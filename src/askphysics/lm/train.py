"""Training for the Fermi models (``docs/MODELS.md``).

Examples from the data factory are tokenized as ``prompt + target``. The
loss only counts target tokens, so the models learn to write answers, not
to parrot inputs. AdamW with linear warmup and cosine decay, gradient
clipping, and bf16 autocast on MPS and CUDA. Checkpoints (weights,
optimizer moments, step) are safetensors and JSON only, never pickle.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from askphysics.lm.checkpoints import save_model
from askphysics.lm.config import ModelConfig
from askphysics.lm.device import select_device
from askphysics.lm.factory import Example, read_examples
from askphysics.lm.generate import encode_task
from askphysics.lm.model import IGNORE_INDEX, FermiLM
from askphysics.lm.tokenizer import Tokenizer

DEFAULT_LR = {
    "fermi-luna-1": 3e-3,
    "fermi-tellus-1": 1e-3,
    "fermi-solem-1": 6e-4,
    "fermi-celeste-1": 3e-4,
}
OPTIMIZER_FILE = "optimizer.safetensors"
STATE_FILE = "train_state.json"
METRICS_FILE = "metrics.jsonl"


@dataclass
class TrainConfig:
    """Knobs for one training run. Defaults suit fermi-luna-1 on a laptop CPU."""

    steps: int = 1000
    batch_size: int = 16
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 8
    checkpoint_every: int = 500
    log_every: int = 20
    seed: int = 0
    device: str | None = None


@dataclass
class TokenizedSet:
    """Examples as (token ids, index of the first target token)."""

    rows: list[tuple[list[int], int]] = field(default_factory=list)
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.rows)


def tokenize_examples(
    examples: Iterator[Example] | Sequence[Example], tokenizer: Tokenizer, context_length: int
) -> TokenizedSet:
    """Tokenize examples; drop any that don't fit the context window."""
    out = TokenizedSet()
    for e in examples:
        prompt = encode_task(tokenizer, e.prompt)
        target = encode_task(tokenizer, e.target)
        ids = prompt + target
        if len(ids) > context_length + 1:
            out.skipped += 1
            continue
        out.rows.append((ids, len(prompt)))
    return out


def make_batch(
    rows: Sequence[tuple[list[int], int]], pad_id: int, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Inputs and next-token labels; prompt and padding positions are ignored by the loss."""
    width = max(len(ids) for ids, _ in rows) - 1
    inputs = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, target_start) in enumerate(rows):
        seq = torch.tensor(ids, dtype=torch.long)
        n = len(ids) - 1
        inputs[i, :n] = seq[:-1]
        labels[i, :n] = seq[1:]
        labels[i, : target_start - 1] = IGNORE_INDEX  # only learn to write the target
    return inputs.to(device), labels.to(device)


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)


def _optimizer(model: FermiLM, cfg: TrainConfig) -> torch.optim.AdamW:
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95))


def _save_optimizer(opt: torch.optim.Optimizer, step: int, directory: Path) -> None:
    state = opt.state_dict()
    tensors: dict[str, Tensor] = {}
    for idx, slots in state["state"].items():
        for name, value in slots.items():
            tensors[f"{idx}.{name}"] = torch.as_tensor(value).detach().to("cpu").contiguous()
    save_file(tensors, str(directory / OPTIMIZER_FILE))
    meta = {"step": step, "param_groups": [
        {k: v for k, v in g.items() if k != "params"} | {"params": g["params"]}
        for g in state["param_groups"]
    ]}  # fmt: skip
    (directory / STATE_FILE).write_text(json.dumps(meta), encoding="utf-8")


def _load_optimizer(opt: torch.optim.Optimizer, directory: Path) -> int:
    meta = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    tensors = load_file(str(directory / OPTIMIZER_FILE))
    state: dict[int, dict[str, Tensor]] = {}
    for key, value in tensors.items():
        idx, name = key.split(".", 1)
        state.setdefault(int(idx), {})[name] = value
    opt.load_state_dict({"state": state, "param_groups": meta["param_groups"]})
    return int(meta["step"])


@torch.no_grad()
def evaluate(
    model: FermiLM, data: TokenizedSet, cfg: TrainConfig, pad_id: int, device: torch.device
) -> float:
    """Mean target-token loss over the first ``eval_batches`` batches of ``data``."""
    model.eval()
    losses = []
    for b in range(cfg.eval_batches):
        rows = data.rows[b * cfg.batch_size : (b + 1) * cfg.batch_size]
        if not rows:
            break
        inputs, labels = make_batch(rows, pad_id, device)
        _, loss = model(inputs, labels)
        assert loss is not None
        losses.append(float(loss))
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def train(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    resume: bool = False,
    on_log: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Train ``config`` on the factory data in ``data_dir``, saving to ``out_dir``."""
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError(
            f"tokenizer has {tokenizer.vocab_size} tokens; {config.name} only {config.vocab_size}"
        )
    device = select_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)

    train_set = tokenize_examples(
        read_examples(data_dir / "train"), tokenizer, config.context_length
    )
    val_set = tokenize_examples(read_examples(data_dir / "val"), tokenizer, config.context_length)
    if not train_set.rows:
        raise ValueError(f"no training examples fit the context in {data_dir / 'train'}")

    model = FermiLM(config).to(device)
    opt = _optimizer(model, cfg)
    start = 0
    if resume and (out_dir / STATE_FILE).exists():
        from askphysics.lm.checkpoints import load_model

        loaded, _ = load_model(out_dir, device)
        model.load_state_dict(loaded.state_dict())
        start = _load_optimizer(opt, out_dir)
    model.train()

    use_autocast = device.type in {"mps", "cuda"}
    order = list(range(len(train_set)))
    rng.shuffle(order)
    cursor = (start * cfg.batch_size) % len(order)
    metrics: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    tokens_seen, t0 = 0, time.perf_counter()

    def log(entry: dict[str, Any]) -> None:
        metrics.append(entry)
        with (out_dir / METRICS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if on_log:
            on_log(entry)

    def checkpoint(step: int) -> None:
        save_model(model, tokenizer, out_dir)
        _save_optimizer(opt, step, out_dir)

    for step in range(start, cfg.steps):
        if cursor + cfg.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        rows = [train_set.rows[i] for i in order[cursor : cursor + cfg.batch_size]]
        cursor += cfg.batch_size
        inputs, labels = make_batch(rows, tokenizer.pad_id, device)
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_autocast):
            _, loss = model(inputs, labels)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tokens_seen += int((labels != IGNORE_INDEX).sum())

        done = step + 1
        if done % cfg.log_every == 0 or done == cfg.steps:
            elapsed = time.perf_counter() - t0
            log({"step": done, "loss": round(float(loss.detach()), 4), "lr": lr_at(step, cfg),
                 "target_tokens_per_s": round(tokens_seen / elapsed, 1)})  # fmt: skip
        if val_set.rows and (done % cfg.eval_every == 0 or done == cfg.steps):
            log(
                {
                    "step": done,
                    "val_loss": round(evaluate(model, val_set, cfg, tokenizer.pad_id, device), 4),
                }
            )
        if done % cfg.checkpoint_every == 0 and done != cfg.steps:
            checkpoint(done)

    checkpoint(cfg.steps)
    summary = {"config": config.name, "train_examples": len(train_set),
               "skipped_too_long": train_set.skipped, "val_examples": len(val_set),
               "device": device.type, "train": asdict(cfg) | {"device": device.type}}  # fmt: skip
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics


def train_tokenizer(data_dir: Path, vocab_size: int, max_examples: int = 50_000) -> Tokenizer:
    """Train the BPE tokenizer on the prompts and targets of the training split."""
    texts: list[str] = []
    for i, e in enumerate(read_examples(data_dir / "train")):
        if i >= max_examples:
            break
        texts += [e.prompt, e.target]
    if not texts:
        raise ValueError(f"no training examples in {data_dir / 'train'}")
    return Tokenizer.train(texts, vocab_size=vocab_size)
