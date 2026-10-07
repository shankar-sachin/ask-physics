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
    pad_multiple: int = 64  # batch widths come in a few fixed sizes (see make_batch)
    # Real prose (ADR-016): the first ``prose_steps`` steps train on it alone, then a
    # ``prose_share`` of later batches keep it from fading. Both need ``prose`` texts.
    prose_steps: int = 0
    prose_share: float = 0.0


@dataclass
class TokenizedSet:
    """Examples as (token ids, index of the first target token), with each one's task."""

    rows: list[tuple[list[int], int]] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
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
        out.tasks.append(e.task)
    return out


def val_sample(data: TokenizedSet, per_task: int, seed: int = 0) -> dict[str, TokenizedSet]:
    """A fixed random sample of up to ``per_task`` validation rows for each task.

    Shard order groups examples by worker, so the first rows of the split are not a fair
    sample; shuffling once with a fixed seed keeps every evaluation comparable.
    """
    order = list(range(len(data)))
    random.Random(seed).shuffle(order)
    out: dict[str, TokenizedSet] = {}
    for i in order:
        task = data.tasks[i]
        sample = out.setdefault(task, TokenizedSet())
        if len(sample) < per_task:
            sample.rows.append(data.rows[i])
            sample.tasks.append(task)
    return dict(sorted(out.items()))


def make_batch(
    rows: Sequence[tuple[list[int], int]],
    pad_id: int,
    device: torch.device,
    *,
    multiple: int = 1,
    max_width: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Inputs and next-token labels; prompt and padding positions are ignored by the loss.

    The width is rounded up to ``multiple`` (capped at ``max_width``) so batches come in
    a handful of shapes. On Apple GPUs (MPS) every new shape grows a per-shape cache, and
    hundreds of distinct widths exhaust memory within a few hundred steps.
    """
    width = max(len(ids) for ids, _ in rows) - 1
    width = -(-width // multiple) * multiple
    if max_width is not None:
        width = max(min(width, max_width), max(len(ids) for ids, _ in rows) - 1)
    inputs = torch.full((len(rows), width), pad_id, dtype=torch.long)
    labels = torch.full((len(rows), width), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, target_start) in enumerate(rows):
        seq = torch.tensor(ids, dtype=torch.long)
        n = len(ids) - 1
        inputs[i, :n] = seq[:-1]
        labels[i, :n] = seq[1:]
        labels[i, : target_start - 1] = IGNORE_INDEX  # only learn to write the target
    return inputs.to(device), labels.to(device)


def _release_cached_memory(device: torch.device) -> None:
    """Hand cached GPU buffers back to the system (MPS keeps them until asked)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


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
    total, tokens = 0.0, 0
    for b in range(cfg.eval_batches):
        rows = data.rows[b * cfg.batch_size : (b + 1) * cfg.batch_size]
        if not rows:
            break
        inputs, labels = make_batch(
            rows,
            pad_id,
            device,
            multiple=cfg.pad_multiple,
            max_width=model.config.context_length,
        )
        _, loss = model(inputs, labels)
        assert loss is not None
        n = int((labels != IGNORE_INDEX).sum())
        total += float(loss) * n
        tokens += n
    model.train()
    return total / tokens if tokens else float("nan")


def evaluate_by_task(
    model: FermiLM,
    samples: dict[str, TokenizedSet],
    cfg: TrainConfig,
    pad_id: int,
    device: torch.device,
) -> dict[str, float]:
    """``val_loss`` over every sample, plus ``val_loss_<task>`` for each task."""
    out: dict[str, float] = {}
    total, weight = 0.0, 0
    for task, sample in samples.items():
        loss = evaluate(model, sample, cfg, pad_id, device)
        out[f"val_loss_{task}"] = round(loss, 4)
        n = sum(len(ids) - start for ids, start in sample.rows)
        total += loss * n
        weight += n
    return {"val_loss": round(total / weight, 4) if weight else float("nan"), **out}


def train(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    resume: bool = False,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    prose: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Train ``config`` on the factory data in ``data_dir``, saving to ``out_dir``.

    With ``prose`` paragraphs, the first ``cfg.prose_steps`` steps are a language-modeling
    stage on them, and ``cfg.prose_share`` of later batches mix them back in (ADR-016).
    Their held-out paragraphs are reported as ``val_loss_prose``, apart from ``val_loss``.
    """
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
    val_samples = val_sample(val_set, cfg.eval_batches * cfg.batch_size, cfg.seed)
    if not train_set.rows:
        raise ValueError(f"no training examples fit the context in {data_dir / 'train'}")
    prose_train, prose_val = TokenizedSet(), TokenizedSet()
    if prose:
        from askphysics.lm.corpus import split_prose, tokenize_prose

        train_texts, val_texts = split_prose(prose)
        prose_train = tokenize_prose(train_texts, tokenizer, config.context_length)
        prose_val = tokenize_prose(val_texts, tokenizer, config.context_length)
    elif cfg.prose_steps or cfg.prose_share:
        raise ValueError("prose_steps and prose_share need prose texts")

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
    if start == 0:
        (out_dir / METRICS_FILE).unlink(missing_ok=True)  # a fresh run starts a fresh log
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
        if prose_train.rows and (step < cfg.prose_steps or rng.random() < cfg.prose_share):
            rows = rng.sample(prose_train.rows, min(cfg.batch_size, len(prose_train.rows)))
        else:
            if cursor + cfg.batch_size > len(order):
                rng.shuffle(order)
                cursor = 0
            rows = [train_set.rows[i] for i in order[cursor : cursor + cfg.batch_size]]
            cursor += cfg.batch_size
        inputs, labels = make_batch(
            rows,
            tokenizer.pad_id,
            device,
            multiple=cfg.pad_multiple,
            max_width=config.context_length,
        )
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
        if val_samples and (done % cfg.eval_every == 0 or done == cfg.steps):
            scores = evaluate_by_task(model, val_samples, cfg, tokenizer.pad_id, device)
            if prose_val.rows:
                prose_loss = evaluate(model, prose_val, cfg, tokenizer.pad_id, device)
                scores["val_loss_prose"] = round(prose_loss, 4)
            log({"step": done, **scores})
            _release_cached_memory(device)
        if done % cfg.checkpoint_every == 0 and done != cfg.steps:
            checkpoint(done)
            _release_cached_memory(device)

    checkpoint(cfg.steps)
    summary = {"config": config.name, "train_examples": len(train_set),
               "skipped_too_long": train_set.skipped, "val_examples": len(val_set),
               "prose_paragraphs": len(prose), "prose_rows": len(prose_train),
               "device": device.type, "train": asdict(cfg) | {"device": device.type}}  # fmt: skip
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return metrics


MAX_TOKENIZER_PROSE = 30_000  # paragraphs; a sample is enough to learn the words


def train_tokenizer(
    data_dir: Path, vocab_size: int, max_examples: int = 50_000, prose: Sequence[str] = ()
) -> Tokenizer:
    """Train the BPE tokenizer on the prompts and targets of the training split, plus any
    ``prose`` paragraphs, so real English words get tokens of their own. A large corpus is
    sampled down to ``MAX_TOKENIZER_PROSE`` paragraphs (a fixed sample, so it's repeatable)."""
    if len(prose) > MAX_TOKENIZER_PROSE:
        prose = random.Random(0).sample(list(prose), MAX_TOKENIZER_PROSE)
    texts: list[str] = list(prose)
    for i, e in enumerate(read_examples(data_dir / "train")):
        if i >= max_examples:
            break
        texts += [e.prompt, e.target]
    if len(texts) == len(prose):
        raise ValueError(f"no training examples in {data_dir / 'train'}")
    return Tokenizer.train(texts, vocab_size=vocab_size)
