"""Training for the Fermi models (``docs/MODELS.md``).

Examples from the data factory are tokenized as ``prompt + target``. The
loss only counts target tokens, so the models learn to write answers, not
to parrot inputs. AdamW with linear warmup and cosine decay, gradient
clipping, and bf16 autocast on MPS and CUDA. Checkpoints (weights,
optimizer moments, step) are safetensors and JSON only, never pickle.

This is the torch trainer. On Apple Silicon the MLX trainer (``lm/mlx_train.py``,
ADR-019) runs the same loop and shares the data, batch, and logging code here.
"""

from __future__ import annotations

import json
import math
import random
import resource
import sys
import time
from array import array
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, overload

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from torch import Tensor

from askphysics.errors import ConfigError
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
    batch_size: int = 16  # micro-batch size; the effective batch is batch_size * grad_accum
    grad_accum: int = 1  # micro-batches per optimizer step
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
    precision: str = "auto"  # auto, bf16, or fp32 (see autocast_dtype)
    pad_multiple: int = 64  # smallest batch width; wider ones step by 1.5x and 2x (bucket_width)
    flush_every: int = 100  # steps between releases of cached device memory; 0 disables
    # Real prose (ADR-016): the first ``prose_steps`` steps train on it alone, then a
    # ``prose_share`` of later batches keep it from fading. Both need ``prose`` texts.
    prose_steps: int = 0
    prose_share: float = 0.0
    # Recompute each block's activations in the backward pass instead of storing them: less
    # memory, more compute. Both backends honour it (ADR-019).
    checkpoint_blocks: bool = False


Row = tuple[list[int], int]  # token ids, index of the first target token


class _RowView(Sequence[Row]):
    """Read-only view of a ``TokenizedSet``'s rows. Rows are built on access, so the view
    holds no copy of the data. It compares equal to any sequence with the same rows."""

    def __init__(self, data: TokenizedSet) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int) -> Row: ...

    @overload
    def __getitem__(self, index: slice) -> list[Row]: ...

    def __getitem__(self, index: int | slice) -> Row | list[Row]:
        if isinstance(index, slice):
            return [self._data.row(i) for i in range(*index.indices(len(self)))]
        if not -len(self) <= index < len(self):
            raise IndexError(f"row {index} out of range for {len(self)} rows")
        return self._data.row(index % len(self))

    def __iter__(self) -> Iterator[Row]:
        for i in range(len(self)):
            yield self._data.row(i)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sequence):
            return NotImplemented
        return list(self) == list(other)


@dataclass
class TokenizedSet:
    """Examples as token ids, the index of each one's first target token, and its task.

    The ids sit in one flat array of 4-byte ints, with row boundaries in ``offsets``. A
    Python list of lists costs about 36 bytes per int, which would take over 10 GB for a
    full training split.
    """

    tokens: array[int] = field(default_factory=lambda: array("i"))
    offsets: array[int] = field(default_factory=lambda: array("q", [0]))
    starts: array[int] = field(default_factory=lambda: array("i"))
    tasks: list[str] = field(default_factory=list)
    skipped: int = 0

    def __len__(self) -> int:
        return len(self.starts)

    @property
    def rows(self) -> _RowView:
        """The rows as (token ids, target start) tuples, read-only."""
        return _RowView(self)

    def add(self, ids: Sequence[int], target_start: int, task: str) -> None:
        self.tokens.extend(ids)
        self.offsets.append(len(self.tokens))
        self.starts.append(target_start)
        self.tasks.append(task)

    def row(self, i: int) -> Row:
        """Row ``i`` (0 <= i < len) as (token ids, index of the first target token)."""
        return self.tokens[self.offsets[i] : self.offsets[i + 1]].tolist(), self.starts[i]


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
        out.add(ids, len(prompt), e.task)
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
            ids, start = data.row(i)
            sample.add(ids, start, task)
    return dict(sorted(out.items()))


def sample_examples_by_task(examples: Iterable[Example], per_task: int, seed: int) -> list[Example]:
    """Up to ``per_task`` examples of each task, chosen uniformly at random (reservoir sampling).

    Only the reservoirs are kept in memory, so a whole split can be streamed through. The
    result is fixed for a given seed and input order, ordered by task and then by reservoir.
    """
    rng = random.Random(seed)
    reservoirs: dict[str, list[Example]] = {}
    seen: dict[str, int] = {}
    for e in examples:
        n = seen.get(e.task, 0)
        seen[e.task] = n + 1
        reservoir = reservoirs.setdefault(e.task, [])
        if len(reservoir) < per_task:
            reservoir.append(e)
        else:
            j = rng.randrange(n + 1)
            if j < per_task:
                reservoir[j] = e
    return [e for task in sorted(reservoirs) for e in reservoirs[task]]


def bucket_width(n: int, smallest: int) -> int:
    """The smallest of ``smallest`` times 1, 1.5, 2, 3, 4, 6, ... that holds ``n`` tokens.

    Few shapes (nine from 64 to 1024) and at most a third of a batch spent on padding;
    powers of two alone would waste up to half (513 tokens padded to 1024).
    """
    width, half_step = max(2, smallest), True
    while width < n:
        width = width * 3 // 2 if half_step else width * 4 // 3
        half_step = not half_step
    return width


def numpy_batch(
    rows: Sequence[Row],
    pad_id: int,
    *,
    multiple: int = 1,
    max_width: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and next-token labels as int64 arrays of shape (rows, width).

    Prompt and padding positions hold ``IGNORE_INDEX`` in the labels, so the loss skips them.
    The width comes from ``bucket_width``: with the default 64 it is one of 64, 96, 128,
    192, 256, 384, 512, 768, or 1024. It is capped at ``max_width`` but never cut below
    the longest row. On Apple GPUs (MPS) every new shape grows a per-shape cache, so a
    handful of shapes keeps memory flat.
    """
    longest = max(len(ids) for ids, _ in rows) - 1
    width = bucket_width(longest, multiple)
    if max_width is not None:
        width = max(min(width, max_width), longest)
    inputs = np.full((len(rows), width), pad_id, dtype=np.int64)
    labels = np.full((len(rows), width), IGNORE_INDEX, dtype=np.int64)
    for i, (ids, target_start) in enumerate(rows):
        seq = np.asarray(ids, dtype=np.int64)
        n = len(ids) - 1
        inputs[i, :n] = seq[:-1]
        labels[i, :n] = seq[1:]
        labels[i, : target_start - 1] = IGNORE_INDEX  # only learn to write the target
    return inputs, labels


def make_batch(
    rows: Sequence[Row],
    pad_id: int,
    device: torch.device,
    *,
    multiple: int = 1,
    max_width: int | None = None,
) -> tuple[Tensor, Tensor]:
    """``numpy_batch`` as torch tensors on ``device``."""
    inputs, labels = numpy_batch(rows, pad_id, multiple=multiple, max_width=max_width)
    return torch.from_numpy(inputs).to(device), torch.from_numpy(labels).to(device)


def _release_cached_memory(device: torch.device) -> None:
    """Hand cached GPU buffers back to the system (MPS keeps them until asked)."""
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


CACHE_SHARE = 0.25  # flush when the unused cache is above this share of device memory


def _cache_is_large(device: torch.device) -> bool:
    """Whether the cache on ``device`` holds more than ``CACHE_SHARE`` of its memory.

    The cache is memory the device has reserved but no tensor uses: on MPS, driver-allocated
    minus current-allocated bytes against the recommended working set; on CUDA, reserved minus
    allocated bytes against the device's total memory. CPU has no such cache, so it is never
    large. When the device's memory API is missing (an older torch) or fails, this returns
    True, so the caller flushes on every width change as it did before this check existed.
    """
    if device.type == "cpu":
        return False
    try:
        if device.type == "mps":
            names = (
                "driver_allocated_memory",
                "current_allocated_memory",
                "recommended_max_memory",
            )
            if not all(hasattr(torch.mps, name) for name in names):
                return True
            cached = torch.mps.driver_allocated_memory() - torch.mps.current_allocated_memory()
            total = torch.mps.recommended_max_memory()
        elif device.type == "cuda":
            names = ("memory_reserved", "memory_allocated", "get_device_properties")
            if not all(hasattr(torch.cuda, name) for name in names):
                return True
            cached = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
            total = torch.cuda.get_device_properties(device).total_memory
        else:
            return True
    except RuntimeError:
        return True
    return cached > CACHE_SHARE * total


class _WidthFlush:
    """Releases cached device memory when an optimizer step's batch width differs from the
    previous step's, but only while the cache is large (over ``CACHE_SHARE`` of the device's
    memory, see ``_cache_is_large``). MPS keeps a cached buffer per shape, so new widths add
    to the cache until the watermark (``mps_env``) frees it; a flush on every change would
    throw away a cache that is still useful, and widths change on about half of all steps.
    CPU never flushes. The training math does not change."""

    def __init__(self, device: torch.device) -> None:
        self._device = device
        self._width: int | None = None

    def before_step(self, width: int) -> None:
        """Record the width of the step about to run, flushing first if it changed and the
        cache is large."""
        if self._width is not None and width != self._width and _cache_is_large(self._device):
            _release_cached_memory(self._device)
        self._width = width


def _peak_memory_gb() -> float:
    """Peak resident memory of this process so far, in GB (ru_maxrss is bytes on macOS)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak * (1 if sys.platform == "darwin" else 1024) / 1e9, 1)


PRECISIONS = ("auto", "bf16", "fp32")


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """The autocast dtype for ``precision`` on ``device``; ``None`` means plain fp32.

    ``auto`` is bf16 on MPS and CUDA and fp32 on CPU.

    Raises:
        ValueError: ``precision`` is not one of ``PRECISIONS``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {', '.join(PRECISIONS)}")
    if precision == "bf16":
        return torch.bfloat16
    if precision == "auto" and device.type in {"mps", "cuda"}:
        return torch.bfloat16
    return None


def lr_at(step: int, cfg: TrainConfig, resumed: tuple[int, float] | None = None) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * lr``.

    ``resumed`` is (step, learning rate) where a resumed run picked up. From there the rate
    decays from that value over the steps left, so a resume that adds steps (3000 to 6000)
    carries on smoothly instead of jumping back up the schedule a longer run would have.
    """
    floor = cfg.lr * cfg.min_lr_ratio
    if resumed is not None and step >= resumed[0]:
        start, at = resumed
        progress = (step - start) / max(1, cfg.steps - start)
        return floor + (at - floor) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
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


def _save_optimizer(
    opt: torch.optim.Optimizer, step: int, directory: Path, lr: float | None = None
) -> None:
    state = opt.state_dict()
    tensors: dict[str, Tensor] = {}
    for idx, slots in state["state"].items():
        for name, value in slots.items():
            tensors[f"{idx}.{name}"] = torch.as_tensor(value).detach().to("cpu").contiguous()
    save_file(tensors, str(directory / OPTIMIZER_FILE))
    meta = {"step": step, "lr": lr, "backend": "torch", "param_groups": [
        {k: v for k, v in g.items() if k != "params"} | {"params": g["params"]}
        for g in state["param_groups"]
    ]}  # fmt: skip
    (directory / STATE_FILE).write_text(json.dumps(meta), encoding="utf-8")


def checkpoint_backend(directory: Path) -> str:
    """The backend that wrote the optimizer state in ``directory``. Checkpoints from before
    the field existed were all torch's."""
    meta = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    return str(meta.get("backend", "torch"))


def check_backend(directory: Path, backend: str) -> None:
    """Raise ``ConfigError`` unless ``directory`` was trained with ``backend``.

    The two trainers keep their optimizer state in different layouts, so a run resumes only
    on the backend that started it (ADR-019).
    """
    found = checkpoint_backend(directory)
    if found != backend:
        raise ConfigError(
            f"{directory} was trained with the {found} backend, not {backend}; resume it with "
            f"--backend {found}, or train into a new --out"
        )


def _resume_lr(directory: Path, step: int) -> float | None:
    """The learning rate a run had reached at ``step``: saved with the checkpoint, or, for
    checkpoints from before that, the last rate logged at or before it."""
    meta = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    if meta.get("lr") is not None:
        return float(meta["lr"])
    logged = None
    metrics = directory / METRICS_FILE
    if metrics.exists():
        for line in metrics.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            if "lr" in entry and entry["step"] <= step:
                logged = float(entry["lr"])
    return logged


def _optimizer_step(
    model: FermiLM,
    opt: torch.optim.Optimizer,
    micro_batches: Sequence[tuple[Tensor, Tensor]],
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype | None,
) -> Tensor:
    """One optimizer step over ``micro_batches`` of (inputs, labels).

    Each micro-batch's loss is divided by their count, so the accumulated gradient is the
    mean over all of them. Returns the mean loss, detached and still on ``device``.
    """
    scale = len(micro_batches)
    opt.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for inputs, labels in micro_batches:
        with torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            _, loss = model(inputs, labels)
        assert loss is not None
        (loss / scale).backward()
        total += loss.detach() / scale
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    return total


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
    return task_scores(samples, lambda sample: evaluate(model, sample, cfg, pad_id, device))


def task_scores(
    samples: dict[str, TokenizedSet], score: Callable[[TokenizedSet], float]
) -> dict[str, float]:
    """Combine per-task losses from ``score``: ``val_loss`` weights each task by its target
    tokens, and each task is also reported as ``val_loss_<task>``. Both backends use this."""
    out: dict[str, float] = {}
    total, weight = 0.0, 0
    for task, sample in samples.items():
        loss = score(sample)
        out[f"val_loss_{task}"] = round(loss, 4)
        n = sum(len(ids) - start for ids, start in sample.rows)
        total += loss * n
        weight += n
    return {"val_loss": round(total / weight, 4) if weight else float("nan"), **out}


@dataclass
class TrainData:
    """The tokenized data for one run: training rows, validation samples, and prose."""

    train: TokenizedSet
    val: TokenizedSet  # the fixed per-task validation sample, before splitting by task
    val_samples: dict[str, TokenizedSet]
    prose_train: TokenizedSet
    prose_val: TokenizedSet
    prose_paragraphs: int


def check_run_settings(config: ModelConfig, tokenizer: Tokenizer, cfg: TrainConfig) -> None:
    """Raise ``ValueError`` for settings no run can use, before any data is read."""
    if cfg.grad_accum < 1:
        raise ValueError(f"grad_accum must be at least 1, not {cfg.grad_accum}")
    if tokenizer.vocab_size > config.vocab_size:
        raise ValueError(
            f"tokenizer has {tokenizer.vocab_size} tokens; {config.name} only {config.vocab_size}"
        )


def prepare_data(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    cfg: TrainConfig,
    prose: Sequence[str],
) -> TrainData:
    """Tokenize the splits and the prose for a run, and check the prose settings against them.

    Raises:
        ValueError: no training example fits the context, or the prose settings can't work.
    """
    train_set = tokenize_examples(
        read_examples(data_dir / "train"), tokenizer, config.context_length
    )
    # Only the validation rows that evaluation reads are tokenized: a fixed sample per task.
    per_task = cfg.eval_batches * cfg.batch_size
    val_set = tokenize_examples(
        sample_examples_by_task(read_examples(data_dir / "val"), per_task, cfg.seed),
        tokenizer,
        config.context_length,
    )
    val_samples = val_sample(val_set, per_task, cfg.seed)
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
    if cfg.prose_steps >= cfg.steps > 0:
        # Prose-only steps come first: as many as the whole run would never train the tasks.
        raise ValueError(
            f"prose_steps ({cfg.prose_steps}) must be fewer than steps ({cfg.steps}), "
            "or the model never trains on the tasks"
        )
    return TrainData(
        train=train_set,
        val=val_set,
        val_samples=val_samples,
        prose_train=prose_train,
        prose_val=prose_val,
        prose_paragraphs=len(prose),
    )


class BatchSampler:
    """Which rows each micro-batch trains on (ADR-016): a shuffled pass over the training
    rows, with prose mixed in as the config says. Both backends draw through this, so one
    seed and one config give the same batches on either."""

    def __init__(self, data: TrainData, cfg: TrainConfig, rng: random.Random, start: int) -> None:
        self._data, self._cfg, self._rng = data, cfg, rng
        self._order = list(range(len(data.train)))
        rng.shuffle(self._order)
        # A resumed run carries on through the pass where its checkpoint stopped.
        self._cursor = (start * cfg.batch_size * cfg.grad_accum) % len(self._order)

    def next_rows(self, step: int) -> list[Row]:
        """The rows of one micro-batch of optimizer step ``step``."""
        cfg, rng = self._cfg, self._rng
        prose = self._data.prose_train.rows
        if prose and (step < cfg.prose_steps or rng.random() < cfg.prose_share):
            return rng.sample(prose, min(cfg.batch_size, len(prose)))
        if self._cursor + cfg.batch_size > len(self._order):
            rng.shuffle(self._order)
            self._cursor = 0
        rows = self._data.train.rows
        picked = [rows[i] for i in self._order[self._cursor : self._cursor + cfg.batch_size]]
        self._cursor += cfg.batch_size
        return picked


def _metric_writer(
    out_dir: Path, metrics: list[dict[str, Any]], on_log: Callable[[dict[str, Any]], None] | None
) -> Callable[[dict[str, Any]], None]:
    """A logger that appends each entry to ``metrics.jsonl`` and to ``metrics``, then calls
    ``on_log``."""

    def log(entry: dict[str, Any]) -> None:
        metrics.append(entry)
        with (out_dir / METRICS_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
        if on_log:
            on_log(entry)

    return log


def write_summary(
    out_dir: Path,
    config: ModelConfig,
    data: TrainData,
    cfg: TrainConfig,
    *,
    device: str,
    backend: str,
) -> None:
    """``training_summary.json``: the model, the data it saw, the settings, and the device."""
    summary = {"config": config.name, "train_examples": len(data.train),
               "skipped_too_long": data.train.skipped, "val_examples": len(data.val),
               "prose_paragraphs": data.prose_paragraphs, "prose_rows": len(data.prose_train),
               "backend": backend, "device": device,
               "train": asdict(cfg) | {"device": device}}  # fmt: skip
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def train(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    resume: bool = False,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    on_step: Callable[[int], None] | None = None,
    prose: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Train ``config`` on the factory data in ``data_dir``, saving to ``out_dir``.

    ``on_step`` is a display hook, called with the number of steps done: once before the first
    step, with the step a resume starts from (0 for a fresh run), then after every step. It
    must stay cheap and must not read the device.

    With ``prose`` paragraphs, the first ``cfg.prose_steps`` steps are a language-modeling
    stage on them, and ``cfg.prose_share`` of later batches mix them back in (ADR-016).
    Their held-out paragraphs are reported as ``val_loss_prose``, apart from ``val_loss``.

    Raises:
        ConfigError: ``resume`` and the checkpoint in ``out_dir`` came from the MLX backend.
    """
    check_run_settings(config, tokenizer, cfg)
    device = select_device(cfg.device)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    data = prepare_data(config, tokenizer, data_dir, cfg, prose)

    model = FermiLM(config).to(device)
    model.checkpoint_blocks = cfg.checkpoint_blocks
    opt = _optimizer(model, cfg)
    start = 0
    resumed: tuple[int, float] | None = None
    if resume and (out_dir / STATE_FILE).exists():
        check_backend(out_dir, "torch")
        from askphysics.lm.checkpoints import load_model

        loaded, _ = load_model(out_dir, device)
        model.load_state_dict(loaded.state_dict())
        start = _load_optimizer(opt, out_dir)
        at = _resume_lr(out_dir, start)
        if at is not None and start >= cfg.warmup_steps:
            resumed = (start, at)
    model.train()

    dtype = autocast_dtype(device, cfg.precision)
    sampler = BatchSampler(data, cfg, rng, start)
    metrics: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if start == 0:
        (out_dir / METRICS_FILE).unlink(missing_ok=True)  # a fresh run starts a fresh log
    # Target tokens are counted on the device and read back only when logging, so the loop
    # never waits for the accelerator. The rate is over the window since the last reset.
    tokens_seen = torch.zeros((), dtype=torch.long, device=device)
    window_tokens, window_t0 = 0, time.perf_counter()
    log = _metric_writer(out_dir, metrics, on_log)
    flush = _WidthFlush(device)

    def checkpoint(step: int) -> None:
        save_model(model, tokenizer, out_dir)
        _save_optimizer(opt, step, out_dir, lr=lr_at(max(0, step - 1), cfg, resumed))

    if on_step:
        on_step(start)
    for step in range(start, cfg.steps):
        micro_batches = [
            make_batch(
                sampler.next_rows(step),
                tokenizer.pad_id,
                device,
                multiple=cfg.pad_multiple,
                max_width=config.context_length,
            )
            for _ in range(cfg.grad_accum)
        ]
        flush.before_step(max(inputs.shape[1] for inputs, _ in micro_batches))
        for group in opt.param_groups:
            group["lr"] = lr_at(step, cfg, resumed)
        loss = _optimizer_step(model, opt, micro_batches, cfg, device, dtype)
        for _, labels in micro_batches:
            tokens_seen += (labels != IGNORE_INDEX).sum()

        done = step + 1
        if done % cfg.log_every == 0 or done == cfg.steps:
            now, total = time.perf_counter(), int(tokens_seen)
            rate = (total - window_tokens) / max(now - window_t0, 1e-9)
            window_tokens, window_t0 = total, now
            log({"step": done, "loss": round(float(loss), 4),
                 "lr": lr_at(step, cfg, resumed),
                 "target_tokens_per_s": round(rate, 1),
                 "mem_gb": _peak_memory_gb()})  # fmt: skip
        if data.val_samples and (done % cfg.eval_every == 0 or done == cfg.steps):
            scores = evaluate_by_task(model, data.val_samples, cfg, tokenizer.pad_id, device)
            if data.prose_val.rows:
                prose_loss = evaluate(model, data.prose_val, cfg, tokenizer.pad_id, device)
                scores["val_loss_prose"] = round(prose_loss, 4)
            log({"step": done, **scores})
            _release_cached_memory(device)
            window_tokens, window_t0 = int(tokens_seen), time.perf_counter()  # not training time
        if done % cfg.checkpoint_every == 0 and done != cfg.steps:
            checkpoint(done)
            _release_cached_memory(device)
            window_tokens, window_t0 = int(tokens_seen), time.perf_counter()
        if cfg.flush_every and done % cfg.flush_every == 0:
            _release_cached_memory(device)

    checkpoint(cfg.steps)
    write_summary(out_dir, config, data, cfg, device=device.type, backend="torch")
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
