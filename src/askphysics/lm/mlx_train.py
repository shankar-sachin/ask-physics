"""Training the Fermi models with MLX on Apple Silicon (ADR-019).

The same run as ``lm/train.py``, on the MLX model in ``lm/mlx_model.py``: the same
data preparation, batch sampling, learning-rate schedule, metrics, and checkpoint
layout, so ``askphysics.lm.checkpoints.load_model`` reads what this writes. The one
difference is the optimizer state, which is kept as ``m.<param>`` and ``v.<param>``
arrays and marked ``"backend": "mlx"`` in ``train_state.json``; a run resumes only on
the backend that wrote it (``train.check_backend``).

Memory: MLX frees buffers into a cache. The cache is capped (``cache_limit_gb``) and
released at the same points the torch trainer releases its device cache.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
from mlx.nn.utils import value_and_grad
from mlx.optimizers import clip_grad_norm
from mlx.utils import tree_map, tree_unflatten
from safetensors.numpy import load_file as load_numpy_file
from safetensors.numpy import save_file as save_numpy_file

from askphysics.errors import ConfigError
from askphysics.lm.checkpoints import (
    check_tokenizer_fits,
    model_metadata,
    read_config,
    write_config_and_tokenizer,
)
from askphysics.lm.config import ModelConfig
from askphysics.lm.mlx_model import FermiLM, _leaves, export_params, import_params
from askphysics.lm.model import IGNORE_INDEX
from askphysics.lm.paths import WEIGHTS_FILE
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.train import (
    METRICS_FILE,
    OPTIMIZER_FILE,
    PRECISIONS,
    STATE_FILE,
    BatchSampler,
    TokenizedSet,
    TrainConfig,
    TrainData,
    _metric_writer,
    _peak_memory_gb,
    _resume_lr,
    check_backend,
    check_run_settings,
    lr_at,
    numpy_batch,
    prepare_data,
    task_scores,
    write_summary,
)

ADAM_BETAS = (0.9, 0.95)  # the same as the torch trainer
ADAM_EPS = 1e-8  # torch's default
DEFAULT_CACHE_LIMIT_GB = 4.0
# The torch trainer evaluates in fp32: its autocast covers the training step only.
EVAL_DTYPE = mx.float32

# A step function for mx.compile: (learning rate, micro-batches) -> mean loss.
StepFn = Callable[[mx.array, list[tuple[mx.array, mx.array]]], mx.array]


def mlx_dtype(precision: str, on_gpu: bool) -> mx.Dtype:
    """The compute dtype of the forward pass for ``precision``. ``bf16`` is bfloat16 and
    ``fp32`` is float32; ``auto`` is bfloat16 on the GPU and float32 on the CPU, as the torch
    trainer's ``auto`` is bf16 on mps and cuda and fp32 on cpu. Parameters are fp32 whatever
    this says.

    Raises:
        ValueError: ``precision`` is not one of ``PRECISIONS``.
    """
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; choose from {', '.join(PRECISIONS)}")
    if precision == "auto":
        return mx.bfloat16 if on_gpu else mx.float32
    return mx.float32 if precision == "fp32" else mx.bfloat16


class AdamW:
    """AdamW with decoupled weight decay on the parameters with at least two dimensions,
    as the torch trainer sets it up (``train._optimizer``).

    Written out rather than taken from ``mlx.optimizers``: that AdamW decays every parameter
    and leaves bias correction off by default. ``state`` holds the moments and the step count
    as plain arrays, so it can be saved, and passed to ``mx.compile`` as state.
    """

    def __init__(self, model: FermiLM, weight_decay: float) -> None:
        self.weight_decay = weight_decay
        params = _leaves(model.parameters())
        self.state: dict[str, Any] = {
            "m": {name: mx.zeros(p.shape) for name, p in params},
            "v": {name: mx.zeros(p.shape) for name, p in params},
            "step": mx.array(0.0),
        }

    def update(self, model: FermiLM, grads: dict[str, mx.array], lr: mx.array) -> None:
        """Apply one step to ``model`` in place, and advance the moments in ``state``."""
        b1, b2 = ADAM_BETAS
        step = self.state["step"] + 1.0
        self.state["step"] = step
        bias1, bias2 = 1.0 - b1**step, 1.0 - b2**step
        updated: list[tuple[str, mx.array]] = []
        for name, p in _leaves(model.parameters()):
            g = grads[name]
            m = b1 * self.state["m"][name] + (1.0 - b1) * g
            v = b2 * self.state["v"][name] + (1.0 - b2) * g * g
            self.state["m"][name] = m
            self.state["v"][name] = v
            if p.ndim >= 2:
                p = p * (1.0 - lr * self.weight_decay)
            p = p - lr * (m / bias1) / (mx.sqrt(v / bias2) + ADAM_EPS)
            updated.append((name, p))
        model.update(tree_unflatten(updated))


def _loss_fn(
    model: FermiLM, dtype: mx.Dtype, checkpoint_blocks: bool
) -> Callable[[mx.array, mx.array], mx.array]:
    def loss(inputs: mx.array, labels: mx.array) -> mx.array:
        return model.loss(inputs, labels, dtype, checkpoint_blocks)

    return loss


def mean_gradients(
    model: FermiLM,
    micro_batches: Sequence[tuple[mx.array, mx.array]],
    dtype: mx.Dtype,
    checkpoint_blocks: bool,
) -> tuple[mx.array, dict[str, mx.array]]:
    """The mean loss and the mean gradient over ``micro_batches``, as
    ``train._optimizer_step`` accumulates them: each micro-batch counts equally.

    The gradients are returned as a flat dict keyed by parameter name. Both are lazy.
    """
    scale = len(micro_batches)
    loss_and_grad = value_and_grad(model, _loss_fn(model, dtype, checkpoint_blocks))
    total = mx.array(0.0)
    summed: Any = None
    for inputs, labels in micro_batches:
        loss, grads = loss_and_grad(inputs, labels)
        total = total + loss
        summed = grads if summed is None else tree_map(lambda a, b: a + b, summed, grads)
    mean = tree_map(lambda g: g / scale, summed)
    return total / scale, dict(_leaves(mean))


def optimizer_step(
    model: FermiLM,
    adam: AdamW,
    micro_batches: Sequence[tuple[mx.array, mx.array]],
    lr: mx.array,
    cfg: TrainConfig,
    dtype: mx.Dtype,
) -> mx.array:
    """One optimizer step over ``micro_batches`` of (inputs, labels), as ``train._optimizer_step``:
    the mean gradient, global-norm clipping, then AdamW. Returns the mean loss, still lazy."""
    loss, grads = mean_gradients(model, micro_batches, dtype, cfg.checkpoint_blocks)
    clipped, _ = clip_grad_norm(grads, cfg.grad_clip)
    adam.update(model, dict(clipped), lr)
    return loss


def _make_step(
    model: FermiLM, adam: AdamW, cfg: TrainConfig, dtype: mx.Dtype, compile_step: bool
) -> StepFn:
    def step(lr: mx.array, micro: list[tuple[mx.array, mx.array]]) -> mx.array:
        return optimizer_step(model, adam, micro, lr, cfg, dtype)

    if not compile_step:
        return step
    # The compiled graph reads and writes the parameters and the optimizer state in place.
    # Shapes are bucketed (see numpy_batch), so only a handful of graphs get compiled.
    return mx.compile(step, inputs=[model.state, adam.state], outputs=[model.state, adam.state])


def _evaluate(
    model: FermiLM,
    data: TokenizedSet,
    cfg: TrainConfig,
    pad_id: int,
    context_length: int,
    dtype: mx.Dtype,
) -> float:
    """Mean target-token loss over the first ``eval_batches`` batches of ``data``."""
    total, tokens = 0.0, 0
    for b in range(cfg.eval_batches):
        rows = data.rows[b * cfg.batch_size : (b + 1) * cfg.batch_size]
        if not rows:
            break
        inputs, labels = numpy_batch(
            rows, pad_id, multiple=cfg.pad_multiple, max_width=context_length
        )
        n = int((labels != IGNORE_INDEX).sum())
        loss = model.loss(_as_mx(inputs), _as_mx(labels), dtype)
        total += float(loss) * n
        tokens += n
    return total / tokens if tokens else float("nan")


def _as_mx(array: np.ndarray) -> mx.array:
    return mx.array(array.astype(np.int32))


def save_checkpoint(
    model: FermiLM, tokenizer: Tokenizer, directory: Path, adam: AdamW, step: int, lr: float
) -> None:
    """Write a checkpoint that ``checkpoints.load_model`` reads, plus the optimizer state."""
    check_tokenizer_fits(model.config, tokenizer)
    directory.mkdir(parents=True, exist_ok=True)
    save_numpy_file(
        export_params(model), str(directory / WEIGHTS_FILE), metadata=model_metadata(model.config)
    )
    write_config_and_tokenizer(directory, model.config, tokenizer)
    tensors = {
        f"{group}.{name}": np.array(value, dtype=np.float32)
        for group in ("m", "v")
        for name, value in adam.state[group].items()
    }
    save_numpy_file(tensors, str(directory / OPTIMIZER_FILE))
    meta = {"step": step, "lr": lr, "backend": "mlx", "optimizer": "adamw"}
    (directory / STATE_FILE).write_text(json.dumps(meta), encoding="utf-8")


def load_optimizer(adam: AdamW, directory: Path) -> int:
    """Restore the optimizer state saved by ``save_checkpoint``; returns the step it was at.

    Raises:
        ConfigError: the saved moments don't cover this model's parameters.
    """
    meta = json.loads((directory / STATE_FILE).read_text(encoding="utf-8"))
    tensors = load_numpy_file(str(directory / OPTIMIZER_FILE))
    expected = {f"{group}.{name}" for group in ("m", "v") for name in adam.state["m"]}
    if set(tensors) != expected:
        raise ConfigError(f"optimizer state in {directory} does not match the model")
    for key, value in tensors.items():
        group, name = key.split(".", 1)
        adam.state[group][name] = mx.array(value)
    adam.state["step"] = mx.array(float(meta["step"]))
    return int(meta["step"])


MEMORY_FRACTION = 0.7  # default hard limit on MLX's memory, as a share of system RAM


def system_memory_gb() -> float:
    """Physical RAM of this machine, in GB."""
    return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9


def _on_gpu() -> bool:
    """True when MLX computes on the GPU (the default on a Mac, where it is the Metal GPU)."""
    return mx.default_device().type == mx.DeviceType.gpu


def _device(prefer: str | None) -> mx.Device | None:
    """The MLX device for ``--device``: None keeps MLX's default (the GPU on a Mac)."""
    if prefer is None:
        return None
    if prefer == "cpu":
        return mx.Device(mx.cpu)
    if prefer == "mps":
        if not mx.metal.is_available():
            raise ConfigError("device 'mps' is not available on this machine")
        return mx.Device(mx.gpu)
    raise ConfigError(
        f"the mlx backend trains on mps or cpu, not {prefer!r}; use --backend torch for cuda"
    )


def train_mlx(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    resume: bool = False,
    on_log: Callable[[dict[str, Any]], None] | None = None,
    prose: Sequence[str] = (),
    cache_limit_gb: float = DEFAULT_CACHE_LIMIT_GB,
    memory_limit_gb: float | None = None,
    compile_step: bool = True,
) -> list[dict[str, Any]]:
    """Train ``config`` with MLX, with the arguments and behaviour of ``train.train``.

    ``cache_limit_gb`` caps the memory MLX keeps for reuse. ``memory_limit_gb`` is the memory
    limit MLX works to stay under (MLX calls it a guideline: once memory and swap run out,
    allocations raise an error rather than swapping the machine). It defaults to 70% of system
    RAM. ``cfg.checkpoint_blocks`` recomputes each block's activations in the backward pass,
    trading compute for memory. ``compile_step`` runs the optimizer step under ``mx.compile``;
    turn it off to debug.

    Raises:
        ConfigError: ``resume`` and the checkpoint in ``out_dir`` came from the torch backend,
            or ``cfg.device`` is not one MLX can use.
    """
    check_run_settings(config, tokenizer, cfg)
    device = _device(cfg.device)
    limit_gb = (
        memory_limit_gb if memory_limit_gb is not None else MEMORY_FRACTION * system_memory_gb()
    )
    previous_device = mx.default_device()
    previous_limit = mx.set_cache_limit(int(cache_limit_gb * 1e9))
    previous_memory = mx.set_memory_limit(int(limit_gb * 1e9))
    try:
        if device is not None:
            mx.set_default_device(device)
        dtype = mlx_dtype(cfg.precision, on_gpu=_on_gpu())
        return _run(
            config,
            tokenizer,
            data_dir,
            out_dir,
            cfg,
            resume=resume,
            on_log=on_log,
            prose=prose,
            dtype=dtype,
            compile_step=compile_step,
        )
    finally:
        mx.set_default_device(previous_device)
        mx.set_memory_limit(previous_memory)
        mx.set_cache_limit(previous_limit)


def _run(
    config: ModelConfig,
    tokenizer: Tokenizer,
    data_dir: Path,
    out_dir: Path,
    cfg: TrainConfig,
    *,
    resume: bool,
    on_log: Callable[[dict[str, Any]], None] | None,
    prose: Sequence[str],
    dtype: mx.Dtype,
    compile_step: bool,
) -> list[dict[str, Any]]:
    mx.random.seed(cfg.seed)
    rng = random.Random(cfg.seed)
    data: TrainData = prepare_data(config, tokenizer, data_dir, cfg, prose)

    model = FermiLM(config)
    adam = AdamW(model, cfg.weight_decay)
    start = 0
    resumed: tuple[int, float] | None = None
    if resume and (out_dir / STATE_FILE).exists():
        check_backend(out_dir, "mlx")
        read_config(out_dir)  # the weights' metadata and format are checked here
        import_params(model, load_numpy_file(str(out_dir / WEIGHTS_FILE)))
        start = load_optimizer(adam, out_dir)
        at = _resume_lr(out_dir, start)
        if at is not None and start >= cfg.warmup_steps:
            resumed = (start, at)

    sampler = BatchSampler(data, cfg, rng, start)
    step_fn = _make_step(model, adam, cfg, dtype, compile_step)
    metrics: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    if start == 0:
        (out_dir / METRICS_FILE).unlink(missing_ok=True)  # a fresh run starts a fresh log
    log = _metric_writer(out_dir, metrics, on_log)
    # Batches are built on the host, so the target-token count needs no device round trip.
    tokens_seen, window_tokens, window_t0 = 0, 0, time.perf_counter()

    def checkpoint(step: int) -> None:
        save_checkpoint(
            model, tokenizer, out_dir, adam, step, lr=lr_at(max(0, step - 1), cfg, resumed)
        )

    for step in range(start, cfg.steps):
        micro: list[tuple[mx.array, mx.array]] = []
        targets = 0
        for _ in range(cfg.grad_accum):
            inputs, labels = numpy_batch(
                sampler.next_rows(step),
                tokenizer.pad_id,
                multiple=cfg.pad_multiple,
                max_width=config.context_length,
            )
            targets += int((labels != IGNORE_INDEX).sum())
            micro.append((_as_mx(inputs), _as_mx(labels)))
        lr = lr_at(step, cfg, resumed)
        loss = step_fn(mx.array(lr, dtype=mx.float32), micro)
        # Evaluate the step now, so the lazy graph never spans more than one step.
        mx.eval(model.parameters(), adam.state, loss)
        tokens_seen += targets

        done = step + 1
        if done % cfg.log_every == 0 or done == cfg.steps:
            now = time.perf_counter()
            rate = (tokens_seen - window_tokens) / max(now - window_t0, 1e-9)
            window_tokens, window_t0 = tokens_seen, now
            log({"step": done, "loss": round(float(loss), 4), "lr": lr,
                 "target_tokens_per_s": round(rate, 1),
                 "mem_gb": _peak_memory_gb(),
                 "gpu_gb": round(mx.get_peak_memory() / 1e9, 1)})  # fmt: skip
        if data.val_samples and (done % cfg.eval_every == 0 or done == cfg.steps):
            scores = task_scores(
                data.val_samples,
                lambda s: _evaluate(
                    model, s, cfg, tokenizer.pad_id, config.context_length, EVAL_DTYPE
                ),
            )
            if data.prose_val.rows:
                prose_loss = _evaluate(
                    model, data.prose_val, cfg, tokenizer.pad_id, config.context_length, EVAL_DTYPE
                )
                scores["val_loss_prose"] = round(prose_loss, 4)
            log({"step": done, **scores})
            mx.clear_cache()
            window_tokens, window_t0 = tokens_seen, time.perf_counter()  # not training time
        if done % cfg.checkpoint_every == 0 and done != cfg.steps:
            checkpoint(done)
            mx.clear_cache()
            window_tokens, window_t0 = tokens_seen, time.perf_counter()
        if cfg.flush_every and done % cfg.flush_every == 0:
            mx.clear_cache()

    checkpoint(cfg.steps)
    device_name = mx.default_device().type.name  # "gpu" on a Mac, "cpu" otherwise
    write_summary(out_dir, config, data, cfg, device=device_name, backend="mlx")
    return metrics
