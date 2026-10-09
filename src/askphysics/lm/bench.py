"""Time full training steps on each device and precision (``askphysics model bench``)."""

from __future__ import annotations

import gc
import resource
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from askphysics.lm.config import ModelConfig
from askphysics.lm.model import FermiLM
from askphysics.lm.train import TrainConfig, _optimizer, _release_cached_memory, autocast_dtype

_GB = 1024**3


@dataclass
class BenchResult:
    """Speed and memory of one (device, precision) setting, or why it failed.

    ``gpu_mem_gb`` is None off mps and cuda. ``rss_gb`` is the peak RSS of the whole process
    so far, so it never falls between settings.
    """

    device: str
    precision: str
    seconds_per_step: float = 0.0
    tokens_per_s: float = 0.0
    gpu_mem_gb: float | None = None
    rss_gb: float = 0.0
    error: str | None = None


def available_settings() -> list[tuple[str, str]]:
    """The (device, precision) pairs worth timing on this machine."""
    settings: list[tuple[str, str]] = []
    if torch.backends.mps.is_available():
        settings += [("mps", "bf16"), ("mps", "fp32")]
    if torch.cuda.is_available():
        settings += [("cuda", "bf16"), ("cuda", "fp32")]
    settings.append(("cpu", "fp32"))
    return settings


def bench_widths(width: int) -> list[int]:
    """The power-of-two widths from 64 up to ``width``, the shapes the training buckets use.

    A ``width`` below 64 benchmarks just that width.
    """
    widths: list[int] = []
    w = 64
    while w <= width:
        widths.append(w)
        w *= 2
    return widths or [width]


def projected_hours(tokens_per_s: float, steps: int, batch_size: int, width: float) -> float:
    """Hours to train ``steps`` steps of ``batch_size * width`` tokens at ``tokens_per_s``.

    ``width`` is the average width of the benchmarked batches.
    """
    return steps * batch_size * width / tokens_per_s / 3600


def format_hours(hours: float) -> str:
    """Hours as h:mm."""
    minutes = round(hours * 60)
    return f"{minutes // 60}:{minutes % 60:02d}"


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _gpu_memory_gb(device: torch.device) -> float | None:
    """GPU memory in GB: bytes the mps allocator holds, or peak bytes allocated on cuda."""
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / _GB
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / _GB
    return None


def _peak_rss_gb() -> float:
    """Peak resident memory of this process in GB (ru_maxrss is bytes on macOS, KB on Linux)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / _GB if sys.platform == "darwin" else peak * 1024 / _GB


def _time_setting(
    config: ModelConfig,
    device_name: str,
    precision: str,
    batch_size: int,
    widths: Sequence[int],
    steps: int,
    warmup: int,
) -> BenchResult:
    device = torch.device(device_name)
    dtype = autocast_dtype(device, precision)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)  # so the peak is this setting's, not earlier
    cfg = TrainConfig(device=device_name, precision=precision, batch_size=batch_size)
    model = FermiLM(config).to(device)
    model.train()
    opt = _optimizer(model, cfg)
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for width in widths:
        inputs = torch.randint(0, config.vocab_size, (batch_size, width), device=device)
        batches.append((inputs, torch.roll(inputs, -1, dims=1)))

    def one_step(inputs: torch.Tensor, labels: torch.Tensor) -> None:
        with torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            _, loss = model(inputs, labels)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    # Round-robin over the widths, so every shape the training buckets use gets exercised.
    for _ in range(warmup):
        for inputs, labels in batches:
            one_step(inputs, labels)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        for inputs, labels in batches:
            one_step(inputs, labels)
    _sync(device)
    seconds = time.perf_counter() - t0
    total_steps = steps * len(widths)
    total_tokens = steps * sum(batch_size * width for width in widths)
    return BenchResult(
        device_name,
        precision,
        seconds_per_step=seconds / total_steps,
        tokens_per_s=total_tokens / seconds if seconds > 0 else float("inf"),
        gpu_mem_gb=_gpu_memory_gb(device),
        rss_gb=_peak_rss_gb(),
    )


def bench(
    config: ModelConfig,
    *,
    batch_size: int,
    width: int,
    steps: int,
    warmup: int,
    settings: Sequence[tuple[str, str]],
) -> list[BenchResult]:
    """Time ``steps`` full training steps per width for each (device, precision) setting.

    ``width`` is the largest width: each power of two from 64 up to it is timed ``steps`` times,
    round-robin. A setting that fails (unknown device, out of memory, unsupported op) is
    reported in ``error`` rather than raised.
    """
    widths = bench_widths(width)
    results: list[BenchResult] = []
    for device_name, precision in settings:
        try:
            result = _time_setting(
                config, device_name, precision, batch_size, widths, steps, warmup
            )
        except Exception as exc:
            results.append(
                BenchResult(device_name, precision, error=f"{type(exc).__name__}: {exc}")
            )
            continue
        finally:
            gc.collect()  # drop the model before the next setting
            if device_name in {"mps", "cuda"}:
                _release_cached_memory(torch.device(device_name))
        results.append(result)
    return results
