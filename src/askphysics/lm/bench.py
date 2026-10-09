"""Time full training steps on each device and precision (``askphysics model bench``)."""

from __future__ import annotations

import gc
import time
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from askphysics.lm.config import ModelConfig
from askphysics.lm.model import FermiLM
from askphysics.lm.train import TrainConfig, _optimizer, _release_cached_memory, autocast_dtype


@dataclass
class BenchResult:
    """Speed of one (device, precision) setting, or why it failed."""

    device: str
    precision: str
    seconds_per_step: float = 0.0
    tokens_per_s: float = 0.0
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


def projected_hours(tokens_per_s: float, steps: int, batch_size: int, width: int) -> float:
    """Hours to train ``steps`` steps of ``batch_size * width`` tokens at ``tokens_per_s``."""
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


def _time_setting(
    config: ModelConfig,
    device_name: str,
    precision: str,
    batch_size: int,
    width: int,
    steps: int,
    warmup: int,
) -> float:
    device = torch.device(device_name)
    dtype = autocast_dtype(device, precision)
    cfg = TrainConfig(device=device_name, precision=precision, batch_size=batch_size)
    model = FermiLM(config).to(device)
    model.train()
    opt = _optimizer(model, cfg)
    inputs = torch.randint(0, config.vocab_size, (batch_size, width), device=device)
    labels = torch.roll(inputs, -1, dims=1)

    def one_step() -> None:
        with torch.autocast(device.type, dtype=dtype or torch.bfloat16, enabled=dtype is not None):
            _, loss = model(inputs, labels)
        assert loss is not None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    for _ in range(warmup):
        one_step()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    _sync(device)
    return (time.perf_counter() - t0) / steps


def bench(
    config: ModelConfig,
    *,
    batch_size: int,
    width: int,
    steps: int,
    warmup: int,
    settings: Sequence[tuple[str, str]],
) -> list[BenchResult]:
    """Time ``steps`` full training steps for each (device, precision) setting.

    A setting that fails (unknown device, out of memory, unsupported op) is reported in
    ``error`` rather than raised.
    """
    results: list[BenchResult] = []
    for device_name, precision in settings:
        try:
            seconds = _time_setting(
                config, device_name, precision, batch_size, width, steps, warmup
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
        tokens = batch_size * width / seconds if seconds > 0 else float("inf")
        results.append(BenchResult(device_name, precision, seconds, tokens))
    return results
