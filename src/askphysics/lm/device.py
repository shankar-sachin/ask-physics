"""Pick the torch device: Apple Silicon (MPS) first, then CUDA, then CPU."""

from __future__ import annotations

import torch

from askphysics.errors import ConfigError

DEVICES = ("mps", "cuda", "cpu")


def select_device(prefer: str | None = None) -> torch.device:
    """Return ``prefer`` if given and available, else the best available device.

    Raises:
        ConfigError: ``prefer`` is not a known device or is not available here.
    """
    if prefer is not None:
        if prefer not in DEVICES:
            raise ConfigError(f"unknown device {prefer!r}; choose from {', '.join(DEVICES)}")
        if not _available(prefer):
            raise ConfigError(f"device {prefer!r} is not available on this machine")
        return torch.device(prefer)
    for name in DEVICES:
        if _available(name):
            return torch.device(name)
    return torch.device("cpu")  # pragma: no cover - cpu is always available


def _available(name: str) -> bool:
    if name == "mps":
        return bool(torch.backends.mps.is_available())
    if name == "cuda":
        return bool(torch.cuda.is_available())
    return True
