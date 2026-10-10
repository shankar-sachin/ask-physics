"""Which framework trains a Fermi model: MLX on Apple Silicon, torch everywhere else (ADR-019).

Torch does inference and training on every platform. MLX trains faster and in less memory on
an arm64 Mac, where torch's MPS backend grew past 40 GB and swapped. This module decides, and
imports nothing heavy, so the CLI and ``scripts/train.sh`` can ask it cheaply.
"""

from __future__ import annotations

import platform
import sys

from askphysics.errors import ConfigError

BACKENDS = ("auto", "torch", "mlx")


def mlx_available() -> bool:
    """True when ``mlx.core`` can be imported here."""
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def on_apple_silicon() -> bool:
    return sys.platform == "darwin" and platform.machine() == "arm64"


def resolve_backend(backend: str, device: str | None) -> str:
    """The backend a training run uses: ``"torch"`` or ``"mlx"``.

    ``auto`` is MLX when mlx imports on an arm64 Mac and ``device`` is not cpu or cuda, and
    torch otherwise. ``mlx`` and ``torch`` are taken as given.

    Raises:
        ConfigError: ``backend`` is unknown, or ``mlx`` is asked for without mlx installed.
    """
    if backend not in BACKENDS:
        raise ConfigError(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if backend == "torch":
        return "torch"
    if backend == "mlx":
        if not mlx_available():
            raise ConfigError(
                "the mlx backend is not installed; on an Apple Silicon Mac run "
                "pip install -e '.[mlx]', or train with --backend torch"
            )
        return "mlx"
    if device in {"cpu", "cuda"} or not (on_apple_silicon() and mlx_available()):
        return "torch"
    return "mlx"
