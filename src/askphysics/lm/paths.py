"""Where installed Fermi models live, without importing torch.

``Pipeline.from_settings`` checks for installed weights before it decides
whether to load torch at all, and the website (Pyodide, no torch) imports the
pipeline, so this module only touches the filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"
TOKENIZER_FILE = "tokenizer.json"


def default_model_dir() -> Path:
    """Installed models root: ``$ASKPHYSICS_MODEL_DIR``, else ``~/.cache/askphysics/models``."""
    override = os.environ.get("ASKPHYSICS_MODEL_DIR")
    return Path(override) if override else Path.home() / ".cache" / "askphysics" / "models"


def is_installed(directory: Path) -> bool:
    """Whether ``directory`` holds a complete saved model (weights, config, tokenizer)."""
    return all((directory / name).is_file() for name in (WEIGHTS_FILE, CONFIG_FILE, TOKENIZER_FILE))


def installed_models(root: Path | None = None) -> list[str]:
    """Names of the complete models under ``root`` (default: ``default_model_dir()``)."""
    root = root or default_model_dir()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and is_installed(p))
