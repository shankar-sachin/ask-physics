"""Save and load Fermi models as safetensors.

Never pickle. ``torch.load`` and ``pickle`` unpickle arbitrary code, so a
malicious checkpoint could take over the machine (``SECURITY.md``, risk R15).
safetensors files hold only tensors plus string metadata.

Layout of a model directory::

    <dir>/model.safetensors   weights, with the config JSON in the metadata
    <dir>/config.json         the ModelConfig, human-readable
    <dir>/tokenizer.json      the tokenizer the model was trained with

The MLX trainer (``lm/mlx_train.py``, ADR-019) writes the same layout through the
helpers below, so a model trained on a Mac loads here unchanged.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from askphysics.errors import ConfigError
from askphysics.lm.config import (
    FORMAT_VERSION,
    ModelConfig,
    config_json,
    require_model_files,
    saved_config,
)
from askphysics.lm.model import FermiLM
from askphysics.lm.paths import CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE, default_model_dir
from askphysics.lm.tokenizer import Tokenizer

__all__ = [
    "CONFIG_FILE",
    "TOKENIZER_FILE",
    "WEIGHTS_FILE",
    "check_tokenizer_fits",
    "default_model_dir",
    "load_model",
    "model_metadata",
    "read_config",
    "save_model",
    "write_config_and_tokenizer",
]


def model_metadata(config: ModelConfig) -> dict[str, str]:
    """The string metadata stored in ``model.safetensors`` with the weights."""
    return {"config": config_json(config), "format_version": str(FORMAT_VERSION)}


def check_tokenizer_fits(config: ModelConfig, tokenizer: Tokenizer) -> None:
    """Raise ``ConfigError`` when the tokenizer has more tokens than the model's embeddings."""
    if tokenizer.vocab_size > config.vocab_size:
        raise ConfigError(
            f"tokenizer has {tokenizer.vocab_size} tokens but the model only {config.vocab_size}"
        )


def write_config_and_tokenizer(directory: Path, config: ModelConfig, tokenizer: Tokenizer) -> None:
    """Write ``config.json`` and ``tokenizer.json`` (the weights are written by the caller)."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / CONFIG_FILE).write_text(config_json(config), encoding="utf-8")
    tokenizer.save(directory / TOKENIZER_FILE)


def save_model(model: FermiLM, tokenizer: Tokenizer, directory: Path) -> None:
    """Write weights, config, and tokenizer to ``directory`` (created if needed)."""
    check_tokenizer_fits(model.config, tokenizer)
    directory.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(directory / WEIGHTS_FILE), metadata=model_metadata(model.config))
    write_config_and_tokenizer(directory, model.config, tokenizer)


def read_config(directory: Path) -> ModelConfig:
    """The config of a model saved by ``save_model``, checked against its weights.

    Raises:
        ConfigError: files are missing, the config doesn't match the weights'
            metadata, or the task format version is different.
    """
    require_model_files(directory)

    from safetensors import safe_open

    with safe_open(str(directory / WEIGHTS_FILE), framework="np") as f:
        metadata = f.metadata() or {}
    return saved_config(directory, metadata)


def load_model(directory: Path, device: torch.device | None = None) -> tuple[FermiLM, Tokenizer]:
    """Load a model saved by ``save_model``, in eval mode.

    Raises:
        ConfigError: files are missing, the config doesn't match the weights'
            metadata, or the task format version is different.
    """
    config = read_config(directory)
    model = FermiLM(config)
    target = device or torch.device("cpu")
    model.load_state_dict(load_file(str(directory / WEIGHTS_FILE), device=str(target)))
    model.to(target).eval()
    return model, Tokenizer.load(directory / TOKENIZER_FILE)
