"""Model configurations for the Fermi family (``docs/MODELS.md``).

Pure Python, no torch import, so the CLI and tokenizer can read configs
cheaply. Parameter counts are computed analytically and pinned by tests;
``lm/model.py`` must build exactly this many parameters.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from askphysics.errors import ConfigError
from askphysics.lm.paths import CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE

# Bumped whenever a task format changes; recorded in traces and eval reports (Q13).
FORMAT_VERSION = 1

VOCAB_SIZE = 8192


def _round_up(value: float, multiple: int) -> int:
    return int(-(-value // multiple) * multiple)


@dataclass(frozen=True)
class ModelConfig:
    """Shape of one Fermi model.

    Attributes:
        name: Model id, for example ``fermi-solem-1``.
        vocab_size: Tokenizer vocabulary size, shared by the family.
        d_model: Embedding and residual width.
        n_layers: Number of transformer blocks.
        n_heads: Attention heads; ``d_model`` must divide evenly.
        context_length: Maximum sequence length in tokens.
        rope_theta: Base for rotary position embeddings.
        mlp_multiple_of: SwiGLU hidden size is rounded up to this multiple.
    """

    name: str
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    context_length: int
    rope_theta: float = 10_000.0
    mlp_multiple_of: int = 32

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"{self.name}: d_model {self.d_model} not divisible by n_heads")
        if self.head_dim % 2:
            raise ValueError(f"{self.name}: head_dim must be even for rotary embeddings")
        if min(self.vocab_size, self.d_model, self.n_layers, self.context_length) < 1:
            raise ValueError(f"{self.name}: sizes must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def mlp_hidden(self) -> int:
        """SwiGLU hidden width: about 8/3 of d_model, so the MLP costs about 8*d^2."""
        return _round_up(8 * self.d_model / 3, self.mlp_multiple_of)

    def num_parameters(self) -> int:
        """Exact trainable parameter count, with the output head tied to the embeddings."""
        d = self.d_model
        attention = 4 * d * d  # q, k, v, out projections, no biases
        mlp = 3 * d * self.mlp_hidden  # gate, up, down
        norms = 2 * d  # two RMSNorm weights per block
        per_layer = attention + mlp + norms
        return self.vocab_size * d + self.n_layers * per_layer + d  # + final norm


TELLUS = ModelConfig(
    name="fermi-tellus-1", vocab_size=VOCAB_SIZE, d_model=160, n_layers=6, n_heads=5,
    context_length=1024,
)  # fmt: skip
SOLEM = ModelConfig(
    name="fermi-solem-1", vocab_size=VOCAB_SIZE, d_model=512, n_layers=8, n_heads=8,
    context_length=1024, mlp_multiple_of=64,
)  # fmt: skip
CELESTE = ModelConfig(
    name="fermi-celeste-1", vocab_size=VOCAB_SIZE, d_model=768, n_layers=16, n_heads=12,
    context_length=1024, mlp_multiple_of=64,
)  # fmt: skip
LUNA = ModelConfig(
    name="fermi-luna-1", vocab_size=512, d_model=64, n_layers=2, n_heads=4, context_length=1024,
)  # fmt: skip


def config_json(config: ModelConfig) -> str:
    """The canonical JSON of a config: ``config.json``, and the weights' metadata."""
    return json.dumps(asdict(config), sort_keys=True)


def require_model_files(directory: Path) -> None:
    """Raise ``ConfigError`` unless ``directory`` has the weights, config, and tokenizer."""
    paths = [directory / name for name in (WEIGHTS_FILE, CONFIG_FILE, TOKENIZER_FILE)]
    if missing := [p.name for p in paths if not p.exists()]:
        raise ConfigError(f"model directory {directory} is missing {', '.join(missing)}")


def saved_config(directory: Path, metadata: Mapping[str, str]) -> ModelConfig:
    """The ``config.json`` in ``directory``, checked against the weights' ``metadata``.

    Both loaders (torch in ``lm/checkpoints.py``, numpy in ``lm/numpy_model.py``) read the
    config through this, so they refuse the same files.

    Raises:
        ConfigError: the config doesn't match the weights' metadata, or the task format
            version is different.
    """
    config_text = (directory / CONFIG_FILE).read_text(encoding="utf-8")
    config = ModelConfig(**json.loads(config_text))
    if metadata.get("config") != config_json(config):
        raise ConfigError(f"{CONFIG_FILE} does not match the weights in {directory}")
    if metadata.get("format_version") != str(FORMAT_VERSION):
        raise ConfigError(
            f"model was trained on task format {metadata.get('format_version')}, "
            f"this askphysics uses {FORMAT_VERSION}; retrain or upgrade"
        )
    return config


PRESETS: dict[str, ModelConfig] = {c.name: c for c in (TELLUS, SOLEM, CELESTE, LUNA)}


def get_config(name: str) -> ModelConfig:
    """Look up a preset by name, or raise ``KeyError`` listing the presets."""
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; presets: {', '.join(PRESETS)}") from None
