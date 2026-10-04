"""The Fermi language models: written and trained from scratch (``docs/MODELS.md``).

Named ``lm`` rather than ``fermi`` so it is not confused with
``askphysics.solver.fermi``, which does Fermi *estimation*.
"""

from askphysics.lm.config import FORMAT_VERSION, PRESETS, ModelConfig, get_config
from askphysics.lm.tokenizer import Tokenizer

__all__ = ["FORMAT_VERSION", "PRESETS", "ModelConfig", "Tokenizer", "get_config"]
