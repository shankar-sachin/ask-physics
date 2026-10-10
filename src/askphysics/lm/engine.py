"""The seam between the constrained decoder and the code that runs a model (ADR-021).

``Decoder`` (``lm/generate.py``) decides what may be written; an ``Engine`` computes the
next-token logits. The logits cross the seam as one-dimensional float32 numpy arrays, so the
masking, the number guard, the slot rules, and the greedy choice are written once, over numpy,
whichever framework produced the logits:

- ``lm.torch_engine.TorchEngine`` wraps ``FermiLM`` (the CLI, training, and evaluation).
- ``lm.numpy_model.NumpyFermiLM`` is a forward pass in plain numpy, for the browser (Pyodide has
  no torch).

This module imports neither framework, so the website can ship it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from askphysics.lm.config import ModelConfig

# Next-token scores for one position, over the whole vocabulary.
Logits = npt.NDArray[np.float32]
# One (keys, values) pair per layer, each of shape (batch, heads, time, head_dim). The array
# type belongs to the engine; the decoder only passes caches back to it.
Cache = list[tuple[Any, Any]]


class Engine(ABC):
    """What the decoder needs from a model."""

    config: ModelConfig
    device: str  # where it runs, for reports: "cpu", "mps", "cuda"

    @abstractmethod
    def next_logits(self, tokens: Sequence[int], past: Cache | None) -> tuple[Logits, Cache]:
        """Logits for the position after ``tokens``, and the cache that now includes them.

        With ``past=None`` ``tokens`` is the whole prompt. With a cache it is exactly one token.
        """

    @abstractmethod
    def generator(self, seed: int) -> Any:
        """A seeded random source for ``sample``."""

    @abstractmethod
    def sample(self, masked: Logits, temperature: float, generator: Any) -> int:
        """One token drawn from ``softmax(masked / temperature)``; ``-inf`` entries never are."""

    def cache_length(self, past: Cache) -> int:
        """Tokens a cache covers."""
        return int(past[0][0].shape[2])

    def truncate(self, past: Cache, n: int) -> Cache:
        """The cache of the first ``n`` tokens (views, so the original stays valid)."""
        return [(k[:, :, :n], v[:, :, :n]) for k, v in past]
