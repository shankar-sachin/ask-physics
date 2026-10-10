"""The Fermi transformer in plain numpy, for the browser (ADR-021).

The same network as ``askphysics.lm.model.FermiLM``: same math, same parameter names and
shapes, so it reads the very same ``model.safetensors`` (bf16 or fp32, cast to fp32). It
imports no torch and no ``safetensors`` package, because Pyodide has neither to spare: the file
is parsed here (``read_safetensors``, a few lines, since the format is a JSON header followed
by raw bytes) and the forward pass is numpy.

It is also an ``Engine`` (``lm/engine.py``), so ``Decoder`` runs the constrained decoding
rules over it unchanged. Decoding computes the output head for the last position only, which
saves most of a long prompt's head cost; ``step`` returns every position for tests.

Speed note: Pyodide's numpy is built without BLAS, so matrix products there are plain loops.
Weights stay in torch's ``(out, in)`` layout and products are ``x @ w.T``: a transposed view
costs nothing, and walks both operands along their contiguous axis.
"""

from __future__ import annotations

import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt

from askphysics.errors import ConfigError, LLMError
from askphysics.lm.config import ModelConfig, require_model_files, saved_config
from askphysics.lm.engine import Cache, Engine, Logits
from askphysics.lm.paths import TOKENIZER_FILE, WEIGHTS_FILE
from askphysics.lm.tokenizer import Tokenizer

Array = npt.NDArray[np.float32]

# The header of a safetensors file is JSON; the format caps it at 100 MB.
_MAX_HEADER = 100_000_000
# Stored dtype -> (numpy storage type, bytes per element).
_STORAGE = {"F32": ("<f4", 4), "F16": ("<f2", 2), "BF16": ("<u2", 2)}


def read_safetensors(
    data: bytes | bytearray | memoryview,
) -> tuple[dict[str, Array], dict[str, str]]:
    """Parse a safetensors file into float32 arrays and its string metadata.

    bf16 weights (how releases ship, ADR-012) become float32 by placing the 16 bits in the
    top half of a float32, which is exact. fp16 and fp32 are accepted too. The arrays are
    copies, so ``data`` can be freed afterwards.

    Raises:
        ConfigError: the file is malformed, or holds a dtype other than F32, F16, or BF16.
    """
    view = memoryview(data).cast("B")
    if len(view) < 8:
        raise ConfigError("not a safetensors file: shorter than its length prefix")
    (size,) = struct.unpack("<Q", view[:8])
    if size > _MAX_HEADER or 8 + size > len(view):
        raise ConfigError("not a safetensors file: the header length is wrong")
    try:
        header = json.loads(bytes(view[8 : 8 + size]))
    except ValueError as exc:
        raise ConfigError(f"not a safetensors file: the header is not JSON ({exc})") from exc
    body = view[8 + size :]
    metadata = {str(k): str(v) for k, v in header.pop("__metadata__", {}).items()}
    tensors: dict[str, Array] = {}
    for name, info in header.items():
        try:
            dtype, shape = info["dtype"], [int(n) for n in info["shape"]]
            begin, end = (int(n) for n in info["data_offsets"])
            storage, width = _STORAGE[dtype]
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"safetensors entry {name!r} is unreadable or not F32/F16/BF16"
            ) from exc
        if not 0 <= begin <= end <= len(body) or end - begin != math.prod(shape) * width:
            raise ConfigError(f"safetensors entry {name!r} does not fit its shape and offsets")
        raw = np.frombuffer(body[begin:end], dtype=storage)
        if dtype == "BF16":
            values = (raw.astype(np.uint32) << 16).view(np.float32)
        else:
            values = raw.astype(np.float32)
        tensors[name] = values.reshape(shape)
    return tensors, metadata


def parameter_shapes(config: ModelConfig) -> dict[str, tuple[int, ...]]:
    """Every weight the network reads, by its ``FermiLM`` parameter name."""
    d, hidden = config.d_model, config.mlp_hidden
    shapes: dict[str, tuple[int, ...]] = {"embed.weight": (config.vocab_size, d)}
    for i in range(config.n_layers):
        prefix = f"blocks.{i}."
        shapes[prefix + "attn_norm.weight"] = (d,)
        shapes[prefix + "attn.qkv.weight"] = (3 * d, d)
        shapes[prefix + "attn.out.weight"] = (d, d)
        shapes[prefix + "mlp_norm.weight"] = (d,)
        shapes[prefix + "mlp.gate.weight"] = (hidden, d)
        shapes[prefix + "mlp.up.weight"] = (hidden, d)
        shapes[prefix + "mlp.down.weight"] = (d, hidden)
    shapes["norm.weight"] = (d,)
    return shapes


def rope_tables(head_dim: int, length: int, theta: float) -> tuple[Array, Array]:
    """Cosine and sine tables of shape ``(length, head_dim // 2)``, as ``model.rope_tables``."""
    inv_freq = np.float32(1.0) / (
        np.float32(theta) ** (np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim))
    )
    angles = np.outer(np.arange(length, dtype=np.float32), inv_freq)
    return np.cos(angles), np.sin(angles)


def _linear(x: Array, weight: Array) -> Array:
    """``x @ weight.T`` for ``weight`` of shape ``(out, in)``, over any leading dimensions."""
    flat = x.reshape(-1, x.shape[-1]) @ weight.T
    return cast(Array, flat.reshape(*x.shape[:-1], weight.shape[0]))


def _rms_norm(x: Array, weight: Array, eps: float = 1e-6) -> Array:
    rms = np.float32(1.0) / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))
    return cast(Array, x * rms * weight)


def _rope(x: Array, cos: Array, sin: Array, start: int) -> Array:
    """Rotate ``x`` of shape ``(batch, heads, time, head_dim)`` by absolute position."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos[start : start + x.shape[-2]], sin[start : start + x.shape[-2]]
    return np.concatenate((x1 * c - x2 * s, x1 * s + x2 * c), axis=-1)


def _silu(x: Array) -> Array:
    with np.errstate(over="ignore"):  # exp(-x) overflows for very negative x, giving silu 0
        return cast(Array, x / (np.float32(1.0) + np.exp(-x)))


def _softmax(x: Array) -> Array:
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return cast(Array, e / np.sum(e, axis=-1, keepdims=True))


@dataclass(frozen=True)
class _Block:
    attn_norm: Array
    qkv: Array
    out: Array
    mlp_norm: Array
    gate: Array
    up: Array
    down: Array


class NumpyFermiLM(Engine):
    """A Fermi language model that runs on numpy alone.

    ``weights`` maps ``FermiLM`` parameter names to arrays (as ``read_safetensors`` returns
    them). The KV cache is a list of ``(keys, values)`` per layer, each ``(batch, heads, time,
    head_dim)``, and a step builds a new cache rather than editing the old one, so slices of an
    old cache stay valid.
    """

    device = "cpu"

    def __init__(self, config: ModelConfig, weights: Mapping[str, Array]) -> None:
        expected = parameter_shapes(config)
        if missing := sorted(set(expected) - set(weights)):
            raise ConfigError(f"weights are missing {', '.join(missing[:3])} ({len(missing)})")
        for name, shape in expected.items():
            if weights[name].shape != shape:
                raise ConfigError(
                    f"weight {name} has shape {weights[name].shape}, expected {shape}"
                )
        self.config = config
        self.embed = weights["embed.weight"]
        self.norm = weights["norm.weight"]
        self.blocks = [
            _Block(
                attn_norm=weights[f"blocks.{i}.attn_norm.weight"],
                qkv=weights[f"blocks.{i}.attn.qkv.weight"],
                out=weights[f"blocks.{i}.attn.out.weight"],
                mlp_norm=weights[f"blocks.{i}.mlp_norm.weight"],
                gate=weights[f"blocks.{i}.mlp.gate.weight"],
                up=weights[f"blocks.{i}.mlp.up.weight"],
                down=weights[f"blocks.{i}.mlp.down.weight"],
            )
            for i in range(config.n_layers)
        ]
        self.rope_cos, self.rope_sin = rope_tables(
            config.head_dim, config.context_length, config.rope_theta
        )

    # ------------------------------------------------------------------ the forward pass

    def step(self, ids: npt.NDArray[np.integer], past: Cache | None = None) -> tuple[Array, Cache]:
        """Logits for every position of ``ids`` (shape ``(batch, time)``), as ``FermiLM.step``.

        Call once with the whole prompt (``past=None``), then once per new token with ``ids`` of
        shape ``(batch, 1)``.
        """
        return self._run(ids, past, last_only=False)

    def _run(
        self, ids: npt.NDArray[np.integer], past: Cache | None, *, last_only: bool
    ) -> tuple[Array, Cache]:
        if past is not None and ids.shape[1] != 1:
            raise ValueError("with a KV cache, decode one token at a time")
        b, t = ids.shape
        start = 0 if past is None else past[0][0].shape[2]
        if start + t > self.config.context_length:
            raise ValueError(
                f"sequence of {start + t} tokens exceeds context {self.config.context_length}"
            )
        heads, head_dim = self.config.n_heads, self.config.head_dim
        x = self.embed[ids]
        new_past: Cache = []
        for i, block in enumerate(self.blocks):
            q, k, v = np.split(_linear(_rms_norm(x, block.attn_norm), block.qkv), 3, axis=-1)
            q, k, v = (z.reshape(b, t, heads, head_dim).transpose(0, 2, 1, 3) for z in (q, k, v))
            q = _rope(q, self.rope_cos, self.rope_sin, start)
            k = _rope(k, self.rope_cos, self.rope_sin, start)
            if past is not None:
                k = np.concatenate((past[i][0], k), axis=2)
                v = np.concatenate((past[i][1], v), axis=2)
            new_past.append((k, v))
            scores = (q @ k.transpose(0, 1, 3, 2)) * np.float32(1.0 / math.sqrt(head_dim))
            if past is None:  # a cached step has one query, which may see everything before it
                scores = np.where(
                    np.triu(np.ones((t, t), dtype=bool), 1), np.float32(-np.inf), scores
                )
            y = _softmax(scores) @ v
            x = x + _linear(y.transpose(0, 2, 1, 3).reshape(b, t, -1), block.out)
            h = _rms_norm(x, block.mlp_norm)
            x = x + _linear(_silu(_linear(h, block.gate)) * _linear(h, block.up), block.down)
        x = _rms_norm(x, self.norm)
        if last_only:
            x = x[:, -1:]
        return _linear(x, self.embed), new_past  # tied output head

    # ------------------------------------------------------------------ Engine

    def next_logits(self, tokens: Sequence[int], past: Cache | None) -> tuple[Logits, Cache]:
        if not tokens:
            raise ValueError("nothing to feed")
        logits, new_past = self._run(
            np.asarray([list(tokens)], dtype=np.int64), past, last_only=True
        )
        return logits[0, -1], new_past

    def generator(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def sample(self, masked: Logits, temperature: float, generator: np.random.Generator) -> int:
        scaled = masked.astype(np.float64) / temperature
        top = float(np.max(scaled))
        if top == -np.inf:
            raise LLMError("no token is allowed here")
        probs = np.exp(scaled - top)
        return int(generator.choice(probs.size, p=probs / probs.sum()))


def load_numpy_model(directory: Path) -> tuple[NumpyFermiLM, Tokenizer]:
    """Load a model saved by ``save_model`` (or the MLX trainer, or a download) into numpy.

    Raises:
        ConfigError: files are missing, malformed, or don't agree (the same checks as
            ``load_model``), or the task format version is different.
    """
    require_model_files(directory)
    tensors, metadata = read_safetensors((directory / WEIGHTS_FILE).read_bytes())
    config = saved_config(directory, metadata)
    return NumpyFermiLM(config, tensors), Tokenizer.load(directory / TOKENIZER_FILE)
