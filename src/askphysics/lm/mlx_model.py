"""The Fermi transformer in MLX, for training on Apple Silicon (ADR-019).

The same network as ``askphysics.lm.model.FermiLM``: same math, the same
parameter names and shapes, so weights move between the two through
``export_params`` and ``import_params`` and the safetensors files are
identical. Nothing outside ``lm/mlx_*`` imports this module, so mlx is only
needed for training on a Mac (or ``mlx[cpu]`` in tests).

Compute precision: parameters are always fp32. ``dtype`` in the forward pass
is the precision of the matmuls and attention (bf16 under ``precision=bf16``);
normalization and the residual stream stay fp32, as they do under torch
autocast, and the loss is computed in fp32.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import mlx.core as mx
import numpy as np

# Imported from their modules: mlx's package re-exports them through star imports, which
# strict mypy does not follow.
from mlx.nn.layers.base import Module
from mlx.nn.layers.embedding import Embedding
from mlx.nn.layers.linear import Linear
from mlx.utils import tree_flatten, tree_unflatten

from askphysics.errors import ConfigError
from askphysics.lm.config import ModelConfig
from askphysics.lm.model import IGNORE_INDEX


def _leaves(tree: Any) -> list[tuple[str, mx.array]]:
    """The arrays of a parameter tree as (dotted name, array) pairs, in tree order."""
    return cast(list[tuple[str, mx.array]], tree_flatten(tree))


class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        x32 = x.astype(mx.float32)
        rms = mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + self.eps)
        return (x32 * rms).astype(x.dtype) * self.weight


def rope_tables(head_dim: int, length: int, theta: float) -> tuple[mx.array, mx.array]:
    """Cosine and sine tables of shape ``(length, head_dim // 2)``, in fp32."""
    inv_freq = 1.0 / (theta ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
    angles = mx.arange(length, dtype=mx.float32)[:, None] * inv_freq[None, :]
    return mx.cos(angles), mx.sin(angles)


def _rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotary embedding of ``x`` (batch, heads, time, head_dim), rotate-half form."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    c, s = cos.astype(x.dtype), sin.astype(x.dtype)
    return mx.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)


def _project(x: mx.array, weight: mx.array, dtype: mx.Dtype) -> mx.array:
    """``x @ weight.T`` in ``dtype``; ``weight`` is (out, in), as in torch's Linear."""
    return x.astype(dtype) @ weight.astype(dtype).T


class Attention(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = Linear(config.d_model, config.d_model, bias=False)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, dtype: mx.Dtype) -> mx.array:
        b, t, d = x.shape
        qkv = _project(x, self.qkv.weight, dtype)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q, k, v = (
            z.reshape(b, t, self.n_heads, self.head_dim).transpose(0, 2, 1, 3) for z in (q, k, v)
        )
        q, k = _rotate(q, cos, sin), _rotate(k, cos, sin)
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask="causal")
        return _project(y.transpose(0, 2, 1, 3).reshape(b, t, d), self.out.weight, dtype)


class SwiGLU(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = Linear(config.d_model, config.mlp_hidden, bias=False)
        self.up = Linear(config.d_model, config.mlp_hidden, bias=False)
        self.down = Linear(config.mlp_hidden, config.d_model, bias=False)

    def __call__(self, x: mx.array, dtype: mx.Dtype) -> mx.array:
        gate = _project(x, self.gate.weight, dtype)
        hidden = gate * mx.sigmoid(gate) * _project(x, self.up.weight, dtype)
        return _project(hidden, self.down.weight, dtype)


class Block(Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, dtype: mx.Dtype) -> mx.array:
        x = x + self.attn(self.attn_norm(x), cos, sin, dtype)
        return x + self.mlp(self.mlp_norm(x), dtype)


class FermiLM(Module):
    """A Fermi language model in MLX. Call it for logits, or ``loss`` for the mean loss."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = Embedding(config.vocab_size, config.d_model)
        self.blocks = [Block(config) for _ in range(config.n_layers)]
        self.norm = RMSNorm(config.d_model)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """The same GPT-2 style init as the torch model: N(0, 0.02), residual outputs scaled."""
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        fresh: list[tuple[str, mx.array]] = []
        for name, param in _leaves(self.parameters()):
            if name.endswith("norm.weight"):
                value = mx.ones(param.shape)
            elif name.endswith(("attn.out.weight", "mlp.down.weight")):
                value = mx.random.normal(param.shape, scale=residual_std)
            else:
                value = mx.random.normal(param.shape, scale=0.02)
            fresh.append((name, value))
        self.update(tree_unflatten(fresh))

    def num_parameters(self) -> int:
        total = 0
        for _, p in _leaves(self.parameters()):
            total += int(p.size)
        return total

    def __call__(
        self, ids: mx.array, dtype: mx.Dtype = mx.float32, checkpoint_blocks: bool = False
    ) -> mx.array:
        """Logits of shape ``(batch, time, vocab_size)`` for ``ids`` of shape (batch, time).

        With ``checkpoint_blocks``, each block's activations are recomputed in the backward
        pass instead of being stored (see ``_checkpointed_block``).
        """
        t = ids.shape[1]
        if t > self.config.context_length:
            raise ValueError(f"sequence of {t} tokens exceeds context {self.config.context_length}")
        cos, sin = rope_tables(self.config.head_dim, t, self.config.rope_theta)
        x = self.embed(ids)
        for block in self.blocks:
            if checkpoint_blocks:
                x = _checkpointed_block(block, x, cos, sin, dtype)
            else:
                x = block(x, cos, sin, dtype)
        return _project(self.norm(x), self.embed.weight, dtype)  # tied output head

    def loss(
        self,
        ids: mx.array,
        targets: mx.array,
        dtype: mx.Dtype = mx.float32,
        checkpoint_blocks: bool = False,
    ) -> mx.array:
        """Mean cross-entropy over target tokens; ``IGNORE_INDEX`` positions don't count."""
        return cross_entropy(self(ids, dtype, checkpoint_blocks), targets)


def _checkpointed_block(
    block: Block, x: mx.array, cos: mx.array, sin: mx.array, dtype: mx.Dtype
) -> mx.array:
    """Run ``block`` under ``mx.checkpoint``: its intermediate activations (the attention and
    MLP inputs and outputs) are dropped after the forward pass and recomputed in the backward
    pass, so only the block's input is kept per layer. The parameters go through the
    checkpointed function as arguments, so gradients still reach them (the mlx-lm pattern)."""

    def inner(params: Any, h: mx.array, c: mx.array, s: mx.array) -> mx.array:
        block.update(params)
        return block(h, c, s, dtype)

    return mx.checkpoint(inner)(block.parameters(), x, cos, sin)


def cross_entropy(logits: mx.array, targets: mx.array) -> mx.array:
    """Mean negative log-likelihood over the non-ignored targets, computed in fp32.

    The logits are upcast once, to fp32 (the one (batch, time, vocab) tensor the loss needs).
    The log-softmax is not materialized: the target logit is gathered from the upcast logits
    and subtracted from their logsumexp, one reduction per row.
    """
    vocab = logits.shape[-1]
    flat = logits.astype(mx.float32).reshape(-1, vocab)
    labels = targets.reshape(-1)
    valid = (labels != IGNORE_INDEX).astype(mx.float32)
    safe = mx.where(labels != IGNORE_INDEX, labels, 0)
    picked = mx.take_along_axis(flat, safe[:, None], axis=-1)[:, 0]
    nll = mx.logsumexp(flat, axis=-1) - picked
    return mx.sum(nll * valid) / mx.maximum(mx.sum(valid), 1.0)


def export_params(model: FermiLM) -> dict[str, np.ndarray]:
    """Every parameter as a numpy fp32 array, under the same names as the torch state_dict."""
    return {name: np.array(p, dtype=np.float32) for name, p in _leaves(model.parameters())}


def import_params(model: FermiLM, params: Mapping[str, np.ndarray]) -> None:
    """Set every parameter of ``model`` from ``params`` (as from ``export_params``).

    Raises:
        ConfigError: the names or shapes don't match this model's.
    """
    expected = {name: p.shape for name, p in _leaves(model.parameters())}
    if set(params) != set(expected):
        missing, extra = sorted(set(expected) - set(params)), sorted(set(params) - set(expected))
        raise ConfigError(f"weights do not match the model: missing {missing}, unexpected {extra}")
    for name, arr in params.items():
        if tuple(arr.shape) != tuple(expected[name]):
            raise ConfigError(
                f"{name} has shape {tuple(arr.shape)}, the model needs {expected[name]}"
            )
    model.update(
        tree_unflatten(
            [(name, mx.array(np.asarray(arr, dtype=np.float32))) for name, arr in params.items()]
        )
    )
