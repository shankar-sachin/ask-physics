"""The Fermi transformer, written from scratch (``docs/MODELS.md``).

Decoder-only, pre-norm: RMSNorm, rotary position embeddings, causal
self-attention through ``scaled_dot_product_attention``, SwiGLU MLP, no
biases, output head tied to the token embeddings. The parameter count must
match ``ModelConfig.num_parameters`` exactly (pinned by tests).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from askphysics.lm.config import ModelConfig

IGNORE_INDEX = -100

# Per-layer (keys, values), each of shape (batch, heads, time, head_dim).
KVCache = list[tuple[Tensor, Tensor]]


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * rms).type_as(x) * self.weight


def rope_tables(head_dim: int, length: int, theta: float) -> tuple[Tensor, Tensor]:
    """Cosine and sine tables of shape ``(length, head_dim // 2)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, start: int = 0) -> Tensor:
    """Rotate ``x`` of shape ``(batch, heads, time, head_dim)`` by absolute position.

    ``start`` is the position of the first time step (non-zero when decoding
    with a KV cache). Uses the rotate-half form.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    end = start + x.shape[-2]
    cos = cos[start:end].to(x.dtype)
    sin = sin[start:end].to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, past: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        b, t, d = x.shape
        start = 0 if past is None else past[0].shape[2]
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v))
        q, k = apply_rope(q, cos, sin, start), apply_rope(k, cos, sin, start)
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        # With a cache, new tokens may attend to everything before them, so no mask is needed
        # as long as only one token is added per step (enforced in FermiLM.step).
        y = F.scaled_dot_product_attention(q, k, v, is_causal=past is None)
        return self.out(y.transpose(1, 2).reshape(b, t, d)), (k, v)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.up = nn.Linear(config.d_model, config.mlp_hidden, bias=False)
        self.down = nn.Linear(config.mlp_hidden, config.d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))  # type: ignore[no-any-return]


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = Attention(config)
        self.mlp_norm = RMSNorm(config.d_model)
        self.mlp = SwiGLU(config)

    def forward(
        self, x: Tensor, cos: Tensor, sin: Tensor, past: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        attn_out, kv = self.attn(self.attn_norm(x), cos, sin, past)
        x = x + attn_out
        return x + self.mlp(self.mlp_norm(x)), kv


class FermiLM(nn.Module):
    """A Fermi language model. ``forward`` returns logits, and the loss when targets are given."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model)
        cos, sin = rope_tables(config.head_dim, config.context_length, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2 style init: N(0, 0.02), residual output projections scaled by depth."""
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, param in self.named_parameters():
            if name.endswith("norm.weight"):
                nn.init.ones_(param)
            elif name.endswith(("attn.out.weight", "mlp.down.weight")):
                nn.init.normal_(param, mean=0.0, std=residual_std)
            else:
                nn.init.normal_(param, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: Tensor, targets: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        """Run the model.

        Args:
            ids: token ids, shape ``(batch, time)``, time at most ``context_length``.
            targets: next-token ids, same shape; ``IGNORE_INDEX`` positions don't count.

        Returns:
            ``(logits, loss)`` with logits of shape ``(batch, time, vocab_size)``.
        """
        logits, _ = self._run(ids, None)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )
        return logits, loss

    @torch.no_grad()
    def step(self, ids: Tensor, past: KVCache | None = None) -> tuple[Tensor, KVCache]:
        """Incremental decoding: logits for ``ids`` given a KV cache of earlier tokens.

        Call once with the whole prompt (``past=None``), then once per new
        token with ``ids`` of shape ``(batch, 1)``.
        """
        if past is not None and ids.shape[1] != 1:
            raise ValueError("with a KV cache, decode one token at a time")
        return self._run(ids, past)

    def _run(self, ids: Tensor, past: KVCache | None) -> tuple[Tensor, KVCache]:
        start = 0 if past is None else past[0][0].shape[2]
        if start + ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"sequence of {start + ids.shape[1]} tokens exceeds context "
                f"{self.config.context_length}"
            )
        cos: Tensor = self.rope_cos  # type: ignore[assignment]
        sin: Tensor = self.rope_sin  # type: ignore[assignment]
        x = self.embed(ids)
        new_past: KVCache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, cos, sin, None if past is None else past[i])
            new_past.append(kv)
        logits = F.linear(self.norm(x), self.embed.weight)  # tied output head
        return logits, new_past
