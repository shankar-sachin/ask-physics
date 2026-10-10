"""Run a torch ``FermiLM`` behind the decoder's ``Engine`` seam (ADR-021)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F

from askphysics.lm.engine import Cache, Engine, Logits
from askphysics.lm.model import FermiLM


class TorchEngine(Engine):
    """A ``FermiLM`` on its own device. The model's ``step`` is looked up on every call."""

    def __init__(self, model: FermiLM) -> None:
        self.model = model
        self.config = model.config
        self._torch_device = next(model.parameters()).device
        self.device = self._torch_device.type

    def next_logits(self, tokens: Sequence[int], past: Cache | None) -> tuple[Logits, Cache]:
        if not tokens:
            raise ValueError("nothing to feed")
        ids = torch.tensor([list(tokens)], device=self._torch_device)
        logits, new_past = self.model.step(ids, past)
        last = logits[0, -1].float().cpu().numpy()
        return np.array(last, dtype=np.float32), new_past

    def generator(self, seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    def sample(self, masked: Logits, temperature: float, generator: torch.Generator) -> int:
        probs = F.softmax(torch.from_numpy(masked) / temperature, dim=-1)
        return int(torch.multinomial(probs, 1, generator=generator))
