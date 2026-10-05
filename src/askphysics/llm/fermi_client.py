"""``FermiClient``: our own models behind the ``LLMClient`` protocol (ADR-009, ADR-010).

One client wraps one model. It reads the same JSON payloads the pipeline
sends any client, looks the cited equations and constants up in the
``DataStore`` (the payload's copies are only references), and answers with
the constrained decoders in ``lm/generate.py``: classify and plan decode
greedily, and the explanation can only spell numbers from its input.

Weights load on first use, so building a pipeline is cheap and a stage that
never runs never loads its model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import LLMError, LLMResponseFormatError
from askphysics.llm.base import Roster
from askphysics.llm.routing import plan_route
from askphysics.lm.checkpoints import load_model
from askphysics.lm.device import select_device
from askphysics.lm.generate import (
    Decoder,
    decode_classification,
    decode_explanation,
    decode_plan,
)
from askphysics.lm.paths import default_model_dir, installed_models
from askphysics.models import Classification, FermiAssumption, Plan

T = TypeVar("T", bound=BaseModel)


class FermiClient:
    """One Fermi model answering classify, plan, and explain payloads."""

    def __init__(
        self,
        name: str,
        data: DataStore,
        *,
        directory: Path | None = None,
        device: str | None = None,
        temperature: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.name = name
        self.data = data
        self.directory = directory or default_model_dir() / name
        self.device = device
        self.temperature = temperature
        self.seed = seed
        self._decoder: Decoder | None = None

    @property
    def decoder(self) -> Decoder:
        """The model's decoder, loading the weights on first use.

        Raises:
            ConfigError: the weights are missing or were trained on another task format.
        """
        if self._decoder is None:
            model, tokenizer = load_model(self.directory, select_device(self.device))
            self._decoder = Decoder(model, tokenizer)
        return self._decoder

    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        payload = _load(user)
        try:
            if schema is Classification:
                result: BaseModel = decode_classification(self.decoder, str(payload["question"]))
            elif schema is Plan:
                result = self._plan(payload)
            else:
                raise LLMResponseFormatError(
                    f"{self.name} has no task format for {schema.__name__}"
                )
        except ValidationError as exc:
            raise LLMResponseFormatError(f"{self.name} wrote an invalid {schema.__name__}") from exc
        except (KeyError, TypeError) as exc:
            raise LLMError(f"{self.name} got a payload it can't read: {exc}") from exc
        return schema.model_validate(result.model_dump())

    def complete_text(self, *, system: str, user: str) -> str:
        payload = _load(user)
        try:
            equations = [self.data.equations[i] for i in payload["equation_ids"]]
            return decode_explanation(
                self.decoder,
                str(payload["question"]),
                float(payload["result"]["value"]),
                str(payload["result"]["unit"]),
                equations,
                [str(a) for a in payload.get("assumptions", [])],
                [str(i) for i in payload.get("sanity", {}).get("issues", [])],
                temperature=self.temperature,
                seed=self.seed,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError(f"{self.name} got a payload it can't read: {exc}") from exc

    def _plan(self, payload: dict[str, Any]) -> Plan:
        equations = [self.data.equations[e["id"]] for e in payload["equations"]]
        constants = [self.data.constants[c["name"]] for c in payload.get("constants", [])]
        fermi = [FermiAssumption.model_validate(a) for a in payload.get("fermi_assumptions", [])]
        category = str(payload["classification"]["category"])
        return decode_plan(
            self.decoder, str(payload["question"]), category, equations, constants, fermi
        )


def _load(user: str) -> dict[str, Any]:
    try:
        payload = json.loads(user)
    except json.JSONDecodeError as exc:
        raise LLMError(f"payload is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMError("payload is not a JSON object")
    return payload


def build_roster(settings: Settings, data: DataStore, root: Path | None = None) -> Roster:
    """Clients for each stage per ADR-010, sharing one loaded model per name.

    Raises:
        ConfigError: no usable models are installed (see ``plan_route``).
    """
    root = root or default_model_dir()
    route = plan_route(
        installed_models(root),
        forced=settings.model,
        attempts=settings.plan_attempts,
        escalations=settings.escalations,
    )
    clients = {
        name: FermiClient(
            name,
            data,
            directory=root / name,
            device=settings.device,
            temperature=settings.temperature,
        )
        for name in route.models
    }
    return Roster(
        classify=clients[route.classify],
        plan=tuple(clients[name] for name in route.plan),
        explain=clients[route.explain],
    )


__all__ = ["FermiClient", "build_roster"]
