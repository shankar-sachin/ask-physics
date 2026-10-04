"""``FakeLLMClient``: deterministic, offline stand-in for a real LLM.

Used by every test and by the CLI's default path, so nothing needs an API
key. It does just enough "language work" (keyword classification, pulling a
height out of the question) to exercise the pipeline, and it refuses to make
things up: if it has no canned plan for a question, it raises ``LLMError``
and the pipeline degrades gracefully.

Canned plans exist only for the demo question shape (an object dropped from
a stated height), which is deliberately not in the eval set (``docs/EVALS.md``).
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from askphysics.errors import LLMError, LLMResponseFormatError
from askphysics.models import Classification, Plan

T = TypeVar("T", bound=BaseModel)

_OUT_OF_SCOPE: list[tuple[re.Pattern[str], str, str]] = [
    (
        re.compile(r"before the big bang"),
        "Asks about conditions before the Big Bang, outside the domain where current "
        "physics makes predictions.",
        "What was the temperature of the universe when the cosmic microwave background "
        "was released?",
    ),
    (
        re.compile(r"\b(colou?r|blue|red|green)\b.*\bweigh"),
        "Category error: a color is a property of light, not an object with mass.",
        "How much momentum does a beam of blue light carry, for example from a 1 W laser?",
    ),
    (
        re.compile(r"meaning of life"),
        "Not a physics question.",
        "How much energy does a human body use in a day?",
    ),
]
_FERMI = re.compile(r"how many .* (would|does) it take|if everyone|rubber duck|bananas? worth")
_DOMAINS: dict[str, tuple[str, ...]] = {
    "kinematics": ("fall", "drop", "fast", "speed", "accelerat"),
    "dynamics": ("force", "push", "weigh"),
    "energy": ("energy", "joule", "banana"),
    "momentum": ("momentum", "stop", "collision", "crash"),
    "gravitation": ("gravit", "planet", "orbit", "earth"),
    "electromagnetism": ("current", "voltage", "resist", "circuit", "battery"),
    "thermodynamics": ("gas", "pressure", "temperature", "heat"),
}
_HEIGHT = re.compile(r"(\d+(?:\.\d+)?)\s*(m|meters?|metres?|km|ft|feet)\b")
_UNIT_ALIASES = {"meter": "m", "meters": "m", "metre": "m", "metres": "m", "feet": "ft"}


class FakeLLMClient:
    """Deterministic ``LLMClient``. Records every call in ``calls`` for test inspection."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(("json:" + schema.__name__, user))
        payload = _load_payload(user)
        question = str(payload.get("question", "")).lower()
        if schema is Classification:
            data = self._classify(question)
        elif schema is Plan:
            data = self._plan(question, payload)
        else:
            raise LLMResponseFormatError(f"FakeLLMClient has no canned {schema.__name__}")
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMResponseFormatError(f"canned {schema.__name__} is invalid: {exc}") from exc

    def complete_text(self, *, system: str, user: str) -> str:
        self.calls.append(("text", user))
        payload = _load_payload(user)
        result = payload.get("result") or {}
        ids = payload.get("equation_ids") or []
        assumptions = payload.get("assumptions") or []
        cited = ", ".join(f"[{i}]" for i in ids) or "no equations"
        text = f"Using {cited}, the computed result is {result.get('value')} {result.get('unit')}."
        if assumptions:
            text += " This assumes: " + "; ".join(str(a) for a in assumptions) + "."
        return text + " (Explanation written by FakeLLMClient.)"

    # ------------------------------------------------------------------ canned behavior

    @staticmethod
    def _classify(question: str) -> dict[str, Any]:
        for pattern, reasoning, closest in _OUT_OF_SCOPE:
            if pattern.search(question):
                return {
                    "category": "out_of_scope",
                    "reasoning": reasoning,
                    "domains": [],
                    "closest_answerable": closest,
                }
        domains = [d for d, words in _DOMAINS.items() if any(w in question for w in words)]
        if _FERMI.search(question):
            return {
                "category": "fermi",
                "reasoning": "Physically meaningful but needs estimated everyday quantities.",
                "domains": domains,
            }
        return {
            "category": "standard",
            "reasoning": "Well-posed question; values are given or are standard constants.",
            "domains": domains,
        }

    @staticmethod
    def _plan(question: str, payload: dict[str, Any]) -> dict[str, Any]:
        retrieved = {e.get("id") for e in payload.get("equations", []) if isinstance(e, dict)}
        height = _HEIGHT.search(question)
        is_drop = any(w in question for w in ("drop", "fall", "falling"))
        if not (is_drop and height and "kin_v_squared" in retrieved):
            raise LLMError("FakeLLMClient has no canned plan for this question")
        value, unit = float(height.group(1)), height.group(2)
        return {
            "target": "v",
            "unknowns": ["v"],
            "known_values": [
                {"symbol": "v0", "value": 0.0, "unit": "m/s", "origin": "assumption"},
                {"symbol": "a", "value": 9.80665, "unit": "m/s^2", "origin": "constant"},
                {
                    "symbol": "d",
                    "value": value,
                    "unit": _UNIT_ALIASES.get(unit, unit),
                    "origin": "given",
                },
            ],
            "equation_ids": ["kin_v_squared"],
            "assumptions": [
                'Released from rest ("dropped")',
                "Air resistance is negligible",
                "Gravitational acceleration is constant at standard g",
            ],
            "strategy": "Use v^2 = v0^2 + 2*a*d with a = g and take the positive root for v.",
        }


def _load_payload(user: str) -> dict[str, Any]:
    try:
        payload = json.loads(user)
    except json.JSONDecodeError as exc:
        raise LLMResponseFormatError("FakeLLMClient expects a JSON user payload") from exc
    if not isinstance(payload, dict):
        raise LLMResponseFormatError("FakeLLMClient expects a JSON object payload")
    return payload
