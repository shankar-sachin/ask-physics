"""Runtime settings.

A frozen dataclass with environment-variable overrides. This avoids a
pydantic-settings dependency for a handful of fields. Environment variables
use the ``ASKPHYSICS_`` prefix (see ``.env.example``).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, cast, get_args

from askphysics.errors import ConfigError
from askphysics.lm.config import PRESETS

# auto: the Fermi models when any are installed, else the fake client (ADR-010).
Provider = Literal["auto", "fake", "fermi"]

ENV_PREFIX = "ASKPHYSICS_"


@dataclass(frozen=True)
class Settings:
    """Pipeline settings.

    Attributes:
        llm_provider: Which ``LLMClient`` to build. ``fake`` needs no weights, ``fermi``
            needs installed models, and ``auto`` picks ``fermi`` when any are installed.
        model: Force one Fermi model for every stage (``askphysics ask --model``). None
            routes per ADR-010: tellus classifies, solem plans and explains.
        top_k: Number of equations retrieved per question.
        temperature: Sampling temperature for the explain stage. Classify and
            plan always decode greedily (ADR-009).
        plan_attempts: Plans tried with the main planner before escalating (ADR-010).
        escalations: Extra plan attempts by celeste after those, when it is installed.
        device: Torch device for the Fermi models (mps, cuda, cpu); None picks the best.
        auto_pull: Download celeste's published weights the first time a question needs
            its escalation try (ADR-012). Off means celeste is used only when installed.
    """

    llm_provider: Provider = "auto"
    model: str | None = None
    top_k: int = 5
    temperature: float = 0.0
    plan_attempts: int = 5
    escalations: int = 1
    device: str | None = None
    auto_pull: bool = True

    def __post_init__(self) -> None:
        if self.llm_provider not in get_args(Provider):
            raise ConfigError(f"unknown llm_provider {self.llm_provider!r}")
        if self.top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}")
        if not 0.0 <= self.temperature <= 1.0:
            raise ConfigError(f"temperature must be between 0 and 1, got {self.temperature}")
        if self.model is not None and self.model not in PRESETS:
            raise ConfigError(f"unknown model {self.model!r}; choose from {', '.join(PRESETS)}")
        if self.plan_attempts < 1:
            raise ConfigError(f"plan_attempts must be at least 1, got {self.plan_attempts}")
        if self.escalations < 0:
            raise ConfigError(f"escalations can't be negative, got {self.escalations}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from ``ASKPHYSICS_*`` environment variables over the defaults."""
        env = os.environ if env is None else env
        defaults = cls()

        def raw(name: str) -> str | None:
            return env.get(ENV_PREFIX + name.upper())

        def number(name: str, kind: type[int] | type[float], default: float) -> float:
            value = raw(name)
            if value is None:
                return default
            try:
                return kind(value)
            except ValueError as exc:
                raise ConfigError(
                    f"invalid value for {ENV_PREFIX}{name.upper()}: {value!r}"
                ) from exc

        return cls(
            llm_provider=cast(Provider, raw("llm_provider") or defaults.llm_provider),
            model=raw("model") or defaults.model,
            top_k=int(number("top_k", int, defaults.top_k)),
            temperature=number("temperature", float, defaults.temperature),
            plan_attempts=int(number("plan_attempts", int, defaults.plan_attempts)),
            escalations=int(number("escalations", int, defaults.escalations)),
            device=raw("device") or defaults.device,
            auto_pull=_flag(raw("auto_pull"), defaults.auto_pull, "auto_pull"),
        )


def _flag(value: str | None, default: bool, name: str) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"invalid value for {ENV_PREFIX}{name.upper()}: {value!r}")
