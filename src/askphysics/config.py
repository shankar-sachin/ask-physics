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

Provider = Literal["fake", "anthropic"]

ENV_PREFIX = "ASKPHYSICS_"


@dataclass(frozen=True)
class Settings:
    """Pipeline settings.

    Attributes:
        llm_provider: Which ``LLMClient`` to build. ``fake`` needs no API key.
        model: Model id passed to real providers.
        top_k: Number of equations retrieved per question.
        temperature: Sampling temperature, sent only to providers that accept
            it. Current Claude models reject sampling parameters (ADR-006).
    """

    llm_provider: Provider = "fake"
    model: str = "claude-opus-5-5"
    top_k: int = 5
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.llm_provider not in get_args(Provider):
            raise ConfigError(f"unknown llm_provider {self.llm_provider!r}")
        if self.top_k < 1:
            raise ConfigError(f"top_k must be at least 1, got {self.top_k}")
        if not 0.0 <= self.temperature <= 1.0:
            raise ConfigError(f"temperature must be between 0 and 1, got {self.temperature}")

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
        )
