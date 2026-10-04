"""``AnthropicClient``: the first real provider (stub until v0.3).

Requires the optional extra: ``pip install -e ".[anthropic]"``. The SDK is
imported lazily so the package works without it.
"""

from __future__ import annotations

import os
from typing import TypeVar

from pydantic import BaseModel

from askphysics.errors import ConfigError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "claude-opus-5-5"


class AnthropicClient:
    """``LLMClient`` backed by the Anthropic Messages API."""

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ConfigError(
                "ANTHROPIC_API_KEY is not set; copy .env.example to .env or use the fake client"
            )

    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        # TODO: Call anthropic.Anthropic().messages.parse(model=self.model, max_tokens=16000,
        # system=system, messages=[{"role": "user", "content": user}], output_format=schema)
        # and return response.parsed_output. Check stop_reason == "refusal" first and raise
        # LLMError. Never send temperature: current Claude models reject it (ADR-006).
        raise NotImplementedError("AnthropicClient lands in v0.3")

    def complete_text(self, *, system: str, user: str) -> str:
        # TODO: Call messages.create with the same system/user shape, join the text blocks
        # of the response, and map SDK errors (RateLimitError, APIStatusError,
        # APIConnectionError, most specific first) to LLMError.
        raise NotImplementedError("AnthropicClient lands in v0.3")
