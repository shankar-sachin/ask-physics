"""The ``LLMClient`` protocol (ADR-003).

Two methods, nothing provider-specific. ``complete_json`` must return a
validated instance of ``schema``, however the provider achieves that (native
structured outputs, JSON mode, or parse-and-retry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClient(Protocol):
    """Anything that can turn a system prompt plus a JSON user payload into text or a model."""

    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        """Return a validated ``schema`` instance.

        Raises:
            LLMError: the provider failed or refused.
            LLMResponseFormatError: the output could not be validated against ``schema``.
        """
        ...

    def complete_text(self, *, system: str, user: str) -> str:
        """Return free text (used only for the explain stage's prose).

        Raises:
            LLMError: the provider failed or refused.
        """
        ...


@dataclass(frozen=True)
class Roster:
    """The client for each stage (ADR-010). ``plan`` has one client per attempt, in order.

    The same client can appear several times: solem's five plan attempts are one model.
    """

    classify: LLMClient
    plan: tuple[LLMClient, ...]
    explain: LLMClient

    @classmethod
    def single(cls, llm: LLMClient) -> Roster:
        """One client for everything, with one plan attempt (the fake client, tests)."""
        return cls(classify=llm, plan=(llm,), explain=llm)


def client_name(client: LLMClient) -> str:
    """What an ``Answer`` records as the model behind a stage: ``fermi-solem-1``, ``fake``."""
    return str(getattr(client, "name", type(client).__name__))
