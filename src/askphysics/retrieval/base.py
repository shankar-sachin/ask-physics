"""Retrieval protocols. Implementations do not inherit from these; they only match the shape."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from askphysics.models import RetrievalResult


@runtime_checkable
class Retriever(Protocol):
    """Finds the equations and worked examples most relevant to a query."""

    def search(
        self, query: str, k: int, *, domains: Sequence[str] | None = None
    ) -> RetrievalResult:
        """Return up to ``k`` equations (and examples), best first, scores in [0, 1].

        ``domains`` is a hint from the classifier. Implementations may boost
        matching domains but must not drop other results on that basis alone.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores embedding vectors by id and returns nearest neighbours."""

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, str]],
    ) -> None:
        """Insert or replace vectors. All three sequences must have the same length."""
        ...

    def query(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(id, similarity)`` pairs, most similar first."""
        ...
