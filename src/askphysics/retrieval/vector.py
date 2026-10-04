"""Vector retrieval (v0.6). Stubs until then; see the v0.6 section of ``docs/ROADMAP.md``."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from askphysics.models import RetrievalResult


class SqliteVecStore:
    """``VectorStore`` backed by sqlite-vec (chosen in ``docs/ROADMAP.md``, v0.6)."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, str]],
    ) -> None:
        # TODO: Create a vec0 virtual table sized to the embedding dimension on first
        # insert, upsert rows keyed by id, and store metadata in a companion table
        # so hybrid search can join it with an FTS5 index.
        raise NotImplementedError("SqliteVecStore lands in v0.6")

    def query(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]:
        # TODO: Run a KNN query against the vec0 table and convert distances to
        # cosine similarity in [0, 1], most similar first.
        raise NotImplementedError("SqliteVecStore lands in v0.6")


class VectorRetriever:
    """``Retriever`` that embeds the query and searches a ``VectorStore``."""

    def __init__(self, store: SqliteVecStore) -> None:
        self.store = store

    def search(
        self, query: str, k: int, *, domains: Sequence[str] | None = None
    ) -> RetrievalResult:
        # TODO: Embed the query with a Fermi model's hidden states (no external API), query
        # the store for k equation ids, hydrate them from the DataStore, and return a
        # RetrievalResult. The v0.6 HybridRetriever fuses this with BM25 by rank fusion.
        raise NotImplementedError("VectorRetriever lands in v0.6")
