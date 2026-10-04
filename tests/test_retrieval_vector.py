import pytest

from askphysics.retrieval.base import Retriever, VectorStore
from askphysics.retrieval.vector import SqliteVecStore, VectorRetriever


def test_store_matches_protocol_but_is_a_stub() -> None:
    store = SqliteVecStore()
    assert isinstance(store, VectorStore)
    with pytest.raises(NotImplementedError):
        store.add(["a"], [[0.1, 0.2]], [{}])
    with pytest.raises(NotImplementedError):
        store.query([0.1, 0.2], k=1)


def test_vector_retriever_is_a_stub() -> None:
    retriever = VectorRetriever(SqliteVecStore())
    assert isinstance(retriever, Retriever)
    with pytest.raises(NotImplementedError):
        retriever.search("anything", k=3)
