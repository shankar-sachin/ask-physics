from __future__ import annotations

from collections.abc import Iterator

import pytest

from askphysics.config import Settings
from askphysics.data.loader import DataStore, load_all
from askphysics.llm.fake import FakeLLMClient
from askphysics.pipeline import Pipeline
from askphysics.retrieval.keyword import KeywordRetriever

DEMO_QUESTION = "How fast does a falling object hit the ground if it is dropped from 20 m?"


@pytest.fixture(autouse=True)
def no_installed_models(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Point the models dir at an empty folder: tests never load real weights (golden rule 6)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ASKPHYSICS_MODEL_DIR", str(tmp_path_factory.mktemp("no-models")))
        yield


@pytest.fixture(scope="session")
def store() -> DataStore:
    return load_all()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    return FakeLLMClient()


@pytest.fixture
def retriever(store: DataStore) -> KeywordRetriever:
    return KeywordRetriever(store.equations.values(), store.examples.values())


@pytest.fixture
def pipeline(store: DataStore, fake_llm: FakeLLMClient, retriever: KeywordRetriever) -> Pipeline:
    return Pipeline(llm=fake_llm, retriever=retriever, data=store, settings=Settings())
