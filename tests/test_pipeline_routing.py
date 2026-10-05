"""The ADR-010 router inside ``Pipeline.run``: retries, escalation, and the model record."""

import json
import subprocess
import sys
from pathlib import Path
from typing import TypeVar

import pytest
from pydantic import BaseModel
from tests.conftest import DEMO_QUESTION

from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import ConfigError, LLMError
from askphysics.llm.base import Roster
from askphysics.llm.fake import FakeLLMClient
from askphysics.models import Plan
from askphysics.pipeline import Pipeline
from askphysics.retrieval.keyword import KeywordRetriever

T = TypeVar("T", bound=BaseModel)


class FlakyPlanner(FakeLLMClient):
    """Fails its first ``failures`` plans, then plans like the fake client."""

    def __init__(self, name: str, failures: int) -> None:
        super().__init__()
        self.name = name
        self.failures = failures
        self.plans = 0
        self.seen: list[list[str]] = []  # equation ids each plan prompt listed, in order

    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
        if schema is Plan:
            self.plans += 1
            self.seen.append([e["id"] for e in json.loads(user)["equations"]])
            if self.plans <= self.failures:
                raise LLMError(f"{self.name} fumbled")
        return super().complete_json(system=system, user=user, schema=schema)


def _pipeline(store: DataStore, retriever: KeywordRetriever, roster: Roster) -> Pipeline:
    return Pipeline(
        llm=roster.classify, retriever=retriever, data=store, settings=Settings(), roster=roster
    )


def test_every_stage_records_its_model(pipeline: Pipeline) -> None:
    answer = pipeline.run(DEMO_QUESTION)
    assert answer.models == {"classify": "fake", "plan": "fake", "explain": "fake"}
    assert answer.plan_attempts == 1


def test_a_failed_plan_is_retried(store: DataStore, retriever: KeywordRetriever) -> None:
    solem = FlakyPlanner("solem", failures=2)
    roster = Roster(classify=FakeLLMClient(), plan=(solem,) * 5, explain=solem)
    answer = _pipeline(store, retriever, roster).run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert answer.plan_attempts == 3 and solem.plans == 3
    assert answer.models == {"classify": "fake", "plan": "solem", "explain": "solem"}


def test_celeste_rescues_solem(store: DataStore, retriever: KeywordRetriever) -> None:
    solem = FlakyPlanner("solem", failures=99)
    celeste = FlakyPlanner("celeste", failures=0)
    roster = Roster(classify=FakeLLMClient(), plan=(solem,) * 5 + (celeste,), explain=solem)
    answer = _pipeline(store, retriever, roster).run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert answer.plan_attempts == 6 and answer.models["plan"] == "celeste"
    assert answer.final_value == pytest.approx(19.8057, rel=1e-5)


def test_every_attempt_failing_degrades(store: DataStore, retriever: KeywordRetriever) -> None:
    solem = FlakyPlanner("solem", failures=99)
    roster = Roster(classify=FakeLLMClient(), plan=(solem,) * 3, explain=solem)
    answer = _pipeline(store, retriever, roster).run(DEMO_QUESTION)
    assert answer.status == "degraded" and answer.plan_attempts == 3
    assert any("All 3 plan attempts failed" in c for c in answer.caveats)
    assert "plan" not in answer.models


def test_retries_see_the_equations_in_a_new_order(
    store: DataStore, retriever: KeywordRetriever
) -> None:
    solem = FlakyPlanner("solem", failures=2)
    roster = Roster(classify=FakeLLMClient(), plan=(solem,) * 3, explain=solem)
    _pipeline(store, retriever, roster).run(DEMO_QUESTION)
    orders = solem.seen
    assert len(orders) == 3
    assert orders[1] == orders[0][1:] + orders[0][:1]
    assert orders[2] == orders[0][2:] + orders[0][:2]


def test_the_template_is_recorded_when_prose_fails(
    store: DataStore, retriever: KeywordRetriever
) -> None:
    class Mute(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            return "   "

    roster = Roster.single(Mute())
    answer = _pipeline(store, retriever, roster).run(DEMO_QUESTION)
    assert answer.models["explain"] == "template"
    assert answer.explanation.startswith("Using kin_v_squared")


def test_auto_falls_back_to_fake_without_models(store: DataStore) -> None:
    assert isinstance(Pipeline.from_settings(Settings(), data=store).llm, FakeLLMClient)


def test_fermi_without_models_says_so(store: DataStore) -> None:
    with pytest.raises(ConfigError, match="no Fermi models are installed"):
        Pipeline.from_settings(Settings(llm_provider="fermi"), data=store)


def test_auto_ignores_a_lone_test_model(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    luna = tmp_path / "fermi-luna-1"
    luna.mkdir()
    for name in ("model.safetensors", "config.json", "tokenizer.json"):
        (luna / name).write_text("{}")
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(tmp_path))
    assert isinstance(Pipeline.from_settings(Settings(), data=store).llm, FakeLLMClient)


def test_the_website_path_never_imports_torch() -> None:
    code = "import sys, askphysics.web; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
