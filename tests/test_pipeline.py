from typing import TypeVar

import pytest
from pydantic import BaseModel
from tests.conftest import DEMO_QUESTION

from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import LLMError, PlanValidationError
from askphysics.llm.fake import FakeLLMClient
from askphysics.models import (
    Classification,
    ComputeResult,
    KnownValue,
    Plan,
    Question,
    RetrievalResult,
)
from askphysics.pipeline import (
    LIMIT_CASE_CAVEAT,
    Pipeline,
    check_limit_cases,
    compute,
    sanity_check,
    score_confidence,
    validate_plan,
)
from askphysics.retrieval.keyword import KeywordRetriever

T = TypeVar("T", bound=BaseModel)


def _plan(**overrides: object) -> Plan:
    data: dict[str, object] = {
        "target": "v",
        "unknowns": ["v"],
        "known_values": [
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=9.80665, unit="m/s^2", origin="constant"),
            KnownValue(symbol="d", value=20, unit="m", origin="given"),
        ],
        "equation_ids": ["kin_v_squared"],
        "assumptions": ["No drag"],
        "strategy": "s",
    }
    data.update(overrides)
    return Plan.model_validate(data)


# ------------------------------------------------------------------ end to end


def test_end_to_end_standard_question(pipeline: Pipeline) -> None:
    answer = pipeline.run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert answer.category == "standard"
    assert answer.final_value == pytest.approx(19.8057, rel=1e-5)
    assert answer.unit == "meter / second"
    assert [e.id for e in answer.equations_used] == ["kin_v_squared"]
    assert answer.assumptions
    assert answer.confidence.label == "high"
    assert 0.75 <= answer.confidence.score <= 1.0
    assert LIMIT_CASE_CAVEAT in answer.caveats
    assert "[kin_v_squared]" in answer.explanation


def test_out_of_scope_is_refused_with_redirect(pipeline: Pipeline) -> None:
    answer = pipeline.run("How much does the color blue weigh?")
    assert answer.status == "refused"
    assert answer.final_value is None
    assert "Category error" in answer.explanation
    assert "close question" in answer.explanation


def test_fermi_without_planner_degrades_honestly(pipeline: Pipeline) -> None:
    answer = pipeline.run("How many rubber ducks would it take to stop a freight train?")
    assert answer.status == "degraded"
    assert answer.category == "fermi"
    assert answer.final_value is None
    assert "v0.5" in answer.explanation


def test_retrieval_miss_degrades(pipeline: Pipeline) -> None:
    answer = pipeline.run("zxqv blorp?")
    assert answer.status == "degraded"
    assert any("retrieve stage failed" in c for c in answer.caveats)


def test_classifier_failure_falls_back_to_standard(
    store: DataStore, retriever: KeywordRetriever
) -> None:
    class BrokenClassifier(FakeLLMClient):
        def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
            if schema is Classification:
                raise LLMError("provider down")
            return super().complete_json(system=system, user=user, schema=schema)

    p = Pipeline(llm=BrokenClassifier(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert any("classify stage failed" in c for c in answer.caveats)


def test_the_llm_never_supplies_the_number(store: DataStore, retriever: KeywordRetriever) -> None:
    class LyingExplainer(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            return "The answer is 42 m/s."

    p = Pipeline(llm=LyingExplainer(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.final_value == pytest.approx(19.8057, rel=1e-5)


def test_explain_falls_back_to_template(store: DataStore, retriever: KeywordRetriever) -> None:
    class MuteExplainer(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            raise LLMError("no prose today")

    p = Pipeline(llm=MuteExplainer(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.explanation.startswith("Using kin_v_squared, v = 19.8057")


def test_from_settings_builds_fake_pipeline(store: DataStore) -> None:
    p = Pipeline.from_settings(Settings(), data=store)
    assert isinstance(p.llm, FakeLLMClient)
    assert p.run(DEMO_QUESTION).status == "answered"


# ------------------------------------------------------------------ stages


def test_validate_plan_rejects_unretrieved_equations(retriever: KeywordRetriever) -> None:
    retrieval = retriever.search(DEMO_QUESTION, k=3)
    with pytest.raises(PlanValidationError, match="not retrieved"):
        validate_plan(_plan(equation_ids=["made_up_equation"]), retrieval)


def test_validate_plan_rejects_empty_and_bad_units(retriever: KeywordRetriever) -> None:
    retrieval = retriever.search(DEMO_QUESTION, k=3)
    with pytest.raises(PlanValidationError, match="no equations"):
        validate_plan(_plan(equation_ids=[]), retrieval)
    bad = KnownValue(symbol="d", value=20, unit="blorps", origin="given")
    with pytest.raises(PlanValidationError, match="invalid units"):
        validate_plan(_plan(known_values=[bad]), retrieval)


def test_compute_multi_equation_is_a_stub(store: DataStore) -> None:
    with pytest.raises(NotImplementedError):
        compute(_plan(equation_ids=["kin_v_squared", "kin_v_at"]), data=store)


def test_sanity_flags_out_of_range_magnitude(store: DataStore) -> None:
    plan = _plan()
    result = ComputeResult(
        target="v", value=1e9, unit="meter / second", symbolic_solution="", substitutions={}
    )
    report = sanity_check(plan, result, data=store)
    assert report.dimensions_ok
    assert report.magnitude_ok is False
    assert report.issues


def test_sanity_flags_wrong_dimensions(store: DataStore) -> None:
    wrong = KnownValue(symbol="d", value=20, unit="s", origin="given")
    plan = _plan(known_values=[wrong])
    result = ComputeResult(
        target="v", value=19.8, unit="meter / second", symbolic_solution="", substitutions={}
    )
    report = sanity_check(plan, result, data=store)
    assert not report.dimensions_ok


def test_limit_cases_are_a_stub(store: DataStore) -> None:
    with pytest.raises(NotImplementedError):
        check_limit_cases(store.equations["kinetic_energy"])


def test_unused_stage_inputs_are_typed() -> None:
    # Question and RetrievalResult are the public stage payloads.
    assert Question(text="q").text == "q"
    assert RetrievalResult(query="q").equation_ids == []


# ------------------------------------------------------------------ confidence


def test_confidence_formula_matches_plan_md() -> None:
    c = score_confidence(
        retrieval_score=0.67,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=3,
        category="standard",
    )
    assert c.score == pytest.approx(0.82, abs=0.01)
    assert c.label == "high"


def test_dimension_failure_caps_confidence() -> None:
    c = score_confidence(
        retrieval_score=1.0,
        dimensions_ok=False,
        magnitude_ok=True,
        n_assumptions=0,
        category="standard",
    )
    assert c.score <= 0.2
    assert c.label == "low"


def test_fermi_caps_confidence() -> None:
    c = score_confidence(
        retrieval_score=1.0,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=0,
        category="fermi",
    )
    assert c.score == 0.6
    assert c.label == "medium"


def test_unchecked_magnitude_counts_half() -> None:
    checked = score_confidence(
        retrieval_score=0.5,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=0,
        category="standard",
    )
    unchecked = score_confidence(
        retrieval_score=0.5,
        dimensions_ok=True,
        magnitude_ok=None,
        n_assumptions=0,
        category="standard",
    )
    assert checked.score - unchecked.score == pytest.approx(0.1, abs=0.01)
