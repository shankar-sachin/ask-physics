import json

import pytest
from pydantic import BaseModel

from askphysics.errors import LLMError, LLMResponseFormatError
from askphysics.llm.base import LLMClient
from askphysics.llm.fake import FakeLLMClient
from askphysics.models import Classification, Plan


def _classify(llm: FakeLLMClient, question: str) -> Classification:
    return llm.complete_json(
        system="", user=json.dumps({"question": question}), schema=Classification
    )


def test_satisfies_protocol(fake_llm: FakeLLMClient) -> None:
    assert isinstance(fake_llm, LLMClient)


@pytest.mark.parametrize(
    ("question", "category"),
    [
        ("How fast does a falling object hit the ground if dropped from 20 m?", "standard"),
        ("How many rubber ducks would it take to stop a freight train?", "fermi"),
        ("If everyone on Earth jumped at once, how far would Earth move?", "fermi"),
        ("How much does the color blue weigh?", "out_of_scope"),
        ("What happened before the Big Bang?", "out_of_scope"),
    ],
)
def test_classification(fake_llm: FakeLLMClient, question: str, category: str) -> None:
    assert _classify(fake_llm, question).category == category


def test_out_of_scope_offers_closest_answerable(fake_llm: FakeLLMClient) -> None:
    result = _classify(fake_llm, "How much does the color blue weigh?")
    assert result.closest_answerable


def test_canned_plan_for_drop_question(fake_llm: FakeLLMClient) -> None:
    user = json.dumps(
        {
            "question": "A rock is dropped from 45.5 meters. How fast does it land?",
            "equations": [{"id": "kin_v_squared"}],
        }
    )
    plan = fake_llm.complete_json(system="", user=user, schema=Plan)
    assert plan.equation_ids == ["kin_v_squared"]
    d = next(k for k in plan.known_values if k.symbol == "d")
    assert (d.value, d.unit) == (45.5, "m")


def test_canned_plan_converts_feet_alias(fake_llm: FakeLLMClient) -> None:
    user = json.dumps(
        {"question": "A ball is dropped from 10 feet.", "equations": [{"id": "kin_v_squared"}]}
    )
    plan = fake_llm.complete_json(system="", user=user, schema=Plan)
    assert next(k.unit for k in plan.known_values if k.symbol == "d") == "ft"


def test_no_canned_plan_refuses_to_invent(fake_llm: FakeLLMClient) -> None:
    user = json.dumps({"question": "What current flows through a resistor?", "equations": []})
    with pytest.raises(LLMError, match="no canned plan"):
        fake_llm.complete_json(system="", user=user, schema=Plan)


def test_no_plan_when_equation_not_retrieved(fake_llm: FakeLLMClient) -> None:
    user = json.dumps({"question": "dropped from 20 m", "equations": [{"id": "ohms_law"}]})
    with pytest.raises(LLMError):
        fake_llm.complete_json(system="", user=user, schema=Plan)


def test_unsupported_schema(fake_llm: FakeLLMClient) -> None:
    class Other(BaseModel):
        x: int

    with pytest.raises(LLMResponseFormatError):
        fake_llm.complete_json(system="", user="{}", schema=Other)


@pytest.mark.parametrize("bad", ["not json", "[1, 2]"])
def test_rejects_non_object_payloads(fake_llm: FakeLLMClient, bad: str) -> None:
    with pytest.raises(LLMResponseFormatError):
        fake_llm.complete_text(system="", user=bad)


def test_explanation_cites_ids_and_echoes_given_result(fake_llm: FakeLLMClient) -> None:
    user = json.dumps(
        {
            "result": {"value": 19.8057, "unit": "m/s"},
            "equation_ids": ["kin_v_squared"],
            "assumptions": ["No drag"],
        }
    )
    text = fake_llm.complete_text(system="", user=user)
    assert "[kin_v_squared]" in text
    assert "19.8057 m/s" in text
    assert "No drag" in text


def test_records_calls(fake_llm: FakeLLMClient) -> None:
    _classify(fake_llm, "test")
    assert fake_llm.calls[0][0] == "json:Classification"
