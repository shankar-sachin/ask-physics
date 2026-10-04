import json

import pytest

from askphysics.data.loader import DataStore
from askphysics.lm.formats import (
    classify_prompt,
    explain_numbers,
    explain_prompt,
    extract_numbers,
    extract_units,
    format_number,
    plan_numbers,
    plan_prompt,
    plan_units,
    relevant_constants,
    serialize_classification,
    serialize_plan,
)
from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, KnownValue, Plan


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (20.0, "20"),
        (0.0, "0"),
        (-3.0, "-3"),
        (9.80665, "9.80665"),
        (6.6743e-11, "6.6743e-11"),
        (0.5, "0.5"),
        (1e20, "1e+20"),
    ],
)
def test_format_number(value: float, text: str) -> None:
    assert format_number(value) == text


def test_format_number_rejects_nan() -> None:
    with pytest.raises(ValueError):
        format_number(float("nan"))


def test_extract_numbers_and_units() -> None:
    q = "A ball dropped from 20 m hits at 9.80 m/s^2; v0 is 0 and 20.0 again, 1e3 J."
    assert extract_numbers(q) == ["20", "9.8", "0", "1000"]
    assert extract_units(q) == ["m", "m/s^2", "J"]


def test_extract_numbers_skips_identifiers() -> None:
    assert extract_numbers("m1 and v0 and x2y") == []


def test_classification_round_trip() -> None:
    c = Classification(category="fermi", reasoning="Needs estimates.", domains=["momentum"])
    text = serialize_classification(c)
    assert text.endswith(END)
    data = json.loads(text[: -len(END)])
    assert list(data) == ["category", "domains", "reasoning", "closest_answerable"]
    assert Classification.model_validate(data) == c


def test_plan_serialization_is_canonical() -> None:
    plan = Plan(
        equation_ids=["kin_v_squared"],
        target="v",
        unknowns=["v"],
        known_values=[KnownValue(symbol="d", value=20.0, unit="m", origin="given")],
        assumptions=["No drag"],
        strategy="Solve for v.",
    )
    text = serialize_plan(plan)
    assert text == (
        '{"equation_ids": ["kin_v_squared"], "target": "v", "unknowns": ["v"], '
        '"known_values": [{"symbol": "d", "value": 20, "unit": "m", "origin": "given"}], '
        '"assumptions": ["No drag"], "strategy": "Solve for v."}' + END
    )
    assert Plan.model_validate(json.loads(text[: -len(END)])) == plan


def test_prompts_start_with_task_tokens(store: DataStore) -> None:
    eqs = list(store.equations.values())[:2]
    consts = list(store.constants.values())
    assert classify_prompt("q").startswith(CLASSIFY)
    standard = plan_prompt("q", "standard", eqs, consts, list(store.fermi))
    fermi = plan_prompt("q", "fermi", eqs, consts, list(store.fermi))
    assert standard.startswith(PLAN) and "fermi_assumptions" not in standard
    assert "rubber_duck_mass" in fermi
    assert explain_prompt("q", 19.8057, "m/s", eqs, ["a"]).startswith(EXPLAIN)
    assert '"value": 19.8057' in explain_prompt("q", 19.8057, "m/s", eqs, ["a"])


def test_allowed_numbers_and_units(store: DataStore) -> None:
    consts = list(store.constants.values())
    nums = plan_numbers("dropped from 20 m", consts)
    assert nums[0] == "20"
    assert "9.80665" in nums and "0" in nums and "1" in nums
    units = plan_units("dropped from 20 ft", [store.equations["kin_v_squared"]], consts)
    assert units[0] == "ft" and "m/s^2" in units
    assert explain_numbers("from 20 m", 19.8057, ["about 25 g"])[:3] == ["19.8057", "20", "25"]


def test_relevant_constants_match_by_dimension(store: DataStore) -> None:
    consts = list(store.constants.values())
    kin = relevant_constants([store.equations["kin_v_squared"]], consts)
    assert {c.name for c in kin} == {"standard_gravity", "speed_of_light"}
    gas = relevant_constants([store.equations["ideal_gas_law"]], consts)
    assert {c.name for c in gas} >= {"molar_gas_constant", "standard_atmosphere"}
    assert "planck_constant" not in {c.name for c in gas}
