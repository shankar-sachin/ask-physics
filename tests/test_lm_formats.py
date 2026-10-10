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
    quantity_ranges,
    question_quantities,
    relevant_constants,
    serialize_classification,
    serialize_plan,
    stated_quantities,
    value_numbers,
)
from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, KnownValue, Plan
from askphysics.solver.units import quantity, unit_string


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


def test_extract_numbers_skips_identifiers_and_exponents() -> None:
    assert extract_numbers("m1 and v0 and x2y") == []
    assert extract_numbers("9.8 meter / second ** 2 or m^3") == ["9.8"]


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
    prompt = explain_prompt("q", quantity(19.8057, "m/s"), eqs, ["a"])
    assert prompt.startswith(EXPLAIN)
    assert '"value": 19.8057' in prompt
    assert '"unit": "meter / second"' in prompt  # spelled as the training data spells units


def test_unit_spelling_survives_the_explain_boundary(store: DataStore) -> None:
    # The pipeline and the training data spell a unit with unit_string, and FermiClient rebuilds
    # the Quantity from that spelling. It must spell the same way again, or the prompt drifts.
    units = {v.unit for eq in store.equations.values() for v in eq.variables}
    units |= {c.unit for c in store.constants.values()}
    for unit in sorted(units):
        spelled = unit_string(quantity(1.0, unit).units)
        assert unit_string(quantity(1.0, spelled).units) == spelled, unit


def test_allowed_numbers_and_units(store: DataStore) -> None:
    consts = list(store.constants.values())
    nums = plan_numbers("dropped from 20 m", consts)
    assert nums[0] == "20"
    assert "9.80665" in nums and "0" in nums and "1" in nums
    assert set(nums) == set(value_numbers("dropped from 20 m", consts)) | {"1"}


def test_values_never_come_from_the_structural_one(store: DataStore) -> None:
    consts = list(store.constants.values())
    values = value_numbers("dropped from 20 m", consts)
    assert values[0] == "20" and "9.80665" in values
    assert "0" in values  # the one assumed filler: "from rest"
    assert "1" not in values  # a 1 is only ever written as a number in the question
    assert "1" in value_numbers("A 1 kg cart is pushed.", consts)  # stated, so legal
    units = plan_units("dropped from 20 ft", [store.equations["kin_v_squared"]], consts)
    assert units[0] == "ft" and "m/s^2" in units
    result = quantity(19.8057, "m/s")
    assert explain_numbers("from 20 m", result, ["about 25 g"])[:3] == ["19.8057", "20", "25"]


def test_relevant_constants_match_by_dimension(store: DataStore) -> None:
    consts = list(store.constants.values())
    kin = relevant_constants([store.equations["kin_v_squared"]], consts)
    assert {c.name for c in kin} == {"standard_gravity", "speed_of_light"}
    gas = relevant_constants([store.equations["ideal_gas_law"]], consts)
    assert {c.name for c in gas} >= {"molar_gas_constant", "standard_atmosphere"}
    assert "planck_constant" not in {c.name for c in gas}


def test_quantities_with_dangling_operators_are_skipped() -> None:
    text = "A force of 5 N/ acts on 2 kg, and 3 m* is not a unit."
    assert extract_units(text) == ["kg"]
    assert question_quantities(text) == [("2", "kg")]


def test_numbers_too_big_for_a_float_are_skipped() -> None:
    text = "A 1e999 m wall and 5 kg of sand"
    assert extract_numbers(text) == ["5"]
    assert question_quantities(text) == [("5", "kg")]


def test_stated_quantities_add_bare_numbers_as_dimensionless() -> None:
    text = "A 5 kg box with a friction coefficient of 0.3 on a 2 m ramp"
    assert stated_quantities(text) == [("5", "kg"), ("2", "m"), ("0.3", "dimensionless")]


def test_e_notation_is_never_split_into_a_number_and_the_unit_e() -> None:
    assert question_quantities("about 7.5e+19 molecules, and 6.3e+20, then") == [
        ("7.5e+19", "molecules")
    ]
    assert stated_quantities("6.3e+20, then 2 C") == [("2", "C"), ("6.3e+20", "dimensionless")]


def test_a_range_gives_the_first_number_the_unit_of_the_second() -> None:
    text = "If a velocity increases from 0 to 20 m/s in 10 s, what is the acceleration?"
    assert quantity_ranges(text) == [(("0", "m/s"), ("20", "m/s"))]
    assert question_quantities(text) == [("0", "m/s"), ("20", "m/s"), ("10", "s")]
    assert stated_quantities(text) == [("0", "m/s"), ("20", "m/s"), ("10", "s")]  # none bare
    assert quantity_ranges("from 5 km/h up to 9 km/h") == [(("5", "km/h"), ("9", "km/h"))]
    assert quantity_ranges("from 20 m/s to 60 m/s") == [(("20", "m/s"), ("60", "m/s"))]


def test_a_range_needs_a_unit_after_the_second_number() -> None:
    assert quantity_ranges("goes from 12 to 3 in 4 s") == []  # "in" is a word here
    assert quantity_ranges("from the first to the second, 4 m/s") == []
    assert question_quantities("from 12 to 3 in 4 s") == [("3", "in"), ("4", "s")]
