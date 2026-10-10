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


# --------------------------------------------------------------------------- number spellings (#97)


@pytest.mark.parametrize(
    ("text", "number", "unit"),
    [
        ("a frequency of 4.00 x 10^14 Hz", "400000000000000", "Hz"),
        ("a frequency of 4.00 \u00d7 10^14 Hz", "400000000000000", "Hz"),
        ("a speed of 1.0*10^6 m/s", "1000000", "m/s"),
        ("an acceleration of 6.30x10^5 m/s^2", "630000", "m/s^2"),
        ("a time of 8.10 X 10^-4 s", "0.00081", "s"),
        ("an energy of 3.56e-13 J", "3.56e-13", "J"),
        ("a charge of 4 x 10^-9 C", "4e-09", "C"),
        ("a length of 2 x 10**3 m", "2000", "m"),
        ("a frequency of 10^14 Hz", "100000000000000", "Hz"),  # a power of ten alone
        ("a distance of 1,530 km", "1530", "km"),
        ("a power of 100,000 W", "100000", "W"),
        ("a mass of 2,000,000 kg", "2000000", "kg"),
        ("a pressure of 1,013.25 Pa", "1013.25", "Pa"),
        ("a charge of -25 nC", "-25", "nC"),
        ("a charge Q = - 25 nC", "-25", "nC"),  # set apart from its number after "="
        ("a force (- 3 x 10^-6 N)z", "-3e-06", "N"),
        ("a charge of \u221225 nC", "-25", "nC"),  # the unicode minus
        ("a field of -4 x 10^-9 C", "-4e-09", "C"),
        ("a wave at 2.5 MeV", "2.5", "MeV"),
        ("a frequency of 1,530 kHz", "1530", "kHz"),
        ("a 5-kg cart", "5", "kg"),  # a hyphen between the number and its unit
        ("a 90.0-MHz station", "90", "MHz"),
        ("a 0.0100-nm-wavelength photon", "0.01", "nm"),
    ],
)
def test_a_spelled_number_is_one_quantity(text: str, number: str, unit: str) -> None:
    assert extract_numbers(text) == [number]
    assert question_quantities(text) == [(number, unit)]
    assert stated_quantities(text) == [(number, unit)]  # none of its parts is left over


def test_the_mantissa_is_read_as_written() -> None:
    """ "1.1 x 10^-5" is the float "1.1e-5", not 1.1 * 10**-5 (1.1000000000000001e-05)."""
    assert extract_numbers("a length of 1.1 x 10^-5 m") == ["1.1e-05"]
    assert extract_numbers("a length of 1.1e-5 m") == extract_numbers("1.1 x 10^-5 m")
    assert extract_numbers("a mass of 5.97 x 10^24 kg") == ["5.97e+24"]
    assert extract_numbers("a mass of 9.109 x 10^-31 kg") == ["9.109e-31"]


def test_a_power_of_ten_never_offers_its_base_or_its_exponent() -> None:
    for text in ("4.00 x 10^14 Hz", "4.00x10^14", "1.0*10^6", "10^14", "3 x 10**-9"):
        numbers = extract_numbers(text)
        assert len(numbers) == 1, text
        assert not {"10", "14", "6", "9", "-9", "4", "1", "3"} & set(numbers), text
        assert len(stated_quantities(text)) == 1, text


def test_a_comma_groups_figures_only_without_a_space() -> None:
    assert extract_numbers("1,530 kHz") == ["1530"]
    assert extract_numbers("speeds of 3, 4 and 5 m/s") == ["3", "4", "5"]
    assert extract_numbers("3,4,5") == ["3", "4", "5"]  # groups are three figures
    assert extract_numbers("3,45 and 6,7890") == ["3", "45", "6", "7890"]
    assert extract_numbers("1,530,000 W and 12,345.5 W") == ["1530000", "12345.5"]
    assert extract_numbers("0,123") == ["0", "123"]  # no leading-zero group
    assert stated_quantities("masses 3, 4 and 5 kg") == [
        ("5", "kg"),
        ("3", "dimensionless"),
        ("4", "dimensionless"),
    ]


def test_a_range_dash_is_not_a_minus() -> None:
    assert extract_numbers("between 10-20 m") == ["10", "20"]
    assert question_quantities("between 10-20 m") == [("20", "m")]
    assert extract_numbers("5-10 s and -3 m") == ["5", "10", "-3"]
    assert extract_numbers("a speed of 10 - 5 m/s") == ["10", "5"]  # a subtraction
    assert extract_numbers("x=-2 m") == ["-2"]
    assert extract_numbers("v(-3 N)") == ["-3"]


def test_unit_exponents_are_still_no_numbers() -> None:
    assert extract_numbers("2 m^-1 and 3 s^-2, 10 kg m^-3") == ["2", "3", "10"]
    assert extract_numbers("5 m/s^2 for 10 s") == ["5", "10"]
    assert extract_numbers("a density of 5 x 10^3 kg/m^3") == ["5000"]


def test_a_power_of_ten_too_big_for_a_float_is_skipped() -> None:
    assert extract_numbers("a 1 x 10^999 m wall and 5 kg of sand") == ["5"]
    assert question_quantities("a 1 x 10^999 m wall and 5 kg of sand") == [("5", "kg")]
    assert stated_quantities("a 2.5 x 10^999 wall and 5 kg") == [("5", "kg")]


def test_a_range_reads_every_spelling() -> None:
    assert quantity_ranges("from 1.0 x 10^3 to 2.0 x 10^3 m/s") == [
        (("1000", "m/s"), ("2000", "m/s"))
    ]
    assert quantity_ranges("from -5 to 20 m/s") == [(("-5", "m/s"), ("20", "m/s"))]
    assert quantity_ranges("from 1,000 to 2,500 W") == [(("1000", "W"), ("2500", "W"))]
    assert question_quantities("from 1.0 x 10^3 to 2.0 x 10^3 m/s") == [
        ("1000", "m/s"),
        ("2000", "m/s"),
    ]


def test_a_messy_question_offers_only_the_numbers_it_states() -> None:
    """Golden rule 1: every offered number is one the question writes, whole."""
    text = (
        "Light of 4.00 x 10^14 Hz and 1,530 kHz carries -25 nC, 3, 4 and 5 m, 10-20 s, "
        "and a 6.30x10^5 m/s^2 push."
    )
    assert [n for n, _ in stated_quantities(text)] == [
        "400000000000000",
        "1530",
        "-25",
        "5",
        "20",
        "630000",
        "3",
        "4",
        "10",
    ]
