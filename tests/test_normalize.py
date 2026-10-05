import pytest

from askphysics.data.loader import DataStore
from askphysics.lm.factory import DataFactory
from askphysics.lm.formats import question_quantities
from askphysics.normalize import normalize_question
from askphysics.pipeline import Pipeline


@pytest.mark.parametrize(
    ("raw", "quantities"),
    [
        ("A 2,000-kg car moves at 27 m/s.", [("2000", "kg"), ("27", "m/s")]),
        ("Light: 3.00 × 10^8 m/s", [("300000000", "m/s")]),
        ("Light: 3.00×10⁸ m/s", [("300000000", "m/s")]),
        ("Light: 3.00 x 10 8 m/s", [("300000000", "m/s")]),  # flattened superscript
        ("G is 6.67 × 10⁻¹¹ N·m²/kg²", [("6.67e-11", "N*m^2/kg^2")]),
        ("10⁻³ kg of salt", [("0.001", "kg")]),
        ("It accelerates at 9.8 m/s² for 4 s.", [("9.8", "m/s^2"), ("4", "s")]),
        ("Its momentum is 12 kg⋅m/s.", [("12", "kg*m/s")]),
        ("A 55-kg lady", [("55", "kg")]),
        ("going 20 meters per second", [("20", "meters/second")]),
        ("braking at 4 meters per second squared", [("4", "meters/second^2")]),
        ("Water at 25 °C", [("25", "degC")]),
        ("a charge of −2.5 µC", [("-2.5", "uC")]),
        ("a 220 Ω resistor and a 4.7 kΩ one", [("220", "ohm"), ("4.7", "kohm")]),
        ("1,234.5 m away", [("1234.5", "m")]),
    ],
)
def test_real_world_quantities_become_readable(raw: str, quantities: list[tuple[str, str]]) -> None:
    assert question_quantities(normalize_question(raw)) == quantities


@pytest.mark.parametrize(
    "text",
    [
        "Friction μ = 0.3 between the surfaces.",  # a symbol, not a unit prefix
        "3 apples per minute",  # "apples/minute" is not a unit
        "items 1,2,3 and 12,34",  # not thousands groups
        "How fast does a ball dropped from 20 m hit the ground?",
    ],
)
def test_text_that_is_not_a_quantity_is_left_alone(text: str) -> None:
    assert normalize_question(text) == text


def test_factory_questions_keep_every_quantity(store: DataStore) -> None:
    factory = DataFactory(store, seed=12)
    checked = 0
    for _ in range(400):
        p = factory.standard_problem()
        if p is None:
            continue
        assert question_quantities(normalize_question(p.question)) == question_quantities(
            p.question
        ), p.question
        checked += 1
    assert checked > 300


def test_normalizing_twice_changes_nothing() -> None:
    once = normalize_question("A 2,000-kg car at 3.0 × 10⁸ m/s² and 220 Ω")
    assert normalize_question(once) == once


def test_the_pipeline_answers_with_the_normalized_question(pipeline: Pipeline) -> None:
    answer = pipeline.run("A ball is dropped from 2,000 cm. How fast does it hit the ground?")
    assert "2000 cm" in answer.question
