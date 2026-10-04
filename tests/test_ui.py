import pytest
from rich.console import Console

from askphysics.data.loader import DataStore
from askphysics.models import Answer, Confidence, EquationRef, KnownValue
from askphysics.ui import (
    THEME,
    answer_card,
    confidence_meter,
    pretty_equation,
    pretty_number,
    pretty_symbol,
    pretty_unit,
)


def _render(renderable: object) -> str:
    console = Console(theme=THEME, width=100, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


@pytest.mark.parametrize(
    ("unit", "pretty"),
    [
        ("meter / second", "m/s"),
        ("meter / second ** 2", "m/s²"),
        ("joule", "J"),
        ("kilogram * meter / second", "kg·m/s"),
        ("not a unit at all", "not a unit at all"),
    ],
)
def test_pretty_unit(unit: str, pretty: str) -> None:
    assert pretty_unit(unit) == pretty


@pytest.mark.parametrize(
    ("value", "pretty"),
    [
        (19.8057, "19.8057"),
        (0, "0"),
        (437400.0, "437400"),
        (6.6743e-11, "6.674 × 10⁻¹¹"),
        (2.4e8, "2.4 × 10⁸"),
    ],
)
def test_pretty_number(value: float, pretty: str) -> None:
    assert pretty_number(value) == pretty


def test_pretty_symbol() -> None:
    assert pretty_symbol("v0") == "v₀"
    assert pretty_symbol("m12") == "m₁₂"
    assert pretty_symbol("KE") == "KE"


def test_pretty_equation_renders_math() -> None:
    assert pretty_equation("v**2 = v0**2 + 2*a*d") == "v² = v₀² + 2·a·d"
    assert pretty_equation("F = G*m1*m2/r**2") == "F = G·m₁·m₂/r²"
    assert pretty_equation("x = y; z") == "x = y; z"  # unparseable input falls back to raw


def test_confidence_meter_width() -> None:
    assert len(confidence_meter(0.5, width=10).plain) == 10


def _answer(**overrides: object) -> Answer:
    data: dict[str, object] = {
        "question": "How fast?",
        "status": "answered",
        "category": "standard",
        "final_value": 19.8057,
        "unit": "meter / second",
        "equations_used": [EquationRef(id="kin_v_squared", name="Velocity relation")],
        "inputs": [KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption")],
        "assumptions": ["No drag"],
        "confidence": Confidence(label="high", score=0.85),
        "caveats": ["Limit-case checks are not implemented yet (v0.8)."],
        "explanation": "Using [kin_v_squared], v = 19.8057 m/s.",
    }
    data.update(overrides)
    return Answer.model_validate(data)


def test_answer_card_shows_everything(store: DataStore) -> None:
    text = _render(answer_card(_answer(), store.equations))
    for expected in (
        "ANSWERED",
        "19.8057 m/s",
        "high · 0.85",
        "[kin_v_squared]",
        "v₀ = 0 m/s",
        "assumption",
        "• No drag",
        "Limit-case",
        "How fast?",
        "Ask Physics",
    ):
        assert expected in text, expected


def test_refused_card_shows_redirect() -> None:
    answer = _answer(
        status="refused",
        category="out_of_scope",
        final_value=None,
        unit=None,
        equations_used=[],
        inputs=[],
        explanation="This can't be answered as asked. Category error. "
        "A close question that can be answered: How heavy is a photon?",
        redirect="How heavy is a photon?",
        confidence=Confidence(label="low", score=0),
    )
    text = _render(answer_card(answer))
    assert "CAN'T ANSWER" in text
    assert "try instead  How heavy is a photon?" in text
    assert "A close question" not in text


def test_degraded_card() -> None:
    answer = _answer(
        status="degraded",
        final_value=None,
        unit=None,
        equations_used=[],
        inputs=[],
        confidence=Confidence(label="low", score=0),
        explanation="No answer: the plan stage failed.",
    )
    assert "PARTIAL" in _render(answer_card(answer))
