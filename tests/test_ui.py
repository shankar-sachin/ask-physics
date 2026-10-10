import io

from rich.console import Console

from askphysics.data.loader import DataStore
from askphysics.models import Answer, Confidence, EquationRef, KnownValue
from askphysics.ui import (
    THEME,
    answer_card,
    confidence_meter,
    tolerate_narrow_encodings,
    training_progress,
)


def _render(renderable: object) -> str:
    console = Console(theme=THEME, width=100, record=True, color_system=None)
    console.print(renderable)
    return console.export_text()


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


def test_narrow_encodings_replace_glyphs_instead_of_crashing() -> None:
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252")
    tolerate_narrow_encodings(stream)
    stream.write("◉ Ask Physics ✓")
    stream.flush()
    assert raw.getvalue() == b"? Ask Physics ?"


def test_utf8_streams_stay_strict() -> None:
    stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    tolerate_narrow_encodings(stream, object())
    assert stream.errors == "strict"


def test_models_line_names_each_stage_and_retries() -> None:
    from askphysics.ui import models_line

    answer = Answer(
        question="q",
        status="degraded",
        category="standard",
        confidence=Confidence(label="low", score=0.0),
        explanation="x",
        models={"classify": "fermi-tellus-1", "plan": "fermi-celeste-1", "explain": "template"},
        plan_attempts=6,
    )
    assert models_line(answer).plain == (
        "tellus classified · celeste planned (attempt 6) · template explained"
    )


def test_training_progress_estimates_speed_over_a_window_longer_than_its_updates() -> None:
    # The trainer logs about every 45 s; Rich's default 30 s window would hold one update at
    # most, so the ETA column would always read -:--:--.
    progress = training_progress(Console(record=True, file=io.StringIO()))
    assert progress.speed_estimate_period == 600
    assert progress.speed_estimate_period > 45
