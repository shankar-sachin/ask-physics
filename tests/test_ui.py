import io
from datetime import datetime

import pytest
from rich.console import Console
from rich.progress import Progress

from askphysics.data.loader import DataStore
from askphysics.models import Answer, Confidence, EquationRef, KnownValue
from askphysics.ui import (
    THEME,
    EtaTracker,
    answer_card,
    confidence_meter,
    format_duration,
    format_finish,
    make_console,
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


class FakeClock:
    """A clock the test winds by hand, so ETA tests take no real time."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _render_wide(progress: Progress) -> str:
    console = make_console(file=io.StringIO(), width=200, force_terminal=False)
    console.print(progress.get_renderable())
    return console.file.getvalue()  # type: ignore[attr-defined]


def _drive(clock: FakeClock, start: int, total: int, updates: int, every: int) -> Progress:
    """A run at one step per second from ``start``, reported every ``every`` steps."""
    progress = training_progress(
        make_console(file=io.StringIO(), width=200),
        clock=clock,
        now=lambda: datetime(2026, 1, 5, 23, 0, 0),
    )
    task = progress.add_task("m", total=total, loss="...", val="...", speed="")
    progress.reset(task, completed=start)
    progress.tracker.start(start)
    step = start
    for _ in range(updates):
        clock.now += every
        step += every
        progress.tracker.record(step)
        progress.update(task, completed=step)
    return progress


def test_eta_shows_after_a_few_updates_with_a_clock_time() -> None:
    progress = _drive(FakeClock(), start=0, total=1000, updates=3, every=50)
    # 150 of 1000 steps done at 1 step/s: 850 s left from 23:00:00 is 23:14:10 on the clock.
    assert "eta 0:14:10 · done ~23:14" in _render_wide(progress)


def test_eta_after_a_resume_does_not_count_the_jump_as_speed() -> None:
    # Resumed at 4000 of 5000, 1 step/s: 850 s left, not the near-zero a 0-to-4050 jump gives.
    progress = _drive(FakeClock(), start=4000, total=5000, updates=3, every=50)
    assert "eta 0:14:10" in _render_wide(progress)


def test_finish_time_names_the_day_when_it_is_not_today() -> None:
    assert format_finish(5 * 3600, datetime(2026, 1, 5, 23, 0)) == "~Tue 04:00"
    assert format_finish(600, datetime(2026, 1, 5, 12, 0)) == "~12:10"
    assert format_duration(3723) == "1:02:03"


def test_own_estimate_fills_in_when_rich_has_none() -> None:
    clock = FakeClock()
    tracker = EtaTracker(clock)
    assert tracker.seconds_left(0, 100) is None  # nothing seen yet
    tracker.start(100)
    assert tracker.seconds_left(100, 1100) is None  # one sample is not a rate
    clock.now += 10
    tracker.record(120)  # 2 steps/s
    assert tracker.seconds_left(120, 1120) == 500
    # A task Rich has no samples for (it was set to 120 directly) uses the tracker's number.
    progress = training_progress(make_console(file=io.StringIO(), width=200), clock=clock)
    task = progress.add_task("m", total=1120, completed=120, loss="", val="", speed="")
    progress.tracker.start(100)
    clock.now += 10
    progress.tracker.record(120)
    assert progress.tasks[task].time_remaining is None
    assert "eta 0:08:20" in _render_wide(progress)


def test_own_estimate_forgets_samples_older_than_its_window() -> None:
    clock = FakeClock()
    tracker = EtaTracker(clock, window=60)
    tracker.start(0)
    clock.now += 600
    tracker.record(60)  # a slow start: 0.1 steps/s
    for step in (160, 260):
        clock.now += 10
        tracker.record(step)  # now 10 steps/s
    assert tracker.steps_per_second() == pytest.approx(10.0)
