"""The live view of ``askphysics model eval``: the ETA, the running rates, the results panel."""

import io
import re
from datetime import datetime

import pytest
from rich.console import Console

from askphysics.eval_view import (
    EvalMonitor,
    KindEta,
    Row,
    meter,
    print_results,
    real_sections,
    report_sections,
    results_panel,
    working,
)
from askphysics.lm.evaluate import EvalReport, EvalTick, RealReport
from askphysics.ui import make_console


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


NOON = datetime(2026, 10, 10, 12, 0, 0)


def _console(*, terminal: bool, width: int = 110) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return make_console(file=buffer, force_terminal=terminal, width=width), buffer


def _shown(buffer: io.StringIO) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())


def _frame(monitor: EvalMonitor, width: int = 110) -> str:
    console, buffer = _console(terminal=True, width=width)
    assert monitor.progress is not None
    console.print(monitor.progress.get_renderable())
    return _shown(buffer)


def tick(
    done: int,
    total: int = 10,
    task: str = "plan",
    *,
    stage: str = "score",
    plans: int = 0,
    valid: int = 0,
    confident: int = 0,
    tried: int = 0,
    rescued: int = 0,
) -> EvalTick:
    return EvalTick(
        stage,
        done,
        total,
        task,
        plans=plans,
        valid=valid,
        confident=confident,
        rescue_tried=tried,
        rescued=rescued,
    )


def test_the_eta_exists_after_the_first_example() -> None:
    clock = FakeClock()
    eta = KindEta({"classify": 5, "plan": 5}, clock)
    eta.start(0)
    assert eta.seconds_left(0, 10) is None  # nothing seen yet
    clock.now += 2
    eta.record_example(1, "classify")
    assert eta.seconds_left(1, 10) == pytest.approx(9 * 2)  # everything priced at 2 s


def test_the_eta_prices_each_kind_by_how_long_it_has_taken() -> None:
    clock = FakeClock()
    eta = KindEta({"classify": 4, "plan": 4}, clock)
    eta.start(0)
    for done, (kind, seconds) in enumerate([("classify", 1), ("classify", 1), ("plan", 9)], 1):
        clock.now += seconds
        eta.record_example(done, kind)
    # 2 classify left at 1 s, 3 plan left at 9 s
    assert eta.seconds_left(3, 8) == pytest.approx(2 * 1 + 3 * 9)


def test_a_run_of_cheap_examples_first_does_not_promise_an_early_finish_for_the_plans() -> None:
    # Classify first, then plans: a single running rate would say "done in a minute".
    clock = FakeClock()
    eta = KindEta({"classify": 100, "plan": 100}, clock)
    eta.start(0)
    for done in range(1, 101):
        clock.now += 0.1
        eta.record_example(done, "classify")
    clock.now += 5
    eta.record_example(101, "plan")
    left = eta.seconds_left(101, 200)
    assert left is not None and left == pytest.approx(99 * 5)  # not 99 * 0.1


def test_the_eta_follows_the_recent_pace_not_a_stale_one() -> None:
    clock = FakeClock()
    eta = KindEta({"plan": 400}, clock, recent=10)
    eta.start(0)
    for done in range(1, 101):
        clock.now += 10.0
        eta.record_example(done, "plan")
    for done in range(101, 121):
        clock.now += 1.0
        eta.record_example(done, "plan")
    assert eta.seconds_left(120, 400) == pytest.approx(280 * 1.0)


def test_the_bar_shows_an_eta_and_a_clock_time_after_one_example_and_the_running_rates() -> None:
    clock = FakeClock()
    console, _ = _console(terminal=True)
    with EvalMonitor(console, "fermi-solem-1", {"classify": 5, "plan": 5}, clock=clock,
                     now=lambda: NOON) as monitor:  # fmt: skip
        assert "eta …" in _frame(monitor)
        clock.now += 30
        monitor.on_tick(tick(1, plans=1, valid=1))
        frame = _frame(monitor)
        assert "eta 0:04:30 (done" not in frame  # the layout is eta H:MM:SS · done ~HH:MM
        assert "eta 0:04:30 · done ~12:04" in frame
        assert "1/10" in frame and " 10%" in frame
        assert "valid plan 100.0%" in frame and "confidently wrong 0" in frame
        clock.now += 30
        monitor.on_tick(tick(2, plans=2, valid=1, confident=1))
        frame = _frame(monitor)
        assert "valid plan 50.0%" in frame and "confidently wrong 1" in frame
        assert "speed 30.0 s/ex" in frame


def test_a_finished_pass_settles_into_a_line_with_a_tick_and_the_time() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    with EvalMonitor(console, "m", {"plan": 2}, clock=clock) as monitor:
        clock.now += 61
        monitor.on_tick(tick(1, 2, plans=1, valid=1))
        clock.now += 61
        monitor.on_tick(tick(2, 2, plans=2, valid=1, confident=1))
    line = next(row for row in _shown(buffer).splitlines() if "Scored" in row)
    line = line.rstrip()
    assert line.startswith("  ✓ Scored 2 examples")
    assert "valid plan 50.0% · confidently wrong 1" in line and line.endswith("2m 02s")


def test_a_rescue_pass_gets_its_own_bar_and_eta() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    with EvalMonitor(console, "m", {"plan": 2}, rescue_with="fermi-celeste-1", clock=clock,
                     now=lambda: NOON) as monitor:  # fmt: skip
        monitor.on_tick(tick(1, 2, plans=1, valid=1))
        monitor.on_tick(tick(2, 2, plans=2, valid=1))
        monitor.on_tick(tick(0, 4, stage="rescue", tried=4))
        frame = _frame(monitor)
        assert "rescue" in frame and "0/4" in frame and "scoring" not in frame
        clock.now += 8
        monitor.on_tick(tick(1, 4, stage="rescue", tried=4, rescued=1))
        frame = _frame(monitor)
        assert "1/4" in frame and "eta 0:00:25 · done ~12:00" in frame
        assert "rescued 1 of 4" in frame
        for done in (2, 3, 4):
            clock.now += 8
            monitor.on_tick(tick(done, 4, stage="rescue", tried=4, rescued=done))
    shown = _shown(buffer)
    assert "✓ Scored 2 examples" in shown
    assert "✓ Rescue pass with fermi-celeste-1" in shown and "rescued 4 of 4" in shown


def test_a_rescue_pass_with_no_misses_still_closes_cleanly() -> None:
    console, buffer = _console(terminal=False)
    with EvalMonitor(console, "m", {"plan": 1}, rescue_with="big") as monitor:
        monitor.on_tick(tick(1, 1, plans=1, valid=1, task="plan"))
        monitor.on_tick(tick(0, 0, stage="rescue"))
    assert "done: Rescue pass with big (0s) | rescued 0 of 0" in buffer.getvalue()


def test_plain_output_is_a_line_about_every_tenth_and_never_has_escape_codes() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=False)
    with EvalMonitor(console, "m", {"classify": 10, "plan": 10}, clock=clock,
                     now=lambda: NOON) as monitor:  # fmt: skip
        for done in range(1, 21):
            clock.now += 1
            monitor.on_tick(tick(done, 20, plans=done, valid=done // 2))
    lines = buffer.getvalue().splitlines()
    assert "\x1b" not in buffer.getvalue()
    assert lines[0].startswith("scoring 1/20 (5%) | valid plan 0.0% | confidently wrong 0 | eta ")
    assert any(line.startswith("scoring 10/20 (50%)") for line in lines)
    assert len(lines) <= 14  # a log stays short
    assert lines[-1] == "done: Scored 20 examples (20s) | valid plan 50.0% | confidently wrong 0"
    assert lines[-2].startswith("scoring 20/20 (100%)") and "eta" not in lines[-2]


def test_the_real_question_bar_has_no_plan_stats() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    monitor_args = {"label": "asking", "noun": "real textbook questions", "stats": False}
    with EvalMonitor(console, "m", {"question": 3}, clock=clock, **monitor_args) as monitor:  # type: ignore[arg-type]
        clock.now += 2
        monitor.on_tick(tick(1, 3, task="question"))
        frame = _frame(monitor)
        assert "asking" in frame and "valid plan" not in frame
        monitor.on_tick(tick(3, 3, task="question"))
    assert "✓ Scored 3 real textbook questions" in _shown(buffer)


def test_a_meter_is_green_when_high_is_good_and_red_when_it_is_low() -> None:
    assert meter(1.0, "high").style == "ok"
    assert meter(0.8, "high").style == "warn"
    assert meter(0.2, "high").style == "bad"
    assert meter(0.01, "low", width=1000).style == "ok"
    assert meter(0.5, "low").style == "bad"
    assert meter(0.5, "plain").style == "accent"
    assert len(meter(0.5, "high", width=20)) == 20


def _report() -> EvalReport:
    return EvalReport(
        classify_examples=120,
        category_accuracy=0.983,
        plan_examples=120,
        equation_accuracy=0.917,
        target_accuracy=0.95,
        knowns_accuracy=0.9,
        valid_plan_rate=0.892,
        attempts=3,
        routed_right_rate=0.961,
        flagged_wrong_rate=0.024,
        confidently_wrong_rate=0.015,
        mean_tries=1.21,
        rescue_tried=31,
        rescued=12,
        rescue_rate=12 / 31,
        rescue_confidently_wrong=1,
        routed_with_rescue_rate=0.98,
    )


def test_the_results_panel_has_a_bar_and_a_percentage_per_check() -> None:
    sections, footer = report_sections(_report(), attempts=3, rescue_with="fermi-celeste-1")
    console, buffer = _console(terminal=True)
    print_results(console, "fermi-solem-1 on held-out questions", sections, footer)
    shown = _shown(buffer)
    assert "fermi-solem-1 on held-out questions" in shown
    assert re.search(r"right category\s+120 classify\s+━+\s+98\.3%", shown)
    assert re.search(r"valid plan \(right answer\)\s+━+\s+89\.2%", shown)
    assert "up to 3 tries · 1.21 avg" in shown
    assert re.search(r"confidently wrong\s+━+\s+1\.5%", shown)
    assert "fermi-celeste-1 tried on 31 misses" in shown
    assert re.search(r"rescued\s+━+\s+38\.7%", shown)
    assert re.search(r"confidently wrong rescues\s+1\b", shown)
    assert "v0.4 target: rescues at least a third of the misses" in shown
    assert max(len(line.rstrip()) for line in shown.splitlines()) <= 100


def test_without_a_rescuer_the_panel_has_no_rescue_section() -> None:
    sections, footer = report_sections(_report(), attempts=3, rescue_with=None)
    assert footer == [] and len(sections) == 2


def test_plain_results_are_label_colon_value_lines() -> None:
    sections, footer = report_sections(_report(), attempts=3, rescue_with="big")
    console, buffer = _console(terminal=False)
    print_results(console, "m on held-out questions", sections, footer)
    lines = buffer.getvalue().splitlines()
    assert lines[0] == "m on held-out questions"
    assert "  right category (120 classify): 98.3%" in lines
    assert "  confidently wrong: 1.5%" in lines
    assert "  confidently wrong rescues: 1" in lines
    assert "\x1b" not in buffer.getvalue()


def test_real_question_results_use_the_same_panel() -> None:
    report = RealReport(questions=49, standard_rate=1.0, retrieved_rate=0.776, right_rate=0.5)
    console, buffer = _console(terminal=True)
    print_results(console, "m on 49 real textbook questions", real_sections(report))
    shown = _shown(buffer)
    assert re.search(r"right equations retrieved\s+━+\s+77\.6%", shown)
    assert re.search(r"right answer as ask gives it\s+━+\s+50\.0%", shown)


def test_a_row_with_no_value_is_a_heading_and_a_count_is_not_a_percentage() -> None:
    panel = results_panel(
        "t", [[Row("heading", None, "a note"), Row("how many", 3, good="low", count=True)]]
    )
    console, buffer = _console(terminal=True)
    console.print(panel)
    shown = _shown(buffer)
    assert "heading" in shown and "a note" in shown and re.search(r"how many\s+3\b", shown)
    assert "%" not in shown


def test_working_is_a_line_off_a_terminal_and_a_spinner_on_one() -> None:
    plain, text = _console(terminal=False)
    with working(plain, "loading m"):
        pass
    assert text.getvalue() == "loading m\n"
    live, buffer = _console(terminal=True)
    with working(live, "loading m"):
        pass
    assert "loading m" in _shown(buffer)
