import io
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from askphysics.train_view import RunInfo, TrainingMonitor, sparkline
from askphysics.ui import make_console

NOW = datetime(2026, 1, 5, 23, 0, 0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def _info(**kwargs: Any) -> RunInfo:
    base: dict[str, Any] = {
        "name": "fermi-luna-1",
        "backend": "torch",
        "device": "cpu",
        "steps": 100,
        "out_dir": Path("/models/fermi-luna-1"),
        "has_prose": True,
        "prose_steps": 40,
        "prose_share": 0.1,
    }
    return RunInfo(**{**base, **kwargs})


def _console(*, terminal: bool, **kwargs: Any) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return make_console(file=buffer, force_terminal=terminal, width=130, **kwargs), buffer


def _shown(buffer: io.StringIO) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())


def _run(monitor: TrainingMonitor, clock: FakeClock, start: int, stop: int, every: int) -> None:
    """One step per second from ``start`` to ``stop``, logging every ``every`` steps."""
    monitor.on_step(start)
    for done in range(start + 1, stop + 1):
        clock.now += 1.0
        monitor.on_step(done)
        if done % every == 0:
            monitor.on_log(
                {
                    "step": done,
                    "loss": 4.0 - done / 50,
                    "lr": 3e-4,
                    "target_tokens_per_s": 4200.0,
                    "mem_gb": 2.1,
                }
            )
        if done % 50 == 0:
            monitor.on_log({"step": done, "val_loss": 3.0 - done / 100, "val_loss_unit": 1.5})


def test_sparkline_scales_to_its_own_range_and_keeps_the_recent_values() -> None:
    assert sparkline([]) == ""
    assert sparkline([5, 5, 5]) == "▁▁▁"
    assert sparkline([0, 1, 2, 3, 4, 5, 6, 7]) == "▁▂▃▄▅▆▇█"
    assert sparkline(range(100), width=4) == "▁▃▆█"


def test_phase_names_the_prose_warm_up_then_the_tasks() -> None:
    info = _info()
    assert info.phase(0) == "prose warm-up (steps 0-40)"
    assert info.phase(39) == "prose warm-up (steps 0-40)"
    assert info.phase(40) == "tasks + prose (steps 40-100)"
    assert _info(prose_share=0.0).phase(60) == "tasks (steps 40-100)"
    assert _info(has_prose=False).phase(0) == "tasks (steps 0-100)"


def test_the_live_view_shows_the_panel_eta_and_phase_switch() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    monitor = TrainingMonitor(console, _info(), clock=clock, now=lambda: NOW)
    assert not monitor.plain
    with monitor:
        _run(monitor, clock, 0, 45, every=5)
        console.print(monitor.renderable())
    text = _shown(buffer)
    # 45 of 100 steps at 1 step/s: 55 s left from 23:00:00.
    assert "eta 0:00:55 · done ~23:00" in text
    assert "→ step 40: tasks + prose (steps 40-100)" in text  # printed when the phase switched
    for expected in ("tasks + prose (steps 40-100)", "torch · cpu", "4,200 tok/s", "2.1 GB peak"):
        assert expected in text
    assert "3.00e-04" in text and any(block in text for block in "▁▂▃▄▅▆▇█")
    assert "best 2.500 at 50" not in text  # step 50 has not happened yet


def test_the_live_view_after_a_resume_estimates_from_the_resumed_step() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    monitor = TrainingMonitor(console, _info(), clock=clock, now=lambda: NOW)
    with monitor:
        _run(monitor, clock, 60, 65, every=5)
        console.print(monitor.renderable())
    assert "eta 0:00:35" in _shown(buffer)  # 35 steps left at 1 step/s, not a jump-inflated rate


def test_plain_output_is_readable_lines_without_escape_codes_or_animation() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=False)
    monitor = TrainingMonitor(console, _info(), clock=clock, now=lambda: NOW)
    assert monitor.plain
    with monitor:
        _run(monitor, clock, 0, 100, every=10)
    monitor.print_summary()
    text = buffer.getvalue()
    assert "\x1b" not in text and "\r" not in text
    lines = text.splitlines()
    assert lines[0] == "phase: prose warm-up (steps 0-40)"
    assert "phase at step 40: tasks + prose (steps 40-100)" in lines
    step_line = next(line for line in lines if line.startswith("step 20/100"))
    assert step_line == (
        "step 20/100 (20%) | loss 3.600 | lr 3.00e-04 | 4,200 tok/s | peak mem 2.1 GB"
        " | eta 0:01:20 (done ~23:01)"
    )
    assert "step 50/100 | val loss 2.500 | best 2.500 at step 50" in lines
    assert "training complete: fermi-luna-1" in lines
    assert "  best val: 2.000 at step 100" in lines
    assert "  weights: /models/fermi-luna-1" in lines
    assert "  total time: 1m 40s" in lines


def test_no_color_gives_plain_lines_even_on_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    console, buffer = _console(terminal=True)
    monitor = TrainingMonitor(console, _info(), clock=FakeClock(), now=lambda: NOW)
    assert monitor.plain
    with monitor:
        monitor.on_step(0)
    assert buffer.getvalue() == "phase: prose warm-up (steps 0-40)\n"


def test_the_summary_panel_reports_the_run() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    monitor = TrainingMonitor(console, _info(), clock=clock, now=lambda: NOW)
    with monitor:
        _run(monitor, clock, 0, 100, every=10)
    monitor.print_summary()
    text = _shown(buffer)
    for expected in (
        "fermi-luna-1 trained",
        "final train loss",
        "2.000",
        "final val loss",
        "best val",
        "at step 100",
        "total time",
        "1m 40s",
        "average speed",
        "4,200 tok/s",
        "val by task",
        "unit 1.500",
        "/models/fermi-luna-1",
    ):
        assert expected in text
