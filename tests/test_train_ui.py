import io
import re
import sys

import pytest
from rich.console import Console
from typer.testing import CliRunner

from askphysics import cli
from askphysics.train_ui import format_elapsed, is_plain, run_phase
from askphysics.ui import make_console


class FakeClock:
    """A clock the test winds by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _console(*, terminal: bool, **kwargs: object) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    console = make_console(file=buffer, force_terminal=terminal, width=100, **kwargs)
    return console, buffer


def _shown(buffer: io.StringIO) -> str:
    """What a terminal would show: the output with its escape codes removed."""
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())


def _py(code: str) -> list[str]:
    return [sys.executable, "-c", code]


def test_elapsed_reads_naturally() -> None:
    assert format_elapsed(4.4) == "4s"
    assert format_elapsed(125) == "2m 05s"
    assert format_elapsed(3789) == "1h 03m 09s"


def test_output_is_plain_off_a_terminal_and_under_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("TERM", raising=False)
    assert is_plain(_console(terminal=False)[0])
    assert not is_plain(_console(terminal=True)[0])
    assert is_plain(_console(terminal=True, no_color=True)[0])
    monkeypatch.setenv("NO_COLOR", "1")
    assert is_plain(make_console(file=io.StringIO(), force_terminal=True))


def test_a_plain_phase_prints_start_and_done_lines_without_escape_codes() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=False)

    code = run_phase(console, "Building data", _py("pass"), clock=clock)
    assert code == 0
    text = buffer.getvalue()
    assert text.splitlines() == ["start: Building data", "done: Building data (0s)"]
    assert "\x1b" not in text


def test_a_failing_phase_reports_its_exit_code() -> None:
    console, buffer = _console(terminal=False)
    assert run_phase(console, "Scoring", _py("raise SystemExit(3)")) == 3
    assert "failed: Scoring (exit 3, " in buffer.getvalue()


def test_a_missing_command_is_a_failed_phase_not_a_traceback() -> None:
    console, buffer = _console(terminal=False)
    assert run_phase(console, "Nope", ["definitely-not-a-command-xyz"]) == 127
    assert "not found" in buffer.getvalue()


def test_a_terminal_phase_ticks_when_done_and_keeps_the_command_output(
    capfd: pytest.CaptureFixture[str],
) -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    code = run_phase(console, "Training the tokenizer", _py("print('vocab 8192')"), clock=clock)
    assert code == 0
    text = _shown(buffer)
    assert "vocab 8192" in text and "✓ Training the tokenizer" in text
    assert any(frame in text for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")  # the spinner drew


def test_a_terminal_phase_marks_a_failure() -> None:
    console, buffer = _console(terminal=True)
    assert run_phase(console, "Scoring", _py("print('boom'); raise SystemExit(2)")) == 2
    assert "✗ Scoring" in _shown(buffer) and "boom" in _shown(buffer)


def test_an_inheriting_phase_prints_a_heading_and_a_closing_line(
    capfd: pytest.CaptureFixture[str],
) -> None:
    console, buffer = _console(terminal=True)
    assert run_phase(console, "Training fermi-luna-1", _py("print('live view')"), inherit=True) == 0
    assert "◉ Training fermi-luna-1" in _shown(buffer)
    assert "✓ Training fermi-luna-1" in _shown(buffer)
    assert "live view" in capfd.readouterr().out  # the command wrote to the real terminal


def test_the_phase_command_passes_the_exit_code_through() -> None:
    runner = CliRunner()
    ok = runner.invoke(cli.app, ["model", "phase", "--title", "T", "--", *_py("print('hi')")])
    assert ok.exit_code == 0 and "done: T" in ok.output
    boom = _py("raise SystemExit(4)")
    bad = runner.invoke(cli.app, ["model", "phase", "--title", "T", "--", *boom])
    assert bad.exit_code == 4
