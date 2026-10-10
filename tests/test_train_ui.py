import io
import re
import sys
from pathlib import Path

import pytest
from rich.console import Console
from rich.text import Text

from askphysics.train_ui import (
    CorpusOutput,
    FirstLine,
    LastLine,
    PytestOutput,
    _Running,
    _undecorated,
    display_command,
    failure_panel,
    finished_row,
    format_elapsed,
    is_plain,
    pytest_text,
    run_phase,
    step_row,
)
from askphysics.ui import make_console


class FakeClock:
    """A clock the test winds by hand."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def _console(*, terminal: bool, **kwargs: object) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    kwargs.setdefault("width", 100)
    console = make_console(file=buffer, force_terminal=terminal, **kwargs)
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


def test_a_plain_phase_shows_its_counter(capfd: pytest.CaptureFixture[str]) -> None:
    console, buffer = _console(terminal=False)
    assert run_phase(console, "Tests", _py("print('hi')"), index=2, total=6) == 0
    assert buffer.getvalue().splitlines() == ["start: [2/6] Tests", "done: [2/6] Tests (0s)"]
    assert capfd.readouterr().out == "hi\n"  # the command writes straight through


def test_a_failed_plain_phase_names_its_command() -> None:
    console, buffer = _console(terminal=False)
    assert run_phase(console, "Boom", ["false"]) == 1
    assert buffer.getvalue().splitlines()[-1] == "  command: false"


def test_the_keep_awake_wrapper_is_left_out_of_the_command_shown() -> None:
    assert display_command(["caffeinate", "-dims", "askphysics-dev", "model", "eval"]) == (
        "askphysics-dev model eval"
    )
    assert display_command(["python", "-c", "print(1)"]) == "python -c 'print(1)'"


def test_a_running_step_keeps_the_last_few_lines_dimmed_under_its_spinner() -> None:
    clock = FakeClock()
    running = _Running("Installing", clock, width=100, counter="3/6", pad=0, tail=3, parser=None)
    for n in range(1, 8):
        running.feed(f"Collecting package-{n}")
    clock.now += 75
    console, buffer = _console(terminal=True)
    console.print(running)
    shown = _shown(buffer)
    assert "3/6" in shown and "Installing" in shown and "1m 15s" in shown
    kept = [line for line in shown.splitlines() if "Collecting" in line]
    assert [line.strip() for line in kept] == [
        "│ Collecting package-5",
        "│ Collecting package-6",
        "│ Collecting package-7",
    ]


def test_a_finished_step_lines_up_its_columns_and_a_failure_says_its_exit_code() -> None:
    console, buffer = _console(terminal=True)
    console.print(finished_row("Lint", 2.4, 0, width=100, counter="1/3", detail="clean", pad=8))
    console.print(finished_row("Types", 65, 2, width=100, counter="2/3", pad=8))
    console.print(finished_row("Shell", None, 0, width=100, counter="3/3", detail="n/a", pad=8,
                               skipped=True))  # fmt: skip
    lint, types, shell = _shown(buffer).splitlines()
    assert lint.startswith("  ✓ 1/3 Lint      clean") and lint.endswith("2s")
    assert types.startswith("  ✗ 2/3 Types     exit 2") and types.endswith("1m 05s")
    assert shell.startswith("  \u2013 3/3 Shell     n/a")
    assert len({len(line) for line in (lint, types)}) == 1  # the time sits at one right edge


def test_the_time_is_at_the_same_edge_whether_running_or_finished() -> None:
    clock = FakeClock()
    console, buffer = _console(terminal=True)
    console.print(_Running("Tests", clock, width=100, counter="", pad=0, tail=3, parser=None))
    console.print(finished_row("Tests", 0, 0, width=100))
    running, done = (line.rstrip() for line in _shown(buffer).splitlines())
    assert len(running) == len(done) == 100


def test_a_failed_step_shows_its_command_the_end_of_its_output_and_the_log(
    tmp_path: Path,
) -> None:
    log = tmp_path / "logs" / "01-boom.log"
    console, buffer = _console(terminal=True)
    script = "print('\\n'.join(f'line {n}' for n in range(40))); raise SystemExit(3)"
    assert run_phase(console, "Boom", _py(script), log=log, index=1, total=2) == 3
    shown = _shown(buffer)
    assert "✗ 1/2 Boom" in shown and "exit 3" in shown
    assert "line 39" in shown and "line 25" not in shown  # only the last lines
    assert f"full log  {log}" in shown.replace("\n", "")
    assert log.read_text().splitlines()[0] == "line 0"  # the log has all of it


def test_the_full_output_is_saved_even_when_the_step_succeeds(tmp_path: Path) -> None:
    log = tmp_path / "ok.log"
    console, _ = _console(terminal=True)
    assert run_phase(console, "Fine", _py("print('a'); print('b')"), log=log) == 0
    assert log.read_text() == "a\nb\n"


def test_an_unwritable_log_never_stops_the_step(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    console, _ = _console(terminal=True)
    assert run_phase(console, "Fine", _py("pass"), log=blocker / "sub" / "x.log") == 0


def test_a_failure_panel_for_a_command_that_printed_nothing() -> None:
    console, buffer = _console(terminal=True)
    console.print(failure_panel(["false"], 1, [], None, width=100))
    shown = _shown(buffer)
    assert "exit 1" in shown and "$ false" in shown and "full log" not in shown


def test_escape_codes_in_a_childs_output_never_reach_the_display() -> None:
    clock = FakeClock()
    running = _Running("Build", clock, width=100, counter="", pad=0, tail=3, parser=None)
    from askphysics.train_ui import clean_line

    running.feed(clean_line("\x1b[31mred\x1b[0m text\t\x1b]0;title\x07"))
    console, buffer = _console(terminal=True)
    console.print(running)
    assert "│ red text" in _shown(buffer)


def test_pytest_counts_its_progress_dots_and_reads_its_closing_line() -> None:
    parser = PytestOutput()
    assert parser.live() == "collecting tests"
    for line in ("..........                    [ 10%]", "....F.s.x.                    [ 20%]"):
        parser.feed(line)
    assert parser.live() == "17 passed · 1 failed · 1 skipped · 1 xfailed  20%"
    parser.feed("=== 1 failed, 1280 passed, 3 skipped in 92.31s (0:01:32) ===")
    assert parser.done() == "1 failed · 1280 passed · 3 skipped"
    quiet = PytestOutput()
    quiet.feed("1284 passed in 3.10s")
    assert quiet.done() == "1284 passed"


def test_a_pytest_step_shows_counts_instead_of_dots_and_keeps_them_when_it_fails() -> None:
    console, buffer = _console(terminal=True)
    script = "print('...F.'); print('1 failed, 4 passed in 0.05s'); raise SystemExit(1)"
    assert run_phase(console, "pytest", _py(script), parse="pytest", pad=8) == 1
    row = next(line for line in _shown(buffer).splitlines() if line.startswith("  ✗"))
    assert "1 failed · 4 passed" in row


def test_the_first_and_last_line_parsers_pick_the_line_that_says_something() -> None:
    first, last = FirstLine(), LastLine()
    for line in ("", "✓ seed data: all valid", "┏━━━┓", "│ equations │ 110 │"):
        first.feed(line)
        last.feed(line)
    assert first.done() == "seed data: all valid"
    assert last.done() == "│ equations │ 110 │"


def test_a_succeeding_quiet_step_says_what_the_script_asked_it_to() -> None:
    console, buffer = _console(terminal=True)
    assert run_phase(console, "shellcheck", _py("pass"), parse="last", ok_detail="no findings") == 0
    assert "no findings" in _shown(buffer)


def test_the_corpus_parser_settles_the_openstax_stage_into_a_line_of_its_own() -> None:
    parser = CorpusOutput()
    parser.feed("[OpenStax 2/11] downloading openstax/osbooks-college-physics")
    assert parser.live() == "OpenStax · book 2 of 11"
    assert parser.settled() is None
    parser.feed("[OpenStax 11/11] reading University Physics Volume 3")
    parser.feed("[OpenStax] done: 3,412,330 words")
    assert parser.settled() == ("OpenStax textbooks", "11 books · 3,412,330 words")
    assert parser.settled() is None  # only once
    parser.feed("[Gutenberg] 9,000,000 of 20,000,000 words")
    assert parser.live() == "Gutenberg · 9,000,000 of 20,000,000 words · 45%"
    parser.feed("20,000,000 words from 11 OpenStax repositories and 24 public-domain books -> x")
    assert parser.done() == "20,000,000 words · 11 textbooks · 24 books"


def test_a_corpus_step_prints_the_openstax_stage_as_it_ends() -> None:
    script = (
        "print('[OpenStax 1/1] reading A'); print('[OpenStax] done: 100 words'); "
        "print('20 words from 1 OpenStax repositories and 2 public-domain books -> x')"
    )
    console, buffer = _console(terminal=True)
    assert run_phase(console, "Corpus", _py(script), parse="corpus") == 0
    shown = _shown(buffer)
    assert "✓ OpenStax textbooks" in shown and "1 books · 100 words" in shown
    assert "20 words · 1 textbooks · 2 books" in shown


def test_a_step_row_is_one_line_at_a_narrow_width() -> None:
    console, buffer = _console(terminal=True, width=60)
    console.print(step_row(Text("✓"), Text("A very long title " * 4), width=60, right=Text("9s")))
    assert len(_shown(buffer).splitlines()) == 1


def test_pytest_counts_are_green_for_passed_red_for_failed_and_the_percent_stays_quiet() -> None:
    text = pytest_text("144 passed · 1 failed · 2 skipped  54%")
    styles = {text.plain[span.start : span.end]: str(span.style) for span in text.spans}
    assert styles["144 passed"] == "ok" and styles["1 failed"] == "bad"
    assert styles["2 skipped"] == "muted" and styles["  54%"] == "muted"
    assert text.plain == "144 passed · 1 failed · 2 skipped  54%"


def test_a_failure_panel_drops_the_rulers_and_reddens_the_error_lines() -> None:
    assert _undecorated("===== FAILURES =====") == "FAILURES"
    assert _undecorated("_____ test_triple _____") == "test_triple"
    assert _undecorated("-- two dashes --") == "-- two dashes --"  # too short to be a ruler
    assert _undecorated("=====") == "====="  # nothing to keep: left as it was
    assert _undecorated("plain line") == "plain line"
    console, buffer = _console(terminal=True)
    output = ["=== FAILURES ===", "E   assert 6 == 9", "FAILED tests/x.py::t - assert 6 == 9", "ok"]
    console.print(failure_panel(["pytest"], 1, output, None, width=100))
    assert "===" not in _shown(buffer)
    raw = buffer.getvalue()
    bad = "\x1b[1;91m"  # the theme's red on a 16-colour terminal
    assert bad + "E   assert 6 == 9" in raw and bad + "FAILED tests/x.py::t" in raw
    assert bad + "ok" not in raw
