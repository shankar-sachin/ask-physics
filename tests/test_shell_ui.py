"""The terminal look of the workflow scripts (askphysics.shell_ui)."""

import io
import re
import subprocess
import sys

import pytest
from rich.console import Console

from askphysics import shell_ui
from askphysics.ui import make_console


def _console(*, terminal: bool, width: int = 100) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return make_console(file=buffer, force_terminal=terminal, width=width), buffer


def _shown(buffer: io.StringIO) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())


def test_rows_split_a_command_from_what_it_does() -> None:
    assert shell_ui.split_row("sh scripts/check.sh::run every gate") == (
        "sh scripts/check.sh",
        "run every gate",
    )
    assert shell_ui.split_row("just a command") == ("just a command", "")


def test_the_header_names_the_script_and_counts_its_steps() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.header(console, "Check", "Everything that must pass", 6)
    lines = [line.rstrip() for line in _shown(buffer).splitlines() if line.strip()]
    assert lines == [
        "  ◉ Ask Physics  ·  Check",
        "    Everything that must pass  ·  6 steps",
    ]


def test_the_plain_header_is_two_ascii_lines() -> None:
    console, buffer = _console(terminal=False)
    shell_ui.header(console, "Check", "Everything that must pass", 1)
    assert buffer.getvalue().splitlines() == [
        "Ask Physics: Check (1 step)",
        "  Everything that must pass",
    ]


def test_a_summary_after_a_clean_run_reads_all_clear_with_the_time_and_what_to_run_next() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.summary(
        console,
        title="All clear",
        steps=6,
        passed=6,
        seconds=112,
        nexts=[("git push -u origin HEAD", "publish the branch"), ("make ask", "try it")],
    )
    shown = _shown(buffer)
    assert "✓ All clear   all 6 steps  ·  1m 52s" in shown
    assert "next  git push -u origin HEAD  publish the branch" in shown
    assert "      make ask                 try it" in shown


def test_a_summary_of_a_single_step_says_step() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.summary(console, title="Scored", steps=1, passed=1, seconds=52)
    assert "✓ Scored   all 1 step  ·  52s" in _shown(buffer)


def test_a_summary_after_a_failure_names_what_failed() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.summary(console, title="Not ready", steps=6, passed=4, failed=["mypy", "pytest"])
    assert "✗ Not ready   2 of 6 failed: mypy, pytest" in _shown(buffer)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"steps": 6, "passed": 6, "seconds": 112}, ["ok: Done (6 of 6 steps, 1m 52s)"]),
        ({"steps": 1, "passed": 1, "seconds": 4}, ["ok: Done (1 of 1 step, 4s)"]),
        ({"seconds": 4}, ["ok: Done (4s)"]),
        ({}, ["ok: Done"]),
        (
            {"steps": 3, "passed": 2, "failed": ["pytest"], "seconds": 9},
            ["failed: Done (1 of 3 steps failed: pytest; 9s)"],
        ),
        (
            {"steps": 3, "failed": ["a", "b"]},
            ["failed: Done (2 of 3 steps failed: a, b)"],
        ),
        (
            {"nexts": [("make x", "do x"), ("make y", "")]},
            ["ok: Done", "  next: make x  - do x", "  next: make y"],
        ),
    ],
)
def test_the_plain_summary_is_one_line_per_fact(
    kwargs: dict[str, object], expected: list[str]
) -> None:
    console, buffer = _console(terminal=False)
    shell_ui.summary(console, title="Done", **kwargs)  # type: ignore[arg-type]
    assert buffer.getvalue().splitlines() == expected


def test_the_ready_panel_lists_commands_with_what_they_do() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.ready(
        console,
        "Ask Physics is ready",
        [("source .venv/bin/activate", "in each new terminal"), ("make ask", "ask something")],
    )
    shown = _shown(buffer)
    assert "╭─ ✓ Ask Physics is ready" in shown
    assert "source .venv/bin/activate    in each new terminal" in shown
    assert re.search(r"make ask {21}ask something", shown)
    assert max(len(line) for line in shown.splitlines()) <= 100


def test_the_plain_ready_lines() -> None:
    console, buffer = _console(terminal=False)
    shell_ui.ready(console, "Ready", [("make ask", "ask something")], footer="Enjoy.")
    assert buffer.getvalue().splitlines() == [
        "ready: Ready",
        "  make ask  - ask something",
        "  Enjoy.",
    ]


def test_results_for_a_tick_a_cross_and_a_skip() -> None:
    console, buffer = _console(terminal=True)
    shell_ui.result(console, "ok", "origin/main", "merges cleanly", index=7, total=8, pad=12)
    shell_ui.result(console, "fail", "origin/x", "conflicts", index=8, total=8, pad=12)
    shell_ui.result(console, "skip", "shellcheck", "not installed")
    ok, bad, skip = _shown(buffer).splitlines()
    assert ok.startswith("  ✓ 7/8 origin/main   merges cleanly")
    assert bad.startswith("  ✗ 8/8 origin/x      conflicts")
    assert skip.startswith("  \u2013 shellcheck ") and "not installed" in skip
    plain, text = _console(terminal=False)
    shell_ui.result(plain, "skip", "shellcheck", "not installed", index=1, total=2)
    assert text.getvalue() == "skipped: [1/2] shellcheck - not installed\n"


def test_main_runs_a_step_and_returns_its_exit_code() -> None:
    console, buffer = _console(terminal=False)
    code = shell_ui.main(
        ["step", "--title", "Fail", "--", sys.executable, "-c", "exit(4)"], console
    )
    assert code == 4 and "failed: Fail (exit 4," in buffer.getvalue()
    assert shell_ui.main(["step", "--title", "Nothing"], console) == 2


def test_main_prints_a_summary_from_the_clock_a_script_recorded() -> None:
    console, buffer = _console(terminal=False)
    args = ["summary", "--title", "Done", "--steps", "2", "--passed", "2", "--next", "make x::do x"]
    assert shell_ui.main(args, console) == 0
    assert buffer.getvalue().splitlines() == ["ok: Done (2 of 2 steps)", "  next: make x  - do x"]


def test_it_runs_as_a_module_and_prints_plain_lines_into_a_pipe() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "askphysics.shell_ui", "header", "--title", "T", "--steps", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["Ask Physics: T (3 steps)"]
    assert "\x1b" not in result.stdout


def test_importing_the_display_pulls_in_no_math_or_models() -> None:
    # Each script step starts this module, so it has to stay light.
    code = (
        "import sys, askphysics.shell_ui; "
        "bad = [m for m in ('sympy', 'pint', 'torch', 'pydantic') if m in sys.modules]; "
        "sys.exit(len(bad))"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
