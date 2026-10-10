"""The look of a training run: phase lines for ``scripts/train.sh``, and the live training view.

Everything here only formats and times; nothing computes. When output is not a terminal (CI,
``| tee``), when ``NO_COLOR`` is set, or when the terminal is dumb, everything prints as plain
ASCII lines with no animation, so a log stays readable.
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Callable, Sequence

from rich.console import Console, RenderableType
from rich.live import Live
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text


def is_plain(console: Console) -> bool:
    """True when the output should be plain lines: not a terminal, a dumb one, or NO_COLOR."""
    return not console.is_terminal or console.is_dumb_terminal or bool(console.no_color)


def format_elapsed(seconds: float) -> str:
    """``45s``, ``2m 05s``, or ``1h 03m 09s``: how long a phase took."""
    whole = max(0, round(seconds))
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


class _Running:
    """The spinner line of a running phase; the elapsed time is read each time it draws."""

    def __init__(self, title: str, clock: Callable[[], float]) -> None:
        self._title = title
        self._clock = clock
        self._t0 = clock()
        self._spinner = Spinner("dots", style="accent")

    def __rich__(self) -> RenderableType:
        line = Table.grid(padding=(0, 1))
        line.add_row(
            self._spinner,
            Text(self._title, style="brand"),
            Text(format_elapsed(self._clock() - self._t0), style="muted"),
        )
        return line


def _finish_line(title: str, seconds: float, code: int, plain: bool) -> Text:
    elapsed = format_elapsed(seconds)
    if plain:
        return Text(
            f"done: {title} ({elapsed})"
            if code == 0
            else f"failed: {title} (exit {code}, {elapsed})"
        )
    text = Text("✓ " if code == 0 else "✗ ", style="ok" if code == 0 else "bad")
    text.append(title, style="brand" if code == 0 else "bad")
    text.append(f"  {elapsed}" if code == 0 else f"  exit {code} after {elapsed}", style="muted")
    return text


def run_phase(
    console: Console,
    title: str,
    command: Sequence[str],
    *,
    inherit: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Run ``command`` as one named phase and return its exit code.

    On a terminal the phase shows a spinner and the time so far while it runs, then a tick and
    the time it took; the command's own output scrolls above the spinner. With ``inherit`` the
    command keeps the terminal to itself (the training view draws its own live display), and
    the phase prints only a heading and the closing line. Plain output prints ``start:`` and
    ``done:`` lines and lets the command write straight through.
    """
    plain = is_plain(console)
    t0 = clock()
    code: int
    try:
        if plain:
            console.print(Text(f"start: {title}"), soft_wrap=True)
            console.file.flush()
            code = subprocess.run(list(command), check=False).returncode
        elif inherit:
            console.print(Text("◉ ", style="accent") + Text(title, style="brand"))
            console.file.flush()
            code = subprocess.run(list(command), check=False).returncode
        else:
            code = _run_with_spinner(console, title, command, clock)
    except FileNotFoundError:
        console.print(Text(f"cannot run {command[0]}: not found"), soft_wrap=True)
        code = 127
    except KeyboardInterrupt:
        code = 130
    console.print(_finish_line(title, clock() - t0, code, plain), soft_wrap=True)
    return code


def _run_with_spinner(
    console: Console, title: str, command: Sequence[str], clock: Callable[[], float]
) -> int:
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # so the output scrolls as it is written
    with subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    ) as proc:
        assert proc.stdout is not None
        try:
            with Live(
                _Running(title, clock), console=console, refresh_per_second=10, transient=True
            ):
                for line in proc.stdout:
                    console.print(Text(line.rstrip("\n")), soft_wrap=True)
            return proc.wait()
        except KeyboardInterrupt:
            proc.wait()  # the terminal sent it the interrupt too
            raise
