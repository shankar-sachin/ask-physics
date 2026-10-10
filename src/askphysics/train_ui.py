"""The look of a step: the spinner, the tick and the time, the dimmed tail of a long command's
output, and the failure block. Used by ``scripts/lib.sh`` (through ``askphysics.shell_ui``) for
every workflow script, and by ``scripts/train.sh`` for its phases.

Everything here only formats and times; nothing computes. When output is not a terminal (CI,
``| tee``), when ``NO_COLOR`` is set, or when the terminal is dumb, everything prints as plain
ASCII lines with no animation, so a log stays readable.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, ClassVar, Protocol

from rich import box
from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.live import Live
from rich.padding import Padding
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

FRAME_WIDTH = 100  # steps never draw wider than this, so a wide terminal stays calm
INDENT = 2
TAIL_LINES = 3  # lines of a long command's output kept under its spinner
FAILURE_LINES = 14  # lines of output shown when a command fails
PLAIN_INDENT = "  "

_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")


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


def frame_width(console: Console) -> int:
    return max(48, min(console.width, FRAME_WIDTH))


def clean_line(line: str) -> str:
    """A line of a child's output without escape codes, tabs, or the line ending."""
    return _ANSI.sub("", line).replace("\t", "    ").rstrip()


def display_command(command: Sequence[str]) -> str:
    """The command as a person would type it; the keep-awake wrapper is left out."""
    parts = list(command)
    if parts and Path(parts[0]).name == "caffeinate":
        parts = parts[1:]
        while parts and parts[0].startswith("-"):
            parts = parts[1:]
    return shlex.join(parts)


def step_row(
    mark: RenderableType,
    title: Text,
    *,
    width: int,
    counter: str = "",
    detail: Text | None = None,
    right: Text | None = None,
    pad: int = 0,
    indent: int = INDENT,
) -> RenderableType:
    """One aligned line: mark, counter, title, detail, and the time on the right.

    The running line and the finished line use the same columns, so a step resolves in place:
    the spinner becomes the tick and nothing else moves.
    """
    grid = Table.grid(padding=(0, 1), expand=True)
    grid.add_column(width=1, no_wrap=True)
    if counter:
        grid.add_column(no_wrap=True, justify="right")
    grid.add_column(no_wrap=True)
    grid.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
    grid.add_column(no_wrap=True, justify="right")
    title = title.copy()
    title.pad_right(max(0, pad - len(title)))  # titles line up when the script gave a width
    title.append(" ")  # two spaces to the detail: this one and the column gap
    cells: list[RenderableType] = [mark]
    if counter:
        cells.append(Text(counter, style="muted"))
    cells += [title, detail or Text(""), right or Text("")]
    grid.add_row(*cells)
    return Padding(Constrain(grid, width - indent), (0, 0, 0, indent))


class OutputParser(Protocol):
    """Reads a command's output as it streams, to say something useful beside its spinner."""

    def feed(self, line: str) -> None: ...

    def live(self) -> str:
        """What to show while the command runs."""
        ...

    def done(self) -> str:
        """What to show once it has finished."""
        ...

    def settled(self) -> tuple[str, str] | None:
        """A stage of the command that just finished, as (title, detail), once; else None."""
        ...


class LastLine:
    """The last line of output that says something: ``All checks passed!``."""

    def __init__(self) -> None:
        self._last = ""

    def feed(self, line: str) -> None:
        if line.strip():
            self._last = line.strip()

    def live(self) -> str:
        return ""

    def done(self) -> str:
        return self._last

    def settled(self) -> tuple[str, str] | None:
        return None


class FirstLine:
    """The first line of output that says something: ``seed data: all valid``."""

    def __init__(self) -> None:
        self._first = ""

    def feed(self, line: str) -> None:
        if not self._first and line.strip():
            self._first = line.strip().removeprefix("✓").strip()

    def live(self) -> str:
        return ""

    def done(self) -> str:
        return self._first

    def settled(self) -> tuple[str, str] | None:
        return None


class PytestOutput:
    """Counts pytest's progress dots as they arrive, then reads its closing summary line."""

    DOTS = re.compile(r"^(?:\S+\.py\s+)?([.sxXFEw]+)\s*(?:\[\s*(\d+)%\])?\s*$")
    SUMMARY = re.compile(
        r"^=*\s*(.*\b(?:passed|failed|errors?|skipped|deselected|xfailed|xpassed|warnings?)\b.*?)"
        r"\s+in\s+[\d.]+s\b.*$"
    )
    KINDS: ClassVar[dict[str, str]] = {
        ".": "passed",
        "F": "failed",
        "E": "errors",
        "s": "skipped",
        "x": "xfailed",
    }

    def __init__(self) -> None:
        self.counts = dict.fromkeys(self.KINDS.values(), 0)
        self.percent = 0
        self._summary = ""

    def feed(self, line: str) -> None:
        text = line.strip()
        hit = self.DOTS.match(text)
        if hit:
            for char in hit.group(1):
                kind = self.KINDS.get(char)
                if kind:
                    self.counts[kind] += 1
            if hit.group(2):
                self.percent = int(hit.group(2))
            return
        closing = self.SUMMARY.match(text)
        if closing:
            self._summary = closing.group(1).strip(" =")

    def _counts(self) -> str:
        shown = [f"{n:,} {kind}" for kind, n in self.counts.items() if n]
        return " · ".join(shown)

    def live(self) -> str:
        counts = self._counts()
        if not counts:
            return "collecting tests" if not self.percent else ""
        return f"{counts}  {self.percent}%" if self.percent else counts

    def done(self) -> str:
        return self._summary.replace(", ", " · ") if self._summary else self._counts()

    def settled(self) -> tuple[str, str] | None:
        return None


class CorpusOutput:
    """Reads ``scripts/build_corpus.py``'s progress lines: books, words, and the stage it is in.

    The OpenStax stage settles into a line of its own when it ends; the Gutenberg stage (and the
    whole step) settles when the script does.
    """

    OPENSTAX = re.compile(r"^\[OpenStax (\d+)/(\d+)\] (?:downloading|reading) (.*)$")
    OPENSTAX_DONE = re.compile(r"^\[OpenStax\] done: ([\d,]+) words$")
    GUTENBERG_WORDS = re.compile(r"^\[Gutenberg\] ([\d,]+) of ([\d,]+) words$")
    GUTENBERG_BOOKS = re.compile(r"^\[Gutenberg\] ([\d,]+) public-domain books qualify")
    FINAL = re.compile(r"^([\d,]+) words from (\d+) OpenStax repositories and (\d+) public-domain")

    def __init__(self) -> None:
        self._live = "starting"
        self._books = 0
        self._pending: tuple[str, str] | None = None
        self._final = ""

    def feed(self, line: str) -> None:
        text = line.strip()
        if hit := self.OPENSTAX.match(text):
            self._books = int(hit.group(2))
            self._live = f"OpenStax · book {hit.group(1)} of {hit.group(2)}"
        elif hit := self.OPENSTAX_DONE.match(text):
            self._pending = ("OpenStax textbooks", f"{self._books} books · {hit.group(1)} words")
            self._live = "Gutenberg · reading the catalog"
        elif hit := self.GUTENBERG_BOOKS.match(text):
            self._live = f"Gutenberg · {hit.group(1)} books qualify"
        elif hit := self.GUTENBERG_WORDS.match(text):
            words, target = (int(hit.group(i).replace(",", "")) for i in (1, 2))
            self._live = f"Gutenberg · {words:,} of {target:,} words · {100 * words // target}%"
        elif hit := self.FINAL.match(text):
            self._final = f"{hit.group(1)} words · {hit.group(2)} textbooks · {hit.group(3)} books"

    def live(self) -> str:
        return self._live

    def done(self) -> str:
        return self._final

    def settled(self) -> tuple[str, str] | None:
        stage, self._pending = self._pending, None
        return stage


PARSERS: dict[str, Callable[[], OutputParser]] = {
    "pytest": PytestOutput,
    "last": LastLine,
    "first": FirstLine,
    "corpus": CorpusOutput,
}


class _Running:
    """The live block of a running step: its line, then the last few lines of output, dimmed."""

    def __init__(
        self,
        title: str,
        clock: Callable[[], float],
        *,
        width: int,
        counter: str,
        pad: int,
        tail: int,
        parser: OutputParser | None,
    ) -> None:
        self.title = title
        self.clock = clock
        self.t0 = clock()
        self.stage_t0 = self.t0
        self.width = width
        self.counter = counter
        self.pad = pad
        self.parser = parser
        self.lines: deque[str] = deque(maxlen=tail)
        self._spinner = Spinner("dots", style="accent")

    def feed(self, line: str) -> None:
        if self.parser is not None:
            self.parser.feed(line)
        if line.strip() and not (isinstance(self.parser, PytestOutput) and self.parser.live()):
            self.lines.append(line)

    def __rich__(self) -> RenderableType:
        detail = Text(self.parser.live() if self.parser else "", style="muted")
        rows: list[RenderableType] = [
            step_row(
                self._spinner,
                Text(self.title, style="brand"),
                width=self.width,
                counter=self.counter,
                detail=detail,
                right=Text(format_elapsed(self.clock() - self.t0), style="muted"),
                pad=self.pad,
            )
        ]
        for line in tuple(self.lines):
            row = Text.assemble(("│ ", "faint"), (line, "faint"), no_wrap=True, overflow="ellipsis")
            rows.append(Padding(Constrain(row, self.width - INDENT - 5), (0, 0, 0, INDENT + 4)))
        return Group(*rows)


def finished_row(
    title: str,
    seconds: float | None,
    code: int,
    *,
    width: int,
    counter: str = "",
    detail: str = "",
    pad: int = 0,
    skipped: bool = False,
    indent: int = INDENT,
) -> RenderableType:
    """The line a step settles into: a tick (or a cross) and the time it took."""
    if skipped:
        mark, title_style, detail_style = Text("\u2013", style="muted"), "muted", "muted"
    elif code == 0:
        mark, title_style, detail_style = Text("✓", style="ok"), "value", "muted"
    else:
        mark, title_style, detail_style = Text("✗", style="bad"), "bad", "bad"
    if code and not skipped and not detail:
        detail = f"exit {code}"
    return step_row(
        mark,
        Text(title, style=title_style),
        width=width,
        counter=counter,
        detail=Text(detail, style=detail_style),
        right=Text(format_elapsed(seconds), style="muted") if seconds is not None else None,
        pad=pad,
        indent=indent,
    )


def _label(title: str, index: int | None, total: int | None) -> str:
    return f"[{index}/{total}] {title}" if index and total else title


def _plain_finish(
    title: str, seconds: float, code: int, index: int | None, total: int | None
) -> Text:
    label = _label(title, index, total)
    elapsed = format_elapsed(seconds)
    if code == 0:
        return Text(f"done: {label} ({elapsed})")
    return Text(f"failed: {label} (exit {code}, {elapsed})")


def failure_panel(
    command: Sequence[str], code: int, output: Sequence[str], log: Path | None, *, width: int
) -> RenderableType:
    """What a failed command left behind: the command, the last lines of output, the log."""
    body: list[RenderableType] = [Text.assemble(("$ ", "muted"), (display_command(command), "eq"))]
    shown = [line for line in output if line.strip()][-FAILURE_LINES:]
    if shown:
        body.append(Text(""))
        body.extend(Text(line, no_wrap=True, overflow="ellipsis") for line in shown)
    panel = Panel(
        Group(*body),
        title=Text(f"exit {code}", style="bad"),
        title_align="left",
        border_style="bad",
        box=box.ROUNDED,
        padding=(0, 1),
    )
    parts: list[RenderableType] = [Constrain(panel, width - INDENT)]
    if log is not None:  # outside the box, so a long path wraps instead of being cut by a border
        parts.append(Text.assemble(("full log  ", "label"), (str(log), "accent")))
    return Padding(Group(*parts), (0, 0, 0, INDENT))


def run_phase(
    console: Console,
    title: str,
    command: Sequence[str],
    *,
    inherit: bool = False,
    clock: Callable[[], float] = time.monotonic,
    index: int | None = None,
    total: int | None = None,
    log: Path | None = None,
    tail: int = TAIL_LINES,
    parse: str | None = None,
    pad: int = 0,
    ok_detail: str = "",
) -> int:
    """Run ``command`` as one named phase and return its exit code.

    On a terminal the phase shows a spinner and the time so far while it runs, with the last
    few lines of the command's output dimmed underneath (``tail``), then settles into a tick
    and the time it took. The full output goes to ``log``. When the command fails, the cross
    is followed by the command, the last lines of its output, and the path of the log.

    With ``inherit`` the command keeps the terminal to itself (the training view draws its own
    live display), and the phase prints only a heading and the closing line. Plain output
    prints ``start:`` and ``done:`` lines and lets the command write straight through.

    ``parse`` names an output reader (``pytest``, ``last``, ``first``, ``corpus``) that puts a
    count or a line of the output beside the step; ``ok_detail`` is said when it has nothing.
    """
    plain = is_plain(console)
    counter = f"{index}/{total}" if index and total else ""
    width = frame_width(console)
    t0 = clock()
    code: int
    detail = ""
    output: list[str] = []
    saved: Path | None = None
    try:
        if plain:
            console.print(Text(f"start: {_label(title, index, total)}"), soft_wrap=True)
            console.file.flush()
            code = subprocess.run(list(command), check=False).returncode
        elif inherit:
            console.print(
                Padding(Text("◉ ", style="accent") + Text(title, style="brand"), (0, 0, 0, INDENT))
            )
            console.file.flush()
            code = subprocess.run(list(command), check=False).returncode
        else:
            parser = PARSERS[parse]() if parse else None
            running = _Running(
                title, clock, width=width, counter=counter, pad=pad, tail=tail, parser=parser
            )
            code, output, saved = _run_captured(console, command, running, log)
            # A failed command shows its exit code, except pytest, whose counts say what failed.
            detail = parser.done() if parser and (code == 0 or parse == "pytest") else ""
    except FileNotFoundError:
        output = [f"cannot run {command[0]}: not found"]
        code = 127
        if plain or inherit:
            console.print(Text(output[0]), soft_wrap=True)
    except KeyboardInterrupt:
        code = 130
    seconds = clock() - t0
    if plain:
        console.print(_plain_finish(title, seconds, code, index, total), soft_wrap=True)
        if code:
            console.print(Text(f"  command: {display_command(command)}"), soft_wrap=True)
        return code
    if inherit and code == 0:
        detail = ""
    if code == 0 and not detail:
        detail = ok_detail  # what to say for a command that prints nothing when it is happy
    console.print(
        finished_row(title, seconds, code, width=width, counter=counter, detail=detail, pad=pad)
    )
    if code and not inherit:
        console.print(failure_panel(command, code, output, saved, width=width))
    return code


def _run_captured(
    console: Console, command: Sequence[str], running: _Running, log: Path | None
) -> tuple[int, list[str], Path | None]:
    """Run under the spinner, saving everything to ``log``; returns the code and last lines."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # so the output arrives as it is written
    handle: IO[str] | None = None
    if log is not None:
        try:
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w", encoding="utf-8", errors="replace")
        except OSError:
            log = None  # a read-only disk should never stop the step
    last: deque[str] = deque(maxlen=200)
    try:
        with subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            env=env,
        ) as proc:
            assert proc.stdout is not None
            try:
                with Live(running, console=console, refresh_per_second=15, transient=True):
                    for raw in proc.stdout:
                        if handle is not None:
                            handle.write(raw)
                            handle.flush()
                        line = clean_line(raw)
                        last.append(line)
                        running.feed(line)
                        stage = running.parser.settled() if running.parser else None
                        if stage is not None:
                            now = running.clock()
                            console.print(
                                finished_row(
                                    stage[0],
                                    now - running.stage_t0,
                                    0,
                                    width=running.width,
                                    detail=stage[1],
                                    pad=running.pad,
                                    indent=INDENT + 2,
                                )
                            )
                            running.stage_t0 = now
                return proc.wait(), list(last), log
            except KeyboardInterrupt:
                proc.wait()  # the terminal sent it the interrupt too
                raise
    finally:
        if handle is not None:
            handle.close()
