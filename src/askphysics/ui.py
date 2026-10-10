"""Terminal presentation for the CLI: answer cards and training progress.

The pretty math and unit formatting lives in ``askphysics.pretty`` so the website
can use it without Rich.

Everything here only formats; nothing computes. Numbers shown are the ones
Noether produced.
"""

from __future__ import annotations

import codecs
import io
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timedelta

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from askphysics.models import Answer, Equation
from askphysics.pretty import pretty_equation, pretty_number, pretty_symbol, pretty_unit

THEME = Theme(
    {
        "brand": "bold #8be9fd",
        "muted": "#8b93a7",
        "label": "bold #a6accd",
        "value": "bold #f8f8f2",
        "accent": "#bd93f9",
        "ok": "bold #50fa7b",
        "warn": "bold #f1fa8c",
        "bad": "bold #ff6e6e",
        "eq": "#ffb86c",
        "badge.ok": "bold #1e1f29 on #50fa7b",
        "badge.warn": "bold #1e1f29 on #f1fa8c",
        "badge.bad": "bold #1e1f29 on #ff6e6e",
    }
)
STATUS = {
    "answered": ("ok", "ANSWERED"),
    "degraded": ("warn", "PARTIAL"),
    "refused": ("bad", "CAN'T ANSWER"),
}
CONFIDENCE_STYLE = {"high": "ok", "medium": "warn", "low": "bad"}
ORIGIN_STYLE = {"given": "value", "constant": "accent", "assumption": "warn"}


def make_console(**kwargs: object) -> Console:
    return Console(theme=THEME, highlight=False, **kwargs)  # type: ignore[arg-type]


def tolerate_narrow_encodings(*streams: object) -> None:
    """Print ``?`` for glyphs a stream can't encode instead of crashing.

    Windows pipes and redirects default to cp1252, which has no ◉, ✓, or ━.
    """
    for stream in streams:
        if isinstance(stream, io.TextIOWrapper) and codecs.lookup(stream.encoding).name != "utf-8":
            stream.reconfigure(errors="replace")


def confidence_meter(score: float, width: int = 20) -> Text:
    filled = round(score * width)
    return Text(
        "━" * filled, style="ok" if score >= 0.75 else "warn" if score >= 0.45 else "bad"
    ) + Text("━" * (width - filled), style="muted")


def banner() -> Text:
    text = Text()
    text.append("◉ ", style="accent")
    text.append("Ask Physics", style="brand")
    text.append("  ·  grounded answers, real math", style="muted")
    return text


def _rows(rows: list[tuple[str, RenderableType]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="label", justify="right", no_wrap=True, min_width=11)
    grid.add_column()
    for label, value in rows:
        grid.add_row(label, value)
    return grid


STAGE_VERBS = {"classify": "classified", "plan": "planned", "explain": "explained"}


def models_line(answer: Answer) -> Text:
    """Who did what: "tellus classified · solem planned (attempt 2) · solem explained"."""
    parts = []
    for stage, verb in STAGE_VERBS.items():
        name = answer.models.get(stage)
        if name is None:
            continue
        short = name.removeprefix("fermi-").removesuffix("-1")
        if stage == "plan" and answer.plan_attempts > 1:
            verb += f" (attempt {answer.plan_attempts})"
        parts.append(f"{short} {verb}")
    return Text(" · ".join(parts), style="muted")


def answer_card(answer: Answer, equations: Mapping[str, Equation] | None = None) -> Panel:
    """The main ``ask`` output."""
    equations = equations or {}
    style, word = STATUS[answer.status]

    header = Text()
    header.append(f" {word} ", style=f"badge.{style}")
    header.append(f"  {answer.category.replace('_', ' ')}", style="muted")

    parts: list[RenderableType] = [header, Text("")]

    if answer.final_value is not None:
        result = Text()
        result.append(pretty_number(answer.final_value), style="value")
        result.append(" " + pretty_unit(answer.unit or ""), style="brand")
        parts.append(Padding(result, (0, 0, 0, 2)))
        parts.append(Text(""))

    rows: list[tuple[str, RenderableType]] = []
    conf = answer.confidence
    meter = confidence_meter(conf.score)
    meter.append(f"  {conf.label} · {conf.score:.2f}", style=CONFIDENCE_STYLE[conf.label])
    rows.append(("confidence", meter))
    if answer.status == "refused" and answer.redirect:
        reason = answer.explanation.split(" A close question that can be answered:")[0]
        rows.append(("why", Text(reason)))
        rows.append(("try instead", Text(answer.redirect, style="accent")))
    else:
        rows.append(("explanation", Text(answer.explanation)))

    if answer.equations_used:
        eq_parts: list[RenderableType] = []
        for ref in answer.equations_used:
            title = Text()
            title.append(ref.id, style="accent")
            title.append(f"  {ref.name}", style="muted")
            eq_parts.append(title)
            eq = equations.get(ref.id)
            if eq is not None:
                eq_parts.append(
                    Padding(Text(pretty_equation(eq.sympy_expr), style="eq"), (0, 0, 0, 2))
                )
        rows.append(("equation", Group(*eq_parts)))

    if answer.inputs:
        inputs = Text()
        for i, k in enumerate(answer.inputs):
            if i:
                inputs.append("\n")
            inputs.append(f"{pretty_symbol(k.symbol)} = ", style="muted")
            inputs.append(
                f"{pretty_number(k.value)} {pretty_unit(k.unit)}", style=ORIGIN_STYLE[k.origin]
            )
            inputs.append(f"  {k.origin}", style="muted")
        rows.append(("inputs", inputs))

    if answer.steps:  # values found along the way when the plan chained equations
        steps = Text()
        for i, s in enumerate(answer.steps):
            if i:
                steps.append("\n")
            steps.append(f"{pretty_symbol(s.symbol)} = ", style="muted")
            steps.append(f"{pretty_number(s.value)} {pretty_unit(s.unit)}", style="value")
            steps.append(f"  from {s.equation_id}", style="muted")
        rows.append(("found", steps))

    if answer.assumptions:
        rows.append(("assumes", Text("\n".join(f"• {a}" for a in answer.assumptions))))
    if answer.models:
        rows.append(("models", models_line(answer)))
    if answer.caveats:
        rows.append(("caveats", Text("\n".join(f"• {c}" for c in answer.caveats), style="muted")))

    parts.append(_rows(rows))
    return Panel(
        Group(*parts),
        title=banner(),
        title_align="left",
        subtitle=Text(answer.question, style="muted"),
        subtitle_align="left",
        border_style=style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


# Rich estimates speed over this many seconds of updates. A run that only reported every
# logged step (about 45 s apart) needs a window longer than Rich's default 30 s, which holds
# at most one update, or the ETA never shows.
TRAINING_SPEED_WINDOW = 600.0


class EtaTracker:
    """Our own steps-per-second estimate, over the recent window of reported steps.

    Rich's estimate is the first choice; this one fills in when Rich has none. It starts at
    the step the run resumed from, so a resume never counts the jump to the checkpoint as
    speed. ``clock`` is injectable so tests need no real time.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.monotonic,
        window: float = TRAINING_SPEED_WINDOW,
    ) -> None:
        self._clock = clock
        self._window = window
        self._samples: deque[tuple[float, int]] = deque()

    def start(self, step: int) -> None:
        """Begin at ``step`` (0, or the resumed step), dropping any earlier samples."""
        self._samples.clear()
        self._samples.append((self._clock(), step))

    def record(self, step: int) -> None:
        now = self._clock()
        self._samples.append((now, step))
        # Keep the recent window, but never fewer than two samples: with one update every
        # 45 s the newest pair is still a usable rate.
        while len(self._samples) > 2 and self._samples[0][0] < now - self._window:
            self._samples.popleft()

    def steps_per_second(self) -> float | None:
        if len(self._samples) < 2:
            return None
        (t0, s0), (t1, s1) = self._samples[0], self._samples[-1]
        if t1 <= t0 or s1 <= s0:
            return None
        return (s1 - s0) / (t1 - t0)

    def seconds_left(self, step: int, total: int) -> float | None:
        rate = self.steps_per_second()
        if rate is None:
            return None
        return max(0, total - step) / rate


def format_duration(seconds: float) -> str:
    """``1:02:03`` for 3723 seconds; hours keep counting past a day."""
    whole = max(0, round(seconds))
    return f"{whole // 3600}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def format_finish(seconds_left: float, now: datetime) -> str:
    """The clock time a run ends: ``~03:42``, with the weekday when it is not today."""
    end = now + timedelta(seconds=seconds_left)
    return "~" + end.strftime("%H:%M" if end.date() == now.date() else "%a %H:%M")


class EtaColumn(ProgressColumn):
    """``eta 0:12:34 · done ~03:42``: Rich's estimate, or ``EtaTracker``'s when Rich has none."""

    def __init__(self, tracker: EtaTracker, now: Callable[[], datetime] = datetime.now) -> None:
        super().__init__()
        self._tracker = tracker
        self._now = now

    def render(self, task: Task) -> Text:
        if task.finished:
            return Text("done", style="ok")
        left: float | None = task.time_remaining
        if left is None and task.total is not None:
            left = self._tracker.seconds_left(int(task.completed), int(task.total))
        if left is None:
            return Text("eta …", style="muted")
        text = Text("eta ", style="muted")
        text.append(format_duration(left), style="value")
        text.append(f" · done {format_finish(left, self._now())}", style="muted")
        return text


class TrainingProgress(Progress):
    """The training bar, with an optional ``footer`` (the live panel) drawn under it."""

    def __init__(
        self,
        console: Console,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
        footer: Callable[[], RenderableType] | None = None,
    ) -> None:
        self.tracker = EtaTracker(clock)
        self.footer = footer
        super().__init__(
            SpinnerColumn(style="accent", finished_text="[ok]✓[/]"),
            TextColumn("[brand]{task.description}"),
            BarColumn(bar_width=32, complete_style="accent", finished_style="ok"),
            MofNCompleteColumn(),
            TextColumn("[label]loss[/] [value]{task.fields[loss]}"),
            TextColumn("[label]val[/] [accent]{task.fields[val]}"),
            TextColumn("[muted]{task.fields[speed]}"),
            TimeElapsedColumn(),
            EtaColumn(self.tracker, now),
            console=console,
            speed_estimate_period=TRAINING_SPEED_WINDOW,
            get_time=clock,
        )

    def get_renderables(self) -> Iterable[RenderableType]:
        yield from super().get_renderables()
        if self.footer is not None:
            yield self.footer()


def training_progress(
    console: Console,
    *,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = datetime.now,
) -> TrainingProgress:
    """Live progress for ``askphysics model train``."""
    return TrainingProgress(console, clock=clock, now=now)


def safe(text: str) -> str:
    """Escape Rich markup in user or model text."""
    return escape(text)
