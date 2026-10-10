"""The live view of ``askphysics model eval`` (and so ``scripts/eval.sh``).

On a terminal: a bar over the examples with an ETA that shows after the first example (and a
clock time to finish), the valid-plan rate and the confidently-wrong count as they settle, and
a bar of its own for the rescue pass. Each pass settles into a one-line summary with a tick, and
the results close as a panel. Anywhere else (CI, ``| tee``, ``NO_COLOR``): a plain line about
every tenth of the run, and plain result lines.

Everything here only formats; it reads what the evaluator already reports.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import BarColumn, ProgressColumn, SpinnerColumn, Task, TaskID, TextColumn
from rich.table import Table
from rich.text import Text

from askphysics.train_ui import INDENT, finished_row, format_elapsed, frame_width, is_plain
from askphysics.ui import (
    EtaColumn,
    EtaTracker,
    FooterProgress,
    format_duration,
    format_finish,
)

if TYPE_CHECKING:
    from askphysics.lm.evaluate import EvalReport, EvalTick, PhrasingReport, RealReport

RECENT = 60  # examples per kind the ETA averages over
ETA_ROUND = 5  # seconds: the ETA moves in steps of this, so it doesn't flicker
PLAIN_EVERY = 10  # plain output: a line each time this many percent more are done


class KindEta(EtaTracker):
    """An ETA that knows the examples come in kinds that cost different amounts.

    Classify examples decode a few tokens; plan examples decode a whole plan and run the
    router. The mix is random, so a single running rate swings with the luck of the draw.
    This keeps the recent mean time of each kind and prices what is left kind by kind. A kind
    not seen yet is priced at the overall mean, so an estimate exists after the first example.
    """

    def __init__(
        self,
        totals: Mapping[str, int],
        clock: Callable[[], float] = time.monotonic,
        recent: int = RECENT,
    ) -> None:
        super().__init__(clock, max_samples=recent + 1)
        self._totals = dict(totals)
        self._done: Counter[str] = Counter()
        self._times: dict[str, deque[float]] = {k: deque(maxlen=recent) for k in totals}
        self._last = clock()

    def start(self, step: int) -> None:
        super().start(step)
        self._last = self._clock()

    def record_example(self, step: int, kind: str) -> None:
        now = self._clock()
        self._times.setdefault(kind, deque(maxlen=RECENT)).append(now - self._last)
        self._last = now
        self._done[kind] += 1
        super().record(step)

    def seconds_left(self, step: int, total: int) -> float | None:
        every = [t for times in self._times.values() for t in times]
        if not every:
            return None
        overall = sum(every) / len(every)
        left = 0.0
        for kind, count in self._totals.items():
            times = self._times.get(kind)
            mean = sum(times) / len(times) if times else overall
            left += max(0, count - self._done[kind]) * mean
        return left


class _Elapsed(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(format_duration(task.finished_time or task.elapsed or 0), style="muted")


class _Count(ProgressColumn):
    def render(self, task: Task) -> Text:
        total = int(task.total or 0)
        width = len(f"{total:,}")
        return Text(f"{int(task.completed):>{width},}/{total:,}", style="value")


class _Percent(ProgressColumn):
    def render(self, task: Task) -> Text:
        return Text(f"{task.percentage:>3.0f}%", style="muted")


class EvalProgress(FooterProgress):
    """The eval bars, indented to match the step lines, with a stats footer under them."""

    def __init__(
        self,
        console: Console,
        *,
        clock: Callable[[], float],
        now: Callable[[], datetime],
        footer: Callable[[], RenderableType],
        fallback: EtaTracker,
    ) -> None:
        self.footer = footer
        self._width = frame_width(console)
        super().__init__(
            SpinnerColumn(style="accent", finished_text="[ok]✓[/]"),
            TextColumn("[brand]{task.description:<8}"),
            BarColumn(bar_width=None, complete_style="accent", finished_style="ok", style="faint"),
            _Count(),
            _Percent(),
            _Elapsed(),
            EtaColumn(fallback, now, prefer_tracker=True, round_to=ETA_ROUND),
            console=console,
            get_time=clock,
            expand=True,
            refresh_per_second=12,
            transient=True,  # each pass settles into a printed line; the live area just goes
        )

    def get_renderables(self) -> Iterable[RenderableType]:
        table = self.make_tasks_table(self.tasks)
        yield Padding(Constrain(table, self._width - INDENT), (0, 0, 0, INDENT))
        if self.footer is not None:
            yield Padding(self.footer(), (0, 0, 0, INDENT + 2))


def working(console: Console, text: str) -> AbstractContextManager[object]:
    """A spinner with ``text`` while something quiet happens, such as loading a model."""
    if is_plain(console):
        console.print(Text(text), soft_wrap=True)
        return nullcontext()
    return console.status(Text(text, style="muted"), spinner="dots", spinner_style="accent")


def _rate(part: int, whole: int) -> str:
    return f"{100 * part / whole:.1f}%" if whole else "…"


class EvalMonitor:
    """Feed it ``on_tick`` (an ``EvalTick`` per example); use it as a context manager.

    ``kinds`` is how many examples of each task there are (``{"classify": 120, "plan": 120}``),
    so the ETA can price what is left. Each pass prints its own settled line when it ends.
    """

    def __init__(
        self,
        console: Console,
        model: str,
        kinds: Mapping[str, int],
        *,
        rescue_with: str | None = None,
        label: str = "scoring",
        noun: str = "examples",
        stats: bool = True,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.console = console
        self.model = model
        self.rescue_with = rescue_with
        self.label = label  # the bar's name
        self.noun = noun  # "Scored 3,000 examples"
        self.stats = stats  # whether valid-plan and confidently-wrong are known (not for real)
        self.plain = is_plain(console)
        self.total = sum(kinds.values())
        self._clock = clock
        self._now = now
        self.score_eta = KindEta(kinds, clock)
        self.rescue_eta = EtaTracker(clock, max_samples=20)
        self._tick: EvalTick | None = None
        self._t0 = clock()
        self._stage_t0 = self._t0
        self._printed = 0  # plain output: the last tenth reported
        self.progress: EvalProgress | None = None
        self._task: TaskID | None = None
        if not self.plain:
            self.progress = EvalProgress(
                console, clock=clock, now=now, footer=self._footer, fallback=self.score_eta
            )

    def __enter__(self) -> EvalMonitor:
        self._t0 = self._stage_t0 = self._clock()
        self.score_eta.start(0)
        if self.progress is not None:
            self.progress.start()
            self._task = self.progress.add_task(
                self.label, total=self.total, tracker=self.score_eta
            )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.progress is not None:
            self.progress.stop()

    def _speed(self) -> str:
        seconds = self._clock() - self._stage_t0
        tick = self._tick
        if tick is None or seconds <= 0 or tick.done == 0:
            return "…"
        rate = tick.done / seconds
        return f"{rate:.1f} ex/s" if rate >= 1 else f"{1 / rate:.1f} s/ex"

    def _footer(self) -> RenderableType:
        tick = self._tick
        grid = Table.grid(padding=(0, 4))
        for _ in range(3):
            grid.add_column(no_wrap=True)
        speed = Text.assemble(("speed ", "muted"), (self._speed(), "value"))
        if tick is not None and tick.stage == "rescue":
            rescued = Text.assemble(
                ("rescued ", "muted"),
                (f"{tick.rescued} of {tick.rescue_tried}", "value"),
                (f"  {_rate(tick.rescued, tick.done)}" if tick.done else "", "muted"),
            )
            wrong = tick.rescue_confident
            bad = Text.assemble(
                ("confidently wrong ", "muted"), (str(wrong), "bad" if wrong else "ok")
            )
            grid.add_row(rescued, bad, speed)
            return grid
        if not self.stats:
            grid.add_row(speed)
            return grid
        plans = tick.plans if tick else 0
        valid = Text.assemble(
            ("valid plan ", "muted"),
            (_rate(tick.valid, plans) if tick else "…", "value"),
        )
        wrong = tick.confident if tick else 0
        confident = Text.assemble(
            ("confidently wrong ", "muted"), (str(wrong), "bad" if wrong else "ok")
        )
        grid.add_row(valid, confident, speed)
        return grid

    def _eta(self, tick: EvalTick) -> str:
        tracker = self.score_eta if tick.stage == "score" else self.rescue_eta
        left = tracker.seconds_left(tick.done, tick.total)
        if left is None:
            return "eta n/a"
        return f"eta {format_duration(left)} (done {format_finish(left, self._now())})"

    def on_tick(self, tick: EvalTick) -> None:
        self._tick = tick
        if tick.stage == "rescue":
            self._rescue_tick(tick)
            return
        self.score_eta.record_example(tick.done, tick.task)
        if self.progress is not None and self._task is not None:
            self.progress.update(self._task, completed=tick.done)
        elif self.plain:
            self._plain_progress(self.label, tick)
        if tick.done == tick.total:
            self._settle(tick)

    def _rescue_tick(self, tick: EvalTick) -> None:
        if tick.done == 0:  # the rescue pass is about to start
            self._stage_t0 = self._clock()
            self.rescue_eta.start(0)
            self._printed = 0
            if tick.total == 0:
                self._settle(tick)
            elif self.progress is not None:
                self._task = self.progress.add_task(
                    "rescue", total=tick.total, tracker=self.rescue_eta
                )
            return
        self.rescue_eta.record(tick.done)
        if self.progress is not None and self._task is not None:
            self.progress.update(self._task, completed=tick.done)
        elif self.plain:
            self._plain_progress(f"rescue with {self.rescue_with}", tick)
        if tick.done == tick.total:
            self._settle(tick)

    def _plain_progress(self, label: str, tick: EvalTick) -> None:
        tenth = 100 * tick.done // tick.total // PLAIN_EVERY
        if tick.done != tick.total and tenth == self._printed and tick.done != 1:
            return
        self._printed = tenth
        if tick.stage == "score":
            facts = (
                [
                    f"valid plan {_rate(tick.valid, tick.plans)}",
                    f"confidently wrong {tick.confident}",
                ]
                if self.stats
                else []
            )
        else:
            facts = [f"rescued {tick.rescued} of {tick.done}"]
        parts = [f"{label} {tick.done}/{tick.total} ({100 * tick.done // tick.total}%)", *facts]
        if tick.done != tick.total:
            parts.append(self._eta(tick))
        self.console.print(Text(" | ".join(parts)), soft_wrap=True)

    def _settle(self, tick: EvalTick) -> None:
        """A pass is over: its bar becomes one line with a tick and the time it took."""
        seconds = self._clock() - self._stage_t0
        if tick.stage == "score":
            title = f"Scored {tick.total:,} {self.noun}"
            detail = (
                f"valid plan {_rate(tick.valid, tick.plans)} · confidently wrong {tick.confident}"
                if self.stats
                else ""
            )
        else:
            title = f"Rescue pass with {self.rescue_with}"
            detail = f"rescued {tick.rescued} of {tick.total}"
        if self.plain:
            plain_detail = detail.replace(" · ", " | ")
            tail = f" | {plain_detail}" if plain_detail else ""
            self.console.print(
                Text(f"done: {title} ({format_elapsed(seconds)}){tail}"), soft_wrap=True
            )
            return
        if self.progress is not None and self._task is not None and tick.total:
            self.progress.update(self._task, completed=tick.total)
            self.progress.remove_task(self._task)
        row = finished_row(title, seconds, 0, width=frame_width(self.console), detail=detail)
        self.console.print(row)


@dataclass(frozen=True)
class Row:
    """One line of the results: a label, a rate (or a count), and which way is good."""

    label: str
    value: float | int | None = None
    hint: str = ""
    good: str = "high"  # "high", "low", or "plain" (no verdict)
    count: bool = False  # show value as a count, not a percentage


def meter(value: float, good: str, width: int = 20) -> Text:
    """A bar for a rate, coloured by how good it is (green, yellow, red) when it has a verdict."""
    filled = round(max(0.0, min(1.0, value)) * width)
    if good == "high":
        style = "ok" if value >= 0.9 else "warn" if value >= 0.7 else "bad"
    elif good == "low":
        style = "ok" if value <= 0.01 else "warn" if value <= 0.05 else "bad"
    else:
        style = "accent"
    return Text("━" * filled, style=style) + Text("━" * (width - filled), style="faint")


def results_panel(
    title: str, sections: Sequence[Sequence[Row]], footer: Sequence[str] = ()
) -> RenderableType:
    """Results as a panel: sections of rows with a bar and a percentage each."""
    grid = Table(
        box=box.HORIZONTALS,
        show_header=False,
        show_edge=False,
        pad_edge=False,
        padding=(0, 1),
        border_style="faint",
    )
    grid.add_column(overflow="fold")
    grid.add_column(style="muted", overflow="fold")
    grid.add_column(no_wrap=True)
    grid.add_column(justify="right", no_wrap=True)
    for i, rows in enumerate(sections):
        if i:
            grid.add_section()
        for row in rows:
            if row.value is None:
                grid.add_row(Text(row.label, style="label"), Text(row.hint), "", "")
            elif row.count:
                style = "bad" if row.value and row.good == "low" else "ok"
                grid.add_row(Text(row.label), Text(row.hint), "", Text(str(row.value), style=style))
            else:
                grid.add_row(
                    Text(row.label),
                    Text(row.hint),
                    meter(float(row.value), row.good),
                    Text(f"{row.value:.1%}", style="value"),
                )
    parts: list[RenderableType] = [grid]
    for line in footer:
        parts.append(Text(line, style="muted"))
    return Panel(
        Group(*parts),
        title=Text(title, style="brand"),
        title_align="left",
        border_style="muted",
        box=box.ROUNDED,
        padding=(1, 2),
    )


def report_sections(
    report: EvalReport, *, attempts: int, rescue_with: str | None
) -> tuple[list[list[Row]], list[str]]:
    tries = f"up to {attempts} tries · {report.mean_tries:.2f} avg"
    sections = [
        [
            Row("right category", report.category_accuracy, f"{report.classify_examples} classify"),
            Row("right equation", report.equation_accuracy, f"{report.plan_examples} plan"),
            Row("right target", report.target_accuracy),
            Row("right numbers and units", report.knowns_accuracy),
            Row("valid plan (right answer)", report.valid_plan_rate),
        ],
        [
            Row("right answer as ask gives it", report.routed_right_rate, tries),
            Row("wrong, but flagged or refused", report.flagged_wrong_rate, good="plain"),
            Row("confidently wrong", report.confidently_wrong_rate, good="low"),
        ],
    ]
    footer: list[str] = []
    if rescue_with is not None:
        sections.append(
            [
                Row(
                    f"{rescue_with} tried on {report.rescue_tried} "
                    f"{'miss' if report.rescue_tried == 1 else 'misses'}",
                    None,
                ),
                Row("rescued", report.rescue_rate),
                Row("right after escalation", report.routed_with_rescue_rate),
                Row(
                    "confidently wrong rescues",
                    report.rescue_confidently_wrong,
                    good="low",
                    count=True,
                ),
            ]
        )
        footer.append("v0.4 target: rescues at least a third of the misses")
    return sections, footer


def real_sections(report: RealReport) -> list[list[Row]]:
    return [
        [
            Row("classified standard", report.standard_rate),
            Row("right equations retrieved", report.retrieved_rate),
        ],
        [
            Row("right answer as ask gives it", report.right_rate),
            Row("no answer, or wrong but flagged", report.flagged_wrong_rate, good="plain"),
            Row("confidently wrong", report.confidently_wrong_rate, good="low"),
        ],
    ]


def phrasing_sections(report: PhrasingReport) -> list[list[Row]]:
    return [
        [
            Row("plain questions answered, not refused", report.answerable_rate),
            Row("look-alikes with no physical subject refused", report.refused_rate),
        ]
    ]


def print_results(
    console: Console,
    title: str,
    sections: Sequence[Sequence[Row]],
    footer: Sequence[str] = (),
) -> None:
    """The results panel, or plain ``label: 97.9%`` lines when output is not a terminal."""
    if is_plain(console):
        console.print(Text(title), soft_wrap=True)
        for rows in sections:
            for row in rows:
                if row.value is None:
                    console.print(Text(f"  {row.label}"), soft_wrap=True)
                    continue
                shown = str(row.value) if row.count else f"{row.value:.1%}"
                hint = f" ({row.hint})" if row.hint else ""
                console.print(Text(f"  {row.label}{hint}: {shown}"), soft_wrap=True)
        for line in footer:
            console.print(Text(f"  {line}"), soft_wrap=True)
        return
    panel = results_panel(title, sections, footer)
    console.print(Padding(Constrain(panel, frame_width(console) - INDENT), (0, 0, 0, INDENT)))
