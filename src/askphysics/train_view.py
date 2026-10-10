"""The live training view for ``askphysics-dev model train``.

On a terminal: the progress bar with a live panel under it (phase, loss sparkline, learning
rate, speed, backend and device, memory), and a summary panel at the end. Anywhere else (CI,
``| tee``, ``NO_COLOR``, a dumb terminal): plain ASCII lines, one per logged step, no animation.

Everything here only formats; it reads what the trainer already reports and never touches the
device, so the training math is the same with or without it.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import TaskID
from rich.table import Table
from rich.text import Text

from askphysics.train_ui import format_elapsed, is_plain
from askphysics.ui import EtaTracker, TrainingProgress, format_duration, format_finish

SPARK_BLOCKS = "▁▂▃▄▅▆▇█"
SPARK_WIDTH = 24  # logged losses shown in the sparkline


def sparkline(values: Sequence[float], width: int = SPARK_WIDTH) -> str:
    """The last ``width`` values as block characters, scaled to their own range."""
    recent = list(values)[-width:]
    if not recent:
        return ""
    low, high = min(recent), max(recent)
    span = high - low
    top = len(SPARK_BLOCKS) - 1
    return "".join(SPARK_BLOCKS[round((v - low) / span * top) if span else 0] for v in recent)


@dataclass(frozen=True)
class RunInfo:
    """What the display needs to know about a training run."""

    name: str
    backend: str
    device: str
    steps: int
    out_dir: Path
    has_prose: bool = False
    prose_steps: int = 0
    prose_share: float = 0.0

    def phase(self, step: int) -> str:
        """Where ``step`` falls: the prose warm-up, or the tasks (mixed with prose)."""
        warm = self.has_prose and self.prose_steps > 0
        if warm and step < self.prose_steps:
            return f"prose warm-up (steps 0-{self.prose_steps})"
        kind = "tasks + prose" if self.has_prose and self.prose_share > 0 else "tasks"
        return f"{kind} (steps {self.prose_steps if warm else 0}-{self.steps})"


def _fmt(value: float | None, spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


class TrainingMonitor:
    """Feed it ``on_step`` (every optimizer step, cheap) and ``on_log`` (each metrics entry).

    Use it as a context manager around the run, then call ``print_summary``.
    """

    def __init__(
        self,
        console: Console,
        info: RunInfo,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.console = console
        self.info = info
        self.plain = is_plain(console)
        self._clock = clock
        self._now = now
        self._task: TaskID | None = None
        self._started = False
        self._t0 = clock()
        self._step = 0
        self._phase = ""
        self._losses: list[float] = []
        self._rates: list[float] = []
        self._lr: float | None = None
        self._mem: float | None = None
        self._val: dict[str, Any] = {}
        self._best_val: tuple[int, float] | None = None
        self._last_loss: float | None = None

        self.progress: TrainingProgress | None = None  # the footer reads the state above
        self.tracker: EtaTracker
        if self.plain:
            self.tracker = EtaTracker(clock)
        else:
            self.progress = TrainingProgress(console, clock=clock, now=now, footer=self._panel)
            self.tracker = self.progress.tracker

    def __enter__(self) -> TrainingMonitor:
        if self.progress is not None:
            self.progress.start()
            self._task = self.progress.add_task(
                self.info.name, total=self.info.steps, loss="…", val="…", speed=""
            )
        return self

    def __exit__(self, *exc: object) -> None:
        if self.progress is not None:
            self.progress.stop()

    def on_step(self, done: int) -> None:
        """Steps done so far; the first call is the step the run starts (or resumes) from."""
        self._step = done
        if not self._started:
            self._started = True
            self._t0 = self._clock()
            self.tracker.start(done)
            if self.progress is not None and self._task is not None:
                self.progress.reset(self._task, completed=done)
            self._announce(done, first=True)
            return
        self.tracker.record(done)
        if self.progress is not None and self._task is not None:
            self.progress.update(self._task, completed=done)
        if done < self.info.steps and self.info.phase(done) != self._phase:
            self._announce(done, first=False)

    def _announce(self, step: int, *, first: bool) -> None:
        self._phase = self.info.phase(step)
        if self.plain:
            lead = "phase" if first else f"phase at step {step}"
            self.console.print(Text(f"{lead}: {self._phase}"), soft_wrap=True)
        elif not first:
            line = Text("→ ", style="accent") + Text(f"step {step:,}: ", style="muted")
            self.console.print(line + Text(self._phase, style="brand"))

    def on_log(self, entry: dict[str, Any]) -> None:
        step = int(entry["step"])
        if "val_loss" in entry:
            self._log_val(step, entry)
            return
        self._step = max(self._step, step)
        self._last_loss = float(entry["loss"])
        self._losses.append(self._last_loss)
        self._lr = float(entry["lr"]) if "lr" in entry else None
        rate = float(entry["target_tokens_per_s"])
        self._rates.append(rate)
        self._mem = float(entry["mem_gb"]) if entry.get("mem_gb") else None
        if self.progress is not None and self._task is not None:
            self.progress.update(
                self._task,
                completed=step,
                loss=f"{self._last_loss:.3f}",
                speed=f"{rate:,.0f} tok/s",
            )
        if self.plain:
            self.console.print(Text(self._plain_line(step, rate)), soft_wrap=True)

    def _log_val(self, step: int, entry: dict[str, Any]) -> None:
        self._val = entry
        value = float(entry["val_loss"])
        if self._best_val is None or value < self._best_val[1]:
            self._best_val = (step, value)
        if self.progress is not None and self._task is not None:
            self.progress.update(self._task, val=f"{value:.3f}")
        if self.plain:
            parts = [f"step {step}/{self.info.steps}", f"val loss {value:.3f}"]
            if "val_loss_prose" in entry:
                parts.append(f"prose {float(entry['val_loss_prose']):.3f}")
            parts.append(f"best {self._best_val[1]:.3f} at step {self._best_val[0]}")
            self.console.print(Text(" | ".join(parts)), soft_wrap=True)

    def _eta_text(self, step: int) -> str:
        left = self.tracker.seconds_left(step, self.info.steps)
        if left is None:
            return "eta n/a"
        return f"eta {format_duration(left)} (done {format_finish(left, self._now())})"

    def _plain_line(self, step: int, rate: float) -> str:
        total = self.info.steps
        parts = [f"step {step}/{total} ({100 * step // total}%)", f"loss {self._last_loss:.3f}"]
        if self._lr is not None:
            parts.append(f"lr {self._lr:.2e}")
        parts.append(f"{rate:,.0f} tok/s")
        if self._mem is not None:
            parts.append(f"peak mem {self._mem:.1f} GB")
        parts.append(self._eta_text(step))
        return " | ".join(parts)

    def _panel(self) -> RenderableType:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="label", justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        grid.add_column(style="label", justify="right", no_wrap=True)
        grid.add_column(no_wrap=True)
        phase = self.info.phase(min(self._step, self.info.steps - 1))
        loss = Text(sparkline(self._losses), style="accent")
        if self._last_loss is not None:
            loss.append(f" {self._last_loss:.3f}", style="value")
        val = Text("…", style="muted")
        if self._best_val is not None and self._val:
            val = Text(f"{float(self._val['val_loss']):.3f}", style="value")
            val.append(f"  best {self._best_val[1]:.3f} at {self._best_val[0]:,}", style="muted")
        rate = f"{self._rates[-1]:,.0f} tok/s" if self._rates else "…"
        memory = "n/a" if self._mem is None else f"{self._mem:.1f} GB peak"
        run = f"{self.info.backend} · {self.info.device}"
        grid.add_row("phase", Text(phase, style="brand"), "run", Text(run))
        grid.add_row("loss", loss, "speed", Text(rate, style="value"))
        grid.add_row("val", val, "lr", Text(_fmt(self._lr, ".2e"), style="value"))
        grid.add_row("", "", "memory", Text(memory, style="value"))
        return Panel(grid, title="live", title_align="left", border_style="muted", box=box.ROUNDED)

    def renderable(self) -> RenderableType:
        """What the live display draws now (the bar and the panel); for tests and screenshots."""
        assert self.progress is not None, "plain output has no live view"
        return self.progress.get_renderable()

    def summary_facts(self) -> list[tuple[str, str]]:
        total = self._clock() - self._t0
        avg = sum(self._rates) / len(self._rates) if self._rates else None
        best = "n/a"
        if self._best_val is not None:
            best = f"{self._best_val[1]:.3f} at step {self._best_val[0]:,}"
        final_val = float(self._val["val_loss"]) if self._val else None
        facts = [
            ("final train loss", _fmt(self._last_loss)),
            ("final val loss", _fmt(final_val)),
            ("best val", best),
            ("total time", format_elapsed(total)),
            ("average speed", "n/a" if avg is None else f"{avg:,.0f} tok/s"),
        ]
        by_task = [
            f"{key.removeprefix('val_loss_')} {float(value):.3f}"
            for key, value in self._val.items()
            if key.startswith("val_loss_")
        ]
        if by_task:
            facts.append(("val by task", " · ".join(by_task)))
        facts.append(("weights", str(self.info.out_dir)))
        return facts

    def print_summary(self) -> None:
        """Close the bar and print what the run came to."""
        if self.progress is not None and self._task is not None:
            self.progress.update(self._task, completed=self.info.steps)
        facts = self.summary_facts()
        if self.plain:
            self.console.print(Text(f"training complete: {self.info.name}"), soft_wrap=True)
            for label, value in facts:
                self.console.print(Text(f"  {label}: {value}"), soft_wrap=True)
            return
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="label", justify="right", no_wrap=True)
        grid.add_column(overflow="fold")
        for label, value in facts:
            grid.add_row(label, Text(value, style="accent" if label == "weights" else "value"))
        title = Text.assemble(("✓ ", "ok"), (f"{self.info.name} trained", "brand"))
        self.console.print(
            Panel(grid, title=title, title_align="left", border_style="ok", box=box.ROUNDED)
        )
