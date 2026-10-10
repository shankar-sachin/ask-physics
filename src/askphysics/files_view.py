"""Files on a terminal: copying a model, downloading one, listing what a release ships.

Every file gets a line that settles into a tick with its size, its sha256 and the verdict
(``verified``), under a header for the model; while a file moves it has a bar with size, speed
and ETA. Without a terminal (CI, ``| tee``, ``NO_COLOR``) the same facts print as plain lines.

Everything here only moves and reports bytes; no model is read or changed.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from rich.console import Console, RenderableType
from rich.constrain import Constrain
from rich.padding import Padding
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
)
from rich.table import Table
from rich.text import Text

from askphysics.train_ui import INDENT, finished_row, format_elapsed, frame_width, is_plain
from askphysics.ui import EtaColumn, EtaTracker

CHUNK = 1024 * 1024
NESTED = INDENT + 2  # files sit one level under their model
SHOWN_FILES = 8  # files listed per model before the smaller ones fold into a count


def human_size(size: float) -> str:
    """``812 B``, ``12.4 KB``, ``240.1 MB``: decimal units, as the release page shows them."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1000 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1000
    return f"{value:.1f} GB"  # unreachable; keeps the type checker content


def sha256_of(path: Path, on_bytes: Callable[[int], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
            if on_bytes:
                on_bytes(len(chunk))
    return digest.hexdigest()


@dataclass(frozen=True)
class FileFact:
    """A finished file: its name, size, sha256, and whether the sha256 checked out."""

    name: str
    size: int
    sha256: str
    verdict: str  # "verified", "matches the pin", "not pinned", "MISMATCH", ...
    seconds: float | None = None

    @property
    def ok(self) -> bool:
        return self.verdict != "MISMATCH" and not self.verdict.startswith("missing")


class _Size(ProgressColumn):
    def render(self, task: Task) -> Text:
        total = int(task.total or 0)
        return Text(f"{human_size(task.completed)} / {human_size(total)}", style="value")


class _Speed(ProgressColumn):
    def render(self, task: Task) -> Text:
        speed = task.finished_speed or task.speed
        return Text(f"{human_size(speed)}/s" if speed else "…", style="muted")


class TransferProgress(Progress):
    """A bar for the file in flight: size, speed, and ETA, indented under its model."""

    def __init__(
        self,
        console: Console,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self._width = frame_width(console)
        super().__init__(
            SpinnerColumn(style="accent"),
            TextColumn("[brand]{task.description}"),
            BarColumn(bar_width=None, complete_style="accent", finished_style="ok", style="faint"),
            _Size(),
            _Speed(),
            EtaColumn(EtaTracker(clock), now, round_to=1),
            console=console,
            get_time=clock,
            expand=True,
            transient=True,
        )

    def get_renderables(self) -> Iterable[RenderableType]:
        table = self.make_tasks_table(self.tasks)
        yield Padding(Constrain(table, self._width - NESTED), (0, 0, 0, NESTED))


def _file_row(fact: FileFact, console: Console, pad: int) -> RenderableType:
    detail = f"{human_size(fact.size)} · sha256 {fact.sha256[:12]} · {fact.verdict}"
    return finished_row(
        fact.name,
        fact.seconds,
        0 if fact.ok else 1,
        width=frame_width(console),
        detail=detail,
        pad=pad,
        indent=NESTED,
    )


def print_files(console: Console, facts: Sequence[FileFact]) -> None:
    """The files of one model as settled lines, the largest first, the rest folded."""
    ordered = sorted(facts, key=lambda f: (-f.size, f.name))
    shown, folded = ordered[:SHOWN_FILES], ordered[SHOWN_FILES:]
    if is_plain(console):
        for fact in ordered:
            console.print(
                Text(
                    f"  file: {fact.name} ({human_size(fact.size)}) "
                    f"sha256 {fact.sha256[:12]} {fact.verdict}"
                ),
                soft_wrap=True,
            )
        return
    pad = max((len(f.name) for f in shown), default=0)
    for fact in shown:
        console.print(_file_row(fact, console, pad))
    if folded:
        bad = [f for f in folded if not f.ok]
        extra = Text(
            f"and {len(folded)} smaller files · {human_size(sum(f.size for f in folded))}",
            style="muted" if not bad else "bad",
        )
        console.print(Padding(extra, (0, 0, 0, NESTED + 2)))


def heading(console: Console, title: str, detail: str = "") -> None:
    """The line a model's files hang from."""
    if is_plain(console):
        shown = detail.replace(" · ", ", ")
        console.print(Text(f"start: {title}" + (f" ({shown})" if shown else "")), soft_wrap=True)
        return
    line = Text.assemble(("◉ ", "accent"), (title, "brand"))
    if detail:
        line.append(f"   {detail}", style="muted")
    console.print(Padding(line, (0, 0, 0, INDENT)))


def settle(
    console: Console, title: str, seconds: float, ok: bool, detail: str = "", indent: int = INDENT
) -> None:
    """The closing line of a model: a tick (or a cross), what happened, the time it took."""
    if is_plain(console):
        word = "done" if ok else "failed"
        console.print(Text(f"{word}: {title} ({format_elapsed(seconds)})"), soft_wrap=True)
        return
    console.print(
        finished_row(
            title,
            seconds,
            0 if ok else 1,
            width=frame_width(console),
            detail=detail,
            indent=indent,
        )
    )


def copy_tree(
    console: Console,
    source: Path,
    dest: Path,
    title: str,
    *,
    clock: Callable[[], float] = time.monotonic,
) -> int:
    """Copy a directory file by file, checking each copy's sha256 against its source.

    The source is hashed as it is read and the copy is hashed after it is written, so a bit
    flipped in transit is caught rather than backed up. Returns 0, or 1 when anything differs.
    """
    if dest.exists():
        console.print(Text(f"{dest} already exists; not overwriting it"), soft_wrap=True)
        return 1
    files = sorted(p for p in source.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    started = clock()
    heading(console, title, f"{len(files)} files · {human_size(total)}")
    facts: list[FileFact] = []
    plain = is_plain(console)
    progress = None if plain else TransferProgress(console, clock=clock)
    if progress is not None:
        progress.start()
    try:
        for path in files:
            relative = path.relative_to(source)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            size = path.stat().st_size
            t0 = clock()
            task = progress.add_task(str(relative), total=2 * size) if progress else None
            digest = hashlib.sha256()
            with path.open("rb") as src, target.open("wb") as out:
                for chunk in iter(lambda: src.read(CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)
                    if progress and task is not None:
                        progress.advance(task, len(chunk))
            shutil.copystat(path, target)
            copied = sha256_of(target, _advancer(progress, task))
            if progress and task is not None:
                progress.remove_task(task)
            verdict = "verified" if copied == digest.hexdigest() else "MISMATCH"
            facts.append(FileFact(str(relative), size, copied, verdict, clock() - t0))
    finally:
        if progress is not None:
            progress.stop()
    print_files(console, facts)
    bad = [f.name for f in facts if not f.ok]
    detail = (
        f"{len(facts)} files · {human_size(total)} · sha256 verified"
        if not bad
        else ("differs: " + ", ".join(bad))
    )
    settle(console, title, clock() - started, not bad, detail)
    return 1 if bad else 0


def _advancer(progress: Progress | None, task: TaskID | None) -> Callable[[int], None] | None:
    if progress is None or task is None:
        return None

    def advance(count: int) -> None:
        progress.advance(task, count)

    return advance


def list_models(console: Console, title: str, root: Path) -> int:
    """What is in a directory of models or backups: name, size, files, and when it changed."""
    entries = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    if is_plain(console):
        console.print(Text(f"{title}: {root}"), soft_wrap=True)
        for path in entries:
            console.print(Text(f"  {path.name}"), soft_wrap=True)
        if not entries:
            console.print(Text("  (none)"), soft_wrap=True)
        return 0
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="brand", no_wrap=True)
    grid.add_column(justify="right", style="value", no_wrap=True)
    grid.add_column(justify="right", style="muted", no_wrap=True)
    grid.add_column(style="muted", no_wrap=True)
    for path in entries:
        files = [p for p in path.rglob("*") if p.is_file()]
        size = sum(p.stat().st_size for p in files)
        changed = datetime.fromtimestamp(max((p.stat().st_mtime for p in files), default=0))
        grid.add_row(
            path.name, human_size(size), f"{len(files)} files", f"{changed:%Y-%m-%d %H:%M}"
        )
    if not entries:
        grid.add_row("(none)", "", "", "")
    heading(console, title, str(root))
    console.print(Padding(grid, (0, 0, 0, NESTED)))
    console.print()
    return 0


def check_assets(console: Console, model: str, asset_dir: Path, manifest_path: Path) -> int:
    """Each release file of ``model`` with its size and sha256, checked against the pin."""
    import json

    pinned: dict[str, dict[str, object]] = {}
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        pinned = data.get("models", {}).get(model, {}).get("files", {})
    by_asset = {str(Path(str(v["url"])).name): v for v in pinned.values()}
    files = sorted(p for p in asset_dir.glob(f"{model}.*") if p.is_file())
    started = time.monotonic()
    heading(console, f"{model} release files", f"{len(files)} files in {asset_dir}")
    facts: list[FileFact] = []
    for path in files:
        t0 = time.monotonic()
        digest = sha256_of(path)
        pin = by_asset.get(path.name)
        if pin is None:
            verdict = "not pinned"
        elif pin["sha256"] == digest and pin["size"] == path.stat().st_size:
            verdict = "matches the pin"
        else:
            verdict = "MISMATCH"
        facts.append(
            FileFact(path.name, path.stat().st_size, digest, verdict, time.monotonic() - t0)
        )
    for name in sorted(set(by_asset) - {f.name for f in facts}):
        facts.append(FileFact(name, 0, "-" * 12, "missing from the folder"))
    print_files(console, facts)
    bad = [f.name for f in facts if not f.ok]
    total = sum(f.size for f in facts)
    detail = (
        f"{len(facts)} files · {human_size(total)} · all pinned in weights.json"
        if not bad
        else "check: " + ", ".join(bad)
    )
    settle(console, f"{model} ready to upload", time.monotonic() - started, not bad, detail)
    return 1 if bad else 0


class PullView:
    """The display of ``askphysics model pull``: a bar per file, then a settled line per file."""

    def __init__(
        self,
        console: Console,
        *,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.console = console
        self._clock = clock
        self.plain = is_plain(console)
        self._progress = None if self.plain else TransferProgress(console, clock=clock, now=now)
        self._tasks: dict[str, TaskID] = {}
        self._started: dict[str, float] = {}
        self._ended: dict[str, float] = {}
        self._last_printed = -1

    def __enter__(self) -> PullView:
        if self._progress is not None:
            self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._progress is not None:
            self._progress.stop()

    def start_model(self, name: str, files: Sequence[tuple[str, int]]) -> None:
        self._tasks.clear()
        self._started.clear()
        self._ended.clear()
        total = sum(size for _, size in files)
        heading(self.console, name, f"downloading {len(files)} files · {human_size(total)}")

    def update(self, name: str, done: int, total: int) -> None:
        """The callback ``askphysics.lm.weights.pull`` reports each chunk to."""
        self._started.setdefault(name, self._clock())
        if done >= total:
            self._ended[name] = self._clock()
        if self._progress is not None:
            if name not in self._tasks:
                self._tasks[name] = self._progress.add_task(name, total=total)
            self._progress.update(self._tasks[name], completed=done)
            if done >= total:
                self._progress.remove_task(self._tasks[name])
        elif total and done * 10 // total > self._last_printed and done < total:
            self._last_printed = done * 10 // total
            self.console.print(
                Text(f"  {name}: {100 * done // total}% of {human_size(total)}"), soft_wrap=True
            )

    def finish_model(
        self, name: str, files: Sequence[tuple[str, int, str]], downloaded: bool, where: Path
    ) -> None:
        """``files`` is (name, size, sha256) per pinned file; install checked every one."""
        verdict = "verified" if downloaded else "already up to date"
        facts = [
            FileFact(
                file,
                size,
                sha,
                verdict,
                (self._ended[file] - self._started[file]) if file in self._ended else None,
            )
            for file, size, sha in files
        ]
        print_files(self.console, facts)
        state = "downloaded and verified" if downloaded else "already up to date"
        started = min(self._started.values(), default=self._clock())
        settle(self.console, f"{name} {state}", self._clock() - started, True, str(where))
