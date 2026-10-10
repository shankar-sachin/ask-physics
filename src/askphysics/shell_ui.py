"""The terminal look of the workflow scripts: ``python -m askphysics.shell_ui``.

``scripts/lib.sh`` calls this for every header, step, result, summary and panel, so ``check.sh``,
``setup.sh``, ``eval.sh`` and the rest read as one product with the CLI (same theme, same
symbols). It uses Rich only and imports nothing heavy, so each call starts in a fraction of a
second.

    python -m askphysics.shell_ui header --title Check --about "..." --steps 6
    python -m askphysics.shell_ui step --title Tests --index 6 --total 6 -- python -m pytest -q
    python -m askphysics.shell_ui summary --title "All clear" --steps 6 --passed 6 --since 17e8
    python -m askphysics.shell_ui ready --title "Ask Physics is ready" --row "cmd::what it does"

Everything here only formats and times. When output is not a terminal, when ``NO_COLOR`` is set,
or when the terminal is dumb, it prints plain lines with no animation and no escape codes; the
pure-sh renderer in ``scripts/lib.sh`` prints the same lines, for a machine with no Python.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from rich import box
from rich.console import Console, Group, RenderableType
from rich.constrain import Constrain
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from askphysics.train_ui import (
    INDENT,
    PARSERS,
    PLAIN_INDENT,
    finished_row,
    format_elapsed,
    frame_width,
    is_plain,
    run_phase,
)
from askphysics.ui import make_console, tolerate_narrow_encodings

PRODUCT = "Ask Physics"
ROW_SEPARATOR = "::"  # between a command and what it does, in --row and --next


def split_row(row: str) -> tuple[str, str]:
    """``"askphysics ask::ask a question"`` as (command, description)."""
    command, _, description = row.partition(ROW_SEPARATOR)
    return command.strip(), description.strip()


def header(console: Console, title: str, about: str = "", steps: int = 0) -> None:
    """What a script is about to do: its name, one line about it, and how many steps."""
    count = f"{steps} step{'s' if steps != 1 else ''}" if steps else ""
    if is_plain(console):
        console.print(
            Text(f"{PRODUCT}: {title}" + (f" ({count})" if count else "")), soft_wrap=True
        )
        if about:
            console.print(Text(f"{PLAIN_INDENT}{about}"), soft_wrap=True)
        return
    top = Text.assemble(("◉ ", "accent"), (PRODUCT, "brand"), ("  ·  ", "muted"), (title, "value"))
    lines: list[RenderableType] = [top]
    sub = "  ·  ".join(part for part in (about, count) if part)
    if sub:
        lines.append(Padding(Text(sub, style="muted"), (0, 0, 0, 2)))
    console.print()
    console.print(Padding(Group(*lines), (0, 0, 0, INDENT)))
    console.print()


def result(
    console: Console,
    kind: str,
    title: str,
    detail: str = "",
    *,
    index: int | None = None,
    total: int | None = None,
    pad: int = 0,
) -> None:
    """A finished line with no command behind it: ``ok``, ``fail`` or ``skip``."""
    if is_plain(console):
        word = {"ok": "done", "fail": "failed", "skip": "skipped"}[kind]
        label = f"[{index}/{total}] {title}" if index and total else title
        console.print(Text(f"{word}: {label}" + (f" - {detail}" if detail else "")), soft_wrap=True)
        return
    counter = f"{index}/{total}" if index and total else ""
    console.print(
        finished_row(
            title,
            None,
            1 if kind == "fail" else 0,
            width=frame_width(console),
            counter=counter,
            detail=detail,
            pad=pad,
            skipped=kind == "skip",
        )
    )


def _facts_grid(facts: Sequence[tuple[str, str]], value_style: str = "value") -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="label", justify="right", no_wrap=True)
    grid.add_column(overflow="fold")
    for label, value in facts:
        grid.add_row(label, Text(value, style=value_style))
    return grid


def summary(
    console: Console,
    *,
    title: str,
    steps: int = 0,
    passed: int = 0,
    failed: Sequence[str] = (),
    seconds: float | None = None,
    facts: Sequence[tuple[str, str]] = (),
    nexts: Sequence[tuple[str, str]] = (),
) -> None:
    """How a script ended: ok or not, how many steps, how long, and what to run next."""
    ok = not failed
    elapsed = format_elapsed(seconds) if seconds is not None else ""
    if is_plain(console):
        if ok:
            bits = [f"{passed} of {steps} steps"] if steps else []
            bits += [elapsed] if elapsed else []
            summary_line = f"ok: {title}" + (f" ({', '.join(bits)})" if bits else "")
            console.print(Text(summary_line), soft_wrap=True)
        else:
            lead = f"{len(failed)} of {steps} steps failed: " if steps else "failed: "
            tail = f"; {elapsed}" if elapsed else ""
            console.print(
                Text(f"failed: {title} ({lead}{', '.join(failed)}{tail})"), soft_wrap=True
            )
        for label, value in facts:
            console.print(Text(f"{PLAIN_INDENT}{label}: {value}"), soft_wrap=True)
        for command, about in nexts:
            console.print(
                Text(f"{PLAIN_INDENT}next: {command}" + (f"  - {about}" if about else "")),
                soft_wrap=True,
            )
        return
    width = frame_width(console)
    line = Text()
    line.append("✓ " if ok else "✗ ", style="ok" if ok else "bad")
    line.append(title, style="ok" if ok else "bad")
    parts: list[str] = []
    if steps and ok:
        parts.append(
            f"{passed} of {steps} steps passed" if passed != steps else f"all {steps} steps"
        )
    elif steps:
        parts.append(f"{len(failed)} of {steps} failed: {', '.join(failed)}")
    if elapsed:
        parts.append(elapsed)
    if parts:
        line.append("   " + "  ·  ".join(parts), style="muted")
    rows: list[RenderableType] = [Rule(style="muted"), line]
    if facts:
        rows += [Text(""), _facts_grid(facts)]
    if nexts:
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="label", justify="right", no_wrap=True)
        grid.add_column(style="brand", no_wrap=True)
        grid.add_column(style="muted", overflow="fold")
        for i, (command, about) in enumerate(nexts):
            grid.add_row("next" if i == 0 else "", command, about)
        rows += [Text(""), grid]
    console.print(Padding(Constrain(Group(*rows), width - INDENT), (0, 0, 0, INDENT)))
    console.print()


def ready(console: Console, title: str, rows: Sequence[tuple[str, str]], footer: str = "") -> None:
    """The panel a setup ends on: it is ready, and these are the commands to try."""
    if is_plain(console):
        console.print(Text(f"ready: {title}"), soft_wrap=True)
        for command, about in rows:
            console.print(
                Text(f"{PLAIN_INDENT}{command}" + (f"  - {about}" if about else "")), soft_wrap=True
            )
        if footer:
            console.print(Text(f"{PLAIN_INDENT}{footer}"), soft_wrap=True)
        return
    grid = Table.grid(padding=(0, 3))
    grid.add_column(style="brand", no_wrap=True)
    grid.add_column(style="muted", overflow="fold")
    for command, about in rows:
        grid.add_row(command, about)
    body: list[RenderableType] = [grid]
    if footer:
        body += [Text(""), Text(footer, style="muted")]
    panel = Panel(
        Group(*body),
        title=Text.assemble(("✓ ", "ok"), (title, "ok")),
        title_align="left",
        border_style="ok",
        box=box.ROUNDED,
        padding=(1, 3),
    )
    console.print(Padding(Constrain(panel, frame_width(console) - INDENT), (0, 0, 0, INDENT)))
    console.print()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m askphysics.shell_ui", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    head = sub.add_parser("header", help="the opening lines of a script")
    head.add_argument("--title", required=True)
    head.add_argument("--about", default="")
    head.add_argument("--steps", type=int, default=0)

    step = sub.add_parser("step", help="run a command as a named step (the command follows --)")
    step.add_argument("--title", required=True)
    step.add_argument("--inherit", action="store_true", help="the command draws its own display")
    step.add_argument("--index", type=int)
    step.add_argument("--total", type=int)
    step.add_argument("--log", type=Path, help="save the command's full output here")
    step.add_argument("--parse", choices=sorted(PARSERS), help="put a count beside the step")
    step.add_argument("--tail", type=int, default=3, help="lines of output kept under the spinner")
    step.add_argument("--pad", type=int, default=0, help="minimum width of the title column")
    step.add_argument("--ok", default="", help="what to say when it succeeds with no output")

    res = sub.add_parser("result", help="a finished line with no command behind it")
    res.add_argument("kind", choices=["ok", "fail", "skip"])
    res.add_argument("--title", required=True)
    res.add_argument("--detail", default="")
    res.add_argument("--index", type=int)
    res.add_argument("--total", type=int)
    res.add_argument("--pad", type=int, default=0)

    end = sub.add_parser("summary", help="the closing lines of a script")
    end.add_argument("--title", required=True)
    end.add_argument("--steps", type=int, default=0)
    end.add_argument("--passed", type=int, default=0)
    end.add_argument("--failed", action="append", default=[], help="a failed step (repeatable)")
    end.add_argument("--since", type=float, help="epoch seconds the script started")
    end.add_argument("--fact", action="append", default=[], help="label::value (repeatable)")
    end.add_argument("--next", action="append", default=[], help="command::about (repeatable)")

    rdy = sub.add_parser("ready", help="the panel a setup ends on")
    rdy.add_argument("--title", required=True)
    rdy.add_argument("--row", action="append", default=[], help="command::about (repeatable)")
    rdy.add_argument("--footer", default="")

    copy = sub.add_parser("copy", help="copy a directory, checking each file's sha256")
    copy.add_argument("source", type=Path)
    copy.add_argument("dest", type=Path)
    copy.add_argument("--title", required=True)

    models = sub.add_parser("list", help="list the model directories under a folder")
    models.add_argument("root", type=Path)
    models.add_argument("--title", required=True)

    assets = sub.add_parser("assets", help="check a model's release files against the pins")
    assets.add_argument("--model", required=True)
    assets.add_argument("--dir", type=Path, required=True)
    assets.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None, console: Console | None = None) -> int:
    args_in = list(sys.argv[1:] if argv is None else argv)
    command: list[str] = []
    if "--" in args_in:  # everything after the first -- is the command to run
        cut = args_in.index("--")
        args_in, command = args_in[:cut], args_in[cut + 1 :]
    args = build_parser().parse_args(args_in)
    if console is None:
        tolerate_narrow_encodings(sys.stdout, sys.stderr)
        console = make_console()
    if args.command == "header":
        header(console, args.title, args.about, args.steps)
    elif args.command == "step":
        if not command:
            print("step needs a command after --", file=sys.stderr)
            return 2
        return run_phase(
            console,
            args.title,
            command,
            inherit=args.inherit,
            index=args.index,
            total=args.total,
            log=args.log,
            tail=args.tail,
            parse=args.parse,
            pad=args.pad,
            ok_detail=args.ok,
        )
    elif args.command == "result":
        result(
            console,
            args.kind,
            args.title,
            args.detail,
            index=args.index,
            total=args.total,
            pad=args.pad,
        )
    elif args.command == "summary":
        seconds = time.time() - args.since if args.since is not None else None
        summary(
            console,
            title=args.title,
            steps=args.steps,
            passed=args.passed,
            failed=args.failed,
            seconds=seconds,
            facts=[split_row(f) for f in args.fact],
            nexts=[split_row(n) for n in args.next],
        )
    elif args.command == "ready":
        ready(console, args.title, [split_row(r) for r in args.row], args.footer)
    elif args.command == "copy":
        from askphysics.files_view import copy_tree

        return copy_tree(console, args.source, args.dest, args.title)
    elif args.command == "list":
        from askphysics.files_view import list_models

        return list_models(console, args.title, args.root)
    elif args.command == "assets":
        from askphysics.files_view import check_assets
        from askphysics.lm.weights import manifest_path

        return check_assets(console, args.model, args.dir, args.manifest or manifest_path())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
