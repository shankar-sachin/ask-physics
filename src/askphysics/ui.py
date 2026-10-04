"""Terminal presentation for the CLI: answer cards, pretty math and units, training progress.

Everything here only formats; nothing computes. Numbers shown are the ones
the symbolic algebra machine produced.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping

from rich import box
from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.padding import Padding
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from askphysics.errors import AskPhysicsError
from askphysics.models import Answer, Equation
from askphysics.solver.symbolic import parse_equation
from askphysics.solver.units import ureg

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
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def make_console(**kwargs: object) -> Console:
    return Console(theme=THEME, highlight=False, **kwargs)  # type: ignore[arg-type]


def pretty_unit(unit: str) -> str:
    """``meter / second ** 2`` -> ``m/s²``; unknown strings pass through unchanged."""
    try:
        return format(ureg.Unit(unit), "~P")
    except Exception:  # Pint raises a zoo of error types for odd strings; show it as written
        return unit


_SUBSCRIPT = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def pretty_symbol(symbol: str) -> str:
    """``v0`` -> ``v₀``, ``m12`` -> ``m₁₂``: trailing digits become subscripts, as in SymPy."""
    stem = symbol.rstrip("0123456789")
    return stem + symbol[len(stem) :].translate(_SUBSCRIPT) if stem else symbol


def pretty_number(value: float, sig: int = 6) -> str:
    """Plain digits for everyday magnitudes, ``4.374 × 10⁵`` style otherwise."""
    if value == 0 or 1e-3 <= abs(value) < 1e6:
        return f"{value:.{sig}g}"
    exponent = math.floor(math.log10(abs(value)))
    mantissa = value / 10**exponent
    return f"{mantissa:.{max(sig - 2, 1)}g} × 10{str(exponent).translate(_SUPERSCRIPT)}"


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POWER = re.compile(r"\*\*(\d+)")


def pretty_equation(sympy_expr: str) -> str:
    """One-line math in the author's term order: ``v**2 = v0**2 + 2*a*d`` -> ``v² = v₀² + 2·a·d``.

    Unparseable strings are returned unchanged.
    """
    try:
        parse_equation(sympy_expr)
    except AskPhysicsError:
        return sympy_expr
    text = _POWER.sub(lambda m: m.group(1).translate(_SUPERSCRIPT), sympy_expr)
    text = _IDENTIFIER.sub(lambda m: pretty_symbol(m.group()), text)
    return text.replace("*", "·")


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

    if answer.assumptions:
        rows.append(("assumes", Text("\n".join(f"• {a}" for a in answer.assumptions))))
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


def training_progress(console: Console) -> Progress:
    """Live progress for ``askphysics model train``."""
    return Progress(
        TextColumn("[brand]{task.description}"),
        BarColumn(bar_width=32, complete_style="accent", finished_style="ok"),
        MofNCompleteColumn(),
        TextColumn("[label]loss[/] [value]{task.fields[loss]}"),
        TextColumn("[label]val[/] [accent]{task.fields[val]}"),
        TextColumn("[muted]{task.fields[speed]}"),
        TimeElapsedColumn(),
        TextColumn("[muted]eta"),
        TimeRemainingColumn(),
        console=console,
    )


def safe(text: str) -> str:
    """Escape Rich markup in user or model text."""
    return escape(text)
