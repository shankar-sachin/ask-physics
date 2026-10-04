"""Pretty math, numbers, and units, shared by the CLI and the website.

Pure formatting with no terminal dependency, so it also runs in the browser
under Pyodide. Everything here only formats; nothing computes.
"""

from __future__ import annotations

import math
import re

from askphysics.errors import AskPhysicsError
from askphysics.solver.symbolic import parse_equation
from askphysics.solver.units import ureg

_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def pretty_unit(unit: str) -> str:
    """``meter / second ** 2`` -> ``m/s²``; unknown strings pass through unchanged."""
    try:
        # Pint 0.26 switched its pretty dot from U+00B7 to U+22C5; pin the old one.
        return format(ureg.Unit(unit), "~P").replace("\u22c5", "·")
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
