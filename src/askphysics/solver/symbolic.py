"""SymPy wrapper: safe equation parsing, dimensional consistency, and ``solve_for``.

Expressions only ever come from reviewed data (``equations.json``), never
from LLM output (see "Security boundary" in ``docs/ARCHITECTURE.md``). They
are still parsed defensively: a character whitelist, no builtins, and a
small function namespace, because ``parse_expr`` uses ``eval`` internally.

Symbols are unit-free inside SymPy. Units are attached by evaluating the
solved expression with Pint quantities substituted in, so Pint catches
dimension errors as part of the arithmetic (ADR-002).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pint
import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    parse_expr,
    standard_transformations,
)

from askphysics.errors import SolverError, UnitMismatchError
from askphysics.models import Equation
from askphysics.solver.units import Quantity, quantity

_ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9_+\-*/^(). =]+$")
_TRANSFORMATIONS = (*standard_transformations, convert_xor)
_FUNCTIONS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "pi": sp.pi,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
}
# Everything parse_expr's auto-symbol transformation needs, and nothing else.
_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    "Symbol": sp.Symbol,
    "Integer": sp.Integer,
    "Float": sp.Float,
    "Rational": sp.Rational,
    **_FUNCTIONS,
}
# Pint quantities support ** but not math.sqrt, so map sqrt to a power.
_PINT_MODULE: dict[str, Callable[[Any], Any]] = {"sqrt": lambda x: x**0.5}


def parse_side(text: str) -> sp.Expr:
    """Parse one side of an equation with the restricted parser."""
    try:
        expr = parse_expr(
            text,
            local_dict={},
            global_dict=dict(_GLOBALS),
            transformations=_TRANSFORMATIONS,
        )
    except (SyntaxError, TypeError, ValueError, NameError, AttributeError) as exc:
        raise SolverError(f"cannot parse expression {text!r}: {exc}") from exc
    if not isinstance(expr, sp.Expr):
        raise SolverError(f"{text!r} did not parse to an expression")
    return expr


def parse_equation(sympy_expr: str) -> sp.Eq:
    """Parse ``"lhs = rhs"`` into a SymPy equation.

    Raises:
        SolverError: disallowed characters, not exactly one ``=``, or a parse failure.
    """
    if not _ALLOWED_CHARS.match(sympy_expr) or "__" in sympy_expr:
        raise SolverError(f"disallowed characters in {sympy_expr!r}")
    if sympy_expr.count("=") != 1:
        raise SolverError(f"expected exactly one '=' in {sympy_expr!r}")
    lhs, rhs = sympy_expr.split("=")
    return sp.Eq(parse_side(lhs), parse_side(rhs), evaluate=False)


def free_symbol_names(sympy_expr: str) -> set[str]:
    """Names of all free symbols in an equation string."""
    return {str(s) for s in parse_equation(sympy_expr).free_symbols}


def _evaluate(expr: sp.Expr, values: Mapping[str, Quantity]) -> Any:
    symbols = sorted(expr.free_symbols, key=str)
    fn = sp.lambdify(symbols, expr, modules=[_PINT_MODULE, "math"])
    return fn(*(values[str(s)] for s in symbols))


def check_dimensional_consistency(equation: Equation) -> bool:
    """True if both sides of the equation have the same dimensions given the declared units.

    Substitutes ``1 * unit`` for every variable and evaluates both sides with
    Pint. Adding incompatible quantities raises inside Pint, which counts as
    inconsistent.
    """
    eq = parse_equation(equation.sympy_expr)
    ones = {v.symbol: quantity(1.0, v.unit) for v in equation.variables}
    try:
        lhs = _evaluate(eq.lhs, ones)
        rhs = _evaluate(eq.rhs, ones)
    except pint.errors.DimensionalityError:
        return False
    lhs_q = lhs if isinstance(lhs, pint.Quantity) else quantity(float(lhs), "dimensionless")
    rhs_q = rhs if isinstance(rhs, pint.Quantity) else quantity(float(rhs), "dimensionless")
    return bool(lhs_q.dimensionality == rhs_q.dimensionality)


@dataclass(frozen=True)
class SolveOutcome:
    """Result of ``solve_for``: the chosen root as a quantity, plus how it was chosen."""

    value: Quantity
    symbolic_solution: str
    notes: list[str] = field(default_factory=list)


def solve_for(equation: Equation, unknown: str, knowns: Mapping[str, Quantity]) -> SolveOutcome:
    """Solve a single algebraic equation for ``unknown`` and evaluate it with units.

    Roots are solved symbolically, evaluated with Pint quantities, filtered
    to real values, and the smallest non-negative root is preferred (see
    ``docs/OPEN_QUESTIONS.md`` Q7). The result is converted to the unknown's
    declared unit.

    Raises:
        SolverError: missing knowns, no closed-form solution, or no real root.
        UnitMismatchError: the result's dimensions don't match the declared unit.
    """
    eq = parse_equation(equation.sympy_expr)
    names = {str(s) for s in eq.free_symbols}
    if unknown not in names:
        raise SolverError(f"{unknown!r} does not appear in {equation.id}")
    missing = names - {unknown} - set(knowns)
    if missing:
        raise SolverError(f"missing known values for {sorted(missing)} in {equation.id}")

    solutions = sp.solve(eq, sp.Symbol(unknown))
    if not solutions:
        raise SolverError(f"no closed-form solution for {unknown!r} in {equation.id}")

    candidates: list[tuple[Quantity, sp.Expr]] = []
    for sol in solutions:
        try:
            val = _evaluate(sol, knowns)
        except pint.errors.DimensionalityError as exc:
            raise UnitMismatchError(f"inconsistent units solving {equation.id}: {exc}") from exc
        q = val if isinstance(val, pint.Quantity) else quantity(float(val), "dimensionless")
        if isinstance(q.magnitude, complex):
            continue
        candidates.append((q, sol))
    if not candidates:
        raise SolverError(f"no real solution for {unknown!r} in {equation.id}")

    candidates.sort(key=lambda c: (c[0].magnitude < 0, abs(c[0].magnitude)))
    chosen, chosen_expr = candidates[0]
    target_unit = equation.variable(unknown).unit
    try:
        chosen = chosen.to(target_unit)
    except pint.errors.DimensionalityError as exc:
        raise UnitMismatchError(
            f"{unknown} came out in {chosen.units}, expected {target_unit}"
        ) from exc

    notes = [
        f"Discarded root {q.to(target_unit).magnitude:.6g} {target_unit}" for q, _ in candidates[1:]
    ]
    return SolveOutcome(value=chosen, symbolic_solution=str(chosen_expr), notes=notes)
