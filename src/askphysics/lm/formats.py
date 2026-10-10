"""Canonical task formats for the Fermi models (``docs/PROMPTS.md``).

The data factory writes training examples with these functions and the
constrained decoder produces outputs in exactly the same layout, so the
models see one consistent format. Any change here means bumping
``FORMAT_VERSION`` and retraining.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any

from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, Constant, Equation, FermiAssumption, Plan
from askphysics.solver.units import check_dimensions, is_valid_unit, quantity

# The one number a standard plan may assume without it being stated: "from rest" means v0 = 0.
# The data factory never assumes anything else. A 1 here let the model invent m = 1 kg (#85).
FILLER_NUMBERS = ("0",)
# Numbers only the prose of a plan (assumptions and strategy) may use: the 0 and 1 in
# "v = 0" or "1/2 m v^2". A structural number is never a known value.
STRUCTURAL_NUMBERS = ("0", "1")

# Not preceded by identifier characters or an exponent: the 2 in m/s^2, the 1 in m^-1, or
# second ** 2 is no number.
_NUMBER = re.compile(
    r"(?<![A-Za-z0-9_.^*])(?<!\*\* )(?<!\^-)(?<!\*\* -)-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
)
# A unit starts with a letter; "^-" allows negative exponents ("s^-1", "m^-1").
# The number is atomic, so "7.5e+19" is never split into 7.5 and the unit "e".
_UNIT_AFTER_NUMBER = re.compile(
    "(?>"
    + _NUMBER.pattern
    + r")\s*([A-Za-z](?:[A-Za-z0-9/*]|\^-?)*(?:\((?:[A-Za-z0-9/*]|\^-?)+\))?)"
)


def dumps(value: Any) -> str:
    """JSON with fixed separators, so every format renders byte-identically."""
    return json.dumps(value, ensure_ascii=True, separators=(", ", ": "))


def format_number(value: float) -> str:
    """Canonical spelling of a number: ``20.0`` -> ``20``, ``9.80665`` -> ``9.80665``."""
    if not math.isfinite(value):
        raise ValueError(f"cannot format non-finite number {value}")
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(float(value))


def extract_numbers(text: str) -> list[str]:
    """Numbers written in ``text``, canonically formatted, in order of appearance."""
    out: list[str] = []
    for match in _NUMBER.finditer(text):
        if not math.isfinite(float(match.group())):
            continue  # "1e999" overflows; it can't be copied into a plan anyway
        formatted = format_number(float(match.group()))
        if formatted not in out:
            out.append(formatted)
    return out


def extract_units(text: str) -> list[str]:
    """Unit strings written right after a number in ``text`` that Pint understands."""
    out: list[str] = []
    for match in _UNIT_AFTER_NUMBER.finditer(text):
        unit = match.group(1)
        if unit not in out and is_valid_unit(unit):
            out.append(unit)
    return out


# "from 0 to 20 m/s", "from 5 km/h up to 9 km/h": the unit after the second number is also
# the first one's when the first has none. The first number may carry a unit of its own.
_UNIT = r"[A-Za-z](?:[A-Za-z0-9/*]|\^-?)*(?:\((?:[A-Za-z0-9/*]|\^-?)+\))?"
_PLAIN_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_RANGE = re.compile(
    r"\bfrom\s+(?P<a>(?>" + _PLAIN_NUMBER + r"))(?:\s*(?P<ua>" + _UNIT + r"))?"
    r"\s+(?:up\s+)?to\s+(?P<b>(?>" + _PLAIN_NUMBER + r"))\s*(?P<ub>" + _UNIT + r")",
    re.IGNORECASE,
)


def _spelled(number: str, unit: str | None) -> tuple[str, str] | None:
    """A (canonical number, unit) pair, or None if the number overflows or the unit is not one."""
    if unit is None or not is_valid_unit(unit) or not math.isfinite(float(number)):
        return None
    return format_number(float(number)), unit


def quantity_ranges(text: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Each "from A to B unit" in ``text``: ((A, unit), (B, unit)), in order of appearance.

    "from 0 to 20 m/s" gives (("0", "m/s"), ("20", "m/s")): the first number takes the
    second's unit when it has none of its own. A start and an end are all this says; which
    variables they fill is for the reading module ("from A to B" is the initial and then the
    final value).
    """
    out: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for m in _RANGE.finditer(text):
        end = _spelled(m.group("b"), m.group("ub"))
        if end is None:
            continue
        start = _spelled(m.group("a"), m.group("ua")) or _spelled(m.group("a"), end[1])
        if start is not None:
            out.append((start, end))
    return out


def _unit_quantities(text: str) -> list[tuple[int, str, str]]:
    """(position, canonical number, unit as written) for every number with a unit."""
    found: dict[int, tuple[str, str]] = {}
    for match in _UNIT_AFTER_NUMBER.finditer(text):
        unit = match.group(1)
        number = match.group()[: match.start(1) - match.start()].strip()
        if is_valid_unit(unit) and math.isfinite(float(number)):
            found[match.start()] = (format_number(float(number)), unit)
    for m in _RANGE.finditer(text):
        # A bare first number ("from 0 to 20 m/s") is a quantity in the second's unit.
        if m.start("a") not in found and _spelled(m.group("a"), m.group("ua")) is None:
            end = _spelled(m.group("b"), m.group("ub"))
            if end is not None:
                found[m.start("a")] = (format_number(float(m.group("a"))), end[1])
    return [(pos, n, u) for pos, (n, u) in sorted(found.items())]


def question_quantities(text: str) -> list[tuple[str, str]]:
    """Every number written with a unit in ``text``: (canonical number, unit as written).

    In order of appearance, repeats kept, so "5 kg and 5 kg" counts two masses. The first
    number of "from 0 to 20 m/s" takes the unit of the second.
    """
    return [(n, u) for _, n, u in _unit_quantities(text)]


def stated_quantities(text: str) -> list[tuple[str, str]]:
    """``question_quantities`` plus every bare number, as a dimensionless quantity.

    A friction coefficient or an efficiency is written without a unit ("a coefficient of
    0.3"), so a number with no unit after it can fill a dimensionless variable. Numbers
    that carry a unit are never offered as dimensionless.
    """
    with_unit = {pos for pos, _, _ in _unit_quantities(text)}
    bare = [
        (format_number(float(m.group())), "dimensionless")
        for m in _NUMBER.finditer(text)
        if m.start() not in with_unit and math.isfinite(float(m.group()))
    ]
    return question_quantities(text) + bare


def _unique(items: Iterable[str]) -> list[str]:
    seen: list[str] = []
    for item in items:
        if item not in seen:
            seen.append(item)
    return seen


# --------------------------------------------------------------------------- classify


def classify_prompt(question: str) -> str:
    return CLASSIFY + dumps({"question": question})


def serialize_classification(c: Classification) -> str:
    """Canonical output text, ending with the end token."""
    body = {
        "category": c.category,
        "domains": list(c.domains),
        "reasoning": c.reasoning,
        "closest_answerable": c.closest_answerable,
    }
    return dumps(body) + END


# --------------------------------------------------------------------------- plan


def relevant_constants(
    equations: Sequence[Equation], constants: Sequence[Constant]
) -> list[Constant]:
    """Constants whose dimensions match some variable of ``equations``.

    Keeps plan inputs short: standard gravity stays for kinematics (it can fill
    an acceleration), the gas constant only shows up next to the ideal gas law.
    """
    units = {v.unit for eq in equations for v in eq.variables}
    return [c for c in constants if any(check_dimensions(quantity(1.0, c.unit), u) for u in units)]


def plan_prompt(
    question: str,
    category: str,
    equations: Sequence[Equation],
    constants: Sequence[Constant],
    fermi: Sequence[FermiAssumption] = (),
) -> str:
    """Plan input. Fermi assumptions are only included for Fermi questions (context budget)."""
    payload: dict[str, Any] = {
        "question": question,
        "category": category,
        "equations": [
            {
                "id": eq.id,
                "sympy_expr": eq.sympy_expr,
                "variables": [{"symbol": v.symbol, "unit": v.unit} for v in eq.variables],
            }
            for eq in equations
        ],
        "constants": [
            {"name": c.name, "symbol": c.symbol, "value": format_number(c.value), "unit": c.unit}
            for c in constants
        ],
    }
    if category == "fermi":
        payload["fermi_assumptions"] = [
            {"quantity": a.quantity, "value": format_number(a.default_value), "unit": a.unit}
            for a in fermi
        ]
    return PLAN + dumps(payload)


def value_numbers(
    question: str, constants: Sequence[Constant], fermi: Sequence[FermiAssumption] = ()
) -> list[str]:
    """Every number a known value of a plan may be: from the question, the tables, or the filler."""
    return _unique(
        [
            *extract_numbers(question),
            *(format_number(c.value) for c in constants),
            *(format_number(a.default_value) for a in fermi),
            *FILLER_NUMBERS,
        ]
    )


def plan_numbers(
    question: str, constants: Sequence[Constant], fermi: Sequence[FermiAssumption] = ()
) -> list[str]:
    """Every number a plan may write: its known values, and the structural numbers in its prose."""
    return _unique([*value_numbers(question, constants, fermi), *STRUCTURAL_NUMBERS])


def plan_units(
    question: str,
    equations: Sequence[Equation],
    constants: Sequence[Constant],
    fermi: Sequence[FermiAssumption] = (),
) -> list[str]:
    """Every unit a plan may use: written in the question, declared by equations, or in tables."""
    return _unique(
        [
            *extract_units(question),
            *(v.unit for eq in equations for v in eq.variables),
            *(c.unit for c in constants),
            *(a.unit for a in fermi),
        ]
    )


def serialize_plan(p: Plan) -> str:
    """Canonical output text: equations first, so the model commits to physics before details."""
    body = {
        "equation_ids": list(p.equation_ids),
        "target": p.target,
        "unknowns": list(p.unknowns),
        "known_values": [
            {
                "symbol": k.symbol,
                "value": _RawNumber(format_number(k.value)),
                "unit": k.unit,
                "origin": k.origin,
            }
            for k in p.known_values
        ],
        "assumptions": list(p.assumptions),
        "strategy": p.strategy,
    }
    return _dumps_raw(body) + END


class _RawNumber(str):
    """Marks a pre-formatted number so it is written without quotes."""


def _dumps_raw(value: Any) -> str:
    if isinstance(value, _RawNumber):
        return str(value)
    if isinstance(value, dict):
        return "{" + ", ".join(f"{dumps(k)}: {_dumps_raw(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_dumps_raw(v) for v in value) + "]"
    return dumps(value)


# --------------------------------------------------------------------------- explain


def explain_prompt(
    question: str,
    value: float,
    unit: str,
    equations: Sequence[Equation],
    assumptions: Sequence[str],
    issues: Sequence[str] = (),
) -> str:
    payload = {
        "question": question,
        "result": {"value": _RawNumber(format_number(value)), "unit": unit},
        "equations": [{"id": eq.id, "name": eq.name} for eq in equations],
        "assumptions": list(assumptions),
        "issues": list(issues),
    }
    return EXPLAIN + _dumps_raw(payload)


def explain_numbers(question: str, value: float, assumptions: Sequence[str]) -> list[str]:
    """Numbers an explanation may mention: the result and anything in the question or plan."""
    return _unique(
        [
            format_number(value),
            *extract_numbers(question),
            *(n for a in assumptions for n in extract_numbers(a)),
        ]
    )
