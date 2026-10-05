"""Pint wrapper: one shared unit registry plus parsing, conversion, and dimension checks.

Every quantity in the package must come from ``ureg``. Quantities from
different Pint registries cannot be combined, so never create another
``UnitRegistry``.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pint

from askphysics.errors import UnitMismatchError, UnitParseError

ureg: pint.UnitRegistry[Any] = pint.UnitRegistry()
Quantity = pint.Quantity[Any]  # annotation alias; use pint.Quantity for isinstance checks


@lru_cache(maxsize=4096)
def _parse_unit(unit: str) -> pint.Unit:
    if not unit.strip():
        raise UnitParseError("empty unit; use 'dimensionless' for pure numbers")
    try:
        parsed = ureg.parse_units(unit)
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError) as exc:
        raise UnitParseError(f"unknown unit {unit!r}") from exc
    except Exception as exc:
        # Pint's expression parser fails in many ways on text that isn't a unit: it asserts
        # on dangling operators ("N/"), raises KeyError on zero powers ("K^0"), and more.
        # Anything it can't parse is simply not a unit.
        raise UnitParseError(f"cannot parse unit {unit!r}") from exc
    return parsed


@lru_cache(maxsize=4096)
def is_valid_unit(unit: str) -> bool:
    """Return True if ``unit`` parses in the shared registry."""
    try:
        _parse_unit(unit)
    except UnitParseError:
        return False
    return True


def quantity(value: float, unit: str) -> Quantity:
    """Build a quantity from a number and a unit string, rejecting unknown units."""
    return ureg.Quantity(value, _parse_unit(unit))


def parse_quantity(text: str) -> Quantity:
    """Parse a string like ``"20 m"`` or ``"9.8 m/s^2"`` into a quantity.

    Raises:
        UnitParseError: the text is not a number followed by a valid unit.
    """
    try:
        parsed: Any = ureg.Quantity(text)
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError) as exc:
        raise UnitParseError(f"cannot parse quantity {text!r}") from exc
    except (AttributeError, TypeError, ValueError, SyntaxError, AssertionError) as exc:
        # Pint's expression parser asserts on dangling operators ("N/", "m*").
        raise UnitParseError(f"cannot parse quantity {text!r}") from exc
    if not isinstance(parsed, pint.Quantity) or parsed.unitless:
        raise UnitParseError(f"{text!r} is a bare number; quantities need units")
    return parsed


def convert(q: Quantity, unit: str) -> Quantity:
    """Convert ``q`` to ``unit``.

    Raises:
        UnitParseError: ``unit`` is not valid.
        UnitMismatchError: the dimensions are incompatible.
    """
    target = _parse_unit(unit)
    try:
        return q.to(target)
    except pint.errors.DimensionalityError as exc:
        raise UnitMismatchError(f"cannot convert {q.units} to {target}: {exc}") from exc


def check_dimensions(q: Quantity, expected_unit: str) -> bool:
    """Return True if ``q`` has the same dimensionality as ``expected_unit``."""
    return bool(q.dimensionality == ureg.Quantity(1, _parse_unit(expected_unit)).dimensionality)


def unit_string(units: Any) -> str:
    """Pint's spelling of ``units``, without a bare leading 1.

    Pint writes a reciprocal unit as "1 / meter". Model-written text may only contain
    numbers from its input, so that 1 can't be copied into an explanation; "meter ** -1"
    is the same unit with nothing but an exponent.
    """
    text = str(units)
    if not text.startswith("1 / "):
        return text
    factors = []
    for name, power in sorted(dict(units._units).items(), key=lambda item: -item[1]):
        exponent = int(power) if float(power).is_integer() else power
        factors.append(name if exponent == 1 else f"{name} ** {exponent}")
    return " * ".join(factors)


def format_quantity(q: Quantity, sig_figs: int = 6) -> str:
    """Format a quantity with a fixed number of significant figures."""
    return f"{q.magnitude:.{sig_figs}g} {q.units}"
