"""Load and validate all seed data.

Validation has two layers (``docs/DATA_SCHEMA.md``):

1. Structural: every entry goes through its pydantic model.
2. Semantic: SymPy parse, symbol coverage, Pint unit validity, dimensional
   consistency, referential integrity, uniqueness, allowed licenses.

All problems are collected and raised together in one ``DataValidationError``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from askphysics.errors import DataValidationError, SolverError
from askphysics.models import Constant, Equation, FermiAssumption, WorkedExample
from askphysics.solver.fermi import AssumptionTable
from askphysics.solver.symbolic import check_dimensional_consistency, free_symbol_names
from askphysics.solver.units import is_valid_unit

ALLOWED_LICENSES = frozenset({"MIT", "CC0-1.0", "CC-BY-4.0", "public-domain"})

EQUATIONS_FILE = "equations.json"
EXAMPLES_FILE = "examples.json"
CONSTANTS_FILE = "constants.json"
FERMI_FILE = "fermi_assumptions.json"

M = TypeVar("M", bound=BaseModel)


@dataclass(frozen=True)
class DataStore:
    """All validated seed data, keyed for lookup."""

    equations: dict[str, Equation]
    examples: dict[str, WorkedExample]
    constants: dict[str, Constant]
    fermi: AssumptionTable

    def summary(self) -> dict[str, int]:
        return {
            "equations": len(self.equations),
            "examples": len(self.examples),
            "constants": len(self.constants),
            "fermi_assumptions": len(self.fermi),
        }


def default_data_dir() -> Path:
    """Directory holding the packaged JSON files."""
    return Path(str(resources.files("askphysics.data")))


def _read_json(path: Path, problems: list[str]) -> list[Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        problems.append(f"{path.name}: file not found")
        return []
    except json.JSONDecodeError as exc:
        problems.append(f"{path.name}: invalid JSON ({exc})")
        return []
    if not isinstance(raw, list):
        problems.append(f"{path.name}: top level must be a JSON array")
        return []
    return raw


def _parse_entries(
    path: Path, model: type[M], key: Callable[[M], str], problems: list[str]
) -> dict[str, M]:
    entries: dict[str, M] = {}
    for i, raw in enumerate(_read_json(path, problems)):
        try:
            entry = model.model_validate(raw)
        except ValidationError as exc:
            for err in exc.errors():
                loc = ".".join(str(p) for p in err["loc"])
                problems.append(f"{path.name}[{i}] {loc}: {err['msg']}")
            continue
        k = key(entry)
        if k in entries:
            problems.append(f"{path.name}[{i}]: duplicate id {k!r}")
            continue
        entries[k] = entry
    return entries


def _check_units(label: str, units: Sequence[str], problems: list[str]) -> None:
    problems.extend(f"{label}: invalid unit {u!r}" for u in units if not is_valid_unit(u))


def validate_equation(eq: Equation) -> list[str]:
    """Semantic checks for one equation. Returns a list of problems (empty if valid)."""
    problems: list[str] = []
    label = f"equation {eq.id}"
    if eq.license not in ALLOWED_LICENSES:
        problems.append(f"{label}: license {eq.license!r} is not allowed")
    units = [v.unit for v in eq.variables]
    _check_units(label, units, problems)
    try:
        symbols = free_symbol_names(eq.sympy_expr)
    except SolverError as exc:
        problems.append(f"{label}: {exc}")
        return problems
    declared = {v.symbol for v in eq.variables}
    if missing := symbols - declared:
        problems.append(f"{label}: symbols missing from variables: {sorted(missing)}")
    if extra := declared - symbols:
        problems.append(f"{label}: variables not in expression: {sorted(extra)}")
    if not problems and not check_dimensional_consistency(eq):
        problems.append(f"{label}: dimensionally inconsistent with declared units")
    return problems


def validate_store(store: DataStore) -> list[str]:
    """Cross-entry semantic checks. Returns a list of problems (empty if valid)."""
    problems: list[str] = []
    for eq in store.equations.values():
        problems.extend(validate_equation(eq))

    for ex in store.examples.values():
        label = f"example {ex.id}"
        if ex.license not in ALLOWED_LICENSES:
            problems.append(f"{label}: license {ex.license!r} is not allowed")
        unknown_ids = [eid for eid in ex.equations_used if eid not in store.equations]
        if unknown_ids:
            problems.append(f"{label}: unknown equation ids {unknown_ids}")
            continue
        symbols: set[str] = set()
        for eid in ex.equations_used:
            symbols |= {v.symbol for v in store.equations[eid].variables}
        used = {k.symbol for k in ex.known_values} | set(ex.unknowns)
        if stray := used - symbols:
            problems.append(f"{label}: symbols not in referenced equations: {sorted(stray)}")
        _check_units(label, [k.unit for k in ex.known_values], problems)
        _check_units(label, [ex.final_answer.unit], problems)

    for c in store.constants.values():
        _check_units(f"constant {c.name}", [c.unit], problems)
    for a in store.fermi:
        _check_units(f"fermi assumption {a.quantity}", [a.unit], problems)
    return problems


def load_all(data_dir: Path | None = None, *, validate: bool = True) -> DataStore:
    """Load every seed data file into a ``DataStore``.

    Raises:
        DataValidationError: any structural or (if ``validate``) semantic problem.
    """
    base = data_dir or default_data_dir()
    problems: list[str] = []
    store = DataStore(
        equations=_parse_entries(base / EQUATIONS_FILE, Equation, lambda e: e.id, problems),
        examples=_parse_entries(base / EXAMPLES_FILE, WorkedExample, lambda e: e.id, problems),
        constants=_parse_entries(base / CONSTANTS_FILE, Constant, lambda c: c.name, problems),
        fermi=AssumptionTable(
            _parse_entries(base / FERMI_FILE, FermiAssumption, lambda a: a.quantity, problems)
        ),
    )
    if validate:
        problems.extend(validate_store(store))
    if problems:
        raise DataValidationError(problems)
    return store
