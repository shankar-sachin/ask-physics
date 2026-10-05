import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from askphysics.data.loader import (
    CONSTANTS_FILE,
    EQUATIONS_FILE,
    EXAMPLES_FILE,
    FERMI_FILE,
    DataStore,
    default_data_dir,
    load_all,
    validate_equation,
)
from askphysics.errors import DataValidationError


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    for name in (EQUATIONS_FILE, EXAMPLES_FILE, CONSTANTS_FILE, FERMI_FILE):
        shutil.copy(default_data_dir() / name, tmp_path / name)
    return tmp_path


def _edit(path: Path, fn: Any) -> None:
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))


def test_every_seed_file_validates(store: DataStore) -> None:
    assert store.summary() == {
        "equations": 110,
        "examples": 4,
        "constants": 11,
        "fermi_assumptions": 8,
    }


def test_every_equation_passes_semantic_checks(store: DataStore) -> None:
    for eq in store.equations.values():
        assert validate_equation(eq) == [], eq.id


def test_every_entry_has_source_and_license(store: DataStore) -> None:
    for eq in store.equations.values():
        assert eq.source and eq.license
    for ex in store.examples.values():
        assert ex.source and ex.license


def test_constants_include_the_required_set(store: DataStore) -> None:
    symbols = {c.symbol for c in store.constants.values()}
    assert {"g", "G", "c", "k_B", "R", "e", "h", "atm"} <= symbols


def _problems(data_dir: Path) -> list[str]:
    with pytest.raises(DataValidationError) as info:
        load_all(data_dir)
    return info.value.problems


def test_bad_unit_is_reported(data_dir: Path) -> None:
    _edit(data_dir / EQUATIONS_FILE, lambda d: d[0]["variables"][0].update(unit="blorps"))
    assert any("invalid unit 'blorps'" in p for p in _problems(data_dir))


def test_missing_variable_is_reported(data_dir: Path) -> None:
    _edit(data_dir / EQUATIONS_FILE, lambda d: d[0]["variables"].pop())
    assert any("symbols missing from variables" in p for p in _problems(data_dir))


def test_unparseable_expression_is_reported(data_dir: Path) -> None:
    _edit(data_dir / EQUATIONS_FILE, lambda d: d[0].update(sympy_expr="v = __import__"))
    assert any("disallowed characters" in p for p in _problems(data_dir))


def test_disallowed_license_is_reported(data_dir: Path) -> None:
    _edit(data_dir / EQUATIONS_FILE, lambda d: d[0].update(license="CC-BY-NC-4.0"))
    assert any("not allowed" in p for p in _problems(data_dir))


def test_dangling_equation_reference_is_reported(data_dir: Path) -> None:
    _edit(data_dir / EXAMPLES_FILE, lambda d: d[0].update(equations_used=["nope"]))
    assert any("unknown equation ids" in p for p in _problems(data_dir))


def test_duplicate_id_is_reported(data_dir: Path) -> None:
    _edit(data_dir / CONSTANTS_FILE, lambda d: d.append(dict(d[0])))
    assert any("duplicate id" in p for p in _problems(data_dir))


def test_schema_errors_are_reported_with_location(data_dir: Path) -> None:
    _edit(data_dir / FERMI_FILE, lambda d: d[0].pop("rationale"))
    assert any("fermi_assumptions.json[0] rationale" in p for p in _problems(data_dir))


def test_all_problems_collected_at_once(data_dir: Path) -> None:
    _edit(data_dir / EQUATIONS_FILE, lambda d: d[0].update(license="proprietary"))
    _edit(data_dir / CONSTANTS_FILE, lambda d: d[0].update(unit="blorps"))
    assert len(_problems(data_dir)) >= 2


def test_broken_and_missing_files(data_dir: Path) -> None:
    (data_dir / EXAMPLES_FILE).write_text("{not json")
    (data_dir / CONSTANTS_FILE).unlink()
    (data_dir / FERMI_FILE).write_text("{}")
    problems = _problems(data_dir)
    assert any("invalid JSON" in p for p in problems)
    assert any("file not found" in p for p in problems)
    assert any("must be a JSON array" in p for p in problems)
