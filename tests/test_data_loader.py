import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from askphysics.data.loader import (
    CONSTANTS_FILE,
    EQUATIONS_FILE,
    EXAMPLE_TOLERANCE,
    EXAMPLES_FILE,
    FERMI_FILE,
    DataStore,
    default_data_dir,
    load_all,
    validate_equation,
    validate_store,
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
        "equations": 111,
        "examples": 20,
        "constants": 11,
        "fermi_assumptions": 12,
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
        load_all(data_dir, solve_examples=True)
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


def _example(entries: list[dict[str, Any]], example_id: str) -> dict[str, Any]:
    return next(e for e in entries if e["id"] == example_id)


def test_every_worked_example_re_solves(store: DataStore) -> None:
    """Noether re-solves each example to within 0.1% of its stated answer (v0.4 exit criterion).

    The check runs in validate_store, so validate-data enforces it as well as pytest.
    """
    assert len(store.examples) >= 15
    assert validate_store(store, solve_examples=True) == []


def test_a_constant_alias_must_name_a_real_constant(data_dir: Path) -> None:
    def rename_constant(d: list[dict[str, Any]]) -> None:
        next(c for c in d if c["symbol"] == "k_B")["symbol"] = "kb"

    _edit(data_dir / CONSTANTS_FILE, rename_constant)
    assert any("'k_B': no constant has that symbol" in p for p in _problems(data_dir))


def test_a_constant_alias_must_name_a_real_variable(data_dir: Path) -> None:
    def rename_variable(d: list[dict[str, Any]]) -> None:
        for eq in d:
            for v in eq["variables"]:
                if v["symbol"] == "kB":
                    v["symbol"] = "kb"
            eq["sympy_expr"] = eq["sympy_expr"].replace("kB", "kb")

    _edit(data_dir / EQUATIONS_FILE, rename_variable)
    assert any("'k_B': no equation has variable 'kB'" in p for p in _problems(data_dir))


def test_a_wrong_example_value_fails_the_re_solve(data_dir: Path) -> None:
    def scale_answer(d: list[dict[str, Any]]) -> None:
        _example(d, "ex_kin_001")["final_answer"]["value"] *= 1000

    _edit(data_dir / EXAMPLES_FILE, scale_answer)
    assert any("ex_kin_001: re-solve gives" in p for p in _problems(data_dir))


def test_a_wrong_example_dimension_fails(data_dir: Path) -> None:
    _edit(
        data_dir / EXAMPLES_FILE,
        lambda d: _example(d, "ex_kin_001")["final_answer"].update(unit="s"),
    )
    assert any("ex_kin_001: final_answer unit 's' does not match" in p for p in _problems(data_dir))


def test_a_re_solve_within_tolerance_passes(data_dir: Path) -> None:
    def nudge_answer(d: list[dict[str, Any]]) -> None:
        _example(d, "ex_kin_001")["final_answer"]["value"] *= 1 + EXAMPLE_TOLERANCE / 2

    _edit(data_dir / EXAMPLES_FILE, nudge_answer)
    load_all(data_dir, solve_examples=True)  # raises DataValidationError on a failed re-solve


def test_worked_examples_do_not_read_like_eval_questions(store: DataStore) -> None:
    from askphysics.lm.factory import LEAK_THRESHOLD, load_blocklist, word_overlap

    evals = load_blocklist(Path(__file__).resolve().parents[1] / "evals" / "questions.yaml")
    assert evals
    for ex in store.examples.values():
        for question in evals:
            assert word_overlap(ex.problem_text, question) < LEAK_THRESHOLD, (ex.id, question)
