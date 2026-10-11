"""``askphysics-dev``: the maintainer commands, and the guard that keeps them off user installs."""

import functools
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from askphysics import devcli
from askphysics.data.loader import (
    CONSTANTS_FILE,
    EQUATIONS_FILE,
    EXAMPLES_FILE,
    FERMI_FILE,
    default_data_dir,
    load_all,
)
from askphysics.errors import DataValidationError

runner = CliRunner()


def test_validate_data_passes() -> None:
    result = runner.invoke(devcli.app, ["validate-data"])
    assert result.exit_code == 0
    assert "all valid" in result.output


def test_validate_data_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_: object) -> None:
        raise DataValidationError(["equation x: invalid unit 'blorps'"])

    monkeypatch.setattr(devcli, "load_all", broken)
    result = runner.invoke(devcli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "blorps" in result.output


def _copy_data_with_example_edit(tmp_path: Path, edit: Callable[[Any], None]) -> Path:
    for name in (EQUATIONS_FILE, EXAMPLES_FILE, CONSTANTS_FILE, FERMI_FILE):
        shutil.copy(default_data_dir() / name, tmp_path / name)
    examples = tmp_path / EXAMPLES_FILE
    data = json.loads(examples.read_text())
    edit(data)
    examples.write_text(json.dumps(data))
    return tmp_path


def _validate_data_from(monkeypatch: pytest.MonkeyPatch, data_dir: Path) -> None:
    """Make ``validate-data`` read the given copy of the seed data."""
    monkeypatch.setattr(devcli, "load_all", functools.partial(load_all, data_dir))


def test_validate_data_fails_a_worked_example_with_the_wrong_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def scale_answer(d: list[dict[str, Any]]) -> None:
        d[0]["final_answer"]["value"] *= 1000  # ex_kin_001: 15 m/s becomes 15000 m/s

    _validate_data_from(monkeypatch, _copy_data_with_example_edit(tmp_path, scale_answer))
    result = runner.invoke(devcli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "data validation failed" in result.output
    assert "re-solve gives" in result.output


def test_validate_data_fails_a_worked_example_with_the_wrong_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong_unit(d: list[dict[str, Any]]) -> None:
        d[0]["final_answer"]["unit"] = "s"  # ex_kin_001 asks for a speed, not a time

    _validate_data_from(monkeypatch, _copy_data_with_example_edit(tmp_path, wrong_unit))
    result = runner.invoke(devcli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "does not match" in result.output


def test_pull_with_nothing_published_says_so() -> None:
    result = runner.invoke(devcli.app, ["model", "pull"])
    assert result.exit_code == 1
    assert "no trained weights are published" in result.output
    quiet = runner.invoke(devcli.app, ["model", "pull", "--if-published"])
    assert quiet.exit_code == 0 and "no trained weights are published" in quiet.output


def test_pull_downloads_and_verifies(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import hashlib
    import io

    from askphysics.lm import weights

    files = {"model.safetensors": b"w" * 10, "config.json": b"{}", "tokenizer.json": b"{}"}
    pinned = weights.PinnedModel("fermi-tellus-1", "r", tuple(
        weights.PinnedFile(n, f"https://example.test/{n}", len(b), hashlib.sha256(b).hexdigest())
        for n, b in files.items()
    ))  # fmt: skip
    monkeypatch.setattr(weights, "read_manifest", lambda path=None: {pinned.name: pinned})
    monkeypatch.setattr(weights, "_open", lambda url: io.BytesIO(files[url.rsplit("/", 1)[1]]))
    result = runner.invoke(devcli.app, ["model", "pull", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fermi-tellus-1 downloaded and verified" in result.output
    assert (tmp_path / "fermi-tellus-1" / "model.safetensors").read_bytes() == files[
        "model.safetensors"
    ]


def test_train_refuses_prose_steps_that_leave_no_task_steps(tmp_path: Path) -> None:
    prose = tmp_path / "prose.jsonl"
    prose.write_text('{"source": "s", "title": "t", "text": "Some prose."}\n')
    result = runner.invoke(
        devcli.app,
        ["model", "train", "--model", "fermi-luna-1", "--steps", "3000", "--prose", str(prose),
         "--prose-steps", "3000", "--tokenizer", str(tmp_path / "t.json")],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "never train on the tasks" in result.output and "--steps 6000" in result.output


# --- the dev-install guard (ADR-022) ----------------------------------------------------------


def test_this_checkout_counts_as_a_dev_install() -> None:
    root = devcli.source_root()
    assert root is not None and (root / "pyproject.toml").is_file()
    assert devcli.is_dev_install()


def test_the_dev_commands_run_inside_a_dev_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devcli, "is_dev_install", lambda: True)
    result = runner.invoke(devcli.app, ["validate-data"])
    assert result.exit_code == 0, result.output
    assert "all valid" in result.output


def test_the_dev_commands_refuse_outside_a_dev_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devcli, "is_dev_install", lambda: False)
    for args in (["validate-data"], ["model", "info"], ["model", "pull", "--if-published"]):
        result = runner.invoke(devcli.app, args)
        assert result.exit_code == 1, args
        assert "source checkout" in result.output
        assert "all valid" not in result.output


def test_the_console_script_refuses_before_it_shows_anything(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(devcli, "is_dev_install", lambda: False)
    monkeypatch.setattr("sys.argv", ["askphysics-dev", "--help"])
    with pytest.raises(SystemExit) as stop:
        devcli.main()
    assert stop.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""  # not even the help text
    assert captured.err.strip() == devcli.REFUSAL
    assert len(devcli.REFUSAL) < 200  # short and clear


def test_the_console_script_runs_inside_a_dev_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(devcli, "is_dev_install", lambda: True)
    monkeypatch.setattr("sys.argv", ["askphysics-dev", "--help"])
    with pytest.raises(SystemExit) as stop:
        devcli.main()
    assert stop.value.code == 0
    assert "validate-data" in capsys.readouterr().out


def _fake_tree(tmp_path: Path, *, src: bool, name: str = "askphysics") -> Path:
    """A tree with ``askphysics/devcli.py`` where a package would be installed."""
    root = tmp_path / "tree"
    package = (root / "src" if src else root / "lib" / "site-packages") / "askphysics"
    package.mkdir(parents=True)
    (package / "devcli.py").write_text("")
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\n')
    return package / "devcli.py"


@pytest.mark.parametrize(
    ("src", "name", "expected"),
    [
        (True, "askphysics", True),  # <repo>/src/askphysics beside the repo's pyproject.toml
        (False, "askphysics", False),  # site-packages: an ordinary install
        (True, "something-else", False),  # a src/ layout, but not this project
    ],
)
def test_the_detection_looks_for_the_repo_beside_src(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, src: bool, name: str, expected: bool
) -> None:
    fake = _fake_tree(tmp_path, src=src, name=name)
    monkeypatch.setattr(devcli, "__file__", str(fake))
    assert devcli.is_dev_install() is expected


def test_the_detection_needs_a_readable_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake = _fake_tree(tmp_path, src=True)
    (tmp_path / "tree" / "pyproject.toml").unlink()
    monkeypatch.setattr(devcli, "__file__", str(fake))
    assert not devcli.is_dev_install()
    (tmp_path / "tree" / "pyproject.toml").write_text("not = [toml")
    assert not devcli.is_dev_install()


def test_the_dev_script_is_declared_beside_the_user_one() -> None:
    import tomllib

    root = devcli.source_root()
    assert root is not None
    scripts = tomllib.loads((root / "pyproject.toml").read_text())["project"]["scripts"]
    assert scripts == {
        "askphysics": "askphysics.cli:app",
        "askphysics-dev": "askphysics.devcli:main",
    }
