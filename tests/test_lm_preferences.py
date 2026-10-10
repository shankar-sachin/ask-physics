"""The saved celeste choice: ``preferences.json`` in the models directory (ADR-022)."""

import json
from pathlib import Path

import pytest

from askphysics.lm.preferences import (
    PREFERENCES_FILE,
    preferences_path,
    read_celeste_choice,
    save_celeste_choice,
)


def test_nothing_is_saved_at_first(tmp_path: Path) -> None:
    assert read_celeste_choice(tmp_path) is None


@pytest.mark.parametrize("choice", ["always", "never", "ask"])
def test_a_saved_choice_reads_back(tmp_path: Path, choice: str) -> None:
    path = save_celeste_choice(choice, tmp_path)  # type: ignore[arg-type]
    assert path == tmp_path / PREFERENCES_FILE
    assert read_celeste_choice(tmp_path) == choice


def test_the_file_lives_in_the_models_directory_the_env_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(tmp_path / "var" / "models"))
    assert preferences_path() == tmp_path / "var" / "models" / PREFERENCES_FILE
    save_celeste_choice("always")  # creates the directory
    assert read_celeste_choice() == "always"


def test_saving_keeps_other_keys_and_deleting_the_file_resets(tmp_path: Path) -> None:
    (tmp_path / PREFERENCES_FILE).write_text(json.dumps({"other": 1}))
    save_celeste_choice("never", tmp_path)
    assert json.loads((tmp_path / PREFERENCES_FILE).read_text()) == {
        "other": 1,
        "celeste_download": "never",
    }
    (tmp_path / PREFERENCES_FILE).unlink()  # the documented reset
    assert read_celeste_choice(tmp_path) is None


@pytest.mark.parametrize("text", ["not json", "[]", '{"celeste_download": "sometimes"}', ""])
def test_a_damaged_file_means_no_choice(tmp_path: Path, text: str) -> None:
    (tmp_path / PREFERENCES_FILE).write_text(text)
    assert read_celeste_choice(tmp_path) is None
    save_celeste_choice("always", tmp_path)  # and it can be written over
    assert read_celeste_choice(tmp_path) == "always"
