"""``scripts/release_notes.py``: one version's section of CHANGELOG.md, for a release page."""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import release_notes

ROOT = Path(__file__).resolve().parents[1]

CHANGELOG = """# Changelog

Intro text.

## [Unreleased]

### Added

- Something not shipped yet.

## [0.4.0] - 2026-11-01

### Added

- A new thing.
  Continued on a second line.

### Fixed

- A bug.

## [0.3.0] - 2026-10-06

### Added

- The old thing.
"""


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "release_notes.py"
    return subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True, check=False
    )


def test_extracts_one_section_without_its_heading() -> None:
    notes = release_notes.extract(CHANGELOG, "0.4.0")
    assert notes.startswith("### Added\n\n- A new thing.\n  Continued on a second line.")
    assert notes.endswith("- A bug.")
    assert "0.4.0" not in notes
    assert "old thing" not in notes
    assert "not shipped" not in notes


def test_the_last_section_runs_to_the_end_of_the_file() -> None:
    assert release_notes.extract(CHANGELOG, "0.3.0") == "### Added\n\n- The old thing."


def test_unreleased_can_be_extracted_by_name() -> None:
    assert "not shipped yet" in release_notes.extract(CHANGELOG, "Unreleased")


def test_missing_version_gives_nothing() -> None:
    assert release_notes.extract(CHANGELOG, "9.9.9") == ""
    assert release_notes.extract(CHANGELOG, "0.4") == ""


def test_an_empty_section_gives_nothing() -> None:
    text = "## [1.0.0] - 2027-01-01\n\n\n## [0.9.0] - 2026-12-01\n\n- x\n"
    assert release_notes.extract(text, "1.0.0") == ""


def test_the_real_changelog_has_notes_for_released_versions() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert release_notes.extract(text, "0.3.0").startswith("### Added")


def test_command_line_prints_the_notes_and_accepts_a_v_prefix(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    done = run_script("v0.4.0", str(changelog))
    assert done.returncode == 0
    assert done.stdout.startswith("### Added")
    assert done.stdout.endswith("- A bug.\n")


def test_command_line_fails_for_a_version_with_no_notes(tmp_path: Path) -> None:
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(CHANGELOG, encoding="utf-8")
    done = run_script("9.9.9", str(changelog))
    assert done.returncode == 1
    assert done.stdout == ""
    assert "no notes for 9.9.9" in done.stderr


@pytest.mark.parametrize("argv", [[], ["a", "b", "c"]])
def test_command_line_rejects_the_wrong_number_of_arguments(argv: list[str]) -> None:
    assert release_notes.main(argv) == 2
