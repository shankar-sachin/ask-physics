import functools
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import DEMO_QUESTION
from typer.testing import CliRunner

from askphysics import cli
from askphysics.data.loader import (
    CONSTANTS_FILE,
    EQUATIONS_FILE,
    EXAMPLES_FILE,
    FERMI_FILE,
    default_data_dir,
    load_all,
)
from askphysics.errors import ConfigError, DataValidationError

runner = CliRunner()


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "0.3.0" in result.output


def test_validate_data_passes() -> None:
    result = runner.invoke(cli.app, ["validate-data"])
    assert result.exit_code == 0
    assert "all valid" in result.output


def test_validate_data_reports_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(**_: object) -> None:
        raise DataValidationError(["equation x: invalid unit 'blorps'"])

    monkeypatch.setattr(cli, "load_all", broken)
    result = runner.invoke(cli.app, ["validate-data"])
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
    monkeypatch.setattr(cli, "load_all", functools.partial(load_all, data_dir))


def test_validate_data_fails_a_worked_example_with_the_wrong_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def scale_answer(d: list[dict[str, Any]]) -> None:
        d[0]["final_answer"]["value"] *= 1000  # ex_kin_001: 15 m/s becomes 15000 m/s

    _validate_data_from(monkeypatch, _copy_data_with_example_edit(tmp_path, scale_answer))
    result = runner.invoke(cli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "data validation failed" in result.output
    assert "re-solve gives" in result.output


def test_validate_data_fails_a_worked_example_with_the_wrong_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def wrong_unit(d: list[dict[str, Any]]) -> None:
        d[0]["final_answer"]["unit"] = "s"  # ex_kin_001 asks for a speed, not a time

    _validate_data_from(monkeypatch, _copy_data_with_example_edit(tmp_path, wrong_unit))
    result = runner.invoke(cli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "does not match" in result.output


def test_ask_renders_answer() -> None:
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0
    assert "19.8057" in result.output
    assert "[kin_v_squared]" in result.output  # citations must survive Rich markup
    assert "confidence" in result.output
    assert "m/s" in result.output  # compact units, not "meter / second"
    assert "ANSWERED" in result.output


def test_ask_json_is_a_structured_answer() -> None:
    result = runner.invoke(cli.app, ["ask", "--json", DEMO_QUESTION])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["final_value"] == pytest.approx(19.8057, rel=1e-5)
    assert data["unit"] == "meter / second"
    assert data["confidence"]["label"] == "high"


def test_ask_refusal_renders() -> None:
    result = runner.invoke(cli.app, ["ask", "How much does the color blue weigh?"])
    assert result.exit_code == 0
    assert "CAN'T ANSWER" in result.output
    assert "try instead" in result.output


@pytest.mark.parametrize("question", ["", "   "])
def test_ask_a_blank_question_gives_a_reason_not_a_traceback(question: str) -> None:
    result = runner.invoke(cli.app, ["ask", question])
    assert result.exit_code == 0
    assert "ValidationError" not in result.output
    assert "Traceback" not in result.output
    assert "CAN'T ANSWER" in result.output
    assert "question is empty" in result.output


def test_ask_json_for_a_blank_question_is_a_refused_answer() -> None:
    result = runner.invoke(cli.app, ["ask", "--json", " "])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["status"] == "refused"
    assert data["final_value"] is None


def test_bad_config_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKPHYSICS_TOP_K", "lots")
    result = runner.invoke(cli.app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "invalid value" in result.output


def test_unknown_provider_exits_cleanly() -> None:
    result = runner.invoke(cli.app, ["ask", "--llm", "skynet", DEMO_QUESTION])
    assert result.exit_code == 1
    assert "unknown llm_provider" in result.output


def test_ask_without_models_names_the_fix() -> None:
    result = runner.invoke(cli.app, ["ask", "--llm", "fermi", DEMO_QUESTION])
    assert result.exit_code == 1
    assert "no Fermi models are installed" in result.output
    result = runner.invoke(cli.app, ["ask", "--model", "gpt-5", DEMO_QUESTION])
    assert result.exit_code == 1
    assert "unknown model" in result.output


def test_ask_card_shows_the_models() -> None:
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert "fake classified · fake planned · fake explained" in result.output


def test_ask_takes_an_unquoted_question() -> None:
    words = DEMO_QUESTION.split()
    result = runner.invoke(cli.app, ["ask", *words])
    assert result.exit_code == 0, result.output
    assert "19.8057" in result.output


def test_pull_with_nothing_published_says_so() -> None:
    result = runner.invoke(cli.app, ["model", "pull"])
    assert result.exit_code == 1
    assert "no trained weights are published" in result.output
    quiet = runner.invoke(cli.app, ["model", "pull", "--if-published"])
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
    result = runner.invoke(cli.app, ["model", "pull", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "fermi-tellus-1 downloaded and verified" in result.output
    assert (tmp_path / "fermi-tellus-1" / "model.safetensors").read_bytes() == files[
        "model.safetensors"
    ]


def test_ask_without_models_says_how_to_get_them() -> None:
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert "No Fermi models are installed" in result.output


def test_train_refuses_prose_steps_that_leave_no_task_steps(tmp_path: Path) -> None:
    prose = tmp_path / "prose.jsonl"
    prose.write_text('{"source": "s", "title": "t", "text": "Some prose."}\n')
    result = runner.invoke(
        cli.app,
        ["model", "train", "--model", "fermi-luna-1", "--steps", "3000", "--prose", str(prose),
         "--prose-steps", "3000", "--tokenizer", str(tmp_path / "t.json")],
    )  # fmt: skip
    assert result.exit_code == 1
    assert "never train on the tasks" in result.output and "--steps 6000" in result.output


# --- ask downloads published models it is missing (ADR-012) ---------------------------------

MODEL_FILES = {"model.safetensors": b"w" * 64, "config.json": b"{}", "tokenizer.json": b"{}"}


def _publish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, names: tuple[str, ...], bad: str | None = None
) -> list[str]:
    """A temporary manifest pinning ``names`` to local files (file:// URLs); no network.

    ``bad`` names a model whose served file differs from what the manifest pins. Returns
    the list that records every URL opened.
    """
    import hashlib

    from askphysics.lm import weights

    served = tmp_path / "served"
    models: dict[str, Any] = {}
    for name in names:
        (served / name).mkdir(parents=True)
        entries = {}
        for file, data in MODEL_FILES.items():
            (served / name / file).write_bytes(data if name != bad else b"x" * len(data))
            entries[file] = {
                "url": (served / name / file).as_uri(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        models[name] = {"release": "models-test", "files": entries}
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": models}), encoding="utf-8")
    monkeypatch.setattr(weights, "manifest_path", lambda: manifest)

    def no_weights(*args: object, **kwargs: object) -> None:
        raise ConfigError("test weights are placeholder bytes")  # no real model loads (rule 6)

    monkeypatch.setattr("askphysics.llm.fermi_client.load_model", no_weights)
    opened: list[str] = []
    real_open = weights._open
    monkeypatch.setattr(weights, "_open", lambda url: (opened.append(url), real_open(url))[1])
    return opened


def _models_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    root = tmp_path / "models"
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(root))
    return root


def _flat(text: str) -> str:
    """Output with the terminal's line wrapping undone."""
    return " ".join(text.split())


def _kept(root: Path, name: str) -> bool:
    return all((root / name / f).read_bytes() == b for f, b in MODEL_FILES.items())


def test_ask_downloads_the_published_models_on_the_first_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert "downloading the Fermi models (tellus, solem" in _flat(result.output)
    assert "won't happen again" in _flat(result.output)
    assert "fermi-tellus-1 downloaded and verified" in _flat(result.output)
    assert "fermi-solem-1 downloaded and verified" in _flat(result.output)
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")
    assert len(opened) == 6
    again = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert again.exit_code == 0 and "downloading" not in _flat(again.output)
    assert len(opened) == 6  # installed now: nothing more is fetched


def test_ask_downloads_only_the_missing_published_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1",))
    runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert _kept(root, "fermi-tellus-1") and not (root / "fermi-solem-1").exists()
    assert all("fermi-tellus-1" in url for url in opened)


def test_ask_does_not_download_when_auto_pull_is_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    monkeypatch.setenv("ASKPHYSICS_AUTO_PULL", "0")
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert opened == [] and not root.exists()
    assert "askphysics model pull" in _flat(result.output)  # the old hint still says how


def test_ask_with_the_fake_provider_never_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["ask", "--llm", "fake", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert opened == []


def test_ask_touches_no_network_when_nothing_is_published(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import socket
    import urllib.request

    from askphysics.lm import weights

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network was used")

    monkeypatch.setattr(weights, "_open", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])  # the shipped manifest is empty
    assert result.exit_code == 0, result.output
    assert "downloading" not in result.output


def test_a_checksum_mismatch_degrades_and_installs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    opened = _publish(
        monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"), bad="fermi-tellus-1"
    )
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert "couldn't download the Fermi models" in _flat(result.output)
    assert "doesn't match the manifest" in _flat(result.output)
    assert "next question will try again" in _flat(result.output)
    assert "askphysics model pull" in _flat(result.output)  # answered by the stand-in, as before
    assert not (root / "fermi-tellus-1").exists()
    assert [p.name for p in root.iterdir()] == []  # no staging leftovers either
    first = len(opened)
    runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert len(opened) > first  # the next ask tries again


def test_a_failed_download_degrades_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from askphysics.lm import weights

    root = _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))

    def offline(url: str) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(weights, "_open", offline)
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert "network is unreachable" in _flat(result.output)
    assert "19.8057" in _flat(result.output)
    assert not (root / "fermi-tellus-1").exists()


def test_a_local_model_is_not_overwritten_by_ask(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    local = root / "fermi-tellus-1"
    local.mkdir(parents=True)
    for file in MODEL_FILES:
        (local / file).write_bytes(b"trained here")
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert all((local / f).read_bytes() == b"trained here" for f in MODEL_FILES)
    assert not any("fermi-tellus-1" in url for url in opened)
    assert _kept(root, "fermi-solem-1")  # the missing one still downloads


def test_ask_json_stays_clean_json_while_it_downloads(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["ask", "--json", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)  # nothing but the answer on stdout
    assert "downloading the Fermi models" in result.stderr  # the display went to stderr
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")


def test_ask_json_stays_clean_when_the_download_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1",), bad="fermi-tellus-1")
    result = runner.invoke(cli.app, ["ask", "--json", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    json.loads(result.stdout)
    assert "couldn't download" in result.stderr


def test_ask_on_a_terminal_shows_the_pull_display(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import io

    from askphysics.ui import make_console

    _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    buffer = io.StringIO()
    monkeypatch.setattr(cli, "console", make_console(file=buffer, force_terminal=True, width=100))
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert "downloading the Fermi models" in _flat(buffer.getvalue())
    assert "downloaded and verified" in _flat(buffer.getvalue())
    assert "downloading" not in result.stdout  # the card prints on the terminal console
