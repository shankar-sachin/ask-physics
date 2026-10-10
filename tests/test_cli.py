import json
import re
from pathlib import Path
from typing import Any

import pytest
from tests.conftest import DEMO_QUESTION
from typer.testing import CliRunner

from askphysics import cli
from askphysics.errors import ConfigError

runner = CliRunner()


def test_version_prints_the_package_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "0.3.0" in result.output


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


def test_ask_without_models_says_why_a_stand_in_answered() -> None:
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert "No Fermi models are installed" in result.output
    assert "aren't published for this version yet" in _flat(result.output)


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
    assert "Automatic downloads are off (ASKPHYSICS_AUTO_PULL=0)" in _flat(result.output)
    assert "askphysics model" not in result.output  # the hint names no command


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
    assert "download when you ask again" in _flat(result.output)  # the stand-in answered
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


# --- the user command list (ADR-022) ----------------------------------------------------------


def test_help_lists_only_what_a_user_needs() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    listed = result.output.split("Commands")[1]
    commands = [line.split()[1] for line in listed.splitlines() if line.startswith("│ ")]
    assert commands == ["ask", "version"]
    assert not re.search(r"\bmodel\b|train|validate-data|install-models|askphysics-dev", listed)


def test_the_registered_commands_are_ask_version_and_one_hidden_hook() -> None:
    registered = {
        (c.name or getattr(c.callback, "__name__", "").replace("_", "-")): c.hidden
        for c in cli.app.registered_commands
    }
    assert registered == {"ask": False, "version": False, "install-models": True}
    assert not cli.app.registered_groups  # no `model` group for users


@pytest.mark.parametrize("command", ["validate-data", "model"])
def test_the_dev_commands_are_not_on_the_user_command(command: str) -> None:
    result = runner.invoke(cli.app, [command])
    assert result.exit_code != 0
    assert "No such command" in result.output


# --- install-models: the installers' download, hidden from users ------------------------------


def test_install_models_downloads_tellus_and_solem(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1", "fermi-celeste-1"))
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 0, result.output
    assert "fermi-tellus-1 downloaded and verified" in _flat(result.output)
    assert "fermi-solem-1 downloaded and verified" in _flat(result.output)
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")
    assert not (root / "fermi-celeste-1").exists()  # celeste downloads when a question needs it
    assert len(opened) == 6
    again = runner.invoke(cli.app, ["install-models"])
    assert again.exit_code == 0 and "already installed" in _flat(again.output)
    assert len(opened) == 6  # nothing more is fetched


def test_install_models_succeeds_quietly_when_nothing_is_published(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import socket
    import urllib.request

    from askphysics.lm import weights

    root = _models_dir(monkeypatch, tmp_path)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the network was used")

    monkeypatch.setattr(weights, "_open", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    result = runner.invoke(cli.app, ["install-models"])  # the shipped manifest is empty
    assert result.exit_code == 0, result.output
    assert "aren't published for this version yet" in _flat(result.output)
    assert not root.exists()


def test_install_models_installs_only_what_is_missing_and_keeps_local_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    local = root / "fermi-tellus-1"
    local.mkdir(parents=True)
    for file in MODEL_FILES:
        (local / file).write_bytes(b"trained here")
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 0, result.output
    assert all((local / f).read_bytes() == b"trained here" for f in MODEL_FILES)
    assert not any("fermi-tellus-1" in url for url in opened)
    assert _kept(root, "fermi-solem-1")


def test_install_models_ignores_the_auto_pull_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # ASKPHYSICS_AUTO_PULL=0 stops the download on a question; an install asked for it.
    root = _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    monkeypatch.setenv("ASKPHYSICS_AUTO_PULL", "0")
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 0, result.output
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")


def test_install_models_fails_with_a_status_the_installers_can_test(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"), bad="fermi-tellus-1")
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 1
    assert "couldn't download the Fermi models" in _flat(result.output)
    assert "doesn't match the manifest" in _flat(result.output)
    assert [p.name for p in root.iterdir()] == []  # nothing half-installed, no staging left


def test_install_models_fails_when_the_network_is_down(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from askphysics.lm import weights

    root = _models_dir(monkeypatch, tmp_path)
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))

    def offline(url: str) -> None:
        raise OSError("network is unreachable")

    monkeypatch.setattr(weights, "_open", offline)
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 1
    assert "network is unreachable" in _flat(result.output)
    assert not (root / "fermi-tellus-1").exists()


def test_install_models_is_hidden_but_answers_help() -> None:
    result = runner.invoke(cli.app, ["install-models", "--help"])
    assert result.exit_code == 0  # the installers probe for it this way
    assert "Download the published tellus and solem models" in _flat(result.output)


# --- ASKPHYSICS_MODEL_DIR: where Homebrew keeps the models (ADR-022) ---------------------------


def test_ask_and_install_models_share_the_models_directory_the_env_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Homebrew's wrapper sets ASKPHYSICS_MODEL_DIR to <prefix>/var/askphysics/models, which does
    # not exist until post_install runs the installer hook.
    root = tmp_path / "prefix" / "var" / "askphysics" / "models"
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(root))
    opened = _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    installed = runner.invoke(cli.app, ["install-models"])
    assert installed.exit_code == 0, installed.output
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")
    assert len(opened) == 6
    answered = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert answered.exit_code == 0, answered.output
    assert "downloading" not in _flat(answered.output)
    assert len(opened) == 6  # ask found what install-models put there


def test_ask_downloads_into_the_directory_the_env_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # post_install failed, or the models were published later: the first question fills the
    # same directory, and nothing lands under the home directory.
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    root = tmp_path / "brew" / "var" / "askphysics" / "models"
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(root))
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["ask", DEMO_QUESTION])
    assert result.exit_code == 0, result.output
    assert _kept(root, "fermi-tellus-1") and _kept(root, "fermi-solem-1")
    assert not home.exists()


def test_the_models_directory_defaults_to_the_cache_when_the_env_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ASKPHYSICS_MODEL_DIR", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"))
    result = runner.invoke(cli.app, ["install-models"])
    assert result.exit_code == 0, result.output
    assert _kept(tmp_path / ".cache" / "askphysics" / "models", "fermi-tellus-1")


# --- no message a user can see tells them to run a maintainer command -------------------------

# What would send a user to a command that isn't theirs.
FORBIDDEN = ("askphysics model", "askphysics-dev", "model pull", "validate-data", "docs/TRAINING")


def _no_dev_commands(text: str) -> None:
    flat = _flat(text)
    for phrase in FORBIDDEN:
        assert phrase not in flat, f"{phrase!r} in: {flat}"


def test_the_messages_ask_prints_never_name_a_dev_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _models_dir(monkeypatch, tmp_path)
    seen: list[str] = []
    # Nothing published: the stand-in answers and says why.
    seen.append(runner.invoke(cli.app, ["ask", DEMO_QUESTION]).output)
    seen.append(runner.invoke(cli.app, ["ask", "--llm", "fermi", DEMO_QUESTION]).output)
    # Published, but downloads are off, or the download fails.
    _publish(monkeypatch, tmp_path, ("fermi-tellus-1", "fermi-solem-1"), bad="fermi-tellus-1")
    seen.append(runner.invoke(cli.app, ["ask", DEMO_QUESTION]).output)
    seen.append(runner.invoke(cli.app, ["ask", "--llm", "fermi", DEMO_QUESTION]).output)
    seen.append(runner.invoke(cli.app, ["ask", "--model", "fermi-tellus-1", DEMO_QUESTION]).output)
    monkeypatch.setenv("ASKPHYSICS_AUTO_PULL", "0")
    seen.append(runner.invoke(cli.app, ["ask", DEMO_QUESTION]).output)
    seen.append(runner.invoke(cli.app, ["ask", "--llm", "fermi", DEMO_QUESTION]).output)
    seen.append(runner.invoke(cli.app, ["install-models"]).output)
    for output in seen:
        assert output.strip()
        _no_dev_commands(output)


def test_the_help_a_user_can_reach_never_names_a_dev_command() -> None:
    for args in (["--help"], ["ask", "--help"], ["version"], ["install-models", "--help"]):
        _no_dev_commands(runner.invoke(cli.app, args).output)


def test_the_installers_never_tell_a_user_to_run_a_dev_command() -> None:
    root = Path(cli.__file__).resolve().parents[2]
    for name in ("install.sh", "install.ps1"):
        for line in (root / name).read_text(encoding="utf-8").splitlines():
            # Comments, and the fallback for a release before v0.4, which runs a command
            # rather than printing one.
            if not line.lstrip().startswith(("#", "set --")):
                _no_dev_commands(line)


def test_the_user_modules_name_no_dev_command_in_their_messages() -> None:
    src = Path(cli.__file__).resolve().parent
    for relative in ("cli.py", "llm/routing.py", "llm/fermi_client.py", "config.py"):
        text = (src / relative).read_text(encoding="utf-8")
        for phrase in ("askphysics model", "askphysics-dev", "askphysics validate-data"):
            assert phrase not in text, (relative, phrase)
