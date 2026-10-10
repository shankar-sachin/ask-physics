import json
from pathlib import Path

import pytest
from tests.conftest import DEMO_QUESTION
from typer.testing import CliRunner

from askphysics import cli
from askphysics.errors import DataValidationError

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
    def broken() -> None:
        raise DataValidationError(["equation x: invalid unit 'blorps'"])

    monkeypatch.setattr(cli, "load_all", broken)
    result = runner.invoke(cli.app, ["validate-data"])
    assert result.exit_code == 1
    assert "blorps" in result.output


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
