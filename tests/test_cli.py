import json

import pytest
from tests.conftest import DEMO_QUESTION
from typer.testing import CliRunner

from askphysics import cli
from askphysics.errors import DataValidationError

runner = CliRunner()


def test_version_prints_0_1_0() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


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
    assert "Confidence" in result.output


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
    assert "REFUSED" in result.output


def test_bad_config_exits_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ASKPHYSICS_TOP_K", "lots")
    result = runner.invoke(cli.app, ["ask", "anything"])
    assert result.exit_code == 1
    assert "invalid value" in result.output


def test_unknown_provider_exits_cleanly() -> None:
    result = runner.invoke(cli.app, ["ask", "--llm", "skynet", DEMO_QUESTION])
    assert result.exit_code == 1
    assert "unknown llm_provider" in result.output
