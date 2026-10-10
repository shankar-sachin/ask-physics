import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from askphysics import cli
from askphysics.errors import ConfigError
from askphysics.lm import backend
from askphysics.lm.backend import resolve_backend


def _plain(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


def test_torch_and_unknown_backends() -> None:
    assert resolve_backend("torch", None) == "torch"
    assert resolve_backend("torch", "mps") == "torch"
    with pytest.raises(ConfigError, match="unknown backend"):
        resolve_backend("jax", None)


def test_auto_is_torch_off_a_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "on_apple_silicon", lambda: False)
    monkeypatch.setattr(backend, "mlx_available", lambda: True)
    assert resolve_backend("auto", None) == "torch"


def test_auto_is_mlx_on_an_apple_silicon_mac_with_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "on_apple_silicon", lambda: True)
    monkeypatch.setattr(backend, "mlx_available", lambda: True)
    assert resolve_backend("auto", None) == "mlx"
    assert resolve_backend("auto", "mps") == "mlx"


def test_auto_stays_on_torch_for_cpu_and_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "on_apple_silicon", lambda: True)
    monkeypatch.setattr(backend, "mlx_available", lambda: True)
    assert resolve_backend("auto", "cpu") == "torch"
    assert resolve_backend("auto", "cuda") == "torch"


def test_auto_falls_back_to_torch_without_mlx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "on_apple_silicon", lambda: True)
    monkeypatch.setattr(backend, "mlx_available", lambda: False)
    assert resolve_backend("auto", None) == "torch"


def test_explicit_mlx_without_mlx_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "mlx_available", lambda: False)
    with pytest.raises(ConfigError, match="not installed"):
        resolve_backend("mlx", None)


def test_cli_refuses_mlx_when_it_is_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "mlx_available", lambda: False)
    r = CliRunner().invoke(cli.app, ["model", "train", "--backend", "mlx"])
    assert r.exit_code == 1 and "not installed" in _plain(r.output)


def test_cli_backend_command_prints_the_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend, "on_apple_silicon", lambda: False)
    r = CliRunner().invoke(cli.app, ["model", "backend", "--backend", "auto", "--device", "cpu"])
    assert r.exit_code == 0 and r.output.strip() == "torch"
    monkeypatch.setattr(backend, "mlx_available", lambda: False)
    r = CliRunner().invoke(cli.app, ["model", "backend", "--backend", "mlx"])
    assert r.exit_code == 1


def test_mlx_memory_limit_needs_mlx() -> None:
    args = ["model", "train", "--backend", "torch", "--mlx-memory-gb", "8"]
    r = CliRunner().invoke(cli.app, args)
    assert r.exit_code == 1 and "mlx backend" in _plain(r.output)


def _small_data(root: Path) -> tuple[Path, Path]:
    from askphysics.lm.factory import build_dataset
    from askphysics.lm.train import train_tokenizer

    data = root / "data"
    build_dataset(data, 120, seed=0)
    tok = train_tokenizer(data, vocab_size=512)
    tok_path = root / "tok.json"
    tok.save(tok_path)
    return data, tok_path


def test_cli_trains_with_the_torch_backend(tmp_path: Path) -> None:
    data, tok = _small_data(tmp_path)
    out = tmp_path / "models" / "fermi-luna-1"
    r = CliRunner().invoke(
        cli.app,
        ["model", "train", "--data", str(data), "--tokenizer", str(tok), "--out", str(out),
         "--steps", "2", "--batch-size", "4", "--device", "cpu", "--backend", "torch"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "torch" in _plain(r.output)
    assert (out / "model.safetensors").exists()


def test_cli_trains_with_the_mlx_backend(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    data, tok = _small_data(tmp_path)
    out = tmp_path / "models" / "fermi-luna-1"
    r = CliRunner().invoke(
        cli.app,
        ["model", "train", "--data", str(data), "--tokenizer", str(tok), "--out", str(out),
         "--steps", "2", "--batch-size", "4", "--device", "cpu", "--backend", "mlx"],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert "mlx" in _plain(r.output)
    assert (out / "model.safetensors").exists() and (out / "metrics.jsonl").exists()


def test_cli_reports_a_refused_resume_without_a_traceback(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    data, tok = _small_data(tmp_path)
    out = tmp_path / "models" / "fermi-luna-1"
    out.mkdir(parents=True)
    # A torch run's optimizer state: no "backend" field, so it counts as torch's.
    (out / "train_state.json").write_text('{"step": 5, "lr": 0.001, "param_groups": []}')
    r = CliRunner().invoke(
        cli.app,
        ["model", "train", "--model", "fermi-luna-1", "--data", str(data), "--tokenizer", str(tok),
         "--out", str(out), "--steps", "2", "--batch-size", "4", "--device", "cpu",
         "--backend", "mlx", "--resume"],
    )  # fmt: skip
    assert r.exit_code == 1 and isinstance(r.exception, SystemExit)
    assert "was trained with the torch backend" in _plain(r.output)
