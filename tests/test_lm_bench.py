import pytest
import torch
from typer.testing import CliRunner

from askphysics import cli
from askphysics.lm.bench import available_settings, bench, format_hours, projected_hours
from askphysics.lm.config import LUNA
from askphysics.lm.train import autocast_dtype


def test_autocast_dtype_mapping() -> None:
    cpu, mps, cuda = (torch.device(d) for d in ("cpu", "mps", "cuda"))
    assert autocast_dtype(cpu, "auto") is None
    assert autocast_dtype(mps, "auto") is torch.bfloat16
    assert autocast_dtype(cuda, "auto") is torch.bfloat16
    assert autocast_dtype(mps, "fp32") is None
    assert autocast_dtype(cpu, "bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_dtype(cpu, "fp16")


def test_bench_times_a_cpu_step_and_survives_a_bad_device() -> None:
    good, bad = bench(
        LUNA,
        batch_size=2,
        width=16,
        steps=1,
        warmup=0,
        settings=[("cpu", "fp32"), ("nosuchdevice", "fp32")],
    )
    assert good.error is None and good.tokens_per_s > 0 and good.seconds_per_step > 0
    assert bad.error and bad.tokens_per_s == 0


def test_available_settings_always_has_cpu() -> None:
    assert ("cpu", "fp32") in available_settings()


def test_projected_hours() -> None:
    assert projected_hours(10.0, 100, 10, 36) == pytest.approx(1.0)
    assert format_hours(1.5) == "1:30"


def test_cli_bench_names_the_fastest() -> None:
    args = ["model", "bench", "--model", "fermi-luna-1", "--batch-size", "2", "--width", "16"]
    r = CliRunner().invoke(cli.app, [*args, "--steps", "1", "--warmup", "0", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert "fastest: cpu fp32" in r.output
    assert "--device cpu --precision fp32" in r.output
    bad = CliRunner().invoke(cli.app, [*args[:-1], "4", "--steps", "1"])
    assert bad.exit_code != 0
