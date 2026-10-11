import pytest
import torch
from typer.testing import CliRunner

from askphysics import devcli
from askphysics.lm.bench import (
    available_settings,
    bench,
    bench_widths,
    format_hours,
    projected_hours,
)
from askphysics.lm.config import LUNA
from askphysics.lm.model import FermiLM
from askphysics.lm.train import autocast_dtype, numpy_batch


def test_autocast_dtype_mapping() -> None:
    cpu, mps, cuda = (torch.device(d) for d in ("cpu", "mps", "cuda"))
    assert autocast_dtype(cpu, "auto") is None
    assert autocast_dtype(mps, "auto") is torch.bfloat16
    assert autocast_dtype(cuda, "auto") is torch.bfloat16
    assert autocast_dtype(mps, "fp32") is None
    assert autocast_dtype(cpu, "bf16") is torch.bfloat16
    with pytest.raises(ValueError, match="unknown precision"):
        autocast_dtype(cpu, "fp16")


def test_bench_widths_are_the_training_buckets_up_to_the_largest() -> None:
    assert bench_widths(1024) == [64, 96, 128, 192, 256, 384, 512, 768, 1024]
    assert bench_widths(512) == [64, 96, 128, 192, 256, 384, 512]
    assert bench_widths(1000) == [64, 96, 128, 192, 256, 384, 512, 768, 1000]
    assert bench_widths(64) == [64]
    assert bench_widths(16) == [16]


@pytest.mark.parametrize("largest", [1024, 1000, 512])
def test_bench_widths_are_exactly_the_widths_training_produces(largest: int) -> None:
    # Every batch length up to the context gets a width from numpy_batch; the bench must
    # time each of them and nothing else.
    produced = set()
    for n in range(1, largest + 1):
        inputs, _ = numpy_batch([([0] * (n + 1), 1)], 0, multiple=64, max_width=largest)
        produced.add(inputs.shape[1])
    assert produced == set(bench_widths(largest))


def test_bench_runs_an_evaluation_pass_without_gradients(monkeypatch: pytest.MonkeyPatch) -> None:
    from askphysics.lm import bench as bench_module

    seen: list[tuple[bool, bool]] = []

    class Recording(FermiLM):
        def forward(
            self, ids: torch.Tensor, targets: torch.Tensor | None = None
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            seen.append((self.training, torch.is_grad_enabled()))
            return super().forward(ids, targets)

    monkeypatch.setattr(bench_module, "FermiLM", Recording)
    bench(LUNA, batch_size=2, width=64, steps=1, warmup=0, settings=[("cpu", "fp32")])
    # One timed training step with gradients, then the evaluation pass without them.
    assert seen == [(True, True), (False, False)]


def test_bench_times_a_cpu_step_and_survives_a_bad_device() -> None:
    good, bad = bench(
        LUNA,
        batch_size=2,
        width=128,
        steps=1,
        warmup=0,
        settings=[("cpu", "fp32"), ("nosuchdevice", "fp32")],
    )
    assert good.error is None and good.tokens_per_s > 0 and good.seconds_per_step > 0
    assert good.rss_gb > 0
    assert good.gpu_mem_gb is None
    assert bad.error and bad.tokens_per_s == 0


def test_available_settings_always_has_cpu() -> None:
    assert ("cpu", "fp32") in available_settings()


def test_projected_hours() -> None:
    assert projected_hours(10.0, 100, 10, 36) == pytest.approx(1.0)
    assert format_hours(1.5) == "1:30"


def test_cli_bench_names_the_fastest() -> None:
    args = ["model", "bench", "--model", "fermi-luna-1", "--batch-size", "2", "--width", "128"]
    r = CliRunner().invoke(devcli.app, [*args, "--steps", "1", "--warmup", "0", "--device", "cpu"])
    assert r.exit_code == 0, r.output
    assert "fastest: cpu fp32" in r.output
    assert "--device cpu --precision fp32" in r.output
    assert "GPU memory" in r.output


def test_cli_bench_defaults_to_the_model_context_and_its_buckets() -> None:
    args = ["model", "bench", "--model", "fermi-luna-1", "--batch-size", "1", "--device", "cpu"]
    r = CliRunner().invoke(devcli.app, [*args, "--steps", "1", "--warmup", "0"])
    assert r.exit_code == 0, r.output
    assert "64, 96, 128, 192, 256, 384, 512, 768, 1024" in " ".join(r.output.split())
    bad = CliRunner().invoke(devcli.app, [*args[:-1], "4", "--steps", "1"])
    assert bad.exit_code != 0
