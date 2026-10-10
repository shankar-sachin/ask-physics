import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from askphysics import cli
from askphysics.lm import train as train_module
from askphysics.lm.checkpoints import load_model
from askphysics.lm.config import LUNA, ModelConfig
from askphysics.lm.factory import Example, build_dataset, read_examples
from askphysics.lm.generate import encode_task
from askphysics.lm.model import IGNORE_INDEX, FermiLM
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.train import (
    STATE_FILE,
    TokenizedSet,
    TrainConfig,
    _load_optimizer,
    _optimizer,
    _optimizer_step,
    _peak_memory_gb,
    _save_optimizer,
    _WidthFlush,
    bucket_width,
    lr_at,
    make_batch,
    sample_examples_by_task,
    tokenize_examples,
    train,
    train_tokenizer,
    val_sample,
)

FAST = TrainConfig(
    steps=50,
    batch_size=8,
    lr=3e-3,
    warmup_steps=5,
    eval_every=25,
    eval_batches=2,
    checkpoint_every=1000,
    log_every=10,
    device="cpu",
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    build_dataset(root, 300, seed=0)
    return root


@pytest.fixture(scope="module")
def tokenizer(dataset: Path) -> Tokenizer:
    return train_tokenizer(dataset, vocab_size=LUNA.vocab_size)


def test_tokenizer_round_trips_every_factory_string(dataset: Path, tokenizer: Tokenizer) -> None:
    for e in read_examples(dataset / "train"):
        for text in (e.prompt, e.target):
            assert tokenizer.decode(encode_task(tokenizer, text)) == text


def test_make_batch_only_learns_targets(tokenizer: Tokenizer) -> None:
    e = Example("classify", "train", "t", '<|classify|>{"question": "q"}', "ok<|end|>")
    data = tokenize_examples([e], tokenizer, 1024)
    ids, target_start = data.rows[0]
    inputs, labels = make_batch(data.rows, tokenizer.pad_id, torch.device("cpu"))
    assert inputs.shape == labels.shape and inputs.shape[1] >= len(ids) - 1
    learned = [int(t) for t in labels[0] if t != IGNORE_INDEX]
    assert learned == ids[target_start:]
    assert learned[-1] == tokenizer.end_id


def test_padding_is_ignored(tokenizer: Tokenizer) -> None:
    rows = [([1, 2, 3, 4], 2), ([1, 2], 1)]
    inputs, labels = make_batch(rows, tokenizer.pad_id, torch.device("cpu"))
    assert inputs[1, 1] == tokenizer.pad_id
    assert labels[1, 1] == IGNORE_INDEX


def test_batch_widths_are_bucketed(tokenizer: Tokenizer) -> None:
    rows = [([1] * 70, 2), ([1] * 10, 1)]
    inputs, labels = make_batch(rows, tokenizer.pad_id, torch.device("cpu"), multiple=64)
    assert inputs.shape == labels.shape == (2, 96)
    assert (labels[:, 69:] == IGNORE_INDEX).all()
    # Rounding never pushes past the model's context, and never cuts a row short.
    inputs, _ = make_batch(rows, tokenizer.pad_id, torch.device("cpu"), multiple=64, max_width=80)
    assert inputs.shape == (2, 80)
    inputs, _ = make_batch(rows, tokenizer.pad_id, torch.device("cpu"), multiple=64, max_width=50)
    assert inputs.shape == (2, 69)


def test_batch_widths_come_in_a_few_buckets() -> None:
    assert [bucket_width(n, 64) for n in (1, 64, 65, 97, 129, 300, 513, 1024)] == [
        64, 64, 96, 128, 192, 384, 768, 1024,
    ]  # fmt: skip
    widths = {bucket_width(n, 64) for n in range(1, 1025)}
    assert len(widths) == 9 and all(n <= bucket_width(n, 64) < n * 1.5 + 64 for n in (65, 700))
    cpu = torch.device("cpu")
    # Widths are the row length minus one (the inputs), so 65 -> 96 needs 66 ids.
    for ids_len, width in [(2, 64), (66, 96), (130, 192), (1025, 1024)]:
        inputs, _ = make_batch([([1] * ids_len, 1)], 0, cpu, multiple=64)
        assert inputs.shape[1] == width
    # A cap is honored, but never cuts below the longest row (129 here).
    rows = [([1] * 130, 1)]
    inputs, _ = make_batch(rows, 0, cpu, multiple=64, max_width=100)
    assert inputs.shape[1] == 129


def test_tokenized_set_stores_compact_rows() -> None:
    data = TokenizedSet()
    data.add([1, 2, 3], 1, "a")
    data.add([4, 5], 1, "b")
    data.add([6, 7, 8, 9], 2, "a")
    assert len(data) == 3 and len(data.rows) == 3
    assert data.tasks == ["a", "b", "a"] and data.tokens.itemsize == 4
    assert data.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    rows = [([1, 2, 3], 1), ([4, 5], 1), ([6, 7, 8, 9], 2)]
    assert data.rows[0] == rows[0] and data.rows[-1] == rows[2]
    assert data.rows[0:2] == rows[0:2] and list(data.rows) == rows
    assert data.rows == rows and data.rows != rows[:2]
    assert {"x": data.rows} == {"x": rows}
    assert all(p in rows for p in random.Random(0).sample(data.rows, 2))
    with pytest.raises(IndexError):
        data.rows[3]


def test_sample_examples_by_task_is_bounded_deterministic_and_complete(
    dataset: Path,
) -> None:
    examples = list(read_examples(dataset / "val"))
    a = sample_examples_by_task(examples, per_task=3, seed=5)
    assert a == sample_examples_by_task(examples, per_task=3, seed=5)
    totals = Counter(e.task for e in examples)
    counts = Counter(e.task for e in a)
    assert counts == {task: min(3, n) for task, n in totals.items()}
    assert [e.task for e in a] == sorted(e.task for e in a)


def test_too_long_examples_skipped(tokenizer: Tokenizer) -> None:
    e = Example("explain", "train", "t", "<|explain|>" + "x " * 600, "y<|end|>")
    data = tokenize_examples([e], tokenizer, 64)
    assert len(data) == 0 and data.skipped == 1


def test_lr_schedule() -> None:
    cfg = TrainConfig(steps=100, lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_at(0, cfg) == pytest.approx(0.1)
    assert lr_at(9, cfg) == pytest.approx(1.0)
    assert lr_at(10, cfg) == pytest.approx(1.0)
    assert lr_at(99, cfg) == pytest.approx(0.1, abs=1e-3)
    assert lr_at(55, cfg) < lr_at(20, cfg)


def test_a_longer_resume_carries_on_from_its_learning_rate() -> None:
    # Resuming a 3000-step run at step 1500 as a 6000-step run once doubled the rate
    # (0.00028 -> 0.00054) and knocked the loss from 4.2 back to 6.4.
    old, longer = TrainConfig(steps=3000, lr=6e-4), TrainConfig(steps=6000, lr=6e-4)
    at = lr_at(1499, old)
    assert lr_at(1500, longer) > at * 1.2  # what a plain schedule would do
    assert lr_at(1500, longer, (1500, at)) == pytest.approx(at)
    assert lr_at(1501, longer, (1500, at)) < at
    assert lr_at(5999, longer, (1500, at)) == pytest.approx(6e-5, rel=1e-3)


def test_luna_trains_and_loss_drops(dataset: Path, tokenizer: Tokenizer, tmp_path: Path) -> None:
    out = tmp_path / "luna"
    metrics = train(LUNA, tokenizer, dataset, out, FAST)
    # Single-batch losses are noisy over 50 steps, so compare the last two to the first,
    # and require held-out loss to fall too.
    losses = [m["loss"] for m in metrics if "loss" in m]
    assert sum(losses[-2:]) / 2 < losses[0] * 0.9
    evals = [m for m in metrics if "val_loss" in m]
    assert len(evals) >= 2 and evals[-1]["val_loss"] < evals[0]["val_loss"]
    assert {"val_loss_classify", "val_loss_plan", "val_loss_explain"} <= evals[-1].keys()
    model, loaded_tok = load_model(out)
    assert model.num_parameters() == LUNA.num_parameters()
    assert loaded_tok.merges == tokenizer.merges
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["config"] == "fermi-luna-1"


def test_fp32_precision_trains_and_is_recorded(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "steps": 5, "precision": "fp32"})
    train(LUNA, tokenizer, dataset, tmp_path / "fp32", cfg)
    summary = json.loads((tmp_path / "fp32" / "training_summary.json").read_text())
    assert summary["train"]["precision"] == "fp32"


def test_log_reports_memory_and_a_windowed_rate(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "steps": 10, "log_every": 5})
    metrics = train(LUNA, tokenizer, dataset, tmp_path / "luna", cfg)
    logged = [m for m in metrics if "loss" in m]
    assert [m["step"] for m in logged] == [5, 10]
    for m in logged:
        assert m["mem_gb"] > 0 and m["target_tokens_per_s"] > 0
    assert _peak_memory_gb() > 0
    # Validation is a fixed sample per task, not the whole split.
    summary = json.loads((tmp_path / "luna" / "training_summary.json").read_text())
    assert 0 < summary["val_examples"] <= 3 * cfg.eval_batches * cfg.batch_size


def test_val_sample_is_fixed_and_per_task(dataset: Path, tokenizer: Tokenizer) -> None:
    val = tokenize_examples(read_examples(dataset / "val"), tokenizer, 1024)
    a = val_sample(val, per_task=5, seed=1)
    assert set(a) == {"classify", "plan", "explain"}
    for task, sample in a.items():
        assert 0 < len(sample) <= 5
        assert set(sample.tasks) == {task}
    assert {t: s.rows for t, s in a.items()} == {
        t: s.rows for t, s in val_sample(val, per_task=5, seed=1).items()
    }


def test_fresh_run_starts_a_fresh_metrics_log(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "luna"
    out.mkdir()
    (out / "metrics.jsonl").write_text('{"step": 850, "loss": 0.071}\n')
    train(LUNA, tokenizer, dataset, out, TrainConfig(**{**FAST.__dict__, "steps": 10}))
    steps = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert steps and max(steps) == 10


def test_resume_continues_from_the_checkpoint(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "luna"
    first = TrainConfig(**{**FAST.__dict__, "steps": 10})
    train(LUNA, tokenizer, dataset, out, first)
    assert json.loads((out / STATE_FILE).read_text())["step"] == 10
    more = TrainConfig(**{**FAST.__dict__, "steps": 20})
    metrics = train(LUNA, tokenizer, dataset, out, more, resume=True)
    assert min(m["step"] for m in metrics) > 10
    state = json.loads((out / STATE_FILE).read_text())
    assert state["step"] == 20 and state["lr"] is not None
    # The resumed run starts from the rate the first one ended at, and only goes down.
    rates = [m["lr"] for m in metrics if "lr" in m]
    assert rates == sorted(rates, reverse=True) and rates[0] <= lr_at(9, first)
    logged = [json.loads(line)["step"] for line in (out / "metrics.jsonl").read_text().splitlines()]
    assert min(logged) <= 10 < max(logged)  # resuming keeps the earlier log


def test_grad_accum_must_be_positive(dataset: Path, tokenizer: Tokenizer, tmp_path: Path) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "grad_accum": 0})
    with pytest.raises(ValueError, match="grad_accum"):
        train(LUNA, tokenizer, dataset, tmp_path / "x", cfg)


def test_accumulated_step_matches_one_big_batch() -> None:
    # Four rows with the same number of target tokens: one step on all four gives the same
    # parameters and loss as one step on two micro-batches of two, each weighted by 1/2.
    config = ModelConfig(
        name="t", vocab_size=64, d_model=16, n_layers=1, n_heads=2, context_length=8
    )
    cpu = torch.device("cpu")
    gen = torch.Generator().manual_seed(1)
    rows = [(torch.randint(0, 64, (6,), generator=gen).tolist(), 3) for _ in range(4)]
    cfg = TrainConfig(lr=1e-2)

    torch.manual_seed(0)
    whole = FermiLM(config)
    accum = FermiLM(config)
    accum.load_state_dict(whole.state_dict())
    opt_whole, opt_accum = _optimizer(whole, cfg), _optimizer(accum, cfg)
    width = {"max_width": config.context_length}
    big = _optimizer_step(whole, opt_whole, [make_batch(rows, 0, cpu, **width)], cfg, cpu, None)
    micro = [make_batch(rows[:2], 0, cpu, **width), make_batch(rows[2:], 0, cpu, **width)]
    small = _optimizer_step(accum, opt_accum, micro, cfg, cpu, None)

    assert torch.allclose(big, small, atol=1e-5)
    for p, q in zip(whole.parameters(), accum.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-5)


def test_grad_accum_trains_and_is_recorded(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "steps": 10, "log_every": 5, "grad_accum": 2})
    metrics = train(LUNA, tokenizer, dataset, tmp_path / "luna", cfg)
    logged = [m for m in metrics if "loss" in m]
    assert [m["step"] for m in logged] == [5, 10]
    summary = json.loads((tmp_path / "luna" / "training_summary.json").read_text())
    assert summary["train"]["grad_accum"] == 2
    assert summary["train"]["batch_size"] == cfg.batch_size


def test_torch_trainer_recomputes_blocks_only_when_asked(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from askphysics.lm import model as model_module

    calls: list[int] = []
    real = model_module.checkpoint

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model_module, "checkpoint", counting)
    # Only the final evaluation runs (in eval mode), so every recomputation is a training one.
    base = {**FAST.__dict__, "steps": 4, "eval_every": 1000, "checkpoint_every": 1000}
    train(LUNA, tokenizer, dataset, tmp_path / "plain", TrainConfig(**base))
    assert calls == []
    checkpointed = TrainConfig(**{**base, "checkpoint_blocks": True})
    train(LUNA, tokenizer, dataset, tmp_path / "checkpointed", checkpointed)
    assert len(calls) == base["steps"] * LUNA.n_layers
    plain_model, _ = load_model(tmp_path / "plain")
    ckpt_model, _ = load_model(tmp_path / "checkpointed")
    for p, q in zip(plain_model.parameters(), ckpt_model.parameters(), strict=True):
        assert torch.allclose(p, q, atol=1e-6)


def test_cli_accepts_checkpoint_blocks_on_torch(
    tokenizer: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, TrainConfig] = {}

    def fake_train(*args: object, **kwargs: object) -> list[dict[str, object]]:
        seen["cfg"] = args[4]  # type: ignore[assignment]
        return []

    monkeypatch.setattr(train_module, "train", fake_train)
    tok_path = tmp_path / "tok.json"
    tokenizer.save(tok_path)
    r = CliRunner().invoke(
        cli.app,
        [
            "model", "train", "--model", "fermi-luna-1", "--backend", "torch", "--device", "cpu",
            "--checkpoint-blocks", "--steps", "5", "--tokenizer", str(tok_path),
            "--data", str(tmp_path), "--out", str(tmp_path / "out"),
        ],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    assert seen["cfg"].checkpoint_blocks is True


def test_cli_rejects_zero_grad_accum() -> None:
    r = CliRunner().invoke(cli.app, ["model", "train", "--grad-accum", "0"])
    # CI forces color, and rich styles each hyphen of the option name on its own.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", r.output)
    assert r.exit_code != 0 and "--grad-accum" in plain


def test_optimizer_state_round_trips(tmp_path: Path) -> None:
    config = ModelConfig(
        name="t", vocab_size=64, d_model=16, n_layers=1, n_heads=2, context_length=8
    )
    model = FermiLM(config)
    opt = _optimizer(model, FAST)
    _, loss = model(torch.randint(0, 64, (2, 8)), torch.randint(0, 64, (2, 8)))
    assert loss is not None
    loss.backward()
    opt.step()
    _save_optimizer(opt, 7, tmp_path)
    fresh = _optimizer(model, FAST)
    assert _load_optimizer(fresh, tmp_path) == 7
    a, b = opt.state_dict()["state"], fresh.state_dict()["state"]
    assert a.keys() == b.keys()
    for k in a:
        assert torch.equal(a[k]["exp_avg"], b[k]["exp_avg"])


def test_tokenizer_bigger_than_model_rejected(dataset: Path, tmp_path: Path) -> None:
    big = train_tokenizer(dataset, vocab_size=2000)
    if big.vocab_size <= LUNA.vocab_size:
        pytest.skip("corpus too small to exceed the luna vocabulary")
    with pytest.raises(ValueError, match="tokenizer has"):
        train(LUNA, big, dataset, tmp_path / "x", FAST)


def test_cli_end_to_end(tmp_path: Path) -> None:
    runner = CliRunner()
    data, tok, models = tmp_path / "data", tmp_path / "tok.json", tmp_path / "models"
    r = runner.invoke(
        cli.app,
        [
            "model",
            "build-data",
            "--out",
            str(data),
            "--examples",
            "200",
            "--blocklist",
            str(tmp_path / "none.yaml"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "no eval questions found" in r.output
    r = runner.invoke(
        cli.app,
        ["model", "train-tokenizer", "--data", str(data), "--out", str(tok), "--vocab-size", "512"],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(
        cli.app,
        [
            "model",
            "train",
            "--data",
            str(data),
            "--tokenizer",
            str(tok),
            "--out",
            str(models / "fermi-luna-1"),
            "--steps",
            "6",
            "--batch-size",
            "4",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli.app, ["model", "info", "--directory", str(models)])
    assert r.exit_code == 0 and "fermi-luna-1" in r.output
    r = runner.invoke(cli.app, ["model", "info", "--directory", str(tmp_path / "empty")])
    assert "No Fermi models installed" in r.output
    r = runner.invoke(cli.app, ["model", "train", "--model", "fermi-blackhole-1"])
    assert r.exit_code == 1 and "unknown model" in r.output


def test_prose_steps_must_leave_task_steps(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    cfg = TrainConfig(steps=10, batch_size=2, prose_steps=10, prose_share=0.1)
    with pytest.raises(ValueError, match="never trains on the tasks"):
        train(
            LUNA, tokenizer, dataset, tmp_path / "luna", cfg, prose=["Some prose here. " * 20] * 30
        )


def test_width_flush_releases_the_cache_only_when_the_width_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    released: list[str] = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: released.append("mps"))
    flush = _WidthFlush(torch.device("mps"))
    for width in (64, 64, 96, 96, 64, 64, 1024):
        flush.before_step(width)
    # The first step has nothing to compare with; then 64 to 96, 96 to 64, and 64 to 1024.
    assert released == ["mps", "mps", "mps"]


def test_train_flushes_on_each_width_change_and_not_otherwise(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    widths: list[int] = []
    real_make_batch = train_module.make_batch

    def recording_make_batch(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, labels = real_make_batch(*args, **kwargs)  # type: ignore[arg-type]
        widths.append(inputs.shape[1])
        return inputs, labels

    released: list[str] = []
    monkeypatch.setattr(train_module, "make_batch", recording_make_batch)
    monkeypatch.setattr(
        train_module, "_release_cached_memory", lambda device: released.append(device.type)
    )
    cfg = TrainConfig(
        **{
            **FAST.__dict__,
            "steps": 16,
            "grad_accum": 2,
            "flush_every": 0,
            "eval_every": 1000,
            "checkpoint_every": 1000,
        }
    )
    train(LUNA, tokenizer, dataset, tmp_path / "flush", cfg)
    # Each optimizer step is grad_accum micro-batches; the step's width is its widest one.
    per_step = [
        max(widths[i : i + cfg.grad_accum]) for i in range(0, cfg.steps * cfg.grad_accum, 2)
    ]
    changes = sum(a != b for a, b in itertools.pairwise(per_step))
    assert changes > 0  # the width does move, so the flush is exercised
    # The evaluation after the last step releases the cache too.
    assert released == ["cpu"] * (changes + 1)
