import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from askphysics import cli
from askphysics.lm.checkpoints import load_model
from askphysics.lm.config import LUNA, ModelConfig
from askphysics.lm.factory import Example, build_dataset, read_examples
from askphysics.lm.generate import encode_task
from askphysics.lm.model import IGNORE_INDEX, FermiLM
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.train import (
    STATE_FILE,
    TrainConfig,
    _load_optimizer,
    _optimizer,
    _save_optimizer,
    lr_at,
    make_batch,
    tokenize_examples,
    train,
    train_tokenizer,
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
    assert inputs.shape == labels.shape == (1, len(ids) - 1)
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
    assert inputs.shape == labels.shape == (2, 128)
    assert (labels[:, 69:] == IGNORE_INDEX).all()
    # Rounding never pushes past the model's context, and never cuts a row short.
    inputs, _ = make_batch(rows, tokenizer.pad_id, torch.device("cpu"), multiple=64, max_width=100)
    assert inputs.shape == (2, 100)
    inputs, _ = make_batch(rows, tokenizer.pad_id, torch.device("cpu"), multiple=64, max_width=50)
    assert inputs.shape == (2, 69)


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


def test_luna_trains_and_loss_drops(dataset: Path, tokenizer: Tokenizer, tmp_path: Path) -> None:
    out = tmp_path / "luna"
    metrics = train(LUNA, tokenizer, dataset, out, FAST)
    losses = [m["loss"] for m in metrics if "loss" in m]
    assert losses[-1] < losses[0] * 0.8
    assert any("val_loss" in m for m in metrics)
    model, loaded_tok = load_model(out)
    assert model.num_parameters() == LUNA.num_parameters()
    assert loaded_tok.merges == tokenizer.merges
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["config"] == "fermi-luna-1"


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
    assert json.loads((out / STATE_FILE).read_text())["step"] == 20


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
