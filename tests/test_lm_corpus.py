import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from askphysics import cli
from askphysics.lm.config import LUNA
from askphysics.lm.corpus import (
    PROSE_TASK,
    VAL_EVERY,
    read_prose,
    split_prose,
    tokenize_prose,
)
from askphysics.lm.factory import build_dataset
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.train import TrainConfig, make_batch, train, train_tokenizer

ROOT = Path(__file__).resolve().parents[1]
OPENSTAX = ROOT / "third_party" / "openstax-physics"
PARAGRAPHS = [
    "Force is a push or a pull, and it always acts between two objects.",
    "Energy can change form, but in a closed system the total stays the same.",
    "A wave carries energy from one place to another without carrying matter along.",
] * 10

FAST = TrainConfig(
    steps=12,
    batch_size=4,
    lr=3e-3,
    warmup_steps=2,
    eval_every=6,
    eval_batches=1,
    checkpoint_every=1000,
    log_every=3,
    device="cpu",
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    build_dataset(root, 200, seed=3)
    return root


@pytest.fixture(scope="module")
def tokenizer(dataset: Path) -> Tokenizer:
    return train_tokenizer(dataset, vocab_size=LUNA.vocab_size, prose=PARAGRAPHS)


def _write(path: Path, texts: list[str]) -> Path:
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts), encoding="utf-8")
    return path


def test_read_and_split(tmp_path: Path) -> None:
    texts = read_prose(_write(tmp_path / "p.jsonl", PARAGRAPHS))
    assert texts == PARAGRAPHS
    train_texts, val_texts = split_prose(texts)
    assert len(val_texts) == len(texts) // VAL_EVERY
    assert len(train_texts) + len(val_texts) == len(texts)
    with pytest.raises(ValueError):
        read_prose(_write(tmp_path / "empty.jsonl", []))


def test_prose_rows_learn_every_token_after_the_first(tokenizer: Tokenizer) -> None:
    rows = tokenize_prose(PARAGRAPHS[:1], tokenizer, LUNA.context_length)
    assert rows.tasks == [PROSE_TASK]
    ids, start = rows.rows[0]
    assert start == 1 and ids[-1] == tokenizer.end_id
    _, labels = make_batch(rows.rows, tokenizer.pad_id, "cpu")  # type: ignore[arg-type]
    assert int((labels >= 0).sum()) == len(ids) - 1


def test_long_paragraphs_are_chunked(tokenizer: Tokenizer) -> None:
    rows = tokenize_prose([" ".join(PARAGRAPHS)], tokenizer, context_length=32)
    assert len(rows) > 1
    assert all(len(ids) <= 33 for ids, _ in rows.rows)


def test_training_with_a_prose_stage(dataset: Path, tokenizer: Tokenizer, tmp_path: Path) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "prose_steps": 6, "prose_share": 0.2})
    metrics = train(LUNA, tokenizer, dataset, tmp_path / "luna", cfg, prose=PARAGRAPHS)
    evals = [m for m in metrics if "val_loss" in m]
    assert evals and all("val_loss_prose" in m for m in evals)
    assert all("val_loss_prose" not in m for m in metrics if "val_loss" not in m)
    summary = json.loads((tmp_path / "luna" / "training_summary.json").read_text())
    assert summary["prose_paragraphs"] == len(PARAGRAPHS)


def test_prose_knobs_need_prose(dataset: Path, tokenizer: Tokenizer, tmp_path: Path) -> None:
    cfg = TrainConfig(**{**FAST.__dict__, "prose_steps": 2})
    with pytest.raises(ValueError, match="need prose"):
        train(LUNA, tokenizer, dataset, tmp_path / "luna", cfg)


def test_cli_trains_with_prose(dataset: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    prose = _write(tmp_path / "p.jsonl", PARAGRAPHS)
    tok = tmp_path / "tok.json"
    r = runner.invoke(
        cli.app,
        ["model", "train-tokenizer", "--data", str(dataset), "--out", str(tok),
         "--vocab-size", "512", "--prose", str(prose)],
    )  # fmt: skip
    assert r.exit_code == 0, r.output
    args = ["model", "train", "--data", str(dataset), "--tokenizer", str(tok),
            "--out", str(tmp_path / "luna"), "--steps", "4", "--batch-size", "2",
            "--device", "cpu"]  # fmt: skip
    r = runner.invoke(cli.app, [*args, "--prose", str(prose), "--prose-steps", "2"])
    assert r.exit_code == 0, r.output
    r = runner.invoke(cli.app, [*args, "--prose-steps", "2"])
    assert r.exit_code == 1 and "need --prose" in r.output


def test_the_shipped_openstax_prose_is_clean_and_attributed() -> None:
    texts = read_prose(OPENSTAX / "prose.jsonl")
    assert len(texts) > 1000
    assert all(t.isascii() and len(t.split()) >= 8 for t in texts)
    assert not any("The student is expected" in t or "Texas" in t for t in texts)
    assert (OPENSTAX / "LICENSE").read_text().startswith("Attribution 4.0 International")
    attribution = (OPENSTAX / "ATTRIBUTION.md").read_text()
    assert "CC BY 4.0" in attribution and "openstax.org" in attribution


def test_a_big_corpus_is_sampled_for_the_tokenizer(
    dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import askphysics.lm.train as train_module

    seen: list[int] = []
    real = Tokenizer.train

    def spy(texts: list[str], vocab_size: int) -> Tokenizer:
        seen.append(sum(1 for t in texts if t in PARAGRAPHS))
        return real(texts, vocab_size=vocab_size)

    monkeypatch.setattr(train_module, "MAX_TOKENIZER_PROSE", 5)
    monkeypatch.setattr(Tokenizer, "train", staticmethod(spy))
    train_tokenizer(dataset, vocab_size=LUNA.vocab_size, prose=PARAGRAPHS)
    assert seen == [5]
