"""``askphysics.lm.package``: what a release ships, run on an untrained fermi-luna-1."""

import json
from pathlib import Path

import pytest
import torch

from askphysics.data.loader import DataStore
from askphysics.lm.checkpoints import CONFIG_FILE, WEIGHTS_FILE, load_model, save_model
from askphysics.lm.config import LUNA
from askphysics.lm.factory import build_dataset
from askphysics.lm.model import FermiLM
from askphysics.lm.package import ATTRIBUTION_FILE, CARD_FILE, measure_throughput, package
from askphysics.lm.train import train_tokenizer
from askphysics.lm.weights import read_manifest, sha256_of


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = tmp_path_factory.mktemp("data")
    build_dataset(data, 200, seed=3)
    torch.manual_seed(0)
    source = tmp_path_factory.mktemp("models") / LUNA.name
    save_model(FermiLM(LUNA), train_tokenizer(data, vocab_size=LUNA.vocab_size), source)
    metrics = [
        {"step": 100, "loss": 2.0, "lr": 0.001, "target_tokens_per_s": 5000.0},
        {"step": 100, "val_loss": 2.5, "val_loss_prose": 4.0},
        {"step": 200, "val_loss": 1.5},
    ]
    (source / "metrics.jsonl").write_text("\n".join(json.dumps(m) for m in metrics))
    (source / "training_summary.json").write_text(json.dumps(
        {"train_examples": 1000, "prose_paragraphs": 0, "device": "cpu",
         "train": {"steps": 200, "batch_size": 32, "prose_steps": 0, "prose_share": 0.0}}
    ))  # fmt: skip
    (source / "eval.json").write_text(json.dumps({
        "classify_examples": 3000, "category_accuracy": 1.0, "plan_examples": 3000,
        "equation_accuracy": 0.99, "target_accuracy": 0.995, "knowns_accuracy": 0.994,
        "valid_plan_rate": 0.989, "attempts": 5, "routed_right_rate": 0.992,
        "flagged_wrong_rate": 0.004, "confidently_wrong_rate": 0.004,
    }))  # fmt: skip
    return source


def test_package_ships_bf16_weights_a_card_and_pins(
    trained: Path, store: DataStore, tmp_path: Path
) -> None:
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": {}}))
    attribution = tmp_path / "ATTRIBUTION.md"
    attribution.write_text("# Attribution\n\nOpenStax, CC BY 4.0.\n")
    out = tmp_path / "release"
    entry = package(LUNA.name, trained, out, "models-test", store, attribution=attribution,
                    measure=False, manifest=manifest)  # fmt: skip

    packaged = out / LUNA.name
    assert (packaged / WEIGHTS_FILE).stat().st_size < (trained / WEIGHTS_FILE).stat().st_size
    original, _ = load_model(trained)
    loaded, _ = load_model(packaged)  # bf16 on disk, float32 once loaded
    ids = torch.randint(0, LUNA.vocab_size, (1, 12))
    assert torch.allclose(original(ids)[0], loaded(ids)[0], atol=0.1)

    card = (packaged / CARD_FILE).read_text()
    for expected in ("Release `models-test`", f"| Parameters | {loaded.num_parameters():,} |",
                     "| 200 | 1.5000 |", "5,000 target tokens per second",
                     "| Confidently wrong | 0.4% |", ATTRIBUTION_FILE):  # fmt: skip
        assert expected in card, expected
    assert (packaged / ATTRIBUTION_FILE).read_text() == attribution.read_text()

    assets = out / "assets"
    assert {p.name for p in assets.iterdir()} == {
        f"{LUNA.name}.safetensors", f"{LUNA.name}.config.json", f"{LUNA.name}.tokenizer.json",
        f"{LUNA.name}.card.md", f"{LUNA.name}.attribution.md",
    }  # fmt: skip
    pinned = read_manifest(manifest)[LUNA.name]
    assert pinned.release == "models-test" and len(pinned.files) == 5
    weights = next(f for f in pinned.files if f.name == WEIGHTS_FILE)
    assert weights.sha256 == sha256_of(assets / f"{LUNA.name}.safetensors")
    assert entry["files"][CONFIG_FILE]["url"].endswith(f"/models-test/{LUNA.name}.config.json")


def test_without_an_eval_the_card_says_so(trained: Path, store: DataStore, tmp_path: Path) -> None:
    bare = tmp_path / LUNA.name
    bare.mkdir()
    for name in (WEIGHTS_FILE, CONFIG_FILE, "tokenizer.json"):
        (bare / name).write_bytes((trained / name).read_bytes())
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": {}}))
    package(LUNA.name, bare, tmp_path / "out", "r", store, measure=False, manifest=manifest)
    card = (tmp_path / "out" / LUNA.name / CARD_FILE).read_text()
    assert "Not evaluated yet" in card and ATTRIBUTION_FILE not in card


def test_throughput_is_timed_through_the_pipeline(trained: Path, store: DataStore) -> None:
    questions = ["A 2 kg cart speeds up at 3 m/s^2. What force acts on it?"] * 2
    result = measure_throughput(LUNA.name, trained, store, device="cpu", questions=questions)
    assert result.device == "cpu" and result.questions == 2
    assert 0 < result.median_s <= result.worst_s
