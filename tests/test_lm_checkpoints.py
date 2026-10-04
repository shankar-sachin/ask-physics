import json
import random
from pathlib import Path

import pytest
import torch

from askphysics.errors import ConfigError
from askphysics.lm import checkpoints
from askphysics.lm.checkpoints import (
    CONFIG_FILE,
    TOKENIZER_FILE,
    WEIGHTS_FILE,
    default_model_dir,
    load_model,
    save_model,
)
from askphysics.lm.config import LUNA
from askphysics.lm.model import FermiLM
from askphysics.lm.tokenizer import Tokenizer

SRC = Path(checkpoints.__file__).parent


@pytest.fixture
def saved(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = FermiLM(LUNA)
    tokenizer = Tokenizer.train(["the quick brown fox jumps over the lazy dog"] * 3, 300)
    save_model(model, tokenizer, tmp_path / "fermi-luna-1")
    return tmp_path / "fermi-luna-1"


def test_round_trip_gives_identical_logits(saved: Path) -> None:
    torch.manual_seed(0)
    original = FermiLM(LUNA).eval()
    loaded, tokenizer = load_model(saved)
    ids = torch.randint(0, LUNA.vocab_size, (1, 10))
    assert torch.equal(original(ids)[0], loaded(ids)[0])
    assert not loaded.training
    assert tokenizer.vocab_size <= LUNA.vocab_size


def test_files_written(saved: Path) -> None:
    assert {p.name for p in saved.iterdir()} == {WEIGHTS_FILE, CONFIG_FILE, TOKENIZER_FILE}


def test_missing_files_reported(saved: Path) -> None:
    (saved / TOKENIZER_FILE).unlink()
    with pytest.raises(ConfigError, match=r"missing tokenizer\.json"):
        load_model(saved)


def test_tampered_config_rejected(saved: Path) -> None:
    config = json.loads((saved / CONFIG_FILE).read_text())
    config["n_layers"] = 3
    (saved / CONFIG_FILE).write_text(json.dumps(config))
    with pytest.raises(ConfigError, match="does not match"):
        load_model(saved)


def test_wrong_format_version_rejected(saved: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(checkpoints, "FORMAT_VERSION", 999)
    with pytest.raises(ConfigError, match="task format"):
        load_model(saved)


def test_tokenizer_bigger_than_model_rejected(tmp_path: Path) -> None:
    rng = random.Random(0)
    words = ["".join(rng.choice("abcdefghijklmnop") for _ in range(6)) for _ in range(3000)]
    big = Tokenizer.train([" ".join(words)] * 2, vocab_size=1000)
    assert big.vocab_size > LUNA.vocab_size
    with pytest.raises(ConfigError, match="tokenizer has"):
        save_model(FermiLM(LUNA), big, tmp_path / "x")


def test_default_model_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(tmp_path))
    assert default_model_dir() == tmp_path
    monkeypatch.delenv("ASKPHYSICS_MODEL_DIR")
    assert default_model_dir().parts[-3:] == (".cache", "askphysics", "models")


def test_never_unpickles() -> None:
    # Risk R15: loading a pickle runs arbitrary code. Weights are safetensors only.
    for path in SRC.rglob("*.py"):
        text = path.read_text()
        assert "torch.load(" not in text, path
        assert "import pickle" not in text, path
