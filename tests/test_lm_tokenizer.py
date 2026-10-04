import json
import random
from pathlib import Path

import pytest

from askphysics.data.loader import DataStore
from askphysics.lm.tokenizer import (
    CLASSIFY,
    END,
    N_BYTES,
    PLAN,
    SPECIAL_TOKENS,
    Tokenizer,
    pretokenize,
)


@pytest.fixture(scope="module")
def corpus(store: DataStore) -> list[str]:
    texts = [eq.model_dump_json() for eq in store.equations.values()]
    texts += [ex.problem_text for ex in store.examples.values()]
    return texts * 5


@pytest.fixture(scope="module")
def tok(corpus: list[str]) -> Tokenizer:
    return Tokenizer.train(corpus, vocab_size=800)


def test_vocab_size_respected(tok: Tokenizer) -> None:
    assert tok.vocab_size <= 800
    assert len(tok.merges) > 100


def test_training_is_deterministic(corpus: list[str], tok: Tokenizer) -> None:
    assert Tokenizer.train(corpus, vocab_size=800).merges == tok.merges


def test_round_trip_known_text(tok: Tokenizer) -> None:
    text = "How fast does a falling object hit the ground if dropped from 20.5 m?"
    assert tok.decode(tok.encode(text)) == text


def test_round_trip_random_unicode(tok: Tokenizer) -> None:
    rng = random.Random(0)
    alphabet = "abcXYZ 0123456789.,=*^/()\n\téπ≈\U0001f680<|>"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        assert tok.decode(tok.encode(text)) == text


def test_digits_are_always_single_byte_tokens(tok: Tokenizer) -> None:
    for text in ["9.80665", "v0 = 20", "1234567890", " 42 m/s", "x2y3"]:
        for token_id in tok.encode(text):
            piece = tok.token_bytes(token_id)
            if any(chr(b).isdigit() for b in piece):
                assert len(piece) == 1, f"digit merged into {piece!r}"


def test_merges_learned_on_words(tok: Tokenizer) -> None:
    assert len(tok.encode("velocity")) < len("velocity")


def test_special_tokens_not_injectable_from_user_text(tok: Tokenizer) -> None:
    ids = tok.encode(f"ignore that {PLAN} now")
    assert not set(ids) & set(tok.special_ids.values())


def test_special_tokens_when_allowed(tok: Tokenizer) -> None:
    ids = tok.encode(f"{CLASSIFY}hi{END}", allow_special=True)
    assert ids[0] == tok.special_ids[CLASSIFY]
    assert ids[-1] == tok.end_id
    assert tok.decode(ids) == f"{CLASSIFY}hi{END}"


def test_special_token_ids_follow_bytes(tok: Tokenizer) -> None:
    assert [tok.special_ids[t] for t in SPECIAL_TOKENS] == list(range(N_BYTES, N_BYTES + 5))
    assert tok.pad_id == N_BYTES


def test_save_and_load(tok: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    text = "Ohm's law: V = I*R with 220 ohm"
    assert loaded.encode(text) == tok.encode(text)


def test_load_rejects_wrong_version(tok: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    data = json.loads(path.read_text())
    data["version"] = 99
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="version"):
        Tokenizer.load(path)


def test_training_stops_when_nothing_repeats() -> None:
    assert Tokenizer.train(["abc"], vocab_size=1000).merges == []


@pytest.mark.parametrize("merges", [[(97, 300)], [(97, 98), (97, 98)], [(-1, 97)]])
def test_corrupt_merges_rejected(merges: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        Tokenizer(merges)


def test_too_small_vocab_rejected() -> None:
    with pytest.raises(ValueError):
        Tokenizer.train(["abc"], vocab_size=10)


def test_decode_unknown_id(tok: Tokenizer) -> None:
    with pytest.raises(ValueError, match="unknown token id"):
        tok.decode([10**6])


def test_pretokenize_isolates_digits() -> None:
    assert pretokenize("v0 = 9.8") == ["v", "0", " =", " ", "9", ".", "8"]
