"""``FermiClient`` and the ``fermi`` provider, run on an untrained fermi-luna-1 on CPU.

An untrained model writes nonsense, but constrained decoding still makes it legal
nonsense: the right schema, retrieved equation ids only, numbers from the input.
"""

import json
from pathlib import Path

import pytest
import torch

from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import ConfigError, LLMError, LLMResponseFormatError
from askphysics.llm.fermi_client import FermiClient, build_roster
from askphysics.lm.checkpoints import save_model
from askphysics.lm.config import LUNA
from askphysics.lm.factory import build_dataset
from askphysics.lm.model import FermiLM
from askphysics.lm.train import train_tokenizer
from askphysics.models import Classification, EquationRef, Plan
from askphysics.pipeline import Pipeline

QUESTION = "A 2 kg cart speeds up at 3 m/s^2. What force acts on it?"


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    data = tmp_path_factory.mktemp("data")
    build_dataset(data, 200, seed=2)
    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("models")
    save_model(FermiLM(LUNA), train_tokenizer(data, vocab_size=LUNA.vocab_size), root / LUNA.name)
    return root


@pytest.fixture
def luna(store: DataStore, models_dir: Path) -> FermiClient:
    return FermiClient(LUNA.name, store, directory=models_dir / LUNA.name, device="cpu")


def test_classifies(luna: FermiClient) -> None:
    out = luna.complete_json(
        system="", user=json.dumps({"question": QUESTION}), schema=Classification
    )
    assert out.category in ("standard", "fermi", "out_of_scope")


def test_plans_with_retrieved_equations_only(luna: FermiClient, store: DataStore) -> None:
    payload = {
        "question": QUESTION,
        "classification": {"category": "standard"},
        "equations": [{"id": "newton_second_law"}, {"id": "kin_v_at"}],
        "constants": [],
        "fermi_assumptions": [],
    }
    out = luna.complete_json(system="", user=json.dumps(payload), schema=Plan)
    assert set(out.equation_ids) <= {"newton_second_law", "kin_v_at"}


def _end_bias(luna: FermiClient, monkeypatch: pytest.MonkeyPatch, bias: float) -> None:
    """Shift the end token's score by ``bias`` on every step of the loaded model."""
    decoder = luna.decoder
    original = decoder.model.step

    def biased(ids: torch.Tensor, past: object = None) -> tuple[torch.Tensor, object]:
        logits, new_past = original(ids, past)  # type: ignore[arg-type]
        logits[..., decoder.tokenizer.end_id] += bias
        return logits, new_past

    monkeypatch.setattr(decoder.model, "step", biased)


EXPLAIN_PAYLOAD = {
    "question": QUESTION,
    "result": {"value": 6.0, "unit": "newton"},
    "equation_ids": ["newton_second_law"],
    "assumptions": [],
    "sanity": {"issues": []},
}


def test_explains(luna: FermiClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _end_bias(luna, monkeypatch, 1000.0)  # an untrained model would otherwise never finish
    assert isinstance(luna.complete_text(system="", user=json.dumps(EXPLAIN_PAYLOAD)), str)


def test_an_unparseable_unit_is_an_llm_error(luna: FermiClient) -> None:
    payload = {**EXPLAIN_PAYLOAD, "result": {"value": 6.0, "unit": "not a unit"}}
    with pytest.raises(LLMError, match="can't read"):
        luna.complete_text(system="", user=json.dumps(payload))


def test_an_unfinished_explanation_is_an_llm_error(
    luna: FermiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _end_bias(luna, monkeypatch, -1e9)  # never writes its end token
    with pytest.raises(LLMError):
        luna.complete_text(system="", user=json.dumps(EXPLAIN_PAYLOAD))


def test_unreadable_payloads_are_llm_errors(luna: FermiClient) -> None:
    with pytest.raises(LLMError):
        luna.complete_json(system="", user="not json", schema=Classification)
    with pytest.raises(LLMError):
        luna.complete_json(system="", user=json.dumps({"q": 1}), schema=Classification)
    with pytest.raises(LLMResponseFormatError):
        luna.complete_json(system="", user=json.dumps({"question": "x"}), schema=EquationRef)
    with pytest.raises(LLMError):
        luna.complete_text(system="", user=json.dumps({"question": "x"}))


def test_missing_weights_are_a_config_error(store: DataStore, tmp_path: Path) -> None:
    client = FermiClient("fermi-solem-1", store, directory=tmp_path / "nope", device="cpu")
    with pytest.raises(ConfigError, match="missing"):
        client.complete_json(system="", user=json.dumps({"question": "x"}), schema=Classification)


def test_roster_shares_one_model_per_name(store: DataStore, models_dir: Path) -> None:
    settings = Settings(llm_provider="fermi", model=LUNA.name, plan_attempts=3, device="cpu")
    roster = build_roster(settings, store, models_dir)
    assert len(roster.plan) == 3
    assert all(c is roster.classify for c in (*roster.plan, roster.explain))


def test_fermi_pipeline_answers_end_to_end(
    store: DataStore, models_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ASKPHYSICS_MODEL_DIR", str(models_dir))
    settings = Settings(llm_provider="auto", model=LUNA.name, plan_attempts=2, device="cpu")
    pipeline = Pipeline.from_settings(settings, data=store)
    answer = pipeline.run(QUESTION)
    assert answer.status in ("answered", "degraded", "refused")
    assert set(answer.models.values()) <= {LUNA.name, "template"}
    assert answer.plan_attempts <= 2


def test_weights_download_on_first_use(store: DataStore, models_dir: Path, tmp_path: Path) -> None:
    calls: list[str] = []
    target = tmp_path / LUNA.name

    def fetch() -> None:
        calls.append("fetch")
        target.mkdir()
        for f in (models_dir / LUNA.name).iterdir():
            (target / f.name).write_bytes(f.read_bytes())

    client = FermiClient(LUNA.name, store, directory=target, device="cpu", fetch=fetch)
    assert calls == []  # building the client downloads nothing
    client.complete_json(system="", user=json.dumps({"question": QUESTION}), schema=Classification)
    client.complete_json(system="", user=json.dumps({"question": QUESTION}), schema=Classification)
    assert calls == ["fetch"]


def test_a_failed_download_is_an_llm_error(store: DataStore, tmp_path: Path) -> None:
    def fetch() -> None:
        raise ConfigError("no network")

    client = FermiClient(LUNA.name, store, directory=tmp_path / "x", device="cpu", fetch=fetch)
    with pytest.raises(LLMError, match="couldn't download"):
        client.complete_json(
            system="", user=json.dumps({"question": QUESTION}), schema=Classification
        )
