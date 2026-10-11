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
from askphysics.errors import ConfigError, DownloadDeclinedError, LLMError, LLMResponseFormatError
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


# --- celeste never downloads by itself (ADR-022) ----------------------------------------------

CELESTE_NAME = "fermi-celeste-1"
CELESTE_FILES = {"model.safetensors": b"c" * 40, "config.json": b"{}", "tokenizer.json": b"{}"}


def _celeste_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, list[str]]:
    """tellus and solem installed (placeholder files), celeste published as file:// URLs.

    Returns the models directory and the list of URLs opened.
    """
    import hashlib

    from askphysics.lm import weights

    root = tmp_path / "models"
    for name in ("fermi-tellus-1", "fermi-solem-1"):
        (root / name).mkdir(parents=True)
        for file in CELESTE_FILES:
            (root / name / file).write_bytes(b"x")
    served = tmp_path / "served"
    served.mkdir()
    entries = {}
    for file, data in CELESTE_FILES.items():
        (served / file).write_bytes(data)
        entries[file] = {
            "url": (served / file).as_uri(),
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest = tmp_path / "weights.json"
    manifest.write_text(
        json.dumps(
            {"format_version": 1, "models": {CELESTE_NAME: {"release": "t", "files": entries}}}
        )
    )
    monkeypatch.setattr(weights, "manifest_path", lambda: manifest)
    opened: list[str] = []
    real_open = weights._open
    monkeypatch.setattr(weights, "_open", lambda url: (opened.append(url), real_open(url))[1])

    def no_weights(*args: object, **kwargs: object) -> None:
        raise ConfigError("placeholder weights")  # no real model loads (golden rule 6)

    monkeypatch.setattr("askphysics.llm.fermi_client.load_model", no_weights)
    return root, opened


def _celeste_client(store: DataStore, roster_plan: tuple[object, ...]) -> FermiClient:
    last = roster_plan[-1]
    assert isinstance(last, FermiClient) and last.name == CELESTE_NAME
    return last


def test_celeste_is_not_on_the_roster_unless_someone_can_be_asked(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The website (Pipeline.from_settings with no callback) and --json look like this.
    root, opened = _celeste_world(monkeypatch, tmp_path)
    roster = build_roster(Settings(llm_provider="fermi", device="cpu"), store, root)
    assert [c.name for c in roster.plan if isinstance(c, FermiClient)] == ["fermi-solem-1"] * 5
    assert opened == []


def test_celeste_is_not_offered_when_auto_pull_is_off(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, _ = _celeste_world(monkeypatch, tmp_path)
    settings = Settings(llm_provider="fermi", device="cpu", auto_pull=False)
    roster = build_roster(settings, store, root, confirm_download=lambda name, size: True)
    assert CELESTE_NAME not in [getattr(c, "name", "") for c in roster.plan]


def test_celeste_asks_once_with_its_real_size_and_downloads_on_yes(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, opened = _celeste_world(monkeypatch, tmp_path)
    asked: list[tuple[str, int]] = []

    def yes(name: str, size: int) -> bool:
        asked.append((name, size))
        return True

    settings = Settings(llm_provider="fermi", device="cpu", escalations=2)
    roster = build_roster(settings, store, root, confirm_download=yes)
    celeste = _celeste_client(store, roster.plan)
    assert asked == [] and opened == []  # building the roster asks and downloads nothing
    with pytest.raises(ConfigError, match="placeholder weights"):  # it got as far as loading
        _ = celeste.decoder
    assert asked == [(CELESTE_NAME, 40 + 2 + 2)]  # the real size from the manifest
    assert len(opened) == 3
    assert all((root / CELESTE_NAME / f).read_bytes() == b for f, b in CELESTE_FILES.items())


def test_a_yes_uses_the_download_hook_the_cli_passes(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, opened = _celeste_world(monkeypatch, tmp_path)
    drawn: list[str] = []
    roster = build_roster(
        Settings(llm_provider="fermi", device="cpu"),
        store,
        root,
        confirm_download=lambda name, size: True,
        download=drawn.append,
    )
    with pytest.raises(ConfigError):
        _ = _celeste_client(store, roster.plan).decoder
    assert drawn == [CELESTE_NAME] and opened == []  # the hook did the downloading


def test_a_no_skips_celeste_and_is_not_asked_again(
    store: DataStore, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root, opened = _celeste_world(monkeypatch, tmp_path)
    asked: list[str] = []

    def no(name: str, size: int) -> bool:
        asked.append(name)
        return False

    settings = Settings(llm_provider="fermi", device="cpu", escalations=3)
    roster = build_roster(settings, store, root, confirm_download=no)
    celeste_tries = [c for c in roster.plan if getattr(c, "name", "") == CELESTE_NAME]
    assert len(celeste_tries) == 3
    for client in celeste_tries:  # every escalation try, one question
        with pytest.raises(DownloadDeclinedError):
            _ = _celeste_client(store, (client,)).decoder
    assert asked == [CELESTE_NAME]
    assert opened == [] and not (root / CELESTE_NAME).exists()


def test_a_declined_download_is_not_a_failed_download(store: DataStore, tmp_path: Path) -> None:
    def declined() -> None:
        raise DownloadDeclinedError("not downloaded")

    client = FermiClient(LUNA.name, store, directory=tmp_path / "x", device="cpu", fetch=declined)
    with pytest.raises(DownloadDeclinedError):
        client.complete_json(
            system="", user=json.dumps({"question": QUESTION}), schema=Classification
        )
