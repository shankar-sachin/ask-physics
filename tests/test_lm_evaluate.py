import json
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from askphysics import cli
from askphysics.data.loader import DataStore
from askphysics.lm.checkpoints import save_model
from askphysics.lm.config import LUNA
from askphysics.lm.evaluate import (
    evaluate_tasks,
    route_plan,
    sample_examples,
    score_classification,
    score_plan,
)
from askphysics.lm.factory import build_dataset, read_examples
from askphysics.lm.generate import Decoder
from askphysics.lm.model import FermiLM
from askphysics.lm.train import train_tokenizer
from askphysics.models import Classification, KnownValue, Plan


def _plan(**changes: object) -> Plan:
    base: dict[str, object] = {
        "equation_ids": ["kin_v_at"],
        "target": "v",
        "unknowns": ["v"],
        "known_values": [
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=2, unit="m/s^2", origin="given"),
            KnownValue(symbol="t", value=10, unit="s", origin="given"),
        ],
        "assumptions": [],
        "strategy": "Solve for v.",
    }
    return Plan.model_validate(base | changes)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    build_dataset(root, 400, seed=1)
    return root


def test_a_gold_plan_scores_perfectly(store: DataStore) -> None:
    score = score_plan(_plan(), _plan(), store)
    assert score.equation and score.target and score.knowns and score.answer


def test_a_wrong_number_gives_a_wrong_answer(store: DataStore) -> None:
    wrong = _plan(
        known_values=[
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=3, unit="m/s^2", origin="given"),
            KnownValue(symbol="t", value=10, unit="s", origin="given"),
        ]
    )
    score = score_plan(wrong, _plan(), store)
    assert score.equation and score.target
    assert not score.knowns and not score.answer


def test_equivalent_units_still_give_the_right_answer(store: DataStore) -> None:
    minutes = _plan(
        known_values=[
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=2, unit="m/s^2", origin="given"),
            KnownValue(symbol="t", value=10 / 60, unit="min", origin="given"),
        ]
    )
    score = score_plan(minutes, _plan(), store)
    assert not score.knowns and score.answer


def test_wrong_equation_or_target(store: DataStore) -> None:
    other = _plan(equation_ids=["kin_x_at"], target="x", unknowns=["x"])
    score = score_plan(other, _plan(), store)
    assert not score.equation and not score.target and not score.answer


def test_classification_scores_the_category() -> None:
    gold = Classification(category="fermi", reasoning="Estimate.", domains=[])
    assert score_classification(gold.model_copy(update={"reasoning": "Other words."}), gold)
    assert not score_classification(gold.model_copy(update={"category": "standard"}), gold)


def test_sample_is_fixed_and_per_task(dataset: Path) -> None:
    a = sample_examples(read_examples(dataset / "val"), per_task=4, seed=3)
    b = sample_examples(read_examples(dataset / "val"), per_task=4, seed=3)
    assert a == b
    assert {e.task for e in a} == {"classify", "plan"}
    assert all(sum(e.task == t for e in a) <= 4 for t in ("classify", "plan"))


def test_evaluate_runs_an_untrained_model(store: DataStore, dataset: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = sample_examples(read_examples(dataset / "val"), per_task=2)
    seen: list[int] = []
    report = evaluate_tasks(decoder, picked, store, on_progress=seen.append)
    assert report.classify_examples == 2 and report.plan_examples == 2
    assert seen == [1, 2, 3, 4]
    for rate in (report.category_accuracy, report.equation_accuracy, report.valid_plan_rate):
        assert 0.0 <= rate <= 1.0
    assert json.loads(report.to_json())["plan_examples"] == 2
    routed = (report.routed_right_rate, report.flagged_wrong_rate, report.confidently_wrong_rate)
    assert sum(routed) == pytest.approx(1.0)
    assert report.mean_tries == pytest.approx(1.0)  # one attempt by default


def test_the_router_retries_in_the_eval(store: DataStore, dataset: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = [
        e for e in sample_examples(read_examples(dataset / "val"), per_task=3) if e.task == "plan"
    ]
    report = evaluate_tasks(decoder, picked, store, attempts=3)
    assert report.attempts == 3
    assert 1.0 <= report.mean_tries <= 3.0
    routed = (report.routed_right_rate, report.flagged_wrong_rate, report.confidently_wrong_rate)
    assert sum(routed) == pytest.approx(1.0)


def test_cli_eval(dataset: Path, tmp_path: Path) -> None:
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    save_model(FermiLM(LUNA), tokenizer, tmp_path / "luna")
    r = CliRunner().invoke(
        cli.app,
        [
            "model",
            "eval",
            "--model",
            "fermi-luna-1",
            "--directory",
            str(tmp_path / "luna"),
            "--data",
            str(dataset),
            "--examples",
            "1",
            "--device",
            "cpu",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "valid plan" in r.output
    assert "confidently wrong" in r.output
    assert json.loads((tmp_path / "luna" / "eval.json").read_text())["plan_examples"] == 1
    r = CliRunner().invoke(cli.app, ["model", "eval", "--directory", str(tmp_path / "none")])
    assert r.exit_code == 1


def _resistor_payload(question: str, store: DataStore) -> dict[str, object]:
    return {
        "question": question,
        "category": "standard",
        "equations": [{"id": "series_resistors"}, {"id": "parallel_resistors"}],
        "constants": [],
    }


def test_the_eval_never_counts_an_impossible_answer(store: DataStore) -> None:
    series = _plan(
        equation_ids=["series_resistors"],
        target="R1",
        unknowns=["R1"],
        known_values=[
            KnownValue(symbol="R", value=1.3, unit="ohm", origin="given"),
            KnownValue(symbol="R2", value=10.1, unit="ohm", origin="given"),
        ],
    )
    q = "In series: total 1.3 ohm, resistance 2 is 10.1 ohm. Find resistance 1."
    decoder = Decoder.__new__(Decoder)  # never called: the first attempt is given
    routed = route_plan(decoder, _resistor_payload(q, store), store, 1, first=series)
    assert routed.plan is None and routed.answer is None and not routed.possible
    flagged = route_plan(
        decoder,
        _resistor_payload(f"{q} The battery is 9 V.", store),
        store,
        1,
        first=_plan(
            equation_ids=["parallel_resistors"],
            target="R1",
            unknowns=["R1"],
            known_values=series.known_values,
        ),
    )
    assert flagged.plan is not None and flagged.possible and not flagged.passed
