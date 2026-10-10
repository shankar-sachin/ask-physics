import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import pint
import pytest
import torch
from typer.testing import CliRunner

from askphysics import cli
from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import PlanValidationError
from askphysics.llm.base import Roster
from askphysics.lm.checkpoints import save_model
from askphysics.lm.config import LUNA
from askphysics.lm.evaluate import (
    EvalTick,
    RealQuestion,
    evaluate_real,
    evaluate_tasks,
    read_real_questions,
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
from askphysics.normalize import normalize_question
from askphysics.pipeline import Pipeline, compute, sanity_check
from askphysics.retrieval.keyword import KeywordRetriever
from askphysics.solver.units import Quantity, quantity

OPENSTAX = Path(__file__).resolve().parents[1] / "third_party" / "openstax-physics"


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


def test_a_plan_that_cannot_be_written_is_a_miss_not_a_crash(
    store: DataStore, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_legal_value(*args: object, **kwargs: object) -> Plan:
        raise PlanValidationError("no legal value for m")

    monkeypatch.setattr("askphysics.lm.evaluate.decode_plan", no_legal_value)
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = sample_examples(read_examples(dataset / "val"), per_task=2)
    report = evaluate_tasks(decoder, picked, store)
    assert report.plan_examples == 2
    assert report.equation_accuracy == 0.0 and report.valid_plan_rate == 0.0


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
    real = tmp_path / "real.jsonl"
    real.write_text((OPENSTAX / "real_eval.jsonl").read_text().splitlines()[0] + "\n")
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
            "--attempts",
            "1",
            "--real",
            str(real),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "valid plan" in r.output
    assert "confidently wrong" in r.output
    assert "1 real textbook questions" in r.output
    report = json.loads((tmp_path / "luna" / "eval.json").read_text())
    assert report["plan_examples"] == 1 and report["real"]["questions"] == 1
    r = CliRunner().invoke(cli.app, ["model", "eval", "--directory", str(tmp_path / "none")])
    assert r.exit_code == 1


def test_cli_eval_with_a_rescuer(dataset: Path, tmp_path: Path) -> None:
    real = tmp_path / "real.jsonl"
    real.write_text((OPENSTAX / "real_eval.jsonl").read_text().splitlines()[0] + "\n")
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
            "--attempts",
            "1",
            "--real",
            str(real),
            "--rescue-with",
            "fermi-luna-1",
            "--rescue-directory",
            str(tmp_path / "luna"),
        ],
    )
    assert r.exit_code == 0, r.output
    assert "tried on" in r.output and "rescued" in r.output
    assert "right after escalation" in r.output
    report = json.loads((tmp_path / "luna" / "eval.json").read_text())
    assert report["rescue_tried"] <= report["plan_examples"]
    assert "routed_with_rescue_rate" in report


def test_the_rescue_fields_stay_zero_without_a_rescuer(store: DataStore, dataset: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = [
        e for e in sample_examples(read_examples(dataset / "val"), per_task=3) if e.task == "plan"
    ]
    report = evaluate_tasks(decoder, picked, store, attempts=2)
    assert report.plan_examples > 0
    assert report.rescue_tried == report.rescued == report.rescue_confidently_wrong == 0
    assert report.rescue_rate == report.routed_with_rescue_rate == 0.0


def test_a_rescue_run_retries_exactly_the_misses(store: DataStore, dataset: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = [
        e for e in sample_examples(read_examples(dataset / "val"), per_task=4) if e.task == "plan"
    ]
    report = evaluate_tasks(decoder, picked, store, attempts=2, rescue=decoder)
    n = report.plan_examples
    main_right = round(report.routed_right_rate * n)
    assert report.rescue_tried + main_right == n
    assert 0 <= report.rescued <= report.rescue_tried
    assert report.rescue_confidently_wrong <= report.rescue_tried - report.rescued
    assert report.rescue_rate == pytest.approx(
        report.rescued / report.rescue_tried if report.rescue_tried else 0.0
    )
    assert report.routed_with_rescue_rate == pytest.approx((main_right + report.rescued) / n)
    assert report.routed_with_rescue_rate >= report.routed_right_rate
    assert json.loads(report.to_json())["rescue_tried"] == report.rescue_tried


def test_ticks_report_each_example_then_the_rescue_pass_over_just_the_misses(
    store: DataStore, dataset: Path
) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = sample_examples(read_examples(dataset / "val"), per_task=3)
    ticks: list[EvalTick] = []
    progress: list[int] = []
    report = evaluate_tasks(
        decoder, picked, store, attempts=2, rescue=decoder, on_tick=ticks.append,
        on_progress=progress.append,
    )  # fmt: skip
    scoring = [t for t in ticks if t.stage == "score"]
    assert [t.done for t in scoring] == list(range(1, len(picked) + 1)) == progress
    assert {t.total for t in scoring} == {len(picked)}
    assert [t.task for t in scoring] == [e.task for e in picked]
    last = scoring[-1]
    assert last.plans == report.plan_examples
    assert last.valid == round(report.valid_plan_rate * report.plan_examples)
    assert last.confident == round(report.confidently_wrong_rate * report.plan_examples)
    # The rescue pass starts after scoring, at 0 of the number of misses, and counts up to it.
    rescue = [t for t in ticks if t.stage == "rescue"]
    assert ticks[len(scoring) :] == rescue
    assert rescue[0].done == 0 and {t.total for t in rescue} == {report.rescue_tried}
    assert [t.done for t in rescue] == list(range(report.rescue_tried + 1))
    assert rescue[-1].rescued == report.rescued
    assert rescue[-1].rescue_confident == report.rescue_confidently_wrong


def test_without_a_rescuer_there_are_only_scoring_ticks(store: DataStore, dataset: Path) -> None:
    torch.manual_seed(0)
    tokenizer = train_tokenizer(dataset, vocab_size=LUNA.vocab_size)
    decoder = Decoder(FermiLM(LUNA).eval(), tokenizer, max_slot_tokens=8)
    picked = sample_examples(read_examples(dataset / "val"), per_task=2)
    ticks: list[EvalTick] = []
    evaluate_tasks(decoder, picked, store, on_tick=ticks.append)
    assert {t.stage for t in ticks} == {"score"} and len(ticks) == len(picked)


def test_eval_sh_passes_the_rescuer_through() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval.sh"
    env = {**os.environ, "DRY_RUN": "1"}
    with_rescue = subprocess.run(
        ["sh", str(script), "fermi-solem-1", "--rescue-with", "fermi-celeste-1"],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    assert "--rescue-with fermi-celeste-1" in with_rescue.stdout
    plain = subprocess.run(
        ["sh", str(script), "fermi-solem-1"], capture_output=True, text=True, env=env, check=True
    )
    assert "--rescue-with" not in plain.stdout


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


# --------------------------------------------------------------------------- real questions

_OPTION = re.compile(
    r"(-?\s?\d{1,3}(?:,\s?\d{3})+(?:\.\d+)?|-?\s?\d+(?:\.\d+)?)(?:\s*x\s*10\^(-?\d+))?"
    r"\s*([A-Za-z][A-Za-z/^\-\d ]*?)?\s*(?:[,.]|$|or)"
)


def _option(text: str) -> Quantity | None:
    """The first quantity in a book's answer option: "1.2 x 10^-4 N, and ..." is 1.2e-4 N."""
    match = _OPTION.search(text)
    if match is None:
        return None
    value = float(re.sub(r"[,\s]", "", match.group(1))) * 10 ** int(match.group(2) or 0)
    try:
        return quantity(value, (match.group(3) or "").strip() or "dimensionless")
    except Exception:  # an option that is a sentence, not a quantity
        return None


def _distance(got: Quantity, option: Quantity | None, signed: bool) -> float:
    """How far apart two quantities are, as |log ratio|; inf if they can't be compared."""
    if option is None:
        return math.inf
    try:
        x, y = float(got.to(option.units).magnitude), float(option.magnitude)
    except pint.DimensionalityError:
        return math.inf
    if not signed:  # the book often gives a magnitude for a signed answer
        x, y = abs(x), abs(y)
    if x == 0 or y == 0 or (x > 0) != (y > 0):
        return math.inf
    return abs(math.log(x / y))


def _real_rows() -> list[dict[str, Any]]:
    lines = (OPENSTAX / "real_eval.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_every_gold_plan_picks_the_books_answer(store: DataStore) -> None:
    """Each gold plan's answer is within 5% of the book's answer and of no other option.

    5%, not 2%: the book rounds to two figures (225 V printed as 2.3 x 10^2 V).
    """
    rows = _real_rows()
    assert len(rows) >= 40
    book = {
        q["id"]: q for q in map(json.loads, (OPENSTAX / "questions.jsonl").read_text().splitlines())
    }
    for row in rows:
        source = book[row["id"]]
        assert (row["question"], row["options"]) == (source["text"], source["options"])
        result = compute(Plan.model_validate(row["plan"]), data=store)
        got = quantity(result.value, result.unit)
        options = [_option(o) for o in row["options"]]
        for signed in (True, False):
            near = sorted(range(len(options)), key=lambda i: _distance(got, options[i], signed))
            if _distance(got, options[near[0]], signed) < math.log(1.05):
                break
        assert "abcd"[near[0]] == row["answer"], (row["id"], got, row["options"])
        assert _distance(got, options[near[0]], signed) < math.log(1.05), row["id"]
        assert _distance(got, options[near[1]], signed) >= math.log(1.05), row["id"]


def test_gold_plans_pass_the_sanity_checks_ask_runs(store: DataStore) -> None:
    """A right plan is never refused, and only a few are flagged (light's speed is far
    outside a speed's everyday range; "deuterium (2 H)" reads as 2 henries)."""
    flagged = []
    for row in _real_rows():
        p = Plan.model_validate(row["plan"])
        question = normalize_question(row["question"])
        report = sanity_check(p, compute(p, data=store), data=store, question=question)
        assert report.possible, row["id"]
        if not report.passed:
            flagged.append(row["id"])
    assert len(flagged) <= 3, flagged


class _Scripted:
    """Calls every question standard and plans it with the plan it was given."""

    name = "scripted"

    def __init__(self, plans: dict[str, Plan]) -> None:
        self.plans = plans

    def complete_json(self, *, system: str, user: str, schema: type[Any]) -> Any:
        if schema is Classification:
            return Classification(category="standard", reasoning="All values are given.")
        return self.plans[json.loads(user)["question"]]

    def complete_text(self, *, system: str, user: str) -> str:
        return ""


def _solve(store: DataStore, plans: dict[str, Plan]) -> Pipeline:
    client = _Scripted({normalize_question(q): p for q, p in plans.items()})
    return Pipeline(
        llm=client,
        retriever=KeywordRetriever(store.equations.values(), store.examples.values()),
        data=store,
        settings=Settings(),
        roster=Roster(classify=client, plan=(client,), explain=client),
    )


def test_a_perfect_planner_is_held_back_only_by_retrieval(store: DataStore) -> None:
    questions = read_real_questions(OPENSTAX / "real_eval.jsonl")
    pipeline = _solve(store, {q.question: q.plan for q in questions})
    report = evaluate_real(pipeline.solve, questions, store)
    assert report.questions == len(questions) and report.standard_rate == 1.0
    assert report.right_rate == report.retrieved_rate > 0.8
    assert report.confidently_wrong_rate == 0.0
    assert {m["stage"] for m in report.misses} <= {"retrieve"}


def test_a_wrong_number_is_a_miss_with_its_stage(store: DataStore) -> None:
    question = "A car starts from rest and accelerates at 2 m/s^2 for 10 s. What is its speed?"
    gold = _plan()
    wrong = _plan(
        known_values=[
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=2, unit="m/s^2", origin="given"),
            KnownValue(symbol="t", value=1, unit="s", origin="given"),
        ]
    )
    pipeline = _solve(store, {question: wrong})
    report = evaluate_real(pipeline.solve, [RealQuestion("q1", question, gold)], store)
    assert report.right_rate == 0.0 and report.retrieved_rate == 1.0
    assert report.flagged_wrong_rate == 1.0  # "10 s" went unused, so it was flagged
    assert report.misses[0]["stage"] == "wrong, flagged"
    assert report.misses[0]["got"].endswith("t=1 s")
