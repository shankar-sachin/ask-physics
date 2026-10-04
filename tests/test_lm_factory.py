import json
from pathlib import Path

import pytest

from askphysics.data.loader import DataStore
from askphysics.lm import templates as tpl
from askphysics.lm.factory import (
    LEAK_THRESHOLD,
    DataFactory,
    Example,
    build_dataset,
    equation_phrase,
    load_blocklist,
    read_examples,
    word_overlap,
)
from askphysics.lm.formats import (
    explain_numbers,
    extract_numbers,
    format_number,
    plan_numbers,
    plan_units,
)
from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, Plan
from askphysics.solver.symbolic import solve_for
from askphysics.solver.units import quantity

EVALS = Path(__file__).resolve().parents[1] / "evals" / "questions.yaml"


@pytest.fixture(scope="module")
def examples(store: DataStore) -> list[Example]:
    factory = DataFactory(store, seed=11, blocklist=load_blocklist(EVALS))
    return list(factory.examples(600))


def _payload(prompt: str, token: str) -> dict:  # type: ignore[type-arg]
    assert prompt.startswith(token)
    return json.loads(prompt[len(token) :])


def test_deterministic_for_a_seed(store: DataStore) -> None:
    a = [e.to_json() for e in DataFactory(store, seed=3).examples(40)]
    b = [e.to_json() for e in DataFactory(store, seed=3).examples(40)]
    c = [e.to_json() for e in DataFactory(store, seed=4).examples(40)]
    assert a == b
    assert a != c


def test_all_tasks_and_splits_present(examples: list[Example]) -> None:
    assert {e.task for e in examples} == {"classify", "plan", "explain"}
    assert {e.split for e in examples} == {"train", "val"}


def test_held_out_templates_never_train(examples: list[Example]) -> None:
    for e in examples:
        if any(part.endswith("_h") for part in e.template.split("+")):
            assert e.split == "val", e.template


def test_plan_targets_are_decodable_and_correct(store: DataStore, examples: list[Example]) -> None:
    plans = [e for e in examples if e.task == "plan"]
    assert len(plans) > 100
    for e in plans:
        payload = _payload(e.prompt, PLAN)
        assert e.target.endswith(END)
        plan = Plan.model_validate(json.loads(e.target[: -len(END)]))
        prompt_ids = [eq["id"] for eq in payload["equations"]]
        assert set(plan.equation_ids) <= set(prompt_ids)
        consts = [store.constants[c["name"]] for c in payload["constants"]]
        eqs = [store.equations[i] for i in prompt_ids]
        numbers = set(plan_numbers(payload["question"], consts))
        units = set(plan_units(payload["question"], eqs, consts))
        for k in plan.known_values:
            assert format_number(k.value) in numbers, (k, payload["question"])
            assert k.unit in units
        for text in [*plan.assumptions, plan.strategy]:
            assert set(extract_numbers(text)) <= numbers, text
        # The gold plan actually solves.
        eq = store.equations[plan.equation_ids[0]]
        knowns = {k.symbol: quantity(k.value, k.unit) for k in plan.known_values}
        assert solve_for(eq, plan.target, knowns).value.magnitude > 0


def test_classify_targets(examples: list[Example]) -> None:
    seen = set()
    for e in (x for x in examples if x.task == "classify"):
        _payload(e.prompt, CLASSIFY)
        c = Classification.model_validate(json.loads(e.target[: -len(END)]))
        seen.add(c.category)
        assert (c.closest_answerable is not None) == (c.category == "out_of_scope")
        assert not extract_numbers(c.reasoning)
    assert seen == {"standard", "fermi", "out_of_scope"}


def test_explain_targets_only_use_allowed_numbers(examples: list[Example]) -> None:
    for e in (x for x in examples if x.task == "explain"):
        payload = _payload(e.prompt, EXPLAIN)
        allowed = set(
            explain_numbers(payload["question"], payload["result"]["value"], payload["assumptions"])
        )
        text = e.target[: -len(END)]
        assert set(extract_numbers(text)) <= allowed, text
        assert f"[{payload['equations'][0]['id']}]" in text


def test_no_eval_leakage(examples: list[Example]) -> None:
    blocked = load_blocklist(EVALS)
    assert len(blocked) == 8
    for e in examples:
        question = json.loads(e.prompt.split("{", 1)[1].join(["{", ""]))["question"]
        assert all(word_overlap(question, b) < LEAK_THRESHOLD for b in blocked)


def test_blocklist_drops_near_duplicates(store: DataStore) -> None:
    factory = DataFactory(
        store, seed=0, blocklist=["How many tennis balls would it take to fill a"]
    )
    questions = [
        json.loads(e.prompt[len(CLASSIFY) :])["question"]
        for e in factory.examples(300)
        if e.task == "classify"
    ]
    assert not any("tennis balls" in q and "fill" in q for q in questions)
    assert factory.dropped > 0


def test_templates_format_cleanly() -> None:
    slots = {k: v[0] for k, v in {**tpl.FERMI_SLOTS, **tpl.OOS_SLOTS}.items()}
    for t in tpl.FERMI:
        t.text.format(**slots)
    for t, reason, closest in tpl.OUT_OF_SCOPE:
        t.text.format(**slots)
        assert not extract_numbers(reason.format(**slots) + closest.format(**slots))
    for reasoning in (*tpl.STANDARD_REASONING, *tpl.FERMI_REASONING):
        assert not extract_numbers(reasoning)
    ids = [t.id for t in (*tpl.GENERIC, *tpl.FERMI, *tpl.EXPLAIN)]
    assert len(ids) == len(set(ids))


def test_equation_phrase() -> None:
    assert equation_phrase("Newton's second law") == "Newton's second law"
    assert equation_phrase("Ohm's law") == "Ohm's law"
    assert equation_phrase("Ideal gas law") == "the ideal gas law"


def test_word_overlap() -> None:
    assert word_overlap("a b c", "a b c") == 1.0
    assert word_overlap("a b", "c d") == 0.0
    assert word_overlap("", "") == 0.0


def test_build_and_read_dataset(tmp_path: Path) -> None:
    manifest = build_dataset(tmp_path, 120, seed=5, workers=1, blocklist=["x y z"])
    assert sum(manifest["counts"].values()) == 120
    assert json.loads((tmp_path / "manifest.json").read_text())["seed"] == 5
    train = list(read_examples(tmp_path / "train"))
    val = list(read_examples(tmp_path / "val"))
    assert len(train) + len(val) == 120
    assert all(e.split == "train" for e in train)


def test_load_blocklist(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text('- id: a\n  question: "Is this blocked?"\n  category: standard\n')
    assert load_blocklist(path) == ["Is this blocked?"]
    assert load_blocklist(tmp_path / "missing.yaml") == []
