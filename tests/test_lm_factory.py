import json
import re
from collections import Counter
from dataclasses import replace
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
    plain_name,
    read_examples,
    word_overlap,
)
from askphysics.lm.formats import (
    explain_numbers,
    extract_numbers,
    extract_units,
    format_number,
    plan_numbers,
    plan_units,
    stated_quantities,
)
from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, Plan
from askphysics.solver.symbolic import solve_for
from askphysics.solver.units import check_dimensions, quantity

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


def test_given_values_are_written_in_the_question(examples: list[Example]) -> None:
    for e in (x for x in examples if x.task == "plan"):
        question = _payload(e.prompt, PLAN)["question"]
        written = set(stated_quantities(question))
        plan = Plan.model_validate(json.loads(e.target[: -len(END)]))
        for k in plan.known_values:
            if k.origin == "given":
                assert (format_number(k.value), k.unit) in written, (question, k)


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
    free_text = (
        *tpl.STANDARD_REASONING,
        *tpl.FERMI_REASONING,
        *tpl.STRATEGIES,
        *tpl.PREAMBLES,
        *tpl.SIGN_OFFS,
        *tpl.ASSUME_LEADS,
    )
    for text in free_text:
        assert not any(ch.isdigit() for ch in text), text
    generic = {
        "target": "t",
        "Target": "T",
        "tsym": "x",
        "knowns": "k",
        "Knowns": "K",
        "facts": "F.",
    }
    for t in tpl.GENERIC:
        t.text.format(**generic)
    for t in tpl.KNOWN_PATTERNS:
        t.text.format(name="mass", a_name="a mass", sym="m", q="2 kg")
    ids = [
        t.id
        for t in (
            *tpl.GENERIC,
            *tpl.KNOWN_PATTERNS,
            *tpl.FERMI,
            *tpl.EXPLAIN,
            *(s.template for s in tpl.SCENARIOS),
            *(o[0] for o in tpl.OUT_OF_SCOPE),
        )
    ]
    assert len(ids) == len(set(ids))
    for family in (tpl.GENERIC, tpl.KNOWN_PATTERNS, tpl.FERMI, tpl.EXPLAIN):
        assert any(t.held_out for t in family) and not all(t.held_out for t in family)
    assert any(s.template.held_out for s in tpl.SCENARIOS)


def test_scenarios_match_their_equations(store: DataStore) -> None:
    for sc in tpl.SCENARIOS:
        eq = store.equations[sc.equation]
        symbols = {v.symbol for v in eq.variables}
        assert sc.target in symbols, sc.template.id
        assert {f[0] for f in sc.forced} <= symbols - {sc.target}, sc.template.id
        fields = set(re.findall(r"{(\w+)}", sc.template.text))
        given = {s.removesuffix("_a") for s in fields} - {
            "object",
            "object2",
            "vehicle",
            "vehicle2",
        }
        assert given <= symbols - {sc.target}, sc.template.id
        assert not extract_numbers(" ".join(sc.assumptions)), sc.template.id


def test_every_scenario_produces_a_solved_problem(
    store: DataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    factory = DataFactory(store, seed=2)
    for sc in tpl.SCENARIOS:
        monkeypatch.setattr(tpl, "SCENARIOS", (sc,))
        problems = [factory._scenario_problem() for _ in range(40)]
        assert any(p is not None for p in problems), sc.template.id


def test_unit_spellings_are_copyable() -> None:
    for symbol, spellings in tpl.UNIT_SPELLINGS.items():
        for spelled in spellings:
            assert extract_units(f"It is 12 {spelled}.") == [spelled], spelled
            assert check_dimensions(quantity(1.0, spelled), symbol), spelled


def test_variable_synonyms_cover_the_database(store: DataStore) -> None:
    names = {v.name for eq in store.equations.values() for v in eq.variables}
    constants = {n for n in names if "constant" in n}
    assert names - constants <= tpl.VAR_SYNONYMS.keys()
    for name in names - constants:
        assert not any(ch.isdigit() for ch in plain_name(name)), name


def test_questions_are_diverse(examples: list[Example]) -> None:
    questions = [_payload(e.prompt, PLAN)["question"] for e in examples if e.task == "plan"]
    assert len(set(questions)) / len(questions) > 0.97
    templates = {e.template.split("+")[0] for e in examples}
    assert len(templates) > 60


def test_casual_dressing_keeps_symbols_intact(store: DataStore) -> None:
    factory = DataFactory(store, seed=0)
    for _ in range(200):
        assert factory._lower_first("V = 5 V. Find I.").startswith("V =")
        assert factory._lower_first("KE is known.").startswith("KE")
        assert factory._lower_first("I need the speed.").startswith("I ")
    assert factory._lower_first("A ball falls.") == "a ball falls."
    assert factory._lower_first("What is it?") == "what is it?"


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


def test_targets_are_balanced(store: DataStore) -> None:
    only_kinematics = replace(store, equations={"kin_v_at": store.equations["kin_v_at"]})
    factory = DataFactory(only_kinematics, seed=9)
    counts: dict[str, Counter[str]] = {}
    for _ in range(400):
        p = factory.standard_problem()
        if p is not None:
            counts.setdefault(p.equation.id, Counter())[p.plan.target] += 1
    kin = counts["kin_v_at"]
    total = sum(kin.values())
    assert set(kin) == {"v", "v0", "a", "t"}
    assert min(kin.values()) / total > 0.15, kin  # "find v0" is not a rarity


def test_out_of_scope_look_alikes_mirror_physics_phrasing() -> None:
    oos = {t.text.split(" {")[0] for t, _, _ in tpl.OUT_OF_SCOPE}
    fermi = {t.text.split(" {")[0] for t in tpl.FERMI}
    assert "How much energy is stored in a" in fermi
    assert {"How much does", "How much force does", "What is the momentum of"} <= oos


def test_every_suggested_question_is_trained_as_answerable(store: DataStore) -> None:
    redirects = {closest.split(" {")[0] for _, _, closest in tpl.OUT_OF_SCOPE}
    seen = Counter()
    for e in DataFactory(store, seed=21).examples(4000):
        if e.task == "classify" and e.template.startswith("redirect_"):
            c = Classification.model_validate_json(e.target[: -len(END)])
            question = json.loads(e.prompt[len(CLASSIFY) :])["question"]
            assert c.category == "fermi" and c.closest_answerable is None
            assert any(r.split(" {")[0][:20].lower() in question.lower() for r in redirects)
            seen[c.category] += 1
    assert seen["fermi"] > 50
