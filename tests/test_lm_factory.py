import itertools
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
    _substitutable,
    build_dataset,
    equation_phrase,
    load_blocklist,
    plain_name,
    read_examples,
    textbook_examples,
    word_overlap,
)
from askphysics.lm.formats import (
    classify_prompt,
    explain_numbers,
    extract_numbers,
    extract_units,
    format_number,
    plan_numbers,
    plan_units,
    stated_quantities,
)
from askphysics.lm.reading import mentions, own_tags, twins
from askphysics.lm.tokenizer import CLASSIFY, END, EXPLAIN, PLAN
from askphysics.models import Classification, Plan
from askphysics.normalize import normalize_question
from askphysics.pipeline import compute
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
        # The gold plan actually solves, chaining its equations if it cites two.
        assert compute(plan, data=store).value > 0


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
        assert not extract_numbers(reason.format(**slots))
        closest.format(**slots)
    for est in tpl.ESTIMATES:
        assert not extract_numbers(est.template.text), est.template.id
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
            *(e.template for e in tpl.ESTIMATES),
        )
    ]
    assert len(ids) == len(set(ids))
    assert any(e.template.held_out for e in tpl.ESTIMATES)
    assert not all(e.template.held_out for e in tpl.ESTIMATES)
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


OPENSTAX = Path(__file__).resolve().parents[1] / "third_party" / "openstax-physics"


def test_textbook_problems_become_standard_classify_examples(tmp_path: Path) -> None:
    rows = [
        {"id": "a", "chapter": "Acceleration", "kind": "problems",
         "text": "A 2,000-kg car speeds up at 3 m/s² for 4 s. How fast is it going?"},
        {"id": "b", "chapter": "Acceleration", "kind": "concept",
         "text": "Why does a 2 kg ball fall as fast as a 4 kg one?"},
        {"id": "c", "chapter": "Acceleration", "kind": "problems",
         "text": "Using the graph, what is the acceleration at 4 s?"},
        {"id": "d", "chapter": "What is Physics?", "kind": "problems",
         "text": "A scale reads 65 kg with 3 percent uncertainty. What is the uncertainty?"},
        {"id": "e", "chapter": "Momentum", "kind": "problems",
         "text": "What is the momentum of a 2 kg ball moving at 3 m/s?"},
        {"id": "f", "chapter": "Acceleration", "kind": "short-answer",
         "text": "What is acceleration?"},
    ]  # fmt: skip
    path = tmp_path / "questions.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    examples = textbook_examples(path, exclude={"e"}, repeats=2)
    assert len(examples) == 2 and len({e.target for e in examples}) == 2  # varied reasoning
    for e in examples:
        assert (e.task, e.split, e.template) == ("classify", "train", "openstax")
        # The prompt is what the pipeline's classifier sees: the normalized question.
        assert e.prompt == classify_prompt(
            "A 2000 kg car speeds up at 3 m/s^2 for 4 s. How fast is it going?"
        )
        c = json.loads(e.target.removesuffix(END))
        assert c["category"] == "standard" and c["domains"] == ["kinematics"]
    build_dataset(tmp_path / "data", 20, seed=1, extra=examples)
    manifest = json.loads((tmp_path / "data" / "manifest.json").read_text())
    assert manifest["textbook_examples"] == 2
    assert (
        sum(1 for e in read_examples(tmp_path / "data" / "train") if e.template == "openstax") == 2
    )


def test_the_real_question_eval_never_becomes_training_data() -> None:
    lines = (OPENSTAX / "real_eval.jsonl").read_text().splitlines()
    held_out = {json.loads(line)["id"] for line in lines}
    examples = textbook_examples(OPENSTAX / "questions.jsonl", exclude=held_out, repeats=1)
    assert len(examples) > 100
    prompts = {e.prompt for e in examples}
    for line in lines:
        question = normalize_question(json.loads(line)["question"])
        assert classify_prompt(question) not in prompts


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
            assert c.category == "standard" and c.closest_answerable is None
            assert c.domains
            assert any(r[:20].lower() in question.lower() for r in redirects)
            seen[c.category] += 1
    assert seen["standard"] > 50


def test_every_redirect_is_answerable(store: DataStore) -> None:
    """Gold path for each redirect: a standard label, retrieval finds the equation, and a
    plan that Noether solves. Fails if any redirect isn't answerable."""
    factory = DataFactory(store, seed=2)
    checked = 0
    for template, _, closest in tpl.OUT_OF_SCOPE:
        keys = re.findall(r"{(\w+)}", closest)
        for combo in itertools.product(*(tpl.OOS_SLOTS[k] for k in keys)):
            redirect = closest.format(**dict(zip(keys, combo, strict=True)))
            label = factory.redirect_label(redirect)
            assert label is not None, (template.id, redirect)
            assert label.category == "standard" and label.closest_answerable is None
            problem = factory.stated_answer(redirect)
            assert problem is not None, (template.id, redirect)
            assert problem.equation.id in {e.id for e in problem.retrieved}, redirect
            assert compute(problem.plan, data=store).value > 0, redirect
            checked += 1
    assert checked >= len(tpl.OUT_OF_SCOPE)


def _classify_labels(factory: DataFactory, ids: set[str], cap: int = 20000) -> dict[str, set[str]]:
    """The categories each template in ``ids`` is trained with, from classify examples.

    Draws until every id has been seen (or ``cap`` draws), so a rare template is still found.
    """
    labels: dict[str, set[str]] = {}
    for _ in range(cap):
        if ids <= labels.keys():
            break
        e = factory.classify_example()
        if e is None or e.template not in ids:
            continue
        c = Classification.model_validate_json(e.target[: -len(END)])
        labels.setdefault(e.template, set()).add(c.category)
    return labels


def test_estimates_are_solved_fermi_questions(store: DataStore) -> None:
    factory = DataFactory(store, seed=4)
    for est in tpl.ESTIMATES:
        assert factory._estimate_solves(est), est.template.id
        assert not any(ch.isdigit() for ch in est.template.text), est.template.id
    estimate_ids = {e.template.id for e in tpl.ESTIMATES}
    labels = _classify_labels(factory, estimate_ids)
    assert set(labels) == estimate_ids
    for tid, categories in labels.items():
        assert categories == {"fermi"}, tid
    for e in factory.examples(3000):
        if e.task == "classify" and e.template in estimate_ids:
            c = Classification.model_validate_json(e.target[: -len(END)])
            assert c.category == "fermi" and c.closest_answerable is None
            question = json.loads(e.prompt[len(CLASSIFY) :])["question"]
            assert not any(ch.isdigit() for ch in question), question


def test_contrast_pairs_share_wording(store: DataStore) -> None:
    """Each pair shares its frame. The out-of-scope side has no physical subject; the Fermi
    side is a physical thing the tables answer. Both kinds are trained, so the boundary is
    learned from both sides."""
    oos = {t.id: t.text for t, _, _ in tpl.OUT_OF_SCOPE}
    est = {e.template.id: e.template.text for e in tpl.ESTIMATES}
    assert oos["oos_heavy_01"] == "How heavy is {abstract}?"
    assert est["est_weight_train_02"] == "How heavy is a freight train?"
    assert oos["oos_speed_03"] == "How fast is {emotion}?"
    assert est["est_takeoff_01"] == "How fast is an adult at takeoff in a standing jump?"
    assert oos["oos_speed_02_h"] == "How fast does {abstract} travel?"
    assert est["est_momentum_train_03"] == "What is the momentum of a freight train?"
    assert oos["oos_momentum_01"] == "What is the momentum of {abstract}?"
    # Plain property lookups (issue #92): "How fast is sound?" next to "How fast is sadness?".
    assert est["est_sound_01"] == "How fast is sound?"
    assert est["est_weight_car_01"] == "How heavy is a car?"
    assert oos["oos_speed_04"] == "What is the speed of {abstract}?"
    assert "sadness" in tpl.OOS_SLOTS["emotion"]
    # The held-out Fermi phrasings must not be the only trained form of a contrast.
    trained = {e.template.id for e in tpl.ESTIMATES if not e.template.held_out}
    assert {"est_weight_train_02", "est_takeoff_01", "est_momentum_train_03"} <= trained
    assert {"est_sound_01", "est_sound_02", "est_weight_car_01", "est_weight_car_02"} <= trained
    wanted = {"oos_heavy_01", "oos_speed_03", "oos_momentum_01", "oos_speed_02_h"}
    wanted |= {"est_weight_train_02", "est_takeoff_01", "est_momentum_train_03"}
    wanted |= {"oos_speed_04", "est_sound_01", "est_weight_car_01"}
    labels = _classify_labels(DataFactory(store, seed=21), wanted)
    assert set(labels) == wanted
    for tid in (
        "oos_heavy_01",
        "oos_speed_03",
        "oos_momentum_01",
        "oos_speed_02_h",
        "oos_speed_04",
    ):
        assert labels[tid] == {"out_of_scope"}, tid
    for tid in (
        "est_weight_train_02",
        "est_takeoff_01",
        "est_momentum_train_03",
        "est_sound_01",
        "est_weight_car_01",
    ):
        assert labels[tid] == {"fermi"}, tid


def test_number_free_questions_avoid_arithmetic_words() -> None:
    """A number-free physical question must not read like the arithmetic refusals, or the
    classifier learns "no numbers means arithmetic" and refuses real physics."""
    arithmetic = {
        "divided", "times", "plus", "minus", "squared", "square", "root", "derivative",
        "integral", "slope", "solve", "evaluate", "compute", "prime", "calculus", "algebra",
        "arithmetic", "homework",
    }  # fmt: skip
    for est in tpl.ESTIMATES:
        words = set(re.findall(r"[a-z]+", est.template.text.lower()))
        assert not words & arithmetic, est.template.text


def test_questions_for_twins_say_which(store: DataStore, examples: list[Example]) -> None:
    # Series and parallel resistors have identical variables: the words must decide.
    seen = 0
    for e in (x for x in examples if x.task == "plan"):
        plan = Plan.model_validate(json.loads(e.target[: -len(END)]))
        eq = store.equations[plan.equation_ids[0]]
        question = _payload(e.prompt, PLAN)["question"]
        for twin in twins(eq, store.equations.values()):
            seen += 1
            assert mentions(question, own_tags(eq, twin)), question
    assert seen > 0


def test_chained_problems_need_both_equations(store: DataStore) -> None:
    factory = DataFactory(store, seed=12)
    for chain in tpl.CHAINS:
        problems = [p for p in (factory.chained_problem(chain) for _ in range(30)) if p]
        assert problems, chain
        for p in problems:
            assert p.plan.equation_ids == [chain.then, chain.first]
            assert p.plan.unknowns == [chain.target, chain.via]
            assert chain.via not in {k.symbol for k in p.plan.known_values}
            assert {chain.then, chain.first} <= {e.id for e in p.retrieved}
            result = compute(p.plan, data=store)
            assert [s.equation_id for s in result.steps] == [chain.first, chain.then]
            assert result.value == pytest.approx(p.value, rel=1e-5)
            assert p.template.startswith("chain+")


def test_some_plan_examples_chain(examples: list[Example]) -> None:
    plans = [e for e in examples if e.task == "plan"]
    chained = [e for e in plans if e.template.startswith("chain+")]
    assert 0.02 < len(chained) / len(plans) < 0.2


def test_plain_property_questions_have_table_answers(store: DataStore) -> None:
    """The speed of sound and a car's weight come from the tables, through Noether."""
    factory = DataFactory(store, seed=1)
    by_id = {e.template.id: e for e in tpl.ESTIMATES}
    sound = by_id["est_sound_01"]
    eq = store.equations[sound.equation]
    outcome = solve_for(eq, sound.target, _substitutable(factory._estimate_knowns(sound), [eq]))
    assert outcome.value.to("m/s").magnitude == pytest.approx(343, rel=0.01)
    car = by_id["est_weight_car_01"]
    eq = store.equations[car.equation]
    outcome = solve_for(eq, car.target, _substitutable(factory._estimate_knowns(car), [eq]))
    assert outcome.value.to("N").magnitude == pytest.approx(1500 * 9.80665)
