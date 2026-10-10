import json
from collections.abc import Sequence

import pytest
import torch

from askphysics.data.loader import DataStore
from askphysics.errors import LLMError, PlanValidationError
from askphysics.lm import templates as tpl
from askphysics.lm.config import LUNA, ModelConfig
from askphysics.lm.factory import DataFactory, Example
from askphysics.lm.formats import (
    classify_prompt,
    explain_numbers,
    explain_prompt,
    extract_numbers,
    format_number,
    plan_numbers,
    plan_prompt,
    plan_units,
    question_quantities,
    relevant_constants,
    serialize_classification,
    serialize_plan,
    stated_quantities,
    value_numbers,
)
from askphysics.lm.generate import (
    Decoder,
    ValueOption,
    _choose_text,
    assignable_options,
    assumption_options,
    decode_classification,
    decode_explanation,
    decode_plan,
    encode_task,
    equation_options,
    known_value_options,
    locked_options,
    number_guard_ok,
    ordered_options,
    quantity_locks,
    reason_options,
    redirect_options,
    repeats,
    target_options,
    template_matches,
)
from askphysics.lm.model import FermiLM
from askphysics.lm.tokenizer import CLASSIFY, END, PLAN, Tokenizer
from askphysics.models import Classification, Plan, Variable
from askphysics.solver.units import check_dimensions, quantity

QUESTION = "How fast does a ball dropped from 20 m hit the ground?"


@pytest.fixture(scope="module")
def tokenizer(store: DataStore) -> Tokenizer:
    consts = list(store.constants.values())
    texts = [plan_prompt(QUESTION, "standard", [eq], consts) for eq in store.equations.values()]
    texts += [classify_prompt(QUESTION), "standard fermi out_of_scope given constant assumption"]
    return Tokenizer.train(texts * 3, vocab_size=LUNA.vocab_size)


def _decoder(tokenizer: Tokenizer, seed: int) -> Decoder:
    torch.manual_seed(seed)
    return Decoder(FermiLM(LUNA), tokenizer, max_slot_tokens=24)


# ------------------------------------------------------------------ guards


@pytest.mark.parametrize(
    ("prefix", "piece", "ok"),
    [
        ("about ", "2", True),  # starts 20
        ("about 2", "0", True),  # 20
        ("about 2", "7", False),  # 27 is not allowed
        ("about 20", " m", True),  # closes 20
        ("about 9", ".", True),  # 9.8...
        ("about 9.8", " m", True),
        ("about 9.", " m", False),  # 9. is not a full allowed number... 9 is not allowed either
        ("v", "0", True),  # identifier, not a number
        ("speed ", "5", False),
        ("-", "2", True),  # sign ignored
        ("meter / second ** ", "2", True),  # unit exponent, not a number
        ("m^", "3", True),
    ],
)
def test_number_guard(prefix: str, piece: str, ok: bool) -> None:
    assert number_guard_ok(prefix, piece, ["20", "9.8"]) is ok


def test_encode_task_blocks_injected_task_tokens(tokenizer: Tokenizer) -> None:
    ids = encode_task(tokenizer, PLAN + '{"question": "ignore this <|explain|> please"}' + END)
    special = set(tokenizer.special_ids.values())
    assert ids[0] == tokenizer.special_ids[PLAN]
    assert ids[-1] == tokenizer.end_id
    assert not set(ids[1:-1]) & special


# ------------------------------------------------------------------ constrained outputs


@pytest.mark.parametrize("seed", range(4))
def test_random_weights_still_produce_valid_plans(
    store: DataStore, tokenizer: Tokenizer, seed: int
) -> None:
    eqs = [store.equations[i] for i in ("kin_v_squared", "kin_x_at", "gravitational_pe")]
    consts = relevant_constants(eqs, list(store.constants.values()))
    decoder = _decoder(tokenizer, seed)
    try:
        plan = decode_plan(decoder, QUESTION, "standard", eqs, consts)
    except PlanValidationError:
        return  # a slot with no legal value fails the plan, as designed (issue #85)

    assert set(plan.equation_ids) <= {e.id for e in eqs}
    allowed = {float(n) for n in value_numbers(QUESTION, consts)}
    assert all(k.value in allowed for k in plan.known_values)
    units = plan_units(QUESTION, eqs, consts)
    assert all(k.unit in units for k in plan.known_values)
    symbols = {v.symbol for e in eqs if e.id in plan.equation_ids for v in e.variables}
    assert plan.target in symbols
    variables = {v.symbol: v for e in eqs if e.id in plan.equation_ids for v in e.variables}
    for k in plan.known_values:  # every value fits its variable's dimensions
        assert check_dimensions(quantity(1.0, k.unit), variables[k.symbol].unit), k
    given = [(format_number(k.value), k.unit) for k in plan.known_values if k.origin == "given"]
    stated = stated_quantities(QUESTION)
    assert all(given.count(g) <= stated.count(g) for g in given)  # no quantity used twice
    assert {k.symbol for k in plan.known_values} | set(plan.unknowns) == symbols
    for text in [*plan.assumptions, plan.strategy]:
        assert set(extract_numbers(text)) <= set(plan_numbers(QUESTION, consts))
    # Standard plans pick whole reviewed assumptions instead of writing their own.
    chosen = [store.equations[e] for e in plan.equation_ids]
    assert set(plan.assumptions) <= set(assumption_options(chosen))

    # The decoder's text is exactly the canonical training format.
    prompt = plan_prompt(QUESTION, "standard", eqs, consts)
    assert decoder.text == prompt + serialize_plan(plan)[: -len(END)]


@pytest.mark.parametrize("seed", range(3))
def test_random_weights_still_produce_valid_classifications(
    tokenizer: Tokenizer, seed: int
) -> None:
    decoder = _decoder(tokenizer, seed)
    c = decode_classification(decoder, QUESTION)
    assert c.category in {"standard", "fermi", "out_of_scope"}
    assert (c.closest_answerable is not None) == (c.category == "out_of_scope")
    assert decoder.text == classify_prompt(QUESTION) + serialize_classification(c)[: -len(END)]
    json.loads(serialize_classification(c)[: -len(END)])
    assert c.reasoning in reason_options(c.category, QUESTION)


@pytest.mark.parametrize("seed", range(3))
def test_random_weights_only_refuse_with_reviewed_sentences(
    tokenizer: Tokenizer, seed: int
) -> None:
    question = "how do you find the slope of a curve?"
    decoder = _decoder(tokenizer, seed)
    decoder.start(classify_prompt(question))
    decoder.emit('{"category": "out_of_scope", "domains": [], "reasoning": "')
    reason = _choose_text(decoder, reason_options("out_of_scope", question))
    assert reason in reason_options("out_of_scope", question)


def test_refusals_can_only_name_what_the_question_names() -> None:
    question = "how do you find the slope of a curve?"
    reasons = reason_options("out_of_scope", question)
    assert "Not a physics question; it is pure math." in reasons
    assert not any("anxiety" in r or "dream" in r for r in reasons)
    assert not any(r.startswith("Category error:") for r in reasons)  # no template matched
    math = "Not a physics question; it is pure math."
    assert redirect_options(question, math) == [
        "How fast is a rock moving after falling 20 m from rest?",
        "How fast is a rock moving after falling 12 m from rest?",
    ]
    assert "How much does a 1.4 kg brain weigh?" in redirect_options("How much does a dream weigh?")


def test_slots_are_filled_only_from_their_own_position() -> None:
    assert (1 + 0, {"abstract": "a promise"}) not in template_matches("x")
    matched = dict(template_matches("Hi! How fast is a promise Thanks!"))
    assert {"abstract": "a promise"} in matched.values()
    reasons = reason_options("out_of_scope", "How fast is a promise?")
    assert "Category error: a promise does not move, so it has no speed." in reasons
    # tellus once filled a slot with "a dropped ball take"; that question matches no template.
    reasons = reason_options(
        "out_of_scope", "How long does a dropped ball take to fall from a table"
    )
    assert not any("dropped ball take" in r for r in reasons)
    # Redirects are fixed, answerable questions (#92): no slot text from the question.
    assert redirect_options(
        "What is the best taco topping?", "Not a physics question; it is a matter of taste."
    ) == ["How much kinetic energy does a 0.2 kg ball have at 3 m/s?"]


def test_gold_classifications_are_always_options(store: DataStore) -> None:
    checked = 0
    for e in DataFactory(store, seed=8).examples(1500):
        if e.task != "classify":
            continue
        gold = Classification.model_validate_json(e.target[: -len(END)])
        question = json.loads(e.prompt[len(CLASSIFY) :])["question"]
        assert gold.reasoning in reason_options(gold.category, question), question
        if gold.category == "out_of_scope":
            assert gold.closest_answerable in redirect_options(question, gold.reasoning), question
        checked += 1
    assert checked > 300


def _end_bias(
    decoder: Decoder, monkeypatch: pytest.MonkeyPatch, *, after: int, bias: float
) -> None:
    """Shift the end token's score by ``bias`` once ``after`` tokens have been written."""
    original = decoder.model.step
    written = 0

    def biased(ids: torch.Tensor, past: object = None) -> tuple[torch.Tensor, object]:
        nonlocal written
        if past is not None:  # one call per generated token, after the prompt
            written += 1
        logits, new_past = original(ids, past)  # type: ignore[arg-type]
        if written >= after:
            logits[..., decoder.tokenizer.end_id] += bias
        return logits, new_past

    monkeypatch.setattr(decoder.model, "step", biased)


def test_explanation_numbers_are_constrained(
    store: DataStore, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = _decoder(tokenizer, 1)
    _end_bias(decoder, monkeypatch, after=20, bias=1000.0)
    eqs = [store.equations["kin_v_squared"]]
    text = decode_explanation(
        decoder, QUESTION, 19.8057, "m/s", eqs, ["No drag"], temperature=1.0, seed=3
    )
    allowed = set(explain_numbers(QUESTION, 19.8057, ["No drag"]))
    assert set(extract_numbers(text)) <= allowed


def test_explanations_are_not_cut_at_the_slot_cap(
    store: DataStore, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(4)
    decoder = Decoder(FermiLM(LUNA), tokenizer)  # the default 48-token slot cap
    _end_bias(decoder, monkeypatch, after=100, bias=1000.0)
    eqs = [store.equations["kin_v_squared"]]
    text = decode_explanation(decoder, QUESTION, 19.8057, "m/s", eqs, ["No drag"])
    assert len(tokenizer.encode(text)) > 48


def test_explanation_without_an_end_token_is_an_llm_error(
    store: DataStore, tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch
) -> None:
    eqs = [store.equations["kin_v_squared"]]
    prompt = explain_prompt(QUESTION, 19.8057, "m/s", eqs, ["No drag"])
    room = 30  # tokens left after the prompt; the model never writes its end token
    tiny = ModelConfig(
        name="tiny",
        vocab_size=LUNA.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        context_length=len(encode_task(tokenizer, prompt)) + room,
    )
    torch.manual_seed(0)
    decoder = Decoder(FermiLM(tiny), tokenizer)
    _end_bias(decoder, monkeypatch, after=0, bias=-1e9)
    with pytest.raises(LLMError, match="end token"):
        decode_explanation(decoder, QUESTION, 19.8057, "m/s", eqs, ["No drag"])


def test_greedy_decoding_is_deterministic(store: DataStore, tokenizer: Tokenizer) -> None:
    eqs = [store.equations["kin_v_squared"], store.equations["kin_x_at"]]
    consts = list(store.constants.values())

    def run() -> Plan | str:
        try:
            return decode_plan(_decoder(tokenizer, 7), QUESTION, "standard", eqs, consts)
        except PlanValidationError as exc:  # no time is stated, so a plan may fail (#85, #91)
            return str(exc)

    assert run() == run()


def test_planning_needs_equations(tokenizer: Tokenizer) -> None:
    with pytest.raises(LLMError, match="no retrieved"):
        decode_plan(_decoder(tokenizer, 0), QUESTION, "standard", [], [])


def test_context_overflow_is_an_llm_error(tokenizer: Tokenizer) -> None:
    tiny = ModelConfig(
        name="tiny",
        vocab_size=LUNA.vocab_size,
        d_model=32,
        n_layers=1,
        n_heads=2,
        context_length=16,
    )
    decoder = Decoder(FermiLM(tiny), tokenizer)
    with pytest.raises(LLMError, match="exceeds"):
        decoder.start(classify_prompt(QUESTION))


@pytest.mark.parametrize("favored", ["standard", "fermi"])
def test_choose_follows_the_models_preference(
    tokenizer: Tokenizer, monkeypatch: pytest.MonkeyPatch, favored: str
) -> None:
    decoder = _decoder(tokenizer, 0)
    boost = tokenizer.encode(favored)[0]
    original = decoder.model.step

    def biased(ids: torch.Tensor, past: object = None) -> tuple[torch.Tensor, object]:
        logits, new_past = original(ids, past)  # type: ignore[arg-type]
        logits[..., boost] += 100.0
        return logits, new_past

    monkeypatch.setattr(decoder.model, "step", biased)
    decoder.start(classify_prompt("x"))
    decoder.emit('{"category": "')
    assert decoder.choose(["standard", "fermi", "out_of_scope"], closer='"') == favored
    assert decoder.text.endswith('"category": "' + favored)


def test_choose_tells_prefix_options_apart(tokenizer: Tokenizer) -> None:
    decoder = _decoder(tokenizer, 0)
    decoder.start(classify_prompt("x"))
    picked = decoder.choose(["1", "12", "120"], closer=", ")
    assert picked in {"1", "12", "120"}
    assert not decoder.text.endswith(",")


# ------------------------------------------------------------------ dimension-aware plans


def test_question_quantities_keep_number_and_unit_together() -> None:
    q = "An object moving at 69.0 m/s has 4.6 kg*m/s of momentum, and 5 kg and 5 kg more."
    assert question_quantities(q) == [
        ("69", "m/s"),
        ("4.6", "kg*m/s"),
        ("5", "kg"),
        ("5", "kg"),
    ]


def test_values_must_fit_their_variable(store: DataStore) -> None:
    eq = store.equations["momentum"]
    q = "An object moving at 69.0 m/s has 4.6 kg*m/s of momentum. What's its mass?"
    by_symbol = {v.symbol: v for v in eq.variables}
    p = known_value_options(by_symbol["p"], q, [])
    v = known_value_options(by_symbol["v"], q, [])
    assert ValueOption("4.6", "kg*m/s", "given") in p
    assert not any(o.number == "69" for o in p)
    assert ValueOption("69", "m/s", "given") in v
    assert not any(o.number == "4.6" for o in v)
    assert not any(o.origin == "assumption" for o in v)  # a velocity here is stated, not 0


def test_units_stay_with_their_number(store: DataStore) -> None:
    eq = store.equations["newton_gravitation"]
    consts = list(store.constants.values())
    q = "Gravitational pull: 0.0019 kN. The first mass is 11 kilograms. Second: 570 pounds."
    m2 = next(v for v in eq.variables if v.symbol == "m2")
    options = known_value_options(m2, q, consts)
    assert ValueOption("570", "pounds", "given") in options
    assert ValueOption("570", "kilograms", "given") not in options
    g = next(v for v in eq.variables if v.symbol == "G")
    assert known_value_options(g, q, consts)[0].origin == "constant"


def test_target_is_what_the_question_leaves_open(store: DataStore) -> None:
    consts = list(store.constants.values())
    grav = store.equations["newton_gravitation"].variables
    q = "The pull is 0.012 kN, the second mass 1100 kg and the first 87.4kg. Distance?"
    assert target_options(grav, q, consts) == ["r"]
    momentum = store.equations["momentum"].variables
    q = "An object moving at 69.0 m/s has 4.6 kg*m/s of momentum. What's its mass?"
    assert target_options(momentum, q, consts) == ["m"]
    # From rest, the counts can't tell which speed is unknown, but the ask can.
    kin = store.equations["kin_v_at"].variables
    q = "A car starts from rest and accelerates at 3 m/s^2 for 4 s. Final speed?"
    assert target_options(kin, q, consts) == ["v"]
    # "How fast" asks for a speed, and not the starting one unless the start is asked about.
    q = "A car starts from rest and accelerates at 3 m/s^2 for 4 s. How fast?"
    assert target_options(kin, q, consts) == ["v"]
    q = "It reaches 9 m/s after accelerating at 3 m/s^2 for 2 s. How fast was it at the start?"
    assert target_options(kin, q, consts) == ["v0"]
    # The gas constant is a table constant, never a target.
    gas = store.equations["ideal_gas_law"].variables
    assert "R" not in target_options(gas, "Some gas.", consts)


def _assert_decodable(store: DataStore, e: Example) -> None:
    """The decoder's constraints allow every choice the gold plan makes, in its order."""
    payload = json.loads(e.prompt[len(PLAN) :])
    gold = Plan.model_validate_json(e.target[: -len(END)])
    eqs = [store.equations[x["id"]] for x in payload["equations"]]
    consts = relevant_constants(eqs, [store.constants[c["name"]] for c in payload["constants"]])
    cited = [store.equations[i] for i in gold.equation_ids]
    variables: dict[str, Variable] = {}
    for eq in cited:  # merged the way decode_plan merges them
        for v in eq.variables:
            variables.setdefault(v.symbol, v)
    q = payload["question"]
    assert gold.equation_ids[0] in equation_options(eqs, q), q
    assert gold.target in target_options(list(variables.values()), q, consts), q
    assert set(gold.assumptions) <= set(assumption_options(cited)), q
    gold_by_symbol = {k.symbol: k for k in gold.known_values}
    order = [s for s in variables if s not in gold.unknowns]
    unused = stated_quantities(q)
    locks = quantity_locks(q, list(variables.values()))
    for i, symbol in enumerate(order):  # in the order the decoder writes them
        k = gold_by_symbol[symbol]
        options = ordered_options(
            locked_options(
                assignable_options(
                    known_value_options(variables[symbol], q, consts),
                    variables[symbol],
                    unused,
                    [variables[x] for x in order[i:]],
                ),
                symbol,
                locks,
                unused,
                order[i:],
            ),
            symbol,
            locks,
            unused,
            order[i:],
        )
        key = (format_number(k.value), k.unit, k.origin)
        assert any((o.number, o.unit, o.origin) == key for o in options), (q, k)
        if k.origin == "given":
            unused.remove(key[:2])


def test_gold_plans_always_fit_the_constraints(store: DataStore) -> None:
    factory = DataFactory(store, seed=4)
    checked = 0
    for e in factory.examples(1500):
        if e.task == "plan":
            _assert_decodable(store, e)
            checked += 1
    assert checked > 400


def test_chained_gold_plans_fit_the_constraints(store: DataStore) -> None:
    factory = DataFactory(store, seed=5)
    for chain in tpl.CHAINS:
        made = 0
        for _ in range(60):
            p = factory.chained_problem(chain)
            if p is None:
                continue
            prompt = plan_prompt(p.question, "standard", p.retrieved, p.constants)
            _assert_decodable(
                store, Example("plan", "train", p.template, prompt, serialize_plan(p.plan))
            )
            made += 1
        assert made >= 10, chain


def test_each_stated_quantity_fills_one_slot(store: DataStore) -> None:
    eq = store.equations["kin_v_at"]
    by_symbol = {v.symbol: v for v in eq.variables}
    q = "end speed: 20.9 mph. Initial velocity: 12 km/h. Time: 0.706 minutes. Work out a."
    pending = [by_symbol[s] for s in ("v", "v0", "t")]
    unused = question_quantities(q)
    v = assignable_options(
        known_value_options(by_symbol["v"], q, []), by_symbol["v"], unused, pending
    )
    assert {(o.number, o.unit) for o in v} == {("20.9", "mph"), ("12", "km/h")}  # no 0 yet
    unused.remove(("20.9", "mph"))
    v0 = assignable_options(
        known_value_options(by_symbol["v0"], q, []), by_symbol["v0"], unused, pending[1:]
    )
    assert v0 == [ValueOption("12", "km/h", "given")]  # not 20.9 again, and not 0


def test_fillers_only_when_nothing_stated_is_left(store: DataStore) -> None:
    consts = list(store.constants.values())
    momentum = {v.symbol: v for v in store.equations["momentum"].variables}
    q = "An object moving at 34 mph has 0.147kg*m/s of momentum. What's its mass?"
    unused = question_quantities(q)
    p_opts = known_value_options(momentum["p"], q, consts)
    p = assignable_options(p_opts, momentum["p"], unused, [momentum["p"], momentum["v"]])
    assert p == [ValueOption("0.147", "kg*m/s", "given")]
    v_opts = known_value_options(momentum["v"], q, consts)
    v = assignable_options(v_opts, momentum["v"], unused, [momentum["v"]])
    assert v == [ValueOption("34", "mph", "given")]  # 0 m/s and c are off the table
    # From rest: no speed is stated, so v0 may be the structural 0.
    kin = {v.symbol: v for v in store.equations["kin_v_at"].variables}
    q = "A car starts from rest and accelerates at 3 m/s^2 for 4 s. Final speed?"
    unused = question_quantities(q)
    pending = [kin["v0"], kin["a"], kin["t"]]
    v0 = assignable_options(known_value_options(kin["v0"], q, consts), kin["v0"], unused, pending)
    assert ValueOption("0", "m/s", "assumption") in v0
    a = assignable_options(known_value_options(kin["a"], q, consts), kin["a"], unused, pending[1:])
    assert a == [ValueOption("3", "m/s^2", "given")]  # not standard gravity


def test_a_constant_fillable_variable_is_not_the_target_when_another_is_open(
    store: DataStore,
) -> None:
    consts = relevant_constants(
        [store.equations["gravitational_pe"]], list(store.constants.values())
    )
    pe = store.equations["gravitational_pe"].variables
    q = "Lifting a wrench by 11 m took 11000 J of work against gravity. What is its mass?"
    assert target_options(pe, q, consts) == ["m"]  # g comes from the table
    # When g is the only thing left open, it can still be the target.
    q = "A 4 kg rock gains 200 J of potential energy when lifted 5 m. What is g there?"
    assert target_options(pe, q, consts) == ["g"]


def test_zero_is_only_assumed_inside_the_typical_range(store: DataStore) -> None:
    consts = list(store.constants.values())
    pe = {v.symbol: v for v in store.equations["gravitational_pe"].variables}
    q = "Lifting it 11 m took 11000 J."
    assert known_value_options(pe["m"], q, consts) == []  # no 1 kg, no 0 kg
    g = known_value_options(pe["g"], q, consts)
    assert ValueOption("9.80665", "m/s^2", "constant") in g
    assert not any(o.origin == "assumption" for o in g)  # g = 0 is not physics
    kin = {v.symbol: v for v in store.equations["kin_v_at"].variables}
    assert not any(o.origin == "assumption" for o in known_value_options(kin["v0"], q, consts))
    rest = known_value_options(kin["v0"], "A ball is dropped from 11 m.", consts)
    assert ValueOption("0", "m/s", "assumption") in rest  # "dropped" means v0 = 0


# The force is the target and no mass is stated: m has no legal value, so the plan must
# fail rather than write m = 1 kg (issue #85). Before the fix, random models wrote the
# structural 1 (and a constant or a stated number with the wrong units) into m here.
NO_MASS = "A ball accelerates at 2 m/s^2. What is the net force on it?"


@pytest.mark.parametrize("seed", range(16))
def test_an_unstated_mass_is_never_invented(
    store: DataStore, tokenizer: Tokenizer, seed: int
) -> None:
    eqs = [store.equations["newton_second_law"]]
    consts = relevant_constants(eqs, list(store.constants.values()))
    try:
        plan = decode_plan(_decoder(tokenizer, seed), NO_MASS, "standard", eqs, consts)
    except PlanValidationError as exc:
        assert "no legal value" in str(exc)
        return
    assert "m" not in {k.symbol for k in plan.known_values}


def test_a_number_the_question_states_is_a_legal_value(store: DataStore) -> None:
    consts = list(store.constants.values())
    m = {v.symbol: v for v in store.equations["newton_second_law"].variables}["m"]
    q = "A 1 kg box is pushed by a 10 N force. What is its acceleration?"
    assert ValueOption("1", "kg", "given") in known_value_options(m, q, consts)


def test_the_structural_one_is_prose_only() -> None:
    # "KE = 1/2 m v^2" is prose the model may write; "m = 1" is a value it may not.
    q = "How fast does a ball dropped from 20 m hit the ground?"
    assert number_guard_ok("KE = ", "1", plan_numbers(q, []))
    assert not number_guard_ok("m = ", "1", value_numbers(q, []))


def test_dimensionless_values_are_bare_numbers(store: DataStore) -> None:
    friction = {v.symbol: v for v in store.equations["kinetic_friction"].variables}
    q = "A crate feels 14 N of friction with a 40 N normal force and coefficient 0.35."
    mu = known_value_options(friction["mu"], q, [])
    assert ValueOption("0.35", "dimensionless", "given") in mu
    assert not any(o.number == "40" for o in mu)  # a number with a unit is never dimensionless
    q = "Sliding friction is 14 N under a 40 N normal force. What is the coefficient?"
    assert target_options(list(friction.values()), q, []) == ["mu"]


def test_equations_need_room_for_every_stated_quantity(store: DataStore) -> None:
    eqs = [store.equations["kinetic_energy"], store.equations["kinetic_energy_momentum"]]
    q = "Something zipping along at 4.1 m/s carries 620 J of kinetic energy. What is its mass?"
    assert equation_options(eqs, q) == ["kinetic_energy"]
    # Two stated speeds don't fit KE = mv^2/2, which has one; nothing has room, so all stay.
    assert equation_options(eqs, "From 3 m/s to 5 m/s with 10 J?") == [
        "kinetic_energy",
        "kinetic_energy_momentum",
    ]
    # Bare numbers are labels as often as values, so they never rule an equation out.
    assert "kinetic_energy" in equation_options(eqs, "Ball 2 moves at 4 m/s with 8 J.")


def test_labels_and_the_ask_pin_the_plan(store: DataStore) -> None:
    eq = store.equations["doppler_approaching"]
    q = "Assuming fs is 758 Hz, v = 43 m/s and 2800 Hz for the heard frequency, find vs."
    assert target_options(eq.variables, q, []) == ["vs"]
    locks = quantity_locks(q, eq.variables)
    # fs and v by symbol, f by name ("2800 Hz for the heard frequency").
    assert locks == {"fs": ("758", "Hz"), "v": ("43", "m/s"), "f": ("2800", "Hz")}
    unused = stated_quantities(q)
    pending = [eq.variable(s) for s in ("f", "fs", "v")]
    f = assignable_options(known_value_options(pending[0], q, []), pending[0], unused, pending)
    assert [o.number for o in locked_options(f, "f", locks, unused, ["f", "fs", "v"])] == ["2800"]
    fs = known_value_options(eq.variable("fs"), q, [])
    assert [o.number for o in locked_options(fs, "fs", locks, unused, ["fs", "v"])] == ["758"]


def test_the_ask_picks_the_equation(store: DataStore) -> None:
    eqs = [store.equations["work_constant_force"], store.equations["gravitational_pe"]]
    q = "I'm studying for a test. Lifting a book by 5.3 m took 207 J of work. What is its mass?"
    assert equation_options(eqs, q) == ["gravitational_pe"]


def test_repeats_blocks_loops_but_not_units() -> None:
    text = {
        1: " roughly",
        2: " a",
        3: " device",
        4: " that",
        5: " draws",
        6: " power",
        7: " m/s",
        8: " 3",
    }
    assert repeats([1], 1, text)  # "roughly roughly"
    assert not repeats([2], 3, text)
    loop = [2, 3, 4, 5, 6, 2]
    assert not repeats(loop[:-1], 2, text)
    assert repeats([*loop, 3, 4, 5, 6], 2, text)  # the same six words again
    assert not repeats([8, 7, 8], 7, text)  # "3 m/s 3 m/s" has no word tokens


def test_assumptions_come_from_the_equation_and_its_scenarios(store: DataStore) -> None:
    eq = store.equations["kin_v_squared"]
    options = assumption_options([eq])
    assert options[: len(eq.assumptions)] == list(eq.assumptions)
    assert "Air resistance is negligible" in options
    assert len(options) == len(set(options))


def test_the_question_picks_between_twins(store: DataStore) -> None:
    eqs = [store.equations["series_resistors"], store.equations["parallel_resistors"]]
    q = "Two resistors: total resistance 3 ohm, resistance 2 is 5 ohm. What is resistance 1?"
    # Unsaid, both stay, and the sanity check flags the answer as ambiguous.
    assert equation_options(eqs, q) == ["series_resistors", "parallel_resistors"]
    assert equation_options(eqs, q.replace("resistors:", "resistors in parallel:")) == [
        "parallel_resistors"
    ]


def test_a_labelled_bare_number_needs_a_home(store: DataStore) -> None:
    eqs = [store.equations["stefan_boltzmann"], store.equations["stefan_boltzmann_emissivity"]]
    q = (
        "A panel at 350 K with surface area 2 m^2 radiates heat. The emissivity comes out "
        "to 0.017. What power does it radiate?"
    )
    assert equation_options(eqs, q) == ["stefan_boltzmann_emissivity"]
    q = "600W for the radiated power. The temperature comes out to 350 K. 0.036 for the emissivity."
    assert equation_options(eqs, f"{q} Find the area.") == ["stefan_boltzmann_emissivity"]


def test_the_first_value_stated_goes_to_the_first_of_a_pair() -> None:
    # "One has mass 0.293 kg and speed 9.27 km/h, the other mass 140 kg and speed 31 mph."
    unused = [("0.293", "kg"), ("9.27", "km/h"), ("140", "kg"), ("31", "mph")]
    first, second = ValueOption("9.27", "km/h", "given"), ValueOption("31", "mph", "given")
    zero = ValueOption("0", "m/s", "assumption")
    options = [first, second, zero]
    assert ordered_options(options, "v1", {}, unused, ["v1", "v2"]) == [first, zero]
    assert ordered_options(options, "v2", {}, unused, ["v1", "v2"]) == [second, zero]
    # A label on either half decides instead, and so does a pair already half filled.
    assert ordered_options(options, "v1", {"v2": ("31", "mph")}, unused, ["v1", "v2"]) == options
    assert ordered_options(options, "v2", {}, unused, ["v2"]) == options
    assert ordered_options(options, "vf", {}, unused, ["vf"]) == options


def test_a_unitless_variable_needs_a_value_or_an_ask(store: DataStore) -> None:
    eqs = [store.equations["stefan_boltzmann"], store.equations["stefan_boltzmann_emissivity"]]
    # solem once filled the emissivity with the area's 1200.
    q = "I need the power radiated. I know the area is 1200 cm^2; the temperature is 610 kelvin."
    assert equation_options(eqs, q) == ["stefan_boltzmann"]
    q = "A 2 m^2 panel at 350 K radiates 40 W. What is its emissivity?"
    assert equation_options(eqs, q) == ["stefan_boltzmann_emissivity"]


def test_how_strongly_asks_for_a_force(store: DataStore) -> None:
    eqs = [store.equations["newton_gravitation"], store.equations["gravitational_pe_universal"]]
    q = "Two boulders of 2500 kg and 1900 kg sit 24 m apart. How strongly do they attract?"
    assert equation_options(eqs, q) == ["newton_gravitation"]


def test_a_chain_starts_from_the_equation_with_the_ask(store: DataStore) -> None:
    eqs = [store.equations["impulse_momentum"], store.equations["kin_x_at"],
           store.equations["newton_second_law"]]  # fmt: skip
    q = "A 3 kg cart starts at 2 m/s and is pushed with 12 N for 4 s. Work out x."
    assert equation_options(eqs, q) == ["kin_x_at"]


# ------------------------------------------------------- real phrasing (issue #91)


class _EveryChoice(Decoder):
    """A stand-in decoder that takes one scripted option at each choice, with no model.

    ``plans_every_way`` re-runs ``decode_plan`` over every combination of options, so a test
    can assert something about every plan any model could write, not just the ones that
    random weights happen to reach. A listed assumption is never picked (they are prose), and
    unless ``chain`` is set the plan has one equation and one unknown, the target.
    """

    def __init__(self, script: list[int], chain: bool) -> None:  # no model: only choices matter
        self.script = script
        self.chain = chain
        self.sizes: list[int] = []

    def start(self, prompt: str) -> None:
        pass

    def emit(self, fixed: str) -> None:
        pass

    def choose(self, options: Sequence[str], closer: str = "") -> str:
        options = list(options)
        if "]" in options and any(o.startswith('"') for o in options):
            return "]"  # the assumptions list: take none
        if "]" in options and not self.chain:
            return "]"  # no second equation, no second unknown
        at = len(self.sizes)
        self.sizes.append(len(options))
        return options[self.script[at] if at < len(self.script) else 0]

    def free_text(self, allowed_numbers: Sequence[str], **_: object) -> str:
        return "Solve for the unknown."


def plans_every_way(
    store: DataStore, question: str, equation_ids: Sequence[str], *, chain: bool = False
) -> tuple[list[Plan], int]:
    """Every plan the decoder can write for ``question``, and how many ways ended in failure."""
    eqs = [store.equations[i] for i in equation_ids]
    consts = relevant_constants(eqs, list(store.constants.values()))
    plans: list[Plan] = []
    failed = 0
    script: list[int] = []
    for _ in range(200_000):
        decoder = _EveryChoice(script, chain)
        try:
            plans.append(decode_plan(decoder, question, "standard", eqs, consts))
        except PlanValidationError:
            failed += 1
        picks = [script[i] if i < len(script) else 0 for i in range(len(decoder.sizes))]
        while picks and picks[-1] + 1 >= decoder.sizes[len(picks) - 1]:
            picks.pop()
        if not picks:
            return plans, failed
        picks[-1] += 1
        script = picks
    raise AssertionError("the options never ran out")


def _value(plan: Plan, symbol: str) -> float | None:
    return next((k.value for k in plan.known_values if k.symbol == symbol), None)


LIGHT_SPEED = 299792458.0

AVG_SPEED_Q = (
    "Work out the time interval if the average speed comes out to 23.2 m/s; "
    "the distance comes out to 98 meters."
)
CENTRIPETAL_Q = (
    "Calculate the centripetal acceleration of an object following a path with a radius of "
    "a curvature of 0.2 m and at an angular velocity of 5 rad/s."
)


def test_a_speed_is_never_the_speed_of_light(store: DataStore) -> None:
    # The speed of light fits the units of any speed; only a question about light may use it.
    for question, ids in (
        (AVG_SPEED_Q, ["avg_speed", "kin_x_avg_velocity"]),
        (CENTRIPETAL_Q, ["centripetal_acceleration", "tangential_speed"]),
    ):
        plans, _ = plans_every_way(store, question, ids, chain=True)
        assert plans, question
        for plan in plans:
            assert all(k.origin != "constant" for k in plan.known_values), (question, plan)
            assert LIGHT_SPEED not in {k.value for k in plan.known_values}, (question, plan)


def test_the_time_interval_plan_uses_each_stated_value_once(store: DataStore) -> None:
    plans, failed = plans_every_way(store, AVG_SPEED_Q, ["avg_speed", "kin_x_avg_velocity"])
    # v = d/t fits the question exactly. v0 and v of the other equation would both need the
    # one stated speed (or the speed of light), so that way fails instead of being written.
    assert [(p.equation_ids, p.target, _value(p, "v"), _value(p, "d")) for p in plans] == [
        (["avg_speed"], "t", 23.2, 98.0)
    ]
    assert failed


def test_the_centripetal_plan_never_fills_the_speed_with_a_constant(store: DataStore) -> None:
    plans, _ = plans_every_way(
        store, CENTRIPETAL_Q, ["centripetal_acceleration", "tangential_speed"], chain=True
    )
    # v = omega r gives the speed, so the chained plan takes omega and r as stated.
    chained = [
        p for p in plans if p.equation_ids == ["centripetal_acceleration", "tangential_speed"]
    ]
    assert any(
        {k.symbol: k.value for k in p.known_values} == {"omega": 5.0, "r": 0.2} for p in chained
    )
    for plan in plans:  # whatever else it writes, a value is a stated number
        assert {k.value for k in plan.known_values} <= {5.0, 0.2}, plan


def test_light_and_electrons_may_use_their_constants(store: DataStore) -> None:
    q = "A radio signal takes 1.28 s to travel between the Earth and the moon. How far is it?"
    plans, _ = plans_every_way(store, q, ["avg_speed"])
    assert any(_value(p, "v") == LIGHT_SPEED for p in plans)
    q = "What is the force on an electron moving at 1000000 m/s in a 1.0 T field?"
    plans, _ = plans_every_way(store, q, ["magnetic_force_charge"])
    assert any(_value(p, "q") == 1.602176634e-19 for p in plans)


KE_QUESTIONS = (
    "A 1200 kg car travels at 25 m/s. What is its kinetic energy?",
    "A ball of mass 0.145 kg is thrown at 40 m/s. How much kinetic energy does it have?",
    "How much energy of motion does a 70 kg runner have at 6.0 m/s?",
)


@pytest.mark.parametrize("question", KE_QUESTIONS)
def test_the_momentum_is_never_the_filler_zero(store: DataStore, question: str) -> None:
    plans, _ = plans_every_way(
        store, question, ["kinetic_energy", "kinetic_energy_momentum", "momentum"]
    )
    assert plans
    for plan in plans:
        assert _value(plan, "p") != 0.0, plan  # m and v are stated: p is never "0 kg*m/s"
        assert plan.equation_ids[0] == "kinetic_energy", plan  # the one with room for v
        assert {k.origin for k in plan.known_values} == {"given"}, plan


RANGE_Q = "If a velocity increases from 0 to 20 m/s in 10 s, what is the average acceleration?"
SPORTS_CAR_Q = (
    "The driver of a sports car traveling at 10.0 m/s steps down hard on the accelerator for "
    "5.0 s and the velocity increases to 30.0 m/s. What was the average acceleration of the "
    "car during the 5.0 s time interval?"
)
IMPULSE_Q = (
    "For how long should a force of 130 N be applied to an object of mass 50 kg to change "
    "its speed from 20 m/s to 60 m/s?"
)


def test_the_asked_acceleration_is_never_set_to_zero(store: DataStore) -> None:
    ids = ["kin_v_at", "kin_x_at", "kin_v_squared"]
    for question in (RANGE_Q, SPORTS_CAR_Q):
        plans, _ = plans_every_way(store, question, ids)
        assert plans, question
        for plan in plans:
            assert plan.target == "a", (question, plan)  # not x or d
            assert _value(plan, "a") is None, (question, plan)
            assert all(k.origin == "given" for k in plan.known_values), (question, plan)


def test_a_stated_zero_start_is_a_given_value(store: DataStore) -> None:
    plans, _ = plans_every_way(store, RANGE_Q, ["kin_v_at"])
    assert plans
    for plan in plans:
        v0 = next(k for k in plan.known_values if k.symbol == "v0")
        assert (v0.value, v0.origin) == (0.0, "given"), plan  # stated, not assumed
        assert _value(plan, "v") == 20.0 and _value(plan, "t") == 10.0, plan


def test_the_sports_car_keeps_its_stated_speeds_and_time(store: DataStore) -> None:
    plans, _ = plans_every_way(store, SPORTS_CAR_Q, ["kin_v_at"])
    assert plans
    for plan in plans:
        assert {k.symbol: k.value for k in plan.known_values}.get("t") == 5.0, plan
        assert sorted(k.value for k in plan.known_values) == [5.0, 10.0, 30.0], plan


def test_from_a_to_b_is_initial_then_final(store: DataStore) -> None:
    plans, _ = plans_every_way(store, IMPULSE_Q, ["impulse_momentum"])
    assert plans
    for plan in plans:
        assert plan.target == "t", plan
        assert (_value(plan, "v0"), _value(plan, "v")) == (20.0, 60.0), plan
        assert (_value(plan, "F"), _value(plan, "m")) == (130.0, 50.0), plan


def test_a_zero_start_needs_a_rest_cue(store: DataStore) -> None:
    moving = "A car accelerates at 3 m/s^2 for 4 s. What is its final speed?"
    plans, failed = plans_every_way(store, moving, ["kin_v_at"])
    assert not plans and failed  # no speed to start from: the plan fails, v0 is not guessed
    for cue in ("starts from rest", "is released from rest", "starts at rest"):
        q = f"A car that {cue} accelerates at 3 m/s^2 for 4 s. What is its final speed?"
        plans, _ = plans_every_way(store, q, ["kin_v_at"])
        assert plans and all(_value(p, "v0") == 0.0 for p in plans), q
        assert all(
            next(k for k in p.known_values if k.symbol == "v0").origin == "assumption"
            for p in plans
        ), q


@pytest.mark.parametrize("symbol", ["p", "F", "m", "t", "a"])
def test_the_filler_zero_never_lands_on_other_variables(store: DataStore, symbol: str) -> None:
    # Even from rest, only a starting speed is assumed 0.
    q = "A cart is released from rest."
    consts = list(store.constants.values())
    for eq in store.equations.values():
        for v in eq.variables:
            if v.symbol == symbol and not v.name.startswith(("initial", "launch")):
                assert not any(
                    o.origin == "assumption" for o in known_value_options(v, q, consts)
                ), (eq.id, v.name)
