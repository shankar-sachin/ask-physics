from typing import TypeVar

import pytest
from pydantic import BaseModel
from tests.conftest import DEMO_QUESTION

from askphysics.config import Settings
from askphysics.data.loader import CONSTANT_SYMBOL_ALIASES, DataStore
from askphysics.errors import EmptyQuestionError, LLMError, PlanValidationError, SolverError
from askphysics.llm.fake import FakeLLMClient
from askphysics.models import (
    Classification,
    ComputeResult,
    KnownValue,
    Plan,
    Question,
    RetrievalResult,
)
from askphysics.pipeline import (
    LIMIT_CASE_CAVEAT,
    Pipeline,
    check_limit_cases,
    compute,
    never_negative,
    refuse,
    resolve_constant_symbols,
    sanity_check,
    score_confidence,
    validate_plan,
)
from askphysics.prose import FALLBACK_REFUSAL
from askphysics.retrieval.keyword import KeywordRetriever
from askphysics.solver.units import check_dimensions, quantity

T = TypeVar("T", bound=BaseModel)


def _plan(**overrides: object) -> Plan:
    data: dict[str, object] = {
        "target": "v",
        "unknowns": ["v"],
        "known_values": [
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="a", value=9.80665, unit="m/s^2", origin="constant"),
            KnownValue(symbol="d", value=20, unit="m", origin="given"),
        ],
        "equation_ids": ["kin_v_squared"],
        "assumptions": ["No drag"],
        "strategy": "s",
    }
    data.update(overrides)
    return Plan.model_validate(data)


# ------------------------------------------------------------------ end to end


def test_end_to_end_standard_question(pipeline: Pipeline) -> None:
    answer = pipeline.run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert answer.category == "standard"
    assert answer.final_value == pytest.approx(19.8057, rel=1e-5)
    assert answer.unit == "meter / second"
    assert [e.id for e in answer.equations_used] == ["kin_v_squared"]
    assert answer.assumptions
    assert answer.confidence.label == "high"
    assert 0.75 <= answer.confidence.score <= 1.0
    assert LIMIT_CASE_CAVEAT in answer.caveats
    assert "[kin_v_squared]" in answer.explanation


def test_out_of_scope_is_refused_with_redirect(pipeline: Pipeline) -> None:
    answer = pipeline.run("How much does the color blue weigh?")
    assert answer.status == "refused"
    assert answer.final_value is None
    assert "Category error" in answer.explanation
    assert "close question" in answer.explanation


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_a_blank_question_is_refused_with_a_reason(pipeline: Pipeline, text: str) -> None:
    answer = pipeline.run(text)
    assert answer.status == "refused"
    assert answer.final_value is None
    assert answer.question == ""
    assert "question is empty" in answer.explanation


def test_solve_rejects_a_blank_question_before_building_one(pipeline: Pipeline) -> None:
    with pytest.raises(EmptyQuestionError):
        pipeline.solve("  ")


def test_fermi_without_planner_degrades_honestly(pipeline: Pipeline) -> None:
    answer = pipeline.run("How many rubber ducks would it take to stop a freight train?")
    assert answer.status == "degraded"
    assert answer.category == "fermi"
    assert answer.final_value is None
    assert "v0.7" in answer.explanation


def test_retrieval_miss_degrades(pipeline: Pipeline) -> None:
    answer = pipeline.run("zxqv blorp?")
    assert answer.status == "degraded"
    assert any("retrieve stage failed" in c for c in answer.caveats)


def test_classifier_failure_falls_back_to_standard(
    store: DataStore, retriever: KeywordRetriever
) -> None:
    class BrokenClassifier(FakeLLMClient):
        def complete_json(self, *, system: str, user: str, schema: type[T]) -> T:
            if schema is Classification:
                raise LLMError("provider down")
            return super().complete_json(system=system, user=user, schema=schema)

    p = Pipeline(llm=BrokenClassifier(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.status == "answered"
    assert any("classify stage failed" in c for c in answer.caveats)


def test_the_llm_never_supplies_the_number(store: DataStore, retriever: KeywordRetriever) -> None:
    class LyingExplainer(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            return "The answer is 42 m/s."

    p = Pipeline(llm=LyingExplainer(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.final_value == pytest.approx(19.8057, rel=1e-5)


def test_explain_falls_back_to_template(store: DataStore, retriever: KeywordRetriever) -> None:
    class MuteExplainer(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            raise LLMError("no prose today")

    p = Pipeline(llm=MuteExplainer(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.explanation.startswith("Using kin_v_squared, v = 19.8057")


def test_from_settings_builds_fake_pipeline(store: DataStore) -> None:
    p = Pipeline.from_settings(Settings(), data=store)
    assert isinstance(p.llm, FakeLLMClient)
    assert p.run(DEMO_QUESTION).status == "answered"


# ------------------------------------------------------------------ stages


def test_validate_plan_rejects_unretrieved_equations(retriever: KeywordRetriever) -> None:
    retrieval = retriever.search(DEMO_QUESTION, k=3)
    with pytest.raises(PlanValidationError, match="not retrieved"):
        validate_plan(_plan(equation_ids=["made_up_equation"]), retrieval)


def test_validate_plan_rejects_empty_and_bad_units(retriever: KeywordRetriever) -> None:
    retrieval = retriever.search(DEMO_QUESTION, k=3)
    with pytest.raises(PlanValidationError, match="no equations"):
        validate_plan(_plan(equation_ids=[]), retrieval)
    bad = KnownValue(symbol="d", value=20, unit="blorps", origin="given")
    with pytest.raises(PlanValidationError, match="invalid units"):
        validate_plan(_plan(known_values=[bad]), retrieval)


def _car_plan(equation_ids: list[str], **overrides: object) -> Plan:
    # A 1500 kg car goes from 0 to 20 m/s in 8 s: a = 2.5 m/s^2, so F = 3750 N.
    data: dict[str, object] = {
        "equation_ids": equation_ids,
        "target": "F",
        "unknowns": ["F", "a"],
        "known_values": [
            KnownValue(symbol="m", value=1500, unit="kg", origin="given"),
            KnownValue(symbol="v", value=20, unit="m/s", origin="given"),
            KnownValue(symbol="v0", value=0, unit="m/s", origin="assumption"),
            KnownValue(symbol="t", value=8, unit="s", origin="given"),
        ],
        "assumptions": [],
        "strategy": "Find a, then F.",
    }
    data.update(overrides)
    return Plan.model_validate(data)


@pytest.mark.parametrize(
    "order", [["newton_second_law", "kin_v_at"], ["kin_v_at", "newton_second_law"]]
)
def test_compute_chains_equations_in_dependency_order(store: DataStore, order: list[str]) -> None:
    result = compute(_car_plan(order), data=store)
    assert result.value == pytest.approx(3750) and result.unit == "newton"
    assert [(s.equation_id, s.symbol) for s in result.steps] == [
        ("kin_v_at", "a"),
        ("newton_second_law", "F"),
    ]
    assert result.steps[0].value == pytest.approx(2.5)
    assert "a" in result.substitutions


def test_a_single_equation_is_one_step(store: DataStore) -> None:
    result = compute(_plan(), data=store)
    assert [s.equation_id for s in result.steps] == ["kin_v_squared"]


def test_a_chain_that_cannot_reach_the_target_says_what_is_missing(store: DataStore) -> None:
    plan = _car_plan(
        ["newton_second_law", "kin_v_at"],
        known_values=[KnownValue(symbol="m", value=1500, unit="kg", origin="given")],
    )
    with pytest.raises(SolverError, match="still missing"):
        compute(plan, data=store)


def _constant_plan(
    constant: KnownValue, equation_id: str, target: str, others: list[KnownValue]
) -> Plan:
    return Plan(
        target=target,
        unknowns=[target],
        known_values=[constant, *others],
        equation_ids=[equation_id],
        assumptions=[],
        strategy="s",
    )


def test_a_table_symbol_copied_by_the_planner_reaches_the_equation(store: DataStore) -> None:
    # Issue #74: the constants table calls it k_B, molecular_kinetic_energy calls it kB.
    plan = _constant_plan(
        KnownValue(symbol="k_B", value=1.380649e-23, unit="J/K", origin="constant"),
        "molecular_kinetic_energy",
        "KE",
        [KnownValue(symbol="T", value=300, unit="K", origin="given")],
    )
    result = compute(plan, data=store)
    assert quantity(result.value, result.unit).to("J").magnitude == pytest.approx(
        1.5 * 1.380649e-23 * 300, rel=1e-9
    )


def test_a_coulomb_constant_is_not_renamed_to_a_spring_constant(store: DataStore) -> None:
    # k_e becomes k only where k has the constant's dimension; in Hooke's law k is N/m.
    plan = _constant_plan(
        KnownValue(symbol="k_e", value=8.99e9, unit="N*m^2/C^2", origin="constant"),
        "hookes_law",
        "F",
        [KnownValue(symbol="x", value=0.1, unit="m", origin="given")],
    )
    renamed = resolve_constant_symbols(plan, store)
    assert [k.symbol for k in renamed.known_values] == ["k_e", "x"]


def test_every_constant_can_fill_its_matching_variable(store: DataStore) -> None:
    # "Matching" is the rule lm/factory.py uses: the variable has the constant's alias (or the
    # table symbol) and the constant's dimension. Each pair is run through compute with the
    # constant under its table symbol, the way a planner copies it.
    covered: set[str] = set()
    for c in store.constants.values():
        name = CONSTANT_SYMBOL_ALIASES.get(c.symbol, c.symbol)
        for eq in store.equations.values():
            if not any(
                v.symbol == name and check_dimensions(quantity(1.0, c.unit), v.unit)
                for v in eq.variables
            ):
                continue
            unfilled = [v for v in eq.variables if v.symbol != name]
            if not unfilled:
                continue
            target, *rest = unfilled
            others = [
                KnownValue(symbol=v.symbol, value=2.0, unit=v.unit, origin="assumption")
                for v in rest
            ]
            constant = KnownValue(symbol=c.symbol, value=c.value, unit=c.unit, origin="constant")
            compute(_constant_plan(constant, eq.id, target.symbol, others), data=store)
            covered.add(c.symbol)
    assert {"k_B", "k_e", "mu_0", "g", "G", "c", "R", "h", "sigma"} <= covered


def test_a_symbol_with_two_meanings_is_never_chained(store: DataStore) -> None:
    # W is a weight in newtons in one and work in joules in the other.
    plan = Plan.model_validate(
        {
            "equation_ids": ["weight", "power_work_time"],
            "target": "P",
            "unknowns": ["P", "W"],
            "known_values": [
                {"symbol": "m", "value": 2, "unit": "kg", "origin": "given"},
                {"symbol": "g", "value": 9.80665, "unit": "m/s^2", "origin": "constant"},
                {"symbol": "t", "value": 3, "unit": "s", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )
    with pytest.raises(SolverError, match="means different quantities"):
        compute(plan, data=store)


def test_sanity_flags_out_of_range_magnitude(store: DataStore) -> None:
    plan = _plan()
    result = ComputeResult(
        target="v", value=1e9, unit="meter / second", symbolic_solution="", substitutions={}
    )
    report = sanity_check(plan, result, data=store)
    assert report.dimensions_ok
    assert report.magnitude_ok is False
    assert report.issues


def test_sanity_flags_wrong_dimensions(store: DataStore) -> None:
    wrong = KnownValue(symbol="d", value=20, unit="s", origin="given")
    plan = _plan(known_values=[wrong])
    result = ComputeResult(
        target="v", value=19.8, unit="meter / second", symbolic_solution="", substitutions={}
    )
    report = sanity_check(plan, result, data=store)
    assert not report.dimensions_ok


def test_limit_cases_are_a_stub(store: DataStore) -> None:
    with pytest.raises(NotImplementedError):
        check_limit_cases(store.equations["kinetic_energy"])


def test_unused_stage_inputs_are_typed() -> None:
    # Question and RetrievalResult are the public stage payloads.
    assert Question(text="q").text == "q"
    assert RetrievalResult(query="q").equation_ids == []


# ------------------------------------------------------------------ confidence


def test_confidence_formula_matches_plan_md() -> None:
    c = score_confidence(
        retrieval_score=0.67,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=3,
        category="standard",
    )
    assert c.score == pytest.approx(0.82, abs=0.01)
    assert c.label == "high"


def test_dimension_failure_caps_confidence() -> None:
    c = score_confidence(
        retrieval_score=1.0,
        dimensions_ok=False,
        magnitude_ok=True,
        n_assumptions=0,
        category="standard",
    )
    assert c.score <= 0.2
    assert c.label == "low"


def test_fermi_caps_confidence() -> None:
    c = score_confidence(
        retrieval_score=1.0,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=0,
        category="fermi",
    )
    assert c.score == 0.6
    assert c.label == "medium"


def test_unchecked_magnitude_counts_half() -> None:
    checked = score_confidence(
        retrieval_score=0.5,
        dimensions_ok=True,
        magnitude_ok=True,
        n_assumptions=0,
        category="standard",
    )
    unchecked = score_confidence(
        retrieval_score=0.5,
        dimensions_ok=True,
        magnitude_ok=None,
        n_assumptions=0,
        category="standard",
    )
    assert checked.score - unchecked.score == pytest.approx(0.1, abs=0.01)


# ------------------------------------------------------------------ model-written prose

ROUGHLY = (
    "Category error: anxiety is a feeling, not a device that draws power does a device that "
    "draws power does it answerable" + " roughly" * 20
)


def test_a_garbled_refusal_is_cleaned_up() -> None:
    garbled = Classification(
        category="out_of_scope", reasoning=ROUGHLY, closest_answerable="reasoning"
    )
    answer = refuse(Question(text="what is ten divided by three"), garbled)
    assert "roughly" not in answer.explanation
    assert FALLBACK_REFUSAL in answer.explanation
    assert answer.redirect is None


def test_looping_prose_falls_back_to_the_template(
    store: DataStore, retriever: KeywordRetriever
) -> None:
    class Looper(FakeLLMClient):
        def complete_text(self, *, system: str, user: str) -> str:
            return "The speed is" + " fast" * 12

    p = Pipeline(llm=Looper(), retriever=retriever, data=store, settings=Settings())
    answer = p.run(DEMO_QUESTION)
    assert answer.explanation.startswith("Using kin_v_squared")
    assert answer.models["explain"] == "template"


def test_the_slope_question_refusal_is_cleaned_up() -> None:
    # tellus's actual output for "how do you find the slope of a curve?" before this fix.
    garbled = Classification(
        category="out_of_scope",
        reasoning=(
            "Category error: a dream is an experience, not an object with mass"
            + " taste" * 11
            + " and solve for the normal temperature gives"
        ),
        closest_answerable=(
            "How much the same interval.constant today. laptop use in uniform typical serving "
            "of a single line.constant today" + " today" * 8 + ".constant height height "
            "height height from rest; air resistance is negligible;;; air resistance"
        ),
    )
    answer = refuse(Question(text="how do you find the slope of a curve?"), garbled)
    assert answer.explanation == f"This can't be answered as asked. {FALLBACK_REFUSAL}"
    assert answer.redirect is None


# ------------------------------------------------------------------ impossible results


@pytest.mark.parametrize(
    ("equation", "symbol", "never"),
    [
        ("series_resistors", "R1", True),
        ("newton_second_law", "m", True),
        ("doppler_approaching", "f", True),
        ("ideal_gas_law", "T", True),
        ("kin_v_at", "v", False),  # velocities can point backward
        ("thin_lens", "di", False),  # virtual images sit at negative distances
    ],
)
def test_which_quantities_are_never_negative(
    store: DataStore, equation: str, symbol: str, never: bool
) -> None:
    assert never_negative(store.equations[equation].variable(symbol)) is never


def test_temperature_changes_can_be_negative(store: DataStore) -> None:
    changes = [v for eq in store.equations.values() for v in eq.variables if v.symbol == "dT"]
    assert changes and not any(never_negative(v) for v in changes)


def test_a_negative_resistance_fails_the_sign_check(store: DataStore) -> None:
    p = Plan.model_validate(
        {
            "equation_ids": ["series_resistors"],
            "target": "R1",
            "unknowns": ["R1"],
            "known_values": [
                {"symbol": "R", "value": 1.3, "unit": "ohm", "origin": "given"},
                {"symbol": "R2", "value": 10.1, "unit": "ohm", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )
    result = compute(p, data=store)
    report = sanity_check(p, result, data=store)
    assert not report.sign_ok and not report.possible and not report.passed
    assert any("negative" in i for i in report.issues)


def _kin_plan(target: str, knowns: list[tuple[str, float, str, str]]) -> Plan:
    return Plan.model_validate(
        {
            "equation_ids": ["kin_v_squared"],
            "target": target,
            "unknowns": [target],
            "known_values": [
                {"symbol": s, "value": v, "unit": u, "origin": o} for s, v, u, o in knowns
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )


def test_a_zero_from_an_assumed_zero_is_trivial(store: DataStore) -> None:
    # tellus solved "How fast does a rock hit the floor after falling 7.4 m?" for a, with v = 0.
    p = _kin_plan(
        "a",
        [("v", 0, "m/s", "assumption"), ("v0", 0, "m/s", "assumption"), ("d", 7.4, "m", "given")],
    )
    report = sanity_check(p, compute(p, data=store), data=store)
    assert report.trivial and not report.possible and not report.passed
    assert any("assumed a zero" in i for i in report.issues)


def test_a_stated_zero_answer_is_not_trivial(store: DataStore) -> None:
    p = _kin_plan(
        "a", [("v", 0, "m/s", "given"), ("v0", 0, "m/s", "given"), ("d", 7.4, "m", "given")]
    )
    report = sanity_check(p, compute(p, data=store), data=store)
    assert not report.trivial and report.possible


def _resistors(equation: str, *, total: float = 3.0, r2: float = 5.0) -> Plan:
    return Plan.model_validate(
        {
            "equation_ids": [equation],
            "target": "R1",
            "unknowns": ["R1"],
            "known_values": [
                {"symbol": "R", "value": total, "unit": "ohm", "origin": "given"},
                {"symbol": "R2", "value": r2, "unit": "ohm", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )


def test_an_unused_stated_value_is_flagged(store: DataStore) -> None:
    p = _resistors("parallel_resistors")
    q = "In parallel: total 3 ohm, resistance 2 is 5 ohm, on a 9 V battery. Find resistance 1."
    report = sanity_check(p, compute(p, data=store), data=store, question=q)
    assert report.unused == ["9 V"] and report.possible and not report.passed
    assert any("gives 9 V, but the plan doesn't use it" in i for i in report.issues)
    q = "In parallel: total 3 ohm, resistance 2 is 5 ohm. Find resistance 1."
    assert sanity_check(p, compute(p, data=store), data=store, question=q).passed
    # A value the question repeats is one value, used once.
    q = "In parallel: total 3 ohm, resistance 2 is 5 ohm. Find resistance 1 next to the 5 ohm."
    assert sanity_check(p, compute(p, data=store), data=store, question=q).passed


def test_an_unnamed_twin_is_ambiguous(store: DataStore) -> None:
    p = _resistors("parallel_resistors")
    q = "Two resistors: total 3 ohm, resistance 2 is 5 ohm. Find resistance 1."
    report = sanity_check(p, compute(p, data=store), data=store, question=q)
    assert report.ambiguous and report.possible and not report.passed
    assert any("series" in i and "parallel" in i for i in report.issues)
    named = sanity_check(p, compute(p, data=store), data=store, question=f"In parallel. {q}")
    assert not named.ambiguous


def test_without_the_question_givens_are_not_checked(store: DataStore) -> None:
    p = _resistors("parallel_resistors")
    report = sanity_check(p, compute(p, data=store), data=store)
    assert report.unused == [] and not report.ambiguous


def test_a_doubtful_answer_caps_confidence() -> None:
    def score(doubtful: bool) -> float:
        return score_confidence(
            retrieval_score=1.0,
            dimensions_ok=True,
            magnitude_ok=True,
            n_assumptions=0,
            category="standard",
            doubtful=doubtful,
        ).score

    assert score(False) > 0.4
    assert score(True) == 0.4


def test_a_borrowed_unitless_value_is_flagged(store: DataStore) -> None:
    # solem filled the emissivity with the area's 1200 ("1200 cm^2").
    p = Plan.model_validate(
        {
            "equation_ids": ["stefan_boltzmann_emissivity"],
            "target": "P",
            "unknowns": ["P"],
            "known_values": [
                {"symbol": "eps", "value": 1200, "unit": "dimensionless", "origin": "given"},
                {"symbol": "sigma", "value": 5.670374419e-08, "unit": "W/(m^2*K^4)",
                 "origin": "constant"},
                {"symbol": "A", "value": 1200, "unit": "cm^2", "origin": "given"},
                {"symbol": "T", "value": 610, "unit": "kelvin", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )  # fmt: skip
    report = sanity_check(p, compute(p, data=store), data=store)
    assert report.magnitude_ok is False and not report.passed
    assert any("eps = 1200 is far outside" in i for i in report.issues)


def test_a_percentage_is_read_as_a_fraction(store: DataStore) -> None:
    # "40.0 percent efficiency" (OpenStax Physics) is 0.4, well inside an efficiency's range.
    p = Plan.model_validate(
        {
            "equation_ids": ["efficiency"],
            "target": "Wout",
            "unknowns": ["Wout"],
            "known_values": [
                {"symbol": "eta", "value": 40, "unit": "percent", "origin": "given"},
                {"symbol": "Win", "value": 1.2e12, "unit": "J", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )
    report = sanity_check(p, compute(p, data=store), data=store)
    assert report.magnitude_ok is not False and not report.issues


def test_celsius_is_converted_by_meaning(store: DataStore) -> None:
    def heat(dt_unit: str, dt: float) -> float:
        p = Plan.model_validate(
            {
                "equation_ids": ["specific_heat"],
                "target": "Q",
                "unknowns": ["Q"],
                "known_values": [
                    {"symbol": "m", "value": 2, "unit": "kg", "origin": "given"},
                    {"symbol": "c", "value": 4186, "unit": "J/(kg*K)", "origin": "given"},
                    {"symbol": "dT", "value": dt, "unit": dt_unit, "origin": "given"},
                ],
                "assumptions": [],
                "strategy": "x",
            }
        )
        return compute(p, data=store).value

    # A change of 30 degC is a change of 30 K (and of 54 degF), never 303 K.
    assert heat("degC", 30) == pytest.approx(heat("K", 30)) == pytest.approx(251160)
    assert heat("degF", 54) == pytest.approx(251160)


def test_an_absolute_celsius_temperature_is_in_kelvin(store: DataStore) -> None:
    def gas(known: tuple[str, float, str], target: str) -> ComputeResult:
        values = {"n": (1, "mol"), "R": (8.314, "J/(mol*K)"), "V": (0.0224, "m^3"),
                  "P": (101325, "Pa"), "T": (27, "degC")}  # fmt: skip
        values[known[0]] = known[1:]
        values.pop(target)
        p = Plan.model_validate(
            {
                "equation_ids": ["ideal_gas_law"],
                "target": target,
                "unknowns": [target],
                "known_values": [
                    {"symbol": s, "value": v, "unit": u, "origin": "given"}
                    for s, (v, u) in values.items()
                ],
                "assumptions": [],
                "strategy": "x",
            }
        )
        return compute(p, data=store)

    assert gas(("T", 27, "degC"), "P").value == pytest.approx(8.314 * 300.15 / 0.0224)
    # Asked for a temperature with Celsius in the question: the answer says it in degC too.
    result = gas(("V", 0.0224, "m^3"), "T")
    assert result.unit == "kelvin" and result.notes == []


def test_a_temperature_asked_in_celsius_is_also_given_in_celsius(store: DataStore) -> None:
    p = Plan.model_validate(
        {
            "equation_ids": ["carnot_efficiency"],
            "target": "Tc",
            "unknowns": ["Tc"],
            "known_values": [
                {"symbol": "eta", "value": 0.2, "unit": "dimensionless", "origin": "given"},
                {"symbol": "Th", "value": 100, "unit": "degC", "origin": "given"},
            ],
            "assumptions": [],
            "strategy": "x",
        }
    )
    result = compute(p, data=store)
    assert result.value == pytest.approx(373.15 * 0.8) and result.unit == "kelvin"
    assert result.notes == ["That is 25.37 degC."]
