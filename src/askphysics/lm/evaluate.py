"""Task accuracy for a trained Fermi model, measured on held-out examples.

Validation loss says how surprised a model is by the gold text, including
free prose it can never predict exactly. This module scores what the
pipeline actually needs: the right category, and plans that compute the
right answer. Each example is decoded with the same constrained decoder the
pipeline uses, then compared with the gold target from the data factory.

Plans are also run through the router the way ``askphysics ask`` runs them
(retries until Noether accepts a plan, ADR-010) and sorted three ways: the
right answer, a wrong answer the sanity checks flag, and a wrong answer that
passes every check. The last, "confidently wrong", is the number that decides
whether an answer can be trusted, and it should be as close to zero as possible.

``evaluate_real`` asks real textbook questions (OpenStax *Physics*, ADR-016) through the
whole pipeline, exactly as ``askphysics ask`` would, and scores the answers against gold
plans the project wrote. The factory never saw these phrasings.

``evaluate_phrasing`` is a smaller check on plain questions with no numbers ("what is the
speed of sound") and their out-of-scope look-alikes ("how fast is loneliness"): does the
classifier file each on the right side of the boundary? Issue #92 was a real question refused
as maths, which the factory eval cannot catch because it is drawn from the same templates.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pint

from askphysics.data.loader import DataStore
from askphysics.errors import AskPhysicsError
from askphysics.lm.factory import Example
from askphysics.lm.formats import format_number
from askphysics.lm.generate import Decoder, decode_classification, decode_plan
from askphysics.lm.tokenizer import CLASSIFY, END, PLAN
from askphysics.models import Classification, Plan
from askphysics.pipeline import compute, sanity_check
from askphysics.solver.units import Quantity, quantity

if TYPE_CHECKING:
    from askphysics.pipeline import Solved

REL_TOLERANCE = 1e-6  # plans copy numbers exactly, so a right plan gives the same answer


@dataclass(frozen=True)
class PlanScore:
    """How a predicted plan compares with the gold plan."""

    equation: bool  # cites exactly the gold equations
    target: bool  # solves for the gold target
    knowns: bool  # same symbols, numbers, and units as the gold plan
    answer: bool  # computes the gold answer (a "valid plan" for the v0.3 exit criterion)


@dataclass
class EvalReport:
    """Accuracies over ``classify`` and ``plan`` examples, plus a few failures to read.

    ``real`` is the real-question eval (``RealReport``), when it was run; ``phrasing`` is the
    plain-question check (``PhrasingReport``).
    """

    classify_examples: int = 0
    category_accuracy: float = 0.0
    plan_examples: int = 0
    equation_accuracy: float = 0.0
    target_accuracy: float = 0.0
    knowns_accuracy: float = 0.0
    valid_plan_rate: float = 0.0
    attempts: int = 1  # plan attempts the router may make per question
    routed_right_rate: float = 0.0  # right answer after the router's retries
    flagged_wrong_rate: float = 0.0  # wrong, but degraded or flagged by a sanity check
    confidently_wrong_rate: float = 0.0  # wrong, and every check passed
    mean_tries: float = 0.0
    # The rescue run (``rescue`` in ``evaluate_tasks``): a bigger model tries the misses.
    rescue_tried: int = 0  # main-model misses the rescuer attempted
    rescued: int = 0  # ...and got right, by the same rule as the main model
    rescue_rate: float = 0.0  # rescued / rescue_tried
    rescue_confidently_wrong: int = 0  # rescuer answers that passed every check but are wrong
    routed_with_rescue_rate: float = 0.0  # right after escalation, over all plan examples
    failures: list[dict[str, Any]] = field(default_factory=list)
    confidently_wrong: list[dict[str, Any]] = field(default_factory=list)
    real: dict[str, Any] | None = None
    phrasing: dict[str, Any] | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


@dataclass(frozen=True)
class EvalTick:
    """Where an eval stands after one example, for a progress display.

    ``stage`` is ``score`` (the model under test, over every example) or ``rescue`` (the bigger
    model, over just the misses; its ``total`` is known only once scoring has finished).
    The counts are running totals: the display shows rates as they settle.
    """

    stage: str
    done: int
    total: int
    task: str  # "classify" or "plan": what the example just finished was
    plans: int = 0  # plan examples scored so far
    valid: int = 0  # ...of which the plan computed the gold answer
    right: int = 0  # ...of which the router's answer was right
    confident: int = 0  # ...confidently wrong
    rescue_tried: int = 0
    rescued: int = 0
    rescue_confident: int = 0


def _payload(prompt: str, token: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(prompt[len(token) :])
    return data


def _gold(target: str) -> str:
    return target[: -len(END)] if target.endswith(END) else target


def _answer(p: Plan, store: DataStore) -> Quantity | None:
    """What a plan computes, chaining its equations if it has several; None if it can't."""
    if not p.equation_ids or any(eid not in store.equations for eid in p.equation_ids):
        return None
    try:
        result = compute(p, data=store)
    except (AskPhysicsError, ValueError, ArithmeticError):
        return None
    return quantity(result.value, result.unit)


def _same_quantity(a: Quantity, b: Quantity) -> bool:
    try:
        x, y = float(a.to(b.units).magnitude), float(b.magnitude)
    except pint.DimensionalityError:  # incompatible dimensions mean a wrong answer
        return False
    return math.isfinite(x) and math.isclose(x, y, rel_tol=REL_TOLERANCE, abs_tol=0.0)


def score_plan(predicted: Plan, gold: Plan, store: DataStore) -> PlanScore:
    """Compare a decoded plan with the gold plan."""

    def knowns(p: Plan) -> set[tuple[str, str, str]]:
        return {(k.symbol, format_number(k.value), k.unit) for k in p.known_values}

    want = _answer(gold, store)
    got = _answer(predicted, store)
    return PlanScore(
        equation=predicted.equation_ids == gold.equation_ids,
        target=predicted.target == gold.target,
        knowns=knowns(predicted) == knowns(gold),
        answer=want is not None and got is not None and _same_quantity(got, want),
    )


@dataclass(frozen=True)
class Routed:
    """What the router would answer for one plan example."""

    plan: Plan | None  # the accepted plan, or the first that computed, or None
    answer: Quantity | None
    possible: bool  # dimensions and sign check out
    passed: bool  # every sanity check passed: the answer would look trustworthy
    tries: int


def route_plan(
    decoder: Decoder,
    payload: dict[str, Any],
    store: DataStore,
    attempts: int,
    first: Plan | None = None,
) -> Routed:
    """Plan like ``Pipeline._plan_and_compute``: retry with the equations rotated until a
    plan computes a possible result that uses every stated value, falling back to a
    possible one that doesn't. An impossible result is never an answer.
    ``first`` reuses an already-decoded first attempt.
    """
    equations = [store.equations[eq["id"]] for eq in payload["equations"]]
    constants = [store.constants[c["name"]] for c in payload["constants"]]
    flagged: Routed | None = None
    for attempt in range(attempts):
        if attempt == 0 and first is not None:
            p = first
        else:
            shift = attempt % len(equations)
            order = equations[shift:] + equations[:shift]
            try:
                p = decode_plan(decoder, payload["question"], payload["category"], order, constants)
            except AskPhysicsError:
                continue
        try:
            result = compute(p, data=store)
        except (AskPhysicsError, NotImplementedError, ValueError, ArithmeticError):
            continue
        sanity = sanity_check(p, result, data=store, question=payload["question"])
        routed = Routed(
            plan=p,
            answer=quantity(result.value, result.unit),
            possible=sanity.possible,
            passed=sanity.passed,
            tries=attempt + 1,
        )
        if sanity.possible and not sanity.unused:
            return routed
        if sanity.possible:
            flagged = flagged or routed
    if flagged is not None:
        return Routed(flagged.plan, flagged.answer, True, flagged.passed, attempts)
    return Routed(plan=None, answer=None, possible=False, passed=False, tries=attempts)


def _is_right(routed: Routed, want: Quantity | None) -> bool:
    """The router's answer is the gold answer, and it is a possible one."""
    return (
        want is not None
        and routed.answer is not None
        and routed.possible
        and _same_quantity(routed.answer, want)
    )


def score_classification(predicted: Classification, gold: Classification) -> bool:
    return predicted.category == gold.category


def sample_examples(examples: Iterable[Example], per_task: int, seed: int = 0) -> list[Example]:
    """Up to ``per_task`` classify and plan examples, a fixed random sample."""
    pool = [e for e in examples if e.task in ("classify", "plan")]
    random.Random(seed).shuffle(pool)
    counts = {"classify": 0, "plan": 0}
    out = []
    for e in pool:
        if counts[e.task] < per_task:
            counts[e.task] += 1
            out.append(e)
    return out


def evaluate_tasks(
    decoder: Decoder,
    examples: Iterable[Example],
    store: DataStore,
    *,
    attempts: int = 1,
    max_failures: int = 10,
    rescue: Decoder | None = None,
    rescue_attempts: int = 1,
    on_progress: Callable[[int], None] | None = None,
    on_tick: Callable[[EvalTick], None] | None = None,
) -> EvalReport:
    """Decode every example and score it against its gold target.

    ``attempts`` above 1 also runs each plan through the router (``route_plan``) and
    sorts the answers into right, flagged wrong, and confidently wrong.

    ``rescue`` is a bigger model that plans every question the main model did not get
    right, as the pipeline's escalation does. Its router answer counts as rescued when it
    is right by the same rule. The rescue pass runs after scoring, over just the misses, so
    it has a known length (and its own progress).

    ``on_progress`` gets the number of examples scored so far; ``on_tick`` gets an
    ``EvalTick`` after each example of either pass.
    """
    report = EvalReport(attempts=attempts)
    category_hits = 0
    plan_hits = {"equation": 0, "target": 0, "knowns": 0, "answer": 0}
    routed_hits = {"right": 0, "flagged": 0, "confident": 0, "tries": 0}
    rescue_hits = {"tried": 0, "rescued": 0, "confident": 0}
    misses: list[tuple[dict[str, Any], Quantity | None]] = []
    pool = list(examples)

    def tick(stage: str, done: int, total: int, task: str) -> None:
        if on_tick is not None:
            on_tick(
                EvalTick(
                    stage,
                    done,
                    total,
                    task,
                    plans=report.plan_examples,
                    valid=plan_hits["answer"],
                    right=routed_hits["right"],
                    confident=routed_hits["confident"],
                    rescue_tried=len(misses) if stage == "rescue" else 0,
                    rescued=rescue_hits["rescued"],
                    rescue_confident=rescue_hits["confident"],
                )
            )

    for done, e in enumerate(pool, start=1):
        if e.task == "classify":
            question = _payload(e.prompt, CLASSIFY)["question"]
            gold_c = Classification.model_validate_json(_gold(e.target))
            predicted_c = decode_classification(decoder, question)
            report.classify_examples += 1
            if score_classification(predicted_c, gold_c):
                category_hits += 1
            elif len(report.failures) < max_failures:
                report.failures.append(
                    {
                        "task": "classify",
                        "template": e.template,
                        "question": question,
                        "expected": gold_c.category,
                        "got": predicted_c.category,
                    }
                )
        elif e.task == "plan":
            payload = _payload(e.prompt, PLAN)
            gold_p = Plan.model_validate_json(_gold(e.target))
            equations = [store.equations[eq["id"]] for eq in payload["equations"]]
            constants = [store.constants[c["name"]] for c in payload["constants"]]
            predicted_p: Plan | None
            try:
                predicted_p = decode_plan(
                    decoder, payload["question"], payload["category"], equations, constants
                )
            except AskPhysicsError:
                predicted_p = None  # a plan that can't be written is a miss, not a crash
            score = (
                score_plan(predicted_p, gold_p, store)
                if predicted_p is not None
                else PlanScore(equation=False, target=False, knowns=False, answer=False)
            )
            report.plan_examples += 1
            for key in plan_hits:
                plan_hits[key] += int(getattr(score, key))
            routed = route_plan(decoder, payload, store, attempts, first=predicted_p)
            want = _answer(gold_p, store)
            right = _is_right(routed, want)
            routed_hits["tries"] += routed.tries
            if right:
                routed_hits["right"] += 1
            elif routed.passed:
                routed_hits["confident"] += 1
                if len(report.confidently_wrong) < max_failures:
                    report.confidently_wrong.append(
                        {
                            "template": e.template,
                            "question": payload["question"],
                            "expected": _summary(gold_p),
                            "got": _summary(routed.plan) if routed.plan else "-",
                        }
                    )
            else:
                routed_hits["flagged"] += 1
            if rescue is not None and not right:
                misses.append((payload, want))
            if not score.answer and len(report.failures) < max_failures:
                report.failures.append(
                    {
                        "task": "plan",
                        "template": e.template,
                        "question": payload["question"],
                        "expected": _summary(gold_p),
                        "got": _summary(predicted_p) if predicted_p is not None else "-",
                    }
                )
        if on_progress:
            on_progress(done)
        tick("score", done, len(pool), e.task)
    if rescue is not None:
        tick("rescue", 0, len(misses), "plan")
        for n, (payload, want) in enumerate(misses, start=1):
            rescue_hits["tried"] += 1
            saved = route_plan(rescue, payload, store, rescue_attempts)
            if _is_right(saved, want):
                rescue_hits["rescued"] += 1
            elif saved.passed:
                rescue_hits["confident"] += 1
            tick("rescue", n, len(misses), "plan")
    if report.classify_examples:
        report.category_accuracy = category_hits / report.classify_examples
    if report.plan_examples:
        n = report.plan_examples
        report.equation_accuracy = plan_hits["equation"] / n
        report.target_accuracy = plan_hits["target"] / n
        report.knowns_accuracy = plan_hits["knowns"] / n
        report.valid_plan_rate = plan_hits["answer"] / n
        report.routed_right_rate = routed_hits["right"] / n
        report.flagged_wrong_rate = routed_hits["flagged"] / n
        report.confidently_wrong_rate = routed_hits["confident"] / n
        report.mean_tries = routed_hits["tries"] / n
        if rescue is not None:
            tried, rescued = rescue_hits["tried"], rescue_hits["rescued"]
            report.rescue_tried = tried
            report.rescued = rescued
            report.rescue_rate = rescued / tried if tried else 0.0
            report.rescue_confidently_wrong = rescue_hits["confident"]
            report.routed_with_rescue_rate = (routed_hits["right"] + rescued) / n
    return report


def _summary(p: Plan) -> str:
    """A plan in one line: "kin_v_at -> v from v0=0 m/s, a=9.8 m/s^2"."""
    knowns = ", ".join(f"{k.symbol}={format_number(k.value)} {k.unit}" for k in p.known_values)
    return f"{'+'.join(p.equation_ids)} -> {p.target} from {knowns}"


# --------------------------------------------------------------------------- real questions


@dataclass(frozen=True)
class RealQuestion:
    """A question from a textbook, with the plan the project wrote to answer it."""

    id: str
    question: str
    plan: Plan  # the gold plan


def read_real_questions(path: Path) -> list[RealQuestion]:
    """The questions in a real-question file (``third_party/openstax-physics/real_eval.jsonl``)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out.append(RealQuestion(row["id"], row["question"], Plan.model_validate(row["plan"])))
    return out


@dataclass
class RealReport:
    """How the pipeline answers real questions, and where the misses stop.

    Every question is standard, and its gold plan's equations are all in the database, so
    each stage has a right answer: classify as standard, retrieve every gold equation, and
    answer with the gold value.
    """

    questions: int = 0
    standard_rate: float = 0.0  # classified standard: not refused, not a Fermi estimate
    retrieved_rate: float = 0.0  # the planner was shown every gold equation
    right_rate: float = 0.0  # the answer ``ask`` gives is the gold answer
    flagged_wrong_rate: float = 0.0  # no answer, or a wrong one the sanity checks flag
    confidently_wrong_rate: float = 0.0  # wrong, and every check passed
    misses: list[dict[str, Any]] = field(default_factory=list)


def evaluate_real(
    solve: Callable[[str], Solved],
    questions: Sequence[RealQuestion],
    store: DataStore,
    *,
    on_progress: Callable[[int], None] | None = None,
) -> RealReport:
    """Run each question through ``solve`` (``Pipeline.solve``) and score the answer.

    A miss records the first stage that went wrong, so the report says whether real
    phrasing trips the classifier, retrieval, or the planner.
    """
    report = RealReport(questions=len(questions))
    hits = {"standard": 0, "retrieved": 0, "right": 0, "flagged": 0, "confident": 0}
    for done, q in enumerate(questions, start=1):
        solved = solve(q.question)
        standard = solved.classification.category == "standard"
        shown = set(solved.retrieval.equation_ids) if solved.retrieval else set()
        retrieved = set(q.plan.equation_ids) <= shown
        attempt = solved.attempt
        got = quantity(attempt.result.value, attempt.result.unit) if attempt else None
        want = _answer(q.plan, store)
        right = got is not None and want is not None and _same_quantity(got, want)
        hits["standard"] += standard
        hits["retrieved"] += retrieved
        if right:
            hits["right"] += 1
        elif attempt is not None and attempt.sanity.passed:
            hits["confident"] += 1
        else:
            hits["flagged"] += 1
        if not right:
            if not standard:
                stage = f"classified {solved.classification.category}"
            elif not retrieved:
                stage = "retrieve"
            elif attempt is None:
                stage = solved.failed_stage or "plan"
            else:
                stage = "confidently wrong" if attempt.sanity.passed else "wrong, flagged"
            report.misses.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "stage": stage,
                    "expected": _summary(q.plan),
                    "got": _summary(attempt.plan) if attempt else "-",
                }
            )
        if on_progress:
            on_progress(done)
    if questions:
        n = len(questions)
        report.standard_rate = hits["standard"] / n
        report.retrieved_rate = hits["retrieved"] / n
        report.right_rate = hits["right"] / n
        report.flagged_wrong_rate = hits["flagged"] / n
        report.confidently_wrong_rate = hits["confident"] / n
    return report


# --------------------------------------------------------------------------- real phrasing


@dataclass(frozen=True)
class PhrasingQuestion:
    """A plain question and which side of the out-of-scope boundary it belongs on."""

    id: str
    question: str
    expected: str  # "answerable" (standard or fermi) or "out_of_scope"


def read_phrasing_questions(path: Path) -> list[PhrasingQuestion]:
    """The questions in a phrasing file (``evals/real_phrasing.jsonl``)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row["expected"] not in ("answerable", "out_of_scope"):
                raise ValueError(f"{row['id']}: expected must be answerable or out_of_scope")
            out.append(PhrasingQuestion(row["id"], row["question"], row["expected"]))
    return out


@dataclass
class PhrasingReport:
    """How the classifier files plain questions: answerable ones must not be refused, and
    look-alikes with no physical subject must be."""

    questions: int = 0
    answerable_rate: float = 0.0  # answerable questions the classifier did not refuse
    refused_rate: float = 0.0  # out-of-scope questions the classifier refused
    misses: list[dict[str, Any]] = field(default_factory=list)


def evaluate_phrasing(
    classify: Callable[[str], Classification], questions: Sequence[PhrasingQuestion]
) -> PhrasingReport:
    """Classify each question (``pipeline.classify`` with the model under test) and score it."""
    report = PhrasingReport(questions=len(questions))
    right = {"answerable": 0, "out_of_scope": 0}
    total = {"answerable": 0, "out_of_scope": 0}
    for q in questions:
        category = classify(q.question).category
        total[q.expected] += 1
        if (category == "out_of_scope") == (q.expected == "out_of_scope"):
            right[q.expected] += 1
        else:
            report.misses.append(
                {"id": q.id, "question": q.question, "expected": q.expected, "got": category}
            )
    if total["answerable"]:
        report.answerable_rate = right["answerable"] / total["answerable"]
    if total["out_of_scope"]:
        report.refused_rate = right["out_of_scope"] / total["out_of_scope"]
    return report
