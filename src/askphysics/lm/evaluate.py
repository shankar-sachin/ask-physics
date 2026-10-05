"""Task accuracy for a trained Fermi model, measured on held-out examples.

Validation loss says how surprised a model is by the gold text, including
free prose it can never predict exactly. This module scores what the
pipeline actually needs: the right category, and plans that compute the
right answer. Each example is decoded with the same constrained decoder the
pipeline uses, then compared with the gold target from the data factory.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import pint

from askphysics.data.loader import DataStore
from askphysics.errors import AskPhysicsError
from askphysics.lm.factory import Example
from askphysics.lm.formats import format_number
from askphysics.lm.generate import Decoder, decode_classification, decode_plan
from askphysics.lm.tokenizer import CLASSIFY, END, PLAN
from askphysics.models import Classification, Plan
from askphysics.solver.symbolic import solve_for
from askphysics.solver.units import Quantity, quantity

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
    """Accuracies over ``classify`` and ``plan`` examples, plus a few failures to read."""

    classify_examples: int = 0
    category_accuracy: float = 0.0
    plan_examples: int = 0
    equation_accuracy: float = 0.0
    target_accuracy: float = 0.0
    knowns_accuracy: float = 0.0
    valid_plan_rate: float = 0.0
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _payload(prompt: str, token: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(prompt[len(token) :])
    return data


def _gold(target: str) -> str:
    return target[: -len(END)] if target.endswith(END) else target


def _answer(p: Plan, store: DataStore) -> Quantity | None:
    """What a single-equation plan computes, or None if it can't be solved."""
    if len(p.equation_ids) != 1 or p.equation_ids[0] not in store.equations:
        return None
    try:
        knowns = {k.symbol: quantity(k.value, k.unit) for k in p.known_values}
        return solve_for(store.equations[p.equation_ids[0]], p.target, knowns).value
    except (AskPhysicsError, ValueError, ArithmeticError):
        return None


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
    max_failures: int = 10,
    on_progress: Callable[[int], None] | None = None,
) -> EvalReport:
    """Decode every example and score it against its gold target."""
    report = EvalReport()
    category_hits = 0
    plan_hits = {"equation": 0, "target": 0, "knowns": 0, "answer": 0}
    for done, e in enumerate(examples, start=1):
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
            predicted_p = decode_plan(
                decoder, payload["question"], payload["category"], equations, constants
            )
            score = score_plan(predicted_p, gold_p, store)
            report.plan_examples += 1
            for key in plan_hits:
                plan_hits[key] += int(getattr(score, key))
            if not score.answer and len(report.failures) < max_failures:
                report.failures.append(
                    {
                        "task": "plan",
                        "template": e.template,
                        "question": payload["question"],
                        "expected": _summary(gold_p),
                        "got": _summary(predicted_p),
                    }
                )
        if on_progress:
            on_progress(done)
    if report.classify_examples:
        report.category_accuracy = category_hits / report.classify_examples
    if report.plan_examples:
        n = report.plan_examples
        report.equation_accuracy = plan_hits["equation"] / n
        report.target_accuracy = plan_hits["target"] / n
        report.knowns_accuracy = plan_hits["knowns"] / n
        report.valid_plan_rate = plan_hits["answer"] / n
    return report


def _summary(p: Plan) -> str:
    """A plan in one line: "kin_v_at -> v from v0=0 m/s, a=9.8 m/s^2"."""
    knowns = ", ".join(f"{k.symbol}={format_number(k.value)} {k.unit}" for k in p.known_values)
    return f"{'+'.join(p.equation_ids)} -> {p.target} from {knowns}"
