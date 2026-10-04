"""Eval harness (stub in v0.1; implemented in v0.5).

What works now: loading and validating ``questions.yaml``, so a malformed
eval file fails CI. Scoring and running are stubs. Rubrics, tolerance rules,
and the regression policy are in ``docs/EVALS.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from askphysics.models import Answer
from askphysics.pipeline import Pipeline

QUESTIONS_FILE = Path(__file__).with_name("questions.yaml")

EvalCategory = Literal["standard", "multi_step", "fermi", "ambiguous", "impossible", "adversarial"]
ExpectedBehavior = Literal["answer", "estimate_range", "refuse_and_redirect"]


class AnswerShape(BaseModel):
    """Reference value and unit, with a relative or an order-of-magnitude tolerance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    value: float
    unit: str
    rel_tolerance: float | None = Field(default=None, gt=0)
    tolerance_decades: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one_tolerance(self) -> AnswerShape:
        if (self.rel_tolerance is None) == (self.tolerance_decades is None):
            raise ValueError("set exactly one of rel_tolerance or tolerance_decades")
        return self


class EvalQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    question: str
    category: EvalCategory
    expected_behavior: ExpectedBehavior
    expected_answer_shape: AnswerShape | None
    must_mention_assumptions: bool
    acceptable_equation_ids: list[list[str]] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _shape_matches_behavior(self) -> EvalQuestion:
        refusing = self.expected_behavior == "refuse_and_redirect"
        if refusing and self.expected_answer_shape is not None:
            raise ValueError(f"{self.id}: refusals must not have an expected answer shape")
        if not refusing and self.expected_answer_shape is None:
            raise ValueError(f"{self.id}: answers need an expected answer shape")
        shape = self.expected_answer_shape
        if self.expected_behavior == "estimate_range" and shape and shape.tolerance_decades is None:
            raise ValueError(f"{self.id}: estimates are graded by order of magnitude")
        return self


def load_questions(path: Path = QUESTIONS_FILE) -> list[EvalQuestion]:
    """Load and validate the eval set. Raises ``ValueError`` on duplicate ids."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path.name}: top level must be a list")
    questions = [EvalQuestion.model_validate(item) for item in raw]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path.name}: duplicate question ids")
    return questions


def score_answer(question: EvalQuestion, answer: Answer) -> float:
    """Score one answer from 0 to 1 using the category rubric in ``docs/EVALS.md``."""
    # TODO: Dispatch on question.category. Convert answer.final_value to the expected unit
    # with Pint, apply rel_tolerance or the log10 decade check, add the equation-id and
    # status components, and apply the assumption-quality rubric (v0.5).
    raise NotImplementedError("eval scoring lands in v0.5")


def run_suite(pipeline: Pipeline, questions: list[EvalQuestion]) -> dict[str, float]:
    """Run every question through the pipeline and return the mean score per category."""
    # TODO: Run each question (3 times for real LLMs, per ADR-006), score it, write
    # evals/reports/<version>.json, and compare against the previous release using the
    # regression thresholds in docs/EVALS.md (v0.5).
    raise NotImplementedError("the eval runner lands in v0.5")
