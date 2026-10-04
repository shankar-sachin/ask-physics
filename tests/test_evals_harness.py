from collections import Counter
from pathlib import Path

import pytest
from evals.harness import AnswerShape, load_questions, run_suite, score_answer
from pydantic import ValidationError

from askphysics.config import Settings
from askphysics.models import Answer, Confidence
from askphysics.pipeline import Pipeline


def test_sample_set_loads_with_required_mix() -> None:
    questions = load_questions()
    assert len(questions) == 8
    counts = Counter(q.category for q in questions)
    assert counts == {"standard": 3, "fermi": 3, "impossible": 2}


def test_fermi_questions_require_assumptions_and_decades() -> None:
    for q in load_questions():
        if q.category == "fermi":
            assert q.must_mention_assumptions
            assert q.expected_answer_shape is not None
            assert q.expected_answer_shape.tolerance_decades is not None


def test_impossible_questions_expect_refusal() -> None:
    for q in load_questions():
        if q.category == "impossible":
            assert q.expected_behavior == "refuse_and_redirect"
            assert q.expected_answer_shape is None


def test_eval_questions_do_not_leak_into_worked_examples(pipeline: Pipeline) -> None:
    texts = {ex.problem_text.lower() for ex in pipeline.data.examples.values()}
    for q in load_questions():
        assert q.question.lower() not in texts


def test_shape_needs_exactly_one_tolerance() -> None:
    with pytest.raises(ValidationError):
        AnswerShape(value=1.0, unit="m")
    with pytest.raises(ValidationError):
        AnswerShape(value=1.0, unit="m", rel_tolerance=0.1, tolerance_decades=1.0)


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    entry = (
        "- id: a\n  question: q\n  category: impossible\n"
        "  expected_behavior: refuse_and_redirect\n  expected_answer_shape: null\n"
        "  must_mention_assumptions: false\n"
    )
    path = tmp_path / "q.yaml"
    path.write_text(entry + entry)
    with pytest.raises(ValueError, match="duplicate"):
        load_questions(path)


def test_scoring_and_running_are_stubs() -> None:
    q = load_questions()[0]
    answer = Answer(
        question=q.question,
        status="degraded",
        category="standard",
        confidence=Confidence(label="low", score=0.0),
        explanation="",
    )
    with pytest.raises(NotImplementedError):
        score_answer(q, answer)
    with pytest.raises(NotImplementedError):
        run_suite(Pipeline.from_settings(Settings()), [q])
