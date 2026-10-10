import json
from pathlib import Path

import pytest

from askphysics.normalize import normalize_question
from askphysics.retrieval.aliases import alias_words
from askphysics.retrieval.base import Retriever
from askphysics.retrieval.keyword import KeywordRetriever, stem, tokenize

REAL_EVAL = (
    Path(__file__).resolve().parents[1] / "third_party" / "openstax-physics" / "real_eval.jsonl"
)
# Recall at top_k 5 with no classifier hint, after the units and vocabulary aliases.
# Lower it only with a reason in the PR; it was 42 of 49 (85.7%) before the aliases.
REAL_RECALL_FLOOR = 49


def test_falling_object_retrieves_kinematics(retriever: KeywordRetriever) -> None:
    result = retriever.search("how fast does a falling object hit the ground", k=3)
    assert result.equations
    top = result.equations[0].equation
    assert top.domain == "kinematics"
    assert top.id == "kin_v_squared"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("What current flows through a 220 ohm resistor across a 9 V battery?", "ohms_law"),
        ("Pressure of 2 mol of an ideal gas in a container at 300 K", "ideal_gas_law"),
        ("Gravitational attraction between two planets", "newton_gravitation"),
        ("What force stops a 1500 kg car braking from 20 m/s in 5 s?", "impulse_momentum"),
    ],
)
def test_top_hit_per_domain(retriever: KeywordRetriever, query: str, expected: str) -> None:
    assert retriever.search(query, k=3).equations[0].equation.id == expected


def test_scores_bounded_sorted_and_limited(retriever: KeywordRetriever) -> None:
    result = retriever.search("speed velocity acceleration time distance mass energy", k=4)
    scores = [h.score for h in result.equations]
    assert len(scores) <= 4
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_nonsense_returns_nothing(retriever: KeywordRetriever) -> None:
    assert retriever.search("zxqv blorp", k=5).equations == []


def test_domain_hint_boosts_but_does_not_filter(retriever: KeywordRetriever) -> None:
    everything = 1000
    plain = retriever.search("mass", k=everything)
    boosted = retriever.search("mass", k=everything, domains=["energy"])
    assert {h.equation.id for h in plain.equations} == {h.equation.id for h in boosted.equations}
    before = {h.equation.id: h.score for h in plain.equations}
    after = {h.equation.id: h.score for h in boosted.equations}
    assert after["kinetic_energy"] > before["kinetic_energy"]
    assert after["momentum"] == before["momentum"]


def test_domain_hint_breaks_ties(retriever: KeywordRetriever) -> None:
    # "mass" ties several equations from different domains; the hint decides which comes first.
    plain = retriever.search("mass", k=1000).equations
    tied = [h for h in plain if h.score == plain[0].score]
    other_domain = next(
        h.equation.domain for h in tied if h.equation.domain != tied[0].equation.domain
    )
    hinted = retriever.search("mass", k=3, domains=[other_domain])
    assert hinted.equations[0].equation.domain == other_domain


def test_deterministic(retriever: KeywordRetriever) -> None:
    a = retriever.search("velocity of a moving mass", k=5)
    b = retriever.search("velocity of a moving mass", k=5)
    assert a == b


def test_examples_are_retrieved(retriever: KeywordRetriever) -> None:
    result = retriever.search("baseball momentum", k=3)
    assert result.examples[0].example.id == "ex_momentum_001"


def test_satisfies_protocol(retriever: KeywordRetriever) -> None:
    assert isinstance(retriever, Retriever)


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("falling", "fall"),
        ("falls", "fall"),
        ("dropped", "drop"),
        ("speed", "speed"),
        ("mass", "mass"),
        ("masses", "mass"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_tokenize_drops_stopwords_and_short_tokens() -> None:
    # Numbers are deliberately not tokens: they carry values, not topics.
    assert tokenize("How fast does a ball fall 20 m?") == {"fast", "ball", "fall"}


@pytest.mark.parametrize(
    ("query", "concept"),
    [
        ("a 5-kg object", "mass"),
        ("a 1200 kg car", "mass"),
        ("100000 W of power", "power"),
        ("it exerts 10000 N", "force"),
        ("6 C of charge", "charge"),
        ("for 1 min", "time"),
        ("in 5 s", "time"),
        ("a 500 nm photon", "wavelength"),
        ("at 20 m/s^2", "acceleration"),
        ("at 3 m/s", "speed"),
        ("how long does it take", "time"),
        ("the velocity of the wave", "speed"),
        ("the child weighs 300 N", "weight"),
    ],
)
def test_aliases_name_the_quantity(query: str, concept: str) -> None:
    assert concept in alias_words(query)


@pytest.mark.parametrize("query", ["a car has 3 wheels", "an arm A of the object", "every car"])
def test_aliases_stay_quiet_on_ordinary_words(query: str) -> None:
    # Units need a number before them, and "A" or "s" alone is not a unit.
    words = alias_words(query)
    assert "current" not in words
    assert "time" not in words


def test_real_question_recall_at_top_5(retriever: KeywordRetriever) -> None:
    rows = [json.loads(line) for line in REAL_EVAL.read_text().splitlines() if line.strip()]
    hits = 0
    for row in rows:
        gold = set(row["plan"]["equation_ids"])
        shown = {
            h.equation.id
            for h in retriever.search(normalize_question(row["question"]), 5).equations
        }
        hits += gold <= shown
    assert len(rows) == 49
    assert hits >= REAL_RECALL_FLOOR, (
        f"{hits}/{len(rows)} real questions retrieved every gold equation"
    )


def test_newton_second_law_from_units_and_acceleration(retriever: KeywordRetriever) -> None:
    query = "How much force needs to be applied to a 5 kg object to accelerate at 20 m/s^2?"
    ids = [h.equation.id for h in retriever.search(query, k=5).equations]
    assert "newton_second_law" in ids


def test_engine_power_question_retrieves_both_chain_equations(retriever: KeywordRetriever) -> None:
    query = (
        "A cars engine generates 100000 W of power as it exerts a force of 10000 N. "
        "How long does it take the car to travel 100 m?"
    )
    ids = [h.equation.id for h in retriever.search(normalize_question(query), k=5).equations]
    assert {"power_force_velocity", "avg_speed"} <= set(ids)
