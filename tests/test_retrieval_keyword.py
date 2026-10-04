import pytest

from askphysics.retrieval.base import Retriever
from askphysics.retrieval.keyword import KeywordRetriever, stem, tokenize


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
    plain = retriever.search("mass", k=12)
    boosted = retriever.search("mass", k=12, domains=["energy"])
    assert {h.equation.id for h in plain.equations} == {h.equation.id for h in boosted.equations}
    before = {h.equation.id: h.score for h in plain.equations}
    after = {h.equation.id: h.score for h in boosted.equations}
    assert after["kinetic_energy"] > before["kinetic_energy"]
    assert after["momentum"] == before["momentum"]


def test_domain_hint_breaks_ties(retriever: KeywordRetriever) -> None:
    # "mass" scores 1.0 for several equations; the hint decides which comes first.
    assert retriever.search("mass", k=3).equations[0].equation.id == "momentum"
    hinted = retriever.search("mass", k=3, domains=["gravitation"])
    assert hinted.equations[0].equation.id == "newton_gravitation"


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
