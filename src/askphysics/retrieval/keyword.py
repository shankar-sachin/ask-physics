"""In-memory keyword and tag retriever: the one retriever that actually works in v0.1.

Scoring is deliberately simple. Each query token earns the weight of the
best field it matches in an equation (tags 3, name 2, domain 2, variable
names and descriptions 1). The sum is normalized by the best possible score
for the query, so scores land in [0, 1]. A classifier domain hint adds a
small boost and breaks ties in favor of hinted domains. Remaining ties break
by id, so ordering is deterministic.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from askphysics.models import (
    Equation,
    RetrievalResult,
    ScoredEquation,
    ScoredExample,
    WorkedExample,
)

TAG_WEIGHT = 3.0
NAME_WEIGHT = 2.0
DOMAIN_WEIGHT = 2.0
VARIABLE_WEIGHT = 1.0
DOMAIN_BOOST = 0.1

STOPWORDS = frozenset(
    """
    a an and are as at be by can could do does for from had has have how i if in into is it its
    long many much of on or so than that the then this to was what when where which while who
    why will with would you your take
    """.split()  # noqa: SIM905 - a word list reads better than 50 quoted strings
)
_WORD = re.compile(r"[a-z][a-z0-9]*")


def stem(word: str) -> str:
    """Strip common English suffixes so "falling", "falls", and "fall" match.

    Crude on purpose: it only has to map query and data words onto the same
    form, not produce real roots.
    """
    if len(word) > 5 and word.endswith("ing"):
        word = word[:-3]
    elif (len(word) > 5 and word.endswith("ed")) or word.endswith("sses"):
        word = word[:-2]
    elif len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]
    if len(word) > 3 and word[-1] == word[-2] and word[-1] not in "aeiouls":
        word = word[:-1]  # "dropp" -> "drop"
    return word


def tokenize(text: str) -> set[str]:
    """Lowercase, split into words, drop stopwords and one-letter tokens, stem."""
    return {stem(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS and len(w) > 1}


def _tokens(texts: Iterable[str]) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out |= tokenize(t)
    return out


class KeywordRetriever:
    """``Retriever`` over in-memory equations and worked examples."""

    def __init__(
        self, equations: Iterable[Equation], examples: Iterable[WorkedExample] = ()
    ) -> None:
        self._equations = sorted(equations, key=lambda e: e.id)
        self._examples = sorted(examples, key=lambda e: e.id)
        self._eq_fields = {
            eq.id: [
                (TAG_WEIGHT, _tokens(eq.tags)),
                (NAME_WEIGHT, tokenize(eq.name)),
                (DOMAIN_WEIGHT, tokenize(eq.domain)),
                (VARIABLE_WEIGHT, _tokens(f"{v.name} {v.description}" for v in eq.variables)),
            ]
            for eq in self._equations
        }
        self._ex_fields = {
            ex.id: [(TAG_WEIGHT, _tokens(ex.tags)), (VARIABLE_WEIGHT, tokenize(ex.problem_text))]
            for ex in self._examples
        }

    @staticmethod
    def _score(query: set[str], fields: list[tuple[float, set[str]]]) -> float:
        if not query:
            return 0.0
        best = max(w for w, _ in fields)
        total = sum(max((w for w, toks in fields if tok in toks), default=0.0) for tok in query)
        return total / (best * len(query))

    def search(
        self, query: str, k: int, *, domains: Sequence[str] | None = None
    ) -> RetrievalResult:
        q = tokenize(query)
        boosted = set(domains or ())

        eq_hits: list[ScoredEquation] = []
        for eq in self._equations:
            score = self._score(q, self._eq_fields[eq.id])
            if score <= 0:
                continue
            if eq.domain in boosted:
                score += DOMAIN_BOOST
            eq_hits.append(ScoredEquation(equation=eq, score=min(score, 1.0)))
        eq_hits.sort(key=lambda h: (-h.score, h.equation.domain not in boosted, h.equation.id))

        ex_hits = [
            ScoredExample(example=ex, score=min(s, 1.0))
            for ex in self._examples
            if (s := self._score(q, self._ex_fields[ex.id])) > 0
        ]
        ex_hits.sort(key=lambda h: (-h.score, h.example.id))

        return RetrievalResult(query=query, equations=eq_hits[:k], examples=ex_hits[:k])
