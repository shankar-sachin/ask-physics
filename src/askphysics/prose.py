"""Checks on model-written prose before anyone sees it.

The Fermi models are small, and small models loop ("roughly roughly roughly") or
borrow a sentence that belongs to another question ("anxiety is a feeling" for
"what is ten divided by three"). The decoder only offers reviewed sentences where
it can; these checks guard the rest, for any ``LLMClient``. Text that fails is
replaced with a plain sentence or dropped, never shown.
"""

from __future__ import annotations

import re
from itertools import pairwise

FALLBACK_REFUSAL = "It doesn't look like a physics question the equation database can answer."
# Refusal reasons that needn't repeat the question's words ("Not a physics question; it is
# history."). Any other reason must share a content word with the question, so a model
# can't transplant one ("anxiety is a feeling" for "what is ten divided by three").
GENERIC_REASONS = (
    "Not a physics",
    "Not answerable",
    "Research-level",
    "Needs unknowable",
    "Unknowable",
)
STOPWORDS = frozenset(
    [
        "about",
        "all",
        "and",
        "are",
        "but",
        "can",
        "for",
        "has",
        "how",
        "its",
        "not",
        "the",
        "was",
        "who",
        "why",
        "you",
        "also",
        "been",
        "could",
        "does",
        "from",
        "have",
        "into",
        "just",
        "like",
        "many",
        "more",
        "most",
        "much",
        "only",
        "over",
        "should",
        "some",
        "than",
        "that",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "very",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "will",
        "with",
        "would",
        "your",
    ]
)


def readable(text: str) -> bool:
    """Whether model-written prose is fit to show: real words, no loops.

    Rejects text with fewer than three words, a word repeated back to back ("roughly
    roughly"), or long text that is mostly the same few words over and over.
    """
    # Whitespace-separated words, letters only: "[orbital_speed] (Speed" is two words.
    words = [w for w in (re.sub(r"[^a-z']", "", t.lower()) for t in text.split()) if w]
    if len(words) < 3:
        return False
    # A loop writes the same token twice with nothing between ("roughly roughly");
    # "point charges; charges at rest" is fine.
    raw = text.lower().split()
    if any(a == b and a.isalpha() and len(a) >= 3 for a, b in pairwise(raw)):
        return False
    return len(words) < 8 or len(set(words)) / len(words) >= 0.5


def content_words(text: str) -> set[str]:
    """Words of three or more letters that carry meaning, lowercase."""
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3} - STOPWORDS


def refusal_reason(question: str, reasoning: str) -> str:
    """The classifier's reason for refusing, if it is readable and about this question."""
    if readable(reasoning) and (
        reasoning.startswith(GENERIC_REASONS) or content_words(question) & content_words(reasoning)
    ):
        return reasoning
    return FALLBACK_REFUSAL


def usable_redirect(text: str | None) -> str | None:
    """The suggested answerable question, if it reads as one; else nothing."""
    if text is None:
        return None
    text = text.strip()
    if not (readable(text) and text.endswith("?") and len(text.split()) >= 4):
        return None
    return text
