"""Query-side vocabulary aliases for the keyword retriever.

Textbook questions name quantities in words the equation tags don't use:
"accelerate" for acceleration, "travel" for distance, "how long" for time,
"5 kg" for mass, "1 min" for time. Each alias adds canonical words to the
query only. The equation data is never changed, so an alias can't add an
equation to the database or change what the planner may cite.

Every canonical word here is a word that appears in equation tags, names, or
variable descriptions in ``data/equations.json``. Word aliases are matched on
the lowercased query word, so list every inflection that should match.
"""

from __future__ import annotations

import re

# Vocabulary: a word that means a quantity, mapped to the canonical words it implies.
# Kept to words that the equation tags don't spell the same way and that the real-question
# misses need. Aliases that were redundant with the unit aliases (accelerate, travel,
# crest, PE, KE) were left out: each one changed the top 5 on ordinary questions for no gain.
WORD_ALIASES: dict[str, tuple[str, ...]] = {
    "velocity": ("speed",),
    "velocities": ("speed",),
    "weigh": ("weight",),
    "weighs": ("weight",),
    "weighed": ("weight",),
}

# Phrases whose meaning is a quantity even though their words are stopwords.
PHRASE_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\bhow long\b"), ("time",)),
    (re.compile(r"\bhow fast\b"), ("speed",)),
    (re.compile(r"\bhow far\b"), ("distance",)),
    (re.compile(r"\bevery\b"), ("period",)),
)

# Units written after a number, matched on the normalized query (case matters: "N" is
# newtons, "n" is nothing). The unit reads as the quantity it measures.
_NUMBER_UNIT = r"(?<=\d)[\s-]*"
UNIT_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(_NUMBER_UNIT + r"m/s\^?2\b"), ("acceleration",)),
    (re.compile(_NUMBER_UNIT + r"m/s\b(?!\^)"), ("speed",)),
    (re.compile(_NUMBER_UNIT + r"kg\b"), ("mass",)),
    (re.compile(_NUMBER_UNIT + r"N\b"), ("force",)),
    (re.compile(_NUMBER_UNIT + r"W\b"), ("power",)),
    (re.compile(_NUMBER_UNIT + r"J\b"), ("energy",)),
    (re.compile(_NUMBER_UNIT + r"C\b"), ("charge",)),
    (re.compile(_NUMBER_UNIT + r"A\b"), ("current",)),
    (re.compile(_NUMBER_UNIT + r"V\b"), ("voltage",)),
    (re.compile(_NUMBER_UNIT + r"ohm\b"), ("resistance",)),
    (re.compile(_NUMBER_UNIT + r"Hz\b"), ("frequency",)),
    (re.compile(_NUMBER_UNIT + r"nm\b"), ("wavelength",)),
    (re.compile(_NUMBER_UNIT + r"(min|s)\b"), ("time",)),
)

_WORD = re.compile(r"[a-z]+")


def alias_words(text: str) -> list[str]:
    """Canonical words the query implies, in a stable order, duplicates allowed.

    Call this on the query as typed (after normalization), before tokenizing. The
    caller appends the result to the query, then tokenizes both together.
    """
    out: list[str] = []
    for word in _WORD.findall(text.lower()):
        out.extend(WORD_ALIASES.get(word, ()))
    lowered = text.lower()
    for pattern, words in PHRASE_ALIASES:
        if pattern.search(lowered):
            out.extend(words)
    for pattern, words in UNIT_ALIASES:
        if pattern.search(text):
            out.extend(words)
    return out
