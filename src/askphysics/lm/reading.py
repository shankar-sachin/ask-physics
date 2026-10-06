"""Facts a question states outright, read with rules instead of the model.

Two things a small model gets wrong even when the question spells them out:

- **Labels.** "the source frequency fs is 758 Hz" says 758 Hz is ``fs``. tellus
  still swapped it with the heard frequency, because both are frequencies.
- **The ask.** "Lifting a book took 207 J. What is its mass?" says the unknown
  is a mass. tellus solved for a force instead.

``labelled_quantities`` and ``asked_names`` read those facts, and the plan
decoder narrows its options to agree with them (ADR-015). These rules only
narrow choices among numbers and ids already in the input: they never compute.
When a rule doesn't fire, or contradicts the dimensions, the decoder ignores it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from askphysics.lm.formats import format_number
from askphysics.lm.templates import VAR_SYNONYMS
from askphysics.models import Variable
from askphysics.solver.units import is_valid_unit

_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_UNIT = r"[A-Za-z](?:[A-Za-z0-9/*]|\^-?)*(?:\((?:[A-Za-z0-9/*]|\^-?)+\))?"
_SYMBOL = r"[A-Za-z][A-Za-z0-9_]*"
# A quantity: a number not glued to an identifier or an exponent, then an optional unit.
_QUANTITY = re.compile(
    r"(?<![A-Za-z0-9_.^*])(?<!\*\* )(?<!\^-)(?<!\*\* -)(?P<number>(?>"
    + _NUMBER
    + r"))(?:\s*(?P<unit>"
    + _UNIT
    + r"))?"
)
# "fs = 758 Hz", "fs is 758 Hz", "fs: 758 Hz", "fs equals 758 Hz", "fs was 758 Hz".
_LABEL_BEFORE = re.compile(
    r"(?:^|[\s(,;])(?P<symbol>" + _SYMBOL + r")\s*(?:=|:|\bis\b|\bequals\b|\bwas\b)\s*$"
)
# "1.8 m (di)".
_LABEL_AFTER = re.compile(r"^\s*\(\s*(?P<symbol>" + _SYMBOL + r")\s*\)")

# Words that start the ask, and words that end it.
_ASK = re.compile(
    r"\b(?:what(?:'s| is| was| would| will| does)?|find|calculate|compute|determine|"
    r"work out|solve for|figure out|get|need|want to know|tell me|how (?:large|big|much)|"
    r"unknown:?)\s+",
    re.IGNORECASE,
)
# "of", "for", and "from" don't end an ask: "the mass of the planet", "the time for one
# cycle", "the distance from the pivot" are all one name. A quantity does (the givens
# start), but a plain integer doesn't: it's part of names like "mass 1".
_ASK_END = re.compile(
    r"[?.,;:!=]|\s(?:when|if|given|where|using|with|since|because)\s",
    re.IGNORECASE,
)
_ASK_SPAN_WORDS = 12  # "the value of the time to reach the top" is nine
# "Speed?" or "Final speed = ? when ..." also ask, with no cue word, if the sentence is short.
_BARE_ASK_WORDS = 4
_BARE_ASK = re.compile(r"(?:^|[.!?]\s+)(?P<span>[^.!?=]{1,60}?)\s*(?:=\s*)?\?")


def _quantity(match: re.Match[str]) -> tuple[str, str] | None:
    number = float(match.group("number"))
    if not math.isfinite(number):
        return None
    unit = match.group("unit")
    if unit is not None and not is_valid_unit(unit):
        unit = None
    return format_number(number), unit or "dimensionless"


def labelled_quantities(text: str) -> list[tuple[str, str, str]]:
    """Quantities the question labels with a symbol: (symbol, canonical number, unit).

    "fs is 758 Hz" gives ("fs", "758", "Hz"); "di = 1.8 m" and "1.8 m (di)" both give
    ("di", "1.8", "m"); "mu = 0.3" gives ("mu", "0.3", "dimensionless"). Units and
    numbers are spelled the way ``stated_quantities`` spells them, so they compare equal.
    Which tokens are really symbols is the caller's call: only it knows the equations.
    """
    out: list[tuple[str, str, str]] = []
    for match in _QUANTITY.finditer(text):
        found = _quantity(match)
        if found is None:
            continue
        before = _LABEL_BEFORE.search(text[: match.start()])
        after = _LABEL_AFTER.match(text[match.end() :])
        for label in (before, after):
            if label is not None:
                out.append((label.group("symbol"), *found))
    return out


def names_for(variable: Variable) -> list[str]:
    """Every phrase the data factory may call ``variable``, lowercase."""
    return [n.lower() for n in (variable.name, *VAR_SYNONYMS.get(variable.name, ()))]


def ask_spans(text: str) -> list[str]:
    """The stretches of ``text`` that say what is wanted.

    "Work out the object height if ..." gives "the object height"; "Solve for vs." gives
    "vs"; "... 53 km/h. Final speed?" gives "Final speed". Each span runs from an ask
    word ("what is", "find", "solve for") to the next clause boundary, capped at a few
    words so a run-on question can't drag the givens in.
    """
    spans: list[str] = []
    for cue in _ASK.finditer(text):
        rest = text[cue.end() :]
        end = _ASK_END.search(rest)
        span = rest[: end.start()] if end else rest
        for q in _QUANTITY.finditer(span):
            number, unit = q.group("number"), q.group("unit")
            if (unit and is_valid_unit(unit)) or not number.isdigit():
                span = span[: q.start()]
                break
        words = span.split()
        if words:
            spans.append(" ".join(words[:_ASK_SPAN_WORDS]))
    for bare in _BARE_ASK.finditer(text):
        span = bare.group("span").strip()
        if span and len(span.split()) <= _BARE_ASK_WORDS and not any(c.isdigit() for c in span):
            spans.append(span)
    return spans


def asked_variables(text: str, variables: Iterable[Variable]) -> list[Variable]:
    """The variables an ask in ``text`` names, out of ``variables``.

    In each ask, the name that starts earliest wins, and the longest name at that spot:
    "the mass of the planet" names M ("mass of the planet") over m ("mass"), and "the
    time for one cycle" names T over t. A symbol counts only when it is the whole ask
    ("Solve for vs"), because symbols like "do" and "a" are also English words. Ties are
    all returned, and the caller decides what an ambiguous ask means. ``variables`` may
    mix several equations, so their names compete with each other.
    """
    variables = list(variables)
    found: list[Variable] = []
    for span in ask_spans(text):
        bare = span.removeprefix("the ").strip()
        picked = [v for v in variables if v.symbol == bare]
        if not picked:
            lowered = span.lower()
            best: tuple[int, int] | None = None  # (start, -length): earliest, then longest
            for v in variables:
                for name in names_for(v):
                    hit = re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", lowered)
                    if hit is None:
                        continue
                    rank = (hit.start(), -len(name))
                    if best is None or rank < best:
                        best, picked = rank, [v]
                    elif rank == best and v not in picked:
                        picked.append(v)
        found += [v for v in picked if v not in found]
    return found


def asked_symbols(text: str, variables: Iterable[Variable]) -> set[str]:
    """Symbols of the variables an ask in ``text`` names (see ``asked_variables``)."""
    return {v.symbol for v in asked_variables(text, variables)}


def symbol_locks(text: str, variables: Sequence[Variable]) -> dict[str, tuple[str, str]]:
    """Symbol -> (number, unit) for labels that name exactly one quantity.

    A symbol labelled with two different quantities is dropped as ambiguous.
    """
    symbols = {v.symbol for v in variables}
    locks: dict[str, tuple[str, str]] = {}
    clashes: set[str] = set()
    for symbol, number, unit in labelled_quantities(text):
        if symbol not in symbols:
            continue
        if symbol in locks and locks[symbol] != (number, unit):
            clashes.add(symbol)
        locks[symbol] = (number, unit)
    return {s: q for s, q in locks.items() if s not in clashes}
