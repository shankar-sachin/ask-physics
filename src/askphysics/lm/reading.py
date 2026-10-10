"""Facts a question states outright, read with rules instead of the model.

Things a small model gets wrong even when the question spells them out:

- **Labels.** "the source frequency fs is 758 Hz" says 758 Hz is ``fs``. tellus
  still swapped it with the heard frequency, because both are frequencies.
- **The ask.** "Lifting a book took 207 J. What is its mass?" says the unknown
  is a mass. tellus solved for a force instead.
- **Transitions and cues.** "from 20 m/s to 60 m/s" says 20 is the initial speed and 60 the
  final one; solem wrote them backwards. "dropped" and "from rest" are the only reasons to
  assume a starting speed of 0 (``transition_locks``, ``rest_cue``).

``labelled_quantities`` and ``asked_names`` read those facts, and the plan
decoder narrows its options to agree with them (ADR-015). These rules only
narrow choices among numbers and ids already in the input: they never compute.
When a rule doesn't fire, or contradicts the dimensions, the decoder ignores it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

from askphysics.errors import AskPhysicsError
from askphysics.lm.formats import format_number, quantity_ranges, question_quantities
from askphysics.lm.templates import VAR_SYNONYMS
from askphysics.models import Equation, Variable
from askphysics.solver.units import check_dimensions, is_valid_unit, quantity

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


# A name labels the quantity right after it: "primary turns: 61000", "the speed at the end
# comes out to 160 mph", "a mass of 5 kg". Searched back only to the start of the clause.
# The connector may be missing: "the second mass 1100 kg".
_NAME_CONNECTOR = re.compile(
    r"\s*(?:=|:|\bis\b|\bequals\b|\bwas measured at\b|\bwas\b|\bcomes? out to\b|\bof\b)?\s*$",
    re.IGNORECASE,
)
_CLAUSE_START = re.compile(r"[,;.!?]|\b(?:and|when|if|given|that|where|while)\b", re.IGNORECASE)
# ... or the quantity comes first: "110 m for the distance to the object".
_FOR_NAME = re.compile(r"^\s*for\s+(?:the\s+)?(?P<rest>[^,;.!?]*)", re.IGNORECASE)


def _named(segment: str, variables: Sequence[Variable], *, suffix: bool) -> Variable | None:
    """The variable whose name ends (or starts) ``segment``, longest name first; None if tied.

    A trailing symbol is allowed after the name: "the image distance di is 1.8 m".
    """
    words = segment.split()
    if suffix and words and any(words[-1] == v.symbol for v in variables):
        words = words[:-1]
    text = " ".join(words).lower()
    best, picked = 0, []
    for v in variables:
        for name in names_for(v):
            hit = (
                text == name or text.endswith(" " + name)
                if suffix
                else text == name or text.startswith(name + " ")
            )
            if hit and len(name) > best:
                best, picked = len(name), [v]
            elif hit and len(name) == best and v not in picked:
                picked.append(v)
    return picked[0] if len(picked) == 1 else None


def name_labels(text: str, variables: Sequence[Variable]) -> list[tuple[str, str, str]]:
    """Quantities the question labels with a variable's name: (symbol, number, unit).

    "primary turns: 61000" gives Np, "the speed at the end comes out to 160 mph" gives v,
    "110 m for the distance to the object" gives do. These are the data factory's own
    ways of stating a value, and the way people write them too.
    """
    variables = list(variables)
    out: list[tuple[str, str, str]] = []
    for match in _QUANTITY.finditer(text):
        found = _quantity(match)
        if found is None:
            continue
        labels: set[str] = set()
        before = text[: match.start()]
        connector = _NAME_CONNECTOR.search(before)
        # Without a connector, only a value with a unit is labelled: "mass 2" is a name.
        if connector is not None and not connector.group().strip() and found[1] == "dimensionless":
            connector = None
        if connector is not None:
            head = before[: connector.start()]
            start = max((c.end() for c in _CLAUSE_START.finditer(head)), default=0)
            clause = head[start:]
            # "Find the heat out: 0.0052 kJ for the work output": that name is the ask.
            if not _ASK.search(clause + " "):
                v = _named(clause, variables, suffix=True)
                if v is not None:
                    labels.add(v.symbol)
        # A bare number's match may have taken the next word as a unit: "0.036 for the".
        end = match.end("number") if found[1] == "dimensionless" else match.end()
        after = _FOR_NAME.match(text[end:])
        if after is not None:
            v = _named(after.group("rest"), variables, suffix=False)
            if v is not None:
                labels.add(v.symbol)
        if len(labels) == 1:  # a value claimed by two names is not locked to either
            out.append((labels.pop(), *found))
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
    if not found:
        found = idiom_variables(text, variables)
    return found


# Asks that name a kind of quantity rather than a variable: "how fast", "how far". Each maps
# to words a variable's name must contain, and words it must not: "how fast does it hit
# the floor" wants a final speed, not the initial one, unless the start is what's asked.
_IDIOMS: tuple[tuple[re.Pattern[str], tuple[str, ...], tuple[str, ...]], ...] = (
    (  # "What speed did a go-kart start at if it reaches 160 mph ...?"
        re.compile(
            r"\b(?:how fast|what (?:speed|velocity))\b[^.?!]*"
            r"\b(?:start(?:s|ed)?|begin|began)\s+(?:at|with|out at|off at)\b",
            re.I,
        ),
        ("initial", "starting", "launch"),
        (),
    ),
    (
        # "after starting from rest" says how it began, not what is asked.
        re.compile(
            r"\bhow fast\b[^.?!]*\b(?:start|initial|launch|thrown|throw|begin)"
            r"(?![a-z]*\s+from\s+rest)",
            re.I,
        ),
        ("initial", "starting", "launch"),
        (),
    ),
    (re.compile(r"\bhow fast\b", re.I), ("speed", "velocity"), ("initial", "starting", "launch")),
    (re.compile(r"\bhow far\b", re.I), ("distance", "displacement", "height", "range"), ()),
    (
        re.compile(r"\b(?:how high|what height)\b", re.I),
        ("height", "displacement", "distance", "depth"),
        (),
    ),
    (re.compile(r"\bhow heavy\b", re.I), ("mass", "weight"), ()),
    # "How strongly do they attract each other?" asks for a force, not an energy.
    (re.compile(r"\bhow (?:strongly|hard)\b", re.I), ("force",), ()),
    (
        re.compile(r"\bhow long (?:does|will|did|would|until)\b|\bhow much time\b", re.I),
        ("time", "duration", "period"),
        (),
    ),
)


def idiom_variables(text: str, variables: Iterable[Variable]) -> list[Variable]:
    """Variables an idiomatic ask ("how fast", "how far", "what height") points at.

    Only the first idiom that matches counts, so "how fast was it at the start" means the
    initial speed and nothing else.
    """
    variables = list(variables)
    for pattern, include, exclude in _IDIOMS:
        if not pattern.search(text):
            continue

        def words(v: Variable) -> set[str]:
            return {w for name in names_for(v) for w in name.split()}

        return [v for v in variables if words(v) & set(include) and not words(v) & set(exclude)]
    return []


def asked_symbols(text: str, variables: Iterable[Variable]) -> set[str]:
    """Symbols of the variables an ask in ``text`` names (see ``asked_variables``)."""
    return {v.symbol for v in asked_variables(text, variables)}


def symbol_locks(text: str, variables: Sequence[Variable]) -> dict[str, tuple[str, str]]:
    """Symbol -> (number, unit) for labels that name exactly one quantity.

    A label is the variable's symbol ("fs is 758 Hz") or one of its names ("primary
    turns: 61000"). A symbol labelled with two different quantities is dropped as ambiguous.
    """
    symbols = {v.symbol for v in variables}
    locks: dict[str, tuple[str, str]] = {}
    clashes: set[str] = set()
    for symbol, number, unit in [*labelled_quantities(text), *name_labels(text, variables)]:
        if symbol not in symbols:
            continue
        if symbol in locks and locks[symbol] != (number, unit):
            clashes.add(symbol)
        locks[symbol] = (number, unit)
    return {s: q for s, q in locks.items() if s not in clashes}


# --------------------------------------------------------------------------- twins and givens


def mentions(text: str, phrases: Iterable[str]) -> bool:
    """Whether ``text`` contains any of ``phrases`` as whole words, ignoring case."""
    lowered = text.lower()
    return any(re.search(rf"(?<![a-z]){re.escape(p.lower())}(?![a-z])", lowered) for p in phrases)


def _signature(eq: Equation) -> frozenset[tuple[str, str]]:
    return frozenset((v.symbol, str(quantity(1, v.unit).dimensionality)) for v in eq.variables)


def twins(eq: Equation, equations: Iterable[Equation]) -> list[Equation]:
    """Equations with exactly ``eq``'s variables: series vs parallel resistors, orbital vs
    escape speed. Only the question's words can tell them apart."""
    sig = _signature(eq)
    return [o for o in equations if o.id != eq.id and _signature(o) == sig]


def own_tags(eq: Equation, twin: Equation) -> list[str]:
    """Tags ``eq`` has and its twin doesn't: "series" for series resistors."""
    return sorted(set(eq.tags) - set(twin.tags))


def contradicted(question: str, eq: Equation, others: Iterable[Equation]) -> bool:
    """Whether the question names a twin of ``eq`` and not ``eq`` ("in parallel" for the
    series formula)."""
    return any(
        mentions(question, own_tags(t, eq)) and not mentions(question, own_tags(eq, t))
        for t in twins(eq, others)
    )


def stated_givens(question: str, equations: Iterable[Equation]) -> list[tuple[str, str]]:
    """Values a question states that a plan should use: (number, unit) pairs.

    Every number written with a unit, plus bare numbers a variable's symbol or name labels
    ("the emissivity comes out to 0.017", "eps = 0.392"). Unlabelled bare numbers are left
    out, because they are as often labels themselves ("resistor 2").
    """
    equations = list(equations)
    # Only a unitless variable can label a bare number: "resistance 2 for the total
    # resistance" is an index, not a resistance of 2.
    unitless = {
        v.symbol for eq in equations for v in eq.variables if quantity(1, v.unit).dimensionless
    }
    bare = {
        n for s, n, u in labelled_quantities(question) if u == "dimensionless" and s in unitless
    }
    for eq in equations:
        bare |= {
            n
            for s, n, u in name_labels(question, eq.variables)
            if u == "dimensionless" and s in unitless
        }
    return [*question_quantities(question), *((n, "dimensionless") for n in sorted(bare))]


# --------------------------------------------------------------------------- start and end


# Variables that hold a quantity at the start of the process, and the end of it.
_START_WORDS = {"initial", "starting", "launch"}
_END_WORDS = {"final"}


def starts_process(variable: Variable) -> bool:
    """Whether ``variable`` is an initial value by its database name: "initial velocity",
    "launch speed". Other names are never read as a start, whatever their unit."""
    return bool(set(variable.name.lower().split()) & _START_WORDS)


def can_start_at_zero(variable: Variable) -> bool:
    """Whether ``variable`` is a value that can be 0 because the process starts at rest.

    An initial or launch speed or position, and the indexed speeds before a collision
    ("velocity 1", "velocity 2": one cart is parked). A final speed, an acceleration, a
    force, a mass, a momentum, or a time is never one, even when 0 is inside its typical range.
    """
    name = variable.name.lower()
    return starts_process(variable) or re.fullmatch(r"velocity [12]", name) is not None


def _fits(unit: str, variable: Variable) -> bool:
    try:
        return check_dimensions(quantity(1.0, unit), variable.unit)
    except AskPhysicsError:
        return False


def transition_locks(text: str, variables: Sequence[Variable]) -> dict[str, tuple[str, str]]:
    """Symbol -> (number, unit) for a "from A to B" that names an initial and a final value.

    "change its speed from 20 m/s to 60 m/s" gives the initial speed 20 and the final speed
    60, the order they are written. It only applies when exactly one variable is an initial
    value ("initial velocity") and exactly one a final value ("final velocity") whose units
    fit A and B, and when exactly one "from A to B" in the text fits them; anything less
    clear gives nothing.
    """
    starts = [v for v in variables if starts_process(v)]
    ends = [v for v in variables if set(v.name.lower().split()) & _END_WORDS]
    if len(starts) != 1 or len(ends) != 1:
        return {}
    start, end = starts[0], ends[0]
    fitting = [(a, b) for a, b in quantity_ranges(text) if _fits(a[1], start) and _fits(b[1], end)]
    if len(fitting) != 1:
        return {}
    a, b = fitting[0]
    return {start.symbol: a, end.symbol: b}


# What the question says that makes a starting value 0 an assumption the reader can see.
_REST_CUE = re.compile(
    r"\b(?:(?:from|at) rest|dropped|drops|released?|lets? go|stationary|motionless|"
    r"(?:sitting|standing|sits|stands) still|parked|(?:from )?a standstill|standing start|"
    r"starts? (?:a |the )?race|falls?|falling|fell|free[- ]fall)\b",
    re.IGNORECASE,
)
# ... and steady motion, the only reason to take an acceleration as 0.
_STEADY_CUE = re.compile(
    r"\b(?:steady|constant (?:speed|velocity)|cruis(?:es|ed|ing)|uniform (?:speed|velocity)|"
    r"(?:without|not) accelerating|no acceleration|zero acceleration)\b",
    re.IGNORECASE,
)


def rest_cue(text: str) -> bool:
    """Whether ``text`` says something starts at rest ("from rest", "dropped", "released")."""
    return _REST_CUE.search(text) is not None


def steady_cue(text: str) -> bool:
    """Whether ``text`` says the motion is steady ("cruises at a steady 30 m/s")."""
    return _STEADY_CUE.search(text) is not None


# ... and things falling, the only reason to read a free acceleration as standard gravity.
_GRAVITY_CUE = re.compile(
    r"\b(?:dropped|drops?|falls?|fell|falling|free[- ]fall(?:ing)?|gravity|gravitational|"
    r"thrown|throws?|tossed|released?|lets? go|knocked|projectile|plummet\w*|"
    r"(?:off|from) (?:a|an|the) (?:\w+ ){0,2}(?:ledge|cliff|balcony|roof|tower|bridge|building))\b",
    re.IGNORECASE,
)


def gravity_cue(text: str) -> bool:
    """Whether ``text`` has something falling or thrown, so an acceleration may be g."""
    return _GRAVITY_CUE.search(text) is not None


# ... and light or radiation, the only reason to read a speed as the speed of light ...
_LIGHT_CUE = re.compile(
    r"\b(?:speed of light|electromagnetic|radio (?:waves?|signals?)|microwaves?|x-?rays?|"
    r"photons?|lasers?|radar|light (?:waves?|rays?|beams?|signals?|pulses?|years?)|"
    r"(?:red|orange|yellow|green|blue|violet|visible|ultraviolet|infrared|white) light)\b",
    re.IGNORECASE,
)
# ... and an electron or proton, the only reason to read a charge as the elementary charge.
_PARTICLE_CUE = re.compile(r"\b(?:electrons?|protons?|positrons?)\b", re.IGNORECASE)


def light_cue(text: str) -> bool:
    """Whether ``text`` is about light or other radiation, so a speed may be ``c``."""
    return _LIGHT_CUE.search(text) is not None


def particle_cue(text: str) -> bool:
    """Whether ``text`` names an electron or proton, so a charge may be ``e``."""
    return _PARTICLE_CUE.search(text) is not None
