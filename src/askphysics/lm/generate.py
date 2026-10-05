"""Constrained decoding for the Fermi models (``docs/MODELS.md``).

The decoder writes every piece of JSON structure itself. The model only fills
slots, and each slot is constrained:

- **Choice slots** (category, equation ids, symbols, numbers, units, origin):
  every legal option is scored by the model's log-probability and the best
  one wins. An equation id that wasn't retrieved, or a number that isn't in
  the input, is never an option, so it can never be emitted.
- **Free-text slots** (reasoning, assumptions, strategy, explanations):
  generated token by token, with a number guard so any digits spell a number
  that appears in the input.

Output text always matches ``lm/formats.py`` byte for byte, which is also
what the models are trained on.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from askphysics.errors import AskPhysicsError, LLMError
from askphysics.lm.formats import (
    classify_prompt,
    explain_numbers,
    explain_prompt,
    extract_numbers,
    format_number,
    plan_numbers,
    plan_prompt,
    plan_units,
    question_quantities,
    relevant_constants,
)
from askphysics.lm.model import FermiLM, KVCache
from askphysics.lm.tokenizer import SPECIAL_TOKENS, Tokenizer, pretokenize
from askphysics.models import (
    Classification,
    Constant,
    Equation,
    FermiAssumption,
    Plan,
    Variable,
)
from askphysics.solver.units import check_dimensions, quantity

CATEGORIES = ("standard", "fermi", "out_of_scope")
DOMAINS = (
    "kinematics",
    "dynamics",
    "energy",
    "momentum",
    "gravitation",
    "electromagnetism",
    "thermodynamics",
    "fluids",
    "waves",
    "optics",
    "modern",
)
ORIGINS = ("given", "constant", "assumption")

# The only number a standard plan may assume without it being stated: "from rest" means v0 = 0.
# The data factory never assumes anything else, and a 1 here let the model invent m = 1 kg.
FILLER_NUMBERS = ("0",)

MAX_EQUATIONS = 3
MAX_DOMAINS = 3
MAX_ASSUMPTIONS = 6

_TRAILING_RUN = re.compile(r"(?<![A-Za-z_0-9.^*])(?<!\*\* )(\d+(?:\.\d*)?)$")
_TRAILING_EXPONENT = re.compile(r"(\^|\*\*\s?)\d*$")
_TRAILING_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def encode_task(tokenizer: Tokenizer, text: str) -> list[int]:
    """Encode task text: a leading task token and a trailing end token are special, nothing else.

    Payloads hold user text, so a ``<|plan|>`` typed inside a question stays plain bytes.
    """
    ids: list[int] = []
    for token in SPECIAL_TOKENS:
        if text.startswith(token):
            ids.append(tokenizer.special_ids[token])
            text = text[len(token) :]
            break
    end = SPECIAL_TOKENS[1]
    has_end = text.endswith(end)
    if has_end:
        text = text[: -len(end)]
    ids.extend(tokenizer.encode(text))
    if has_end:
        ids.append(tokenizer.end_id)
    return ids


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def number_guard_ok(prefix: str, piece: str, allowed: Sequence[str]) -> bool:
    """Would appending ``piece`` keep every number in the text inside ``allowed``?

    Digits glued to letters (``v0``, ``m1``) are identifiers, not numbers, and
    are not checked. Signs are ignored: ``-9.8`` is fine if ``9.8`` is allowed.
    """
    stripped = [a.lstrip("-") for a in allowed]
    match = _TRAILING_RUN.search(prefix)
    run = match.group(1) if match else ""
    if piece[:1].isdigit():
        if not run and _TRAILING_IDENTIFIER.search(prefix):
            return True  # part of an identifier like v0 or m12
        if not run and _TRAILING_EXPONENT.search(prefix):
            return True  # a unit exponent like m^3 or second ** 2
        return any(a.startswith(run + piece) for a in stripped)
    if piece == "." and run:
        return any(a.startswith(run + ".") for a in stripped)
    if run:  # this piece closes the number
        return run.rstrip(".") in stripped
    return True


@dataclass
class _State:
    text: str
    ids: list[int]
    past: KVCache | None
    logits: Tensor | None = field(default=None)


class Decoder:
    """Stateful constrained decoder over one model and tokenizer."""

    def __init__(self, model: FermiLM, tokenizer: Tokenizer, max_slot_tokens: int = 48) -> None:
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.max_slot_tokens = max_slot_tokens
        self.device = next(model.parameters()).device
        vocab = model.config.vocab_size
        usable = torch.zeros(vocab, dtype=torch.bool)
        string_safe = torch.zeros(vocab, dtype=torch.bool)
        prose_safe = torch.zeros(vocab, dtype=torch.bool)
        quote_start = torch.zeros(vocab, dtype=torch.bool)
        special = set(tokenizer.special_ids.values())
        self._text: dict[int, str] = {}
        for i in range(min(vocab, tokenizer.vocab_size)):
            if i in special:
                continue
            raw = tokenizer.token_bytes(i)
            usable[i] = True
            printable = all(0x20 <= b < 0x7F for b in raw)
            if printable:
                self._text[i] = raw.decode("ascii")
                prose_safe[i] = True
                string_safe[i] = b'"' not in raw and b"\\" not in raw
                quote_start[i] = raw.startswith(b'"')
        self._string_safe = string_safe.to(self.device)
        self._prose_safe = prose_safe.to(self.device)
        self._quote_start = quote_start.to(self.device)
        self._ids_by_text = {text: i for i, text in self._text.items()}
        self._ids_by_first: dict[str, list[int]] = {}
        for i, text in self._text.items():
            self._ids_by_first.setdefault(text[:1], []).append(i)
        self._state = _State(text="", ids=[], past=None)

    # ------------------------------------------------------------------ cache plumbing

    @property
    def text(self) -> str:
        return self._state.text

    def _feed(self, past: KVCache | None, tokens: Sequence[int]) -> tuple[KVCache, Tensor]:
        if not tokens:
            raise ValueError("nothing to feed")
        used = 0 if past is None else past[0][0].shape[2]
        if used + len(tokens) > self.model.config.context_length:
            raise LLMError(
                f"input of {used + len(tokens)} tokens exceeds the model's "
                f"{self.model.config.context_length}"
            )
        if past is None:
            ids = torch.tensor([list(tokens)], device=self.device)
            logits, new_past = self.model.step(ids)
            return new_past, logits[0, -1]
        logits_t: Tensor | None = None
        for t in tokens:
            out, past = self.model.step(torch.tensor([[t]], device=self.device), past)
            logits_t = out[0, -1]
        assert past is not None and logits_t is not None
        return past, logits_t

    def _rollback(self, n: int) -> tuple[KVCache | None, Tensor | None, int]:
        """Cache state covering the first ``n - 1`` tokens, ready to re-feed from ``n - 1``."""
        state = self._state
        if n == len(state.ids) and state.logits is not None:
            return state.past, state.logits, n
        keep = max(n - 1, 0)
        if keep == 0 or state.past is None:
            return None, None, 0
        past = [(k[:, :, :keep], v[:, :, :keep]) for k, v in state.past]
        return past, None, keep

    def _sync(self, text: str) -> None:
        ids = encode_task(self.tokenizer, text)
        if len(ids) > self.model.config.context_length:
            raise LLMError(
                f"input of {len(ids)} tokens exceeds the model's {self.model.config.context_length}"
            )
        n = _common_prefix(self._state.ids, ids)
        past, logits, start = self._rollback(n)
        if start < len(ids):
            past, logits = self._feed(past, ids[start:])
        self._state = _State(text=text, ids=ids, past=past, logits=logits)

    def _score(self, text: str) -> float:
        """Log-probability of the tokens ``text`` adds after the current state."""
        ids = encode_task(self.tokenizer, text)
        n = _common_prefix(self._state.ids, ids)
        past, logits, start = self._rollback(n)
        if start < n:  # re-feed the shared tokens we rolled back over
            past, logits = self._feed(past, ids[start:n])
        total = 0.0
        for t in ids[n:]:
            assert logits is not None
            total += float(F.log_softmax(logits.float(), dim=-1)[t])
            past, logits = self._feed(past, [t])
        return total

    # ------------------------------------------------------------------ public slots

    def start(self, prompt: str) -> None:
        self._state = _State(text="", ids=[], past=None)
        self._sync(prompt)

    def emit(self, fixed: str) -> None:
        """Append structure the decoder controls (JSON punctuation, forced values)."""
        self._sync(self._state.text + fixed)

    def choose(self, options: Sequence[str], closer: str = "") -> str:
        """Append the option the model writes when it may only write legal options.

        Greedy decoding through a trie of ``option + closer`` strings: at each
        step only tokens that keep some option reachable are allowed, and the
        model's best one is taken. This matches how the model was trained
        (next-token prediction) and has no bias toward short options, unlike
        comparing summed log-probabilities. ``closer`` marks where an option
        ends (so "1" and "1200" are told apart) and is not kept.
        """
        if not options:
            raise ValueError("no options to choose from")
        if len(options) == 1:
            self.emit(options[0])
            return options[0]
        base = self._state.text
        # Hold back the last pretokenizer chunk: in training it may merge with what follows
        # (`"` + `],` is one chunk `"],`), so the model must be free to write them together.
        cut = self._pending_start(base)
        pending = base[cut:]
        self._sync(base[:cut])
        targets = {pending + option + closer: option for option in options}
        written = ""
        for _ in range(max(len(t) for t in targets) + 1):
            if any(written.startswith(t) for t in targets):
                break
            logits = self._state.logits
            assert logits is not None
            best_id, best_logit = -1, float("-inf")
            for token in self._continuations(written, targets):
                if float(logits[token]) > best_logit:
                    best_id, best_logit = token, float(logits[token])
            if best_id < 0:  # pragma: no cover - single bytes always exist
                raise LLMError("no token can continue any option")
            written += self._text[best_id]
            self._state.past, self._state.logits = self._feed(self._state.past, [best_id])
            self._state.ids.append(best_id)
            self._state.text += self._text[best_id]
        chosen = targets[max((t for t in targets if written.startswith(t)), key=len)]
        self._sync(base + chosen)  # drop the closer and re-canonicalize the tokens
        return chosen

    def _pending_start(self, text: str) -> int:
        """Index where the last pretokenizer chunk of ``text`` starts (after any task token)."""
        start = next((len(t) for t in SPECIAL_TOKENS if text.startswith(t)), 0)
        chunks = pretokenize(text[start:])
        return len(text) - len(chunks[-1]) if chunks else len(text)

    def _continuations(self, written: str, targets: Iterable[str]) -> set[int]:
        """Tokens that keep some target reachable, including ones that run past its end."""
        out: set[int] = set()
        for target in targets:
            if not target.startswith(written):
                continue
            rest = target[len(written) :]
            for k in range(1, len(rest) + 1):
                token = self._ids_by_text.get(rest[:k])
                if token is not None:
                    out.add(token)
            for token in self._ids_by_first.get(rest[:1], ()):
                if self._text[token].startswith(rest):
                    out.add(token)
        return out

    def free_text(
        self,
        allowed_numbers: Sequence[str],
        *,
        until_end_token: bool = False,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> str:
        """Generate text for a JSON string (stops before a quote) or for prose (stops at END)."""
        written = ""
        allowed = self._prose_safe if until_end_token else self._string_safe
        for _ in range(self.max_slot_tokens):
            logits = self._state.logits
            assert logits is not None
            masked = logits.float().masked_fill(~allowed, float("-inf"))
            if until_end_token:
                stop_score = float(logits[self.tokenizer.end_id])
            else:
                stop_score = float(logits.float().masked_fill(~self._quote_start, -1e9).max())
            choice = self._pick(masked, written, allowed_numbers, temperature, generator)
            run = _TRAILING_RUN.search(written)
            number_open = bool(run) and run.group(1).rstrip(".") not in [  # type: ignore[union-attr]
                a.lstrip("-") for a in allowed_numbers
            ]
            if choice is None or (
                not number_open and written and stop_score >= float(masked[choice])
            ):
                break
            piece = self._text[choice]
            written += piece
            self._state.past, self._state.logits = self._feed(self._state.past, [choice])
            self._state.ids.append(choice)
            self._state.text += piece
        # Re-encode so the cache matches the canonical tokenization of the text.
        self._sync(self._state.text)
        return written

    def _pick(
        self,
        masked: Tensor,
        written: str,
        allowed_numbers: Sequence[str],
        temperature: float,
        generator: torch.Generator | None,
    ) -> int | None:
        if temperature > 0:
            probs = F.softmax(masked / temperature, dim=-1)
            for _ in range(8):
                token = int(torch.multinomial(probs, 1, generator=generator))
                if number_guard_ok(written, self._text.get(token, ""), allowed_numbers):
                    return token
        order = torch.argsort(masked, descending=True)
        for token_t in order[:256]:
            token = int(token_t)
            if masked[token] == float("-inf"):
                return None
            if number_guard_ok(written, self._text[token], allowed_numbers):
                return token
        return None


# --------------------------------------------------------------------------- tasks


def decode_classification(decoder: Decoder, question: str) -> Classification:
    """Classify a question; output is always a valid ``Classification``."""
    numbers = extract_numbers(question)
    decoder.start(classify_prompt(question))
    decoder.emit('{"category": "')
    category = decoder.choose(CATEGORIES, closer='"')
    decoder.emit('", "domains": [')
    domains: list[str] = []
    while len(domains) < MAX_DOMAINS:
        remaining = [d for d in DOMAINS if d not in domains]
        sep = ", " if domains else ""
        picked = decoder.choose(["]"] + [f'{sep}"{d}"' for d in remaining])
        if picked == "]":
            break
        domains.append(picked.split('"')[1])
    else:
        decoder.emit("]")
    decoder.emit(', "reasoning": "')
    reasoning = decoder.free_text(numbers)
    decoder.emit('", "closest_answerable": ')
    closest: str | None = None
    if category == "out_of_scope":
        decoder.emit('"')
        closest = decoder.free_text(numbers)
        decoder.emit('"')
    else:
        decoder.emit("null")
    decoder.emit("}")
    return Classification.model_validate(
        {
            "category": category,
            "domains": domains,
            "reasoning": reasoning,
            "closest_answerable": closest,
        }
    )


def _fits(unit: str, variable: Variable) -> bool:
    """Whether a quantity in ``unit`` can fill ``variable`` (same dimensions)."""
    try:
        return check_dimensions(quantity(1.0, unit), variable.unit)
    except AskPhysicsError:
        return False


def _table_constant(variable: Variable, constants: Sequence[Constant]) -> Constant | None:
    """The table constant a variable stands for (``G``, ``R``), as the data factory decides."""
    if "constant" not in variable.name:
        return None
    return next((c for c in constants if _fits(c.unit, variable)), None)


@dataclass(frozen=True)
class ValueOption:
    """A legal (number, unit, origin) for one known value in a plan."""

    number: str
    unit: str
    origin: str


def known_value_options(
    variable: Variable, question: str, constants: Sequence[Constant]
) -> list[ValueOption]:
    """What a standard plan may write for ``variable``.

    A quantity written in the question with matching dimensions (its number and unit stay
    together), a table constant with matching dimensions, or an assumed 0 in the variable's
    own unit ("dropped" means v0 = 0), only where 0 is inside the variable's typical range:
    a speed can start at rest, but g and a mass can't be zero. A mass can never be filled
    with a speed, and "570 pounds" can't turn into 570 kilograms. Noether still checks
    units later; this only stops the model from writing values that could never be right.
    """
    options: list[ValueOption] = []

    def add(option: ValueOption) -> None:
        if option not in options:
            options.append(option)

    for number, unit in question_quantities(question):
        if _fits(unit, variable):
            add(ValueOption(number, unit, "given"))
    for c in constants:
        if _fits(c.unit, variable):
            add(ValueOption(format_number(c.value), c.unit, "constant"))
    low, high = variable.typical_range or (1.0, 0.0)  # no range: assume nothing
    for number in FILLER_NUMBERS:
        if low <= float(number) <= high:
            add(ValueOption(number, variable.unit, "assumption"))
    return options


def assignable_options(
    options: Sequence[ValueOption],
    variable: Variable,
    unused: Sequence[tuple[str, str]],
    pending: Sequence[Variable],
) -> list[ValueOption]:
    """Narrow ``known_value_options`` to what is still free as a plan is written.

    Each quantity in the question fills at most one variable, so ``unused`` holds the
    (number, unit) pairs not yet taken. A table constant or an assumed 0 is only
    allowed when the variables still to fill (``pending``, including ``variable``) outnumber
    the unused quantities that fit them; otherwise a stated value would go unused, and a
    question never states a value for nothing. Falls back to ``options`` if nothing is left.
    """
    fitting = [q for q in unused if _fits(q[1], variable)]
    slots = sum(1 for v in pending if _fits(v.unit, variable))
    fillers_ok = slots > len(fitting)
    narrowed = [
        o
        for o in options
        if (o.origin == "given" and (o.number, o.unit) in fitting)
        or (o.origin != "given" and fillers_ok)
    ]
    return narrowed or list(options)


def target_options(
    variables: Sequence[Variable], question: str, constants: Sequence[Constant]
) -> list[str]:
    """Symbols a standard plan may solve for: ones the question doesn't already give.

    A variable is a candidate when the question states fewer quantities of its dimensions
    than the equations have variables of those dimensions. "Given a force, two masses,
    find..." leaves only the distance. Table constants are never targets, and a variable a
    constant could fill (g) only is when nothing else is open. Falls back to every
    non-constant symbol if the count rules them all out.
    """
    free = [v for v in variables if _table_constant(v, constants) is None]
    given = question_quantities(question)
    out: list[Variable] = []
    for v in free:
        slots = sum(1 for other in free if _fits(other.unit, v))
        stated = sum(1 for _, unit in given if _fits(unit, v))
        if stated < slots:
            out.append(v)
    # A variable a table constant can fill (g by standard gravity) is only the unknown if no
    # open variable lacks such a fallback: "lifting it 11 m took 11000 J, what is its mass?"
    # asks for m, and g comes from the table.
    without_fallback = [v for v in out if not any(_fits(c.unit, v) for c in constants)]
    picked = without_fallback or out
    return [v.symbol for v in picked] or [v.symbol for v in free] or [v.symbol for v in variables]


def decode_plan(
    decoder: Decoder,
    question: str,
    category: str,
    equations: Sequence[Equation],
    constants: Sequence[Constant],
    fermi: Sequence[FermiAssumption] = (),
) -> Plan:
    """Write a plan that can only cite ``equations`` and numbers present in the input.

    Standard plans are also dimension-checked as they are written: the target must be a
    variable the question leaves open, each known value must be a quantity whose units fit
    its variable, and each stated quantity is used once before any constant or 0 fills a
    slot (``target_options``, ``known_value_options``, ``assignable_options``).
    """
    if not equations:
        raise LLMError("no retrieved equations to plan with")
    fermi_used = fermi if category == "fermi" else ()
    constants = relevant_constants(equations, constants)
    numbers = plan_numbers(question, constants, fermi_used)
    units = plan_units(question, equations, constants, fermi_used)
    by_id = {eq.id: eq for eq in equations}
    dimensional = category == "standard"

    decoder.start(plan_prompt(question, category, equations, constants, fermi_used))
    decoder.emit('{"equation_ids": [')
    chosen = [decoder.choose([f'"{i}"' for i in by_id], closer="]").strip('"')]
    while len(chosen) < min(MAX_EQUATIONS, len(by_id)):
        remaining = [i for i in by_id if i not in chosen]
        picked = decoder.choose(["]"] + [f', "{i}"' for i in remaining])
        if picked == "]":
            break
        chosen.append(picked.split('"')[1])
    else:
        decoder.emit("]")

    variables: dict[str, Variable] = {}
    for eid in chosen:
        for v in by_id[eid].variables:
            variables.setdefault(v.symbol, v)
    symbols = list(variables)

    decoder.emit(', "target": "')
    targets = (
        target_options(list(variables.values()), question, constants) if dimensional else symbols
    )
    target = decoder.choose(targets, closer='"')
    decoder.emit(f'", "unknowns": ["{target}"')
    unknowns = [target]
    while True:
        remaining = [s for s in symbols if s not in unknowns]
        if not remaining:
            decoder.emit("]")
            break
        picked = decoder.choose(["]"] + [f', "{s}"' for s in remaining])
        if picked == "]":
            break
        unknowns.append(picked.split('"')[1])

    decoder.emit(', "known_values": [')
    knowns = []
    to_fill = [s for s in symbols if s not in unknowns]
    unused = question_quantities(question)
    for i, symbol in enumerate(to_fill):
        decoder.emit(("" if i == 0 else ", ") + f'{{"symbol": "{symbol}", "value": ')
        options = (
            assignable_options(
                known_value_options(variables[symbol], question, constants),
                variables[symbol],
                unused,
                [variables[s] for s in to_fill[i:]],
            )
            if dimensional
            else []
        )
        # Nothing fits (a mass with no mass stated and nothing to assume): fall back to the
        # loose rules, and Noether's checks turn the plan into an honest degraded answer.
        if options:
            value = decoder.choose(list(dict.fromkeys(o.number for o in options)), closer=", ")
            decoder.emit(', "unit": "')
            fitting = [o for o in options if o.number == value]
            unit = decoder.choose(list(dict.fromkeys(o.unit for o in fitting)), closer='"')
            origins = [o.origin for o in fitting if o.unit == unit]
        else:
            value = decoder.choose(numbers, closer=", ")
            decoder.emit(', "unit": "')
            unit = decoder.choose(units, closer='"')
            origins = list(ORIGINS)
        decoder.emit('", "origin": "')
        origin = decoder.choose(list(dict.fromkeys(origins)), closer='"')
        if origin == "given" and (value, unit) in unused:
            unused.remove((value, unit))
        decoder.emit('"}')
        knowns.append({"symbol": symbol, "value": float(value), "unit": unit, "origin": origin})
    decoder.emit('], "assumptions": [')

    assumptions: list[str] = []
    picked = decoder.choose(["]", '"'])
    while picked != "]":
        assumptions.append(decoder.free_text(numbers))
        if len(assumptions) >= MAX_ASSUMPTIONS:
            decoder.emit('"]')
            break
        picked = decoder.choose(['"]', '", "'])
        if picked == '"]':
            break
    decoder.emit(', "strategy": "')
    strategy = decoder.free_text(numbers)
    decoder.emit('"}')
    return Plan.model_validate(
        {
            "equation_ids": chosen,
            "target": target,
            "unknowns": unknowns,
            "known_values": knowns,
            "assumptions": assumptions,
            "strategy": strategy,
        }
    )


def decode_explanation(
    decoder: Decoder,
    question: str,
    value: float,
    unit: str,
    equations: Sequence[Equation],
    assumptions: Sequence[str],
    issues: Sequence[str] = (),
    *,
    temperature: float = 0.0,
    seed: int = 0,
) -> str:
    """Write prose about a computed result; digits can only spell numbers from the input."""
    decoder.start(explain_prompt(question, value, unit, equations, assumptions, issues))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    numbers = explain_numbers(question, value, assumptions)
    return decoder.free_text(
        numbers, until_end_token=True, temperature=temperature, generator=generator
    ).strip()


__all__ = [
    "Decoder",
    "ValueOption",
    "assignable_options",
    "decode_classification",
    "decode_explanation",
    "decode_plan",
    "encode_task",
    "known_value_options",
    "number_guard_ok",
    "target_options",
]
