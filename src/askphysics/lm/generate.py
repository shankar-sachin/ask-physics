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
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache

import torch
import torch.nn.functional as F
from torch import Tensor

from askphysics.errors import AskPhysicsError, LLMError, PlanValidationError
from askphysics.lm import templates as tpl
from askphysics.lm.formats import (
    FILLER_NUMBERS,
    classify_prompt,
    dumps,
    explain_numbers,
    explain_prompt,
    format_number,
    plan_numbers,
    plan_prompt,
    plan_units,
    question_quantities,
    relevant_constants,
    stated_quantities,
    value_numbers,
)
from askphysics.lm.model import FermiLM, KVCache
from askphysics.lm.reading import (
    asked_symbols,
    asked_variables,
    can_start_at_zero,
    contradicted,
    gravity_cue,
    light_cue,
    mentions,
    names_for,
    particle_cue,
    rest_cue,
    starts_process,
    stated_givens,
    steady_cue,
    symbol_locks,
    transition_locks,
)
from askphysics.lm.tokenizer import SPECIAL_TOKENS, Tokenizer, pretokenize
from askphysics.models import (
    Classification,
    Constant,
    Equation,
    FermiAssumption,
    Plan,
    Variable,
)
from askphysics.prose import GENERIC_REASONS, content_words
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

MAX_EQUATIONS = 3
MAX_DOMAINS = 3
MAX_ASSUMPTIONS = 6
# Free text may not repeat a run of this many words, so a small model can't loop
# ("roughly roughly roughly ...").
NO_REPEAT_WORDS = 6

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


def repeats(history: Sequence[int], token: int, text: Mapping[int, str]) -> bool:
    """Whether writing ``token`` after ``history`` would start a loop.

    A word token may not follow itself ("roughly roughly"), and a run of
    ``NO_REPEAT_WORDS`` word tokens may not appear twice. Tokens with digits or symbols are
    exempt, so "3 m/s and 4 m/s" is fine.
    """

    def word(t: int) -> bool:
        return text.get(t, "").strip().isalpha()

    if not word(token):
        return False
    if history and history[-1] == token and len(text[token].strip()) >= 3:
        return True
    n = NO_REPEAT_WORDS
    gram = (*history[len(history) - (n - 1) :], token) if len(history) >= n - 1 else ()
    if not gram or not all(word(t) for t in gram):
        return False
    return any(tuple(history[i : i + n]) == gram for i in range(len(history) - n + 1))


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
        history: list[int] = []
        allowed = self._prose_safe if until_end_token else self._string_safe
        for _ in range(self.max_slot_tokens):
            logits = self._state.logits
            assert logits is not None
            masked = logits.float().masked_fill(~allowed, float("-inf"))
            if until_end_token:
                stop_score = float(logits[self.tokenizer.end_id])
            else:
                stop_score = float(logits.float().masked_fill(~self._quote_start, -1e9).max())
            choice = self._pick(masked, written, history, allowed_numbers, temperature, generator)
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
            history.append(choice)
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
        history: Sequence[int],
        allowed_numbers: Sequence[str],
        temperature: float,
        generator: torch.Generator | None,
    ) -> int | None:
        def ok(token: int) -> bool:
            piece = self._text.get(token, "")
            return number_guard_ok(written, piece, allowed_numbers) and not repeats(
                history, token, self._text
            )

        if temperature > 0:
            probs = F.softmax(masked / temperature, dim=-1)
            for _ in range(8):
                token = int(torch.multinomial(probs, 1, generator=generator))
                if ok(token):
                    return token
        order = torch.argsort(masked, descending=True)
        for token_t in order[:256]:
            token = int(token_t)
            if masked[token] == float("-inf"):
                return None
            if ok(token):
                return token
        return None


# --------------------------------------------------------------------------- tasks


def decode_classification(decoder: Decoder, question: str) -> Classification:
    """Classify a question; output is always a valid ``Classification``."""
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
    reasoning = _choose_text(decoder, reason_options(category, question))
    decoder.emit('", "closest_answerable": ')
    closest: str | None = None
    if category == "out_of_scope":
        decoder.emit('"')
        closest = _choose_text(decoder, redirect_options(question, reasoning))
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


# A slot ends at punctuation, the end, or a sign-off ("... a promise Thanks!"). A sign-off
# needs a space before it: "Ty" must not end "anxie|ty".
_END_OF_ASK = (
    "(?=\\s*(?:[?.!]|$)|\\s+(?:"
    + "|".join(re.escape(x.rstrip(".!")) for x in tpl.SIGN_OFFS)
    + ")\\b)"
)


@cache
def _template_pattern(text: str) -> re.Pattern[str]:
    """An out-of-scope question template as a regex whose slots capture the question's words."""
    parts = re.split(r"(\{\w+\})", text.rstrip("?.!"))
    body = "".join(f"(?P<{p[1:-1]}>.+?)" if p.startswith("{") else re.escape(p) for p in parts)
    return re.compile(body + _END_OF_ASK, re.IGNORECASE)


def template_matches(question: str) -> list[tuple[int, dict[str, str]]]:
    """(index into ``OUT_OF_SCOPE``, slot values) for each template ``question`` matches.

    "How fast is a promise?" matches "How fast is {abstract}?" with abstract = "a promise".
    A slot is only ever filled with the words in its own position, never a stray run of
    words ("a dropped ball take does not move").
    """
    out: list[tuple[int, dict[str, str]]] = []
    for i, (template, _, _) in enumerate(tpl.OUT_OF_SCOPE):
        match = _template_pattern(template.text).search(question)
        if match is not None:
            out.append((i, {k: v.strip() for k, v in match.groupdict().items()}))
    return out


def reason_options(category: str, question: str) -> list[str]:
    """The reasons a classification may give: reviewed sentences, never free writing.

    Standard and Fermi reasons come from the data factory's lists. An out-of-scope reason
    with a slot ("Category error: {emotion} is a feeling") is offered only when the
    question matches its template, filled with the words in that slot's position. A fixed
    one that names something ("a dream is an experience") must share a word with the
    question; a generic one ("Not a physics question; it is history.") always fits.
    """
    if category == "standard":
        return [r.format(domain=d) for d in DOMAINS for r in tpl.STANDARD_REASONING]
    if category == "fermi":
        return list(tpl.FERMI_REASONING)
    words = content_words(question)
    out = [tpl.OUT_OF_SCOPE[i][1].format(**slots) for i, slots in template_matches(question)]
    for reason in dict.fromkeys(r for _, r, _ in tpl.OUT_OF_SCOPE):
        if "{" not in reason and (
            reason.startswith(GENERIC_REASONS) or content_words(reason) & words
        ):
            out.append(reason)
    return list(dict.fromkeys(out))


def redirect_options(question: str, reason: str | None = None) -> list[str]:
    """The answerable questions a refusal may suggest, given the chosen ``reason``.

    Only redirects written for that reason are offered (a math refusal suggests the math
    redirect), with slots filled from the matching template. A redirect whose slot can't
    be filled that way is never offered.
    """
    matched = dict(template_matches(question))
    out: list[str] = []
    for i, (_, r, closest) in enumerate(tpl.OUT_OF_SCOPE):
        slots = matched.get(i, {})
        try:
            filled_reason, filled = r.format(**slots), closest.format(**slots)
        except KeyError:
            continue  # a slot the question didn't fill
        if reason is None or filled_reason == reason:
            out.append(filled)
    if not out:
        out = [c for _, _, c in tpl.OUT_OF_SCOPE if "{" not in c]
    return list(dict.fromkeys(out))


def _choose_text(decoder: Decoder, options: Sequence[str]) -> str:
    """Write one of ``options`` inside a JSON string and return it unescaped."""
    by_escaped = {dumps(o)[1:-1]: o for o in options}
    return by_escaped[decoder.choose(list(by_escaped), closer='"')]


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


def constant_fills(constant: Constant, variable: Variable, question: str) -> bool:
    """Whether a table constant may fill ``variable``: only a slot that is that constant.

    That is a variable named "... constant" with the constant's units (``G``, ``k``, ``R``,
    ``c`` in the relativity equations) or one with the constant's own symbol. Matching units
    are never enough: ``v = c`` is a speed of 299792458 m/s, which is how "an average speed of
    23.2 m/s" once became a light-speed plan (#91). Three constants also fill the plain
    variable they are the value of when the question says so: standard gravity for an
    acceleration when something falls, the speed of light for a speed when the question is
    about light, and the elementary charge for a charge when it names an electron or proton.
    """
    if not _fits(constant.unit, variable):
        return False
    if "constant" in variable.name or variable.symbol == constant.symbol:
        return True
    if starts_process(variable) or variable.name.lower().startswith("final"):
        return False  # a start or an end is a stated value, never a constant
    if constant.name == "standard_gravity":
        return variable.name == "acceleration" and gravity_cue(question)
    if constant.name == "speed_of_light":
        return light_cue(question)
    if constant.name == "elementary_charge":
        return particle_cue(question)
    return False


def zero_allowed(variable: Variable, question: str) -> bool:
    """Whether a plan may assume ``variable`` is 0 for this question.

    Only a starting speed or position, and only when the question says the thing starts at
    rest ("from rest", "dropped", "released"); a stated "from 0 m/s" is a given value, not an
    assumption. Never a momentum, force, mass, or time, and a final speed or an asked-for
    quantity is never one (the target is not a known value at all). The one other zero is an
    acceleration when the motion is steady ("cruises at a steady 30 m/s"). 0 must also be
    inside the variable's typical range.
    """
    low, high = variable.typical_range or (1.0, 0.0)  # no range: assume nothing
    if not low <= 0.0 <= high:
        return False
    if can_start_at_zero(variable):
        return rest_cue(question)
    return variable.name == "acceleration" and steady_cue(question)


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
    together), the table constant that is this variable (``constant_fills``), or an assumed 0
    in the variable's own unit where the question says it starts at rest (``zero_allowed``:
    "dropped" means v0 = 0, but a momentum, mass, or time is never 0). A mass can never be
    filled with a speed, and "570 pounds" can't turn into 570 kilograms. Noether still checks
    units later; this only stops the model from writing values that could never be right.
    """
    options: list[ValueOption] = []

    def add(option: ValueOption) -> None:
        if option not in options:
            options.append(option)

    for number, unit in stated_quantities(question):
        if _fits(unit, variable):
            add(ValueOption(number, unit, "given"))
    for c in constants:
        if constant_fills(c, variable, question):
            add(ValueOption(format_number(c.value), c.unit, "constant"))
    if zero_allowed(variable, question):
        for number in FILLER_NUMBERS:
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
    given = stated_quantities(question)
    out: list[Variable] = []
    for v in free:
        slots = sum(1 for other in free if _fits(other.unit, v))
        stated = sum(1 for _, unit in given if _fits(unit, v))
        if stated < slots:
            out.append(v)
    # A variable a table constant can fill (g by standard gravity) is only the unknown if no
    # open variable lacks such a fallback: "lifting it 11 m took 11000 J, what is its mass?"
    # asks for m, and g comes from the table.
    without_fallback = [
        v for v in out if not any(constant_fills(c, v, question) for c in constants)
    ]
    picked = without_fallback or out
    # A symbol the question labels with a value ("fs is 758 Hz") is given, not wanted, and
    # a variable the ask names ("what is its mass?") is the one wanted. Either rule only
    # narrows: if it would rule everything out, it is ignored.
    locks = quantity_locks(question, variables)
    picked = [v for v in picked if v.symbol not in locks] or picked
    asked = asked_symbols(question, picked)
    picked = [v for v in picked if v.symbol in asked] or picked
    return [v.symbol for v in picked] or [v.symbol for v in free] or [v.symbol for v in variables]


def quantity_locks_bare(question: str, eq: Equation) -> list[str]:
    """The bare numbers the question labels with one of ``eq``'s variables."""
    return [n for n, u in quantity_locks(question, eq.variables).values() if u == "dimensionless"]


def quantity_locks(question: str, variables: Sequence[Variable]) -> dict[str, tuple[str, str]]:
    """Symbol -> the (number, unit) the question labels it with, when the units fit.

    Includes the initial and final value of a "from A to B" (``transition_locks``)."""
    by_symbol = {v.symbol: v for v in variables}
    locks = {
        s: q for s, q in symbol_locks(question, variables).items() if _fits(q[1], by_symbol[s])
    }
    # "from 20 m/s to 60 m/s" is the initial and then the final value, unless a label differs.
    moved = transition_locks(question, variables)
    if (
        moved
        and not any(s in locks for s in moved)
        and not set(moved.values()) & set(locks.values())
    ):
        locks.update(moved)
    return locks


def _partner(symbol: str) -> str | None:
    """The other half of an indexed pair: "m1" and "m2", "v2" and "v1"."""
    if symbol[-1:] == "1":
        return symbol[:-1] + "2"
    if symbol[-1:] == "2":
        return symbol[:-1] + "1"
    return None


def ordered_options(
    options: Sequence[ValueOption],
    symbol: str,
    locks: Mapping[str, tuple[str, str]],
    unused: Sequence[tuple[str, str]],
    pending: Sequence[str],
) -> list[ValueOption]:
    """With neither half of a pair labelled, the first value stated goes to the "1" variable.

    "One has mass 0.293 kg and speed 9.27 km/h, the other mass 140 kg and speed 31 mph":
    v1 is 9.27 km/h because it is stated first. Falls back to ``options``.
    """
    partner = _partner(symbol)
    if partner is None or partner not in pending or symbol in locks or partner in locks:
        return list(options)
    given = [(o.number, o.unit) for o in options if o.origin == "given"]
    order = [q for q in unused if q in given]
    if len(order) < 2:
        return list(options)
    want = order[0] if symbol.endswith("1") else order[-1]
    narrowed = [o for o in options if o.origin != "given" or (o.number, o.unit) == want]
    return narrowed or list(options)


def locked_options(
    options: Sequence[ValueOption],
    symbol: str,
    locks: Mapping[str, tuple[str, str]],
    unused: Sequence[tuple[str, str]],
    pending: Sequence[str],
) -> list[ValueOption]:
    """Narrow a variable's options to agree with the question's labels.

    A labelled symbol ("di is 1.8 m") may only take its own quantity, and other variables
    may not take a quantity a label reserves for a symbol still to be written (``pending``),
    unless the question states it more than once. Falls back to ``options`` if nothing is
    left, so a wrong label can never leave a variable without a value.
    """
    if symbol in locks:
        narrowed = [
            o for o in options if o.origin == "given" and (o.number, o.unit) == locks[symbol]
        ]
        return narrowed or list(options)
    reserved = Counter(locks[s] for s in pending if s in locks and s != symbol)
    narrowed = [
        o
        for o in options
        if o.origin != "given" or unused.count((o.number, o.unit)) > reserved[(o.number, o.unit)]
    ]
    return narrowed or list(options)


def _unfilled_unitless(eq: Equation, question: str, asked: Sequence[Variable]) -> bool:
    """Whether ``eq`` has a unitless variable the question never mentions, states, or asks
    for ("the Lorentz factor is unknown... what is it?" mentions one)."""
    labelled = {
        s for s, (_, u) in quantity_locks(question, eq.variables).items() if u == "dimensionless"
    }
    return any(
        v.unit == "dimensionless"
        and v.symbol not in labelled
        and v not in asked
        and not mentions(question, names_for(v))
        and not known_value_options(v, question, [])
        for v in eq.variables
    )


def _dimensions(unit: str) -> str:
    return str(quantity(1.0, unit).dimensionality)


def equation_options(equations: Sequence[Equation], question: str) -> list[str]:
    """Ids a standard plan may start with: equations with room for every stated quantity.

    Each quantity the question writes with a unit must fit a variable, and no dimensions may
    be stated more often than the equation has variables of those dimensions. "A 620 J thing
    moving at 4.1 m/s, what is its mass?" rules out KE = p^2/2m, which has nowhere to put the
    speed (the model would otherwise assume p = 0). Bare numbers are left out because "the
    resistance 2" is a label, not a value. Falls back to every id if nothing has room.
    """
    stated = question_quantities(question)

    def same_dimensions(a: str, b: str) -> bool:
        try:
            return check_dimensions(quantity(1.0, a), b)
        except AskPhysicsError:
            return False

    def has_room(eq: Equation) -> bool:
        for _, unit in stated:
            slots = sum(1 for v in eq.variables if _fits(unit, v))
            same = sum(1 for _, other in stated if same_dimensions(other, unit))
            if slots < same:
                return False
        return True

    roomy = [eq for eq in equations if has_room(eq)] or list(equations)
    # When no equation with room has the variable the ask names, the answer takes two steps
    # ("the force, mass, and time are given: how far does it go?"): start from the
    # equations that do have it, and chain to the rest.
    # Kinds of quantity are compared, not names: "its speed" names the speed in v = d/t,
    # but the final velocity in v = v0 + at is a speed too.
    asked_any = asked_variables(question, [v for eq in equations for v in eq.variables])
    asked_kinds = {_dimensions(v.unit) for v in asked_any}
    if asked_any and not any(
        _dimensions(v.unit) in asked_kinds for eq in roomy for v in eq.variables
    ):
        roomy = [eq for eq in equations if any(v in asked_any for v in eq.variables)]
    # "in parallel" rules out the series formula, whose variables are otherwise identical.
    roomy = [eq for eq in roomy if not contradicted(question, eq, equations)] or roomy
    # A bare number the question labels ("the emissivity comes out to 0.017") needs a home
    # too: prefer the equations with a variable it names.
    labelled = {n for n, u in stated_givens(question, roomy) if u == "dimensionless"}
    if labelled:
        housing = [eq for eq in roomy if labelled <= set(quantity_locks_bare(question, eq))]
        roomy = housing or roomy
    # A unitless variable (an emissivity, a coefficient) needs its value stated or asked
    # for: "the area is 1200 cm^2 and the temperature 610 K, what power?" rules out the
    # law with an emissivity, which would otherwise borrow the 1200.
    asked_here = asked_variables(question, [v for eq in roomy for v in eq.variables])
    roomy = [eq for eq in roomy if not _unfilled_unitless(eq, question, asked_here)] or roomy
    # Prefer equations with the variable the ask names: "what is its mass?" rules out
    # W = Fd. Names compete across equations, so "time to reach the top" beats "time".
    asked = asked_variables(question, [v for eq in roomy for v in eq.variables])
    named = [eq for eq in roomy if any(v in asked for v in eq.variables)]
    return [eq.id for eq in named or roomy]


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
    slot (``target_options``, ``known_value_options``, ``assignable_options``). A known value
    with no legal option raises ``PlanValidationError``: nothing is filled in for it.

    Raises:
        LLMError: there are no equations to plan with.
        PlanValidationError: a standard plan's known value has no legal number.
    """
    if not equations:
        raise LLMError("no retrieved equations to plan with")
    fermi_used = fermi if category == "fermi" else ()
    constants = relevant_constants(equations, constants)
    numbers = plan_numbers(question, constants, fermi_used)
    values = value_numbers(question, constants, fermi_used)
    units = plan_units(question, equations, constants, fermi_used)
    by_id = {eq.id: eq for eq in equations}
    dimensional = category == "standard"

    decoder.start(plan_prompt(question, category, equations, constants, fermi_used))
    decoder.emit('{"equation_ids": [')
    first = equation_options(equations, question) if dimensional else list(by_id)
    chosen = [decoder.choose([f'"{i}"' for i in first], closer="]").strip('"')]
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
    unused = stated_quantities(question)
    locks = quantity_locks(question, list(variables.values())) if dimensional else {}
    for i, symbol in enumerate(to_fill):
        decoder.emit(("" if i == 0 else ", ") + f'{{"symbol": "{symbol}", "value": ')
        options = (
            ordered_options(
                locked_options(
                    assignable_options(
                        known_value_options(variables[symbol], question, constants),
                        variables[symbol],
                        unused,
                        [variables[s] for s in to_fill[i:]],
                    ),
                    symbol,
                    locks,
                    unused,
                    to_fill[i:],
                ),
                symbol,
                locks,
                unused,
                to_fill[i:],
            )
            if dimensional
            else []
        )
        if options:
            value = decoder.choose(list(dict.fromkeys(o.number for o in options)), closer=", ")
            decoder.emit(', "unit": "')
            fitting = [o for o in options if o.number == value]
            unit = decoder.choose(list(dict.fromkeys(o.unit for o in fitting)), closer='"')
            origins = [o.origin for o in fitting if o.unit == unit]
        elif dimensional:
            # Nothing fits (a mass the question never states): the plan fails, and the router
            # tries again or degrades. A filler here is how the model wrote m = 1 kg (#85).
            raise PlanValidationError(
                f"no legal value for {symbol}: not in the question, the tables, or an assumed 0"
            )
        else:
            value = decoder.choose(values, closer=", ")
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
    if dimensional:
        assumptions = _choose_assumptions(decoder, assumption_options([by_id[e] for e in chosen]))
        picked = "]"
    else:
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


def assumption_options(equations: Sequence[Equation]) -> list[str]:
    """The assumptions a standard plan may list: the equations' own, and their scenarios'.

    These are exactly the phrasings the data factory writes (``Equation.assumptions`` and
    ``Scenario.assumptions``), so a plan picks whole reviewed sentences instead of writing
    its own: a small model can't garble "Point masses or spherically symmetric bodies".
    """
    ids = {eq.id for eq in equations}
    out: list[str] = []
    for text in [
        *(a for eq in equations for a in eq.assumptions),
        *(a for sc in tpl.SCENARIOS if sc.equation in ids for a in sc.assumptions),
    ]:
        if text not in out:
            out.append(text)
    return out


def _choose_assumptions(decoder: Decoder, options: Sequence[str]) -> list[str]:
    """Pick assumptions one at a time from ``options``, each at most once, then close the list."""
    chosen: list[str] = []
    remaining = list(options)
    while remaining and len(chosen) < MAX_ASSUMPTIONS:
        lead = ", " if chosen else ""
        by_text = {lead + dumps(a): a for a in remaining}
        picked = decoder.choose(["]", *by_text])
        if picked == "]":
            return chosen
        chosen.append(by_text[picked])
        remaining.remove(by_text[picked])
    decoder.emit("]")
    return chosen


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
    "assumption_options",
    "decode_classification",
    "decode_explanation",
    "decode_plan",
    "encode_task",
    "equation_options",
    "known_value_options",
    "locked_options",
    "number_guard_ok",
    "ordered_options",
    "quantity_locks",
    "reason_options",
    "redirect_options",
    "repeats",
    "target_options",
]
