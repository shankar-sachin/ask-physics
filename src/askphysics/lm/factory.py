"""The data factory: training examples for the Fermi models, generated from our own data.

No external model writes anything here (ADR-009). Standard problems are built
from the equation database and **solved by Noether** (SymPy + Pint)
before they're kept; anything it can't solve is dropped. Every plan is
checked against the same allowed numbers and units the constrained decoder
uses, so training targets are always outputs the decoder could produce.

Examples whose question resembles an eval question are dropped
(``docs/EVALS.md``, leakage policy).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from askphysics.data.loader import DataStore
from askphysics.errors import AskPhysicsError
from askphysics.lm import templates as tpl
from askphysics.lm.config import FORMAT_VERSION
from askphysics.lm.formats import (
    classify_prompt,
    explain_prompt,
    format_number,
    plan_numbers,
    plan_prompt,
    plan_units,
    relevant_constants,
    serialize_classification,
    serialize_plan,
)
from askphysics.lm.reading import mentions, own_tags, twins
from askphysics.lm.tokenizer import END
from askphysics.models import Classification, Constant, Equation, KnownValue, Plan, Variable
from askphysics.retrieval.keyword import KeywordRetriever
from askphysics.solver.symbolic import solve_for
from askphysics.solver.units import check_dimensions, quantity, unit_string

Task = Literal["classify", "plan", "explain"]
LEAK_THRESHOLD = 0.7
RETRIES_PER_TARGET = 6
_WORDS = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class Example:
    """One training example: the model reads ``prompt`` and learns to write ``target``."""

    task: Task
    split: Literal["train", "val"]
    template: str
    prompt: str
    target: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True)


@dataclass(frozen=True)
class StandardProblem:
    question: str
    template: str
    equation: Equation
    retrieved: list[Equation]
    constants: list[Constant]
    plan: Plan
    value: float
    unit: str


def word_overlap(a: str, b: str) -> float:
    """Jaccard similarity of the word sets of two strings."""
    wa, wb = set(_WORDS.findall(a.lower())), set(_WORDS.findall(b.lower()))
    return len(wa & wb) / len(wa | wb) if wa | wb else 0.0


def equation_phrase(name: str) -> str:
    """Name an equation mid-sentence: "Newton's second law", "the ideal gas law"."""
    first = name.split()[0]
    if first.endswith("'s") or (first[:1].isupper() and first[1:2].isupper()):
        return name
    return "the " + name[:1].lower() + name[1:]


def _join(phrases: Sequence[str], style: int = 0) -> str:
    """List phrases: "a, b and c" (style 0), "a, b, and c" (1), or "a; b; c" (2)."""
    if len(phrases) == 1:
        return phrases[0]
    if style == 2:
        return "; ".join(phrases)
    comma = "," if style == 1 and len(phrases) > 2 else ""
    return ", ".join(phrases[:-1]) + f"{comma} and " + phrases[-1]


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def _article(name: str) -> str:
    return ("an " if name[:1].lower() in "aeiou" else "a ") + name


def plain_name(var_name: str) -> str:
    """A name for a variable with no digits ("second mass", not "mass 2").

    Model-written text may only contain numbers from the question, so prose never says
    "mass 2" unless the question did.
    """
    for name in tpl.VAR_SYNONYMS.get(var_name, ()):
        if not any(ch.isdigit() for ch in name):
            return name
    return var_name


def _is_plain_word(token: str) -> bool:
    """A word whose case can change freely: not a symbol like "V", "KE", or "v0"."""
    return len(token) > 1 and token[1:].isalpha() and token[1:].islower()


@dataclass(frozen=True)
class _Phrase:
    text: str
    symbol_first: bool  # starts with a case-sensitive symbol, so never recapitalize it

    def sentence(self) -> str:
        return (self.text if self.symbol_first else _capitalize(self.text)) + "."


class DataFactory:
    """Deterministic example generator for a given ``seed``."""

    def __init__(self, store: DataStore, seed: int = 0, blocklist: Sequence[str] = ()) -> None:
        self.store = store
        self.rng = random.Random(seed)
        self.blocklist = list(blocklist)
        self.constants = list(store.constants.values())
        self.by_symbol = {c.symbol: c for c in self.constants}
        self.retriever = KeywordRetriever(store.equations.values(), store.examples.values())
        self.dropped = 0

    # ------------------------------------------------------------------ helpers

    def _constant_for(self, var_name: str, unit: str) -> Constant | None:
        """The table constant a variable stands for (``G``, ``R``), if it is one."""
        if "constant" not in var_name:
            return None
        for c in self.constants:
            if check_dimensions(quantity(1.0, c.unit), unit):
                return c
        return None

    def _sample(
        self,
        unit: str,
        *,
        adjective: bool = False,
        typical: tuple[float, float] | None = None,
    ) -> tuple[str, str, str]:
        """A realistic value for a variable measured in ``unit``.

        Returns (number text, unit as written, the two as they appear in the question).
        Units are sometimes converted ("km/h"), spelled out ("meters"), or written without
        a space ("20m"); adjectives ("a 5 kg ball") keep the symbol. A variable whose
        ``typical`` range lies outside the everyday range for its unit (a molecule's mass,
        an electron's charge) is sampled from its own range instead.
        """
        low, high = tpl.FRIENDLY_RANGES.get(unit, (1.0, 1000.0))
        if typical is not None and typical[1] > 0 and (typical[1] < low or typical[0] > high):
            high = typical[1]
            low = max(typical[0], high * 1e-3)
        if unit == "dimensionless":  # a coefficient or a count: a bare number in its own range
            low, high = typical if typical and typical[0] > 0 else (0.1, 10.0)
            value = math.exp(self.rng.uniform(math.log(low), math.log(high)))
            number = format_number(float(f"{value:.{self.rng.choice((2, 2, 3))}g}"))
            return number, unit, number
        value = math.exp(self.rng.uniform(math.log(low), math.log(high)))
        shown_unit = self.rng.choice(tpl.ALT_UNITS.get(unit, (unit,)))
        shown = quantity(value, unit).to(shown_unit).magnitude
        digits = self.rng.choice((2, 2, 3))
        number = format_number(float(f"{shown:.{digits}g}"))
        if "." not in number and "e" not in number and self.rng.random() < 0.05:
            number += ".0"
        spellings = tpl.UNIT_SPELLINGS.get(shown_unit)
        if spellings and not adjective and self.rng.random() < 0.3:
            spelled = self.rng.choice(spellings)
            return number, spelled, f"{number} {spelled}"
        gap = "" if self.rng.random() < 0.1 else " "
        return number, shown_unit, f"{number}{gap}{shown_unit}"

    def _name(self, var_name: str) -> str:
        return self.rng.choice(tpl.VAR_SYNONYMS.get(var_name, (var_name,)))

    def _lower_first(self, text: str) -> str:
        words = text.split(" ", 2)
        first = words[0].rstrip(",:.!?")
        article = first == "A" and len(words) > 1 and words[1] != "="
        return text[:1].lower() + text[1:] if article or _is_plain_word(first) else text

    def _dress(self, question: str) -> str:
        """Casual framing: an opener, a sign-off, dropped punctuation, a lowercase start."""
        if self.rng.random() < 0.1:
            question = question.rstrip("?.")
        if self.rng.random() < 0.08:
            question = self._lower_first(question)
        if self.rng.random() < 0.2:
            preamble = self.rng.choice(tpl.PREAMBLES)
            if preamble[-1] not in ".:!?":
                question = self._lower_first(question)
            question = f"{preamble} {question}"
        if self.rng.random() < 0.1:
            question = f"{question} {self.rng.choice(tpl.SIGN_OFFS)}"
        return question

    def _leaks(self, question: str) -> bool:
        return any(word_overlap(question, b) >= LEAK_THRESHOLD for b in self.blocklist)

    def _retrieved(self, question: str, gold: Equation) -> list[Equation]:
        hits = [
            h.equation for h in self.retriever.search(question, 3, domains=[gold.domain]).equations
        ]
        if gold.id not in {e.id for e in hits}:
            hits = hits[:2]
            hits.insert(self.rng.randrange(len(hits) + 1), gold)
        return hits

    # ------------------------------------------------------------------ standard problems

    def standard_problem(self) -> StandardProblem | None:
        """One solved standard problem, or None if this attempt failed a check.

        The equation and then the target are drawn uniformly, so every unknown is asked
        for about equally often ("find v0" as often as "find v"). A worded scenario for
        that equation and target is used when one exists, otherwise a generic question.
        """
        eq = self.rng.choice(list(self.store.equations.values()))
        targets = [v for v in eq.variables if self._constant_for(v.name, v.unit) is None]
        target = self.rng.choice(targets)
        # Some targets fail often (a random v0 = v - a*t is frequently negative), so retry
        # the same equation and target rather than letting easy targets crowd them out.
        for _ in range(RETRIES_PER_TARGET):
            problem = self._problem_for(eq, target)
            if problem is not None:
                return problem
        return None

    def _problem_for(self, eq: Equation, target: Variable) -> StandardProblem | None:
        scenarios = [
            sc for sc in tpl.SCENARIOS if sc.equation == eq.id and sc.target == target.symbol
        ]
        if scenarios and self.rng.random() < 0.6:
            return self._scenario_problem(self.rng.choice(scenarios))
        frame = self.rng.choice(tpl.GENERIC)
        pattern = self.rng.choice(tpl.KNOWN_PATTERNS)
        mixed = self.rng.random() < 0.25
        used = [pattern]
        knowns: list[KnownValue] = []
        phrases: list[_Phrase] = []
        for v in eq.variables:
            if v.symbol == target.symbol:
                continue
            const = self._constant_for(v.name, v.unit)
            if const is not None:
                knowns.append(
                    KnownValue(
                        symbol=v.symbol, value=const.value, unit=const.unit, origin="constant"
                    )
                )
                continue
            number, unit, shown = self._sample(v.unit, typical=v.typical_range)
            knowns.append(
                KnownValue(symbol=v.symbol, value=float(number), unit=unit, origin="given")
            )
            if mixed:
                pattern = self.rng.choice(tpl.KNOWN_PATTERNS)
                used.append(pattern)
            name = self._name(v.name)
            text = pattern.text.format(name=name, a_name=_article(name), sym=v.symbol, q=shown)
            phrases.append(_Phrase(text, pattern.text.startswith("{sym}")))
        if self.rng.random() < 0.5:
            self.rng.shuffle(phrases)
        knowns_text = _join([p.text for p in phrases], self.rng.randrange(3))
        target_name = self._name(target.name)
        question = frame.text.format(
            target=target_name,
            Target=_capitalize(target_name),
            tsym=target.symbol,
            knowns=knowns_text,
            Knowns=knowns_text if phrases[0].symbol_first else _capitalize(knowns_text),
            facts=" ".join(p.sentence() for p in phrases),
        )
        pattern_id = "kp_mix" if mixed else pattern.id
        if mixed and any(p.held_out for p in used):
            pattern_id += "_h"
        return self._finish(
            self._dress(question),
            f"{frame.id}+{pattern_id}",
            eq,
            target.symbol,
            knowns,
            list(eq.assumptions),
        )

    def _scenario_problem(self, sc: tpl.Scenario | None = None) -> StandardProblem | None:
        sc = sc or self.rng.choice(tpl.SCENARIOS)
        eq = self.store.equations[sc.equation]
        forced = {f[0]: f for f in sc.forced}
        objects = self.rng.sample(tpl.OBJECTS, 2)
        vehicles = self.rng.sample(tpl.VEHICLES, 2)
        slots: dict[str, str] = {
            "object": objects[0],
            "object2": objects[1],
            "vehicle": vehicles[0],
            "vehicle2": vehicles[1],
        }
        knowns: list[KnownValue] = []
        for v in eq.variables:
            if v.symbol == sc.target:
                continue
            if v.symbol in forced:
                _, value, unit, origin = forced[v.symbol]
                table = self.by_symbol.get(value)
                number = table.value if table else float(value)
                unit = table.unit if table else unit
                knowns.append(KnownValue(symbol=v.symbol, value=number, unit=unit, origin=origin))
                continue
            const = self._constant_for(v.name, v.unit)
            if const is not None:
                knowns.append(
                    KnownValue(
                        symbol=v.symbol, value=const.value, unit=const.unit, origin="constant"
                    )
                )
                continue
            adjective = "{" + v.symbol + "_a}" in sc.template.text
            number_text, unit, shown = self._sample(
                v.unit, adjective=adjective, typical=v.typical_range
            )
            knowns.append(
                KnownValue(symbol=v.symbol, value=float(number_text), unit=unit, origin="given")
            )
            slots[v.symbol] = slots[v.symbol + "_a"] = shown
        question = self._dress(sc.template.text.format(**slots))
        assumptions = [*sc.assumptions, *eq.assumptions]
        return self._finish(question, sc.template.id, eq, sc.target, knowns, assumptions)

    def _finish(
        self,
        question: str,
        template_id: str,
        eq: Equation,
        target: str,
        knowns: list[KnownValue],
        assumptions: list[str],
    ) -> StandardProblem | None:
        context = tpl.EQUATION_CONTEXT.get(eq.id)
        if context and any(
            not mentions(question, own_tags(eq, t))
            for t in twins(eq, self.store.equations.values())
        ):
            question = f"{self.rng.choice(context)} {question}"
        if self._leaks(question):
            self.dropped += 1
            return None
        try:
            outcome = solve_for(eq, target, {k.symbol: quantity(k.value, k.unit) for k in knowns})
        except AskPhysicsError:
            self.dropped += 1
            return None
        value = float(outcome.value.magnitude)
        if not math.isfinite(value) or value <= 0:
            self.dropped += 1
            return None
        retrieved = self._retrieved(question, eq)
        constants = relevant_constants(retrieved, self.constants)
        target_name = plain_name(eq.variable(target).name)
        phrase = equation_phrase(eq.name)
        plan = Plan(
            equation_ids=[eq.id],
            target=target,
            unknowns=[target],
            known_values=knowns,
            assumptions=assumptions,
            strategy=self.rng.choice(tpl.STRATEGIES).format(
                eq=phrase, Eq=_capitalize(phrase), target=target_name, sym=target
            ),
        )
        # Training targets must be outputs the constrained decoder could produce.
        allowed_numbers = set(plan_numbers(question, constants))
        allowed_units = set(plan_units(question, retrieved, constants))
        if any(
            format_number(k.value) not in allowed_numbers or k.unit not in allowed_units
            for k in knowns
        ):
            self.dropped += 1
            return None
        return StandardProblem(
            question=question,
            template=template_id,
            equation=eq,
            retrieved=retrieved,
            constants=constants,
            plan=plan,
            value=float(f"{value:.6g}"),
            unit=unit_string(outcome.value.units),
        )

    # ------------------------------------------------------------------ examples per task

    @staticmethod
    def _split(template_id: str) -> Literal["train", "val"]:
        """Validation if any part of the template id ("gen_01+kp_10_h") is held out."""
        return "val" if any(p.endswith("_h") for p in template_id.split("+")) else "train"

    def plan_example(self) -> Example | None:
        p = self.standard_problem()
        if p is None:
            return None
        prompt = plan_prompt(p.question, "standard", p.retrieved, p.constants)
        return Example("plan", self._split(p.template), p.template, prompt, serialize_plan(p.plan))

    def explain_example(self) -> Example | None:
        p = self.standard_problem()
        if p is None:
            return None
        template = self.rng.choice(tpl.EXPLAIN)
        assume = ""
        if p.plan.assumptions:
            listed = "; ".join(a[:1].lower() + a[1:] for a in p.plan.assumptions)
            assume = self.rng.choice(tpl.ASSUME_LEADS).format(listed=listed)
        text = template.text.format(
            id=p.equation.id,
            name=p.equation.name,
            Name=_capitalize(p.equation.name),
            target=plain_name(p.equation.variable(p.plan.target).name),
            value=format_number(p.value),
            unit=p.unit,
            assume=assume,
        )
        prompt = explain_prompt(p.question, p.value, p.unit, [p.equation], p.plan.assumptions)
        split = "val" if template.held_out else self._split(p.template)
        return Example("explain", split, f"{p.template}+{template.id}", prompt, text + END)

    def classify_example(self) -> Example | None:
        roll = self.rng.random()
        if roll < 0.5:
            p = self.standard_problem()
            if p is None:
                return None
            reasoning = self.rng.choice(tpl.STANDARD_REASONING).format(domain=p.equation.domain)
            c = Classification(
                category="standard", reasoning=reasoning, domains=[p.equation.domain]
            )
            template_id = p.template
            question = p.question
        elif roll < 0.68:
            template = self.rng.choice(tpl.FERMI)
            question = self._dress(
                template.text.format(**{k: self.rng.choice(v) for k, v in tpl.FERMI_SLOTS.items()})
            )
            domains = ["energy"] if "energy" in question else []
            c = Classification(
                category="fermi", reasoning=self.rng.choice(tpl.FERMI_REASONING), domains=domains
            )
            template_id = template.id
        elif roll < 0.75:
            # Every question a refusal suggests must itself be answerable: tellus once
            # refused "How long does a dropped ball take to fall from a table?", which our
            # own math refusal had just suggested.
            template, _, closest = self.rng.choice(tpl.OUT_OF_SCOPE)
            slots = {k: self.rng.choice(v) for k, v in tpl.OOS_SLOTS.items()}
            question = self._dress(closest.format(**slots))
            domains = ["energy"] if "energy" in question else []
            c = Classification(
                category="fermi", reasoning=self.rng.choice(tpl.FERMI_REASONING), domains=domains
            )
            template_id = f"redirect_{template.id}"
        else:
            template, reason, closest = self.rng.choice(tpl.OUT_OF_SCOPE)
            slots = {k: self.rng.choice(v) for k, v in tpl.OOS_SLOTS.items()}
            question = self._dress(template.text.format(**slots))
            c = Classification(
                category="out_of_scope",
                reasoning=reason.format(**slots),
                closest_answerable=closest.format(**slots),
            )
            template_id = template.id
        if self._leaks(question):
            self.dropped += 1
            return None
        return Example(
            "classify",
            self._split(template_id),
            template_id,
            classify_prompt(question),
            serialize_classification(c),
        )

    # ------------------------------------------------------------------ the stream

    def examples(
        self, n: int, mix: tuple[float, float, float] = (0.35, 0.45, 0.20)
    ) -> Iterator[Example]:
        """Yield ``n`` examples; ``mix`` is the (classify, plan, explain) share."""
        made = 0
        attempts = 0
        while made < n:
            attempts += 1
            if attempts > 20 * n + 100:
                raise RuntimeError("data factory keeps failing; check the templates and data")
            roll = self.rng.random()
            if roll < mix[0]:
                example = self.classify_example()
            elif roll < mix[0] + mix[1]:
                example = self.plan_example()
            else:
                example = self.explain_example()
            if example is not None:
                made += 1
                yield example


# --------------------------------------------------------------------------- datasets on disk

_EVAL_QUESTION = re.compile(r'^\s*(?:-\s*)?question:\s*"(.*)"\s*$')


def load_blocklist(path: Path) -> list[str]:
    """Questions from an eval YAML file (``evals/questions.yaml``), no YAML parser needed."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _EVAL_QUESTION.match(line)
        if match:
            out.append(match.group(1))
    return out


def _write_shard(args: tuple[Path, int, int, int, list[str]]) -> dict[str, Any]:
    out_dir, shard, n, seed, blocklist = args
    from askphysics.data.loader import load_all

    factory = DataFactory(load_all(), seed=seed, blocklist=blocklist)
    counts: Counter[str] = Counter()
    files = {split: (out_dir / split / f"shard-{shard:03d}.jsonl") for split in ("train", "val")}
    for path in files.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with (
        files["train"].open("w", encoding="utf-8") as train,
        files["val"].open("w", encoding="utf-8") as val,
    ):
        for example in factory.examples(n):
            (val if example.split == "val" else train).write(example.to_json() + "\n")
            counts[f"{example.split}/{example.task}"] += 1
    return {"counts": dict(counts), "dropped": factory.dropped}


def build_dataset(
    out_dir: Path,
    examples: int,
    seed: int = 0,
    workers: int = 1,
    blocklist: Sequence[str] = (),
) -> dict[str, Any]:
    """Write ``examples`` examples as JSONL shards in ``out_dir/{train,val}/``, plus a manifest."""
    workers = max(1, workers)
    per = [examples // workers + (1 if i < examples % workers else 0) for i in range(workers)]
    jobs = [(out_dir, i, n, seed * 1000 + i, list(blocklist)) for i, n in enumerate(per) if n]
    if workers == 1:
        results = [_write_shard(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_write_shard, jobs))
    counts: Counter[str] = Counter()
    for r in results:
        counts.update(r["counts"])
    manifest = {
        "format_version": FORMAT_VERSION,
        "seed": seed,
        "examples": examples,
        "workers": workers,
        "counts": dict(sorted(counts.items())),
        "dropped": sum(r["dropped"] for r in results),
        "blocklist_size": len(blocklist),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def read_examples(directory: Path) -> Iterator[Example]:
    """Every example in a directory of JSONL shards, in shard order."""
    for path in sorted(directory.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield Example(**json.loads(line))
