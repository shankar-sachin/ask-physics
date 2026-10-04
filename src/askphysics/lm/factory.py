"""The data factory: training examples for the Fermi models, generated from our own data.

No external model writes anything here (ADR-009). Standard problems are built
from the equation database and **solved by the symbolic algebra machine**
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
from askphysics.lm.tokenizer import END
from askphysics.models import Classification, Constant, Equation, KnownValue, Plan
from askphysics.retrieval.keyword import KeywordRetriever
from askphysics.solver.symbolic import solve_for
from askphysics.solver.units import check_dimensions, quantity

Task = Literal["classify", "plan", "explain"]
LEAK_THRESHOLD = 0.7
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


def _join(phrases: Sequence[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    return ", ".join(phrases[:-1]) + " and " + phrases[-1]


class DataFactory:
    """Deterministic example generator for a given ``seed``."""

    def __init__(self, store: DataStore, seed: int = 0, blocklist: Sequence[str] = ()) -> None:
        self.store = store
        self.rng = random.Random(seed)
        self.blocklist = list(blocklist)
        self.constants = list(store.constants.values())
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

    def _sample(self, unit: str) -> tuple[str, str]:
        """A realistic value for a variable measured in ``unit``: (number text, unit as written)."""
        low, high = tpl.FRIENDLY_RANGES.get(unit, (1.0, 1000.0))
        value = math.exp(self.rng.uniform(math.log(low), math.log(high)))
        shown_unit = self.rng.choice(tpl.ALT_UNITS.get(unit, (unit,)))
        shown = quantity(value, unit).to(shown_unit).magnitude
        digits = self.rng.choice((2, 2, 3))
        return format_number(float(f"{shown:.{digits}g}")), shown_unit

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
        """One solved standard problem, or None if this attempt failed a check."""
        if self.rng.random() < 0.35:
            return self._scenario_problem()
        eq = self.rng.choice(list(self.store.equations.values()))
        targets = [v for v in eq.variables if self._constant_for(v.name, v.unit) is None]
        target = self.rng.choice(targets)
        template = self.rng.choice(tpl.GENERIC)
        knowns: list[KnownValue] = []
        phrases: list[str] = []
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
            number, unit = self._sample(v.unit)
            knowns.append(
                KnownValue(symbol=v.symbol, value=float(number), unit=unit, origin="given")
            )
            phrases.append(f"the {v.name} is {number} {unit}")
        knowns_text = _join(phrases)
        question = template.text.format(
            target=target.name,
            Target=target.name.capitalize(),
            knowns=knowns_text,
            Knowns=knowns_text[:1].upper() + knowns_text[1:],
        )
        return self._finish(question, template.id, eq, target.symbol, knowns, list(eq.assumptions))

    def _scenario_problem(self) -> StandardProblem | None:
        eq_id, template, target, forced = self.rng.choice(tpl.SCENARIOS)
        eq = self.store.equations[eq_id]
        forced_symbols = {f[0] for f in forced}
        knowns: list[KnownValue] = []
        slots: dict[str, str] = {
            "object": self.rng.choice(tpl.OBJECTS),
            "vehicle": self.rng.choice(tpl.VEHICLES),
        }
        g = self.store.constants["standard_gravity"]
        for v in eq.variables:
            if v.symbol == target:
                continue
            if v.symbol in forced_symbols:
                _, value, unit, origin = next(f for f in forced if f[0] == v.symbol)
                number = g.value if value == "g" else float(value)
                unit = g.unit if value == "g" else unit
                knowns.append(KnownValue(symbol=v.symbol, value=number, unit=unit, origin=origin))
                continue
            number_text, unit = self._sample(v.unit)
            knowns.append(
                KnownValue(symbol=v.symbol, value=float(number_text), unit=unit, origin="given")
            )
            slots[v.symbol] = f"{number_text} {unit}"
        question = template.text.format(**slots)
        assumptions = [*tpl.SCENARIO_ASSUMPTIONS.get(template.id, ()), *eq.assumptions]
        return self._finish(question, template.id, eq, target, knowns, assumptions)

    def _finish(
        self,
        question: str,
        template_id: str,
        eq: Equation,
        target: str,
        knowns: list[KnownValue],
        assumptions: list[str],
    ) -> StandardProblem | None:
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
        target_name = eq.variable(target).name
        plan = Plan(
            equation_ids=[eq.id],
            target=target,
            unknowns=[target],
            known_values=knowns,
            assumptions=assumptions,
            strategy=self.rng.choice(
                (
                    f"Use {equation_phrase(eq.name)} and solve for the {target_name}.",
                    f"Rearrange {equation_phrase(eq.name)} for the {target_name}.",
                    f"Solve {equation_phrase(eq.name)} for {target}.",
                )
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
            unit=str(outcome.value.units),
        )

    # ------------------------------------------------------------------ examples per task

    @staticmethod
    def _split(template_id: str) -> Literal["train", "val"]:
        return "val" if template_id.endswith("_h") else "train"

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
            assume = f" This assumes {listed}."
        text = template.text.format(
            id=p.equation.id,
            name=p.equation.name,
            target=p.equation.variable(p.plan.target).name,
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
        elif roll < 0.75:
            template = self.rng.choice(tpl.FERMI)
            question = template.text.format(
                **{k: self.rng.choice(v) for k, v in tpl.FERMI_SLOTS.items()}
            )
            domains = ["energy"] if "energy" in question else []
            c = Classification(
                category="fermi", reasoning=self.rng.choice(tpl.FERMI_REASONING), domains=domains
            )
            template_id = template.id
        else:
            template, reason, closest = self.rng.choice(tpl.OUT_OF_SCOPE)
            slots = {k: self.rng.choice(v) for k, v in tpl.OOS_SLOTS.items()}
            question = template.text.format(**slots)
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
