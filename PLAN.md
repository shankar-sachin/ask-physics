# Ask Physics: Master Plan

This is the source of truth for what Ask Physics is, how it works, and how we
know it works. Detailed docs live in [`docs/`](docs/); this file links to them
instead of repeating them.

---

## 1. Vision

A user types any physics question, from "how long does a ball dropped from
20 m take to land?" to "how many rubber ducks would it take to stop a freight
train?", and gets back an answer they can trust: the equations it used (with
ids and sources), every assumption spelled out, units checked end to end, a
number computed by a real math engine, and an honest confidence label. When
the question is silly, the answer treats it seriously. When the question
cannot be answered, the answer says why and offers the closest version that
can be.

## 2. Core thesis

**Retrieval plus symbolic computation beats a language model guessing
arithmetic.**

Language models are good at language: reading a messy question, deciding
which physics applies, laying out a plan, and explaining the result. They are
bad at arithmetic, careless with units, and happy to invent formulas that
look right. So we split the work, and we build both halves ourselves:

| Job | Who does it |
|-----|-------------|
| Understand the question, pick a strategy, write the explanation | Our Fermi language models ([`docs/MODELS.md`](docs/MODELS.md)) |
| Know which equations exist and where they came from | Curated database + retrieval |
| Rearrange equations, substitute values, compute numbers | SymPy |
| Track, convert, and check units | Pint |
| Decide how much to trust the answer | Deterministic confidence formula (section 7) |

The language model never produces a number that appears in the final answer.
Constrained decoding makes it impossible for a plan to cite an unretrieved
equation or a number that wasn't in the input. Every equation in an answer
traces back to an entry in the database with a source and a license.

The models are the Fermi family, written and trained from scratch in this
repo, running locally with no external APIs (ADR-009):
`fermi-tellus-1` (~3M params), `fermi-solem-1` (~30M), and
`fermi-celeste-1` (~120M). Throughout this plan, "the LLM" means them.

## 3. Non-goals for v0.x

- **No external LLM APIs and no pretrained weights.** The Fermi models are
  ours, trained from scratch (ADR-009).
- **No giant models.** Celeste stays around 120M parameters until there is
  both the data and the compute to justify more.
- **No multimodal input.** No diagrams, photos of homework, or handwriting.
- **No web UI in v0.x.** CLI first; an optional API server is a v0.9
  stretch. A public website comes after v1.0, once the LLM layer and the
  symbolic algebra machine are solid (ADR-008).
- **No claim of correctness on research-level physics.** Quantum field
  theory, general relativity beyond textbook formulas, and open research
  questions are out of scope; the classifier should route them to an honest
  "out of scope" answer.
- **No homework-cheating features** like step-by-step answer formatting for
  specific textbooks. We explain physics; we do not mimic answer keys.

## 4. Pipeline overview

Six stages, each a plain function in `src/askphysics/pipeline.py`, each with
a typed input and output (pydantic models in `src/askphysics/models.py`).
Full payload examples are in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

```
classify -> retrieve -> plan -> compute -> sanity_check -> explain
```

### 4.1 classify

- **Input:** `Question` (raw text).
- **Output:** `Classification`: `category` (`standard` | `fermi` |
  `out_of_scope`), `reasoning`, `domains` (hints such as `kinematics`), and
  `closest_answerable` (for out-of-scope questions).
- **Failure modes:** a standard question labelled Fermi (we over-hedge and
  invent assumptions the user already gave); a Fermi question labelled
  standard (the planner can't find values and fails); a valid question
  labelled out of scope (a false refusal, the worst UX failure); malformed
  LLM output.
- **Graceful degradation:** malformed output gets one retry, then falls back
  to `standard` with a caveat. Defaulting to `standard` is deliberate: a
  failed attempt at an answer is more useful than a false refusal, and the
  later stages catch real problems.

### 4.2 retrieve

- **Input:** `Question`, `Classification`, `k`.
- **Output:** `RetrievalResult`: scored equations and worked examples, best
  first, with scores normalized to 0 to 1.
- **Failure modes:** the right equation is in the DB but not retrieved
  (vocabulary mismatch: "how hard does it hit" vs "momentum"); the right
  equation is not in the DB at all; irrelevant high-scoring hits.
- **Graceful degradation:** no hits raises `RetrievalEmptyError`, and the
  pipeline returns a `degraded` answer that names the domain it looked in
  and says the equation database lacks coverage. It never falls through to
  "let the LLM recall a formula" silently (see section 6 and
  [`docs/PROMPTS.md`](docs/PROMPTS.md)).

### 4.3 plan

- **Input:** `Question`, `RetrievalResult`, plus the constants and Fermi
  assumption tables.
- **Output:** `Plan`: `target` symbol, `unknowns`, `known_values` (each with
  value, unit, and origin: given, constant, or assumption), `equation_ids`,
  `assumptions` (plain-English), `strategy`.
- **Failure modes:** the LLM references an equation id that was not
  retrieved (hallucination); extracts the wrong number or unit from the
  question; forgets a needed known; picks a valid but inapplicable equation
  (for example, constant-acceleration kinematics for a rocket burning fuel).
- **Graceful degradation:** the plan is validated before compute: unknown
  equation ids, unparseable units, or a `target` not in `unknowns` raise
  `PlanValidationError`. In v0.1 that ends in a degraded answer. From v0.3
  the router retries solem (up to 5 attempts), then escalates once to
  celeste, then degrades (ADR-010).

### 4.4 compute

- **Input:** `Plan` and the referenced `Equation` entries.
- **Output:** `ComputeResult`: target, value, unit, the symbolic solution,
  and the substitutions made.
- **Failure modes:** the equation cannot be solved for the target in closed
  form; multiple roots (a negative time, a negative speed); a known is
  missing; units do not combine.
- **Graceful degradation:** SymPy and Pint errors become `SolverError` or
  `UnitMismatchError` and produce a degraded answer naming the failed step.
  Multiple roots are filtered to real roots, then non-negative ones, and the
  choice is recorded as a caveat. Multi-equation chains are not supported in
  v0.1 and say so.

### 4.5 sanity_check

- **Input:** `Plan`, `ComputeResult`, the equation's declared variables.
- **Output:** `SanityReport`: `dimensions_ok`, `magnitude_ok` (true, false,
  or null when there is no typical range), `limit_cases_checked`, `issues`.
- **Failure modes:** the check passes a wrong answer (dimensionally right,
  numerically absurd); the check fails a right answer (a typical range too
  narrow for an absurd question).
- **Graceful degradation:** sanity failures never block the answer. They
  lower confidence and add caveats. A dimension failure caps confidence at
  0.2 because a dimensionally wrong answer is wrong.

### 4.6 explain

- **Input:** `Question`, `Plan`, `ComputeResult`, `SanityReport`,
  `RetrievalResult`.
- **Output:** `Answer`: `final_value`, `unit`, `value_range` (Fermi),
  `equations_used` (ids and names), `assumptions`, `confidence` (label and
  score), `caveats`, `explanation`, `status` (`answered` | `degraded` |
  `refused`).
- **Failure modes:** the LLM restates numbers incorrectly or does new
  arithmetic in prose; omits assumptions; cites equations not used.
- **Graceful degradation:** every structured field is filled by code, not
  the LLM. The LLM only writes the `explanation` string. If that call fails,
  a deterministic template explanation is used instead.

## 5. Handling absurd questions (Fermi policy)

"How many rubber ducks would it take to stop a freight train?" is a real
physics question wearing a silly hat. The policy:

1. **Decompose.** Rewrite the question as a chain of estimable quantities:
   the train's momentum (mass x speed), the momentum one duck can absorb in
   a collision, and the ratio between them.
2. **List every assumption explicitly.** "A loaded freight train masses
   about 6,000 t", "the ducks are stationary and stick to the train", "stop
   means slow to half speed, because perfectly inelastic collisions never
   reach zero". The user should be able to disagree with any line.
3. **Take defaults from the assumptions table, not from vibes.** Every
   Fermi quantity has a `default_value`, `low`, `high`, a `rationale`, and a
   `source` in `data/fermi_assumptions.json`. The planner may only use a
   number that is not in the table if it labels that number "estimated, not
   from table", which costs confidence.
4. **Compute with the same SymPy and Pint path** as standard questions.
5. **Report a range, not fake precision.** The answer is an order of
   magnitude ("about 10^8 ducks, plausibly 3x10^7 to 2x10^9"), produced by
   propagating `low` and `high` through the calculation (Monte Carlo or
   interval arithmetic, decided in v0.7). Never "243,912,016 ducks".
6. **Flag the silly parts with a straight face.** Caveats state the physical
   absurdities plainly ("this many ducks would form a pile roughly X m
   high; the train would derail before stopping") without jokes overriding
   the content. Deadpan is the house style.

Confidence for Fermi answers is capped at 0.6: the method is sound, but the
inputs are guesses by construction.

## 6. Handling impossible or out-of-scope questions

Three kinds of question get a refusal-with-redirect instead of a number:

| Kind | Example | Why it cannot be answered |
|------|---------|---------------------------|
| No physical meaning | "How much does the color blue weigh?" | Category error: color is a property of light, not an object with mass |
| Requires unknowable data | "What was the temperature before the Big Bang?" | Outside the domain where current physics makes predictions |
| Out of project scope | "Derive the QCD beta function" | Research-level; we would be guessing |

Every refusal must:

1. **Say why**, in one or two sentences, naming which kind it is.
2. **Offer the closest answerable version**, phrased as a question the user
   could ask next ("Blue light does carry momentum. Want the radiation
   pressure of a 1 W blue laser?").
3. **Not lecture.** One redirect, no moralizing.

Ambiguous questions (missing a needed value) are *not* refusals: they get
treated as Fermi questions with the missing value assumed and flagged, or,
when no defensible default exists, a degraded answer that names the missing
quantity.

## 7. Confidence model

Crude on purpose, documented so it can be replaced with something better
once evals exist. Implemented in `askphysics.pipeline.score_confidence`.

**Inputs:**

| Symbol | Meaning | Range |
|--------|---------|-------|
| `r` | Top retrieval score for an equation the plan actually used | 0 to 1 |
| `d` | Dimensional consistency: 1 if `dimensions_ok`, else 0 | 0 or 1 |
| `s` | Magnitude check: 1 if passed, 0.5 if no typical range, 0 if failed | 0, 0.5, 1 |
| `n` | Number of assumptions in the plan | 0 or more |

**Formula:**

```
a     = 1 / (1 + 0.25 * n)               # 0 assumptions -> 1.0, 4 -> 0.5
score = 0.35*r + 0.30*d + 0.20*s + 0.15*a

if d == 0:            score = min(score, 0.2)   # dimensionally wrong is wrong
if category == fermi: score = min(score, 0.6)   # inputs are guesses
```

**Labels:** `high` at 0.75 or above, `medium` at 0.45 or above, `low`
below that. Refusals and degraded answers report `low` with score 0.0.

**Known weaknesses:** the weights are made up; the retrieval score is not
calibrated; a large `n` penalizes careful answers that list obvious
assumptions. v0.5 eval data will be used to fit the weights (see
[`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)).

## 8. Success metrics per version

| Version | Metric | Target |
|---------|--------|--------|
| v0.1 | `pytest`, `ruff`, `mypy` clean; CLI runs end to end with the fake LLM | Pass |
| v0.2 | `fermi-luna-1` trains in CI with falling loss; constrained decoding property-tested | Pass |
| v0.3 | solem valid-plan rate on held-out templates; celeste rescue rate | 90% or more; a third or more of solem failures |
| v0.4 | Standard questions within 2% relative tolerance, correct unit | 80% or more |
| v0.5 | Eval set size; automated scoring coverage | 100+ questions; 100% scored |
| v0.6 | Recall@5 on the retrieval set; equations / examples with validated license | 90% or more; 500+ / 1000+ |
| v0.7 | Fermi answers within one order of magnitude of reference | 70% or more |
| v0.8 | Wrong answers flagged by self-verification | 50% or more of wrong answers caught |
| v1.0 | All eval categories at threshold; no category regressed | See [`docs/ROADMAP.md`](docs/ROADMAP.md) |

## 9. Risks and mitigations

The full register, with likelihood, impact, and owner, is in
[`docs/RISKS.md`](docs/RISKS.md). The top three:

1. **Hallucinated equations.** Plans can only reference retrieved equation
   ids; the validator rejects anything else.
2. **False confidence.** Confidence is computed by code from checks, never
   self-reported by the LLM.
3. **License contamination.** Every data entry carries `source` and
   `license`; incompatible licenses fail validation (see
   [`docs/DATA_SOURCING.md`](docs/DATA_SOURCING.md)).

## 10. Milestones

v0.1 Skeleton -> v0.2 Fermi foundations -> v0.3 Fermi models trained and
wired in -> v0.4 Solver expansion -> v0.5 Eval harness -> v0.6 Retrieval and
data expansion -> v0.7 Fermi engine -> v0.8 Self-verification -> v0.9
Hardening -> v1.0 Release -> website.

Goals, deliverables, exit criteria, and effort for each:
[`docs/ROADMAP.md`](docs/ROADMAP.md). The near-term backlog is in
[`TODO.md`](TODO.md).
