# Roadmap

Effort is rough: **S** is a few sittings, **M** is a week or two of evenings,
**L** is a month or more. Each milestone ships as a tagged release with a
`CHANGELOG.md` entry. No milestone starts until the previous one's exit
criteria are met, except where noted.

---

## v0.1.0 Skeleton (S)

**Goal:** an architecture you can run, test, and extend, with nothing fake
pretending to be real.

**Deliverables**
- Planning docs: `PLAN.md`, `CLAUDE.md`, `TODO.md`, everything in `docs/`.
- `askphysics` package: six-stage pipeline, pydantic models, error taxonomy.
- `FakeLLMClient` with canned responses; `AnthropicClient` stub.
- Working in-memory keyword retriever; vector retriever stub.
- Pint and SymPy wrappers with the single-equation solve path.
- Seed data: about 12 equations, 4 worked examples, 8 constants, 8 Fermi
  assumptions.
- Typer CLI: `ask`, `version`, `validate-data`.
- Tests for every module; `evals/questions.yaml` with 8 questions; stub
  harness.
- CI on GitHub Actions.

**Exit criteria**
- `pytest`, `ruff check`, `ruff format --check`, and `mypy` are clean.
- `askphysics ask "..."` runs end to end with the fake LLM.
- Every stub raises `NotImplementedError` with a TODO.

---

## v0.2.0 Real retrieval (M)

**Goal:** find the right equation even when the question shares no words with
it.

**Vector store: sqlite-vec.** Justification:
- The equation DB stays small for a long time (500 entries at v0.6). Index
  speed is irrelevant at that size; simplicity and portability are what
  matter.
- sqlite-vec keeps vectors, metadata, and keyword search (SQLite FTS5) in one
  file with one dependency, which makes hybrid search a single SQL query
  instead of two systems glued together.
- Chroma adds a server-or-embedded runtime and a heavier dependency tree for
  features we do not need yet. FAISS is excellent at millions of vectors but
  has no metadata or keyword story, so we would bolt on SQLite anyway.
- Hidden behind the `VectorStore` protocol, so switching later is cheap.

**Deliverables**
- Embedding interface plus one implementation (a small local
  sentence-transformer by default, so tests and CI stay offline-capable).
- `SqliteVecStore` implementing `VectorStore`.
- Hybrid retriever: BM25 (FTS5) + vector similarity, merged with reciprocal
  rank fusion.
- Retrieval eval set: 50 questions, each labelled with the correct equation
  ids, written without looking at equation names or tags.
- Reranking experiment: cross-encoder rerank of the top 20 vs none, results
  written up in `docs/DECISIONS.md` whichever way it goes.

**Exit criteria**
- Recall@5 of 90% or more on the retrieval eval set.
- Keyword retriever stays as the fallback, and its tests still pass.
- Retrieval p95 latency under 200 ms on a laptop for the seed DB.

---

## v0.3.0 Real solver (L)

**Goal:** plan-to-compute works for intro mechanics, kinematics, energy,
basic E&M, and thermodynamics, unit-safe end to end.

**Deliverables**
- `AnthropicClient` implemented with structured outputs for classify and plan.
- Real prompts from `docs/PROMPTS.md`, versioned in code.
- Multi-equation chaining: solve equation A for an intermediate, feed it to
  equation B, with the dependency graph built from the plan.
- Re-plan loop: one retry with the validation error fed back to the planner.
- Equation-level dimensional consistency checked at load time for every
  entry.
- Seed DB grown to cover the five domains (about 60 equations).

**Exit criteria**
- 80% or more of standard eval questions within 2% relative tolerance with
  the correct unit.
- Zero answers where a number in `explanation` differs from `final_value`
  (checked automatically).
- No plan ever references an equation id outside the retrieved set.

---

## v0.4.0 Eval harness (M)

**Goal:** know, with numbers, whether a change made things better or worse.

**Deliverables**
- `evals/harness.py` implemented: run all questions, score, write a JSON
  report and a Markdown summary.
- 100+ questions across all six categories in `docs/EVALS.md`.
- Automated scoring: numeric tolerance, unit match, equation-id match,
  refusal correctness, and an assumption-quality rubric (LLM-graded with a
  human-audited sample).
- Regression tracking: reports committed to `evals/reports/`, with a
  comparison against the last release.
- CI job that runs the eval suite against the fake LLM on every PR (smoke),
  and against the real LLM on demand.

**Exit criteria**
- Harness produces a stable score (variance under 2 points across 3 runs).
- Regression policy from `docs/EVALS.md` enforced in the release checklist.

---

## v0.5.0 Fermi engine (M)

**Goal:** absurd questions get defensible ranges instead of fake precision.

**Deliverables**
- Assumption library grown to 100+ entries with `low`, `high`, rationale, and
  source.
- Uncertainty propagation: Monte Carlo (seeded, log-uniform between `low`
  and `high`) as the default, with interval arithmetic as a cheap cross-check.
  The decision and reasoning are recorded in `docs/DECISIONS.md`.
- `value_range` populated on Fermi answers; formatting as order of
  magnitude.
- Absurdity caveats generated from computed side quantities (pile height,
  energy released, and so on).

**Exit criteria**
- 70% or more of Fermi eval answers within one order of magnitude of the
  reference.
- Every Fermi answer lists its assumptions, and every assumed number traces
  to the table or is flagged.

---

## v0.6.0 Data expansion (L)

**Goal:** coverage. Most intro-physics questions should find their equation.

**Deliverables**
- 500+ equations and 1000+ worked examples, following
  `docs/DATA_SOURCING.md`.
- Licensing tracked per entry, with a generated `DATA_LICENSES.md`
  attribution file.
- Synthetic example pipeline: LLM-generated problems, SymPy-verified answers,
  human spot-check of 10%.
- Data QA report generated by `askphysics validate-data --report`.

**Exit criteria**
- 100% of entries pass schema validation, SymPy parse, and the unit check.
- Zero entries with an unknown or incompatible license.
- Retrieval Recall@5 does not drop below the v0.2 bar as the DB grows.

---

## v0.7.0 Fine-tuning (L)

**Goal:** find out whether a small open model can do the plan stage as well
as a prompted frontier model, cheaper.

**Deliverables**
- Instruction dataset built from pipeline traces (question, retrieval
  context, validated plan), filtered to plans whose answers passed evals.
- Fine-tune of a small open model (around 1B to 8B parameters) for the plan
  stage only.
- Head-to-head comparison: plan validity rate, downstream answer accuracy,
  latency, cost per question.

**Exit criteria**
- A written result in `docs/DECISIONS.md`, positive or negative. A negative
  result is a valid outcome; the prompted baseline stays the default unless
  the fine-tune is at least as accurate.

---

## v0.8.0 Self-verification (M)

**Goal:** catch wrong answers before the user does.

**Deliverables**
- Second-pass checker: re-derive the answer by an independent route where
  one exists (energy vs kinematics for the same fall).
- Limit-case tests per equation: symbolic limits as a variable goes to 0 or
  infinity, checked against expected behavior declared in the data.
- Conflict detection between retrieved sources that disagree.
- Confidence formula refit on v0.4 eval data.

**Exit criteria**
- 50% or more of known-wrong answers in the eval set are flagged.
- False-flag rate under 10% on correct answers.

---

## v0.9.0 Hardening (M)

**Goal:** something other people can run without reading the source.

**Deliverables**
- Caching of LLM calls and pipeline traces keyed by question hash and
  prompt version.
- Rate limiting and per-run cost caps for real LLM providers.
- Structured logging and tracing for each stage, with timings.
- Error taxonomy finalized and mapped to user-facing messages.
- CLI polish: `--json`, `--verbose`, trace dump, shell completion.
- Optional FastAPI server behind an extra (`pip install askphysics[server]`).

**Exit criteria**
- No unhandled exception reaches the user in the eval suite.
- A cold `pip install` and first answer works in under 5 minutes, following
  only the README.

---

## v1.0.0 Release (S)

**Release criteria**
- Eval thresholds: standard 85% or more, multi-step 70% or more, Fermi 70%
  or more within an order of magnitude, impossible/ambiguous 90% or more
  correct behavior, adversarial 95% or more with no injection success.
- No category regressed by more than the threshold in `docs/EVALS.md`
  relative to v0.9.
- Docs complete: every public function documented; README quickstart
  verified on a clean machine.
- Reproducible setup: pinned lockfile, `make install && make test` green on
  Linux and macOS, Python 3.11 to 3.13.
- Published to PyPI.

---

## After v1.0: Website (L)

**Goal:** put Ask Physics online for people who will never open a terminal
(ADR-008).

**Deliverables (to be detailed when v1.0 ships)**
- Web front end that consumes the same `Answer` JSON the CLI prints:
  value and unit, cited equations, assumptions, confidence, caveats.
- Hosted API built from the v0.9 server, with caching, rate limits, and cost
  caps sized for public traffic.
- Rendered LaTeX for equations and a "show the math" view of the symbolic
  steps.

**Entry criteria**
- v1.0 release criteria met, with eval thresholds holding on the hosted
  configuration.
