# Roadmap

Effort is rough: **S** is a few sittings, **M** is a week or two of evenings,
**L** is a month or more. Each milestone ships as a tagged release with a
`CHANGELOG.md` entry, and only with the maintainer's approval. No milestone
starts until the previous one's exit criteria are met, except where noted.

The language side is the Fermi model family, built from scratch in this repo
([`MODELS.md`](MODELS.md), ADR-009). The math side is Noether (SymPy + Pint). The CLI is the main interface; an in-browser website
at askphysics.vercel.app runs the same package (ADR-013). The hosted API comes
after v1.0 (ADR-008).

---

## v0.1.0 Skeleton (S), done

**Goal:** an architecture you can run, test, and extend, with nothing fake
pretending to be real.

**Delivered:** planning docs; six-stage pipeline with graceful degradation;
pydantic models and error taxonomy; `FakeLLMClient`; keyword retriever;
SymPy and Pint wrappers with a single-equation `solve_for`; validated seed
data (12 equations, 4 examples, 8 constants, 8 Fermi assumptions); Typer CLI;
156 tests; 8-question eval set; CI.

---

## v0.2.0 Fermi foundations (L), done

**Goal:** every piece needed to train a Fermi model exists, works, and is
tested on `fermi-luna-1`, without needing a GPU.

**Deliverables**
- `src/askphysics/lm/`: config presets (tellus, solem, celeste, luna),
  byte-level BPE tokenizer with digit splitting and task tokens, the
  transformer, safetensors checkpoints, device selection.
- Constrained decoding: schema masks, allowed equation ids, allowed numbers,
  valid units.
- Data factory: standard, Fermi, out-of-scope, and explanation examples from
  the database, every standard example verified by Noether.
- Training loop with seeded runs, bf16 autocast, checkpoints, and resumption.
- CLI: `askphysics model build-data`, `train-tokenizer`, `train`, `info`.
- `torch` and `safetensors` as dependencies; CI on CPU wheels.

**Exit criteria**
- `fermi-luna-1` trains for 50 steps in CI and its loss drops.
- A constrained plan decode can only emit retrieved equation ids and numbers
  from the question (property-tested).
- Tokenizer round-trips every string in the data factory output.

---

## v0.3.0 Fermi models trained and wired in (L)

**Goal:** the CLI answers with our own models.

**Deliverables**
- Train `fermi-tellus-1`, `fermi-solem-1`, and `fermi-celeste-1` (~120M) on
  the maintainer's M5 Pro, with the commands and configs committed.
- `FermiClient` implementing `LLMClient`, with the ADR-010 router: tellus
  classifies, solem plans and explains, up to 5 solem attempts, one
  celeste escalation, tellus-only fallback when bigger weights are missing.
- `Answer` records which model handled each stage.
- `fermi` becomes the default provider once weights are installed; the fake
  client stays for tests.
- Model cards with measured throughput, training curves, and eval results.
- Ship the trained weights (ADR-012): GitHub release assets pinned by a
  checksum manifest, `askphysics model pull`, the installers and Homebrew
  formula fetching tellus and solem, and celeste on first escalation.

**Exit criteria**
- On a held-out set of factory-style questions (unseen templates), solem
  produces a valid plan for 90% or more, and celeste escalation recovers at
  least a third of solem's failures.
- Zero invented equation ids or numbers (guaranteed by constraints, verified
  by tests).
- `askphysics ask` with solem answers in under 2 seconds on an M5 Pro.
- A fresh install on a clean machine answers with solem, with no training
  and no manual download step.

---

## v0.4.0 Solver expansion (M)

**Goal:** Noether handles real intro problems.

**Deliverables**
- Multi-equation chaining: dependency order over the plan's unknowns.
- Offset units (Celsius, Fahrenheit) converted before substitution.
- About 60 equations across mechanics, energy, E&M, and thermodynamics;
  15+ worked examples.
- Factory regenerated from the bigger database; models retrained.
- **The Ask Physics Wiki**, one source with two homes:
  - Pages live in this repo under `docs/wiki/`: guides, FAQ, glossary, the
    model cards, and a page per equation generated from the database
    (formula, variables with units, validity conditions, source, license).
  - A CI workflow publishes them to the repo's GitHub Wiki.
  - The website renders the same pages at askphysics.vercel.app/wiki/, with
    rendered math, linked from answer cards (each equation id links to its
    page).

**Exit criteria**
- 80% or more of standard eval questions within 2% with the correct unit.
- Every worked example re-solves within 0.1%.

---

## v0.5.0 Eval harness (M)

**Goal:** know, with numbers, whether a change made things better or worse.

**Deliverables**
- `evals/harness.py` implemented; 100+ questions across all six categories.
- Automated scoring (numeric tolerance, unit, equation ids, refusal
  correctness) plus a rubric for assumption quality, graded by humans on a
  sample (no external LLM grader).
- Per-model results: tellus vs solem vs celeste vs routed.
- Reports committed to `evals/reports/`; regression policy enforced.

**Exit criteria**
- Stable scores across 3 runs; regression policy in the release checklist.

---

## v0.6.0 Retrieval and data expansion (L)

**Goal:** find the right equation for messy questions, at scale.

**Deliverables**
- Embeddings from a Fermi model's hidden states (no external embedding API),
  stored in sqlite-vec, fused with BM25 (SQLite FTS5) by reciprocal rank
  fusion. sqlite-vec because the database stays small and one file holds
  vectors, metadata, and keyword search.
- 500+ equations and 1000+ worked examples per `docs/DATA_SOURCING.md`, with
  licenses tracked; CC-BY physics text added to the training corpus.
- Retrieval eval set (50+ questions) with Recall@5.

**Exit criteria**
- Recall@5 of 90% or more; 100% of entries pass validation.

---

## v0.7.0 Fermi engine (M)

**Goal:** absurd questions get defensible ranges instead of fake precision.

**Deliverables**
- Assumption library of 100+ entries.
- Seeded Monte Carlo propagation (log-uniform between `low` and `high`),
  interval arithmetic as a cross-check.
- `value_range` on Fermi answers; absurdity caveats from computed side
  quantities.

**Exit criteria**
- 70% or more of Fermi eval answers within one order of magnitude.

---

## v0.8.0 Self-verification (M)

**Goal:** catch wrong answers before the user does.

**Deliverables**
- Second-pass checks via an independent route where one exists.
- Limit-case tests per equation.
- Confidence formula refit on eval data.

**Exit criteria**
- 50% or more of known-wrong answers flagged; false-flag rate under 10%.

---

## v0.9.0 Hardening (M)

**Goal:** something other people can run without reading the source.

**Deliverables**
- Structured logging and per-stage timings; error taxonomy mapped to
  user-facing messages.
- CLI polish: `--json`, `--verbose`, `--model`, trace dump.
- Optional API server (groundwork for the website), including the ADR-010
  usage tiers behind config, and the API key design (Q18): key format,
  hashed storage, per-key quotas, rotation and revocation.

**Exit criteria**
- No unhandled exception reaches the user in the eval suite.
- Fresh install to first answer in under 5 minutes, following only the README.

---

## v1.0.0 Release (S)

**Release criteria**
- Eval thresholds: standard 85%+, multi-step 70%+, Fermi 70%+ within an order
  of magnitude, impossible/ambiguous 90%+ correct behavior, adversarial 95%+.
- No category regressed beyond the `docs/EVALS.md` threshold relative to v0.9.
- Docs complete; reproducible setup on macOS and Linux, Python 3.11 to 3.13.
- Published to PyPI, with weights as release assets.

---

## After v1.0: Hosted API (L)

**Goal:** let developers call Ask Physics over HTTP. (The in-browser website
already ships, ADR-013; the hosted version can reuse its front end.)

**Deliverables (detailed when v1.0 ships)**
- Point the existing web front end at the hosted API for people whose
  devices can't run the in-browser engine.
- Hosted API built from the v0.9 server, with the ADR-010 usage tiers: start
  on solem, one celeste answer per day, tellus after five solem answers.
- Rendered LaTeX and a "show the math" view of the symbolic steps.
- **Ask Physics API keys:** self-serve keys for the hosted API, stored hashed
  with a recognizable prefix (`ap_live_...`) so leaked keys are easy to find,
  per-key quotas tied to the usage tiers, rotation, and revocation. The local
  CLI never needs a key.

**Entry criteria**
- v1.0 release criteria met, holding on the hosted configuration.
