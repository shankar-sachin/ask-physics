# Architecture Decision Records

Each record has context, decision, and consequences. Records are append-only.
To change a decision, add a new record that supersedes the old one, and mark
the old one "Superseded by ADR-NNN".

---

## ADR-001: RAG plus symbolic computation over pure fine-tuning

**Status:** Accepted (v0.1.0)

**Context.** Two broad ways to build a physics answerer: fine-tune a model
until it "knows" physics and does the math, or have a model plan around a
curated equation database and hand computation to a math engine. LLMs are
unreliable at multi-digit arithmetic and unit bookkeeping, and fine-tuning
does not fix that reliably. It also hides where a formula came from, and it
costs data and compute we do not have.

**Decision.** Retrieval-augmented generation for choosing equations,
symbolic computation for all math. The LLM classifies, plans, and explains.
Fine-tuning is allowed later only for a narrow stage (planning, v0.7) and
only if it beats the prompted baseline.

**Consequences.**
- Every answer is auditable: equation ids, sources, substitutions.
- Coverage is limited by the database. A missing equation is a visible gap,
  not a confident guess, which is better but means data work is never done.
- More moving parts than one model call: six stages, each needing tests.
- Plans must be structured enough for a machine to execute, which constrains
  prompts (see ADR-003 and `docs/PROMPTS.md`).

---

## ADR-002: SymPy and Pint as the compute layer

**Status:** Accepted (v0.1.0)

**Context.** We need symbolic rearrangement (solve `v^2 = v0^2 + 2ad` for
`v`), numeric evaluation, and unit tracking with dimensional analysis. The
options were SymPy plus Pint; SymPy's own `sympy.physics.units`; writing our
own; or numeric-only (NumPy plus `astropy.units`).

**Decision.** SymPy for algebra and Pint for units, with symbols kept
unit-free inside SymPy. Equations are solved symbolically first, then the
solution is evaluated with Pint quantities substituted in, so Pint catches
dimensional errors during evaluation.

**Consequences.**
- Both are mature, pure-Python, and permissively licensed (BSD).
- `sympy.physics.units` was rejected: weaker unit parsing and a smaller
  ecosystem than Pint, and mixing units into symbolic expressions slows
  `solve` and complicates simplification.
- Evaluating SymPy expressions on Pint quantities needs a small lambdify
  bridge (`sqrt` mapped to `** 0.5`). Transcendental functions require
  dimensionless arguments, which is physically correct and will need care.
- Offset units (degrees Celsius) do not multiply cleanly in Pint; banned in
  data for now (`docs/OPEN_QUESTIONS.md`).
- SymPy's parser uses `eval`; mitigated as described in `SECURITY.md` and
  the architecture doc's security boundary section.
- SymPy has incomplete type hints; mypy is configured to ignore missing
  imports for it.

---

## ADR-003: Provider-agnostic LLM interface

**Status:** Accepted (v0.1.0)

**Context.** The project should not be married to one LLM vendor: costs and
quality shift fast, v0.7 may swap in a local fine-tuned model, and tests
must run offline with no API key. The repo's original description also
mentions an on-device model (see `docs/OPEN_QUESTIONS.md`).

**Decision.** A two-method `typing.Protocol`, `LLMClient`:
`complete_json(system, user, schema) -> validated pydantic model` and
`complete_text(system, user) -> str`. Provider SDKs are optional extras
(`pip install askphysics[anthropic]`). `FakeLLMClient` is the default in
config, tests, and the demo. Anthropic is the first real provider.

**Consequences.**
- A provider is one file. The fake client keeps the protocol honest because
  the pipeline cannot rely on anything it doesn't do.
- Provider-specific strengths (prompt caching, strict structured outputs)
  must be used inside the client, not exposed to the pipeline.
- Structured output quality varies by provider. Clients without native
  schema enforcement must parse, validate, and retry internally.

---

## ADR-004: src layout and a Typer CLI

**Status:** Accepted (v0.1.0)

**Context.** We need a package layout that prevents accidentally importing
from the working directory instead of the installed package, and a CLI that
is quick to extend and typed.

**Decision.** PEP 621 `pyproject.toml`, a `src/askphysics/` layout, the
Hatchling build backend, and Typer (built on Click) for the CLI, with Rich
for output. Entry point: `askphysics = "askphysics.cli:app"`.

**Consequences.**
- Tests run against the installed package (`pip install -e .`), which
  catches packaging mistakes such as missing data files early.
- Typer derives CLI options from type hints, which matches the
  typed-everything convention. Click is available underneath if Typer
  becomes limiting.
- Rich output is for humans; `--json` provides the machine-readable form, so
  scripts never parse Rich tables.

---

## ADR-005: JSON seed data before a real database

**Status:** Accepted (v0.1.0)

**Context.** v0.1 has about 30 data entries in total. A database (SQLite,
Postgres, or a vector DB) would add migrations, setup, and tooling before we
know the schema is right.

**Decision.** Plain JSON files in `src/askphysics/data/`, loaded and
validated through pydantic at startup. The JSON files are the source of
truth through v0.5. From v0.2, the sqlite-vec index is a derived artifact
rebuilt from the JSON, never edited directly.

**Consequences.**
- Data changes are reviewable diffs in pull requests, which is the main QA
  mechanism.
- Loading everything at startup is fine at this scale. Somewhere around
  thousands of entries (v0.6), we revisit: probably JSON Lines plus a built
  SQLite file shipped as package data.
- Schema changes are a find-and-replace across JSON files; acceptable while
  the schema is still moving.

---

## ADR-006: Determinism without sampling temperature on Anthropic models

**Status:** Accepted (v0.1.0)

**Context.** The original spec called for "temperature 0 for planning".
Current Claude models (including the default, `claude-opus-5-5`) reject
`temperature` and `top_p` with a 400 error. Other providers still accept
temperature.

**Decision.** `Settings.temperature` stays (default 0.0) and is passed only
to providers that support it. For Anthropic, determinism comes from strict
structured outputs (`client.messages.parse` with the pydantic schema), a
pinned model id, the lowest effort level that holds eval quality, and, from
v0.9, cached plan traces. Monte Carlo uses explicit seeds. Logged as an open
question to revisit if provider behavior changes.

**Consequences.**
- Plans for the same question may vary slightly between runs on Anthropic.
  The eval harness must run each question more than once to measure that
  variance (`docs/ROADMAP.md`, v0.4 exit criteria).
- Everything after `plan` is deterministic given a plan, so replaying a
  cached plan reproduces an answer exactly.

---

## ADR-007: mypy configuration

**Status:** Accepted (v0.1.0)

**Context.** The definition of done for v0.1.0 requires `mypy` to pass, with
any exceptions listed here.

**Decision.** `mypy --strict` on `src/` and `evals/`. Known exceptions:
- `sympy.*` and `pint.*`: `ignore_missing_imports` and untyped-call allowances
  where their stubs are incomplete. Values crossing those boundaries are
  annotated at our wrapper functions (`solver/units.py`,
  `solver/symbolic.py`).
- `anthropic`: the optional extra is not installed in CI, so imports of it
  happen inside the client method and are covered by
  `ignore_missing_imports`.

**Consequences.** Type safety stops at the wrapper boundary for SymPy and
Pint objects; the wrappers' own signatures are fully typed.
