# Architecture Decision Records

Each record has context, decision, and consequences. Records are append-only.
To change a decision, add a new record that supersedes the old one, and mark
the old one "Superseded by ADR-NNN".

---

## ADR-001: RAG plus symbolic computation over pure fine-tuning

**Status:** Accepted (v0.1.0). Amended by ADR-009: the LLM is now our own from-scratch Fermi family; retrieval plus symbolic computation is unchanged.

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

**Status:** Accepted (v0.1.0). Amended by ADR-009: the `LLMClient` protocol stays, external providers are dropped, and the real implementation is `FermiClient`.

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

**Status:** Superseded by ADR-009. We control decoding, so classify and plan are greedy and fully deterministic.

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
- `sympy.*`: `ignore_missing_imports`, because SymPy ships without complete
  type information. Values crossing that boundary are annotated at our
  wrapper functions in `solver/symbolic.py`. Pint ships type hints and needs
  no exception; annotations use the `Quantity = pint.Quantity[Any]` alias
  from `solver/units.py`.

**Consequences.** Type safety stops at the wrapper boundary for SymPy
objects; the wrappers' own signatures are fully typed. v0.1.0 passes with no
other suppressions.

---

## ADR-008: CLI first, website after v1.0

**Status:** Accepted (v0.1.0), decided by the maintainer

**Context.** Ask Physics needs a user-facing surface. A website reaches more
people, but building one now would mean designing UI around an LLM layer and
a symbolic algebra machine that are still stubs and will change shape.

**Decision.** The CLI (`askphysics ask`) is the only interface through v1.0.
An optional API server stays in v0.9 as groundwork. A public website comes
after v1.0, once the models and the symbolic algebra machine have passed the
release eval thresholds.

**Consequences.**
- `Answer` and `--json` are the contract a future website will consume, so
  they must stay stable and complete (every field a UI needs, no Rich-only
  information).
- Hosting, auth, rate limiting, and cost control for public traffic are
  deferred, but the v0.9 caching and cost caps are designed with them in
  mind.

---

## ADR-009: The Fermi model family, built from scratch, no external APIs

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** v0.1 planned an API-backed LLM (Anthropic) behind the
`LLMClient` protocol. The maintainer wants Ask Physics powered by its own
language models, written in Python, trained from scratch, running locally,
with no external API calls. The original repo description already promised
"a simple Python LLM that runs on your device." Training hardware is an M5
Pro with 48 GB of unified memory.

**Decision.**
- Three decoder-only transformers written in PyTorch in this repo:
  `fermi-tellus-1` (~3M params), `fermi-solem-1` (~30M), and
  `fermi-celeste-1` (~120M), plus `fermi-luna-1` for tests. Design and
  training plan in `docs/MODELS.md`.
- Our own byte-level BPE tokenizer with single-digit tokens and task tokens.
- Training data comes from a data factory in this repo, built from the
  equation database and verified by the symbolic algebra machine. No
  pretrained weights, no external models generating data.
- **Constrained decoding** enforces the golden rules structurally: only
  retrieved equation ids, only numbers present in the input or tables, only
  valid units, only schema-valid JSON.
- Classify and plan decode greedily, so they're deterministic.
  `Settings.temperature` applies only to explain.
- `AnthropicClient`, the `[anthropic]` extra, and API keys are removed. The
  `LLMClient` protocol and `FakeLLMClient` stay.
- `torch` and `safetensors` become dependencies in v0.2. Weights are
  safetensors in `~/.cache/askphysics/models/`, never in git, never pickled.

**Consequences.**
- No API cost, no network, no key management; Ask Physics works offline.
- Small from-scratch models understand phrasing close to their training
  data. Unusual wording fails more often than with a big pretrained model.
  Constraints keep failures safe (degraded, not invented), and v0.5 evals
  measure the rate.
- Language quality now depends on our data factory. Template diversity and
  leakage control become core engineering work, not side tasks.
- ~1B-parameter training is out of reach on a laptop; celeste stays ~120M
  until there's both the data and the compute to justify more.
- Supersedes ADR-006; amends ADR-001 and ADR-003. The v0.1 non-goal "no
  training from scratch" is dropped.

---

## ADR-010: Model routing: split and escalate locally, usage tiers online

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** Three model sizes trade speed for quality. The maintainer
specified a scheme: start on solem, get one celeste, drop to tellus after
five solem tries, or otherwise use the models in different ways. On a
local machine there's no cost to meter, so quotas protect nothing. On a
hosted website, they cap the cost.

**Decision.** Both, each where it makes sense.
- **CLI (v0.3): split and escalate.** tellus classifies every question.
  solem plans and explains. If solem's plan fails validation, or the
  symbolic algebra machine rejects it, solem retries up to 5 attempts in
  total (varying context order and seed). Then celeste gets one attempt.
  Then the answer degrades. Missing weights are skipped, and tellus alone
  can run every task.
- **Website (after v1.0): usage tiers.** A visitor starts on solem, gets one
  celeste answer per day, and drops to tellus after five solem answers in
  a window. Limits are server config, built with the v0.9 API server. The
  CLI never enforces them.
- Every `Answer` records which model handled each stage.

**Consequences.**
- Worst case for a hard question is 5 solem attempts plus 1 celeste
  attempt. Latency is bounded, and stays acceptable at these model sizes.
- Escalation statistics (how often celeste rescues solem) become an eval
  metric that justifies celeste's existence, or not.
- The exact numbers (5 attempts, 1 escalation, the website limits) live in
  `Settings`, not in code.

---

## ADR-011: Install channels: curl, irm, and a Homebrew tap

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** People should be able to try Ask Physics with one command,
without knowing what a virtualenv is. The package depends on torch, whose
default Linux wheel bundles about 2 GB of CUDA libraries the CLI never uses.

**Decision.**
- **Recommended:** `install.sh` (`curl ... | sh`) for macOS and Linux and
  `install.ps1` (`irm ... | iex`) for Windows. Both install uv if missing and
  run `uv tool install`, which gives `askphysics` its own isolated
  environment with a uv-managed Python 3.12. On Linux, torch comes from
  PyTorch's CPU-only index, pinned for torch alone by a uv source in
  `pyproject.toml` (passing the index to uv globally makes it resolve every
  package there first). The default installs the latest GitHub release;
  `ASKPHYSICS_REF` pins a tag or branch.
- **Also:** a Homebrew tap, `shankar-sachin/homebrew-tap` (`brew install shankar-sachin/tap/askphysics`), for people
  who live in brew. Its formula builds a virtualenv in `libexec` from the
  release tarball.
- **Not:** WinGet (not submitting), and PyPI only at v1.0.
- A CI workflow runs both installers on real Linux, macOS, and Windows
  machines whenever they change.

**Consequences.**
- Each release needs the tap's formula bumped (URL and sha256); the steps
  are in `docs/RELEASING.md`.
- Installs need network access to GitHub, PyPI, and (on Linux) PyTorch's
  index. Offline installs are out of scope until wheels ship on PyPI.
- Installers never touch the system Python, so uninstalling is one command.
