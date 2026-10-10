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

**Status:** Accepted (v0.1.0), decided by the maintainer. Superseded in part
by ADR-013: an in-browser website ships early; the hosted API still waits.

**Context.** Ask Physics needs a user-facing surface. A website reaches more
people, but building one now would mean designing UI around an LLM layer and
Noether that are still stubs and will change shape.

**Decision.** The CLI (`askphysics ask`) is the only interface through v1.0.
An optional API server stays in v0.9 as groundwork. A public website comes
after v1.0, once the models and Noether have passed the
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
  equation database and verified by Noether. No
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
  solem plans and explains. If solem's plan fails validation, or Noether rejects it, solem retries up to 5 attempts in
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

**Implementation (v0.3).** `llm/routing.py` picks the models from what is
installed, without loading weights; `llm/fermi_client.py` wraps each one.
- A plan is rejected when it fails validation, when Noether can't compute
  it, or when the result is impossible: the wrong dimensions, or negative
  for a quantity that can't be (a mass, a resistance, a frequency, an
  absolute temperature; `never_negative`). R1 = R - R2 with the givens in
  the wrong equation gives a negative resistor, and the retry fixes it. A
  result of exactly 0 is also rejected when the plan assumed a 0 (`trivial`):
  "how fast does a rock hit the floor?" solved for the acceleration with
  v = 0 answers a question nobody asked. A zero from stated values stands.
- Plans are tiered. A possible plan that uses every value the question
  states (`stated_givens`) is answered at once. A possible plan that leaves
  one unused ("the emissivity comes out to 0.017", never plugged in) is kept
  as a fallback while the remaining attempts look for one that uses them
  all; if none does, it is shown with the unused value in its caveats and
  confidence capped at 0.4. An impossible result is never shown: when every
  attempt is impossible, the answer degrades and says why. (Until v0.4 the
  first impossible plan was kept as a last resort, which showed "0 N" for a
  braking car.)
- Plans decode greedily, so a seed changes nothing. Each retry rotates the
  order the retrieved equations are listed in, which changes the prompt.
- The `auto` provider (the default) uses the Fermi models when tellus,
  solem, or celeste is installed, and the fake client otherwise.
  `fermi-luna-1` is only used when forced with `--model`. Torch is only
  imported on the Fermi path, so the website (Pyodide) never needs it.
- `Answer.models` maps each stage to its model (`"template"` when the
  explanation fell back), and `Answer.plan_attempts` counts plans tried.
  That count is the escalation statistic this ADR promises.

---

## ADR-011: Install channels: curl, irm, and a Homebrew tap

**Status:** Accepted (v0.2.0), decided by the maintainer. Amended by ADR-023: WinGet is now a channel.

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
- **Not:** WinGet (not submitting; reversed by ADR-023), and PyPI only at v1.0.
- A CI workflow runs both installers on real Linux, macOS, and Windows
  machines whenever they change.
- The installers are served from `askphysics.vercel.app/installers/`, built
  from this repo's `main` by `vercel.json` (copied, never edited, with a
  `text/plain` content type so `irm | iex` works).

**Consequences.**
- Each release needs the tap's formula bumped (URL and sha256); the steps
  are in `docs/RELEASING.md`.
- Installs need network access to GitHub, PyPI, and (on Linux) PyTorch's
  index. Offline installs are out of scope until wheels ship on PyPI.
- Installers never touch the system Python, so uninstalling is one command.

## ADR-012: Ship trained weights; users never train

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** The Fermi models are trained from scratch (ADR-009), which takes
hours to a day on an M5 Pro. Asking every user to do that would make Ask
Physics unusable. ROADMAP had weight downloads in v0.9, but the CLI needs
them the moment v0.3 wires the models in.

**Decision.**
- The maintainer trains the models; users only download them. The `model`
  training commands are for maintainers and contributors.
- Weights ship as GitHub release assets, never in git: one
  `fermi-<name>-1.safetensors` per model in bf16 (about 6 MB for tellus,
  60 MB for solem, 240 MB for celeste), plus its config and the shared
  tokenizer.
- The package carries a manifest pinning each asset's URL, size, and
  sha256. `askphysics model pull` downloads to the models directory and
  refuses anything that doesn't match. Safetensors never run code on load.
- The curl and irm installers run `askphysics model pull` at the end
  (tellus and solem, about 66 MB). The Homebrew formula declares the same
  files as checksummed resources, so a brew install works offline after.
- celeste downloads on the first question that needs its escalation shot,
  or up front with `model pull --all`. Most questions never need it.
- With no weights installed, `ask` still runs (degraded, as today) and says
  to run `askphysics model pull`.

**Consequences.**
- Moves "weight download and verification" from v0.9 to v0.3.
- Every model release bumps the manifest; a release can't ship weights the
  package doesn't pin.
- A full install, torch included, is a few hundred MB. Accepted: the
  maintainer chose to keep torch for inference rather than add a separate
  numpy engine.

**Implementation (v0.4).**
- `askphysics model package --model NAME --release TAG` (maintainers) writes
  a bf16 copy of the model, a model card (`MODEL_CARD.md`: architecture,
  training settings and curve, eval results, answering speed through the
  pipeline, credit), and the prose attribution when `--attribution` is given.
  It puts the release assets in `build/release/assets/` and pins each one in
  `src/askphysics/lm/weights.json` by URL, size, and sha256. Loading casts
  bf16 back to float32, so nothing else changes.
- `askphysics model pull` (`lm/weights.py`, no torch) downloads tellus and
  solem, or `--model NAME`, or `--all` for celeste too. Every file goes to a
  staging directory and is checked against the manifest; the model directory
  is replaced only when all of them match, so a failed or tampered download
  never leaves a half-installed model. A locally trained model of the same
  name is kept unless `--force`.
- celeste is routed as *available* when it is published but not downloaded
  (`Settings.auto_pull`, `ASKPHYSICS_AUTO_PULL=0` to turn it off). Its client
  downloads the weights the first time a question reaches its escalation
  try; a failed download fails that try like any other, so the answer
  degrades instead of crashing.
- `askphysics ask` pulls tellus and solem itself on the first question when they
  are published but not installed, so nobody has to run `model pull` (pip or
  brew installs, a skipped installer step, models published after install).
  The logic is `_auto_pull` in `cli.py`, which calls the same verified `pull`,
  shows the same `PullView` after a one-line heads-up, and uses the same
  `Settings.auto_pull` switch as celeste. It prints to stderr with `--json` or
  without a terminal, never replaces a complete local model, makes no network
  request while the manifest is empty, and on any failure leaves `ask`
  degraded as before; the next `ask` tries again. `Pipeline` and
  `askphysics.web` never download.
- The installers run `askphysics model pull --if-published` last, which
  succeeds quietly while nothing is published (`ASKPHYSICS_SKIP_MODELS=1`
  skips it). With no models installed, `ask` says how to get them.
- Homebrew: a formula can't write to the user's home directory, so instead
  of resources it prints a caveat to run `askphysics model pull`
  (`docs/RELEASING.md`). A brew install needs the network once for the
  models.

---

## ADR-013: An in-browser website, early

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** ADR-008 put the website after v1.0, mostly to avoid hosting,
auth, rate limiting, and cost control for public traffic. The maintainer
wants Ask Physics usable at askphysics.vercel.app now, with the actual
program, not a mock-up.

**Decision.**
- The site runs the real `askphysics` package in the visitor's browser with
  Pyodide (Python on WebAssembly), in a Web Worker. SymPy, pydantic, and
  numpy come from Pyodide; Pint comes from PyPI. Nothing runs on our servers,
  so there is no hosting cost per question, nothing to rate-limit, and no
  accounts or keys.
- `askphysics.web` is the only bridge: a question in, the `Answer` plus
  pre-formatted strings out, formatted by `askphysics.pretty` exactly like
  the CLI card. The browser bundle leaves out the terminal (`cli`, `ui`) and
  torch (`lm`) code.
- One page: the pitch, a live "Ask a question" box, how it works, the model
  portraits, and the install commands. The engine (about 15 MB) loads only
  when the visitor reaches the question box, then is cached.
- `scripts/build_site.sh` builds it (Vercel runs it, ADR-011); a CI workflow
  drives the built page in headless Chromium against the real Pyodide CDN
  and PyPI on every change.

**Consequences.**
- Until v0.3 the site runs the same stand-in planner as the CLI and says so
  on the page.
- The Fermi models can't run in the browser through torch. Before they
  power the site, they need a browser-capable inference path (for example a
  small numpy forward pass reading the same safetensors weights); that
  choice gets its own ADR in v0.3.
- Accounts stay unnecessary while all compute is on the visitor's device.
  They arrive with the hosted API after v1.0 (Q18), the first thing that
  costs us money per question.

---

## ADR-014: The math engine is called Noether

**Status:** Accepted (v0.2.0), decided by the maintainer

**Context.** "The symbolic algebra machine" was a mouthful, and it hid what
we built: SymPy and Pint are libraries, but the restricted parser,
unit-aware solving, root selection, sanity checks, and Fermi engine around
them are ours.

**Decision.** The math layer (`src/askphysics/solver/`) is named **Noether**,
for Emmy Noether, whose theorem ties symmetries to conservation laws. The
models read the question; Noether does the math. We keep building on SymPy
and Pint rather than writing our own computer algebra: correctness is the
project's core promise, and decades of SymPy users find bugs a homemade
engine would ship.

**Consequences.** Docs, the README, and the website say Noether. Module
names stay as they are (`askphysics.solver`); the name is branding, not a
code move.

---

## ADR-015: Dimension-aware constrained decoding for plans

**Status:** Accepted (v0.3), decided by the maintainer

**Context.** The first `askphysics model eval` of fermi-tellus-1 found the
right category 99% of the time and the right equation 93% of the time, but
only 48% of plans computed the right answer. The misses were mostly values
with the wrong dimensions: in "moving at 69 m/s has 4.6 kg*m/s of momentum"
it set p = 69 and v = 4.6; it turned "570 pounds" into 570 kilograms; asked
for a distance, it solved for a mass and made 87.4 kg a distance. The
decoder allowed all of this, because a value only had to be some number in
the question and a unit only had to be some unit.

**Decision.** For standard plans the decoder enforces dimensions as it
writes:

- A known value is chosen from (number, unit, origin) options: quantities
  written in the question whose units fit the variable, with the unit that
  follows the number kept attached (origin `given`); table constants that
  fit (`constant`); or an assumed 0 in the variable's unit
  (`assumption`).
- The target is chosen from variables the question leaves open: those whose
  dimensions have fewer stated quantities than the chosen equations have
  variables. Table constants are never targets. If the count rules out
  everything, every non-constant variable stays open.
- Each stated quantity fills at most one variable, and a table constant or
  assumed 0 may only fill a slot when the variables still to fill
  outnumber the unused quantities that fit them. Added after the second
  eval (71.5% valid plans), whose misses wrote one speed into both v and v0
  or wrote v = 0 with "34 mph" sitting unused.

- The only assumed filler is 0, and only for an initial or launch speed (or
  one of two collision speeds) when the question has a rest cue ("from rest",
  "dropped", "released", "sitting still"), or for an acceleration when the
  motion is steady ("cruises at a steady 30 m/s"); 0 must also be inside the
  variable's `typical_range`. A stated "from 0 to 20 m/s" is two given values,
  not an assumption. 0 never fills a final speed, momentum, force, mass, or
  time, and never the target (the fifth eval wrote g = 0; the solem eval wrote
  p = 0 for kinetic energy and a = 0 for the acceleration the question asked
  for, #91). If no value is legal for a slot, the plan fails
  (`PlanValidationError`) and the router tries again or the answer degrades;
  the decoder never fills the slot in with a loose value, and never writes
  one stated quantity into two variables.
  The data factory never assumes anything else.
- A table constant fills only a slot that is that constant: a variable named
  "... constant" with its units (`G`, `k`, `R`, and `c` in the relativity
  equations) or one with its own symbol (`g`). Matching units are not enough:
  v = c is a speed of 299792458 m/s, and the solem eval wrote it into "an
  average speed of 23.2 m/s" and "an angular velocity of 5 rad/s" (#91).
  Three constants also fill the plain variable they are the value of, only
  when the question says so: g for an acceleration when something falls or is
  thrown, c for a speed when the question is about light or radiation, and e
  for a charge when it names an electron or proton. A variable a table
  constant can fill (g) is only the target when no open variable lacks such a
  fallback. Added after the fourth eval, where "lifting it 11 m took 11000 J,
  what is its mass?" was solved for g with an invented m = 1 kg.
- "From A to B" (a number, a unit, then a second number and unit) names the
  initial and the final value, in that order, when the equation has exactly one
  initial and one final variable that fit: "change its speed from 20 m/s to
  60 m/s" is v0 = 20 and v = 60 (solem had them backwards). It is read by a
  rule (`transition_locks`) like a label, so a label that disagrees wins.

- A number with no unit after it is a dimensionless quantity, so it can
  fill a friction coefficient, an efficiency, or a count; a number with a
  unit never can. Added with the dimensionless equations (v0.3).

The first equation is constrained the same way (`equation_options`): it
must have room for every quantity the question writes with a unit, meaning
each one fits some variable and no dimensions are stated more often than
the equation has variables of those dimensions. Without this, "a thing at
4.1 m/s carries 620 J, what is its mass?" could pick KE = p²/2m, leave
the speed unused, and assume p = 0. Bare numbers don't count, since "the
resistance 2" is a label. If no retrieved equation has room, every one
stays on offer.

Two facts the question states outright are read by rules, not the model
(`lm/reading.py`), and the decoder narrows its options to agree:
- **Labels.** "fs is 758 Hz", "di = 1.8 m", or "1.8 m (di)" lock that
  quantity to that symbol. A locked symbol can't be the target, and no
  other variable can take its quantity. A symbol labelled with two values
  is not locked.
- **The ask.** The words after "what is", "find", "solve for", and the like
  name the unknown, matched against each variable's names (the database
  name and the factory's synonyms). The name that starts earliest wins,
  then the longest, so "the mass of the planet" means M and not m, and
  names compete across the retrieved equations, so "what is its mass?"
  rules out W = Fd. A symbol counts only when it is the whole ask ("Solve
  for vs"), because "do" and "a" are also words.
Both rules only narrow, and are ignored when they would leave nothing. On
20,227 generated plans they never ruled out the gold equation, target, or
value, and they pin the equation 85% of the time and the target 99%. They
fixed five of the six misses in tellus's 110-equation eval without
retraining.

Model-written text is held to the same standard. A standard plan's
assumptions are picked whole from the equation's own list and its
scenarios' (`assumption_options`), the exact sentences the data factory
trains on, so a small model can't garble them. Free text (reasons,
redirects, strategies, explanations) can't repeat a word back to back or
repeat a run of six words, which stops greedy loops; on 13,000 factory
texts this never blocked a gold one. Afterwards the pipeline checks what it
shows (`readable`, `refusal_reason`, `usable_redirect`): looping or
off-topic text is replaced with a plain sentence, and a redirect that
isn't a question is dropped. On 6,047 factory explanations and 2,601
refusals these checks rejected nothing.

Classifications go further: their reasoning and redirect are never free
text. The reason is picked from the factory's reviewed sentences
(`reason_options`): standard and Fermi from their lists, and out-of-scope
ones with their slot ("{emotion}", "{abstract}") filled by a run of words
copied from the question. A fixed sentence that names something ("a dream
is an experience") is only offered when it shares a word with the
question. The redirect comes from the same template as the chosen reason
(`redirect_options`), so a math refusal suggests the math redirect. Pure
math questions ("slope of a curve", "ten divided by three") got their own
out-of-scope templates, since nothing in the bank fit them. On 10,665
factory classifications, the gold reason and redirect were always options.

Labels and asks were widened after the v0.3 evals, where every confidently
wrong answer was a swap or a misread ask. A variable's *name* labels a value
the way its symbol does ("primary turns: 61000", "the speed at the end comes
out to 160 mph", "110 m for the distance to the object"), unless that clause
is the ask ("Find the heat out: ...") or two names claim the same value. Asks
that name a kind of quantity map to variables by the words in their names:
"how fast" to a speed or velocity that isn't the initial one (unless the
start is asked about), "how far" and "what height" to a distance or height,
"how long does it take" to a time. Refusal slots are only filled from the
question's match against that refusal's template, never from an arbitrary
run of words. On 10,669 generated plans and 10,665 classifications, none of
these ruled out a gold answer.

The solem 3,000-question eval added four more rules, one per kind of miss.
- *Twins.* Some equations have identical variables and differ only in
  meaning: series and parallel resistors, orbital and escape speed. Words
  decide: an equation whose twin's own tags the question uses, and whose
  own it doesn't, is ruled out ("in parallel" rules out the series
  formula). When the question names neither, both stay, and the sanity
  check marks the answer ambiguous (confidence capped at 0.4) instead of
  confidently picking one. The data factory opens each twin's questions with
  a sentence that says which (`EQUATION_CONTEXT`) when its own wording
  doesn't, so training never teaches a coin flip.
- *Labelled bare numbers need a home.* A number without a unit that a
  variable's name or symbol labels ("the emissivity comes out to 0.017") is
  a value, so equations with a variable for it are preferred. Unlabelled
  bare numbers still never count ("resistor 2" is a name).
- *Pairs go in order.* When neither half of an indexed pair (m1 and m2, v1
  and v2) is labelled, the first value stated goes to the "1" variable and
  the last to the "2" one.
- *A name labels without a connector* ("the second mass 1100 kg") only when
  the value has a unit, since "mass 2" is a name, not a mass of 2.
On 10,813 generated plans and 10,558 classifications, none of these ruled
out a gold answer; on 7,187 gold plans the sanity check found no unused
value, no ambiguity, and no trivial zero; and none of the 219 factory
questions for twin equations is ambiguous.

v0.4 added three rules from solem's evals and from chaining:
- *An ask needs a home.* When no equation with room for the stated values
  has a variable of the kind the ask names (compared by units, not names:
  the final velocity in v = v0 + at answers "its speed"), the equations that
  do have it come first, and the plan chains to the rest. "Pushed with 12 N
  for 4 s from 2 m/s: work out x" starts from x = v0 t + a t^2/2, not the
  impulse-momentum theorem, which has room for every value but no x.
- *A unitless slot needs a value.* An equation with a unitless variable (an
  emissivity, a coefficient) is ruled out when the question never mentions
  it, states it, or asks for it; solem once filled an emissivity with the
  area's 1200. The sanity check also flags any unitless input more than ten
  times outside its usual range, so a borrowed value is never confidently
  wrong.
- *"How strongly" and "how hard" ask for a force*, not an energy.

**Chaining (v0.4).** A plan may cite several equations and list the
intermediate values as unknowns. Noether solves them in dependency order: any
equation with exactly one value still missing is solved for it, until the
target is found (`compute`, `Answer.steps`). A symbol shared between cited
equations must have the same dimensions in each, so weight W (N) never stands
in for work W (J). The data factory writes chained problems from a reviewed
list (`templates.CHAINS`), about one plan example in ten, keeping only chains
no single equation in the database answers.

**Temperatures (v0.4).** Celsius and Fahrenheit values are converted to
kelvin before substitution, by what the variable means: "water at 20 degC"
is 293.15 K, "heated by 20 degC" is a change of 20 K (`Variable.is_change`).
A temperature answer asked in Celsius or Fahrenheit is also given on that
scale. The factory writes some temperatures in degC and degF, changes
included.

Fermi plans keep the looser rules until the Fermi engine (v0.7). A question
that states an irrelevant quantity with the same dimensions as a variable
would have it forced into the plan; the data factory never writes one, and
real questions rarely do. Revisit with the v0.5 evals.

**Consequences.** This is the decoder doing bookkeeping on units, not
arithmetic: the model still picks among legal options, and Noether still
solves and checks everything afterwards. The training format is unchanged,
so no retraining is needed, and every gold plan from the data factory fits
the constraints (a test checks this). What remains for the model is
genuine reading: which of two masses is the first, and whether a speed is
the initial or the final one.

---

## ADR-016: Real-phrasing data from OpenStax Physics (CC BY 4.0), text only

**Status:** Accepted (v0.3), decided by the maintainer

**Context.** fermi-tellus-1 reaches 86.5% valid plans on held-out
*templates*, but every question it has seen was written by our data
factory. Real people phrase things differently, and we have no measurement
of that at all. We need human-written physics questions and prose. The
project and its weights are MIT, so the source's license has to allow
commercial use without share-alike.

Checking licenses at the source (each book's `LICENSE` file and collection
metadata in the `openstax/osbooks-*` repositories) showed that OpenStax
*University Physics* and *College Physics* are **CC BY-NC-SA 4.0**, not CC
BY as our docs said. OpenStax *Physics* (2020, high school) is **CC BY 4.0**:
about 2,000 exercises and 100 sections of prose. Its preface notes that some
artwork came through separate permissions.

**Decision.**

- Use OpenStax *Physics* (2020) as the source of real-phrasing data. Pulls
  part of the v0.6 "CC-BY text" deliverable forward.
- **Text only.** No figures, captions, or media.
- Imported text keeps its license. It lives in its own directory with its
  own `LICENSE` (CC BY 4.0) and an attribution file naming the book, its
  authors as OpenStax credits them, the URL, the license, and that we
  extracted and reformatted it. It is never relabelled MIT.
- Uses, in order: (1) an eval of real questions our equations can answer,
  with gold plans written by the project and reviewed by the maintainer;
  (2) its other questions as classify examples; (3) its prose as an extra
  language-modeling stage before task training.
- Model cards and `THIRD_PARTY_LICENSES.md` credit OpenStax for any weights
  trained on it. The OpenStax name and logo are not used beyond that
  attribution.
- Rejected sources: *University Physics* and *College Physics* (NC-SA),
  SciQ (CC BY-NC), ScienceQA (CC BY-NC-SA), GPT-generated sets such as
  camel-ai physics (ADR-009), MMLU and GPQA (benchmarks; training on them
  contaminates evaluation). Physics StackExchange (CC BY-SA) needs its own
  ADR first, because share-alike may reach trained weights.

**Consequences.** We get the first measurement on human phrasing, and
training text that isn't ours. Gold plans are labor: an external model may
not write them (ADR-009), so the project writes them and the maintainer
reviews them. CC BY attribution travels with every artifact that contains
or was trained on the text. This is a careful reading of the licenses, not
legal advice; a lawyer should review `THIRD_PARTY_LICENSES.md`, this ADR,
and the model cards before any commercial launch.

**Implementation (v0.3), use (3).** `scripts/extract_openstax.py` pulls
the body prose from the book's source at a pinned commit, refusing anything
whose `LICENSE` isn't CC BY 4.0. It keeps prose paragraphs, including the
worked examples and boxed features OpenStax wrote, and drops figures,
captions, media, tables, exercises, display equations, the preface, and all
teacher-support material (it quotes state standards that aren't OpenStax's
to license). Inline math is written as text and figure references as "the
figure". The result, about 1,650 paragraphs and 118,000 words, is in
`third_party/openstax-physics/` with the book's `LICENSE` and an
`ATTRIBUTION.md`. `askphysics model train --prose ... --prose-steps N
--prose-share F` trains on it alone for N steps and mixes it into a share F
of later batches; the tokenizer can learn from it too.

**Implementation (v0.4), uses (1) and (2).** The extractor also writes
`questions.jsonl`: 969 exercises from the practice, end-of-chapter, and
test-prep sections, with their answer options and any solution text,
leaving out every exercise that needs a figure or table. The project wrote
gold plans for 49 multiple-choice problems our equations can answer
(`real_eval.jsonl`); a test checks each plan's answer is within 5% of the
book's keyed option and of no other option (the book rounds to two
figures). `askphysics model eval` asks them through `Pipeline.solve`, the
model under test doing every stage, and reports classification, retrieval,
and answers separately. Of the rest, 180 problems that state a quantity and
need no picture become standard classify examples in their chapter's
domain, five times each with different reasoning; the eval's 49 never do.
Conceptual questions are left out for now (open question Q19).

---

## ADR-017: A 20M-word prose corpus from pre-relicensing OpenStax and public-domain books

**Status:** Accepted (v0.4), decided by the maintainer

**Context.** solem learned English from one 118,000-word book (ADR-016), and its
held-out prose loss (5.07) shows it. Writing well needs millions of words. The
project is MIT and the maintainer's rule stands: only text whose license lets
anyone use it commercially without share-alike.

Checking the `openstax/osbooks-*` repositories in October 2026 found that
OpenStax relicensed almost every book from CC BY 4.0 to CC BY-NC-SA 4.0 between
2026-03-12 and 2026-03-23 (one commit per repository, e.g. `b375d3f` "updating
license to CC BY NC-SA"). Before that, each repository's `LICENSE` and every
book's collection metadata said CC BY 4.0. Calculus and Organic Chemistry were
never CC BY; Physics and Statistics still are.

**Decision.**

- **OpenStax, pinned before the switch.** CC BY 4.0 grants a "worldwide,
  royalty-free, non-sublicensable, non-exclusive, irrevocable license" (section
  2.a.1). Versions published under it stay usable under it after the publisher
  relicenses newer ones. `third_party/corpus/sources.json` pins each repository
  to its last commit before the switch, and the build refuses to read a commit
  unless its `LICENSE` and the book's own metadata both say CC BY 4.0. Nothing
  after a switch is used, so later corrections are not in the corpus. The
  maintainer asked for anything in English, because explaining well comes
  from reading good explanations, so every OpenStax book that was CC BY
  counts: 52 books across 33 repositories, from physics, chemistry, biology,
  and mathematics to history, economics, sociology, philosophy, writing,
  business, and psychology, about 6.8M words after removing repeats. Left
  out: books that were never CC BY (Calculus, Organic Chemistry, Business
  Law, Accounting), and any whose `LICENSE` file is non-standard or disagrees
  with the book's metadata (Introduction to Business, Neuroscience, Workplace
  Software Skills, Foundations of Information Systems).
- **Public-domain books for the rest.** Project Gutenberg texts in English:
  the sciences first, then technology and medicine, then expository
  non-fiction (history, geography, philosophy, social science, law,
  education, the arts), then language and literature, until the corpus
  reaches 20M words. Poetry and drama are skipped. A book
  qualifies only if every person the catalog credits (authors, translators,
  editors) died in or before 1955, which makes it public domain in the US and in
  life-plus-70 countries. Gutenberg's header, footer, and trademark are removed.
- **Not committed.** The corpus is about 130 MB, so `scripts/build_corpus.py`
  builds it locally under `build/`. The repository holds the manifest, the
  extractors, and their tests; each build writes `ATTRIBUTION.md`, the CC BY
  license text, and a lock file listing exact commits, Gutenberg ids, and file
  hashes.
- **Cleaning.** Text only, as in ADR-016; reference-list entries and repeated
  paragraphs removed; paragraphs containing slurs or swear words dropped, by
  the maintainer's rule of none at all in what the models learn from (old
  books and quoted history have both). The list is stored as hashes, and
  words with innocent meanings (a surname, a crack in a shutter, a bloody
  nose, a finger prick, a rooster, a donkey) are left off it.
- Model cards for weights trained on the corpus carry its attribution.

**Consequences.** About 20x the real English solem has seen, with no
non-commercial or share-alike text. The pre-switch reading of CC BY is the
standard one, but it is a legal judgment: a lawyer should review this ADR with
ADR-016 before any commercial launch, and if OpenStax asserts otherwise the
pinned books come out of the manifest. Old public-domain prose is
old-fashioned and some of its physics is obsolete; that is acceptable because
the models only learn language from it, and Noether does the physics. The
build downloads a few hundred files from a volunteer-run site, so it pauses
between requests.

---

## ADR-018: Noether and Fermi as their own packages, in three repositories

**Status:** Proposed by the maintainer; work starts after `fermi-celeste-1` is trained

**Context.** One package, `askphysics`, holds three things that change at
different speeds and could each be useful alone: Noether, the math engine
(`solver/`, units, the equation database and its validation); the Fermi models
(`lm/`, training, decoding, weights, the data factory); and Ask Physics itself
(the pipeline, retrieval, the CLI, the website, the docs). Today a change to the
website's CSS and a change to the training loop land in the same repository,
the same CI, and the same release.

**Decision (proposed).**

- Three repositories under the same owner, each MIT:
  - **noether**: the `noether` package. SymPy and Pint math, dimension checks,
    the equation, constant, and Fermi-assumption databases with their loader
    and `validate-data`, and the compute and sanity-check stages. No torch.
  - **fermi**: the `fermi` package. The model configs and architecture, the
    tokenizer, training, constrained decoding, the data factory, `model pull`
    and weights, `bench` and `eval`. It depends on `noether` (the factory
    solves every example with it) and keeps `third_party/` with its licenses
    and attribution, since that text trains the models.
  - **ask-physics**: the app. The six-stage pipeline wiring the two together,
    retrieval, the CLI, the website, Ask Physics Docs, and the end-to-end evals.
    It depends on `noether` and `fermi` (the website only on `noether`, as now).
- Each package publishes to PyPI on its own version; ask-physics pins ranges.
- The golden rules travel with the code: noether owns "every number has units"
  and "every equation has a source and license"; fermi owns "the models never
  do arithmetic" and "tests never need weights".

**Open before starting.**

- `lm/evaluate.py` imports `compute` and `sanity_check` from the pipeline, and
  `lm/` reads `normalize` and the shared models; those move into noether, or
  behind an interface, so fermi doesn't depend on the app.
- Where the shared Pydantic models (`Plan`, `Classification`, `Answer`) live:
  likely noether, since plans are its input.
- Moving history: `git filter-repo` per package keeps each file's history.
- The release, installer, and Homebrew flows, which today ship one package.

**Consequences.** Smaller, faster CI per repository, and Noether usable on its
own. The cost is three release trains and cross-repo changes, which is why it
waits until the models stop changing weekly.

## ADR-019: Train on Apple Silicon with MLX; PyTorch stays for inference and other platforms

**Status:** Proposed by the maintainer's request; implemented on `feat/mlx-trainer`, to be
confirmed by the first full runs on the M5 Pro

**Context.** The maintainer trains the Fermi models on an M5 Pro with 48 GB. Training with
torch's MPS backend grew past 40 GB of process memory and then swapped, and throughput fell to
about 256 target tokens per second. Attention is the main cost: with batch 16 at 1024 tokens,
each layer's (heads, T, T) attention matrix is about 0.5 to 1 GB, so the activations of
`fermi-celeste-1` alone need more memory than the machine has. MLX is Apple's array framework
for Apple Silicon (MIT licensed, `pip install mlx`). It keeps unified-memory use under control
with a hard memory limit and a cache limit, and it can recompute each block's activations in
the backward pass.

**Decision.**

- Training on an arm64 Mac uses MLX. `askphysics model train --backend auto` (the default)
  picks MLX when `mlx.core` imports on an arm64 Mac and `--device` is not `cpu` or `cuda`, and
  torch otherwise. `--backend torch` forces torch; `--backend mlx` fails with a clear message
  when mlx is missing. `askphysics model backend` prints the choice, and `scripts/train.sh`
  passes it through.
- Inference (`ask`, `model eval`, the website), and training on Linux and Windows, stay on
  torch. Nothing else in the package imports mlx, and the MLX modules are imported only when
  the MLX backend is chosen.
- The weights format is unchanged. `lm/mlx_model.py` uses the torch parameter names and
  shapes, `lm/mlx_train.py` writes `model.safetensors`, `config.json`, and `tokenizer.json`
  through the same helpers as `lm/checkpoints.py`, and `load_model` in torch reads them.
  Logits and loss agree with torch within 1e-4 in fp32, and a test checks it.
- The two trainers share the data preparation, the batch sampler (so one seed gives the same
  batches on either), the learning-rate schedule, the metric keys, and the training summary.
  The optimizer state differs (MLX keeps its moments under `m.<param>` and `v.<param>`), so
  `train_state.json` records `"backend"`, and a run resumes only on the backend that wrote it.
- Memory on MLX: a memory limit (`mx.set_memory_limit`, 70% of system RAM by default,
  `--mlx-memory-gb` to change it), a cap on the cache kept for reuse (`--mlx-cache-gb`, 4 GB by
  default), and `--checkpoint-blocks` to recompute each block's activations in the backward
  pass (`mx.checkpoint`). The torch trainer takes the same `--checkpoint-blocks` flag
  (`torch.utils.checkpoint`, non-reentrant), used only while training, so evaluation and
  decoding never recompute. The loss upcasts the logits to fp32 once and does not materialize a
  log-softmax copy. The optimizer step runs under `mx.compile`, which is on by default.
- Precision: parameters are always fp32. `bf16` runs the forward pass in bfloat16 with fp32
  master weights and an fp32 loss, as torch autocast does on MPS. `auto` does that on the GPU
  and runs fp32 on the CPU, as torch's `auto` does. Validation runs in fp32 on both backends,
  since torch's autocast covers only the training step.

**Consequences.** Two training loops must stay in step: a change to the batch, schedule, or
logging code lands in `lm/train.py` and is shared where possible, but the model and optimizer
steps exist twice, and `tests/test_lm_mlx_train.py` checks one step of each against the other.
The MLX trainer is new code on a young framework, so the torch trainer remains the reference
and the fallback. Weights are unchanged, so a model trained either way is installed, evaluated,
and packaged the same way. MLX is MIT licensed, so it adds no license obligation beyond the
attribution in `THIRD_PARTY_LICENSES.md`. Linux and Windows users see no change at run time: the
`[mlx]` extra is Mac-only, and `auto` resolves to torch there. The `dev` extra adds `mlx[cpu]`
on Linux, so CI runs the MLX tests on CPU; without it those tests skip.

## ADR-020: One terminal look for every script, drawn by a small Rich module with a sh fallback

**Status:** Accepted (issue #88, building on the training view of #87)

**Context.** The workflow scripts printed bare `==>` lines and then went quiet while pip, pytest,
or a download ran, and each script styled its own output. The installers run before anything is
installed, and `build_site.sh` runs on Vercel with no askphysics, so the look cannot depend on the
package being importable.

**Decision.**

- `scripts/lib.sh` is the one set of helpers (`ui_begin`, `phase`, `gate`, `ui_result`, `ui_ready`,
  `ui_end`). When `python -m askphysics.shell_ui` can be imported it draws with Rich, in the theme
  of `ui.py`; when it cannot, a pure-sh renderer with the same glyphs and columns draws instead.
  `install.sh` embeds that renderer (it is piped from `curl`), and a test keeps the copies equal.
  `install.ps1` has the matching look natively in PowerShell.
- The renderer is a module run with `python -m`, not a CLI subcommand: importing `askphysics.cli`
  costs about 0.7 s (sympy, Pint), and every step of every script calls it. `ui.py` no longer
  imports `pretty`, so the module starts in about 0.1 s. The #87 `model phase` command is gone;
  scripts call the module.
- Off a terminal, under `NO_COLOR`, or with `TERM=dumb`, output is plain `start:` and `done:`
  lines with no escape codes; `DRY_RUN=1` prints `-- title` and `+ command` as before.
- A long command's output is folded into three dimmed lines and saved in full to
  `$TMPDIR/askphysics-logs/`; a failed step prints the command, the last lines, and the log path.
- The eval ETA prices the remaining examples by kind (classify and plan cost different amounts),
  so it shows after the first example and does not swing with the mix. The rescue pass now runs
  after scoring, over just the misses, so it has a known length and its own bar; the results are
  the same as before because every example is decoded greedily and independently.

**Consequences.** There are two renderers to keep alike (Rich and sh), checked by a test that
compares their plain output and the width of the ready panel. Rich and the module are used only by
the scripts and the CLI; the website bundle leaves them out. Rich is already a dependency, so
nothing new is installed.


## ADR-021: Run the Fermi models in the browser with a numpy forward pass

**Status:** Proposed. Phase 1 (the backend, the decoder seam, and parity tests) is implemented on
`feat/browser-inference`; the speed and memory below are estimates until phase 2 measures them in
real Pyodide (issue #66).

**Context.** The website runs the real `askphysics` package in Pyodide in a Web Worker (ADR-013).
Pyodide has numpy, SymPy, and pydantic, but no torch, so the Fermi models cannot run there as
written. ADR-013 said this choice "gets its own ADR": a browser-capable inference path that reads
the same weights. Issue #66 needs it before tellus, solem, and celeste can answer on the site.
ADR-012 chose to keep torch for inference rather than add a numpy engine; that holds for the CLI,
but the browser has no torch.

**Decision.**

- **A numpy forward pass** (`lm/numpy_model.py`, `NumpyFermiLM`) for the Fermi transformer: the same
  RMSNorm, rotary embeddings, causal attention with a KV cache, SwiGLU, and tied head as
  `lm/model.py`. It reads the same `model.safetensors` that `askphysics model pull` installs: bf16
  weights become float32 by placing the 16 bits in the top half of a float32, which is exact.
  Weights stay in torch's `(out, in)` layout. It imports no torch.
- **A small safetensors reader of our own** (`read_safetensors`, about 40 lines: an 8-byte length,
  a JSON header, raw bytes). Pyodide's package set does include `safetensors` 0.7.0 (a recipe in
  `pyodide/pyodide-recipes`, depending on numpy), but I could not confirm that the Pyodide version
  pinned in `web/worker.js` ships it, and its numpy path cannot return bf16 without `ml_dtypes`.
  Reading the file ourselves removes both questions. The reader accepts F32, F16, and BF16,
  validates every offset against its shape, and refuses anything else with `ConfigError`.
- **Constrained decoding is unchanged and written once.** `Decoder` (`lm/generate.py`) now sits on a
  seam, `Engine` (`lm/engine.py`): an engine turns tokens and a cache into the next-token logits,
  and the logits cross the seam as one-dimensional float32 numpy arrays. The masks, the number
  guard, the slot rules, the no-repeat rule, and the greedy choice all run over numpy, whichever
  framework produced the logits. `TorchEngine` (`lm/torch_engine.py`) wraps `FermiLM`;
  `NumpyFermiLM` is itself an `Engine`. `Decoder(model, tokenizer)` takes either a `FermiLM` or an
  `Engine`, and `generate.py` no longer imports torch at module level. The seam is five small
  methods: `next_logits`, `generator`, `sample`, `cache_length`, and `truncate`. `FORMAT_VERSION`,
  the formats, and the training code are untouched.
- **Both loaders refuse the same files.** The checks on a saved model (files present, `config.json`
  against the weights' metadata, task format version) moved from `checkpoints.py` into
  `lm/config.py`, torch-free, and both loaders call them.
- **Not decided here:** where the weights are hosted for the site (Q20 in `OPEN_QUESTIONS.md`, which
  needs the maintainer), and which matrix-product kernel Pyodide should use (below).

**Alternatives.**

- *ONNX Runtime Web (wasm or WebGPU).* Faster, with SIMD and threads in wasm and far more on
  WebGPU. But it needs an export step that has to track `model.py` (a graph with a KV cache,
  dynamic shapes, and rotary tables), a second runtime of several MB, and a bridge: ONNX Runtime
  is JavaScript and asynchronous, while `Decoder` is Python that calls the model synchronously
  hundreds of times per question. Bridging means rewriting the decoder as async or blocking across
  workers with `Atomics.wait`. That is a lot of machinery for models small enough to run in numpy.
- *WebGPU kernels by hand.* The same bridge problem, narrower browser support, and the most code.
- *A hosted API.* ADR-013 chose a site where nothing runs on our servers: no cost per question, no
  rate limits, no accounts or keys. ADR-009 says no external model APIs. A hosted model would bring
  back everything those ADRs avoid (Q17, Q18).
- *numpy wins here because* there is one decoder, in Python, and one model definition per
  framework in the same package, tested against each other in CI. There is no export step to drift,
  no new runtime to download (numpy is already in the Pyodide load), and no server. Its cost is
  speed, estimated below. The seam keeps the door open: if phase 2 shows numpy is too slow for
  solem, an ONNX or WebGPU engine can be added behind `Engine` by a later ADR without touching a
  decoding rule.

**Parity (measured, CPU).** For random weights of `fermi-luna-1` and of tellus's shape, the largest
absolute logit difference from torch is about 1e-5 at logits of size 5 to 6 (about 2e-6 relative),
with and without the KV cache, and for bf16 weights written by `model package`. Classify, plan, and
explain give identical text on both backends for five questions over three weight seeds
(`tests/test_lm_numpy_model.py`). Greedy decoding is the default and is what the site uses. A
sampled explanation (`temperature > 0`) draws from each engine's own random generator, so it differs
between backends, as it differs between seeds; the number guard holds on both.

**Speed and memory in Pyodide (estimates).** Measured on a 4-core box, one thread, native numpy 2.5
with BLAS, on random weights and the real decoder:

| | tellus-shaped (3.2M) | solem-shaped (29.9M) |
|---|---|---|
| Prefill, 300 tokens: numpy / torch | 42 ms / 34 ms | 219 ms / 175 ms |
| One cached step: numpy / torch | 1.8 ms / 3.2 ms | 12.4 ms / 16.2 ms |
| Whole question on one model, classify + plan + explain: numpy / torch | 0.55 s / 0.89 s | 3.7 s / 4.5 s |

The work in a question, counted through the engine, is about 420 prompt tokens and about 290 cached
single-token steps (a classify prompt is about 35 tokens, a plan prompt about 350, an explanation
prompt about 100; counted with a 3,130-token tokenizer trained on factory text, so the shipped
8,192-token one will be a little shorter). In floating point operations that is about 4 GFLOP for a
whole question on tellus and about 43 GFLOP on solem. Of those, tellus's classify share is under
0.5 GFLOP, and on solem the plan is about 29 and the explanation about 9.

Pyodide's numpy is built without BLAS (`-Dallow-noblas=true` in its recipe), so every matrix
product is numpy's plain triple loop, single-threaded. A C copy of that loop runs at about 2.9
GFLOP/s on the box above, so I expect roughly 1 to 2 GFLOP/s in wasm on a recent laptop and about
half that on a phone. That gives, per question:

- **tellus doing every stage:** about 2 to 4 s on a laptop. Classify alone is well under a second.
- **solem planning and explaining:** about 20 to 40 s for one plan attempt plus the explanation
  (the plan is three quarters of that), and a failed plan retries up to five times (ADR-010), so a
  hard question can take minutes. This is the number phase 2 must beat or hide.
- **celeste (113M non-embedding parameters, about 4 times solem):** roughly 1.5 to 3 minutes per
  attempt on this path. Not viable here.
- A different kernel may change this a lot. Natively, `np.einsum` runs the same product at about
  10 GFLOP/s against 2.9 for numpy's plain `matmul` loop, because einsum has SIMD inner loops.
  Whether Pyodide's build keeps that edge is unknown until measured. Other levers, none built:
  reuse the KV cache of the shared prompt prefix across plan retries, and avoid re-feeding tokens
  the decoder already holds.

Memory: tellus is 6 MB on the wire, 13 MB as float32, and about 25 MB at its peak while loading.
solem is 60 MB on the wire and 120 MB as float32. Its peak while loading is about 240 MB (the
downloaded buffer, the Python copy of it, and the float32 arrays, before the first two are freed;
60 MB more if the file is first written to Pyodide's file system), and the KV cache adds up to 34
MB at the full 1,024 tokens. Both models resident come to about 135 MB of weights on top of
Pyodide's own footprint (I estimate 150 to 250 MB with SymPy loaded): comfortable on a laptop and
tight on a low-end phone. celeste is 476 MB as float32 with a peak near 1 GB: not on this path.

**What the visitor sees while a model loads.** The engine already shows four boot steps from the
worker. A fifth, "Loading the language models", would show a determinate download bar in megabytes.
tellus (6 MB) loads first so a question can be answered at once, because tellus alone can run every
stage (ADR-010); solem keeps downloading in the background and takes over for later questions. The
answer card's model line, which already records each stage's model, says which one answered. While
a long answer runs, the page shows the stage it is on ("Planning, attempt 2 of 5"). Weights are
fetched once and kept by the browser cache under content-hashed names, so a return visit does not
download them again. A model that cannot be fetched or does not fit in memory degrades to the
smaller one with a plain message instead of failing (issue #66, third item). All of this is phase
2; none of it exists yet.

**Consequences.**

- Two forward passes (`model.py`, `numpy_model.py`) plus the MLX trainer's (ADR-019) must stay in
  step. `tests/test_lm_numpy_model.py` compares numpy with torch on every run, so a change to
  `model.py` that is not mirrored fails there.
- The site bundle is unchanged: `scripts/build_site.sh` still leaves `lm/` out except the torch-free
  modules, and `llm/fermi_client.py` imports torch at module level. Phase 2 ships `engine.py`,
  `generate.py`, and `numpy_model.py`, and loads torch lazily in `FermiClient`.
- The CLI, evaluation, packaging, and training keep using torch (and MLX) exactly as before; ADR-012
  and ADR-019 stand for them.
- Phase 2 (needs real Pyodide, so not done here): wire `FermiClient` to the numpy engine in the
  worker; measure speed and memory in headless Chromium (`scripts/site_smoke.mjs`) and choose the
  kernel; settle where the weights are hosted (Q20); add the loading and progress states; smoke-test
  the eval questions through the built site. Tellus and solem ship only if those numbers are
  acceptable; celeste waits for a faster engine.

---

## ADR-023: Ship on WinGet, with a per-user installer, and let a tag create the Release

**Status:** Accepted (2026-10-10), decided by the maintainer. Reverses the "Not: WinGet (not
submitting)" line of ADR-011.

**Context.** ADR-011 left Windows with a PowerShell one-liner and said WinGet was not worth
submitting. Windows users expect `winget install`, and the maintainer wants it. WinGet needs a
real installer (an `.exe`, `.msi` or MSIX) at a stable URL with a known hash, which `install.ps1`
alone is not. Separately, releases were drafted by hand in the GitHub UI, which the maintainer's
tooling cannot do; it can push a tag.

**Decision.**
- **WinGet package `shankars.askphysics`** (display name "Ask Physics", publisher "Sachin Shankar",
  MIT). The identifier is lowercase to share the publisher folder `manifests/s/shankars/` with the
  maintainer's other package; WinGet matches ids case-insensitively, so `winget install
  ShankarS.AskPhysics` works too. Manifests are schema 1.9.0, templated in `packaging/winget/` and
  rendered by `scripts/winget.sh`. Submitting them to `microsoft/winget-pkgs` stays a manual pull
  request from the maintainer's fork (`docs/RELEASING.md`); no token is stored in this repository.
- **The installer is Inno Setup** (`packaging/windows/askphysics.iss`), per-user
  (`PrivilegesRequired=lowest`), silent-capable, with a fixed `AppId`. It is a thin wrapper: it
  copies `install.ps1` into the app folder and runs it pinned to its own release tag, so the
  install logic stays in one place (ADR-011). Its uninstaller runs `uv tool uninstall askphysics`.
  Models in `~/.cache/askphysics/models` are left alone (user data, shared with other install
  methods, reused on reinstall); neither uv nor its Python is removed. The manifest declares
  `astral-sh.uv` as a dependency though `install.ps1` installs uv anyway.
- **x64 only** in the manifest. Windows on ARM would run the same installer (x64 compatible) but
  `install.ps1` has not been tried there and torch wheels for it are unverified; this is left as
  an open question (Q11).
- **Pushing a tag `vX.Y.Z` creates the GitHub Release** (`.github/workflows/release.yml`): it
  builds and smoke-tests the installer, then attaches it (versioned, plus an unversioned
  `AskPhysicsSetup.exe` for a stable download link) with its SHA-256, the rendered WinGet
  manifests, and the CHANGELOG section as notes (`scripts/release_notes.py`). Pull requests that
  touch the installer build and smoke-test it too.

**Consequences.**
- A release needs the CHANGELOG section written before the tag, and the tag must equal the version
  in `pyproject.toml`; the workflow refuses otherwise.
- The installer is not code-signed; SmartScreen may warn on a direct download. WinGet verifies the
  hash instead.
- Installing takes minutes (it downloads Python, torch and the models), which Microsoft's
  validation pipeline or a reviewer may question. If WinGet review pushes back, the fallback is
  the PowerShell one-liner and the direct download.
- The installer's `AppId` must never change; the manifest's `ProductCode` is derived from it.
- Manifest changes requested in review must be made in the templates as well, or the next version
  loses them.
