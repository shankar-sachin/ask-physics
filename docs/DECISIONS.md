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
  v = 0 answers a question nobody asked. A zero from stated values stands. If every attempt is
  rejected, the first plan that computed at all is kept (its sanity report
  lowers confidence); with none, the answer degrades.
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

- The only assumed filler is 0 ("from rest"), and only for a variable whose
  `typical_range` includes 0: a speed can start at rest, g and a mass can't
  be zero (the fifth eval wrote g = 0). If no value is legal for a slot, the
  decoder falls back to the loose rules and Noether degrades the answer.
  The data factory never assumes anything else. And a variable a table constant can fill (g) is
  only the target when no open variable lacks such a fallback. Added after
  the fourth eval, where "lifting it 11 m took 11000 J, what is its mass?"
  was solved for g with an invented m = 1 kg.

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
of later batches; the tokenizer can learn from it too. Uses (1) and (2),
the real-question eval and the exercises as classify examples, come next.

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
  after a switch is used, so later corrections are not in the corpus. 25 books
  across 13 repositories: physics, chemistry, biology, astronomy, anatomy,
  microbiology, algebra through precalculus, statistics, and psychology,
  about 2.8M words after removing repeats.
- **Public-domain books for the rest.** Project Gutenberg texts in English,
  physics, astronomy, chemistry, mathematics, and general science first, then
  English and American literature, until the corpus reaches 20M words. A book
  qualifies only if every person the catalog credits (authors, translators,
  editors) died in or before 1955, which makes it public domain in the US and in
  life-plus-70 countries. Gutenberg's header, footer, and trademark are removed.
- **Not committed.** The corpus is about 130 MB, so `scripts/build_corpus.py`
  builds it locally under `build/`. The repository holds the manifest, the
  extractors, and their tests; each build writes `ATTRIBUTION.md`, the CC BY
  license text, and a lock file listing exact commits, Gutenberg ids, and file
  hashes.
- **Cleaning.** Text only, as in ADR-016; reference-list entries and repeated
  paragraphs removed; paragraphs containing slurs dropped (old literature has
  them), with the list stored as hashes.
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

