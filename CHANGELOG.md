# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `askphysics model pull` downloads the published Fermi models (tellus and
  solem; `--all` adds celeste), checking every file's size and sha256
  against a manifest the package pins, and never leaving a half-installed
  model. The installers run it last. celeste downloads by itself the first
  time a question needs it (`ASKPHYSICS_AUTO_PULL=0` turns that off).
- `askphysics model package` (maintainers) turns a trained model into
  release assets: bf16 weights, a model card with its training curve, eval
  results, answering speed, and credit for the prose it learned from, all
  pinned in the manifest.
- The Ask Physics Wiki on GitHub: guides for using it, reading an answer, and
  asking good questions, an FAQ, the Fermi models and Noether explained, and a
  page for every equation (formula, variables and units, assumptions,
  look-alikes, source, license) generated from the database. Pages live in
  `docs/wiki/`, `scripts/build_wiki.py` builds them, and a workflow publishes
  them on every push to main.
- Workflow scripts for the jobs that used to be long command chains:
  `setup.sh`, `update.sh`, `check.sh` (with a merge-conflict check),
  `corpus.sh`, `train.sh` (back up, train, score old and new, compare),
  `eval.sh`, `models.sh` (backup and restore), and `release_weights.sh`, plus
  `compare_evals.py`. Each has `--help` and a `DRY_RUN=1` mode, and CI runs
  shellcheck on all of them. `make train MODEL=...` and friends call them.
- A 20-million-word prose corpus for the Fermi models (ADR-017):
  `scripts/build_corpus.py` builds it locally from 52 OpenStax textbooks
  under CC BY 4.0 (sciences, math, history, economics, social sciences,
  writing, and more), read at commits from before OpenStax relicensed them,
  and public-domain books from Project Gutenberg, explanatory non-fiction
  ahead of fiction. Every pinned commit's license
  is checked at build time, repeated paragraphs, citations, slurs, and swear words are
  removed, and each build writes its attribution and a lock file.
- Multi-equation answers: a plan can chain equations ("a 3 kg cart is pushed
  with 12 N for 4 s: how far does it go?" finds the acceleration first), and
  the answer shows each value found along the way. The training data now
  includes two-step problems.
- Celsius and Fahrenheit: "20 °C", "68 degrees Fahrenheit", and "100 degrees
  centigrade" all work, converted as a temperature or as a change in one by
  what the equation means, and a temperature asked in Celsius is answered in
  Celsius too.

### Fixed

- `model train` refuses `--prose-steps` as large as `--steps`. Prose-only steps
  come first, so such a run reads prose all night and never trains on the
  tasks; `scripts/train.sh` defaulted to exactly that for solem and celeste
  (3000 and 3000) and now runs 5000 steps, 2000 of them prose.
- Every answer v0.3's eval caught confidently wrong is fixed by rules, for
  any model: a value labelled by a variable's name ("primary turns: 61000",
  "the speed at the end comes out to 160 mph") is locked to that variable;
  "how fast", "how far", "what height", and "how long does it take" pick the
  kind of quantity asked for; and an answer of exactly 0 that only an
  assumed 0 produced is rejected and retried instead of shown.
- A refusal's slot is filled only by the words in that slot's position of a
  matching template, so tellus can't write "a dropped ball take does not
  move", and every question a refusal suggests is trained as answerable.
- Spelled-out numbers before a unit are read as digits ("an eight kilogram
  ball" is "an 8 kilogram ball"), and the training data has arithmetic with
  digits too, so number words no longer mean "math question".
- Every confidently wrong answer in solem's 3,000-question eval is fixed.
  When a question doesn't say whether resistors are in series or parallel
  (or whether a speed is orbital or escape), the answer says so and its confidence is
  low; "in parallel" picks the parallel formula. A stated value the plan
  doesn't use ("the emissivity comes out to 0.017") is retried, and flagged
  if no plan uses it. Unlabelled pairs of values go in the order stated.
  "What speed did it start at" asks for the initial speed, and "after
  starting from rest" no longer does.
- An impossible result (a negative resistance, a 0 N braking force from an
  assumed 0) is never shown as the answer; the answer degrades and says why.
  "Stop" and "braking" questions now find the impulse equation.
- The last two confidently wrong answers in solem's 3,000-question eval:
  "how strongly do they attract?" asks for a force, not an energy, and a
  question that never mentions an emissivity no longer gets one borrowed
  from another number (and a borrowed unitless value is flagged).

## [0.3.0] - 2026-10-06

### Added

- Real prose for the Fermi models: about 118,000 words of OpenStax
  *Physics* (CC BY 4.0, attributed in `third_party/openstax-physics/`),
  extracted by `scripts/extract_openstax.py` (ADR-016). `model train
  --prose ... --prose-steps N --prose-share F` trains on it before and
  alongside the tasks, reporting `val_loss_prose`, and `model
  train-tokenizer --prose ...` learns its words.
- `askphysics model eval` reports what `ask` would answer: the share right
  after the router's retries, wrong but flagged, and confidently wrong
  (wrong with every check passing), which is the number that has to stay
  near zero. `--attempts` sets the retries.
- A result that can't be negative (a mass, a resistance, a frequency, an
  absolute temperature) but is fails the sanity check, caps confidence at
  low, and makes the router try another plan.
- `askphysics ask` answers with the Fermi models once any are installed
  (ADR-010). tellus classifies, solem plans and explains, a rejected plan is
  retried up to 5 times, and celeste gets one more try. Missing models are
  skipped, so tellus alone can do everything. The answer card and
  `Answer.models` say which model handled each stage, and
  `Answer.plan_attempts` counts the plans tried. `--model` forces one model,
  `--llm fake` brings back the fake client, and `ASKPHYSICS_PLAN_ATTEMPTS`,
  `ASKPHYSICS_ESCALATIONS`, and `ASKPHYSICS_DEVICE` tune the router.
- 12 more equations (98 → 110) with dimensionless quantities: kinetic and
  static friction, efficiency, Carnot efficiency, refractive index, the
  ideal transformer, magnification, the ideal gas law by molecule count,
  the Lorentz factor, emissivity, dielectric capacitors, and Faraday's law
  for a coil. Questions write these as bare numbers ("a friction
  coefficient of 0.3"), and the plan decoder now reads a number with no unit
  after it as a dimensionless quantity.
- 43 more equations (55 → 98) in thermodynamics (heat, latent heat,
  expansion, conduction, radiation, the first law, kinetic theory, heat
  engines), electromagnetism (Coulomb's law, fields, potential, electric
  power, resistivity, series and parallel resistors, capacitors, RC circuits,
  magnetic forces, the field of a wire and a solenoid, motional emf,
  inductors), optics (thin lens, lens power, magnification), and modern
  physics (photon energy and momentum, E = mc², de Broglie, photoelectric
  effect, time dilation, length contraction). Three new constants (CODATA
  2022): the Coulomb constant, the magnetic constant, and the
  Stefan-Boltzmann constant.
- The data factory samples a variable from its own typical range when the
  everyday range for its unit doesn't fit (a molecule's mass, a particle's
  charge).
- 43 new equations (12 → 55): average speed, vertical launches, centripetal
  acceleration and force, angular velocity, period and frequency, Hooke's law,
  impulse, torque, rotational dynamics and energy, moment of inertia, work,
  power, spring energy, the work-energy theorem, angular momentum, recoil,
  surface gravity, orbital and escape speed, Kepler's third law, density,
  pressure, hydrostatic pressure, buoyancy, continuity, flow rate, spring and
  pendulum periods, wave speed, string waves, intensity, and the Doppler
  effect. New domains: fluids, waves (plus optics and modern, used by the
  next batch). Every equation is checked to solve for each of its variables.
  Retrain models after this; the factory covers every new equation.
- Questions are normalized before anything reads them, so real-world
  spellings work: "2,000-kg", "3.00 × 10⁸ m/s", "m/s²", "kg·m/s", "55-kg",
  "20 meters per second", "25 °C", "2.5 µC", "220 Ω". Across the OpenStax
  *Physics* text, 12% more quantities are read, and accelerations written
  "m/s²" are no longer read as speeds.
- `askphysics model eval`: decodes held-out questions with the constrained
  decoder and reports category accuracy and the valid plan rate (plans that
  compute the right answer), with sample failures and an `eval.json` report.
- `THIRD_PARTY_LICENSES.md`: every library, font, and image source we use,
  with its license.
- A website at https://askphysics.vercel.app that runs the real `askphysics`
  package in your browser (Pyodide): ask a question, get the full answer card.
  No account, nothing runs on a server (ADR-013).
- `askphysics.web` (question in, display-ready JSON out) and
  `askphysics.pretty` (number, unit, and equation formatting shared by the CLI
  and the site).

### Changed

- The default provider is now `auto`: the Fermi models when installed, the
  fake client otherwise. `ASKPHYSICS_MODEL` now forces one model for every
  stage instead of naming the planner.
- The one-line installers now live at
  `https://askphysics.vercel.app/installers/install.sh` and `install.ps1`.
- The data factory composes questions from parts (44 frames, 12 ways to state
  a value, variable synonyms, spelled-out and converted units, openers,
  sign-offs, casual punctuation) and has 63 worded scenarios, 24 Fermi and 24
  out-of-scope templates, and more varied explanations. 20,000 examples now
  hold about 16,800 distinct question shapes, up from 3,000, so models stop
  memorizing phrasings. Rebuild the dataset and tokenizer before training.
- Every unknown is asked for about equally often: "find v0" went from 1 in
  13 kinematics questions to 1 in 4, with new scenarios for starting speeds,
  roofs, and cliffs. New look-alike questions pair "How much energy is in a
  rumor?" style category errors with real Fermi questions about batteries
  and lightning, so classification has to read the object, not the opening
  words. Rebuild the dataset.
- Training reports validation loss for each task (classify, plan, explain) as
  well as overall, on a fixed random sample of the validation split.

### Fixed

- Refusals are built from reviewed sentences. The classifier picks the
  reason, with its slot filled by words from the question, and a
  "try instead" written for that reason, so tellus can no longer answer
  "how do you find the slope of a curve?" with "a dream is an experience
  ... taste taste taste". Pure math questions have their own refusal ("Not
  a physics question; it is pure math.") in the training data.
- The plan decoder reads what a question states outright. A symbol written
  next to a value ("fs is 758 Hz", "di = 1.8 m") locks that value to it, and
  the words after "what is" or "find" pick the unknown and the equation
  ("What is its mass?" can't be answered with W = Fd). tellus had swapped
  labelled values and solved for the wrong unknown.
- No more word salad from the models. Plans pick assumptions from each
  equation's reviewed list instead of writing their own ("The mass or
  spherically symmetric bodies"), free text can't loop ("roughly roughly
  roughly..."), and a refusal's reason or suggested question is replaced
  or dropped when it is unreadable or about something else.
- `askphysics ask` no longer needs quotes around the question.
- The plan decoder no longer picks an equation that has nowhere to put a
  quantity the question states. tellus answered "a thing at 4.1 m/s carries
  620 J, what is its mass?" with KE = p²/2m and an assumed p = 0.
- A number in e-notation with no unit after it ("6.3e+20,") was split into
  6.3 and the unit "e" (Pint's elementary charge).
- Quantities with negative exponents in their unit ("5 s^-1", "230 m^-1",
  including "s⁻¹" after normalization) weren't read, and the 1 in "m^-1" was
  read as a number.
- A reciprocal result unit was written "1 / meter", whose 1 an explanation
  isn't allowed to copy; it is now "meter ** -1".
- The dimensional check set every variable to 1, so an equation like
  `f = fs*v/(v - vs)` crashed validation with a division by zero.
- More unit strings crashed Pint's parser ("K^0", "m^^2"); any string it
  can't parse is now simply not a unit. A number too large for a float
  ("1e999") no longer crashes number extraction.
- A question containing a dangling unit operator ("5 N/", "3 m*") crashed
  the unit parser instead of ignoring the fragment.
- Our docs said OpenStax *University Physics* is CC BY 4.0; it is CC BY-NC-SA
  4.0. Equation entries only cite it as a reference and copy no text, so
  nothing changes in the data. Real-phrasing data will come from OpenStax
  *Physics* (2020), which is CC BY 4.0 (ADR-016).
- Plans can no longer put a value with the wrong dimensions into a variable
  (a speed as a momentum), change the unit written after a number (570
  pounds into 570 kilograms), solve for something the question already
  gives, use one stated quantity twice, or fill a slot with 0 while a stated
  value goes unused (ADR-015). They also can no longer invent a 1 as a
  filler ("m = 1 kg"), assume 0 for something that can't be zero (g = 0),
  or solve for g when the question asks for a mass and g can come from the
  constants table.
- The data factory sometimes wrote a spelled-out unit one way in the
  question and another in the plan ("115 ohms" but `ohm`), and dropped
  examples that spelled meters as metres. Rebuild the dataset.
- Training on Apple GPUs no longer runs out of memory after a few hundred
  steps.
- Validation loss was measured on the first rows of the first shard only; it
  now uses a shuffled sample, so runs are comparable.
- A fresh training run starts a fresh `metrics.jsonl` instead of appending to
  the last run's log (`--resume` still appends).

## [0.2.0] - 2026-10-04

### Added

- Fermi model code in `src/askphysics/lm/`: configs, byte-level BPE
  tokenizer, transformer with KV cache, safetensors checkpoints, canonical
  task formats, constrained decoding, the data factory, and the training loop.
- `askphysics model build-data`, `train-tokenizer`, `train`, and `info`.
- A redesigned CLI: answer cards with one-line math, the inputs behind every
  number, a confidence meter, a "try instead" redirect for refusals, and a
  live training progress bar. `Answer` gains `inputs` and `redirect`.
- README screenshots, regenerated with `make screenshots`.
- One-line installers: `install.sh` (curl, macOS and Linux) and `install.ps1`
  (irm, Windows), tested on all three platforms in CI, plus a Homebrew tap
  (ADR-011).
- Plan for Ask Physics API keys on the future hosted API (PLAN.md, Q18).
- An Ask Physics logo, a README banner, and artwork for each Fermi model,
  rendered by `make brand`: a ray-traced black hole quasar, a cratered
  crescent Moon from LRO data, the Earth from NASA's Blue Marble, and the Sun.

### Fixed

- The solver raised a raw `ZeroDivisionError` on a zero in a division; it is
  now a `SolverError`, so answers degrade instead of crashing.
- Output piped or redirected on Windows (cp1252) crashed on symbols like ◉;
  unencodable glyphs now print as `?`.

### Changed

- The language side is now the Fermi model family, built and trained from
  scratch in this repo: `fermi-tellus-1` (~3M params), `fermi-solem-1`
  (~30M), `fermi-celeste-1` (~120M). Design in `docs/MODELS.md`; ADR-009
  and ADR-010.
- Roadmap rebuilt around the Fermi models (v0.2 foundations, v0.3 trained
  and wired in); later milestones renumbered.

### Removed

- `AnthropicClient`, the `[anthropic]` extra, and API keys. Ask Physics makes
  no external API calls.

## [0.1.0] - 2026-10-03

The skeleton release: architecture, planning documents, and stubs. No real
LLM calls or vector search yet.

### Added

- Repository foundation: README with badges, MIT license, security policy,
  contributing guide, code of conduct, pull request template.
- Planning documents: `PLAN.md`, `CLAUDE.md`, `TODO.md`, and `docs/`
  (roadmap, architecture, data schema, data sourcing, evals, prompts, risks,
  decisions, open questions, glossary).
- Project configuration: `pyproject.toml`, `Makefile`, CI workflow.
- `askphysics` package skeleton with the six-stage pipeline, pydantic models,
  error taxonomy, `FakeLLMClient`, working keyword retriever, Pint and SymPy
  wrappers, and seed JSON data.
- Typer CLI: `askphysics ask`, `askphysics version`, `askphysics validate-data`.
- Test suite (156 tests, 96% coverage) and an eval question set with a stub
  harness.
- ADR-008: the CLI is the interface through v1.0; a website follows after.
