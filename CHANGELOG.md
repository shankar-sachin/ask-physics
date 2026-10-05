# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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
