# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Fermi model code in `src/askphysics/lm/`: configs, byte-level BPE
  tokenizer, transformer with KV cache, safetensors checkpoints, canonical
  task formats, constrained decoding, the data factory, and the training loop.
- `askphysics model build-data`, `train-tokenizer`, `train`, and `info`.
- A redesigned CLI: answer cards with one-line math, the inputs behind every
  number, a confidence meter, a "try instead" redirect for refusals, and a
  live training progress bar. `Answer` gains `inputs` and `redirect`.
- README screenshots, regenerated with `make screenshots`.

### Fixed

- The solver raised a raw `ZeroDivisionError` on a zero in a division; it is
  now a `SolverError`, so answers degrade instead of crashing.

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

## [0.1.0] - Unreleased

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
