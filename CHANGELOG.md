# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
- Test suite and an eval question set with a stub harness.
