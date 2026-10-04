# CLAUDE.md

Guidance for Claude Code sessions working on this repo.

## Project

Ask Physics answers physics questions, from textbook to absurd, through a
six-stage pipeline: classify, retrieve, plan, compute, sanity_check, explain.
The LLM handles language; SymPy and Pint do all math and units. v0.1.0 is a
skeleton: many functions are stubs that raise `NotImplementedError`.

Read [`PLAN.md`](PLAN.md) first, then the relevant file in [`docs/`](docs/):
`ARCHITECTURE.md` (modules, interfaces, payloads), `DATA_SCHEMA.md`,
`DECISIONS.md` (ADRs), `OPEN_QUESTIONS.md`, `ROADMAP.md`. The backlog is in
`TODO.md`.

## Commands

```bash
make install     # pip install -e ".[dev]"
make test        # pytest with coverage
make lint        # ruff check + ruff format --check
make format      # ruff format + ruff check --fix
make typecheck   # mypy (strict on src/ and evals/)
make ask Q="How fast does a falling object hit the ground if dropped from 20 m?"
askphysics validate-data
```

All four of lint, typecheck, test, and validate-data must pass before a
commit.

## Golden rules

1. **The LLM never does arithmetic.** It classifies, plans (ids and numbers
   copied from the question or tables), and writes prose. Every number in an
   `Answer` comes from SymPy and Pint.
2. **Every number has units.** Use Pint quantities from
   `askphysics.solver.units` (one shared registry). No bare floats across
   module boundaries.
3. **Every equation has a source and a license.** No exceptions.
4. **Never silently add data to the DB.** Data changes go through
   `askphysics validate-data`, are listed in the PR, and are never made as a
   side effect of another task.
5. **The LLM never emits expressions.** Plans reference equation ids;
   expressions come only from reviewed JSON. Never call `sympify` on
   untrusted text.
6. **Tests never need an API key.** Use `FakeLLMClient`.

## Conventions

- Python 3.11+, `src/` layout, package `askphysics`.
- Full type hints; mypy strict. Pydantic v2 models in `models.py`.
- Ruff for lint and formatting, line length 100.
- Unimplemented code raises `NotImplementedError` with a `# TODO:` comment of
  1 to 3 sentences describing the intended implementation.
- Raise exceptions from `askphysics.errors`. Pipeline stages convert failures
  into degraded answers instead of crashing.
- One test file per module: `tests/test_<area>_<module>.py`.
- Ambiguity goes in `docs/OPEN_QUESTIONS.md`; decisions go in
  `docs/DECISIONS.md` as a new ADR. Don't guess silently.
- Don't add features outside `docs/ROADMAP.md` for the current milestone.
- Update `CHANGELOG.md` under `Unreleased` for user-visible changes.

## Workflow

Branch from `main` with a prefix (`feat/`, `fix/`, `docs/`, `test/`,
`data/`, `build/`, `chore/`), keep commits conceptually separated, and open
a PR using the template. CI runs ruff, mypy, and pytest on 3.11 and 3.12.
PRs may be merged once CI is green. Never tag or publish a release without
asking the maintainer first.
