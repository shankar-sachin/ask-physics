# Contributing to Ask Physics

Thanks for wanting to help. This project has a few hard rules that exist to
keep the answers honest, so read the golden rules before writing code.

## Golden rules

1. **The LLM never does arithmetic.** Our Fermi models classify, plan, and
   explain.
   Numbers come out of SymPy and Pint, full stop.
2. **Every number has units.** Bare floats do not cross module boundaries.
   Use Pint quantities or the `(value, unit)` fields on the pydantic models.
3. **Every equation has a source and a license.** No source, no merge.
4. **Never silently add data.** Every equation, worked example, constant, or
   Fermi assumption goes through `askphysics validate-data` and code review.

The reasoning behind these is in [`PLAN.md`](PLAN.md) and
[`docs/DECISIONS.md`](docs/DECISIONS.md).

## Development setup

Requires Python 3.11 or newer.

```bash
git clone https://github.com/shankar-sachin/ask-physics.git
cd ask-physics
python -m venv .venv && source .venv/bin/activate
make install          # pip install -e ".[dev]"
make test             # pytest with coverage
make lint             # ruff check + ruff format --check
make typecheck        # mypy
```

No API keys, no GPU, and no trained weights are needed for development.
Tests and the default CLI path use `FakeLLMClient`; model code is tested on
the tiny `fermi-nano` config on CPU. Training the real Fermi models is
covered in [`docs/MODELS.md`](docs/MODELS.md).

## Workflow

1. **Open or pick an issue first** for anything bigger than a typo, so we can
   agree on scope before you sink time into it. Check
   [`docs/ROADMAP.md`](docs/ROADMAP.md) and [`TODO.md`](TODO.md) to see what
   is planned.
2. **Branch from `main`** using a prefix that says what kind of change it is:
   `feat/`, `fix/`, `docs/`, `test/`, `data/`, `build/`, `chore/`, `refactor/`.
3. **Keep commits conceptually separated.** One idea per commit. Commit
   messages use an imperative subject line under 72 characters, for example
   `Add kinematics seed equations`.
4. **Open a pull request** against `main` and fill in the template. CI runs
   ruff, mypy, and pytest on Python 3.11 and 3.12; all of it must be green.
5. **Update `CHANGELOG.md`** under `Unreleased` for anything user-visible.

## Code conventions

- `src/` layout; the package is `askphysics`.
- Full type hints. `mypy --strict` settings apply to `src/`.
- Ruff handles lint and formatting. Do not hand-format.
- Unimplemented functions raise `NotImplementedError` and carry a `TODO`
  comment of 1 to 3 sentences describing the intended implementation.
- New modules get a matching `tests/test_<module>.py`.
- Raise exceptions from `askphysics.errors`, not bare `ValueError`, when the
  error is something the pipeline should handle.

## Contributing data

Data lives in `src/askphysics/data/*.json` and follows
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md). Before a data PR:

- [ ] `askphysics validate-data` passes.
- [ ] Every equation's `sympy_expr` parses and every symbol in it is listed
      under `variables`.
- [ ] Every unit is a valid Pint unit.
- [ ] `source` points to something a reviewer can check, and `license` is
      compatible with MIT redistribution (see
      [`docs/DATA_SOURCING.md`](docs/DATA_SOURCING.md)).
- [ ] Nothing was copied from `evals/questions.yaml`. Eval questions must
      never leak into retrieval data.

## Contributing eval questions

Eval questions go in `evals/questions.yaml`, follow the format in
[`evals/README.md`](evals/README.md), and must not be paraphrases of worked
examples in the data directory.

## Reporting bugs

Open an issue with the exact question you asked, the output you got, what you
expected, and `askphysics version`. Security problems go through
[`SECURITY.md`](SECURITY.md) instead.

## Code of conduct

Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
