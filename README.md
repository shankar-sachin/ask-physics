# Ask Physics

[![CI](https://github.com/shankar-sachin/ask-physics/actions/workflows/ci.yml/badge.svg)](https://github.com/shankar-sachin/ask-physics/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](CHANGELOG.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)](docs/ROADMAP.md)

Ask any physics question, from a textbook problem to "how many rubber ducks
would it take to stop a freight train?", and get an answer that is grounded in
retrieved equations, computed by SymPy and Pint rather than guessed by a
language model, unit-checked, and honest about its assumptions.

## Status

**v0.1.0 skeleton, under construction.** The planning documents, project
configuration, code skeleton, and tests land in a series of pull requests.
This README gets its quickstart and full status table in the last one.

## How it works (the short version)

```
question -> classify -> retrieve -> plan -> compute -> sanity_check -> explain -> answer
             (LLM)      (search)    (LLM)   (SymPy +    (units, order    (LLM)
                                             Pint)       of magnitude)
```

The LLM handles language: classifying the question, writing a plan, and
explaining the result. The math is done by symbolic and units libraries, so a
wrong unit is a hard error instead of a confident typo.

## Project docs

- [`PLAN.md`](PLAN.md): vision, pipeline, confidence model, milestones
- [`docs/`](docs/): architecture, roadmap, data schema, evals, risks, decisions

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Security reports go through
[`SECURITY.md`](SECURITY.md).

## License

[MIT](LICENSE)
