# Ask Physics

[![CI](https://github.com/shankar-sachin/ask-physics/actions/workflows/ci.yml/badge.svg)](https://github.com/shankar-sachin/ask-physics/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](CHANGELOG.md)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Status: pre-alpha](https://img.shields.io/badge/status-pre--alpha-red.svg)](docs/ROADMAP.md)

Ask any physics question, from a textbook problem to "how many rubber ducks
would it take to stop a freight train?", and get an answer you can check: the
equations it used (with ids and sources), every assumption spelled out, units
verified end to end, and an honest confidence score. The LLM never does the
math. It reads the question, finds the right equations, and writes the
explanation. A symbolic algebra machine built on SymPy and Pint solves the
equations and tracks the units.

## How it works

```
                 "How fast does a falling object hit the ground if dropped from 20 m?"
                                              |
                                              v
  +-----------+   +-----------+   +--------+   +-----------+   +--------------+   +---------+
  | 1 classify|-->| 2 retrieve|-->| 3 plan |-->| 4 compute |-->| 5 sanity     |-->| 6 explain|
  |   (LLM)   |   | equation  |   | (LLM:  |   | symbolic  |   |   check      |   | (LLM:    |
  | standard/ |   | database  |   | ids +  |   | algebra   |   | units, order |   |  prose   |
  | fermi/    |   | search    |   | values |   | machine:  |   | of magnitude |   |  only)   |
  | nonsense  |   |           |   | only)  |   | SymPy+Pint|   |              |   |          |
  +-----------+   +-----------+   +--------+   +-----------+   +--------------+   +---------+
                                                                                        |
                        19.8057 m/s  [kin_v_squared]  confidence: high (0.85)  <--------+
                        assumptions: released from rest, no air resistance, standard g
```

The planner can only cite equations that retrieval actually found, and it
copies numbers and units out of the question without computing anything. The
machine does the algebra, and Pint rejects any calculation whose units don't
add up. Confidence comes from a documented formula, never from the LLM grading
itself. Full details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Quickstart

Requires Python 3.11 or newer. No API key needed: the default LLM is a
deterministic fake.

```bash
git clone https://github.com/shankar-sachin/ask-physics.git
cd ask-physics
python -m venv .venv && source .venv/bin/activate
make install            # pip install -e ".[dev]"

make test               # pytest with coverage
make lint typecheck     # ruff + mypy --strict

askphysics version
askphysics validate-data
askphysics ask "How fast does a falling object hit the ground if it is dropped from 20 m?"
askphysics ask --json "A rock is dropped from 45 meters. How fast does it land?"
askphysics ask "How much does the color blue weigh?"     # refused, with a redirect
```

`FakeLLMClient` only has a canned plan for "dropped from a height" questions,
on purpose. Ask it anything else and the pipeline tells you exactly which
stage it couldn't complete, instead of making something up. Real LLM planning
arrives in v0.3.

## Project status: v0.1.0 skeleton

| Area | Status |
|------|--------|
| Six-stage pipeline with graceful degradation | Works |
| Symbolic algebra machine: single-equation solve with units (SymPy + Pint) | Works |
| Dimensional consistency and order-of-magnitude sanity checks | Works |
| Keyword retrieval over the equation database | Works |
| Seed data with full validation: 12 equations, 4 examples, 8 constants, 8 Fermi assumptions | Works |
| CLI: `ask`, `version`, `validate-data` | Works |
| Confidence scoring (crude, documented formula) | Works |
| Eval set (8 questions) with a validating loader | Works |
| Real LLM (`AnthropicClient`) | Stub until v0.3 |
| Multi-equation chaining | Stub until v0.3 |
| Vector and hybrid retrieval | Stub until v0.2 |
| Eval scoring and runner | Stub until v0.4 |
| Fermi range propagation | Stub until v0.5 |
| Limit-case checks | Stub until v0.8 |

Every stub raises `NotImplementedError` with a TODO describing the intended
implementation.

## Roadmap

v0.2 real retrieval, v0.3 real solver, v0.4 eval harness, v0.5 Fermi engine,
v0.6 data expansion, v0.7 fine-tuning, v0.8 self-verification, v0.9
hardening, v1.0 release. After v1.0 comes a website. Until then the CLI is
the interface. Details, deliverables, and exit criteria are in
[`docs/ROADMAP.md`](docs/ROADMAP.md); the near-term backlog is
[`TODO.md`](TODO.md).

## Documentation

- [`PLAN.md`](PLAN.md): vision, pipeline, Fermi and refusal policies, confidence model
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): modules, interfaces, one question traced through every stage
- [`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md) and [`docs/DATA_SOURCING.md`](docs/DATA_SOURCING.md): what data looks like and where it comes from
- [`docs/EVALS.md`](docs/EVALS.md): how answers are graded
- [`docs/DECISIONS.md`](docs/DECISIONS.md), [`docs/RISKS.md`](docs/RISKS.md), [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version: the LLM
never does arithmetic, every number has units, every equation has a source
and a license, and data never enters the database without passing
`askphysics validate-data`. Security reports go through
[`SECURITY.md`](SECURITY.md). Participation is covered by the
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

## Limitations

Being straight about what this is right now:

- **It answers almost nothing yet.** With the fake LLM, only "dropped from a
  height" questions get a number. That is the point of a skeleton.
- **The equation database is tiny.** Twelve equations cover a slice of intro
  mechanics, E&M, and thermodynamics. Anything else hits a retrieval miss.
- **Keyword retrieval is brittle.** Phrasing that shares no words with an
  equation's tags won't find it. Vector search arrives in v0.2.
- **Single equations only.** Problems that chain two or more equations
  degrade until v0.3.
- **Scalars only.** No vectors, no Celsius, no numerical-only solutions yet
  (see [`docs/OPEN_QUESTIONS.md`](docs/OPEN_QUESTIONS.md)).
- **The confidence weights are invented.** They get fit to eval data in v0.8.
- **Not for research-level physics**, now or at v1.0.

## License

[MIT](LICENSE). Data entries carry their own `source` and `license` fields;
see [`docs/DATA_SOURCING.md`](docs/DATA_SOURCING.md).
