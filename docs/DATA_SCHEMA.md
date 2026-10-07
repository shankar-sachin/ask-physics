# Data Schema

All data lives in `src/askphysics/data/*.json`. Each file is a JSON array of
objects. The pydantic models in `src/askphysics/models.py` are the
executable version of this spec. If the two disagree, the code wins and this
doc gets fixed.

Validation runs in two layers:

1. **Structural** (pydantic, per entry): types, required fields, value
   ranges.
2. **Semantic** (`data/loader.py`, across entries): SymPy parse, Pint unit
   validity, symbol coverage, referential integrity, uniqueness, dimensional
   consistency.

`askphysics validate-data` runs both and exits non-zero on any failure.

---

## Shared conventions

- **ids** are lowercase snake case, `^[a-z][a-z0-9_]*$`, and unique within a
  file.
- **Units** are Pint-parseable strings (`"m/s^2"`, `"J/(mol*K)"`, `"kg"`).
  Dimensionless quantities use `"dimensionless"`. Offset units such as
  `degC` are banned for now; use `K` (see `docs/OPEN_QUESTIONS.md`).
- **Symbols** in expressions are valid Python identifiers. Greek letters are
  spelled out (`theta`, `lam`; `lambda` is a Python keyword).
- **Licenses** are SPDX identifiers (`MIT`, `CC-BY-4.0`, `CC-BY-SA-4.0`,
  `CC0-1.0`) or `public-domain` for facts and government works. The allowed
  set is in `docs/DATA_SOURCING.md`.

---

## Equation

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | str | yes | Unique, snake case |
| `name` | str | yes | Human-readable |
| `latex` | str | yes | Display form only, never parsed |
| `sympy_expr` | str | yes | Exactly one `=`; both sides parse with the restricted parser |
| `variables` | list[Variable] | yes | One entry per free symbol in `sympy_expr` |
| `domain` | str | yes | One of `kinematics`, `dynamics`, `energy`, `momentum`, `gravitation`, `electromagnetism`, `thermodynamics`, `fluids`, `waves`, `optics`, `modern` |
| `assumptions` | list[str] | yes | Physical idealizations baked into the formula |
| `validity_conditions` | list[str] | yes | When the formula stops applying |
| `tags` | list[str] | yes | Retrieval keywords; at least one |
| `source` | str | yes | Where a reviewer can verify it |
| `license` | str | yes | SPDX id or `public-domain` |
| `confidence_in_entry` | float | yes | 0 to 1; reviewer's confidence the entry is correct and complete |

### Variable

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `symbol` | str | yes | As it appears in `sympy_expr` |
| `name` | str | yes | Human-readable |
| `unit` | str | yes | Pint unit the symbol is measured in |
| `description` | str | yes | One line |
| `typical_range` | [float, float] or null | no | Plausible magnitude window used by the sanity check; `low <= high` |

### Example

```json
{
  "id": "kin_v_squared",
  "name": "Velocity-displacement relation (constant acceleration)",
  "latex": "v^2 = v_0^2 + 2 a d",
  "sympy_expr": "v**2 = v0**2 + 2*a*d",
  "variables": [
    {"symbol": "v", "name": "final velocity", "unit": "m/s", "description": "Velocity after covering displacement d", "typical_range": [0.0, 1000.0]},
    {"symbol": "v0", "name": "initial velocity", "unit": "m/s", "description": "Velocity at the start", "typical_range": [0.0, 1000.0]},
    {"symbol": "a", "name": "acceleration", "unit": "m/s^2", "description": "Constant acceleration along the motion", "typical_range": [0.0, 100.0]},
    {"symbol": "d", "name": "displacement", "unit": "m", "description": "Displacement along the line of motion", "typical_range": [0.0, 100000.0]}
  ],
  "domain": "kinematics",
  "assumptions": ["Acceleration is constant", "Motion is along a straight line"],
  "validity_conditions": ["Speeds far below c", "No drag, or drag negligible over d"],
  "tags": ["kinematics", "free fall", "falling", "drop", "speed", "velocity", "height", "constant acceleration"],
  "source": "OpenStax University Physics Volume 1, Chapter 3 (reference; entry written by the project)",
  "license": "MIT",
  "confidence_in_entry": 0.99
}
```

### Validation rules

- `sympy_expr` contains exactly one `=` and only characters from
  `[A-Za-z0-9_+\-*/^(). =]`.
- Both sides parse with the restricted SymPy parser (no builtins, a
  whitelisted function namespace).
- The set of free symbols in the expression equals the set of
  `variables[].symbol`. Missing and extra variables are both errors.
- Every `variables[].unit` parses in the shared Pint registry.
- Dimensional consistency: substituting `1 * unit` for each symbol, both
  sides evaluate without a `DimensionalityError` and have the same
  dimensionality.
- `confidence_in_entry` is in [0, 1].
- `license` is in the allowed set.

---

## WorkedExample

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `id` | str | yes | Unique |
| `problem_text` | str | yes | The problem as a student would read it |
| `known_values` | list[KnownValue] | yes | Values given in or implied by the problem |
| `unknowns` | list[str] | yes | Symbols to find: the answer first, then any values found on the way |
| `equations_used` | list[str] | yes | Equation ids; each must exist |
| `solution_steps` | list[str] | yes | Ordered, human-readable |
| `final_answer` | QuantityValue | yes | `{value, unit}` |
| `difficulty` | str | yes | `intro`, `intermediate`, or `advanced` |
| `tags` | list[str] | yes | |
| `source` | str | yes | |
| `license` | str | yes | |

`KnownValue` is `{symbol, value, unit, origin}` where `origin` is `given`,
`constant`, or `assumption`. `QuantityValue` is `{value, unit}`.

### Example

```json
{
  "id": "ex_kin_001",
  "problem_text": "A car starts from rest and accelerates uniformly at 3 m/s^2 for 5 s. What is its final speed?",
  "known_values": [
    {"symbol": "v0", "value": 0.0, "unit": "m/s", "origin": "given"},
    {"symbol": "a", "value": 3.0, "unit": "m/s^2", "origin": "given"},
    {"symbol": "t", "value": 5.0, "unit": "s", "origin": "given"}
  ],
  "unknowns": ["v"],
  "equations_used": ["kin_v_at"],
  "solution_steps": ["Start from v = v0 + a*t.", "Substitute v0 = 0, a = 3 m/s^2, t = 5 s.", "v = 15 m/s."],
  "final_answer": {"value": 15.0, "unit": "m/s"},
  "difficulty": "intro",
  "tags": ["kinematics", "acceleration", "car"],
  "source": "Project-authored",
  "license": "MIT"
}
```

### Validation rules

- Every id in `equations_used` exists in `equations.json`.
- Every `unknowns` entry and every `known_values[].symbol` appears in at
  least one referenced equation.
- All units parse; `final_answer.unit` has the same dimensionality as the
  unknown's declared unit in the equation that defines it.
- Every example re-solves: Noether solves `equations_used` for the first
  unknown (chaining them when there are several) and lands within 0.1% of
  `final_answer` (`tests/test_data_loader.py`, a v0.4 exit criterion).
- No example reads like an eval question (same word-overlap check the data
  factory uses).
- (v0.4) Re-solving with `solve_for` reproduces `final_answer` within 0.1%.
- No `problem_text` may be a near-duplicate of an eval question (see
  `docs/EVALS.md`, leakage policy).

---

## Constant

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | str | yes | Snake case, unique (`standard_gravity`) |
| `symbol` | str | yes | Conventional symbol (`g`); not required to be unique |
| `value` | float | yes | |
| `unit` | str | yes | Pint unit |
| `uncertainty` | float | yes | Standard uncertainty in `unit`; `0.0` for exact (defined) values |
| `source` | str | yes | CODATA release, SI definition, or standard |

### Example

```json
{
  "name": "gravitational_constant",
  "symbol": "G",
  "value": 6.6743e-11,
  "unit": "m^3/(kg*s^2)",
  "uncertainty": 1.5e-15,
  "source": "CODATA 2022 recommended values (NIST)"
}
```

### Validation rules

- `name` unique; `unit` parses; `uncertainty >= 0`.
- `uncertainty < abs(value)`.

---

## FermiAssumption

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `quantity` | str | yes | Snake case, unique (`rubber_duck_mass`) |
| `default_value` | float | yes | Best single guess |
| `unit` | str | yes | Pint unit |
| `low` | float | yes | Plausible lower bound, not an absolute minimum |
| `high` | float | yes | Plausible upper bound |
| `rationale` | str | yes | Why these numbers; one or two sentences |
| `source` | str | yes | Reference, or `"estimate"` with reasoning in `rationale` |

### Example

```json
{
  "quantity": "freight_train_mass",
  "default_value": 6000000.0,
  "unit": "kg",
  "low": 2000000.0,
  "high": 20000000.0,
  "rationale": "A loaded North American freight train of about 100 cars at 60 to 130 t each. Heavy-haul ore trains exceed the high bound and are out of scope for the default.",
  "source": "estimate"
}
```

### Validation rules

- `quantity` unique; `unit` parses.
- `0 < low <= default_value <= high` (Fermi quantities are positive
  magnitudes; signed quantities belong in equations).
- `high / low <= 1e4`. A wider range means the assumption is not doing any
  work and should be split or rethought.
