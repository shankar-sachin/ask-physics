# Architecture

## Diagram

```
                         +-------------------------------+
   "How fast does a      |            CLI (Typer)        |
    falling object  ---> |  askphysics ask / validate    |
    hit the ground       +---------------+---------------+
    if dropped from 20 m?"               |
                                         v
+-------------------------------------------------------------------------+
|                         Pipeline (pipeline.py)                          |
|                                                                         |
|  1 classify --> 2 retrieve --> 3 plan --> 4 compute --> 5 sanity --> 6 explain
|      |              |             |           |            |            |
|      v              v             v           v            v            v
|  +--------+   +-----------+   +--------+  +---------+  +---------+  +--------+
|  |LLMClient|  | Retriever |   |LLMClient| | Solver  |  | units + |  |LLMClient|
|  |(classify)| | keyword / |   | (plan, |  | SymPy + |  | ranges  |  |(prose  |
|  +--------+   | vector    |   | strict |  | Pint    |  +---------+  | only)  |
|               +-----+-----+   | JSON)  |  +----+----+               +--------+
|                     |         +--------+       |                          |
+---------------------|---------------------------|--------------------------+
                      v                           v                          v
          +-----------------------+    +---------------------+     +----------------+
          | DataStore (loader.py) |    | constants.json      |     | Answer         |
          | equations.json        |--->| fermi_assumptions   |     | value + unit   |
          | examples.json         |    | .json               |     | equations used |
          +-----------------------+    +---------------------+     | assumptions    |
                                                                   | confidence     |
                                                                   | caveats        |
                                                                   +----------------+
```

Arrows point in the direction data flows. The LLM (our own Fermi models,
[`MODELS.md`](MODELS.md)) touches three stages and never produces the
numbers that end up in `Answer.final_value`. From v0.3, `LLMClient` is
implemented by `FermiClient`: tellus classifies, solem plans and explains,
and celeste gets one escalation shot (ADR-010).

## Module responsibilities

| Module | Responsibility | Depends on |
|--------|----------------|------------|
| `cli.py` | Parse commands, build a `Pipeline`, render `Answer` with Rich | `pipeline`, `config`, `data.loader` |
| `config.py` | `Settings` dataclass with env var overrides | stdlib |
| `pipeline.py` | The six stage functions, `Pipeline.run`, confidence scoring, degradation | everything below |
| `models.py` | Pydantic models for every payload; structural validation only | pydantic |
| `errors.py` | Exception taxonomy rooted at `AskPhysicsError` | stdlib |
| `llm/base.py` | `LLMClient` protocol | `models` |
| `llm/fake.py` | `FakeLLMClient`: deterministic canned responses for tests and demos | `models` |
| `llm/routing.py` | Which installed model handles each stage (ADR-010), without loading weights | stdlib |
| `llm/fermi_client.py` | `FermiClient`: one Fermi model behind `LLMClient`, using the constrained decoders; `build_roster` makes the per-stage clients | `lm`, torch |
| `lm/` (v0.2) | The Fermi models: config, tokenizer, transformer, constrained decoding, data factory, training | torch, safetensors |
| `normalize.py` (v0.3) | Rewrites real-world quantity spellings (commas, powers of ten, superscripts, middle dots, "per", µ, Ω) into canonical forms when a question enters the pipeline | Pint (unit check) |
| `retrieval/base.py` | `Retriever` and `VectorStore` protocols | `models` |
| `retrieval/keyword.py` | `KeywordRetriever`: in-memory tag and token scorer | `models` |
| `retrieval/vector.py` | `VectorRetriever` (stub until v0.6) | `retrieval/base` |
| `solver/units.py` | One shared Pint registry; parse, convert, dimension checks | pint |
| `solver/symbolic.py` | Safe equation parsing; `solve_for` | sympy, `solver/units` |
| `solver/fermi.py` | `AssumptionTable`; range propagation (stub) | `models` |
| `data/loader.py` | Load JSON, validate through pydantic plus cross-checks, return `DataStore` | `models`, `solver` |

Dependency direction is strictly downward: `models` and `errors` import
nothing from the package, solvers never import the LLM layer, and only
`pipeline` and `cli` know about everything.

## One question through all six stages

Question: *"How fast does a falling object hit the ground if it is dropped
from 20 m?"* These payloads match the pydantic models and are the real v0.1 output with
`FakeLLMClient` (`askphysics ask --json` shows the final one).

### 1. classify

```json
{
  "category": "standard",
  "reasoning": "Free-fall kinematics with a given height; all inputs are stated or are standard constants.",
  "domains": ["kinematics"],
  "closest_answerable": null
}
```

### 2. retrieve

```json
{
  "query": "How fast does a falling object hit the ground if it is dropped from 20 m?",
  "equations": [
    {"equation_id": "kin_v_squared", "score": 0.77},
    {"equation_id": "kin_x_at", "score": 0.43},
    {"equation_id": "gravitational_pe", "score": 0.17}
  ],
  "examples": [
    {"example_id": "ex_energy_001", "score": 0.06}
  ]
}
```

(The real `RetrievalResult` embeds the full `Equation` objects; ids are shown
here for brevity.)

### 3. plan

```json
{
  "target": "v",
  "unknowns": ["v"],
  "known_values": [
    {"symbol": "v0", "value": 0.0, "unit": "m/s", "origin": "assumption"},
    {"symbol": "a", "value": 9.80665, "unit": "m/s^2", "origin": "constant"},
    {"symbol": "d", "value": 20.0, "unit": "m", "origin": "given"}
  ],
  "equation_ids": ["kin_v_squared"],
  "assumptions": [
    "Released from rest (\"dropped\")",
    "Air resistance is negligible",
    "Gravitational acceleration is constant at standard g"
  ],
  "strategy": "Use v^2 = v0^2 + 2*a*d with a = g and solve for the positive root of v."
}
```

### 4. compute

```json
{
  "target": "v",
  "value": 19.8057,
  "unit": "meter / second",
  "symbolic_solution": "sqrt(2*a*d + v0**2)",
  "substitutions": {"v0": "0.0 meter / second", "a": "9.80665 meter / second ** 2", "d": "20.0 meter"},
  "notes": ["Discarded negative root -19.8057 m/s"]
}
```

### 5. sanity_check

```json
{
  "dimensions_ok": true,
  "magnitude_ok": true,
  "limit_cases_checked": false,
  "issues": []
}
```

### 6. explain

```json
{
  "question": "How fast does a falling object hit the ground if it is dropped from 20 m?",
  "status": "answered",
  "category": "standard",
  "final_value": 19.8057,
  "unit": "meter / second",
  "value_range": null,
  "equations_used": [{"id": "kin_v_squared", "name": "Velocity-displacement relation (constant acceleration)"}],
  "assumptions": ["Released from rest (\"dropped\")", "Air resistance is negligible", "Gravitational acceleration is constant at standard g"],
  "confidence": {"label": "high", "score": 0.85},
  "caveats": ["Discarded negative root -19.8057 m/s", "Limit-case checks are not implemented yet (v0.8)."],
  "explanation": "Using [kin_v_squared], the computed result is 19.8057 meter / second. This assumes: ..."
}
```

Confidence: `r = 0.77` (keyword score 0.67 plus the 0.1 boost for the
classifier's `kinematics` hint), `d = 1`, `s = 1`, `n = 3` gives `a = 0.571`
and `0.35*0.77 + 0.30 + 0.20 + 0.15*0.571 = 0.85`, which is `high`.

## Interfaces and why each is swappable

All four are `typing.Protocol`s, so implementations do not inherit from
anything and tests can pass any object with the right methods.

### `LLMClient` (`llm/base.py`)

```python
class LLMClient(Protocol):
    def complete_json(self, *, system: str, user: str, schema: type[T]) -> T: ...
    def complete_text(self, *, system: str, user: str) -> str: ...
```

`complete_json` returns a validated pydantic instance. `FermiClient`
guarantees that with constrained decoding against the schema, then
validates with pydantic anyway. A `Roster` names the client for each stage:
one classifier, one client per plan attempt, and an explainer. `Pipeline.run`
tries the plan clients in order until Noether accepts a plan (it computes
with the right dimensions), and records each stage's model on
`Answer.models` and the number of plans tried on `Answer.plan_attempts`. **Swappable because** the fake client must
be drop-in for fast tests, and the router swaps tellus, solem, and celeste
behind the same two methods.

### `Retriever` (`retrieval/base.py`)

```python
class Retriever(Protocol):
    def search(self, query: str, k: int, *, domains: Sequence[str] | None = None) -> RetrievalResult: ...
```

**Swappable because** v0.1 keyword, v0.6 hybrid, and any reranked variant
must be comparable on the same eval set by changing one constructor argument.

### `VectorStore` (`retrieval/base.py`)

```python
class VectorStore(Protocol):
    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]], metadata: Sequence[Mapping[str, str]]) -> None: ...
    def query(self, vector: Sequence[float], k: int) -> list[tuple[str, float]]: ...
```

**Swappable because** the sqlite-vec choice in v0.6 is a bet on scale staying
small. If that bet is wrong, FAISS or a hosted store plugs in without
touching the retriever logic.

### Solver (`solver/symbolic.py`, `solver/units.py`)

```python
def solve_for(equation: Equation, unknown: str, knowns: Mapping[str, Quantity]) -> SolveOutcome: ...
```

The solver is a set of functions rather than a class, but the contract is
fixed: equations in, Pint quantities in, a Pint quantity out. **Swappable
because** some problems need numeric root finding (`scipy.optimize`) or ODE
integration instead of closed-form SymPy, and those should slot in behind the
same signature (see `docs/OPEN_QUESTIONS.md`).

## Security boundary: what the LLM may emit

The planner emits **equation ids and numbers with units**, never expressions.
From v0.3 this is enforced during decoding: the Fermi models physically
cannot emit an unretrieved equation id, a number absent from the input, or
an invalid unit.
Expressions only come from `equations.json`, which is reviewed and validated.
That keeps untrusted text away from SymPy's parser, which uses `eval`
internally. Seed expressions are additionally parsed with a restricted
namespace and a character whitelist (`solver/symbolic.py`). Units from the
LLM go through Pint's unit parser, which does not evaluate code.

## Where non-determinism lives and how we tame it

| Source | Where | Mitigation |
|--------|-------|------------|
| Model decoding | classify, plan, explain | Classify and plan decode greedily, so the same input always gives the same output (ADR-009). Only explain samples, at `Settings.temperature`, and its digits are constrained. |
| Model retraining | all model stages | Weights versioned by name (`fermi-solem-1`) and checksum; task format version recorded in traces; evals re-run on every retrain. |
| Training runs | `lm/train.py` | Seeded initialization and data order. MPS kernels are not bit-exact across runs, so retrains are compared by eval scores, not weights. |
| Retrieval ties | `KeywordRetriever` | Ties broken by equation id, so ordering is stable. |
| Vector search (v0.6) | ANN index | Exact search at our scale; if approximate, a fixed index build seed. |
| SymPy solution ordering | `solve_for` | Roots sorted by value after filtering; the choice rule is documented and noted in caveats. |
| Monte Carlo (v0.7) | `solver/fermi.py` | Fixed seed per question (hash of question text plus format version), sample count in `Settings`. |
| Floating point | compute | Values are reported to 6 significant figures; comparisons in evals use relative tolerance, never equality. |

Because plans are greedy, a given model and input always produce the same
plan, and everything after `plan` is deterministic given a plan. The whole
pipeline is reproducible on a fixed set of weights.
