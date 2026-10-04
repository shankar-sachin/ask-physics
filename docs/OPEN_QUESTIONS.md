# Open Questions

Things we have not decided. Per the working rules, ambiguity gets logged here
instead of being guessed at silently in code. Each question has a
**default for now**, meaning what the code does today, so nothing is
blocked. When a question is resolved, move it to `docs/DECISIONS.md` as an
ADR and delete it here.

---

### Q1. How do we represent vector and tensor quantities in the Equation schema?

`F = m*a` is a vector equation; we store it as scalar magnitudes. Projectile
motion, torque (`tau = r x F`), and the Lorentz force need components or
cross products. Options: component-wise scalar equations (`Fx = m*ax`); SymPy
`Matrix` expressions with a `shape` field on `Variable`;
`sympy.physics.vector`.

**Default for now:** scalar magnitudes only. Equations whose direction
matters say so in `assumptions` ("magnitudes along a single line").

### Q2. How do we handle piecewise, implicit, or numerical-only solutions?

Some problems have no closed form (a pendulum at large amplitude, drag with
v^2, transcendental equations). Should `solve_for` fall back to
`sympy.nsolve` or `scipy.optimize`, and how is the initial guess chosen (from
`typical_range`)? Is SciPy worth the dependency?

**Default for now:** closed-form `sympy.solve` only. No solution raises
`SolverError`, and the answer is degraded.

### Q3. Do we support relativistic or quantum regimes in v1?

Textbook special relativity (time dilation, `E = mc^2`) and intro quantum
(photon energy, de Broglie wavelength) are formula-level and cheap to add.
The risk is users then asking QFT questions and getting confident nonsense.

**Default for now:** not in the seed data. The classifier routes
research-level questions to `out_of_scope`. Leaning towards allowing
formula-level relativity and quantum from v0.6, with strict
`validity_conditions`.

### Q4. How do we represent equation validity regimes machine-readably?

`validity_conditions` is free text ("speeds far below c"). To check it
automatically we would need structured conditions, such as
`{"expr": "v/c", "op": "<", "value": 0.1}`. Who writes those, and how does
the sanity check evaluate them when the condition involves a quantity the
plan doesn't compute?

**Default for now:** free text, shown as caveats when the equation is used.

### Q5. How do we deal with multiple correct approaches?

A falling-object speed can come from kinematics or energy conservation. Do
evals accept either equation set? Should the planner prefer one? Should v0.8
self-verification deliberately use the other route?

**Default for now:** evals list *acceptable* equation-id sets, any of which
earns full equation credit. The planner picks whichever retrieval ranks
higher.

### Q6. What should happen with symbol collisions across equations and constants?

`R` is resistance in Ohm's law and the gas constant in the ideal gas law.
`G`, `g`, `e`, and `E` all collide with something eventually. Today symbols
are scoped per equation and constants are looked up by `name`, not symbol.
Once multi-equation chaining exists (v0.3), how do we tell a shared symbol
(the same `v` across two equations) from a coincidence?

**Default for now:** symbols are per-equation; the plan uses one equation;
constants are referenced by `name` in plans.

### Q7. Which root does the solver pick when there are several?

Solving `v^2 = ...` gives plus and minus roots; a quadratic in `t` can give
two positive times (up and down). We currently drop non-real roots, prefer
non-negative roots, take the smallest of those, and record the choice in
caveats. That rule is wrong for "when does the ball pass 10 m on the way
down?"

**Default for now:** the rule above. Should the plan carry a
`root_preference` field?

### Q8. How should Fermi uncertainty be propagated, and what does "stop" mean?

Monte Carlo with log-uniform sampling between `low` and `high`, or interval
arithmetic? MC handles correlations and non-monotonic functions; intervals
are deterministic and cheap but blow up. Separately: absurd questions have
fuzzy targets ("stop" a train with inelastic collisions never reaches zero
speed). Who defines the operational target, the planner or the assumptions
table?

**Default for now:** propagation is a stub (`solver/fermi.py`). The planner
states its operational definition as an assumption.

### Q9. How do we fit the confidence weights?

The `PLAN.md` formula weights (0.35/0.30/0.20/0.15) are invented. Fit them by
logistic regression on eval outcomes (v0.4 data)? Calibrate so "0.8" means
right 80% of the time? Is a single number even the right output, versus
separate "inputs confidence" and "method confidence"?

**Default for now:** the documented crude formula.

### Q10. Offset units: how do we support Celsius and Fahrenheit?

Pint offset units cannot be multiplied (`degC * J/K` is ambiguous). Users
will write "at 25 degrees C".

**Default for now:** data uses kelvin only. In v0.3 the planner is told to
pass temperatures as given, and the compute stage converts offset units to
kelvin before substitution.

### Q11. How is the CLI distributed?

The interface question is settled: CLI now, website after v1.0 (ADR-008).
Still open: do we publish to PyPI during v0.x or only at v1.0? Do we want a
Homebrew **formula** (for CLI tools) later? A Homebrew **cask** is for GUI
`.app` bundles and does not fit a CLI.

**Default for now:** pip-installable from source; PyPI at v1.0 per the
roadmap. Releases are tagged only with the maintainer's approval.

### Q12. The repo description says "a simple Python LLM that runs on your device". Is local inference a goal?

The original one-line README promises on-device inference. This spec
defaults to the fake client and an optional Anthropic API extra. A local
model (llama.cpp, Ollama, or a small Hugging Face model) would fit behind
`LLMClient`, but would need to manage structured outputs without native
schema enforcement, and quality for the plan stage is unproven.

**Maintainer intent (2026-10-04):** "we make both the LLM and the symbolic
algebra machine." The symbolic algebra machine is ours: the solver layer in
`solver/` built on SymPy and Pint (not a from-scratch CAS). Still to
confirm: does "make the LLM" mean training or fine-tuning our own model,
which would pull the v0.7 fine-tuning milestone forward and make on-device
a goal, or building the LLM layer (prompts, plans, validation) around a
hosted model?

**Default for now:** provider-agnostic interface; Anthropic is the first
real provider; v0.7's fine-tuned small open model is the natural path to
on-device. The repo description should be updated once decided.

### Q13. Should few-shot examples be in the plan prompt, and how are they chosen?

Retrieved worked examples are natural few-shot material, but they make the
prompt longer and more variable (bad for caching) and risk the planner
copying numbers from the example.

**Default for now:** worked examples are retrieved and passed to the planner
as context, but not formatted as few-shot demonstrations.

### Q14. How are prompts versioned alongside eval reports?

Each eval report should record the prompt versions and model id it ran
against. Hash the prompt text, or bump a manual version string?

**Default for now:** not tracked (no real prompts in v0.1).
