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
Once multi-equation chaining exists (v0.4), how do we tell a shared symbol
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
logistic regression on eval outcomes (v0.5 data)? Calibrate so "0.8" means
right 80% of the time? Is a single number even the right output, versus
separate "inputs confidence" and "method confidence"?

**Default for now:** the documented crude formula.

### Q10. Offset units: how do we support Celsius and Fahrenheit?

Pint offset units cannot be multiplied (`degC * J/K` is ambiguous). Users
will write "at 25 degrees C".

**Default for now:** data uses kelvin only. In v0.4 the planner is trained to
pass temperatures as given, and the compute stage converts offset units to
kelvin before substitution.

### Q11. How is the CLI distributed?

The interface question is settled: CLI now, website after v1.0 (ADR-008).
Still open: do we publish to PyPI during v0.x or only at v1.0? Do we want a
Homebrew **formula** (for CLI tools) later? A Homebrew **cask** is for GUI
`.app` bundles and does not fit a CLI.

**Update (v0.2):** install channels are decided in ADR-011: curl and irm
installers plus a Homebrew tap, no WinGet. Still open: publishing to PyPI
before v1.0.

**Default for now:** installers and the tap install from GitHub releases;
PyPI at v1.0. Releases are tagged only with the maintainer's approval.

### Q12. Should worked examples be part of the plan input?

Retrieved worked examples could help the Fermi models plan, but they eat
the 1024-token context and risk the model copying numbers from the example
(constrained decoding allows any number in the input, including the
example's).

**Default for now:** plan inputs include equations, constants, and Fermi
assumptions only. Worked examples are training data, not inference context.

### Q13. How are task formats and weights versioned alongside eval reports?

Each eval report should record the model names, weight checksums, and
`FORMAT_VERSION` it ran against. Is a manual version string enough, or do
we hash the format templates?

**Default for now:** a manual `FORMAT_VERSION` integer in `src/askphysics/lm/`
(v0.2), recorded in every eval report (v0.5).

### Q14. Is an 8,192-token vocabulary right for physics text?

A bigger vocabulary shortens sequences (more context for equations) but
costs embedding parameters, which matters a lot for tellus: at d=160, 8k
tokens is already about 1.3M of its ~3.2M parameters.

**Default for now:** 8,192 shared across the family. Revisit with tellus's
v0.3 eval numbers.

### Q15. What happens when retrieved context outgrows 1024 tokens?

At 12 equations it fits easily. At 500 (v0.6), top-k retrieval keeps the
input small, but long equations with many variables add up.

**Default for now:** `top_k` caps context; the plan input lists only each
equation's id, expression, and variable symbols and units. Longer contexts
(2048) are a retrain away if needed.

### Q16. Is there enough training data to justify celeste?

~120M parameters wants roughly 2.4B training tokens. A data factory built
from 12 to 60 equations will repeat itself long before that. Celeste may
just memorize templates better than solem does.

**Default for now:** train celeste last, and keep it only if its rescue
rate on solem failures (ADR-010) justifies the extra weights. CC-BY text
(v0.6) is the main path to more real tokens.

### Q17. How are the website usage tiers windowed?

ADR-010 says a visitor starts on solem, gets one celeste answer per day,
and drops to tellus after five solem answers. Five per what window, per
visitor how (account, IP, cookie), and does the tellus fallback reset daily?

**Default for now:** undecided until the website milestone; values live in
server config so they can change without a release.

### Q18. How do API keys for the hosted API work?

The post-v1.0 website comes with a public HTTP API, and that needs API keys.
Open: who can get a key (self-serve sign-up, or approval), how keys map to
the ADR-010 usage tiers (a free tier on solem, trusted keys getting more
celeste answers), how keys are stored (hashed, with a recognizable prefix
such as `ap_live_` so leaks are easy to search for), rotation and
revocation, per-key rate limits and quotas, and whether the local CLI ever
needs a key (default: no).

**Default for now:** the CLI needs no key, ever. Keys are designed with the
v0.9 API server and ship with the website.

### Q19. How should a conceptual question be classified?

OpenStax *Physics* has hundreds of questions like "Why does a heavier ball
not fall faster?" They are physics, and well posed, but there is nothing to
compute. None of the three categories fits: `standard` and `fermi` promise
a number, and `out_of_scope` refuses with a redirect, which is wrong for a
good question. A fourth category ("conceptual", answered with prose and the
relevant equation) is possible but changes the task format.

**Default for now:** conceptual questions are not used as classify examples
(ADR-016), and the classifier keeps three categories. Revisit when the
explain stage can answer in prose without a number.
