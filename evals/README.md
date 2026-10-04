# Evals

The question set that decides whether Ask Physics is getting better. Design,
rubrics, tolerance rules, and the regression policy live in
[`docs/EVALS.md`](../docs/EVALS.md); this file covers the format and how the
harness will use it.

## Status (v0.1)

- `questions.yaml`: 8 sample questions: 3 standard, 3 Fermi (absurd), and 2
  impossible.
- `harness.py`: `load_questions()` works and is tested, so a malformed eval
  file fails CI. `score_answer()` and `run_suite()` raise
  `NotImplementedError` until v0.5.

## Question format

```yaml
- id: std_001                      # unique, snake case
  question: "A 1200 kg car is moving at 27 m/s. What is its kinetic energy?"
  category: standard               # standard | multi_step | fermi | ambiguous | impossible | adversarial
  expected_behavior: answer        # answer | estimate_range | refuse_and_redirect
  expected_answer_shape:           # null for refusals
    value: 437400.0                # reference value
    unit: J                        # Pint unit; answers are converted before comparing
    rel_tolerance: 0.02            # standard: relative tolerance
    # tolerance_decades: 1.0       # fermi: allowed |log10(answer) - log10(value)|
  must_mention_assumptions: false  # true for Fermi and ambiguous questions
  acceptable_equation_ids:         # optional: any one set earns full equation credit
    - [kinetic_energy]
  rationale: "How the reference was derived, for reviewers."
```

Validation rules enforced by `load_questions()`:

- ids are unique and snake case.
- Exactly one of `rel_tolerance` or `tolerance_decades` is set.
- `refuse_and_redirect` questions have no answer shape; every other behavior
  has one.
- `estimate_range` questions use `tolerance_decades`.

## How the harness will consume this (v0.5)

1. Load and validate `questions.yaml` (works today).
2. For each question, run `Pipeline.run(question)` once per model setup
   (pulsar, quasar, magnetar, and the ADR-010 router). Plans are greedy, so
   one run per setup is deterministic.
3. Score each `Answer` with the category rubric from `docs/EVALS.md`:
   - **Numeric:** convert `final_value` to `unit` with Pint, then apply
     `rel_tolerance` or the decade check against `value`.
   - **Equations:** credit if `equations_used` covers any set in
     `acceptable_equation_ids`.
   - **Assumptions:** rubric checklist (automated checks plus a human-graded
     sample) when `must_mention_assumptions`
     is true.
   - **Refusals:** `status == "refused"`, a stated reason, and an answerable
     redirect.
4. Write `evals/reports/<version>.json` plus a Markdown summary, and compare
   against the last release under the regression policy.

## Rules

- **No leakage.** Never copy or paraphrase these questions into
  `src/askphysics/data/`, the data factory, task formats, or training data.
- **Fake LLM runs are smoke tests.** `FakeLLMClient` has no canned plans for
  these questions, by design. Scores from it say nothing about quality.
- **Changing a reference value** needs a `rationale` update and a reviewer.
