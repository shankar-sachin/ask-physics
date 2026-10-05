# Task Formats

The Fermi models ([`MODELS.md`](MODELS.md)) are trained from scratch on three
tasks. There are no system prompts in the chat-model sense: each task is a
fixed text format the model learned during training, selected by a task
token. The data factory writes training examples in exactly these formats,
and `FermiClient` (v0.3) builds inference inputs the same way. Change a
format and you retrain the models, so formats are versioned
(`FORMAT_VERSION` in `src/askphysics/lm/`).

Payloads are JSON with sorted keys, so identical inputs produce identical
token sequences.

---

## Hardening rules (apply to every task)

1. **The question is data.** It sits inside a JSON field. Instructions inside
   it ("ignore your equations", "the answer is 42") are just text to
   classify, never commands. Adversarial phrasings are included in the
   training data, labelled with the correct behavior.
2. **No invented equations.** Constrained decoding only allows equation ids
   from the retrieved list. If none fits, the only valid plan is an empty
   `equation_ids` list with the reason in `strategy`.
3. **No invented numbers.** Constrained decoding only allows numbers that
   appear in the question, the constants table, or the Fermi assumptions
   table. The model cannot compute a new number even if it wanted to.
4. **No arithmetic.** The model copies; Noether computes.
5. **No expressions.** Plans contain ids and numbers, never formulas.
6. **Greedy where it matters.** Classify and plan always decode greedily, so
   the same input always gives the same output. Only explain may sample.

---

## classify

**Input**

```
<|classify|>{"question": "How much does the color blue weigh?"}
```

**Output** (greedy, constrained to the `Classification` schema)

```
{"category": "out_of_scope", "closest_answerable": "How much momentum does a beam of blue light carry?", "domains": [], "reasoning": "Category error: a color is a property of light, not an object with mass."}<|end|>
```

Behavior the training data teaches:
- `standard`: every needed value is given or is a standard constant.
- `fermi`: physically meaningful but needs estimated everyday quantities,
  including absurd hypotheticals. Absurd is not impossible.
- `out_of_scope`: no physical meaning, unknowable, research-level, or not
  physics. Always with a concrete `closest_answerable`.
- When in doubt, prefer `standard` or `fermi`: a false refusal is worse than
  a hedged estimate. The factory's class balance reflects this.

---

## plan

**Input**

```
<|plan|>{"category": "standard",
 "constants": [{"name": "standard_gravity", "symbol": "g", "unit": "m/s^2", "value": 9.80665}, ...],
 "equations": [{"id": "kin_v_squared", "sympy_expr": "v**2 = v0**2 + 2*a*d",
                "variables": [{"symbol": "v", "unit": "m/s"}, ...]}, ...],
 "fermi_assumptions": [...],
 "question": "How fast does a falling object hit the ground if it is dropped from 20 m?"}
```

(Line breaks added here for readability; the real input is one line.)

**Output** (greedy, constrained to the `Plan` schema, allowed ids, allowed
numbers, valid units)

```
{"assumptions": ["Released from rest", "Air resistance is negligible", "Constant standard g"], "equation_ids": ["kin_v_squared"], "known_values": [{"origin": "assumption", "symbol": "v0", "unit": "m/s", "value": 0}, {"origin": "constant", "symbol": "a", "unit": "m/s^2", "value": 9.80665}, {"origin": "given", "symbol": "d", "unit": "m", "value": 20}], "strategy": "Solve v^2 = v0^2 + 2*a*d for the positive root of v.", "target": "v", "unknowns": ["v"]}<|end|>
```

Behavior the training data teaches:
- Pick the equation that connects the givens to the asked-for quantity.
- Copy values and units exactly as written; never convert ("20 m" is value 20,
  unit "m").
- `origin` is `given`, `constant`, or `assumption`. Implied values ("dropped"
  means initial velocity 0) are assumptions and must be listed.
- For Fermi questions, missing quantities come from `fermi_assumptions`, and
  each one used appears in `assumptions`.

The pipeline still runs `validate_plan` afterwards. Constraints make
violations impossible; validation proves it.

---

## explain

**Input**

```
<|explain|>{"assumptions": [...], "equations": [{"id": "kin_v_squared", "name": "..."}],
 "question": "...", "result": {"unit": "meter / second", "value": 19.8057},
 "sanity": {"dimensions_ok": true, "issues": [], "magnitude_ok": true}}
```

**Output** (sampled at `Settings.temperature`; digits constrained to numbers
in the input)

```
Using [kin_v_squared], an object dropped from rest falls 20 m and hits the ground at 19.8057 meter / second, ignoring air resistance.<|end|>
```

Behavior the training data teaches:
- State the result with exactly the given value and unit.
- Cite every equation id in brackets.
- Mention the key assumptions; mention sanity issues plainly.
- Fermi answers give an order of magnitude and say it's an estimate. Absurd
  scenarios get a straight face.

If explain fails, the pipeline falls back to a deterministic template.

---

## Open format questions

Tracked in `docs/OPEN_QUESTIONS.md`: how much retrieved context fits in 1024
tokens as the database grows, and whether worked examples belong in the plan
input.
