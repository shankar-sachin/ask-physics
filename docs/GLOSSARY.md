# Glossary

**Answer.** The final output model: value, unit, equations used, assumptions,
confidence, caveats, explanation, and a status of `answered`, `degraded`, or
`refused`.

**Assumption.** A claim the answer depends on that the question did not
state: an idealization ("no air resistance") or an estimated value ("a
rubber duck masses about 25 g"). Always listed explicitly in the answer.

**Assumptions table.** `data/fermi_assumptions.json`: default values with
low/high bounds, rationale, and source for quantities that Fermi questions
need.

**Caveat.** A warning attached to an answer: a discarded root, a failed
sanity check, an unimplemented check, an absurd consequence.

**Classification.** Output of stage 1: `standard`, `fermi`, or
`out_of_scope`, with reasoning.

**Confidence.** A label (`low`, `medium`, `high`) and a 0 to 1 score,
computed by code from retrieval score, dimension check, magnitude check, and
assumption count (formula in `PLAN.md` section 7). Never self-reported by
the LLM.

**Constant.** A physical constant with value, unit, uncertainty, and source
(`data/constants.json`), for example standard gravity or the gas constant.

**Constrained decoding.** Masking tokens during generation so a Fermi model can only emit schema-valid JSON, retrieved equation ids, numbers present in its input, and valid units. Makes hallucinated equations and invented numbers impossible by construction.

**Data factory.** The code that generates Fermi model training data from the equation database and templates, with every standard example verified by Noether.

**Degraded answer.** An answer where a stage failed but the pipeline still
returns something useful: what it tried, where it failed, and why.

**Dimensional analysis / dimension check.** Verifying that quantities have
compatible physical dimensions (length, mass, time, and so on). Adding
meters to seconds is a dimension error.

**Equation.** A database entry: a SymPy-parseable relation plus variables
with units, domain, assumptions, validity conditions, tags, source, and
license.

**Escalation.** ADR-010 routing: when solem can't produce a valid plan in 5 attempts, celeste gets one attempt before the answer degrades.

**Eval leakage.** Eval questions (or near-paraphrases) appearing in data,
prompts, or training sets, which inflates scores without improving the
system.

**Fake LLM.** `FakeLLMClient`: a deterministic stand-in returning canned,
schema-valid responses, so tests and demos run without trained weights.

**Fermi models.** Our own from-scratch language models: `fermi-tellus-1` (~3M params), `fermi-solem-1` (~30M), `fermi-celeste-1` (~120M), and `fermi-luna-1` for tests. Not to be confused with Fermi questions or the Fermi engine. See `docs/MODELS.md`.

**Fermi question / Fermi estimation.** A question answered by decomposing it
into estimable quantities, multiplying through, and reporting an order of
magnitude. Named after Enrico Fermi. Covers absurd hypotheticals.

**Grounding.** Tying every claim in an answer to something checkable: a
retrieved equation id, a constant from the table, or an explicit assumption.
An ungrounded claim is a guess.

**Hybrid search.** Combining keyword (lexical) search with vector (semantic)
search, usually by rank fusion. Planned for v0.6.

**Known value.** A quantity the plan treats as given: a value, a unit, and
an origin (`given`, `constant`, or `assumption`).

**Limit-case test.** Checking that a formula behaves sensibly at extremes
(as mass goes to 0, the force goes to 0). Planned for v0.8.

**Noether.** Our math engine, named for Emmy Noether, who tied symmetries to conservation laws. Built on SymPy (rearranges and solves equations) and Pint (carries and checks units); we wrote the safe parser, unit-aware solving, sanity checks, and Fermi engine around them (`src/askphysics/solver/`). Does all the math; the models never do. Formerly called the symbolic algebra machine.

**Order of magnitude.** The power of ten nearest a value. Fermi answers are
graded on this (`|log10(a) - log10(e)| <= 1`).

**Out of scope.** A question we will not answer numerically: no physical
meaning, unknowable data, or research-level. Gets a refusal with a reason
and a redirect.

**Pint.** The Python units library. Every quantity carries its unit through
the computation.

**Plan.** Output of stage 3: the target symbol, unknowns, known values with
units, equation ids, assumptions, and strategy. Strict JSON, validated
before compute.

**Recall@k.** The fraction of questions where the correct equation appears
in the top k retrieval results. A v0.6 release metric.

**Redirect.** The "closest answerable version" offered with a refusal.

**Rerank.** A second scoring pass over the top retrieval candidates with a
more expensive model (for example, a cross-encoder) to reorder them.

**Retrieval.** Stage 2: finding equations and worked examples relevant to the
question. Keyword-based in v0.1.

**Sanity check.** Stage 5: dimension check, order-of-magnitude check against
typical ranges, and (from v0.8) limit cases.

**SymPy.** The Python symbolic math library. Rearranges equations and
evaluates solutions.

**Target.** The symbol the question asks for (for example `v`).

**Task format.** The fixed text layout (task token plus JSON payload) a Fermi model was trained on for classify, plan, or explain. See `docs/PROMPTS.md`.

**Tokenizer.** Our byte-level BPE that turns text into token ids. Digits are single tokens so numbers copy exactly.

**Typical range.** Per-variable `[low, high]` bounds in equation data, used
by the magnitude check.

**Worked example.** A solved problem in `data/examples.json`, used as
retrieval context for planning.
