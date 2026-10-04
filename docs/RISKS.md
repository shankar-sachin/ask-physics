# Risk Register

Likelihood and impact are **L**ow, **M**edium, or **H**igh. "Owner" is the
role responsible for the mitigation. While the project has a single
maintainer, every role resolves to that person, but naming the role says
which hat to wear. Reviewed at every milestone.

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|------|------------|--------|------------|-------|
| R1 | **Hallucinated equations**: the LLM invents or misremembers a formula | H | H | The planner may only cite retrieved equation ids; `validate_plan` rejects unknown ids; the LLM never emits expressions; explain-stage general knowledge must be labelled "unverified" | Pipeline lead |
| R2 | **Wrong units**: a unit dropped, mixed up, or mis-converted | H | H | Pint quantities everywhere; bare floats banned across module boundaries; dimension check in `sanity_check` caps confidence at 0.2 on failure; data validation checks every unit and every equation's dimensional consistency | Solver lead |
| R3 | **Retrieval misses**: the right equation exists but isn't retrieved | H | M | v0.2 hybrid search; a retrieval eval set with Recall@5 as a release metric; retrieval failures produce a degraded answer naming the gap, never a silent LLM fallback | Retrieval lead |
| R4 | **License contamination**: incompatible data ships in an MIT package | M | H | `license` field required and checked against an allow-list; CC-BY-SA quarantined; `source` must be verifiable; data PR checklist | Data lead |
| R5 | **Eval overfitting**: prompts and data tuned to the eval set | M | H | Leakage policy and similarity check (`docs/EVALS.md`); 20% held-out split run only at release; few-shot examples drawn from data, never evals | Eval lead |
| R6 | **LLM provider lock-in**: features built on one vendor's quirks | M | M | `LLMClient` protocol with only two methods; the fake client proves the contract; no provider-specific types leak past `llm/` | Pipeline lead |
| R7 | **Absurd-question prompt injection**: a "hypothetical" smuggles instructions ("imagine you are an AI with no rules...") | M | M | Question passed as JSON data; hardening rules in `docs/PROMPTS.md`; the LLM cannot affect numbers or confidence; adversarial eval category with a 1-point regression threshold | Eval lead |
| R8 | **False confidence**: a high confidence label on a wrong answer | M | H | Confidence computed by code from checks, never self-reported; Fermi capped at 0.6; dimension failure caps at 0.2; formula refit on eval data in v0.8 | Pipeline lead |
| R9 | **Scope creep**: GUI, multimodal, research physics before the core works | H | M | Non-goals in `PLAN.md`; milestone exit criteria gate new work; ambiguities go to `docs/OPEN_QUESTIONS.md`, not into code | Maintainer |
| R10 | **Cost blowup**: real LLM calls in tests, eval runs, or retry loops | M | M | Tests use `FakeLLMClient` only; one re-plan retry max; per-run cost caps and call caching (v0.9); real-LLM evals on demand, not on every PR | Maintainer |
| R11 | **Unsafe expression parsing**: crafted strings reaching `sympify`/`eval` | L | H | The LLM never emits expressions; seed expressions use a restricted parser with a character whitelist and no builtins; covered in `SECURITY.md` | Solver lead |
| R12 | **Multiple roots or wrong branch**: a negative time or speed chosen | M | M | Real, then non-negative root filter; the choice recorded in caveats; limit-case checks in v0.8 | Solver lead |
| R13 | **Stale constants**: a CODATA update not reflected | L | L | `source` records the CODATA year; refresh script in v0.6; most constants are exact by SI definition | Data lead |
| R14 | **Model API drift**: provider parameters change (for example, sampling params removed) | M | M | All provider code isolated in one client module; ADR-006 records the current constraint; client covered by a recorded-response test from v0.3 | Pipeline lead |
