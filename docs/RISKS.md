# Risk Register

Likelihood and impact are **L**ow, **M**edium, or **H**igh. "Owner" is the
role responsible for the mitigation. While the project has a single
maintainer, every role resolves to that person, but naming the role says
which hat to wear. Reviewed at every milestone.

| # | Risk | Likelihood | Impact | Mitigation | Owner |
|---|------|------------|--------|------------|-------|
| R1 | **Hallucinated equations**: the model invents or misremembers a formula | H | H | Constrained decoding only allows retrieved equation ids; `validate_plan` double-checks; models never emit expressions; explain-stage digits are constrained to input numbers | Model lead |
| R2 | **Wrong units**: a unit dropped, mixed up, or mis-converted | H | H | Pint quantities everywhere; bare floats banned across module boundaries; dimension check in `sanity_check` caps confidence at 0.2 on failure; data validation checks every unit and every equation's dimensional consistency | Solver lead |
| R3 | **Retrieval misses**: the right equation exists but isn't retrieved | H | M | v0.2 hybrid search; a retrieval eval set with Recall@5 as a release metric; retrieval failures produce a degraded answer naming the gap, never a silent LLM fallback | Retrieval lead |
| R4 | **License contamination**: incompatible data ships in an MIT package | M | H | `license` field required and checked against an allow-list; CC-BY-SA quarantined; `source` must be verifiable; data PR checklist | Data lead |
| R5 | **Eval overfitting**: prompts and data tuned to the eval set | M | H | Leakage policy and similarity check (`docs/EVALS.md`); 20% held-out split run only at release; few-shot examples drawn from data, never evals | Eval lead |
| R6 | **Small-model generalization**: a from-scratch 3M to 120M model fails on phrasings unlike its training data | H | H | Diverse templates with held-out templates for validation; adversarial and messy phrasings in the factory; magnetar escalation; failures degrade safely (constraints); measured per model in v0.5 evals | Model lead |
| R7 | **Absurd-question prompt injection**: a "hypothetical" smuggles instructions ("imagine you are an AI with no rules...") | M | M | Question passed as JSON data; adversarial examples in training data; constraints mean the model cannot affect numbers, equations, or confidence; adversarial eval category with a 1-point regression threshold | Eval lead |
| R8 | **False confidence**: a high confidence label on a wrong answer | M | H | Confidence computed by code from checks, never self-reported; Fermi capped at 0.6; dimension failure caps at 0.2; formula refit on eval data in v0.8 | Pipeline lead |
| R9 | **Scope creep**: GUI, multimodal, research physics before the core works | H | M | Non-goals in `PLAN.md`; milestone exit criteria gate new work; ambiguities go to `docs/OPEN_QUESTIONS.md`, not into code | Maintainer |
| R10 | **Training cost blowup**: magnetar runs eat days of laptop time, or someone pushes for 1B params | M | M | Budgets in `docs/MODELS.md` (~20 tokens per parameter); nano config for all CI; pulsar and quasar first, magnetar only after quasar's eval numbers justify it; 1B is out of scope (ADR-009) | Maintainer |
| R11 | **Unsafe expression parsing**: crafted strings reaching `sympify`/`eval` | L | H | The LLM never emits expressions; seed expressions use a restricted parser with a character whitelist and no builtins; covered in `SECURITY.md` | Solver lead |
| R12 | **Multiple roots or wrong branch**: a negative time or speed chosen | M | M | Real, then non-negative root filter; the choice recorded in caveats; limit-case checks in v0.8 | Solver lead |
| R13 | **Stale constants**: a CODATA update not reflected | L | L | `source` records the CODATA year; refresh script in v0.6; most constants are exact by SI definition | Data lead |
| R14 | **Synthetic-data artifacts**: models learn template quirks instead of language | H | M | Many templates per task, randomized slot order and wording, held-out templates, CC-BY text from v0.6, and evals written by humans | Data lead |
| R15 | **Unsafe weight loading**: a malicious checkpoint executes code when loaded | L | H | Weights are safetensors only, never pickle; `torch.load` banned in the package; checksums verified on download (v0.9); covered in `SECURITY.md` | Model lead |
| R16 | **Weights distribution**: weights too big for git, missing on a fresh install | M | M | Weights in `~/.cache/askphysics/models/`; router falls back to pulsar or the fake client; release assets with checksums (v0.9), published only with maintainer approval | Maintainer |
| R17 | **Platform drift**: MPS, CUDA, and CPU kernels give slightly different results | M | L | Greedy decoding plus constraints keep plans stable; retrains compared by eval scores, not weights; CI runs the CPU path | Model lead |
| R18 | **LLM provider lock-in**: largely retired by ADR-009, since there's no external provider. What remains is lock-in to PyTorch and our own formats | L | L | `LLMClient` protocol with two methods; task formats documented in `docs/PROMPTS.md`; weights in a framework-neutral format (safetensors) | Model lead |
