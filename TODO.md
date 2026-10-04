# TODO

Prioritized backlog for the next two milestones. Each item should fit in one
sitting. Within each area, items are in priority order. Check items off in
the PR that completes them. Milestone details are in
[`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## v0.2.0 Real retrieval

### Retrieval
- [ ] Write the 50-question retrieval eval set (`evals/retrieval.yaml`):
      question plus correct equation ids, written without looking at tags.
- [ ] Add a `recall_at_k` script that scores any `Retriever` on that set;
      record the `KeywordRetriever` baseline in `docs/DECISIONS.md`.
- [ ] Define an `Embedder` protocol in `retrieval/base.py` (`embed(texts) -> list[list[float]]`).
- [ ] Implement a local sentence-transformer `Embedder` behind an optional
      extra (`askphysics[vector]`).
- [ ] Decide what text gets embedded per equation (name + tags +
      variable descriptions + assumptions) and document it in
      `docs/ARCHITECTURE.md`.
- [ ] Implement `SqliteVecStore` (`add`, `query`) with an in-memory test.
- [ ] Implement `VectorRetriever.search` on top of the store.
- [ ] Add a SQLite FTS5 BM25 keyword index as a second lexical scorer.
- [ ] Implement `HybridRetriever` with reciprocal rank fusion of BM25 and
      vector results.
- [ ] Add an index build command, `askphysics build-index`, that rebuilds from JSON.
- [ ] Run the reranking experiment (cross-encoder over the top 20) and write
      up the result as an ADR.
- [ ] Make the retriever choice configurable in `Settings` (`retriever = "keyword" | "hybrid"`).

### Data
- [ ] Add 20 more hand-curated equations to stress retrieval vocabulary
      (circular motion, springs, power, pressure, density).
- [ ] Add `synonyms` handling to keyword tokenization ("speed" ~ "velocity",
      "hit the ground" ~ "fall") or decide it is unnecessary given hybrid search.

### Evals
- [ ] Add retrieval Recall@5 to CI as a non-blocking report.

### Tooling
- [ ] Add a `make bench` target timing retrieval p95 on the seed DB.
- [ ] Add a pre-commit config running ruff and mypy.

---

## v0.3.0 Real solver

### Solver
- [ ] Implement `AnthropicClient.complete_json` with `client.messages.parse`
      and pydantic schemas; handle the `refusal` stop reason.
- [ ] Implement `AnthropicClient.complete_text` for the explain stage.
- [ ] Record-and-replay test fixture for `AnthropicClient`, so tests stay offline.
- [ ] Move the prompts from `docs/PROMPTS.md` into code with a `PROMPT_VERSION`.
- [ ] Implement the re-plan loop: one retry with the `PlanValidationError`
      message fed back.
- [ ] Multi-equation chaining in `compute`: build a dependency order from
      the plan and solve intermediates first.
- [ ] Offset unit handling: convert degC and degF to K before substitution (Q10).
- [ ] Add a `root_preference` field to `Plan`, or document why not (Q7).
- [ ] Check that numbers in `explanation` match `final_value` (regex extract
      plus tolerance) and add a caveat on mismatch.
- [ ] Map `validity_conditions` into caveats on every answer that uses the
      equation.

### Data
- [ ] Grow the seed DB to about 60 equations across kinematics, dynamics,
      energy, basic E&M, and thermodynamics.
- [ ] Add 15 worked examples, at least 3 per domain, none resembling eval
      questions.
- [ ] Re-solve every worked example in `validate-data` and compare to
      `final_answer` within 0.1%.

### Evals
- [ ] Grow `evals/questions.yaml` to 30 standard and 10 multi-step questions.
- [ ] Implement `evals/harness.py` numeric and unit scoring for the standard
      category (the rest lands in v0.4).
- [ ] Run each eval question 3 times against the real LLM and record the
      plan variance (ADR-006).

### Tooling
- [ ] Add `askphysics ask --trace` to dump every stage's payload as JSON.
- [ ] Add `askphysics ask --llm anthropic` with a clear error when the extra
      or `ANTHROPIC_API_KEY` is missing.
