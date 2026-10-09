# TODO

Prioritized backlog for the next two milestones. Each item should fit in one
sitting. Within each area, items are in priority order. Check items off in
the PR that completes them. Milestone details: [`docs/ROADMAP.md`](docs/ROADMAP.md);
model design: [`docs/MODELS.md`](docs/MODELS.md).

---

## v0.2.0 Fermi foundations

### Tooling
- [x] Add `torch` and `safetensors` dependencies; CI installs the CPU-only
      torch wheel.
- [x] `lm/config.py`: `ModelConfig` presets for luna, tellus, solem, and
      celeste, with a parameter-count function and a test pinning each count.
- [x] `lm/device.py`: pick MPS, then CUDA, then CPU; honor an override.

### Tokenizer
- [x] Byte-level BPE trainer: merges learned from a text corpus, digits never
      merged, task tokens reserved.
- [x] Encoder and decoder with byte fallback; round-trip property test.
- [x] Save and load the tokenizer as JSON alongside the weights.

### Model
- [x] RMSNorm, rotary embeddings, causal attention via
      `scaled_dot_product_attention`, SwiGLU MLP, tied embeddings.
- [x] Shape tests for every config; a causal-mask test (future tokens can't
      change past logits).
- [x] safetensors save and load, with a config hash check; a test that
      `torch.load` is never used.

### Decoding
- [x] Greedy and sampled generation with a token budget and `<|end|>` stop.
- [x] JSON schema constraint for `Classification` and `Plan` (prefix-valid
      masking).
- [x] Allowed-id constraint for `equation_ids`; allowed-number constraint
      for `value` fields; Pint-unit constraint for `unit` fields.
- [x] Property tests: random weights plus constraints can only ever produce
      valid plans citing retrieved ids and input numbers.

### Data factory
- [x] Phrasing templates for standard questions (at least 30 per target
      quantity kind), with held-out templates kept aside for validation.
- [x] Standard-question generator: sample values in `typical_range`, render,
      build the gold `Plan`, solve with `solve_for`, drop failures.
- [x] Fermi and out-of-scope generators from the assumptions table and
      category-error templates, each with reasons and redirects.
- [x] Explanation generator from plan plus computed result.
- [x] Eval leakage check: drop anything too similar to `evals/questions.yaml`.
- [x] `askphysics model build-data` writing sharded JSONL with a manifest.

### Training
- [x] Training loop: AdamW, warmup plus cosine schedule, gradient clipping,
      bf16 autocast on MPS and CUDA, seeded data order.
- [x] Checkpointing and resumption; loss and throughput logging.
- [x] `askphysics model train-tokenizer`, `train`, and `info` commands.
- [x] CI test: `fermi-luna-1` trains 50 steps on CPU and its loss drops.

---

## v0.3.0 Fermi models wired in (done)

The step-by-step training commands are in [`docs/TRAINING.md`](docs/TRAINING.md).

- [x] Train `fermi-tellus-1`; record the loss curve and held-out plan validity.
- [x] `FermiClient` implementing `LLMClient` with the task formats from
      `docs/PROMPTS.md`.
- [x] Router per ADR-010: tellus classifies, solem plans and explains, up
      to 5 solem attempts, one celeste escalation, tellus-only fallback.
- [x] Record the model used per stage on `Answer`.
- [x] Add the `fermi` provider to `Settings`; make it the default when
      weights are installed, with a clear message when they aren't.
- [x] `askphysics ask --model` to force a specific Fermi model.
- [x] Held-out factory eval (`model eval`): valid-plan rate, answers after
      the router's retries, and confidently wrong answers.
- [x] 110 equations; the question normalizer; OpenStax *Physics* prose.

---

## v0.4.0 solem, weights, solver, and the docs

### Training (on the maintainer's M5 Pro)
- [ ] Train `fermi-solem-1` with the prose stage; record the loss curve,
      held-out plan validity, and confidently wrong rate.
- [ ] Measure real tokens per second for each config; update `docs/MODELS.md`.
- [ ] Train `fermi-celeste-1` (~120M) only if solem's held-out results leave
      room for it to help (open question Q16).

### Distribution (ADR-012)
- [x] Export bf16 safetensors per model and a manifest (URL, size, sha256)
      (`model package`, #51).
- [x] `askphysics model pull [--all]` with checksum verification (#51).
- [x] Installers run `model pull` (#51). Homebrew can't write to the home
      directory, so its formula prints a caveat instead (`docs/RELEASING.md`).
- [x] Download celeste on first escalation; clear message when no weights
      are installed (#51).

### Evals
- [x] `model eval --rescue-with` measures how many of solem's misses celeste
      rescues; `--grad-accum` lets celeste train in half batches.
- [ ] Celeste rescue rate on the held-out factory set (needs a trained celeste).
- [x] Real-question eval from OpenStax *Physics* exercises, with
      project-written gold plans (ADR-016): 49 questions, each gold answer
      checked against the book's answer key, run by `model eval`.
- [ ] Maintainer review of the 49 gold plans in
      `third_party/openstax-physics/real_eval.jsonl`.
- [ ] Keyword retrieval misses the gold equations of 7 of the 49 real
      questions at top_k 5 ("a 5-kg object ... accelerate at 20 m/s^2" never
      sees Newton's second law), capping any model at 85.7%.
- [x] OpenStax questions as extra classify examples (180 problems, held out
      from the real-question eval).
- [ ] Run the 8 eval questions per model and record results in the model cards.

### Solver
- [x] Multi-equation chaining: dependency order over the plan's unknowns (#52).
- [x] Offset units (Celsius, Fahrenheit) converted before substitution (#52).
- [x] 15+ worked examples, each re-solved within 0.1% by a test.

### Ask Physics Docs
- [ ] Model cards for each trained model: config, data, tokens, time, curves,
      results, known failure modes, OpenStax attribution.
- [x] `docs/pages/` as the single source: guides, FAQ, glossary (#50). Model
      cards join once the trained models are packaged.
- [x] Generate an equation page per database entry (formula, variables with
      units, validity, source, license) (#50).
- [x] Ask Physics Docs at `/docs/` built from the same pages, with rendered math
      and search; link each equation id on the answer card to its page.

## After fermi-celeste-1: three packages (ADR-018, proposed)

- [ ] Untangle imports so `lm/` depends only on the solver and data (move
      compute, sanity checks, normalization, and the shared models).
- [ ] Split into `noether`, `fermi`, and `ask-physics` repositories with
      history, each with its own CI and PyPI release.
- [ ] Installers, Homebrew, and the website unchanged for users.
