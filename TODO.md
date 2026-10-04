# TODO

Prioritized backlog for the next two milestones. Each item should fit in one
sitting. Within each area, items are in priority order. Check items off in
the PR that completes them. Milestone details: [`docs/ROADMAP.md`](docs/ROADMAP.md);
model design: [`docs/MODELS.md`](docs/MODELS.md).

---

## v0.2.0 Fermi foundations

### Tooling
- [ ] Add `torch` and `safetensors` dependencies; CI installs the CPU-only
      torch wheel.
- [ ] `lm/config.py`: `ModelConfig` presets for luna, tellus, solem, and
      celeste, with a parameter-count function and a test pinning each count.
- [ ] `lm/device.py`: pick MPS, then CUDA, then CPU; honor an override.

### Tokenizer
- [ ] Byte-level BPE trainer: merges learned from a text corpus, digits never
      merged, task tokens reserved.
- [ ] Encoder and decoder with byte fallback; round-trip property test.
- [ ] Save and load the tokenizer as JSON alongside the weights.

### Model
- [ ] RMSNorm, rotary embeddings, causal attention via
      `scaled_dot_product_attention`, SwiGLU MLP, tied embeddings.
- [ ] Shape tests for every config; a causal-mask test (future tokens can't
      change past logits).
- [ ] safetensors save and load, with a config hash check; a test that
      `torch.load` is never used.

### Decoding
- [ ] Greedy and sampled generation with a token budget and `<|end|>` stop.
- [ ] JSON schema constraint for `Classification` and `Plan` (prefix-valid
      masking).
- [ ] Allowed-id constraint for `equation_ids`; allowed-number constraint
      for `value` fields; Pint-unit constraint for `unit` fields.
- [ ] Property tests: random weights plus constraints can only ever produce
      valid plans citing retrieved ids and input numbers.

### Data factory
- [ ] Phrasing templates for standard questions (at least 30 per target
      quantity kind), with held-out templates kept aside for validation.
- [ ] Standard-question generator: sample values in `typical_range`, render,
      build the gold `Plan`, solve with `solve_for`, drop failures.
- [ ] Fermi and out-of-scope generators from the assumptions table and
      category-error templates, each with reasons and redirects.
- [ ] Explanation generator from plan plus computed result.
- [ ] Eval leakage check: drop anything too similar to `evals/questions.yaml`.
- [ ] `askphysics model build-data` writing sharded JSONL with a manifest.

### Training
- [ ] Training loop: AdamW, warmup plus cosine schedule, gradient clipping,
      bf16 autocast on MPS and CUDA, seeded data order.
- [ ] Checkpointing and resumption; loss and throughput logging.
- [ ] `askphysics model train-tokenizer`, `train`, and `info` commands.
- [ ] CI test: `fermi-luna-1` trains 50 steps on CPU and its loss drops.

---

## v0.3.0 Fermi models trained and wired in

### Training (on the maintainer's M5 Pro)
- [ ] Measure real tokens per second for each config; update `docs/MODELS.md`.
- [ ] Train `fermi-tellus-1`; record the loss curve and held-out plan validity.
- [ ] Train `fermi-solem-1`; same.
- [ ] Train `fermi-celeste-1` (~120M) only if solem's held-out results leave
      room for it to help (open question Q16).

### Integration
- [ ] `FermiClient` implementing `LLMClient` with the task formats from
      `docs/PROMPTS.md`.
- [ ] Router per ADR-010: tellus classifies, solem plans and explains, up
      to 5 solem attempts, one celeste escalation, tellus-only fallback.
- [ ] Record the model used per stage on `Answer`.
- [ ] Add the `fermi` provider to `Settings`; make it the default when
      weights are installed, with a clear message when they aren't.
- [ ] `askphysics ask --model` to force a specific Fermi model.

### Evals
- [ ] Held-out factory set (unseen templates): valid-plan rate per model and
      celeste rescue rate.
- [ ] Run the 8 eval questions per model and record results in the model cards.

### Docs
- [ ] Model cards for each trained model: config, data, tokens, time, curves,
      results, known failure modes.
