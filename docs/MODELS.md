# The Fermi Model Family

Ask Physics runs on its own language models, written and trained from
scratch in this repo. No pretrained weights, no external APIs. They do the
language work: classify the question, pick equations and copy values into a
plan, and explain the result. Noether (SymPy + Pint)
does every piece of math. Decisions: ADR-009 (from scratch) and ADR-010
(routing) in [`DECISIONS.md`](DECISIONS.md).

## The family

<p align="center">
  <img src="images/fermi-luna-1.jpg" alt="fermi-luna-1" width="24%">
  <img src="images/fermi-tellus-1.jpg" alt="fermi-tellus-1" width="24%">
  <img src="images/fermi-solem-1.jpg" alt="fermi-solem-1" width="24%">
  <img src="images/fermi-celeste-1.jpg" alt="fermi-celeste-1" width="24%">
</p>

Each model is named for a body that grows with it: luna the Moon, tellus the
Earth, solem the Sun, and celeste a quasar, the brightest thing in the sky.
The artwork comes from `make brand` (`scripts/brand.py`).

| Model | Params | Layers | d_model | Heads | Context | Role |
|-------|--------|--------|---------|-------|---------|------|
| `fermi-tellus-1` | ~3.2M | 6 | 160 | 5 | 1024 | Classifies every question; full fallback when bigger weights are missing |
| `fermi-solem-1` | ~29M | 8 | 512 | 8 | 1024 | Default planner and explainer |
| `fermi-celeste-1` | ~119M | 16 | 768 | 12 | 1024 | One escalation shot when solem can't produce a valid plan |
| `fermi-luna-1` | ~0.1M | 2 | 64 | 4 | 1024 | Tests and CI only; never ships |

All share one tokenizer (vocabulary 8,192), so data, prompts, and decoding
constraints are identical across sizes. Parameter counts are approximate
until the configs are frozen in `src/askphysics/lm/config.py` (v0.2).

The model code lives in `src/askphysics/lm/` ("language models"), not
`fermi/`, so it can't be confused with `solver/fermi.py`, which does Fermi
*estimation* (absurd-question math). Same surname, different jobs.

## Architecture

Decoder-only transformer, written in PyTorch:

- **Token embeddings** tied with the output head.
- **RMSNorm**, pre-norm residual blocks.
- **Rotary position embeddings (RoPE)**.
- **Causal self-attention** through `torch.nn.functional.scaled_dot_product_attention`.
- **SwiGLU MLP** with hidden size about 8/3 of `d_model`.
- No biases, no dropout at these sizes (revisit if celeste overfits).

Devices: Apple Silicon (MPS) first, then CUDA, then CPU, picked automatically.

## Tokenizer

Our own byte-level BPE, trained locally on the training corpus:

- **Digits are always single tokens.** Numbers are spelled digit by digit, so
  the model copies them exactly and constrained decoding can restrict them.
- **Task tokens:** `<|classify|>`, `<|plan|>`, `<|explain|>`, `<|end|>`, plus
  `<|pad|>`.
- Byte fallback, so any input encodes without unknown tokens.

## Tasks and formats

One model, three tasks, selected by the task token. Formats are in
[`PROMPTS.md`](PROMPTS.md). Outputs:

| Task | Output | Decoding |
|------|--------|----------|
| classify | A `Classification` JSON object | Greedy, constrained to the schema and the category set |
| plan | A `Plan` JSON object | Greedy, constrained (below) |
| explain | Prose | Sampled at `Settings.temperature`, numbers constrained |

## Constrained decoding

At every step the decoder masks tokens that would break the rules, so these
failures are impossible by construction rather than caught afterwards:

- **Schema:** output must stay a valid prefix of the target JSON schema.
- **Equation ids:** only ids from the retrieval result can appear in
  `equation_ids`. Hallucinated equations cannot be emitted.
- **Numbers:** a `value` can only be a number that appears in the question,
  the constants table, the Fermi assumptions table, or the assumed 0. A table
  constant fills only its own slot (`c` for `c`, `g` for `g`), not every
  variable with matching units, and the 0 fills only an initial speed when the
  question says the thing starts at rest ("from rest", "dropped", "released").
  A stated "from 0 m/s" is a given value, and a momentum, force, mass, or time
  is never 0. The structural 1 is never a value; it is only for prose (the
  `assumptions` and `strategy` text, for example `1/2 m v^2`). A slot with no
  legal value fails the plan (`PlanValidationError`): the router tries again
  or the answer degrades. The decoder never fills it in.
- **Units:** only strings that parse in the shared Pint registry.
- **Dimensions (standard plans, ADR-015):** a known value must be a quantity
  from the question (its number and the unit written after it, together), a
  table constant, or an assumed 0, and its units must fit the
  variable. The target must be a variable the question leaves open: fewer
  quantities of its dimensions are given than the equations have variables
  of those dimensions. Each stated quantity fills one variable at most, and
  a constant or 0 only fills a slot once no stated quantity that fits is
  left.
- **Explain:** digits in prose may only spell numbers present in the
  computed result or the plan.

## Routing (ADR-010)

**CLI (now): split and escalate.**

```
question -> tellus: classify
         -> solem: plan  --valid?--> compute -> solem: explain
                      |
                      +-- invalid plan or compute failure: retry solem (up to 5 attempts,
                          varying the retrieved context order and sampling seed)
                      +-- still failing: ONE attempt on celeste
                      +-- still failing: degraded answer
```

If solem or celeste weights are not installed, the router skips them, and
tellus does every task on its own. Every answer records which model produced
each stage.

**Website (after v1.0): usage tiers.** Each visitor starts on solem, gets
one celeste answer per day, and drops to tellus after five solem answers
in a window. Limits live in server config. They exist to cap hosting cost,
so the local CLI never enforces them.

## Training data

All generated by a data factory in this repo (v0.2), never by an external
model:

1. **Standard questions** from the equation database: for each equation and
   target symbol, sample values inside each variable's `typical_range`,
   render through phrasing templates, and pair with the gold `Plan`.
   Questions are composed from parts (frame, how each value is stated,
   variable synonyms, unit spellings and conversions, openers, sign-offs,
   casual punctuation) or come from worded scenarios, so the models can't
   just memorize sentences. Any held-out part sends an example to validation. Noether solves every example; anything it can't solve is
   dropped.
2. **Fermi questions** from the assumptions table ("how many X to Y").
3. **Out-of-scope questions** from category-error, unknowable, and non-physics
   templates, each paired with a reason and a redirect.
4. **Explanations** generated from plan plus computed result.
5. **Real human phrasing (ADR-016):** questions and prose from OpenStax
   *Physics* (2020, CC BY 4.0, text only, attributed). Not *University
   Physics* or *College Physics*, which are CC BY-NC-SA. The prose is in
   `third_party/openstax-physics/` and trains as a language-modeling stage
   before the tasks (`--prose`); its calculation problems are classify
   examples, except the 49 kept for the real-question eval. A model card for weights trained with it
   must repeat the credit in that directory's `ATTRIBUTION.md`. The same holds
   for the 20M-word corpus built by `scripts/build_corpus.py` (ADR-017),
   whose build writes its own `ATTRIBUTION.md`.

The factory never reads `evals/`, and a similarity check enforces it
(`docs/EVALS.md`).

## Training

- AdamW, cosine learning-rate schedule with warmup, gradient clipping.
- bf16 autocast on MPS and CUDA, fp32 on CPU.
- Seeded data order and initialization; checkpoints every N steps.
- Rough budgets at about 20 tokens per parameter, on an M5 Pro:

| Model | Tokens | Estimated time |
|-------|--------|----------------|
| tellus | ~60M | Minutes |
| solem | ~600M | 1 to 2 hours |
| celeste | ~2.4B | About a day |

These are estimates until v0.4 measures real throughput. The step-by-step
commands are in [`TRAINING.md`](TRAINING.md).

## Weights

- Saved to `~/.cache/askphysics/models/<model-name>/`, never committed to git.
- Stored as `safetensors`, never pickled `torch.save` files, because loading a
  pickle runs arbitrary code (see `SECURITY.md`).
- Users never train: the maintainer trains and ships the weights as GitHub
  release assets, pinned by a checksum manifest and fetched with
  `askphysics model pull` (ADR-012). The installers and the Homebrew formula
  fetch tellus and solem; celeste comes down on first escalation.

## Known limits

- A from-scratch model this small understands phrasings close to its
  training data. Unusual wording will fail more often than with a large
  pretrained model. Constrained decoding keeps those failures safe (a
  degraded answer, never an invented one), and v0.5 evals measure how often
  they happen.
- Synthetic data can teach template artifacts instead of language. Mitigated
  by template diversity, held-out templates, and evals.
