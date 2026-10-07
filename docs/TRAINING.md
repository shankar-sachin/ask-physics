# Training the Fermi models

The runbook for training `fermi-tellus-1`, `fermi-solem-1`, and maybe
`fermi-celeste-1` on the maintainer's M5 Pro (48 GB). Users never do this;
they download the weights we ship (ADR-012). Design background is in
[`MODELS.md`](MODELS.md).

Numbers marked *estimate* get replaced with measurements as we go.

## 1. Set up

```bash
git clone https://github.com/shankar-sachin/ask-physics && cd ask-physics
python3 -m venv .venv && source .venv/bin/activate
make install                      # pip install -e ".[dev]"; macOS torch includes MPS
python -c "import torch; print(torch.backends.mps.is_available())"   # want: True
make check                        # lint, typecheck, tests
askphysics validate-data          # the seed data the factory builds from
```

## 2. Smoke test (about a minute)

Train the tiny test model end to end, so a broken setup fails in seconds, not
hours:

```bash
askphysics model build-data --out build/smoke --examples 5000 --workers 4
askphysics model train-tokenizer --data build/smoke --out build/smoke-tok.json --vocab-size 512
askphysics model train --model fermi-luna-1 --data build/smoke --tokenizer build/smoke-tok.json \
  --out build/smoke-luna --steps 300
```

## 3. Build the real dataset

```bash
askphysics model build-data --out build/data --examples 1000000 --workers 10
askphysics model train-tokenizer --data build/data --out build/tokenizer.json --vocab-size 8192 \
  --prose third_party/openstax-physics/prose.jsonl
```

- `--prose` adds about 1,650 paragraphs of real physics writing from OpenStax
  *Physics* (CC BY 4.0, ADR-016) to what the tokenizer learns from, so
  ordinary English words get tokens of their own. It is already in the
  repository; `scripts/extract_openstax.py` rebuilds it from the pinned
  OpenStax source. Anything trained on it carries the credit in
  `third_party/openstax-physics/ATTRIBUTION.md`.

- About 4 minutes and 1 GB *(estimate: ~600 examples/s per worker)*. Every
  example is solved by SymPy; the factory drops anything it can't solve and
  anything too close to an eval question.
- An example averages about 260 tokens *(measured on a 6,000-example sample
  with a small vocabulary; a full-size tokenizer makes them shorter)*.
- The text is template-generated, so the tokenizer may stop short of 8,192
  merges. That's fine; record the size it reaches.
- Rebuild both whenever the factory's templates change. A tokenizer trained
  on old data can't spell the new phrasings efficiently, and the old val split
  measures the old held-out templates.

## 4. Measure throughput

```bash
askphysics model train --model fermi-tellus-1 --steps 200 --batch-size 32 --out build/probe/tellus
askphysics model train --model fermi-solem-1  --steps 100 --batch-size 32 --out build/probe/solem
```

Read tokens per second off the progress bar and write them into the table in
`MODELS.md`. Each step at batch 32 sees about 8,000 tokens.

## 5. Train

Budgets are about 20 tokens per parameter. Steps ≈ tokens ÷ (batch × 260):

| Model | Params | Token budget | Batch | Steps *(estimate)* |
|-------|--------|--------------|-------|--------------------|
| tellus | 3.2M | ~60M | 32 | ~7,500 |
| solem | 29.9M | ~600M | 32 | ~72,000 |
| celeste | 119.6M | ~2.4B | 32 | ~290,000 |

These budgets assume varied data. On template data a model stops improving
on held-out phrasings long before its budget runs out (see the results
below), so train in short runs and extend only while validation loss keeps
falling.

```bash
caffeinate -dims askphysics model train --model fermi-tellus-1 \
  --data build/data --tokenizer build/tokenizer.json --steps 1500 --batch-size 32
```

- `caffeinate` keeps the Mac awake. Weights land in
  `~/.cache/askphysics/models/<model>/`, where `askphysics` looks for them.
- Runs checkpoint as they go. If one stops, rerun the same command with
  `--resume`. A fresh run (no `--resume`) starts a fresh `metrics.jsonl`.
- Train tellus first and check it before spending hours on solem. Train
  celeste only if solem's held-out results leave room for it to help
  (open question Q16).

### The prose corpus (ADR-017)

The 118,000-word *Physics* book is small. The full corpus is about 20 million
words: 52 OpenStax textbooks under CC BY 4.0 plus public-domain books. Build it
once (about 15 minutes for the first run, which downloads the books):

```bash
python scripts/build_corpus.py
askphysics model train-tokenizer --data build/data --out build/tokenizer.json --vocab-size 8192 \
  --prose build/corpus/prose.jsonl
```

Then pass `--prose build/corpus/prose.jsonl` to `model train` instead of the
single book. At 20M words, 200 prose steps cover about 3% of it, so a corpus
run wants far more prose steps, for example `--prose-steps 3000 --prose-share
0.1`, which is a job for a fast GPU (see the celeste plan in the roadmap).

### Real prose first (solem and celeste)

Every task example is written by our templates, so on its own a model never
reads real English, and its prose shows it. Give the models that write
(solem, celeste) a language-modeling stage on the OpenStax text first:

```bash
caffeinate -dims askphysics model train --model fermi-solem-1 \
  --data build/data --tokenizer build/tokenizer.json --steps 3000 --batch-size 32 \
  --prose third_party/openstax-physics/prose.jsonl --prose-steps 200 --prose-share 0.05
```

- `--prose-steps 200` trains on the prose alone first: about 160k tokens, so
  roughly six passes at batch 32. `--prose-share 0.05` keeps one batch in
  twenty on prose afterwards, so it doesn't fade.
- The eval lines gain `val_loss_prose` (held-out paragraphs), reported apart
  from `val_loss` so task numbers stay comparable with earlier runs. If it
  climbs while training loss falls, the model is memorizing the book: lower
  `--prose-steps`.
- tellus only classifies, so it can skip this stage. It can still use the
  tokenizer trained with `--prose`; each model saves its own copy anyway.

## 6. Check the results

```bash
askphysics model info                                  # params, size, last validation loss
askphysics model eval --model fermi-tellus-1 --examples 200   # task accuracy, a few minutes
```

- The validation split uses held-out templates, so a falling validation loss
  means the model is generalizing, not memorizing phrasings.
- Watch for validation loss rising while training loss keeps falling: that's
  overfitting, and the answer is more varied data, not more steps.
- Each evaluation logs `val_loss` plus `val_loss_classify`, `val_loss_plan`,
  and `val_loss_explain`, so you can see which task is overfitting. The CLI
  prints the final per-task numbers when training ends.
- Loss includes free prose the model can never predict exactly (which of
  eight reasoning sentences, which explanation wording), so it has a floor.
  `model eval` scores what the pipeline needs instead: it decodes held-out
  questions with the constrained decoder and reports the share with the right
  category and the share of plans that compute the right answer ("valid
  plan"). Failures print with what was expected; everything lands in
  `eval.json` next to the weights.
- The eval also runs each plan through the router the way `askphysics ask`
  does (`--attempts`, default 5: retry until Noether accepts a plan) and
  sorts the answers three ways: right, wrong but flagged (degraded, or a
  sanity check caught it, so the answer card says low confidence), and
  **confidently wrong** (every check passed, the answer is still wrong).
  Confidently wrong is the trust number: the target is at most 1 in 1,000.
  Those cases print first, because each one is a hole a rule or more data
  should close.
- Send the `model info` table, `metrics.jsonl`, and the `model eval` table;
  they feed the model cards and the v0.4 exit criteria (90% valid plans on
  unseen templates, 0.1% or less confidently wrong).

## 7. Results so far

### fermi-tellus-1 on the original templates (v0.2 factory)

1,500 steps at batch 32, 10.7K target tokens/s on the M5 Pro, 7 minutes.
Train loss 0.014, val loss flat at about 1.17 from step 150 on. With 14
generic templates and 7 scenarios the model memorized every phrasing within
150 steps; the tokenizer only reached 1,455 tokens.

### fermi-tellus-1 on composed templates (PR #26)

1,500 steps at batch 32, 10.4K target tokens/s, 7.5 minutes. Tokenizer:
2,557 tokens.

| Step | Train | Val | Classify | Plan | Explain |
|------|-------|-----|----------|------|---------|
| 150 | 0.56 | 0.828 | 1.035 | 0.520 | 1.715 |
| 300 | 0.24 | 0.584 | 0.918 | 0.264 | 1.396 |
| 600 | 0.16 | 0.555 | 1.126 | 0.184 | 1.320 |
| 900 | 0.06 | 0.445 | 1.116 | 0.099 | 1.028 |
| 1,200 | 0.05 | 0.441 | 1.151 | 0.079 | 1.045 |
| 1,500 | 0.05 | 0.430 | 1.148 | 0.061 | 1.049 |

- **Plan** kept improving the whole run (0.52 to 0.061). That's the task
  that matters most, since plans are what Noether computes from.
- **Explain** levelled off near 1.05 by step 900. Much of that is the
  irreducible choice among explanation and assumption wordings.
- **Classify** bottomed out at step 300 (0.918) and then crept up to 1.15:
  the reasoning and redirect sentences of held-out Fermi and out-of-scope
  templates are text the model never saw, and it grew overconfident in the
  training ones. The category itself may still be right; `model eval`
  measures that.

### Next

1. Run `askphysics model eval` on tellus. If category accuracy and the valid
   plan rate are high, tellus is good enough for its job (reading every
   question) and the classify loss is just prose.
2. Probe solem's throughput (section 4), then train it in short runs, about
   3,000 steps first, watching per-task val loss. Extend only while val
   keeps falling.
3. If classify accuracy is weak: more Fermi and out-of-scope templates, and
   reasoning sentences tied to the question (its domain or the reason it is
   out of scope) rather than picked at random.
