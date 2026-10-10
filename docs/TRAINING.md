# Training the Fermi models

The runbook for training `fermi-tellus-1`, `fermi-solem-1`, and maybe
`fermi-celeste-1` on the maintainer's M5 Pro (48 GB). Users never do this;
they download the weights we ship (ADR-012). Design background is in
[`MODELS.md`](MODELS.md).

Numbers marked *estimate* get replaced with measurements as we go.

## The short way: scripts

Every step below is also a script that keeps the Mac awake (`caffeinate`) and only runs
a step when the one before it worked. Each prints its options with `--help`, and
`DRY_RUN=1` shows the commands without running them.

| Script | What it does |
|---|---|
| `sh scripts/corpus.sh` | Builds the prose corpus; safe to rerun after an interruption |
| `sh scripts/train.sh fermi-solem-1` | Backs up the installed model, builds the data and tokenizer if missing, scores the old model, trains (prose corpus first for solem and celeste), scores the new one, and compares the two |
| `sh scripts/train.sh fermi-solem-1 --resume` | Continues a run that stopped |
| `sh scripts/eval.sh fermi-solem-1 --against DIR` | Scores a model, and another copy on the same questions, side by side |
| `sh scripts/models.sh list` / `backup` / `restore` | Keeps every replaced model in `~/askphysics-backup`; a restore backs up first |
| `sh scripts/release_weights.sh models-v0.4.0` | Packages each installed model and scores the packaged copy (see `RELEASING.md`) |

`python3 scripts/compare_evals.py OLD/eval.json NEW/eval.json` compares any two reports.

`train.sh` starts with a one-minute speed check (`askphysics model bench`). It times a few
training steps on each device and precision this machine has, and prints the tokens per
second for each and how many hours the run will take. If the projection is far longer than
a night, stop and find out why rather than leaving it running. On a Mac, train plugged in
with the lid open: on battery macOS throttles the GPU and sleeps even under `caffeinate`.
`--precision fp32` (or `bf16`) trains in the format the check found fastest, and
`--no-bench` skips the check. On a Mac the run trains with MLX (below), and the check, which
times torch only, is skipped.

The rest of this page is what those scripts run, step by step.

## 1. Set up

```bash
git clone https://github.com/shankar-sachin/ask-physics && cd ask-physics
python3 -m venv .venv && source .venv/bin/activate
make install                      # pip install -e ".[dev]"; on an arm64 Mac it also installs mlx
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
- `build-data` also adds 180 real problems from OpenStax *Physics*
  (`third_party/openstax-physics/questions.jsonl`) as classify examples, five
  times each. The 49 in the real-question eval are never among them.
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

### Backends and memory (ADR-019)

On an Apple Silicon Mac, `model train` trains with MLX by default: it is faster, and it stays
inside a memory limit where torch's MPS backend grew past 40 GB and swapped. The first line of
output names the backend (`... steps · mlx · → dir`). Nothing else changes: the weights,
`metrics.jsonl`, and `training_summary.json` are the same, and `ask`, `model eval`, and the
website still run on torch.

- `--backend torch` trains on torch instead. `--backend mlx` fails with a message if mlx is
  missing (`pip install -e ".[mlx]"` on the Mac). `askphysics model backend` prints the choice.
- A run resumes only on the backend that started it. Resuming a torch run with MLX, or the
  reverse, stops with a message that names the backend to use.
- `--mlx-memory-gb N` sets the memory limit MLX works to stay under (default 70% of system
  RAM). It is a guideline: when memory and swap run out, MLX raises an error instead of
  swapping the machine.
- `--mlx-cache-gb N` caps the memory MLX keeps for reuse between steps (default 4).
- `--checkpoint-blocks` recomputes each block's activations in the backward pass instead of
  storing them. Attention's (heads, T, T) matrix is the largest activation: at 1024 tokens a
  layer holds about 0.5 to 1 GB of it. Use the flag when a model's activations don't fit, for
  example celeste at batch 16; it costs extra compute. It works on torch too, where it only
  applies to training (evaluation never recomputes).
- `scripts/train.sh` takes the same choices as `--backend`, `--checkpoint-blocks`, and
  `--device`, and skips the torch speed check when the run is MLX.

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
single book. A corpus run wants far more prose steps, for example `--steps 5000
--prose-steps 2000 --prose-share 0.1`: 2000 prose-only steps (about two passes
over the corpus at batch 32), then 3000 on the tasks with one batch in ten still
prose. `--prose-steps` must be fewer than `--steps`, because the prose-only steps
come first; the CLI refuses a run that would never reach the tasks. Measured on an
M5 Pro, solem's prose steps take about 23 seconds each (passages are long), so
2000 of them take about 13 hours; task steps are much shorter. A fast GPU helps
(see the celeste plan in the roadmap). A run checkpoints every quarter of
`--steps` (at most every 2000), and `--resume` with new `--steps` and
`--prose-steps` continues from the last checkpoint.

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

### celeste (after solem)

celeste (119.6M parameters) exists to rescue the questions solem gets wrong:
when all of solem's plan attempts fail, celeste gets one more (ADR-010). Train
it only once solem is trained and scored, and keep it only if it earns its
size (open question Q16): the v0.4 bar is that it rescues at least a third of
solem's misses.

1. Score solem and look at what it misses:

   ```bash
   sh scripts/eval.sh fermi-solem-1 --examples 500
   ```

   If solem is already right on nearly everything, celeste has little to
   rescue; stop here.

2. Check celeste's speed and memory at full width (about two minutes):

   ```bash
   askphysics model bench --model fermi-celeste-1 --width 1024 --batch-size 32
   ```

   It has four times solem's parameters, so expect roughly a quarter of solem's
   tokens per second. If a setting fails or its GPU memory is more than about
   30 GB, try `--batch-size 16` and train with `--batch-size 16 --grad-accum 2`:
   the same 32 sequences per step, in two halves, so less memory at the same
   quality.

3. Train it, plugged in with the lid open, on the same data and tokenizer as
   solem (2000 prose steps, then 3000 on the tasks):

   ```bash
   sh scripts/train.sh fermi-celeste-1 --device mps --precision bf16 --no-eval
   ```

   Add `--batch-size 16 --grad-accum 2` if step 2 said so. The speed check
   before training prints the projected hours; if it is far longer than a
   night, stop and send the bench table.

4. Measure the rescue rate: solem answers, and every question it misses goes to
   celeste.

   ```bash
   sh scripts/eval.sh fermi-solem-1 --examples 500 --rescue-with fermi-celeste-1
   ```

   The table's "rescued" row is the number that decides whether celeste ships.

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
- Then it asks 49 real textbook questions (OpenStax *Physics*, with gold
  plans the project wrote) through the whole pipeline, the model doing every
  stage, and prints a second table: classified standard, right equations
  retrieved, right answer, flagged, confidently wrong, and where each miss
  stopped. The factory never wrote these questions, so this is the first
  number on how real people phrase things. Retrieval caps it at 85.7% for
  now. `--real` points at another file; it is skipped when the file is
  missing (outside a checkout).
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
