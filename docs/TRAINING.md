# Training the Fermi models

The runbook for v0.3: training `fermi-tellus-1`, `fermi-solem-1`, and maybe
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
askphysics model train-tokenizer --data build/data --out build/tokenizer.json --vocab-size 8192
```

- About 4 minutes and 1 GB *(estimate: ~600 examples/s per worker)*. Every
  example is solved by SymPy; the factory drops anything it can't solve and
  anything too close to an eval question.
- An example averages about 260 tokens *(measured on a 6,000-example sample
  with a small vocabulary; a full-size tokenizer makes them shorter)*.
- The text is template-generated, so the tokenizer may stop short of 8,192
  merges. That's fine; record the size it reaches.

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

```bash
caffeinate -dims askphysics model train --model fermi-tellus-1 \
  --data build/data --tokenizer build/tokenizer.json --steps 7500 --batch-size 32
```

- `caffeinate` keeps the Mac awake. Weights land in
  `~/.cache/askphysics/models/<model>/`, where `askphysics` looks for them.
- Runs checkpoint as they go. If one stops, rerun the same command with
  `--resume`.
- Train tellus first and check it before spending hours on solem. Train
  celeste only if solem's held-out results leave room for it to help
  (open question Q16).

## 6. Check the results

```bash
askphysics model info             # params, size, last validation loss
```

- The validation split uses held-out templates, so a falling validation loss
  means the model is generalizing, not memorizing phrasings.
- Watch for validation loss rising while training loss keeps falling: that's
  overfitting, and the answer is more varied data, not more steps.
- Send the `model info` table and `metrics.jsonl` from the model directory;
  they feed the model cards and the v0.3 exit criteria (90% valid plans on
  unseen templates).
