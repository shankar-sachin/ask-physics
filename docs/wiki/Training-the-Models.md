You don't need to train anything to use Ask Physics once weights ship with v0.4. This page
is for building the models yourself. The full runbook, with timings and results, is
[`docs/TRAINING.md`](https://github.com/shankar-sachin/ask-physics/blob/main/docs/TRAINING.md).

## Set up

```bash
git clone https://github.com/shankar-sachin/ask-physics.git
cd ask-physics
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
askphysics validate-data
```

An Apple Silicon Mac (MPS) or an NVIDIA GPU (CUDA) is picked automatically; the CPU works
for the tiny test model.

## A one-minute smoke test

```bash
askphysics model build-data --out build/smoke --examples 5000 --workers 4
askphysics model train-tokenizer --data build/smoke --out build/smoke-tok.json --vocab-size 512
askphysics model train --model fermi-luna-1 --data build/smoke --tokenizer build/smoke-tok.json --steps 300
```

## The real thing

1. Build the data: `askphysics model build-data --out build/data --examples 1000000 --workers 10`.
2. Build the prose corpus: `python scripts/build_corpus.py`. It downloads the textbooks
   and books (see [Corpus and licenses](Corpus-and-Licenses)) and takes a while; it is
   quiet while it downloads books, and safe to rerun.
3. Train the tokenizer, then tellus, solem, and celeste, with the exact commands in
   `docs/TRAINING.md`.
4. Score each one: `askphysics model eval --model fermi-solem-1 --examples 3000`.

Trained weights land in `~/.cache/askphysics/models/<model>/`, and `askphysics ask` uses
them automatically.
