Our own language models, written and trained from scratch in the repository. No
pretrained weights and no outside service. They are named for bodies that grow with them:
luna the Moon, tellus the Earth, solem the Sun, and celeste a quasar.

| Model | Parameters | Layers | Width | Heads | Job |
|---|---|---|---|---|---|
| `fermi-tellus-1` | ~3.2M | 6 | 160 | 5 | Classifies every question. Does everything when the bigger models aren't installed. |
| `fermi-solem-1` | ~29.9M | 8 | 512 | 8 | Plans the solution and writes the explanation. |
| `fermi-celeste-1` | ~119.6M | 16 | 768 | 12 | One more try when solem can't make a plan work. |
| `fermi-luna-1` | ~0.14M | 2 | 64 | 4 | Runs the test suite. Never ships. |

All four share one tokenizer (8,192 tokens, every digit its own token), so a number is
always copied digit by digit, and the decoder can restrict which numbers are possible.

## Architecture

A decoder-only transformer in PyTorch: tied embeddings, RMSNorm, rotary position
embeddings, SwiGLU, no biases. Training uses AdamW with a cosine schedule, and bf16 on
Apple Silicon and CUDA. Weights are stored as `safetensors`, never as pickles, because
loading a pickle can run arbitrary code.

## What they learn from

- **Generated problems.** A data factory turns every equation in the database into
  thousands of worded questions with gold plans, and Noether solves each one; anything it
  can't solve is dropped. Fermi and out-of-scope questions come from their own templates.
- **Real prose.** About 20 million words of English: OpenStax textbooks under CC BY 4.0
  and public-domain books. See [Corpus and licenses](Corpus-and-Licenses).

No outside model ever writes training data.

## Results

Held-out questions from templates the models never trained on (`askphysics model eval`,
3,000 classification and 3,000 planning questions each):

| Check | tellus | solem |
|---|---|---|
| Right category | 100.0% | |
| Valid plan, right answer | 98.9% | |
| Right answer after retries | 99.2% | 99.3% |
| Confidently wrong | 0.4% | 0.4% |

*Confidently wrong* means a wrong answer that passed every sanity check. The goal is
0.1% or less. Wrong answers that come back flagged or PARTIAL don't count against it,
because saying "not sure" is fine. Every confidently wrong answer in these runs has a fix
in the v0.4 decoder and checks, and the models will be scored again after they land.
Speed and real-question results get measured for the v0.4 model cards.

## Getting them

`askphysics model pull` downloads them, checked byte for byte against the manifest this
version pins; the installers run it for you. Each model comes with its model card
(`MODEL_CARD.md`: training, results, speed, credit) in its folder under
`~/.cache/askphysics/models/`. You can also [train your own](Training-the-Models).
