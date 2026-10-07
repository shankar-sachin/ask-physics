# The prose corpus

About 20 million words of real English for the Fermi models to learn from
(ADR-017). It is built locally and never committed:

```bash
python scripts/build_corpus.py          # writes build/corpus/ (about 130 MB)
```

- **52 OpenStax textbooks** (about 6.8M words) under CC BY 4.0, each read at
  the last commit of its repository before OpenStax relicensed it to CC
  BY-NC-SA: sciences, mathematics, history, economics, social sciences,
  philosophy, writing, business, and more. The build stops if a pinned
  commit's `LICENSE` or a book's metadata says anything else.
- **Public-domain books** from Project Gutenberg: the sciences first, then
  expository non-fiction (history, philosophy, essays, and the like), then
  literature, until the target is reached. Everyone credited must have died in
  or before 1955.

`sources.json` is the manifest: the pinned commits and the Gutenberg selection
rules. The build writes `build/corpus/prose.jsonl`, `ATTRIBUTION.md`,
`LICENSE-CC-BY-4.0.txt`, and `sources.lock.json` (exact commits, book ids, and
file hashes). Keep the attribution with the corpus, and copy it into the model
card of any weights trained on it.

The first run downloads a few hundred books with a pause between each (about
15 minutes); later runs reuse `build/corpus-cache/`. `--offline` uses only the
cache, and `--target-words` changes the size.
