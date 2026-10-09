Ask Physics is MIT licensed. Everything it learns from or ships is either written by the
project or licensed so that it can be used, and credit travels with it.

## The prose corpus

About 20 million words of English, built locally by `scripts/build_corpus.py` and never
committed (ADR-017):

- **OpenStax textbooks** (33 repositories, 52 books: physics, biology, chemistry,
  history, economics, writing, and more), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). OpenStax moved later
  versions to CC BY-NC-SA in March 2026. A CC BY license can't be taken back from a
  version already published under it, so each book is read at the last commit from
  before the change, and the build checks both the license file and the book's own
  metadata at that commit and stops if either isn't CC BY 4.0.
- **Public-domain books** from Project Gutenberg: only books where everyone credited died
  in or before 1955, sciences first, then other non-fiction, then literature. Project
  Gutenberg's header, footer, and trademark are removed.

The build keeps prose only (no figures, tables, exercises, or display math), drops
repeated paragraphs and reference-list entries, and drops every paragraph with a slur or
a swear word. The word list is stored only as hashes.

Each build writes `ATTRIBUTION.md` (every source and its license), the CC BY license
text, and a lock file of exactly what went in. Any model trained on the corpus carries
the same attribution on its model card.

## Equations and constants

Every equation entry was written by the project, with a source cited for checking (mostly
OpenStax, as a reference) and its own license field, almost always MIT. Constants cite
CODATA and NIST. See [Equations](Equations) and [Constants](Constants).

## Everything else

Fonts, imagery, and software dependencies keep their own licenses, listed in
[`THIRD_PARTY_LICENSES.md`](https://github.com/shankar-sachin/ask-physics/blob/main/THIRD_PARTY_LICENSES.md).
