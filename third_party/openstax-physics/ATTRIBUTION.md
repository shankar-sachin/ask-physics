# Attribution: OpenStax *Physics*

`prose.jsonl` in this directory contains text from:

> **Physics** (2020), by Fatih Gozuacik, Denise Pattison, and Catherine Tabor
> (authors as OpenStax's collection metadata credits them).
> Published by OpenStax, Rice University. Access for free at
> <https://openstax.org/details/books/physics>.
> Source files: <https://github.com/openstax/osbooks-physics>, commit
> `dfdfd7a5356ecdd42e504de3df50d9153e33ea49`.

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0),
<https://creativecommons.org/licenses/by/4.0/>. The full license text is in
[`LICENSE`](LICENSE). This text is **not** covered by Ask Physics' MIT license.

**Changes made by the Ask Physics project** (`scripts/extract_openstax.py`):

- Kept only body prose paragraphs. Removed all figures, captions, media,
  tables, exercises, display equations, the preface, and teacher-support
  material.
- Wrote inline math as plain text (for example `v = d / t`), replaced
  automatic references to figures and tables with "the figure" and "the
  table", and skipped paragraphs that embed exercises or begin mid-sentence.
- Converted typographic characters to ASCII (curly quotes, dashes, Greek
  letters as names, accents removed).

The text is used to train the Fermi language models (ADR-016). Any model
trained on it carries this attribution in its model card. OpenStax and Rice
University do not endorse Ask Physics, and the OpenStax name and logo are
used here only to give credit.
