# Data Sourcing

Where the equations, worked examples, constants, and Fermi assumptions come
from, what each source costs us, and the gate every entry has to pass.

**The rule:** no entry enters the database without passing schema
validation (`askphysics validate-data`) and a SymPy parse test. No
exceptions for "obviously correct" entries, bulk imports, or the
maintainer's own additions.

## Licensing baseline

The code is MIT. Data ships inside the package, so data licenses matter.

| License | Allowed | Obligation |
|---------|---------|------------|
| `MIT`, `CC0-1.0`, `public-domain` | Yes | None beyond keeping the `source` field |
| `CC-BY-4.0` | Yes | Attribution: listed in a generated `DATA_LICENSES.md` (v0.6) |
| `CC-BY-SA-4.0` | **Quarantined** | Share-alike may extend to the data files. Allowed only in a separate `data/cc-by-sa/` directory shipped as its own optional data package, never mixed into MIT files |
| `CC-BY-NC-*`, `CC-BY-ND-*`, proprietary, unknown | **No** | Validation fails |

Physics equations themselves are facts and are not copyrightable. What *is*
copyrightable is the expression around them: a textbook's worked example
text, its problem wording, its explanations. So equation entries written by
the project, with a textbook cited as the `source` for verification, are
`MIT`. Worked examples copied from a textbook carry the textbook's license.

---

## Sources

### 1. Hand-curated core set

- **What we get:** the 50 to 100 equations that cover most intro questions,
  written by maintainers with careful `assumptions`, `validity_conditions`,
  and `typical_range` values. This is the quality bar for everything else.
- **License risk:** none (`MIT`), provided entries are written rather than
  pasted.
- **Cleaning effort:** high per entry, low total. About 10 to 15 minutes per
  equation including review.
- **Recommendation:** **Do first, keep forever.** v0.1 seeds about 12; v0.4
  grows this to about 60 across the five target domains.

### 2. Open textbooks (OpenStax)

- **What we get:** OpenStax publishes several physics books, and **their
  licenses differ** (checked against each book's own `LICENSE` file and
  collection metadata in the `openstax/osbooks-*` repositories, October 2026):

  | Book | License | Can we train on it or ship its text? |
  |------|---------|------|
  | *Physics* (high school, 2020) | `CC-BY-4.0` | Yes, with attribution (ADR-016) |
  | *University Physics* volumes 1 to 3 | `CC-BY-NC-SA-4.0` | No: non-commercial and share-alike conflict with MIT |
  | *College Physics* | `CC-BY-NC-SA-4.0` | No, same reason |

  Earlier versions of this document said every OpenStax book was
  `CC-BY-4.0`. That was wrong for *University Physics* and *College
  Physics*, because OpenStax relicensed them (and most of its catalog) to
  `CC-BY-NC-SA-4.0` in March 2026. Versions published before that stay
  `CC-BY-4.0` (the license is irrevocable), so ADR-017 uses those, pinned to
  the last commit before each switch, for the prose corpus.
- **License risk:** low for *Physics* text, with attribution to OpenStax and
  its contributors. Some *Physics* artwork was provided through separate
  permissions, so we use **text only**: no figures, no captions. The
  OpenStax name and logo are trademarks and are never used to imply
  endorsement. Citing *University Physics* as a reference for a fact (an
  equation, a definition) is fine; copying or training on its text is not.
- **Cleaning effort:** medium. Problems are in HTML or CNXML with MathML.
  Converting to our schema means extracting known values and units by hand
  or with the data factory's extraction helpers plus verification. Answers are only given for some
  problems.
- **Recommendation:** ***Physics* (2020) is the source for real-phrasing
  questions and prose (ADR-016).** Any OpenStax book may be cited as a
  verification reference (the `source` field) for hand-curated equations,
  because the entries are written by the project and equations are facts.
  Every imported question records the book, chapter, and exercise, keeps its
  `CC-BY-4.0` license, and lives apart from our MIT data.

### 3. Wikipedia and Wikidata

- **What we get:** Wikidata has structured physics formulas (property P2534,
  "defining formula", with symbols linked to quantities via P416/P7235).
  Wikipedia has broad coverage, including obscure formulas.
- **License risk:** split. **Wikidata is `CC0-1.0`**, so it is safe.
  **Wikipedia text is `CC-BY-SA-4.0`**: share-alike obligations, so it goes
  to the quarantined directory or is used only as a verification reference.
- **Cleaning effort:** high. Wikidata formulas are in LaTeX/MathML with
  inconsistent symbol conventions. Units are linked through quantity items,
  not stated directly. Expect about 30% of formulas to need manual repair or
  rejection.
- **Recommendation:** **Use Wikidata for v0.6 bulk expansion** with an
  import script that converts formulas, auto-generates `variables` from the
  linked quantities, and routes anything failing validation to a review
  queue. Use Wikipedia only as a citation for verification.

### 4. NIST CODATA (constants)

- **What we get:** authoritative values and standard uncertainties for every
  fundamental constant; machine-readable ASCII table at physics.nist.gov.
- **License risk:** none. NIST data is a US government work (public domain
  in the US); cite the CODATA release year.
- **Cleaning effort:** low. One script, one table. Map the NIST names to our
  snake-case names and their unit strings to Pint units.
- **Recommendation:** **Authoritative source for `constants.json`.** Refresh
  when a new CODATA release lands (2022 is current; next expected around
  2026 to 2027). Defined constants (c, h, e, k_B, N_A) have uncertainty 0.

### 5. Synthetic examples (data factory, SymPy-verified)

- **What we get:** unlimited worked examples, targeted at gaps (unusual
  equation combinations, unit conversions, Fermi decompositions).
- **License risk:** none for licensing (templates are project-authored, `MIT`),
  medium for contamination: the factory must never read eval questions.
  No external model generates data (ADR-009).
- **Cleaning effort:** low per item, but needs a pipeline: render a problem
  from templates and sampled values, solve it with `solve_for`, keep only
  items Noether solves with consistent units, then
  human spot-check 10% of each batch.
- **Recommendation:** **v0.2 for Fermi model training data; v0.6 for worked
  examples in the database, after the solver handles chains (v0.4).**
  Tag every synthetic entry with `source: "synthetic:<generator-version>"`
  so it can be filtered or dropped wholesale. Cap synthetic examples at 50%
  of the database examples set, so retrieval does not learn the generator's
  style instead of physics.

### 6. Community contributions

- **What we get:** domain coverage the maintainers lack, plus corrections.
- **License risk:** medium. Contributors may paste from copyrighted books
  without realizing it. Contributions are accepted under the repo's MIT
  license (see `CONTRIBUTING.md`); the PR template requires a checkable
  `source`.
- **Cleaning effort:** medium; review load scales with volume.
- **Recommendation:** **Open from v0.2, for small PRs.** CI runs
  `validate-data`; reviewers check the source and wording. Bulk
  contributions (more than 20 entries) need an issue first.

---

## Data QA checklist

Every data PR, human or scripted, must pass:

- [ ] `askphysics validate-data` exits 0 (schema, SymPy parse, unit parse,
      symbol coverage, dimensional consistency, referential integrity).
- [ ] `source` is specific enough for a reviewer to verify (book plus
      chapter, URL plus access date, or CODATA year).
- [ ] `license` is in the allowed set; CC-BY-SA entries are in the
      quarantine directory only.
- [ ] For equations: `assumptions` and `validity_conditions` are non-empty
      and specific ("speeds far below c", not "valid in classical regime").
- [ ] For equations: `typical_range` is set on any variable that could be
      an answer, and the range is defensible.
- [ ] For worked examples: re-solving with the pipeline reproduces
      `final_answer` within 0.1% (enforced in v0.4).
- [ ] For Fermi assumptions: `rationale` explains `low` and `high`, not
      just the default.
- [ ] No near-duplicate of any question in `evals/` (enforced by a
      similarity check from v0.5; reviewer judgment before that).
- [ ] No duplicated entry: same expression under a different id. Search
      before adding.
- [ ] `confidence_in_entry` reflects the reviewer's honest opinion; below
      0.8 needs a comment explaining the doubt.
