Contributions are welcome. Start with
[`CONTRIBUTING.md`](https://github.com/shankar-sachin/ask-physics/blob/main/CONTRIBUTING.md)
and [`PLAN.md`](https://github.com/shankar-sachin/ask-physics/blob/main/PLAN.md).

## The rules that never bend

1. The models never do arithmetic. Every number in an answer comes from SymPy and Pint.
2. Every number has units.
3. Every equation has a source and a license.
4. Data never enters the database without passing `askphysics validate-data`.
5. Models never write expressions; plans name equation ids from the reviewed database.
6. Tests never need trained weights or a GPU.

## Adding an equation

Equations live in
[`src/askphysics/data/equations.json`](https://github.com/shankar-sachin/ask-physics/blob/main/src/askphysics/data/equations.json).
Each needs an id, a name, LaTeX, a SymPy expression with exactly one `=`, every variable
with its unit, description, and usual range, assumptions, validity conditions, tags (the
words that find it), a source, and a license. Run `askphysics validate-data`, and the
equation gets its own page in this wiki when the change reaches main.

## Before a pull request

```bash
make lint
make typecheck
make test
askphysics validate-data
```

All four must pass. Branch from `main` with a prefix (`feat/`, `fix/`, `docs/`, `data/`,
...) and fill in the pull request template.

## Editing this wiki

Wiki pages live in
[`docs/wiki/`](https://github.com/shankar-sachin/ask-physics/tree/main/docs/wiki), and
the equation pages are generated from the database. Edits made directly on the wiki are
overwritten by the next publish, so change the repository instead.
