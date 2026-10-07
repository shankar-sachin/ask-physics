Noether is the math engine, named for Emmy Noether. It is built on
[SymPy](https://www.sympy.org) for algebra and [Pint](https://pint.readthedocs.io) for
units, and it does every piece of math in Ask Physics.

## What it does

- **Rearranges.** Each equation is stored once (for example `v**2 = v0**2 + 2*a*d`).
  Noether solves it symbolically for whichever variable is asked for, so one entry answers
  questions about any of its variables.
- **Keeps units.** Values go in as Pint quantities, `20 m` or `55 mph`, and units are
  carried through the algebra. If they don't combine into the unit the answer needs, the
  calculation fails instead of returning a wrong number.
- **Picks the right root.** When the algebra gives several answers (a square root has two
  signs), it keeps the real ones and prefers the smallest non-negative one. The others are
  listed as caveats ("discarded root -19.8 m/s").
- **Checks the database.** Every equation's units must balance before it is accepted;
  `askphysics validate-data` checks all of them.

## Safety

Equations come only from the reviewed database. Text from a question or a model is never
parsed as math, so nothing typed into a question can run as code.

## What it can't do yet

- Chain two or more equations (v0.4).
- Celsius and Fahrenheit, whose zero points need converting before the math (v0.4).
- Vectors, and equations with no closed-form solution.
- Fermi range propagation, for estimates with low and high bounds (v0.7).
