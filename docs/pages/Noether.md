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
- **Chains equations.** When a question needs two steps (the acceleration first, then the
  distance), Noether solves the equations in order, feeding each value into the next, and
  the answer lists every value found along the way. A symbol shared between equations must
  mean the same kind of quantity in both, so a weight is never used as a work.
- **Handles Celsius and Fahrenheit.** They're converted to kelvin before the math, by what
  the value means: "water at 20 °C" is 293.15 K, but "heated by 20 °C" is a change of 20 K.
- **Checks the database.** Every equation's units must balance before it is accepted;
  `askphysics-dev validate-data` checks all of them.

## Safety

Equations come only from the reviewed database. Text from a question or a model is never
parsed as math, so nothing typed into a question can run as code.

## What it can't do yet

- Vectors, and equations with no closed-form solution.
- Fermi range propagation, for estimates with low and high bounds (v0.7).
