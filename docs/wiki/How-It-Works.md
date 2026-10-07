Every question goes through six stages. Language models handle three of them, and none of
those three produces a number.

```
question -> 1 classify -> 2 retrieve -> 3 plan -> 4 compute -> 5 sanity check -> 6 explain
            (tellus)      (database)    (solem)   (Noether)    (Noether)          (solem)
```

1. **Classify.** Is this a standard problem, a Fermi estimate ("how many piano tuners
   are in Chicago?"), or not physics at all? Out-of-scope questions stop here with a reason
   and, when possible, a physics question to ask instead.
2. **Retrieve.** Search the [equation database](Equations) for equations whose tags match
   the question's words.
3. **Plan.** A [Fermi model](The-Fermi-Models) writes a plan: which equation, which
   variable to solve for, and which value goes where. It can only pick equations that
   retrieval found and numbers that appear in the question or the [constants](Constants),
   and every value's units must fit its variable. The decoder makes anything else
   impossible to write, not just unlikely.
4. **Compute.** [Noether](Noether) rearranges the equation symbolically, plugs in the
   values with their units, and converts the result to the right unit.
5. **Sanity check.** Do the dimensions work out? Is the result negative where it can't
   be, or zero only because the plan assumed a zero? Is it within a believable range? Did
   the plan use every value the question gave? Does the question fit a look-alike
   equation just as well? These checks set the [confidence](Reading-an-Answer#confidence).
6. **Explain.** The model writes a short explanation. Any digit it writes must be a number
   from the plan or the result.

## When a plan doesn't work

The router (ADR-010) tries again instead of showing a bad answer:

- A plan that fails validation, can't be computed, or gives an impossible result is
  thrown away, and solem tries again with the equations listed in a new order, up to 5
  times. Then celeste, the biggest model, gets one try.
- A plan that works but leaves one of the question's values unused is kept as a fallback
  while the router looks for one that uses everything. If none does, the fallback is shown,
  flagged.
- If nothing works, the answer is PARTIAL: it says which stage stopped and why. An
  impossible number is never shown.

## Why this design

A small model can't be trusted with arithmetic, and a big one can't either, not every
time. So the models only read and write; every number comes from symbolic algebra, every
unit is tracked by Pint, and every equation comes from a reviewed database with a source
and a license. The full design is in
[`ARCHITECTURE.md`](https://github.com/shankar-sachin/ask-physics/blob/main/docs/ARCHITECTURE.md)
and the decisions behind it in
[`DECISIONS.md`](https://github.com/shankar-sachin/ask-physics/blob/main/docs/DECISIONS.md).
