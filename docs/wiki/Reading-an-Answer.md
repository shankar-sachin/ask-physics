Every answer comes with what you need to check it yourself.

## The status

| Card says | Meaning |
|---|---|
| **ANSWERED** | Every stage worked. The number is shown with its unit. |
| **PARTIAL** | A stage couldn't finish, so no number is shown. The card says which stage stopped and why, instead of guessing. |
| **REFUSED** | Not a physics question ("how much does the color blue weigh?"). You get the reason and, when there is one, a physics question you could ask instead. |

## The parts of an answer

- **The value and unit.** Computed by [Noether](Noether), never by a language model.
- **Equations used.** Each one by name and id, with a page in [Equations](Equations) giving
  its source and license.
- **Inputs.** Every value plugged in, and where it came from: *given* (you wrote it),
  *constant* (from [Constants](Constants)), or *assumption* (for example, "released from
  rest" means an initial speed of 0).
- **Assumptions.** Everything the answer depends on that you didn't say: no air
  resistance, constant acceleration, and so on.
- **Caveats.** Anything a check didn't like.
- **Confidence.** A label and a score from 0 to 1.
- **Models.** Which [Fermi model](The-Fermi-Models) handled each stage, and how many plans
  were tried.

## Confidence

The score comes from a fixed, published formula, never from a model grading itself:

```
a     = 1 / (1 + 0.25 * number_of_assumptions)
score = 0.35 * retrieval + 0.30 * dimensions + 0.20 * magnitude + 0.15 * a
```

*High* is 0.75 or more, *medium* 0.45 or more, *low* below that. Some results are capped:

| Cap | When |
|---|---|
| 0.2 | The units don't work out, the result is negative where it can't be (a negative mass or resistance), or it is 0 only because the plan assumed a 0. |
| 0.4 | The question states a value the plan didn't use, or the question fits two look-alike equations and doesn't say which (series or parallel resistors). |
| 0.6 | A Fermi estimate: the inputs are educated guesses. |

A low-confidence answer comes with caveats saying why. Take it as "probably, but check
this part". An impossible result (a negative resistance, say) is never shown at all: the
answer becomes PARTIAL and says what went wrong.
