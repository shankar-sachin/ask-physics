Ask Physics reads questions the way a careful student would. A few habits get you better
answers.

## Give every value a unit

"Dropped from 20 m" works. "Dropped from 20" doesn't say whether that is metres, feet, or
storeys. Common spellings work (`m/s`, `km/h`, `mph`, `kilograms`, `ohms`), and a number
can be digits or words before a unit ("an eight kilogram ball"). Temperatures in Celsius
and Fahrenheit aren't supported yet; they arrive in v0.4.

## Say which situation you mean

Some equations share every variable and differ only in meaning. If you don't say which,
the answer is flagged as ambiguous with low confidence:

| Say | Not just |
|---|---|
| "two resistors **in series**" or "**in parallel**" | "two resistors" |
| "the **orbital** speed" or "the **escape** speed" | "the speed it needs" |

Each equation's page lists its look-alikes (for example,
[series resistors](series_resistors) and [parallel resistors](parallel_resistors)).

## Label values that could be confused

When a question has two masses or two speeds, say which is which: "the first cart, 2 kg,
moves at 3 m/s; the second, 5 kg, is at rest". With no labels, the first value stated is
taken as the first of a pair. Labels like `R1 = 40 ohm`, "the emissivity is 0.017", and
"110 m for the distance to the object" all work.

## Ask for one thing

"What is its final speed?" or "find the time" tells it what to solve for. "How fast",
"how far", "what height", and "how long does it take" work too.

## Every value counts

If you give a value, the answer is expected to use it. When no equation uses something you
stated, the answer still comes back, but flagged with "the question gives X, but the plan
doesn't use it", because the extra value might mean it picked the wrong equation.

## What it can't do yet

- Problems that need two or more equations chained together (v0.4).
- Vectors, Celsius and Fahrenheit, and answers that need a numerical solver.
- Research-level physics, now or at v1.0.
