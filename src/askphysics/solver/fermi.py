"""Fermi estimation support: the assumptions table (working) and range propagation (stub).

The table loader is real in v0.1 so the planner and data validation can use
it. Uncertainty propagation lands in v0.5 (``docs/ROADMAP.md``); the method
choice is open question Q8.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from askphysics.models import FermiAssumption, ValueRange
from askphysics.solver.units import Quantity, quantity


@dataclass(frozen=True)
class AssumptionTable:
    """Lookup over ``FermiAssumption`` entries, keyed by ``quantity``."""

    entries: Mapping[str, FermiAssumption]

    @classmethod
    def from_list(cls, assumptions: list[FermiAssumption]) -> AssumptionTable:
        return cls(entries={a.quantity: a for a in assumptions})

    def __contains__(self, name: object) -> bool:
        return name in self.entries

    def __iter__(self) -> Iterator[FermiAssumption]:
        return iter(self.entries.values())

    def __len__(self) -> int:
        return len(self.entries)

    def get(self, name: str) -> FermiAssumption:
        """Return the assumption named ``name``, or raise ``KeyError`` listing what exists."""
        try:
            return self.entries[name]
        except KeyError:
            known = ", ".join(sorted(self.entries))
            raise KeyError(f"no Fermi assumption {name!r}; known: {known}") from None

    def default_quantity(self, name: str) -> Quantity:
        """The default value of an assumption as a Pint quantity."""
        a = self.get(name)
        return quantity(a.default_value, a.unit)


def propagate_range(
    expression: str,
    assumptions: Mapping[str, FermiAssumption],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> ValueRange:
    """Propagate assumption ranges through an expression to get a plausible output range.

    Args:
        expression: right-hand side of the solved equation, in assumption names.
        assumptions: the assumptions the expression depends on.
        samples: Monte Carlo sample count.
        seed: RNG seed, derived from the question so results are reproducible.
    """
    # TODO: Sample each assumption log-uniformly between low and high with a seeded
    # numpy Generator, evaluate the lambdified expression on the samples, and return
    # the 5th and 95th percentiles. Cross-check with interval arithmetic on the bounds (v0.5, Q8).
    raise NotImplementedError("Fermi range propagation lands in v0.5")


def order_of_magnitude(value: float) -> int:
    """Power of ten nearest to ``value`` on a log scale (``2.4e8`` gives 8, ``4e8`` gives 9)."""
    if value == 0:
        raise ValueError("zero has no order of magnitude")
    return round(math.log10(abs(value)))
