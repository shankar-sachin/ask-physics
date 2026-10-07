"""Which Fermi model handles which stage (ADR-010), decided without loading any weights.

tellus classifies every question. solem plans and explains, with up to
``plan_attempts`` tries, and celeste gets ``escalations`` more tries after that.
A missing model is skipped: with only tellus installed, tellus does everything. celeste
can also be *available*, published but not yet downloaded: its client fetches the weights
the first time a question needs its try (ADR-012).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from askphysics.errors import ConfigError

TELLUS = "fermi-tellus-1"
SOLEM = "fermi-solem-1"
CELESTE = "fermi-celeste-1"

# Preference order per role. fermi-luna-1 is a test model: it is only used when forced.
CLASSIFIERS = (TELLUS, SOLEM, CELESTE)
PLANNERS = (SOLEM, TELLUS, CELESTE)


@dataclass(frozen=True)
class Route:
    """Model names per stage. ``plan`` has one entry per attempt, in order."""

    classify: str
    plan: tuple[str, ...]
    explain: str

    @property
    def models(self) -> list[str]:
        """Every model the route uses, each once, in first-use order."""
        return list(dict.fromkeys((self.classify, *self.plan, self.explain)))


def can_route(installed: Sequence[str], forced: str | None = None) -> bool:
    """Whether ``plan_route`` has a model to work with (what ``auto`` checks)."""
    if forced is not None:
        return forced in installed
    return any(m in installed for m in CLASSIFIERS)


def plan_route(
    installed: Sequence[str],
    *,
    forced: str | None = None,
    attempts: int = 5,
    escalations: int = 1,
    available: Sequence[str] = (),
) -> Route:
    """The route for the installed models; ``available`` ones may escalate too.

    Raises:
        ConfigError: no usable model is installed, or ``forced`` isn't installed.
    """
    have = set(installed)
    hint = "train one (docs/TRAINING.md) or set ASKPHYSICS_MODEL_DIR"
    if forced is not None:
        if forced not in have:
            raise ConfigError(f"{forced} is not installed; {hint}")
        return Route(classify=forced, plan=(forced,) * attempts, explain=forced)
    classifier = next((m for m in CLASSIFIERS if m in have), None)
    planner = next((m for m in PLANNERS if m in have), None)
    if classifier is None or planner is None:
        raise ConfigError(f"no Fermi models are installed; {hint}")
    can_rescue = CELESTE in have or CELESTE in available
    rescue = (CELESTE,) * escalations if can_rescue and planner != CELESTE else ()
    return Route(classify=classifier, plan=(planner,) * attempts + rescue, explain=planner)
