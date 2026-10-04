"""Exception taxonomy.

Every exception the pipeline is expected to handle derives from
``AskPhysicsError``. Pipeline stages catch these and turn them into degraded
answers; anything else is a bug and is allowed to propagate.

    AskPhysicsError
    ├── ConfigError
    ├── DataValidationError
    ├── LLMError
    │   └── LLMResponseFormatError
    ├── OutOfScopeError
    ├── RetrievalEmptyError
    ├── PlanValidationError
    ├── SolverError
    ├── UnitError
    │   ├── UnitParseError
    │   └── UnitMismatchError
    └── SanityCheckError
"""

from __future__ import annotations

from collections.abc import Sequence


class AskPhysicsError(Exception):
    """Base class for all expected Ask Physics failures."""


class ConfigError(AskPhysicsError):
    """Invalid or missing configuration (bad env var, unknown provider, missing extra)."""


class DataValidationError(AskPhysicsError):
    """Seed data failed schema or semantic validation.

    Carries every problem found, not just the first, so ``validate-data`` can
    report them all in one run.
    """

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        summary = f"{len(self.problems)} data validation problem(s)"
        detail = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{summary}:\n{detail}" if detail else summary)


class LLMError(AskPhysicsError):
    """The LLM provider failed (network, auth, refusal, rate limit)."""


class LLMResponseFormatError(LLMError):
    """The LLM returned output that does not match the requested schema."""


class OutOfScopeError(AskPhysicsError):
    """The question has no physical meaning, needs unknowable data, or is too advanced."""


class RetrievalEmptyError(AskPhysicsError):
    """Retrieval found no relevant equations."""


class PlanValidationError(AskPhysicsError):
    """The plan cites unknown equations, uses invalid units, or contradicts itself."""


class SolverError(AskPhysicsError):
    """SymPy could not solve or evaluate the equation for the requested unknown."""


class UnitError(AskPhysicsError):
    """Base class for unit problems."""


class UnitParseError(UnitError):
    """A unit string is not a valid Pint unit."""


class UnitMismatchError(UnitError):
    """Quantities have incompatible dimensions (for example, converting meters to seconds)."""


class SanityCheckError(AskPhysicsError):
    """A sanity check could not be performed (as opposed to performed and failed)."""
