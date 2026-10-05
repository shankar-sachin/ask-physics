"""Pydantic models for every payload in the pipeline and every data entry.

These models do structural validation only (types, required fields, value
ranges). Semantic checks that need SymPy or Pint (expressions parse, units
are valid, symbols are covered) live in ``askphysics.data.loader`` so this
module stays dependency-free. See ``docs/DATA_SCHEMA.md``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_PATTERN = r"^[a-z][a-z0-9_]*$"

Domain = Literal[
    "kinematics",
    "dynamics",
    "energy",
    "momentum",
    "gravitation",
    "electromagnetism",
    "thermodynamics",
    "fluids",
    "waves",
    "optics",
    "modern",
]
Category = Literal["standard", "fermi", "out_of_scope"]
Origin = Literal["given", "constant", "assumption"]
Difficulty = Literal["intro", "intermediate", "advanced"]
AnswerStatus = Literal["answered", "degraded", "refused"]
ConfidenceLabel = Literal["low", "medium", "high"]


class _Model(BaseModel):
    """Base model: unknown fields are errors, so typos in JSON data fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- data entries


class Variable(_Model):
    symbol: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    name: str
    unit: str
    description: str
    typical_range: tuple[float, float] | None = None

    @field_validator("typical_range")
    @classmethod
    def _range_ordered(cls, v: tuple[float, float] | None) -> tuple[float, float] | None:
        if v is not None and v[0] > v[1]:
            raise ValueError(f"typical_range low {v[0]} is greater than high {v[1]}")
        return v


class Equation(_Model):
    id: str = Field(pattern=ID_PATTERN)
    name: str
    latex: str
    sympy_expr: str
    variables: list[Variable] = Field(min_length=1)
    domain: Domain
    assumptions: list[str]
    validity_conditions: list[str]
    tags: list[str] = Field(min_length=1)
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)
    confidence_in_entry: float = Field(ge=0.0, le=1.0)

    @field_validator("sympy_expr")
    @classmethod
    def _single_equals(cls, v: str) -> str:
        if v.count("=") != 1:
            raise ValueError("sympy_expr must contain exactly one '='")
        return v

    @model_validator(mode="after")
    def _unique_symbols(self) -> Equation:
        symbols = [var.symbol for var in self.variables]
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"duplicate variable symbols in {self.id}")
        return self

    def variable(self, symbol: str) -> Variable:
        """Return the variable with this symbol, or raise ``KeyError``."""
        for var in self.variables:
            if var.symbol == symbol:
                return var
        raise KeyError(f"{symbol!r} is not a variable of {self.id}")


class QuantityValue(_Model):
    value: float
    unit: str


class KnownValue(_Model):
    symbol: str
    value: float
    unit: str
    origin: Origin


class WorkedExample(_Model):
    id: str = Field(pattern=ID_PATTERN)
    problem_text: str
    known_values: list[KnownValue]
    unknowns: list[str] = Field(min_length=1)
    equations_used: list[str] = Field(min_length=1)
    solution_steps: list[str] = Field(min_length=1)
    final_answer: QuantityValue
    difficulty: Difficulty
    tags: list[str]
    source: str = Field(min_length=1)
    license: str = Field(min_length=1)


class Constant(_Model):
    name: str = Field(pattern=ID_PATTERN)
    symbol: str
    value: float
    unit: str
    uncertainty: float = Field(ge=0.0)
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _uncertainty_smaller_than_value(self) -> Constant:
        if self.uncertainty >= abs(self.value):
            raise ValueError(f"uncertainty of {self.name} is not smaller than its value")
        return self


class FermiAssumption(_Model):
    quantity: str = Field(pattern=ID_PATTERN)
    default_value: float
    unit: str
    low: float
    high: float
    rationale: str
    source: str = Field(min_length=1)

    @model_validator(mode="after")
    def _bounds(self) -> FermiAssumption:
        if not 0 < self.low <= self.default_value <= self.high:
            raise ValueError(f"{self.quantity}: need 0 < low <= default_value <= high")
        if self.high / self.low > 1e4:
            raise ValueError(f"{self.quantity}: high/low exceeds 1e4; split the assumption")
        return self


# --------------------------------------------------------------------------- pipeline payloads


class Question(_Model):
    text: str = Field(min_length=1)


class Classification(_Model):
    category: Category
    reasoning: str
    domains: list[Domain] = Field(default_factory=list)
    closest_answerable: str | None = None


class ScoredEquation(_Model):
    equation: Equation
    score: float = Field(ge=0.0, le=1.0)


class ScoredExample(_Model):
    example: WorkedExample
    score: float = Field(ge=0.0, le=1.0)


class RetrievalResult(_Model):
    query: str
    equations: list[ScoredEquation] = Field(default_factory=list)
    examples: list[ScoredExample] = Field(default_factory=list)

    @property
    def equation_ids(self) -> list[str]:
        return [hit.equation.id for hit in self.equations]

    def score_for(self, equation_id: str) -> float:
        """Retrieval score for an equation id, or 0.0 if it was not retrieved."""
        return next((h.score for h in self.equations if h.equation.id == equation_id), 0.0)


class Plan(_Model):
    target: str
    unknowns: list[str] = Field(min_length=1)
    known_values: list[KnownValue]
    equation_ids: list[str]
    assumptions: list[str]
    strategy: str

    @model_validator(mode="after")
    def _target_is_unknown(self) -> Plan:
        if self.target not in self.unknowns:
            raise ValueError(f"target {self.target!r} must be listed in unknowns")
        known = {k.symbol for k in self.known_values}
        overlap = known & set(self.unknowns)
        if overlap:
            raise ValueError(f"symbols cannot be both known and unknown: {sorted(overlap)}")
        return self


class ComputeResult(_Model):
    target: str
    value: float
    unit: str
    symbolic_solution: str
    substitutions: dict[str, str]
    notes: list[str] = Field(default_factory=list)


class SanityReport(_Model):
    dimensions_ok: bool
    magnitude_ok: bool | None
    limit_cases_checked: bool = False
    issues: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.dimensions_ok and self.magnitude_ok is not False


class EquationRef(_Model):
    id: str
    name: str


class Confidence(_Model):
    label: ConfidenceLabel
    score: float = Field(ge=0.0, le=1.0)


class ValueRange(_Model):
    low: float
    high: float


class Answer(_Model):
    question: str
    status: AnswerStatus
    category: Category
    final_value: float | None = None
    unit: str | None = None
    value_range: ValueRange | None = None
    equations_used: list[EquationRef] = Field(default_factory=list)
    inputs: list[KnownValue] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    confidence: Confidence
    caveats: list[str] = Field(default_factory=list)
    explanation: str
    redirect: str | None = None

    @model_validator(mode="after")
    def _value_has_unit(self) -> Answer:
        if (self.final_value is None) != (self.unit is None):
            raise ValueError("final_value and unit must be set together")
        if self.status == "refused" and self.final_value is not None:
            raise ValueError("a refused answer cannot carry a value")
        return self
