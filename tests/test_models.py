import pytest
from pydantic import ValidationError

from askphysics.models import (
    Answer,
    Confidence,
    Constant,
    Equation,
    FermiAssumption,
    KnownValue,
    Plan,
    SanityReport,
    Variable,
)


def _var(symbol: str, unit: str = "m") -> Variable:
    return Variable(symbol=symbol, name=symbol, unit=unit, description=symbol)


def _equation(**overrides: object) -> Equation:
    data: dict[str, object] = {
        "id": "test_eq",
        "name": "test",
        "latex": "x = y",
        "sympy_expr": "x = y",
        "variables": [_var("x"), _var("y")],
        "domain": "kinematics",
        "assumptions": [],
        "validity_conditions": [],
        "tags": ["test"],
        "source": "test",
        "license": "MIT",
        "confidence_in_entry": 1.0,
    }
    data.update(overrides)
    return Equation.model_validate(data)


def test_equation_requires_single_equals() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        _equation(sympy_expr="x == y")


def test_equation_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        _equation(variables=[_var("x"), _var("x")])


def test_equation_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _equation(colour="blue")


def test_equation_variable_lookup() -> None:
    eq = _equation()
    assert eq.variable("x").symbol == "x"
    with pytest.raises(KeyError):
        eq.variable("z")


def test_typical_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        Variable(symbol="v", name="v", unit="m/s", description="v", typical_range=(10, 1))


def test_plan_target_must_be_unknown() -> None:
    with pytest.raises(ValidationError, match="target"):
        Plan(
            target="v",
            unknowns=["t"],
            known_values=[],
            equation_ids=["e"],
            assumptions=[],
            strategy="",
        )


def test_plan_symbol_cannot_be_known_and_unknown() -> None:
    known = KnownValue(symbol="v", value=1.0, unit="m/s", origin="given")
    with pytest.raises(ValidationError, match="both known and unknown"):
        Plan(
            target="v",
            unknowns=["v"],
            known_values=[known],
            equation_ids=["e"],
            assumptions=[],
            strategy="",
        )


def test_fermi_assumption_bounds() -> None:
    base = {"quantity": "q", "unit": "kg", "rationale": "r", "source": "s"}
    FermiAssumption.model_validate({**base, "default_value": 2, "low": 1, "high": 3})
    with pytest.raises(ValidationError):
        FermiAssumption.model_validate({**base, "default_value": 5, "low": 1, "high": 3})
    with pytest.raises(ValidationError, match="1e4"):
        FermiAssumption.model_validate({**base, "default_value": 2, "low": 1, "high": 1e5})


def test_constant_uncertainty_smaller_than_value() -> None:
    with pytest.raises(ValidationError):
        Constant(name="c", symbol="c", value=1.0, unit="m", uncertainty=2.0, source="s")


def test_answer_value_and_unit_go_together() -> None:
    conf = Confidence(label="low", score=0.0)
    with pytest.raises(ValidationError, match="together"):
        Answer(
            question="q",
            status="answered",
            category="standard",
            final_value=1.0,
            confidence=conf,
            explanation="e",
        )


def test_refused_answer_cannot_carry_value() -> None:
    conf = Confidence(label="low", score=0.0)
    with pytest.raises(ValidationError, match="refused"):
        Answer(
            question="q",
            status="refused",
            category="out_of_scope",
            final_value=1.0,
            unit="m",
            confidence=conf,
            explanation="e",
        )


def test_sanity_report_passed() -> None:
    assert SanityReport(dimensions_ok=True, magnitude_ok=None).passed
    assert not SanityReport(dimensions_ok=True, magnitude_ok=False).passed
    assert not SanityReport(dimensions_ok=False, magnitude_ok=True).passed
