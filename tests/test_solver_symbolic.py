import pytest

from askphysics.data.loader import DataStore
from askphysics.errors import SolverError, UnitMismatchError
from askphysics.models import Equation, Variable
from askphysics.solver.symbolic import (
    check_dimensional_consistency,
    free_symbol_names,
    parse_equation,
    solve_for,
)
from askphysics.solver.units import quantity


def test_parse_equation_and_symbols() -> None:
    eq = parse_equation("v**2 = v0**2 + 2*a*d")
    assert {str(s) for s in eq.free_symbols} == {"v", "v0", "a", "d"}


def test_caret_means_power_and_single_letters_are_symbols() -> None:
    # I, E, N, S, Q are SymPy built-ins by default; here they must be plain symbols.
    assert free_symbol_names("V = I*R") == {"V", "I", "R"}
    assert free_symbol_names("E = m*c^2") == {"E", "m", "c"}
    assert free_symbol_names("N = S + Q") == {"N", "S", "Q"}


@pytest.mark.parametrize(
    "bad",
    ["__import__('os') = 1", "x = y.__class__", "x = y; z", "x == y", "x = y = z", "x"],
)
def test_parser_rejects_unsafe_or_malformed(bad: str) -> None:
    with pytest.raises(SolverError):
        parse_equation(bad)


def test_solve_free_fall_speed(store: DataStore) -> None:
    out = solve_for(
        store.equations["kin_v_squared"],
        "v",
        {"v0": quantity(0, "m/s"), "a": quantity(9.80665, "m/s^2"), "d": quantity(20, "m")},
    )
    assert out.value.magnitude == pytest.approx(19.8057, rel=1e-5)
    assert str(out.value.units) == "meter / second"
    assert out.symbolic_solution == "sqrt(2*a*d + v0**2)"
    assert any("Discarded root" in n for n in out.notes)


def test_solve_converts_input_units(store: DataStore) -> None:
    out = solve_for(
        store.equations["kin_v_squared"],
        "d",
        {"v0": quantity(0, "m/s"), "a": quantity(9.80665, "m/s^2"), "v": quantity(36, "km/h")},
    )
    assert out.value.magnitude == pytest.approx(5.0986, rel=1e-4)
    assert str(out.value.units) == "meter"


def test_solve_ideal_gas_pressure(store: DataStore) -> None:
    out = solve_for(
        store.equations["ideal_gas_law"],
        "P",
        {
            "V": quantity(0.05, "m^3"),
            "n": quantity(2, "mol"),
            "R": quantity(8.314462618, "J/(mol*K)"),
            "T": quantity(300, "K"),
        },
    )
    assert out.value.magnitude == pytest.approx(99773.55, rel=1e-6)


def test_missing_known_raises(store: DataStore) -> None:
    with pytest.raises(SolverError, match="missing"):
        solve_for(store.equations["kin_v_squared"], "v", {"v0": quantity(0, "m/s")})


def test_unknown_not_in_equation_raises(store: DataStore) -> None:
    with pytest.raises(SolverError, match="does not appear"):
        solve_for(store.equations["ohms_law"], "x", {})


def test_wrong_units_raise_mismatch(store: DataStore) -> None:
    with pytest.raises(UnitMismatchError):
        solve_for(
            store.equations["kin_v_squared"],
            "v",
            {"v0": quantity(0, "m/s"), "a": quantity(9.8, "m/s^2"), "d": quantity(20, "s")},
        )


def test_no_real_solution_raises(store: DataStore) -> None:
    # Decelerating to a stop before covering d: v^2 would be negative.
    with pytest.raises(SolverError, match="no real solution"):
        solve_for(
            store.equations["kin_v_squared"],
            "v",
            {"v0": quantity(1, "m/s"), "a": quantity(-10, "m/s^2"), "d": quantity(100, "m")},
        )


def test_dimensional_consistency(store: DataStore) -> None:
    assert all(check_dimensional_consistency(eq) for eq in store.equations.values())
    broken = Equation(
        id="broken",
        name="broken",
        latex="x = v + t",
        sympy_expr="x = v + t",
        variables=[
            Variable(symbol="x", name="x", unit="m", description="x"),
            Variable(symbol="v", name="v", unit="m/s", description="v"),
            Variable(symbol="t", name="t", unit="s", description="t"),
        ],
        domain="kinematics",
        assumptions=[],
        validity_conditions=[],
        tags=["x"],
        source="test",
        license="MIT",
        confidence_in_entry=0.0,
    )
    assert not check_dimensional_consistency(broken)
