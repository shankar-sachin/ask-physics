import pytest

from askphysics.pretty import pretty_equation, pretty_number, pretty_symbol, pretty_unit


@pytest.mark.parametrize(
    ("unit", "pretty"),
    [
        ("meter / second", "m/s"),
        ("meter / second ** 2", "m/s²"),
        ("joule", "J"),
        ("kilogram * meter / second", "kg·m/s"),
        ("not a unit at all", "not a unit at all"),
    ],
)
def test_pretty_unit(unit: str, pretty: str) -> None:
    assert pretty_unit(unit) == pretty


@pytest.mark.parametrize(
    ("value", "pretty"),
    [
        (19.8057, "19.8057"),
        (0, "0"),
        (437400.0, "437400"),
        (6.6743e-11, "6.674 × 10⁻¹¹"),
        (2.4e8, "2.4 × 10⁸"),
    ],
)
def test_pretty_number(value: float, pretty: str) -> None:
    assert pretty_number(value) == pretty


def test_pretty_symbol() -> None:
    assert pretty_symbol("v0") == "v₀"
    assert pretty_symbol("m12") == "m₁₂"
    assert pretty_symbol("KE") == "KE"


def test_pretty_equation_renders_math() -> None:
    assert pretty_equation("v**2 = v0**2 + 2*a*d") == "v² = v₀² + 2·a·d"
    assert pretty_equation("F = G*m1*m2/r**2") == "F = G·m₁·m₂/r²"
    assert pretty_equation("x = y; z") == "x = y; z"  # unparseable input falls back to raw
