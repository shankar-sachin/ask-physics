import pytest

from askphysics.errors import UnitMismatchError, UnitParseError
from askphysics.solver.units import (
    check_dimensions,
    convert,
    format_quantity,
    is_valid_unit,
    parse_quantity,
    quantity,
)


def test_convert_km_per_hour_to_m_per_s() -> None:
    q = convert(quantity(36, "km/h"), "m/s")
    assert q.magnitude == pytest.approx(10.0)


def test_parse_quantity() -> None:
    q = parse_quantity("9.8 m/s^2")
    assert q.magnitude == pytest.approx(9.8)
    assert check_dimensions(q, "m/s^2")


def test_deliberate_dimension_mismatch_raises() -> None:
    with pytest.raises(UnitMismatchError):
        convert(quantity(20, "m"), "s")


def test_check_dimensions_detects_mismatch() -> None:
    assert check_dimensions(quantity(1, "J"), "kg*m^2/s^2")
    assert not check_dimensions(quantity(1, "J"), "N")


@pytest.mark.parametrize("bad", ["furlongs_per_fortnightzzz", "", "   "])
def test_invalid_units(bad: str) -> None:
    assert not is_valid_unit(bad)
    with pytest.raises(UnitParseError):
        quantity(1.0, bad)


def test_parse_quantity_rejects_bare_numbers_and_garbage() -> None:
    with pytest.raises(UnitParseError):
        parse_quantity("42")
    with pytest.raises(UnitParseError):
        parse_quantity("20 blorps")


def test_convert_rejects_invalid_target_unit() -> None:
    with pytest.raises(UnitParseError):
        convert(quantity(1, "m"), "blorps")


def test_format_quantity() -> None:
    assert format_quantity(quantity(19.80570624, "m/s")) == "19.8057 meter / second"


@pytest.mark.parametrize("text", ["N/", "m*", "GeV/", "Air*", "K^0", "m^^2", "/s"])
def test_dangling_operators_are_invalid_not_a_crash(text: str) -> None:
    # Pint's parser raises AssertionError on these; real text ("5 N/ ...") contains them.
    assert not is_valid_unit(text)
    with pytest.raises(UnitParseError):
        quantity(1.0, text)
