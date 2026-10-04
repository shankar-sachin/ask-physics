import pytest

from askphysics.data.loader import DataStore
from askphysics.solver.fermi import AssumptionTable, order_of_magnitude, propagate_range


def test_table_loaded_from_seed_data(store: DataStore) -> None:
    table = store.fermi
    assert isinstance(table, AssumptionTable)
    assert len(table) >= 8
    assert "rubber_duck_mass" in table
    assert table.get("freight_train_mass").unit == "kg"


def test_default_quantity_has_units(store: DataStore) -> None:
    q = store.fermi.default_quantity("rubber_duck_mass")
    assert q.to("g").magnitude == pytest.approx(25.0)


def test_missing_assumption_lists_known_ones(store: DataStore) -> None:
    with pytest.raises(KeyError, match="rubber_duck_mass"):
        store.fermi.get("unicorn_mass")


def test_from_list_round_trip(store: DataStore) -> None:
    table = AssumptionTable.from_list(list(store.fermi))
    assert set(table.entries) == set(store.fermi.entries)


@pytest.mark.parametrize(("value", "expected"), [(2.4e8, 8), (4e8, 9), (3.4e-14, -13), (-50, 2)])
def test_order_of_magnitude(value: float, expected: int) -> None:
    assert order_of_magnitude(value) == expected


def test_order_of_magnitude_of_zero() -> None:
    with pytest.raises(ValueError):
        order_of_magnitude(0)


def test_propagate_range_is_a_stub(store: DataStore) -> None:
    with pytest.raises(NotImplementedError):
        propagate_range("m_train / m_duck", dict(store.fermi.entries), seed=1)
