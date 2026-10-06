import pytest

from askphysics.data.loader import DataStore
from askphysics.lm.reading import (
    ask_spans,
    asked_symbols,
    asked_variables,
    labelled_quantities,
    symbol_locks,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("the source frequency fs is 758 Hz", [("fs", "758", "Hz")]),
        ("v = 43 m/s and di: 1.8 m", [("v", "43", "m/s"), ("di", "1.8", "m")]),
        # Any word can sit where a label goes; the decoder keeps only real symbols.
        ("the image is 5 cm tall, 1.8 m (di) away", [("image", "5", "cm"), ("di", "1.8", "m")]),
        ("mu equals 0.3", [("mu", "0.3", "dimensionless")]),
        ("the speed was 6.3e+20 m/s", [("speed", "6.3e+20", "m/s")]),
        ("a 5 kg ball moving at 3 m/s", []),
    ],
)
def test_labels(text: str, expected: list[tuple[str, str, str]]) -> None:
    assert labelled_quantities(text) == expected


def test_a_symbol_with_two_values_is_not_locked(store: DataStore) -> None:
    eq = store.equations["doppler_approaching"]
    assert symbol_locks("fs is 758 Hz and fs is 800 Hz, v = 43 m/s", eq.variables) == {
        "v": ("43", "m/s")
    }
    assert symbol_locks(
        "Ok so do = 110 m, what do I do?", store.equations["lens_magnification"].variables
    ) == {"do": ("110", "m")}


@pytest.mark.parametrize(
    ("text", "spans"),
    [
        ("Work out the object height if hi is 5 m.", ["the object height"]),
        ("How do I get the first area when v1 is 3 m/s?", ["the first area"]),
        ("What is the mass 1 when m2 = 27 kg?", ["the mass 1"]),
        ("Find the speed for 3 kg moving", ["the speed for"]),
        ("v = 3 m/s, t = 2 s. Solve for a.", ["a"]),
        ("v = 3 m/s and t = 2 s. Final speed?", ["Final speed"]),
        ("Help me find the time to reach the top given v0 = 3 m/s", ["the time to reach the top"]),
    ],
)
def test_ask_spans(text: str, spans: list[str]) -> None:
    assert spans[0] in ask_spans(text)


@pytest.mark.parametrize(
    ("equation", "text", "asked"),
    [
        ("lens_magnification", "Work out the object height if hi = 2 m.", {"ho"}),
        ("continuity_equation", "How do I get the first area when v1 = 3 m/s?", {"A1"}),
        (
            "parallel_resistors",
            "R = 1.3 ohm, R2 = 10.1 ohm: what would the resistance 1 be?",
            {"R1"},
        ),
        ("gravitational_pe", "Lifting it 5.3 m took 207 J. What is its mass?", {"m"}),
        ("doppler_approaching", "f = 800 Hz, fs = 700 Hz. Solve for vs.", {"vs"}),
        # "do" is a symbol and an English word; only a whole-ask symbol counts.
        ("lens_magnification", "What image distance do you get when hi: 2 m?", {"di"}),
        ("work_constant_force", "Lifting it 5.3 m took 207 J. What is its mass?", set()),
    ],
)
def test_asked_symbols(store: DataStore, equation: str, text: str, asked: set[str]) -> None:
    assert asked_symbols(text, store.equations[equation].variables) == asked


def test_longer_names_win_across_equations(store: DataStore) -> None:
    variables = [
        v for eid in ("kin_v_at", "vertical_launch_time") for v in store.equations[eid].variables
    ]
    text = "Determine the value of the time to reach the top given v0 = 16 m/s."
    picked = asked_variables(text, variables)
    assert picked == [store.equations["vertical_launch_time"].variable("t")]
