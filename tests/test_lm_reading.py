import pytest

from askphysics.data.loader import DataStore
from askphysics.lm.reading import (
    ask_spans,
    asked_symbols,
    asked_variables,
    contradicted,
    labelled_quantities,
    mentions,
    name_labels,
    own_tags,
    stated_givens,
    symbol_locks,
    twins,
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


@pytest.mark.parametrize(
    ("equation", "text", "expected"),
    [
        (
            "ideal_transformer",
            "Vs is 46 volts; primary turns: 61000; the turns on the secondary is 455, find Vp.",
            {("Np", "61000", "dimensionless"), ("Ns", "455", "dimensionless")},
        ),
        (
            "first_law",
            "The internal energy change is 0.0045 kJ. The thermal energy is 190 kilojoules.",
            {("dU", "0.0045", "kJ"), ("Q", "190", "kilojoules")},
        ),
        (
            "recoil",
            "the first mass comes out to 23 kg, the velocity 1 comes out to 66.3 km/h and "
            "the second velocity comes out to 35 m/s",
            {("m1", "23", "kg"), ("v1", "66.3", "km/h"), ("v2", "35", "m/s")},
        ),
        (
            "impulse_momentum",
            "the speed at the end comes out to 160 mph and the starting velocity comes out "
            "to 23.1 m/s",
            {("v", "160", "mph"), ("v0", "23.1", "m/s")},
        ),
        ("lens_magnification", "110 m for the distance to the object.", {("do", "110", "m")}),
        ("newton_second_law", "a mass of 5 kg", {("m", "5", "kg")}),
        # The name before the colon is the ask, not a label.
        (
            "heat_engine_work",
            "Find the heat out: 0.0052kJ for the work output.",
            {("W", "0.0052", "kJ")},
        ),
    ],
)
def test_name_labels(
    store: DataStore, equation: str, text: str, expected: set[tuple[str, str, str]]
) -> None:
    assert set(name_labels(text, store.equations[equation].variables)) == expected


def test_a_value_claimed_by_two_names_is_not_locked(store: DataStore) -> None:
    eq = store.equations["heat_engine_work"]
    assert name_labels("the heat out is 5 kJ for the work output", eq.variables) == []


@pytest.mark.parametrize(
    ("equation", "text", "asked"),
    [
        ("kin_v_squared", "How fast does a rock hit the floor after falling 7.4 m?", {"v"}),
        ("kin_v_at", "It ends at 9 m/s after 2 s. How fast was it at the start?", {"v0"}),
        ("kin_v_squared", "From what height would a plant fall to land at 16 m/s?", {"d"}),
        ("kin_x_at", "How far does a car travel in 4 s at 3 m/s^2?", {"x"}),
        ("kin_x_at", "How long does it take to cover 20 m from rest at 2 m/s^2?", {"t"}),
        ("kin_x_at", "How fast does a rock hit the floor after falling 7.4 m?", set()),
    ],
)
def test_idioms_name_a_kind_of_quantity(
    store: DataStore, equation: str, text: str, asked: set[str]
) -> None:
    assert asked_symbols(text, store.equations[equation].variables) == asked


def test_a_name_without_a_connector_labels_only_a_value_with_a_unit(store: DataStore) -> None:
    variables = store.equations["kinetic_energy"].variables
    assert name_labels("a ball with mass 3 kg moves at 4 m/s", variables) == [("m", "3", "kg")]
    # "mass 2" names the second mass; the 2 is not a mass.
    assert name_labels("the mass 2 moves at 4 m/s", variables) == []


def test_twins_share_every_variable(store: DataStore) -> None:
    series, parallel = store.equations["series_resistors"], store.equations["parallel_resistors"]
    assert twins(series, store.equations.values()) == [parallel]
    assert own_tags(series, parallel) == ["series"]
    assert twins(store.equations["kinetic_energy"], store.equations.values()) == []


@pytest.mark.parametrize(
    ("question", "series", "parallel"),
    [
        ("Two resistors in parallel have 3 ohm total.", True, False),
        ("A series circuit has 3 ohm total.", False, True),
        ("Two resistors have 3 ohm total.", False, False),  # says neither: rule neither out
        ("Series or parallel? 3 ohm total.", False, False),
    ],
)
def test_naming_one_twin_rules_out_the_other(
    store: DataStore, question: str, series: bool, parallel: bool
) -> None:
    eqs = store.equations
    assert contradicted(question, eqs["series_resistors"], eqs.values()) is series
    assert contradicted(question, eqs["parallel_resistors"], eqs.values()) is parallel


def test_mentions_matches_whole_words() -> None:
    assert mentions("Wired in Parallel.", ["parallel"])
    assert not mentions("unparalleled series", ["parallel"])
    assert mentions("the space station", ["space station"])


def test_stated_givens_include_labelled_bare_numbers(store: DataStore) -> None:
    q = "A panel at 350 K radiates 40 W. The emissivity comes out to 0.017. Resistor 2 is 5 ohm."
    assert stated_givens(q, store.equations.values()) == [
        ("350", "K"),
        ("40", "W"),
        ("5", "ohm"),
        ("0.017", "dimensionless"),  # "Resistor 2" is a name, not a value
    ]
    q = "Compute the resistance 2 for the total resistance is 940 ohm."
    assert stated_givens(q, store.equations.values()) == [("940", "ohm")]
    q = "600W for the radiated power. 0.036 for the emissivity. Find the area."
    assert ("0.036", "dimensionless") in stated_givens(q, store.equations.values())
    assert stated_givens("eps = 0.392 and T = 300 K", store.equations.values()) == [
        ("300", "K"),
        ("0.392", "dimensionless"),
    ]


@pytest.mark.parametrize(
    ("text", "asked"),
    [
        ("What speed did a go-kart start at if it reaches 160 mph after 3 s at 18 m/s^2?", {"v0"}),
        ("How fast did the scooter start out at, if it hits 16 kph after 4.7 s?", {"v0"}),
        # Starting from rest says how it began, not what is asked.
        ("How fast is it going after starting from rest at 2 m/s^2 for 3 s?", {"v"}),
    ],
)
def test_asking_how_it_started_means_the_initial_speed(
    store: DataStore, text: str, asked: set[str]
) -> None:
    assert asked_symbols(text, store.equations["kin_v_at"].variables) == asked
