"""Phrasing templates for the data factory (``lm/factory.py``).

Every template has a stable id. Ids ending in ``_h`` are **held out**: they
never appear in training data, only in the validation split, so validation
measures how well a model handles phrasings it has never seen.

Rules for writing templates:
- Never copy or paraphrase anything from ``evals/`` (``docs/EVALS.md``).
- Free text the models must reproduce (reasoning, assumptions, strategy) may
  only contain digits that appear in the question, because constrained
  decoding forbids any other number.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Template:
    id: str
    text: str

    @property
    def held_out(self) -> bool:
        return self.id.endswith("_h")


# Realistic question values per unit (low, high), sampled log-uniformly.
FRIENDLY_RANGES: dict[str, tuple[float, float]] = {
    "m/s": (1.0, 120.0),
    "m/s^2": (0.5, 30.0),
    "s": (1.0, 120.0),
    "m": (0.5, 800.0),
    "kg": (0.05, 3000.0),
    "N": (1.0, 20000.0),
    "J": (1.0, 200000.0),
    "kg*m/s": (0.1, 50000.0),
    "V": (1.5, 240.0),
    "A": (0.01, 20.0),
    "ohm": (1.0, 10000.0),
    "Pa": (1000.0, 1000000.0),
    "m^3": (0.001, 10.0),
    "mol": (0.1, 50.0),
    "K": (200.0, 900.0),
}

# Alternative units a question may use; values are converted, the plan copies the unit as written.
ALT_UNITS: dict[str, tuple[str, ...]] = {
    "m": ("m", "m", "cm", "km", "ft"),
    "m/s": ("m/s", "m/s", "km/h"),
    "kg": ("kg", "kg", "g"),
    "s": ("s", "s", "min"),
    "N": ("N", "N", "kN"),
    "J": ("J", "J", "kJ"),
    "Pa": ("Pa", "kPa"),
    "m^3": ("m^3", "L"),
    "A": ("A", "mA"),
}

# Generic question templates. {target} is a variable name, {knowns} a natural-language list.
GENERIC: tuple[Template, ...] = (
    Template("gen_01", "What is the {target} when {knowns}?"),
    Template("gen_02", "Find the {target} given that {knowns}."),
    Template("gen_03", "If {knowns}, what is the {target}?"),
    Template("gen_04", "Calculate the {target}. We know that {knowns}."),
    Template("gen_05", "Suppose {knowns}. Determine the {target}."),
    Template("gen_06", "Given {knowns}, how large is the {target}?"),
    Template("gen_07", "{Knowns}. What {target} does that give?"),
    Template("gen_08", "Work out the {target} if {knowns}."),
    Template("gen_09", "I have a setup where {knowns}. Can you tell me the {target}?"),
    Template("gen_10_h", "With {knowns}, compute the {target}."),
    Template("gen_11", "what's the {target} if {knowns}"),
    Template("gen_12", "Quick one: {knowns}. {Target}?"),
    Template("gen_13_h", "Assuming {knowns}, what would the {target} be?"),
    Template("gen_14", "Please solve for the {target}, where {knowns}."),
)

# Scenario templates for specific equations: (equation id, template, forced knowns).
# Forced knowns are (symbol, value or "g" for standard gravity, unit, origin).
SCENARIOS: tuple[tuple[str, Template, str, tuple[tuple[str, str, str, str], ...]], ...] = (
    (
        "kin_v_squared",
        Template(
            "drop_speed_01", "A {object} is dropped from {d}. How fast is it going when it lands?"
        ),
        "v",
        (("v0", "0", "m/s", "assumption"), ("a", "g", "m/s^2", "constant")),
    ),
    (
        "kin_v_squared",
        Template("drop_speed_02_h", "How fast does a {object} hit the floor after falling {d}?"),
        "v",
        (("v0", "0", "m/s", "assumption"), ("a", "g", "m/s^2", "constant")),
    ),
    (
        "kin_x_at",
        Template("drop_dist_01", "How far does a {object} fall in {t} after being released?"),
        "x",
        (("v0", "0", "m/s", "assumption"), ("a", "g", "m/s^2", "constant")),
    ),
    (
        "kin_v_at",
        Template(
            "start_rest_01",
            "A {vehicle} starts from rest and accelerates at {a} for {t}. What is its final speed?",
        ),
        "v",
        (("v0", "0", "m/s", "assumption"),),
    ),
    (
        "weight",
        Template("weight_earth_01", "How much does a {m} {object} weigh on Earth?"),
        "W",
        (("g", "g", "m/s^2", "constant"),),
    ),
    (
        "gravitational_pe",
        Template(
            "lift_pe_01", "How much potential energy does a {m} {object} gain when lifted {h}?"
        ),
        "U",
        (("g", "g", "m/s^2", "constant"),),
    ),
    (
        "ohms_law",
        Template(
            "battery_01",
            "What current flows when a {V} battery is connected across a {R} resistor?",
        ),
        "I",
        (),
    ),
)

SCENARIO_ASSUMPTIONS: dict[str, tuple[str, ...]] = {
    "drop_speed_01": ("Released from rest", "Air resistance is negligible"),
    "drop_speed_02_h": ("Released from rest", "Air resistance is negligible"),
    "drop_dist_01": ("Released from rest", "Air resistance is negligible"),
    "start_rest_01": ("Starts from rest", "Acceleration stays constant"),
    "weight_earth_01": ("Standard gravity at the surface",),
    "lift_pe_01": ("Standard gravity near the surface",),
    "battery_01": ("Ideal battery with no internal resistance",),
}

OBJECTS = ("ball", "rock", "phone", "book", "watermelon", "brick", "coin", "toy car", "wrench")
VEHICLES = ("car", "bike", "train", "scooter", "sled", "cart", "boat")

STANDARD_REASONING = (
    "All the needed values are given, so this is a standard {domain} problem.",
    "The question gives the inputs directly; it is {domain}.",
    "A well-posed {domain} question with every value stated or a standard constant.",
)

# Fermi questions (classification only until the Fermi engine lands in v0.7).
FERMI: tuple[Template, ...] = (
    Template("fermi_01", "How many {small} would it take to fill a {container}?"),
    Template("fermi_02", "How many {small} would you need to stack to reach the top of a {tall}?"),
    Template("fermi_03", "How many {small} weigh as much as a {heavy}?"),
    Template("fermi_04_h", "Roughly how many {small} could fit inside a {container}?"),
    Template("fermi_05", "If you lined up {small} end to end, how many would span a {long}?"),
    Template("fermi_06", "How much energy would it take to lift a {heavy} to the top of a {tall}?"),
)
FERMI_SLOTS: dict[str, tuple[str, ...]] = {
    "small": ("tennis balls", "marbles", "ping pong balls", "golf balls", "pennies", "dice"),
    "container": ("school bus", "bathtub", "swimming pool", "classroom", "shipping container"),
    "tall": ("skyscraper", "lighthouse", "redwood tree", "radio tower"),
    "heavy": ("school bus", "elephant", "pickup truck", "grand piano", "blue whale"),
    "long": ("football field", "city block", "suspension bridge", "runway"),
}
FERMI_REASONING = (
    "Physically meaningful, but it needs estimated everyday quantities.",
    "A Fermi estimate: the answer depends on assumed sizes and masses.",
    "Answerable with order-of-magnitude assumptions rather than given values.",
)

# Out-of-scope questions with a reason and a redirect.
OUT_OF_SCOPE: tuple[tuple[Template, str, str], ...] = (
    (
        Template("oos_mass_01", "What is the mass of {abstract}?"),
        "Category error: {abstract} is an idea, not an object with mass.",
        "What is the mass of a typical paper notebook?",
    ),
    (
        Template("oos_temp_01", "What temperature is {emotion}?"),
        "Category error: {emotion} is a feeling, not a physical system with a temperature.",
        "What is the normal temperature of the human body?",
    ),
    (
        Template("oos_sound_01_h", "How loud is the {abstract2}?"),
        "Category error: the {abstract2} has no physical form to make sound.",
        "How loud is a typical conversation in decibels?",
    ),
    (
        Template(
            "oos_future_01",
            "What will the exact air temperature be in {city} on a day a thousand years from now?",
        ),
        "Needs unknowable future data; physics cannot predict weather that far ahead.",
        "What is the average yearly temperature in {city} today?",
    ),
    (
        Template("oos_beyond_01", "What lies outside the observable universe?"),
        "Unknowable: no signal from beyond the observable universe can ever reach us.",
        "How big is the observable universe?",
    ),
    (
        Template("oos_offtopic_01", "What is the best {food} topping?"),
        "Not a physics question; it is a matter of taste.",
        "How much energy is in a typical slice of {food}?",
    ),
    (
        Template("oos_research_01", "Derive the full quantum theory of gravity."),
        "Research-level physics with no settled answer; out of scope.",
        "What is the gravitational force between two people standing a meter apart?",
    ),
)
OOS_SLOTS: dict[str, tuple[str, ...]] = {
    "abstract": ("a promise", "Tuesday", "a good idea", "justice", "a rumor"),
    "abstract2": ("letter Q", "concept of zero", "word silence", "idea of blue sky"),
    "emotion": ("jealousy", "boredom", "nostalgia", "pride"),
    "city": ("Lisbon", "Nairobi", "Osaka", "Denver"),
    "food": ("pizza", "taco", "bagel", "burger"),
}

EXPLAIN: tuple[Template, ...] = (
    Template("exp_01", "Using [{id}], the {target} comes out to {value} {unit}.{assume}"),
    Template("exp_02", "From [{id}] ({name}), {target} = {value} {unit}.{assume}"),
    Template("exp_03", "Solving [{id}] for the {target} gives {value} {unit}.{assume}"),
    Template("exp_04_h", "The {target} is {value} {unit}, from [{id}].{assume}"),
)
