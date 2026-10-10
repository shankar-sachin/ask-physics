"""Phrasing for the data factory (``lm/factory.py``).

Questions are composed from parts (a frame, a way of stating each known
value, variable synonyms, unit spellings, a preamble, a sign-off, and a
little casual noise) so a model sees thousands of distinct phrasings instead
of memorizing a few dozen sentences (risk R14).

Every frame and pattern has a stable id. Ids ending in ``_h`` are **held
out**: an example built with any held-out part goes to the validation split
only, so validation measures how well a model handles phrasings it has never
seen.

Rules for writing templates:
- Never copy or paraphrase anything from ``evals/`` (``docs/EVALS.md``).
- Free text the models must reproduce (reasoning, assumptions, strategy) may
  only contain digits that appear in the question, because constrained
  decoding forbids any other number. Preambles and sign-offs contain no
  digits at all.
- A unit written in a question must be one token Pint understands
  ("meters", not "meters per second"), right after its number, so the
  constrained decoder can copy it.
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


# --------------------------------------------------------------------------- values and units

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
    "Hz": (0.5, 20000.0),
    "rad/s": (0.1, 500.0),
    "rad/s^2": (0.1, 200.0),
    "N*m": (0.5, 5000.0),
    "N*s": (0.1, 5000.0),
    "N/m": (5.0, 50000.0),
    "W": (1.0, 100000.0),
    "W/m^2": (0.001, 2000.0),
    "kg*m^2": (0.01, 500.0),
    "kg*m^2/s": (0.1, 5000.0),
    "kg/m": (0.0005, 2.0),
    "kg/m^3": (0.5, 20000.0),
    "m^2": (0.0005, 50.0),
    "m^3/s": (0.0001, 100.0),
    "1/K": (1e-6, 1e-4),
    "1/m": (10.0, 10000.0),
    "C": (1e-9, 1e-4),
    "F": (1e-12, 1e-3),
    "H": (1e-6, 1.0),
    "J/(kg*K)": (100.0, 5000.0),
    "J/kg": (1e5, 3e6),
    "N/C": (1.0, 1e6),
    "T": (1e-5, 5.0),
    "V/m": (1.0, 1e6),
    "W/(m*K)": (0.02, 400.0),
    "ohm*m": (1e-8, 1e3),
}

# Alternative units a question may use; values are converted, the plan copies the unit as written.
# Offset temperature scales: degrees per kelvin, for writing a temperature *change*.
TEMPERATURE_SCALES = {"degC": 1.0, "degF": 1.8}

ALT_UNITS: dict[str, tuple[str, ...]] = {
    "m": ("m", "m", "m", "cm", "km", "ft"),
    "m/s": ("m/s", "m/s", "m/s", "km/h", "mph"),
    "kg": ("kg", "kg", "kg", "g", "lb"),
    "K": ("K", "K", "K", "K", "degC", "degC", "degF"),
    "s": ("s", "s", "s", "min"),
    "N": ("N", "N", "kN"),
    "J": ("J", "J", "kJ"),
    "Pa": ("Pa", "kPa", "kPa", "atm"),
    "m^3": ("m^3", "L", "L"),
    "A": ("A", "A", "mA"),
    "Hz": ("Hz", "Hz", "kHz"),
    "W": ("W", "W", "kW"),
    "m^2": ("m^2", "m^2", "cm^2"),
    "kg/m^3": ("kg/m^3", "kg/m^3", "g/cm^3"),
    # Questions can't write a unit starting with a digit; "1/m" is written "m^-1".
    "1/m": ("m^-1",),
    "1/K": ("K^-1",),
}

# Spelled-out forms of unit symbols. One token each, so the decoder can copy them.
UNIT_SPELLINGS: dict[str, tuple[str, ...]] = {
    "m": ("meters", "metres", "meters"),
    "cm": ("centimeters",),
    "km": ("kilometers",),
    "ft": ("feet",),
    "s": ("seconds", "seconds", "sec"),
    "min": ("minutes",),
    "km/h": ("kph",),
    "kg": ("kilograms",),
    "g": ("grams",),
    "lb": ("pounds",),
    "N": ("newtons",),
    "kN": ("kilonewtons",),
    "J": ("joules",),
    "kJ": ("kilojoules",),
    "V": ("volts",),
    "A": ("amperes", "amp"),
    "mA": ("milliamps",),
    "ohm": ("ohms", "ohms", "ohm"),
    "Pa": ("pascals",),
    "kPa": ("kilopascals",),
    "L": ("liters", "litres"),
    "mol": ("moles",),
    "K": ("kelvin",),
}

# --------------------------------------------------------------------------- variable names

# Ways to name each database variable in a question. The first is the database name.
VAR_SYNONYMS: dict[str, tuple[str, ...]] = {
    "final velocity": ("final velocity", "final speed", "speed at the end", "end speed"),
    "initial velocity": (
        "initial velocity",
        "initial speed",
        "starting speed",
        "starting velocity",
    ),
    "acceleration": ("acceleration", "acceleration", "rate of acceleration"),
    "time": ("time", "elapsed time", "duration", "time interval"),
    "displacement": ("displacement", "distance traveled", "distance covered", "displacement"),
    "net force": ("net force", "total force", "resultant force", "net force"),
    "mass": ("mass",),
    "gravitational acceleration": (
        "gravitational acceleration",
        "acceleration due to gravity",
        "local gravity",
        "surface gravity",
    ),
    "weight": ("weight", "weight", "weight force"),
    "kinetic energy": ("kinetic energy", "kinetic energy", "energy of motion"),
    "speed": ("speed", "speed", "velocity"),
    "potential energy": ("potential energy", "gravitational potential energy", "stored energy"),
    "height": ("height", "height", "elevation", "vertical height"),
    "momentum": ("momentum", "momentum", "linear momentum"),
    "velocity": ("velocity", "speed"),
    "mass 1": ("mass 1", "first mass", "mass of the first object", "mass of object one"),
    "mass 2": ("mass 2", "second mass", "mass of the second object", "mass of object two"),
    "velocity 1": ("velocity 1", "first velocity", "velocity of the first object"),
    "velocity 2": ("velocity 2", "second velocity", "velocity of the second object"),
    "gravitational force": (
        "gravitational force",
        "force of gravity",
        "gravitational pull",
        "gravitational attraction",
    ),
    "separation": ("separation", "distance between them", "center-to-center distance"),
    "voltage": ("voltage", "voltage", "potential difference", "voltage drop"),
    "current": ("current", "current", "electric current"),
    "resistance": ("resistance",),
    "pressure": ("pressure", "pressure", "gas pressure"),
    "volume": ("volume", "volume", "container volume"),
    "amount": ("amount of gas", "number of moles", "amount of substance"),
    "temperature": ("temperature", "temperature", "absolute temperature"),
    # Mechanics, rotation, gravitation, fluids, and waves (v0.3 equation expansion).
    "angular acceleration": ("angular acceleration", "rotational acceleration"),
    "angular frequency": ("angular frequency", "angular frequency"),
    "angular momentum": ("angular momentum", "spin angular momentum"),
    "angular velocity": ("angular velocity", "angular speed", "rotation rate"),
    "area": ("area", "cross-sectional area", "surface area"),
    "area 1": ("area 1", "first area", "area at the wide end"),
    "area 2": ("area 2", "second area", "area at the narrow end"),
    "average force": ("average force", "average force", "mean force"),
    "average speed": ("average speed", "mean speed", "speed"),
    "buoyant force": ("buoyant force", "buoyancy", "upthrust"),
    "central mass": ("central mass", "mass of the planet", "mass of the central body"),
    "centripetal acceleration": ("centripetal acceleration", "inward acceleration"),
    "centripetal force": ("centripetal force", "inward force", "net inward force"),
    "contact time": ("contact time", "collision time", "duration of the impact"),
    "density": ("density", "mass density"),
    "depth": ("depth", "depth below the surface"),
    "displaced volume": ("displaced volume", "submerged volume", "volume of fluid displaced"),
    "distance": ("distance", "distance traveled", "path length"),
    "distance from center": ("distance from center", "distance from the center", "radius"),
    "elastic potential energy": ("elastic potential energy", "spring energy", "stored energy"),
    "escape speed": ("escape speed", "escape velocity"),
    "extension": ("extension", "stretch", "compression"),
    "final speed": ("final speed", "end speed", "speed at the end"),
    "flow rate": ("flow rate", "volume flow rate", "discharge"),
    "flow speed": ("flow speed", "flow velocity", "speed of the flow"),
    "flow speed 1": ("flow speed 1", "first flow speed", "flow speed at the wide end"),
    "flow speed 2": ("flow speed 2", "second flow speed", "flow speed at the narrow end"),
    "fluid density": ("fluid density", "density of the fluid", "liquid density"),
    "force": ("force", "applied force"),
    "frequency": ("frequency", "frequency"),
    "fundamental frequency": ("fundamental frequency", "fundamental", "lowest frequency"),
    "gauge pressure": ("gauge pressure", "pressure", "water pressure"),
    "gravitational potential energy magnitude": (
        "gravitational potential energy magnitude",
        "magnitude of the gravitational potential energy",
        "gravitational binding energy",
    ),
    "impulse": ("impulse", "impulse"),
    "initial speed": ("initial speed", "starting speed"),
    "intensity": ("intensity", "intensity", "power per area"),
    "launch speed": ("launch speed", "initial speed", "speed at launch"),
    "lever arm": ("lever arm", "moment arm", "distance from the pivot"),
    "linear density": ("linear density", "mass per unit length"),
    "maximum height": ("maximum height", "peak height", "max height"),
    "moment of inertia": ("moment of inertia", "rotational inertia"),
    "net torque": ("net torque", "torque"),
    "net work": ("net work", "work done", "total work"),
    "observed frequency": ("observed frequency", "heard frequency", "frequency heard"),
    "orbital period": ("orbital period", "period of the orbit", "orbit time"),
    "orbital radius": ("orbital radius", "radius of the orbit", "orbit radius"),
    "orbital speed": ("orbital speed", "orbit speed", "orbital velocity"),
    "pendulum length": ("pendulum length", "length of the pendulum", "string length"),
    "period": ("period", "period", "time for one cycle"),
    "power": ("power", "power output"),
    "radius": ("radius", "radius"),
    "rotational kinetic energy": ("rotational kinetic energy", "spin energy"),
    "source frequency": ("source frequency", "emitted frequency", "frequency of the source"),
    "source power": ("source power", "power of the source", "emitted power"),
    "source speed": ("source speed", "speed of the source"),
    "speed of sound": ("speed of sound", "sound speed"),
    "spring constant": ("spring constant", "spring stiffness", "force constant"),
    "spring force": ("spring force", "restoring force", "force on the spring"),
    "string length": ("string length", "length of the string"),
    "tangential speed": ("tangential speed", "linear speed", "rim speed"),
    "tension": ("tension", "string tension"),
    "time to peak": ("time to peak", "time to reach the top", "rise time"),
    "torque": ("torque", "turning moment"),
    "wave speed": ("wave speed", "speed of the wave", "propagation speed"),
    "wavelength": ("wavelength", "wavelength"),
    "work": ("work", "work done"),
    # Thermodynamics, electromagnetism, optics, and modern physics.
    "average kinetic energy": ("average kinetic energy", "mean kinetic energy per molecule"),
    "capacitance": ("capacitance",),
    "change in internal energy": ("change in internal energy", "internal energy change"),
    "change in length": ("change in length", "expansion", "length increase"),
    "charge": ("charge", "electric charge"),
    "charge 1": ("charge 1", "first charge", "charge on the first object"),
    "charge 2": ("charge 2", "second charge", "charge on the second object"),
    "contracted length": ("contracted length", "measured length", "observed length"),
    "cross-sectional area": ("cross-sectional area", "wire cross section", "area"),
    "de Broglie wavelength": ("de Broglie wavelength", "matter wavelength", "wavelength"),
    "dilated time": ("dilated time", "time measured on Earth", "observed time"),
    "electric field": ("electric field", "field strength", "electric field strength"),
    "electric force": ("electric force", "electrostatic force", "Coulomb force"),
    "electric potential": ("electric potential", "potential", "voltage"),
    "equivalent resistance": ("equivalent resistance", "total resistance", "combined resistance"),
    "expansion coefficient": ("expansion coefficient", "coefficient of linear expansion"),
    "focal length": ("focal length",),
    "heat": ("heat", "heat energy", "thermal energy"),
    "heat flow rate": ("heat flow rate", "rate of heat flow", "heat loss rate"),
    "heat in": ("heat in", "heat absorbed", "heat from the hot reservoir"),
    "heat out": ("heat out", "heat rejected", "waste heat"),
    "image distance": ("image distance", "distance to the image"),
    "image height": ("image height", "height of the image"),
    "induced emf": ("induced emf", "induced voltage", "emf"),
    "inductance": ("inductance",),
    "latent heat": ("latent heat", "latent heat of fusion"),
    "lens power": ("lens power", "optical power", "power of the lens"),
    "magnetic field": ("magnetic field", "field strength", "magnetic flux density"),
    "magnetic force": ("magnetic force", "force from the field"),
    "maximum kinetic energy": ("maximum kinetic energy", "kinetic energy of the electrons"),
    "molecular mass": ("molecular mass", "mass of one molecule", "molecule mass"),
    "object distance": ("object distance", "distance to the object"),
    "object height": ("object height", "height of the object"),
    "original length": ("original length", "initial length", "starting length"),
    "particle mass": ("particle mass", "mass of the particle"),
    "path radius": ("path radius", "radius of the path", "orbit radius"),
    "photon energy": ("photon energy", "energy of the photon"),
    "photon momentum": ("photon momentum", "momentum of the photon"),
    "plate area": ("plate area", "area of each plate"),
    "plate separation": ("plate separation", "gap between the plates", "plate spacing"),
    "potential difference": ("potential difference", "voltage"),
    "proper length": ("proper length", "rest length"),
    "proper time": ("proper time", "time on the moving clock"),
    "radiated power": ("radiated power", "power radiated", "emitted power"),
    "relative speed": ("relative speed", "speed"),
    "resistance 1": ("resistance 1", "first resistance", "first resistor"),
    "resistance 2": ("resistance 2", "second resistance", "second resistor"),
    "resistivity": ("resistivity",),
    "rest energy": ("rest energy", "energy equivalent", "mass energy"),
    "rms speed": ("rms speed", "root-mean-square speed", "typical molecular speed"),
    "rod length": ("rod length", "length of the rod"),
    "specific heat": ("specific heat", "specific heat capacity"),
    "stored energy": ("stored energy", "energy stored"),
    "surface area": ("surface area", "area"),
    "temperature change": ("temperature change", "temperature rise", "change in temperature"),
    "temperature difference": ("temperature difference", "difference in temperature"),
    "thermal conductivity": ("thermal conductivity",),
    "thickness": ("thickness", "wall thickness"),
    "turns per length": ("turns per length", "turn density", "turns per meter"),
    "wire length": ("wire length", "length of the wire"),
    "work function": ("work function",),
    "work output": ("work output", "work done", "useful work"),
    # Dimensionless quantities, written as bare numbers.
    "kinetic friction force": ("kinetic friction force", "friction force", "friction"),
    "coefficient of kinetic friction": (
        "coefficient of kinetic friction",
        "friction coefficient",
        "coefficient of friction",
    ),
    "normal force": ("normal force", "normal force", "contact force"),
    "maximum static friction": ("maximum static friction", "max static friction force"),
    "coefficient of static friction": (
        "coefficient of static friction",
        "static friction coefficient",
    ),
    "efficiency": ("efficiency", "efficiency"),
    "useful energy out": ("useful energy out", "useful output energy", "useful work"),
    "energy in": ("energy in", "input energy", "energy supplied"),
    "maximum efficiency": ("maximum efficiency", "Carnot efficiency", "ideal efficiency"),
    "cold reservoir temperature": ("cold reservoir temperature", "cold side temperature"),
    "hot reservoir temperature": ("hot reservoir temperature", "hot side temperature"),
    "refractive index": ("refractive index", "index of refraction"),
    "speed of light in the medium": (
        "speed of light in the medium",
        "speed of light in the material",
    ),
    "secondary voltage": ("secondary voltage", "output voltage"),
    "primary voltage": ("primary voltage", "input voltage"),
    "secondary turns": ("secondary turns", "turns on the secondary"),
    "primary turns": ("primary turns", "turns on the primary"),
    "magnification": ("magnification", "magnifying power"),
    "number of molecules": ("number of molecules", "molecule count"),
    "Lorentz factor": ("Lorentz factor", "gamma factor"),
    "emissivity": ("emissivity",),
    "relative permittivity": ("relative permittivity", "dielectric constant"),
    "average induced emf": ("average induced emf", "average emf", "induced voltage"),
    "number of turns": ("number of turns", "turns", "coil turns"),
    "coil area": ("coil area", "area of the coil"),
}

# --------------------------------------------------------------------------- generic questions

# How one known value is stated. {name} the variable name, {a_name} with "a"/"an",
# {sym} its symbol, {q} the number and unit.
KNOWN_PATTERNS: tuple[Template, ...] = (
    Template("kp_01", "the {name} is {q}"),
    Template("kp_02", "{a_name} of {q}"),
    Template("kp_03", "{name} = {q}"),
    Template("kp_04", "{sym} = {q}"),
    Template("kp_05", "the {name} equals {q}"),
    Template("kp_06", "{name}: {q}"),
    Template("kp_07", "the {name} {sym} is {q}"),
    Template("kp_08", "the {name} was measured at {q}"),
    Template("kp_09", "{q} for the {name}"),
    Template("kp_10_h", "the {name} comes out to {q}"),
    Template("kp_11", "the {name} is {q} ({sym})"),
    Template("kp_12", "{name} {sym} = {q}"),
)

# Question frames. {target}/{Target} the unknown's name, {tsym} its symbol, {knowns}/{Knowns}
# the known values as a list, {facts} the known values as sentences.
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
    Template("gen_15", "{facts} What is the {target}?"),
    Template("gen_16", "{facts} Find the {target}."),
    Template("gen_17", "{facts} Solve for {tsym}."),
    Template("gen_18", "{facts} How big is the {target}?"),
    Template("gen_19", "Solve for {tsym} when {knowns}."),
    Template("gen_20", "Find {tsym}, given {knowns}."),
    Template("gen_21", "Determine {tsym} if {knowns}."),
    Template("gen_22", "What {target} do you get when {knowns}?"),
    Template("gen_23", "How much {target} is there if {knowns}?"),
    Template("gen_24", "Using {knowns}, find the {target}."),
    Template("gen_25", "{Knowns}. Find the {target}."),
    Template("gen_26", "{Knowns}. Solve for the {target}."),
    Template("gen_27", "{Knowns}, so what is the {target}?"),
    Template("gen_28", "Known: {knowns}. Unknown: the {target}."),
    Template("gen_29", "The {target} is unknown, but {knowns}. What is it?"),
    Template("gen_30", "I need the {target}. I know {knowns}."),
    Template("gen_31", "Can you figure out the {target}? {Knowns}."),
    Template("gen_32", "Help me find the {target} when {knowns}."),
    Template("gen_33_h", "{facts} What does the {target} come out to?"),
    Template("gen_34", "Given that {knowns}, solve for {tsym}."),
    Template("gen_35", "If we have {knowns}, what's {tsym}?"),
    Template("gen_36", "How do I get the {target} when {knowns}?"),
    Template("gen_37", "Compute the {target} for {knowns}."),
    Template("gen_38_h", "Say {knowns}. What's the {target} then?"),
    Template("gen_39", "{facts} I want to know the {target}."),
    Template("gen_40", "Find the {target}: {knowns}."),
    Template("gen_41", "{Target} = ? when {knowns}."),
    Template("gen_42", "What {target} results if {knowns}?"),
    Template("gen_43_h", "{facts} Work out {tsym}."),
    Template("gen_44", "Determine the value of the {target} given {knowns}."),
)

# Equations whose variables are identical to another's (series vs parallel resistors) can
# only be told apart by the question's words, so a question for one says which, with one of
# these sentences when its own wording doesn't already. Each names one of the equation's
# tags that its twin lacks.
EQUATION_CONTEXT: dict[str, tuple[str, ...]] = {
    "series_resistors": (
        "Two resistors are connected in series.",
        "The resistors are wired in series.",
        "This is a series circuit.",
    ),
    "parallel_resistors": (
        "Two resistors are connected in parallel.",
        "The resistors are wired in parallel.",
        "This is a parallel circuit.",
    ),
    "orbital_speed": (
        "It moves in a circular orbit.",
        "A satellite is in a circular orbit.",
    ),
    "escape_velocity": (
        "A rocket has to reach escape velocity.",
        "It has to reach escape speed to leave for good.",
    ),
}

# --------------------------------------------------------------------------- framing noise

# Openers and sign-offs, chosen for a minority of questions. No digits, ever.
PREAMBLES: tuple[str, ...] = (
    "Physics question:",
    "Homework help:",
    "Quick question.",
    "Need help with this one.",
    "From my textbook:",
    "Here's a problem I'm stuck on.",
    "Okay so",
    "Hey,",
    "Exam practice:",
    "Lab question:",
    "Can you help?",
    "Problem:",
    "Hi!",
    "My teacher asked this:",
    "Practice problem.",
    "Real quick,",
    "Help please:",
    "I'm studying for a test.",
    "This came up in class:",
    "So",
)
SIGN_OFFS: tuple[str, ...] = (
    "Thanks!",
    "Show the steps.",
    "Please explain.",
    "Thanks in advance.",
    "Any help appreciated.",
    "Explain your reasoning.",
    "Ty",
    "Cheers.",
)

# --------------------------------------------------------------------------- scenarios


@dataclass(frozen=True)
class Scenario:
    """A worded problem for one equation.

    Slots: ``{object}``, ``{object2}``, ``{vehicle}``, ``{vehicle2}``, and one per given
    symbol (``{t}`` reads "12 s"). ``{m_a}`` writes the value as an adjective ("a 5 kg
    ball"), so it always uses the unit symbol. ``forced`` are knowns the wording implies:
    (symbol, value or "g" for standard gravity, unit, origin).
    """

    equation: str
    template: Template
    target: str
    forced: tuple[tuple[str, str, str, str], ...] = ()
    assumptions: tuple[str, ...] = ()


_REST = ("v0", "0", "m/s", "assumption")
_GRAVITY = ("a", "g", "m/s^2", "constant")
_G = ("g", "g", "m/s^2", "constant")
_DROP = ("Released from rest", "Air resistance is negligible")
_FROM_REST = ("Starts from rest", "Acceleration stays constant")

SCENARIOS: tuple[Scenario, ...] = (
    # kin_v_squared
    Scenario(
        "kin_v_squared",
        Template(
            "drop_speed_01", "A {object} is dropped from {d}. How fast is it going when it lands?"
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template("drop_speed_02_h", "How fast does a {object} hit the floor after falling {d}?"),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "drop_speed_03",
            "Someone lets go of a {object} from a balcony {d} up. What speed does it reach just "
            "before impact?",
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "drop_speed_04",
            "I knocked a {object} off a ledge that's {d} high. How fast was it moving at the "
            "bottom?",
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "ramp_speed_01",
            "A {vehicle} starts from rest and speeds up at {a} over {d}. What speed does it reach?",
        ),
        "v",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "ramp_speed_02",
            "A {vehicle} already doing {v0} accelerates at {a} for a stretch of {d}. What is its "
            "final speed?",
        ),
        "v",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "runway_01",
            "A {vehicle} has to reach {v} from a standstill within {d}. What acceleration does "
            "that take?",
        ),
        "a",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "drop_height_01_h", "From what height would a {object} have to fall to land at {v}?"
        ),
        "d",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "drop_height_02",
            "A {object} hits the ground at {v}. If it fell from rest, how far did it drop?",
        ),
        "d",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "start_speed_01",
            "After speeding up at {a} over {d}, a {vehicle} is doing {v}. "
            "How fast was it going at the start?",
        ),
        "v0",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "roof_drop_01",
            "A {object} tumbles off a roof {d} high. How fast is it going when it hits the ground?",
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "cliff_height_01",
            "A {object} dropped from the top of a cliff lands at {v}. How tall is the cliff?",
        ),
        "d",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "rest_release_v_sq_01",
            "A {object} is released from rest {d} above the ground. How fast is it moving on "
            "impact?",
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_squared",
        Template(
            "rest_ramp_v_sq_01_h",
            "A {vehicle} starts at rest, speeds up at {a}, and covers {d}. What is its final "
            "speed?",
        ),
        "v",
        (_REST,),
        _FROM_REST,
    ),
    # kin_v_at
    Scenario(
        "kin_v_at",
        Template(
            "start_rest_01",
            "A {vehicle} starts from rest and accelerates at {a} for {t}. What is its final speed?",
        ),
        "v",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_at",
        Template(
            "speed_up_01",
            "A {vehicle} moving at {v0} speeds up at {a} for {t}. How fast is it going afterward?",
        ),
        "v",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "speed_up_02_h",
            "Traveling at {v0}, a {vehicle} gains speed at {a}. What is its speed {t} later?",
        ),
        "v",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "reach_time_01",
            "How long does a {vehicle} need to get from {v0} up to {v} if it accelerates at {a}?",
        ),
        "t",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "reach_time_02",
            "Starting from a standstill, a {vehicle} accelerates at {a}. How many seconds until "
            "it hits {v}?",
        ),
        "t",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_at",
        Template("accel_01", "A {vehicle} goes from {v0} to {v} in {t}. What is its acceleration?"),
        "a",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "accel_02",
            "In {t}, a {vehicle} that started at rest is doing {v}. What was its average "
            "acceleration?",
        ),
        "a",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_at",
        Template("fall_speed_01", "A {object} falls from rest. How fast is it moving after {t}?"),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_at",
        Template(
            "throw_down_01",
            "A {object} is thrown straight down at {v0}. What is its speed after {t} of falling?",
        ),
        "v",
        (_GRAVITY,),
        ("Air resistance is negligible",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "start_speed_02",
            "A {vehicle} accelerating at {a} is doing {v} after {t}. What was its starting speed?",
        ),
        "v0",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "start_speed_03_h",
            "What speed did a {vehicle} start at if it reaches {v} "
            "after {t} of accelerating at {a}?",
        ),
        "v0",
        (),
        ("Acceleration stays constant",),
    ),
    # Velocity changes stated "from A to B", with the average acceleration asked for.
    Scenario(
        "kin_v_at",
        Template(
            "change_speed_accel_01",
            "A {vehicle} changes its speed from {v0} to {v} in {t}. What is its acceleration?",
        ),
        "a",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "speed_up_accel_01_h",
            "The {vehicle} speeds up from {v0} to {v} in {t}. Find its acceleration.",
        ),
        "a",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "avg_accel_01",
            "Over {t}, the velocity of a {vehicle} climbs from {v0} to {v}. What is its average "
            "acceleration?",
        ),
        "a",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "avg_accel_02_h",
            "What average acceleration does a {vehicle} have if its velocity increases from {v0} "
            "to {v} during {t}?",
        ),
        "a",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "change_time_01",
            "How long must a {vehicle} accelerate at {a} to change its speed from {v0} to {v}?",
        ),
        "t",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_v_at",
        Template(
            "speed_up_time_01_h",
            "A {vehicle} speeds up from {v0} to {v} at a steady {a}. How many seconds does that "
            "take?",
        ),
        "t",
        (),
        ("Acceleration stays constant",),
    ),
    # Rest cues the database assumes as v0 = 0: "starts at rest", "released from rest".
    Scenario(
        "kin_v_at",
        Template(
            "rest_release_v_01",
            "A {object} is released from rest and falls for {t}. What is its speed at that moment?",
        ),
        "v",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_v_at",
        Template(
            "rest_start_v_01_h",
            "A {vehicle} starts at rest and builds speed at {a} for {t}. What speed does it have "
            "then?",
        ),
        "v",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_v_at",
        Template(
            "rest_avg_accel_01",
            "A {vehicle} starts at rest and its velocity is {v} after {t}. What is its average "
            "acceleration?",
        ),
        "a",
        (_REST,),
        _FROM_REST,
    ),
    # kin_x_at
    Scenario(
        "kin_x_at",
        Template(
            "rest_drop_dist_01",
            "A {object} is dropped and falls for {t}. How far does it drop?",
        ),
        "x",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_x_at",
        Template(
            "rest_start_dist_01_h",
            "A {vehicle} that starts at rest accelerates at {a} for {t}. How far does it go?",
        ),
        "x",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_x_at",
        Template("drop_dist_01", "How far does a {object} fall in {t} after being released?"),
        "x",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_x_at",
        Template(
            "drop_dist_02",
            "A {object} slips off a cliff and falls for {t}. How far has it dropped?",
        ),
        "x",
        (_REST, _GRAVITY),
        _DROP,
    ),
    Scenario(
        "kin_x_at",
        Template(
            "launch_dist_01",
            "A {vehicle} starts from rest and accelerates at {a} for {t}. How far does it travel?",
        ),
        "x",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_x_at",
        Template(
            "launch_dist_02_h",
            "What distance does a {vehicle} cover in {t} if it begins at {v0} and accelerates at "
            "{a}?",
        ),
        "x",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_x_at",
        Template("cruise_01", "A {vehicle} cruises at a steady {v0} for {t}. How far does it go?"),
        "x",
        (("a", "0", "m/s^2", "assumption"),),
        ("Speed stays constant",),
    ),
    Scenario(
        "kin_x_at",
        Template(
            "start_speed_04",
            "A {vehicle} accelerating at {a} covers {x} in {t}. What was its initial speed?",
        ),
        "v0",
        (),
        ("Acceleration stays constant",),
    ),
    Scenario(
        "kin_x_at",
        Template(
            "track_accel_01",
            "Starting from rest, a {vehicle} covers {x} in {t}. What is its acceleration?",
        ),
        "a",
        (_REST,),
        _FROM_REST,
    ),
    Scenario(
        "kin_x_at",
        Template("fall_time_01", "How long does it take a {object} to fall {x} from rest?"),
        "t",
        (_REST, _GRAVITY),
        _DROP,
    ),
    # newton_second_law
    Scenario(
        "newton_second_law",
        Template(
            "push_force_01", "What net force does it take to accelerate a {m_a} {vehicle} at {a}?"
        ),
        "F",
        (),
        ("Friction is ignored",),
    ),
    Scenario(
        "newton_second_law",
        Template(
            "push_accel_01",
            "A net force of {F} acts on a {m_a} {object}. What is its acceleration?",
        ),
        "a",
    ),
    Scenario(
        "newton_second_law",
        Template(
            "push_accel_02",
            "You shove a {object} with a mass of {m} using a net force of {F}. How quickly does "
            "it accelerate?",
        ),
        "a",
        (),
        ("Friction is ignored",),
    ),
    Scenario(
        "newton_second_law",
        Template(
            "push_mass_01",
            "A {F_a} net force makes a {object} accelerate at {a}. What is its mass?",
        ),
        "m",
    ),
    Scenario(
        "newton_second_law",
        Template(
            "push_mass_02_h",
            "Pushing a {vehicle} with {F} of net force gives it an acceleration of {a}. How heavy "
            "is it in kilograms?",
        ),
        "m",
        (),
        ("Friction is ignored",),
    ),
    Scenario(
        "newton_second_law",
        Template(
            "brake_force_01", "A {m_a} {vehicle} slows down at {a}. What net force is acting on it?"
        ),
        "F",
    ),
    # weight
    Scenario(
        "weight",
        Template("weight_earth_01", "How much does a {m_a} {object} weigh on Earth?"),
        "W",
        (_G,),
        ("Standard gravity at the surface",),
    ),
    Scenario(
        "weight",
        Template("weight_earth_02", "What is the weight of a {object} with a mass of {m}?"),
        "W",
        (_G,),
        ("Standard gravity at the surface",),
    ),
    Scenario(
        "weight",
        Template("weight_mass_01", "A {object} weighs {W} on Earth. What is its mass?"),
        "m",
        (_G,),
        ("Standard gravity at the surface",),
    ),
    Scenario(
        "weight",
        Template(
            "weight_planet_01",
            "On a planet where gravity is {g}, how much would a {m_a} {object} weigh?",
        ),
        "W",
    ),
    Scenario(
        "weight",
        Template(
            "weight_planet_02_h",
            "A {m_a} {object} weighs {W} on some alien world. What is the gravitational "
            "acceleration there?",
        ),
        "g",
    ),
    Scenario(
        "weight",
        Template(
            "scale_01", "A bathroom scale reads a force of {W} under a {object}. What mass is that?"
        ),
        "m",
        (_G,),
        ("Standard gravity at the surface",),
    ),
    # gravitational_pe
    Scenario(
        "gravitational_pe",
        Template(
            "lift_pe_01", "How much potential energy does a {m_a} {object} gain when lifted {h}?"
        ),
        "U",
        (_G,),
        ("Standard gravity near the surface",),
    ),
    Scenario(
        "gravitational_pe",
        Template(
            "shelf_pe_01",
            "A {m_a} {object} sits on a shelf {h} above the floor. How much gravitational "
            "potential energy does it have?",
        ),
        "U",
        (_G,),
        ("Standard gravity near the surface", "The floor is the reference level"),
    ),
    Scenario(
        "gravitational_pe",
        Template(
            "lift_height_01",
            "How high do you have to raise a {m_a} {object} to store {U} of potential energy?",
        ),
        "h",
        (_G,),
        ("Standard gravity near the surface",),
    ),
    Scenario(
        "gravitational_pe",
        Template(
            "lift_mass_01_h",
            "Lifting a {object} by {h} took {U} of work against gravity. What is its mass?",
        ),
        "m",
        (_G,),
        ("Standard gravity near the surface",),
    ),
    Scenario(
        "gravitational_pe",
        Template(
            "crane_pe_01",
            "A crane hoists a {m_a} load up {h}. How much energy goes into potential energy?",
        ),
        "U",
        (_G,),
        ("Standard gravity near the surface",),
    ),
    # kinetic_energy
    Scenario(
        "kinetic_energy",
        Template(
            "ke_01",
            "A {m_a} {object} flies through the air at {v}. How much kinetic energy does it carry?",
        ),
        "KE",
    ),
    Scenario(
        "kinetic_energy",
        Template(
            "ke_02",
            "How much energy of motion does a {vehicle} with a mass of {m} have when it does {v}?",
        ),
        "KE",
    ),
    Scenario(
        "kinetic_energy",
        Template(
            "ke_speed_01", "A {m_a} {object} has {KE} of kinetic energy. How fast is it moving?"
        ),
        "v",
    ),
    Scenario(
        "kinetic_energy",
        Template(
            "ke_mass_01_h",
            "Something zipping along at {v} carries {KE} of kinetic energy. What is its mass?",
        ),
        "m",
    ),
    # momentum
    Scenario(
        "momentum",
        Template("p_01", "A {m_a} {vehicle} rolls along at {v}. How much momentum does it have?"),
        "p",
    ),
    Scenario(
        "momentum",
        Template(
            "p_speed_01",
            "A {object} with a mass of {m} has a momentum of {p}. How fast is it going?",
        ),
        "v",
    ),
    Scenario(
        "momentum",
        Template("p_mass_01_h", "An object moving at {v} has {p} of momentum. What's its mass?"),
        "m",
    ),
    # inelastic_collision
    Scenario(
        "inelastic_collision",
        Template(
            "couple_01",
            "A {m1_a} {vehicle} moving at {v1} bumps into a {m2_a} {vehicle2} moving at {v2} in "
            "the same direction, and they lock together. How fast do they move afterward?",
        ),
        "vf",
        (),
        ("Both move in the same direction before the collision",),
    ),
    Scenario(
        "inelastic_collision",
        Template(
            "couple_02",
            "A {m1_a} cart rolling at {v1} hits a {m2_a} cart sitting still, and the two latch "
            "together. What is their shared speed?",
        ),
        "vf",
        (("v2", "0", "m/s", "assumption"),),
        ("The second cart starts at rest",),
    ),
    Scenario(
        "inelastic_collision",
        Template(
            "couple_03_h",
            "Two lumps of clay collide and stick: one has mass {m1} and speed {v1}, the other "
            "mass {m2} and speed {v2}, moving the same way. What is the final velocity?",
        ),
        "vf",
        (),
        ("Both move in the same direction before the collision",),
    ),
    # impulse_momentum: a speed change "from A to B" asks for the force or the contact time.
    Scenario(
        "impulse_momentum",
        Template(
            "impulse_force_01",
            "A {m_a} {object} changes its speed from {v0} to {v} during a {t} impact. What average "
            "force acted on it?",
        ),
        "F",
        (),
        ("The force is constant during the impact",),
    ),
    Scenario(
        "impulse_momentum",
        Template(
            "impulse_force_02_h",
            "A {vehicle} with a mass of {m} speeds up from {v0} to {v} in {t}. What average net "
            "force was needed?",
        ),
        "F",
        (),
        ("The force is constant during the impact",),
    ),
    Scenario(
        "impulse_momentum",
        Template(
            "impulse_time_01",
            "A net force of {F} acts on a {m_a} {object} that changes its speed from {v0} to {v}. "
            "How long does the force act?",
        ),
        "t",
        (),
        ("The force is constant during the impact",),
    ),
    Scenario(
        "impulse_momentum",
        Template(
            "impulse_time_02_h",
            "Pushing a {m_a} {vehicle} with {F} takes it from {v0} up to {v}. How long was the "
            "push?",
        ),
        "t",
        (),
        ("The force is constant during the impact",),
    ),
    Scenario(
        "impulse_momentum",
        Template(
            "impulse_rest_force_01",
            "A {m_a} {object} starts at rest, and a steady push brings it to {v} in {t}. What "
            "average force was applied?",
        ),
        "F",
        (_REST,),
        ("Starts at rest", "The force is constant during the impact"),
    ),
    # newton_gravitation
    Scenario(
        "newton_gravitation",
        Template(
            "grav_01",
            "What is the gravitational pull between a {m1_a} {object} and a {m2_a} {object2} that "
            "are {r} apart?",
        ),
        "F",
    ),
    Scenario(
        "newton_gravitation",
        Template(
            "grav_02_h",
            "Two boulders of {m1} and {m2} sit {r} apart. How strongly do they attract each other?",
        ),
        "F",
    ),
    # ohms_law
    Scenario(
        "ohms_law",
        Template(
            "battery_01",
            "What current flows when a {V_a} battery is connected across a {R_a} resistor?",
        ),
        "I",
        (),
        ("Ideal battery with no internal resistance",),
    ),
    Scenario(
        "ohms_law",
        Template(
            "heater_01", "How much current does a {R_a} heating element draw from a {V_a} supply?"
        ),
        "I",
        (),
        ("Ideal supply with no internal resistance",),
    ),
    Scenario(
        "ohms_law",
        Template(
            "drop_v_01",
            "A current of {I} runs through a {R_a} resistor. What is the voltage across it?",
        ),
        "V",
    ),
    Scenario(
        "ohms_law",
        Template(
            "resist_01",
            "A {V_a} source pushes {I} through a circuit. What is the circuit's resistance?",
        ),
        "R",
        (),
        ("The circuit acts as a single resistance",),
    ),
    Scenario(
        "ohms_law",
        Template("resist_02_h", "What resistor would let exactly {I} flow from a {V_a} battery?"),
        "R",
        (),
        ("Ideal battery with no internal resistance",),
    ),
    Scenario(
        "ohms_law",
        Template(
            "led_v_01",
            "An LED circuit carries {I} through a {R_a} resistor. How many volts drop across the "
            "resistor?",
        ),
        "V",
    ),
    # ideal_gas_law
    Scenario(
        "ideal_gas_law",
        Template(
            "tank_p_01",
            "A sealed {V_a} tank holds {n} of gas at {T}. How much pressure does the gas exert on "
            "the walls?",
        ),
        "P",
        (),
        ("The gas behaves ideally",),
    ),
    Scenario(
        "ideal_gas_law",
        Template("gas_v_01", "What volume do {n} of gas take up at {P} and {T}?"),
        "V",
        (),
        ("The gas behaves ideally",),
    ),
    Scenario(
        "ideal_gas_law",
        Template(
            "gas_n_01", "A {V_a} cylinder contains gas at {P} and {T}. How many moles are inside?"
        ),
        "n",
        (),
        ("The gas behaves ideally",),
    ),
    Scenario(
        "ideal_gas_law",
        Template(
            "gas_t_01_h",
            "At what temperature would {n} of gas in a {V_a} container reach a pressure of {P}?",
        ),
        "T",
        (),
        ("The gas behaves ideally",),
    ),
    Scenario(
        "ideal_gas_law",
        Template(
            "balloon_01",
            "A weather balloon holds {n} of helium at {T} and {P}. What is its volume?",
        ),
        "V",
        (),
        ("Helium behaves as an ideal gas",),
    ),
)

OBJECTS = (
    "ball", "rock", "phone", "book", "watermelon", "brick", "coin", "toy car", "wrench",
    "apple", "bowling ball", "stone", "pumpkin", "laptop", "basketball", "hammer", "mug",
    "backpack", "dumbbell", "pebble", "baseball", "suitcase", "bucket", "shoe", "melon",
    "crate", "box", "tennis ball", "water bottle", "potted plant",
)  # fmt: skip
VEHICLES = (
    "car", "bike", "train", "scooter", "sled", "cart", "boat", "truck", "bus", "motorcycle",
    "skateboarder", "go-kart", "tram", "drone", "rocket sled", "jet ski", "snowmobile",
)  # fmt: skip

# --------------------------------------------------------------------------- reasoning and strategy

STANDARD_REASONING = (
    "All the needed values are given, so this is a standard {domain} problem.",
    "The question gives the inputs directly; it is {domain}.",
    "A well-posed {domain} question with every value stated or a standard constant.",
    "Everything required is stated, so a single {domain} equation answers it.",
    "Standard {domain}: the known values pin down the unknown.",
    "This is textbook {domain} with the inputs supplied.",
    "The inputs are given outright, so no estimation is needed; it is {domain}.",
    "A direct {domain} calculation from the stated values.",
)

STRATEGIES = (
    "Use {eq} and solve for the {target}.",
    "Rearrange {eq} for the {target}.",
    "Solve {eq} for {sym}.",
    "Apply {eq} and isolate {sym}.",
    "Plug the known values into {eq} and solve for the {target}.",
    "Start from {eq}, then solve it for {sym}.",
    "{Eq} links the knowns to the {target}; solve it for {sym}.",
    "Substitute the givens into {eq} to get the {target}.",
)

# --------------------------------------------------------------------------- chained problems


@dataclass(frozen=True)
class Chain:
    """A two-step problem: ``first`` gives ``via``, which ``then`` needs for ``target``.

    The question states the values of both equations except ``via`` and ``target``, so no
    single equation can answer it ("a 1500 kg car goes from 0 to 20 m/s in 8 s: what net
    force?" needs the acceleration first).
    """

    first: str
    via: str
    then: str
    target: str


# Only chains no single equation in the database answers: "from rest to 20 m/s in 8 s,
# what force?" is not here, because the impulse-momentum theorem answers it in one step,
# and density then weight is the buoyant force's formula.
CHAINS: tuple[Chain, ...] = (
    Chain("newton_second_law", "a", "kin_x_at", "x"),
    Chain("newton_second_law", "a", "kin_v_squared", "v"),
    Chain("kin_v_squared", "v", "kinetic_energy", "KE"),
    Chain("kin_v_squared", "v", "momentum", "p"),
    Chain("kin_v_at", "v", "kinetic_energy", "KE"),
    Chain("kin_v_at", "v", "momentum", "p"),
    Chain("work_constant_force", "W", "power_work_time", "P"),
    Chain("density", "m", "gravitational_pe", "U"),
    Chain("density", "m", "kinetic_energy", "KE"),
)

CHAIN_STRATEGIES = (
    "Find the {via} with {first}, then use it in {then} to get the {target}.",
    "First solve {first} for {via_sym}; then {then} gives {sym}.",
    "Get {via_sym} from {first}, then put it into {then} and solve for {sym}.",
    "Two steps: {first} gives the {via}, and {then} turns it into the {target}.",
)

# --------------------------------------------------------------------------- Fermi questions

# Fermi questions (classification only until the Fermi engine lands in v0.7).
FERMI: tuple[Template, ...] = (
    Template("fermi_01", "How many {small} would it take to fill a {container}?"),
    Template("fermi_02", "How many {small} would you need to stack to reach the top of a {tall}?"),
    Template("fermi_03", "How many {small} weigh as much as a {heavy}?"),
    Template("fermi_04_h", "Roughly how many {small} could fit inside a {container}?"),
    Template("fermi_05", "If you lined up {small} end to end, how many would span a {long}?"),
    Template("fermi_06", "How much energy would it take to lift a {heavy} to the top of a {tall}?"),
    Template("fermi_07", "About how many {small} are there in a packed {container}?"),
    Template("fermi_08", "How many {people} could you fit in a {container}?"),
    Template("fermi_09", "How much does all the air in a {container} weigh?"),
    Template("fermi_10", "How many {small} tall is a {tall}?"),
    Template("fermi_11_h", "Estimate how many {small} it would take to cover a {area}."),
    Template(
        "fermi_12",
        "If a {huge} sat on one side of a giant scale, how many {people} would balance it?",
    ),
    Template("fermi_13", "How long would it take to walk the length of a {long} a thousand times?"),
    Template("fermi_14", "How much water does a {container} hold?"),
    Template("fermi_15", "How many {small} would it take to pave a {area}?"),
    Template("fermi_16", "Ballpark: how many {people} weigh as much as a {huge}?"),
    Template("fermi_17", "How many times does a person's heart beat in a lifetime?"),
    Template("fermi_18", "How many {small} could you buy with all the coins in a {container}?"),
    Template(
        "fermi_19_h", "What's a rough guess for the number of {small} that would bury a {heavy}?"
    ),
    Template(
        "fermi_20", "How much energy does a {heavy} have when it rolls down a {tall}-sized hill?"
    ),
    Template("fermi_21", "How many {small} laid flat would cover a {area}?"),
    Template("fermi_22", "How many {people} holding hands would stretch across a {long}?"),
    Template("fermi_23", "Order of magnitude: how many {small} fit in a {container}?"),
    Template("fermi_24", "How many {small} does the average person go through in a year?"),
    # Look-alikes of the out-of-scope energy questions, about real things.
    Template("fermi_25", "How much energy is stored in a {energetic}?"),
    Template("fermi_26", "How much energy is released by a {energetic}?"),
)
FERMI_SLOTS: dict[str, tuple[str, ...]] = {
    "small": (
        "tennis balls", "marbles", "ping pong balls", "golf balls", "pennies", "dice",
        "jelly beans", "grains of rice", "sugar cubes", "LEGO bricks", "paper clips",
        "soccer balls", "eggs", "playing cards", "popcorn kernels", "bricks",
    ),
    "container": (
        "school bus", "bathtub", "swimming pool", "classroom", "shipping container",
        "car trunk", "refrigerator", "stadium", "hot air balloon", "concert hall",
    ),
    "tall": ("skyscraper", "lighthouse", "redwood tree", "radio tower", "mountain", "giraffe"),
    "heavy": (
        "school bus", "elephant", "pickup truck", "grand piano", "blue whale", "horse",
        "cow", "minivan",
    ),
    "huge": ("cruise ship", "jumbo jet", "aircraft carrier", "space shuttle", "train"),
    "energetic": (
        "car battery", "candy bar", "lightning bolt", "gallon of gasoline", "thunderstorm",
        "AA battery", "campfire", "hurricane",
    ),
    "long": ("football field", "city block", "suspension bridge", "runway", "marathon course"),
    "area": ("football field", "parking lot", "basketball court", "city park", "tennis court"),
    "people": ("people", "students", "adults", "kids"),
}  # fmt: skip
FERMI_REASONING = (
    "Physically meaningful, but it needs estimated everyday quantities.",
    "A Fermi estimate: the answer depends on assumed sizes and masses.",
    "Answerable with order-of-magnitude assumptions rather than given values.",
    "No values are given, so it needs rough estimates of everyday sizes.",
    "An estimation problem: reasonable assumptions give an order of magnitude.",
    "Real physics, but the inputs have to be estimated, not read off.",
    "A back-of-the-envelope estimate built from typical quantities.",
    "Meaningful but underspecified; typical values make it answerable roughly.",
)

# --------------------------------------------------------------------------- out of scope

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
    (
        Template("oos_speed_01", "How fast is {abstract}?"),
        "Category error: {abstract} does not move, so it has no speed.",
        "How fast does sound travel through air?",
    ),
    (
        Template("oos_volume_01", "What is the volume of {emotion}?"),
        "Category error: {emotion} is a feeling and takes up no space.",
        "What is the volume of a typical coffee mug?",
    ),
    (
        Template("oos_energy_01_h", "How much energy is in {abstract}?"),
        "Category error: {abstract} is not a physical system that stores energy.",
        "How much energy is stored in a typical phone battery?",
    ),
    (
        Template("oos_density_01", "What is the density of {abstract}?"),
        "Category error: {abstract} has neither mass nor volume.",
        "What is the density of seawater?",
    ),
    (
        Template("oos_stock_01", "Will {company} stock go up next week?"),
        "Not a physics question; it is a financial prediction.",
        "How much electricity does a typical data center use?",
    ),
    (
        Template("oos_opinion_01", "Which is cooler, {topic_a} or {topic_b}?"),
        "Not a physics question; it asks for an opinion.",
        "How much energy does it take to heat a cup of water to boiling?",
    ),
    (
        Template("oos_lottery_01", "What numbers will win the lottery in {city} next month?"),
        "Unknowable: a fair lottery draw is random by design.",
        "How does a ball bounce when dropped onto a hard floor?",
    ),
    (
        Template("oos_meaning_01", "What is the meaning of life?"),
        "Not a physics question; it is philosophy.",
        "How much energy does the human body use in a day?",
    ),
    (
        Template("oos_recipe_01", "How do I make the perfect {food}?"),
        "Not a physics question; it is cooking advice.",
        "How long does it take to bring a pot of water to a boil?",
    ),
    (
        Template("oos_history_01_h", "Who was the first ruler of {city}?"),
        "Not a physics question; it is history.",
        "How tall can a stone tower be before it crushes its own base?",
    ),
    (
        Template("oos_color_01", "What color is {emotion}?"),
        "Category error: {emotion} is a feeling and reflects no light.",
        "Why does the sky look blue during the day?",
    ),
    (
        Template("oos_dream_01", "How much does a dream weigh?"),
        "Category error: a dream is an experience, not an object with weight.",
        "How much does the human brain weigh?",
    ),
    (
        Template("oos_before_01", "What happened before time existed?"),
        "Unknowable: physics has no settled account of anything without time.",
        "How old is the universe?",
    ),
    (
        Template("oos_mind_01", "What am I thinking right now?"),
        "Not answerable: physics cannot read anyone's thoughts.",
        "How fast do signals travel along a human nerve?",
    ),
    (
        Template("oos_code_01", "Write me a poem about {food}."),
        "Not a physics question; it asks for creative writing.",
        "How many calories are in a typical serving of {food}?",
    ),
    (
        Template("oos_exact_01", "Exactly how many grains of sand are on every beach on Earth?"),
        "Unknowable to an exact count; only a rough estimate is possible.",
        "Roughly how many grains of sand fit in a bucket?",
    ),
    (
        Template("oos_friction_01", "What is the friction coefficient of {abstract}?"),
        "Category error: {abstract} has no surface, so it has no friction.",
        "What is the friction coefficient of rubber on dry concrete?",
    ),
    # Look-alikes: phrased like real physics questions, about things that aren't physical.
    (
        Template("oos_weigh_02", "How much does {abstract} weigh?"),
        "Category error: {abstract} is an idea and has no weight.",
        "How much does a typical paperback book weigh?",
    ),
    (
        Template("oos_force_01", "How much force does {emotion} exert?"),
        "Category error: {emotion} is a feeling and pushes on nothing.",
        "How much force does it take to push a shopping cart?",
    ),
    (
        Template("oos_energy_02", "How much energy does {abstract} contain?"),
        "Category error: {abstract} is not a physical system, so it stores no energy.",
        "How much energy does a candy bar contain?",
    ),
    (
        Template("oos_momentum_01", "What is the momentum of {abstract}?"),
        "Category error: {abstract} has no mass and does not move.",
        "What is the momentum of a thrown baseball?",
    ),
    (
        Template("oos_power_01", "How much power does {emotion} use?"),
        "Category error: {emotion} is a feeling, not a device that draws power.",
        "How much power does a laptop use?",
    ),
    # Pure math: no physical quantity at all, however much it looks like a homework problem.
    (
        Template("oos_math_01", "What is {num_a} divided by {num_b}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
    (
        Template("oos_math_02", "What is the slope of {curve}?"),
        "Not a physics question; it is pure math.",
        "How fast is a dropped rock moving after falling for a while?",
    ),
    (
        Template("oos_math_03_h", "How do you find the slope of {curve}?"),
        "Not a physics question; it is pure math.",
        "How fast is a dropped rock moving after falling for a while?",
    ),
    (
        Template("oos_math_04", "What is the derivative of {func}?"),
        "Not a physics question; it is calculus.",
        "How does the speed of a falling ball change with time?",
    ),
    (
        Template("oos_math_05", "What is the integral of {func}?"),
        "Not a physics question; it is calculus.",
        "How far does a car travel while it speeds up?",
    ),
    (
        Template("oos_math_06", "Solve for x: {equation}"),
        "Not a physics question; it is algebra.",
        "How much force does it take to accelerate a shopping cart?",
    ),
    (
        Template("oos_math_07", "Is {integer} a prime number?"),
        "Not a physics question; it is number theory.",
        "How many atoms are in a grain of sand, roughly?",
    ),
    # The same arithmetic with digits, so spelled-out numbers don't come to mean "math".
    (
        Template("oos_math_08", "What is {int_a} divided by {int_b}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
    (
        Template("oos_math_09_h", "What is {int_a} times {int_b}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
    # Arithmetic as a textbook or homework line would phrase it, with no physical quantity.
    (
        Template("oos_math_10", "This came up in class: what is {int_a} times {int_b}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
    (
        Template("oos_math_11", "what's {int_a} divided by {int_b}"),
        "Not a physics question; it is arithmetic.",
        "How fast does a dropped ball hit the floor?",
    ),
    (
        Template("oos_math_12", "compute {dec_a} squared"),
        "Not a physics question; it is arithmetic.",
        "How far does a car travel while it speeds up?",
    ),
    (
        Template("oos_math_13_h", "What is the square root of {int_a}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
    (
        Template("oos_math_14", "Quick check for my homework: what is {dec_a} plus {int_b}?"),
        "Not a physics question; it is arithmetic.",
        "How fast does a dropped ball hit the floor?",
    ),
    (
        Template("oos_math_15", "Evaluate {int_a} minus {int_b}."),
        "Not a physics question; it is arithmetic.",
        "How far does a car travel while it speeds up?",
    ),
    (
        Template("oos_math_16_h", "What's {int_b} times {dec_a}?"),
        "Not a physics question; it is arithmetic.",
        "How long does a dropped ball take to fall from a table?",
    ),
)
OOS_SLOTS: dict[str, tuple[str, ...]] = {
    "abstract": (
        "a promise", "Tuesday", "a good idea", "justice", "a rumor", "a secret",
        "an opinion", "a memory", "the alphabet", "a song title", "a wish",
    ),
    "abstract2": (
        "letter Q", "concept of zero", "word silence", "idea of blue sky", "number seven",
    ),
    "emotion": ("jealousy", "boredom", "nostalgia", "pride", "joy", "anxiety", "hope", "regret"),
    "city": ("Lisbon", "Nairobi", "Osaka", "Denver", "Lima", "Oslo", "Hanoi", "Perth"),
    "food": ("pizza", "taco", "bagel", "burger", "pancake", "sushi", "salad", "curry"),
    "company": ("Acme", "a tech giant", "my favorite company", "an airline"),
    "topic_a": ("cats", "summer", "jazz", "basketball"),
    "topic_b": ("dogs", "winter", "rock music", "soccer"),
    "num_a": ("ten", "seven", "twelve", "a hundred", "fifty"),
    "num_b": ("three", "four", "nine", "eleven"),
    "curve": ("a curve", "a parabola", "a straight line", "the line y = 2x + 1", "a circle"),
    "func": ("x squared", "sin x", "e to the x", "the natural log of x", "x cubed"),
    "equation": ("2x + 3 = 11", "x squared = 49", "5x - 4 = 21", "3x = 12"),
    "integer": ("91", "1001", "221", "97"),
    "int_a": ("10", "48", "7", "144"),
    "int_b": ("3", "4", "6", "8", "12"),
    "dec_a": ("3.5", "2.5", "7.2", "0.8", "4.5", "1.25"),
}  # fmt: skip

# --------------------------------------------------------------------------- explanations

EXPLAIN: tuple[Template, ...] = (
    Template("exp_01", "Using [{id}], the {target} comes out to {value} {unit}.{assume}"),
    Template("exp_02", "From [{id}] ({name}), {target} = {value} {unit}.{assume}"),
    Template("exp_03", "Solving [{id}] for the {target} gives {value} {unit}.{assume}"),
    Template("exp_04_h", "The {target} is {value} {unit}, from [{id}].{assume}"),
    Template("exp_05", "{Name} [{id}] gives a {target} of {value} {unit}.{assume}"),
    Template("exp_06", "Rearranging [{id}] for the {target}: {value} {unit}.{assume}"),
    Template("exp_07", "The {target} works out to {value} {unit} by [{id}].{assume}"),
    Template(
        "exp_08",
        "Applying [{id}] ({name}) to the given values, the {target} is {value} {unit}.{assume}",
    ),
    Template(
        "exp_09_h",
        "Plugging the numbers into [{id}] gives {value} {unit} for the {target}.{assume}",
    ),
    Template("exp_10", "With [{id}], we get {target} = {value} {unit}.{assume}"),
    Template(
        "exp_11",
        "Answer: {value} {unit}. That's the {target}, from [{id}] ({name}).{assume}",
    ),
    Template("exp_12", "By [{id}], the {target} equals {value} {unit}.{assume}"),
)  # fmt: skip

# How the assumptions are listed at the end of an explanation. {listed} is "a; b".
ASSUME_LEADS: tuple[str, ...] = (
    " This assumes {listed}.",
    " Assumptions: {listed}.",
    " It assumes {listed}.",
    " This takes {listed} as given.",
    " Assuming {listed}.",
)
