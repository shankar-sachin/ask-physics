"""Rewrite how people write quantities into the forms the pipeline reads.

Real questions write numbers and units in ways the plan decoder's extractors
(``lm/formats.py``) don't recognize: "2,000-kg car", "3.00 × 10⁸ m/s",
"9.8 m/s²", "kg·m/s", "20 meters per second". Without this, a perfect model
would still be handed the wrong options: "9.8 m/s²" read as a speed and a stray
2. ``normalize_question`` runs once, when a question enters the pipeline, so
the models, retrieval, and Noether all see the same canonical text.

Every rewrite is conservative: it only fires on a number next to something
Pint accepts as a unit, and it never changes a number's value.
"""

from __future__ import annotations

import re

from askphysics.solver.units import is_valid_unit, quantity

_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_RUN = re.compile(r"([A-Za-zµΩ)])([⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)")
_MINUS = str.maketrans({"−": "-", "–": "-", "‒": "-"})  # unicode minus, en dash, figure dash
_UNIT_DOT = re.compile(r"(?<=[A-Za-zµΩ])\s*[·⋅•∙]\s*(?=[A-Za-zµΩ])")
_THOUSANDS = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d,])")
_SUP = "[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+"
_TIMES_TEN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:×|x|X|\*)\s*10\s*(?:\^|\*\*)?\s*([-+]?\d+)(?![\d.])"
)
_TIMES_TEN_SUP = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:×|x|X|\*)\s*10(" + _SUP + ")")
_POWER_OF_TEN = re.compile(r"(?<![\w.])10(" + _SUP + ")")
_HYPHEN_UNIT = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)-([A-Za-zµΩ][A-Za-z0-9µΩ/^*]*)")
_PER = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?(?:e[-+]?\d+)?)\s+([A-Za-z]+)\s+per\s+([A-Za-z]+)(\s+squared)?\b"
)
_DEGREES = re.compile(r"(\d)\s*°\s*([CF])\b")
# "20 degrees Celsius", "68 degrees F", "15 Celsius", "-4 deg F". Bare "C" is a coulomb.
_DEGREE_WORDS = re.compile(
    r"(\d)\s*(?:(?:degrees?|deg)\s+(?:(Celsius|centigrade|C)|(Fahrenheit|F))|"
    r"(Celsius|centigrade)|(Fahrenheit))\b",
    re.IGNORECASE,
)


def _degree_words(m: re.Match[str]) -> str:
    celsius = m.group(2) or m.group(4)
    return f"{m.group(1)} {'degC' if celsius else 'degF'}"


_MICRO = re.compile(r"(\d\s*)[µμ](?=[A-Za-z])")  # micro sign or Greek mu, as a unit prefix
_OHM = re.compile(r"(\d\s*)([kMm]?)[ΩΩ]")  # Greek omega or the ohm sign
_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_WORD_NUMBER = re.compile(
    r"\b(?:(?P<tens>"
    + "|".join(_TENS)
    + r")(?:[- ](?P<unit_digit>"
    + "|".join(_ONES[1:10])
    + r"))?|(?P<ones>"
    + "|".join(_ONES)
    + r"))(?:\s+(?P<scale>hundred|thousand))?"
    + r"\s+(?P<unit>[A-Za-z][A-Za-z/^*]*)\b",
    re.IGNORECASE,
)
# Words Pint reads as units that are usually just English: "one in a million" is no inch.
_NOT_UNITS = frozenset(("in", "at", "a", "are", "as", "us", "ha", "mil", "point"))


def _superscript(match: re.Match[str]) -> str:
    return f"{match.group(1)}^{match.group(2).translate(_SUPERSCRIPTS)}"


def _times_ten(match: re.Match[str]) -> str:
    return f"{match.group(1)}e{int(match.group(2))}"


def _hyphen_unit(match: re.Match[str]) -> str:
    number, unit = match.groups()
    return f"{number} {unit}" if is_valid_unit(unit) else match.group()


def _word_number(match: re.Match[str]) -> str:
    unit = match.group("unit")
    # Only units with dimensions: "the time for one cycle" names a variable, and a cycle,
    # like a radian, is dimensionless to Pint.
    if unit.lower() in _NOT_UNITS or not is_valid_unit(unit) or quantity(1, unit).dimensionless:
        return match.group()
    # "One" is in too many names ("the mass of one molecule", "object one") to read as 1.
    if (match.group("ones") or "").lower() == "one" and not match.group("scale"):
        return match.group()
    if match.group("tens"):
        value = 10 * (_TENS.index(match.group("tens").lower()) + 2)
        if match.group("unit_digit"):
            value += _ONES.index(match.group("unit_digit").lower())
    else:
        value = _ONES.index(match.group("ones").lower())
    scale = (match.group("scale") or "").lower()
    value *= {"hundred": 100, "thousand": 1000}.get(scale, 1)
    return f"{value} {unit}"


def _per(match: re.Match[str]) -> str:
    number, top, bottom, squared = match.groups()
    unit = f"{top}/{bottom}" + ("^2" if squared else "")
    return f"{number} {unit}" if is_valid_unit(unit) else match.group()


def normalize_question(text: str) -> str:
    """Canonical spellings of the quantities in ``text``; everything else is untouched.

    - Unicode minus signs and dashes between digits become ``-``.
    - Thousands separators go: "2,000" is "2000".
    - Powers of ten become e-notation: "3.00 × 10^8", "3.00×10⁸" are "3.00e8", and
      "10⁻¹¹" is "1e-11".
    - Superscripts in units become exponents: "m/s²" is "m/s^2", "s⁻¹" is "s^-1".
    - A middle dot between units is a product: "kg·m/s" is "kg*m/s".
    - A hyphen between a number and a unit goes: "55-kg" is "55 kg".
    - "per" between units is a slash: "20 meters per second" is "20 meters/second",
      "9.8 meters per second squared" is "9.8 meters/second^2".
    - Degree temperatures get Pint's names: "25 °C", "25 degrees Celsius", and "25 Celsius"
      are "25 degC"; Fahrenheit likewise is "degF".
    - Symbols the extractors can't start a unit with get letters: "2.5 µC" is "2.5 uC",
      "220 Ω" is "220 ohm", "4.7 kΩ" is "4.7 kohm".
    - A number spelled out before a unit becomes digits: "an eight kilogram ball" is "an
      8 kilogram ball", "twenty-five meters" is "25 meters". Without a unit after it, a
      word stays a word ("ten divided by three").
    """
    text = text.translate(_MINUS)
    text = _WORD_NUMBER.sub(_word_number, text)
    text = _THOUSANDS.sub(lambda m: m.group(1).replace(",", ""), text)
    text = _TIMES_TEN.sub(_times_ten, text)
    text = _TIMES_TEN_SUP.sub(
        lambda m: f"{m.group(1)}e{int(m.group(2).translate(_SUPERSCRIPTS))}", text
    )
    text = _POWER_OF_TEN.sub(lambda m: f"1e{int(m.group(1).translate(_SUPERSCRIPTS))}", text)
    text = _SUPERSCRIPT_RUN.sub(_superscript, text)
    text = _UNIT_DOT.sub("*", text)
    text = _HYPHEN_UNIT.sub(_hyphen_unit, text)
    text = _PER.sub(_per, text)
    text = _DEGREES.sub(lambda m: f"{m.group(1)} deg{m.group(2)}", text)
    text = _DEGREE_WORDS.sub(_degree_words, text)
    text = _MICRO.sub(r"\1u", text)
    text = _OHM.sub(r"\1\2ohm", text)
    return text
