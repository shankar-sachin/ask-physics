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

from askphysics.solver.units import is_valid_unit

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
_MICRO = re.compile(r"(\d\s*)[µμ](?=[A-Za-z])")  # micro sign or Greek mu, as a unit prefix
_OHM = re.compile(r"(\d\s*)([kMm]?)[ΩΩ]")  # Greek omega or the ohm sign


def _superscript(match: re.Match[str]) -> str:
    return f"{match.group(1)}^{match.group(2).translate(_SUPERSCRIPTS)}"


def _times_ten(match: re.Match[str]) -> str:
    return f"{match.group(1)}e{int(match.group(2))}"


def _hyphen_unit(match: re.Match[str]) -> str:
    number, unit = match.groups()
    return f"{number} {unit}" if is_valid_unit(unit) else match.group()


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
    - Degree temperatures get Pint's names: "25 °C" is "25 degC".
    - Symbols the extractors can't start a unit with get letters: "2.5 µC" is "2.5 uC",
      "220 Ω" is "220 ohm", "4.7 kΩ" is "4.7 kohm".
    """
    text = text.translate(_MINUS)
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
    text = _MICRO.sub(r"\1u", text)
    text = _OHM.sub(r"\1\2ohm", text)
    return text
