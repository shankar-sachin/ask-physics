"""The website's bridge to the pipeline: a question in, display-ready JSON out.

The site (``web/``) runs this in the visitor's browser under Pyodide, so it
imports nothing that needs a terminal or torch. It mirrors the CLI answer card
(``askphysics.ui.answer_card``): same fields, same formatting.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from askphysics import __version__
from askphysics.config import Settings
from askphysics.models import Answer, Equation
from askphysics.pipeline import Pipeline
from askphysics.pretty import pretty_equation, pretty_number, pretty_symbol, pretty_unit

REDIRECT_MARKER = " A close question that can be answered:"


@lru_cache(maxsize=1)
def _pipeline() -> Pipeline:
    return Pipeline.from_settings(Settings.from_env())


def answer_payload(answer: Answer, equations: Mapping[str, Equation]) -> dict[str, Any]:
    """The Answer plus pre-formatted strings, so the page never formats numbers itself."""
    display: dict[str, Any] = {
        "result": None,
        "why": None,
        "equations": [],
        "inputs": [],
    }
    if answer.final_value is not None:
        display["result"] = {
            "value": pretty_number(answer.final_value),
            "unit": pretty_unit(answer.unit or ""),
        }
    if answer.status == "refused" and answer.redirect:
        display["why"] = answer.explanation.split(REDIRECT_MARKER)[0]
    for ref in answer.equations_used:
        eq = equations.get(ref.id)
        display["equations"].append(
            {
                "id": ref.id,
                "name": ref.name,
                "math": pretty_equation(eq.sympy_expr) if eq else None,
                "source": eq.source if eq else None,
            }
        )
    for known in answer.inputs:
        display["inputs"].append(
            {
                "symbol": pretty_symbol(known.symbol),
                "value": pretty_number(known.value),
                "unit": pretty_unit(known.unit),
                "origin": known.origin,
            }
        )
    return {"answer": answer.model_dump(mode="json"), "display": display}


def ask(question: str) -> str:
    """Run the full pipeline on ``question`` and return the payload as JSON."""
    pipeline = _pipeline()
    answer = pipeline.run(question)
    return json.dumps(answer_payload(answer, pipeline.data.equations), ensure_ascii=False)


def info() -> str:
    """Version and seed-data counts, for the page footer."""
    return json.dumps({"version": __version__, "data": _pipeline().data.summary()})
