"""Compare two ``model eval`` reports side by side.

    python3 scripts/compare_evals.py OLD/eval.json NEW/eval.json

Prints each score for both, the change, and which confidently wrong questions are new
and which were fixed. Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROWS = (
    ("category_accuracy", "right category", True),
    ("equation_accuracy", "right equation", True),
    ("target_accuracy", "right target", True),
    ("knowns_accuracy", "right numbers and units", True),
    ("valid_plan_rate", "valid plan (right answer)", True),
    ("routed_right_rate", "right answer as ask gives it", True),
    ("flagged_wrong_rate", "wrong, but flagged or refused", False),
    ("confidently_wrong_rate", "confidently wrong", False),
)


def compare(old: dict[str, Any], new: dict[str, Any]) -> str:
    """The comparison as text."""
    lines = [f"{'check':32} {'old':>8} {'new':>8} {'change':>9}"]
    for key, label, higher_is_better in ROWS:
        a, b = float(old.get(key, 0.0)), float(new.get(key, 0.0))
        delta = b - a
        better = (delta > 0) == higher_is_better
        mark = "" if abs(delta) < 1e-9 else (" better" if better else " WORSE")
        lines.append(f"{label:32} {a:8.1%} {b:8.1%} {delta:+9.1%}{mark}")

    def wrong(report: dict[str, Any]) -> set[str]:
        return {str(f["question"]) for f in report.get("confidently_wrong", [])}

    fixed, broke = sorted(wrong(old) - wrong(new)), sorted(wrong(new) - wrong(old))
    if fixed:
        lines += ["", "No longer confidently wrong:", *(f"  {q}" for q in fixed)]
    if broke:
        lines += ["", "Newly confidently wrong:", *(f"  {q}" for q in broke)]
    if old.get("plan_examples") != new.get("plan_examples"):
        lines += ["", "Different question counts: the two reports aren't on the same set."]
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    reports = []
    for arg in argv:
        path = Path(arg)
        if not path.is_file():
            print(f"no eval report at {path}; run askphysics-dev model eval first", file=sys.stderr)
            return 1
        reports.append(json.loads(path.read_text(encoding="utf-8")))
    print(compare(reports[0], reports[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
