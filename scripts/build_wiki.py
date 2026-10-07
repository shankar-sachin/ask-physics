"""Build the Ask Physics Wiki: the pages in ``docs/wiki/`` plus pages generated from data.

    python scripts/build_wiki.py [build/wiki]

Writes a flat folder of Markdown pages, the layout GitHub Wikis use, which the Wiki
workflow (``.github/workflows/wiki.yml``) publishes to the repository's wiki on every
push to main. Generated:

- one page per equation, named by its id (``kin_v_at.md``), from ``equations.json``;
- ``Equations.md``, every equation grouped by domain;
- ``Constants.md``, from ``constants.json``;
- ``Glossary.md``, from ``docs/GLOSSARY.md``, so the glossary has one source.

Only the standard library is used, so the workflow needs no install.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PAGES = ROOT / "docs" / "wiki"
DATA = ROOT / "src" / "askphysics" / "data"
GLOSSARY = ROOT / "docs" / "GLOSSARY.md"
REPO = "https://github.com/shankar-sachin/ask-physics"

DOMAINS = {
    "kinematics": "Kinematics",
    "dynamics": "Dynamics",
    "energy": "Energy",
    "momentum": "Momentum",
    "gravitation": "Gravitation",
    "fluids": "Fluids",
    "thermodynamics": "Thermodynamics",
    "waves": "Waves",
    "optics": "Optics",
    "electromagnetism": "Electricity and magnetism",
    "modern": "Modern physics",
}


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _range(r: list[float] | None) -> str:
    return "" if r is None else f"{r[0]:g} to {r[1]:g}"


def look_alikes(eq: dict[str, Any], equations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Equations with exactly ``eq``'s variables and units: only the words tell them apart."""

    def signature(e: dict[str, Any]) -> frozenset[tuple[str, str]]:
        return frozenset((v["symbol"], v["unit"]) for v in e["variables"])

    return [o for o in equations if o["id"] != eq["id"] and signature(o) == signature(eq)]


def equation_page(eq: dict[str, Any], equations: list[dict[str, Any]]) -> str:
    """The wiki page for one equation."""
    lines = [
        f"**{eq['name']}** · {DOMAINS.get(eq['domain'], eq['domain'])} · id `{eq['id']}`",
        "",
        "```math",
        eq["latex"],
        "```",
        "",
        "| Symbol | Quantity | Unit | Meaning | Usual range |",
        "|---|---|---|---|---|",
    ]
    for v in eq["variables"]:
        lines.append(
            f"| `{v['symbol']}` | {_cell(v['name'])} | `{v['unit']}` | "
            f"{_cell(v['description'])} | {_range(v.get('typical_range'))} |"
        )
    lines += ["", "### Assumptions", ""]
    lines += [f"- {a}" for a in eq["assumptions"]] or ["- None beyond the equation itself."]
    lines += ["", "### Valid when", ""]
    lines += [f"- {c}" for c in eq["validity_conditions"]] or ["- No extra conditions."]
    twins = look_alikes(eq, equations)
    if twins:
        lines += ["", "### Look-alikes", ""]
        for t in twins:
            words = ", ".join(f"*{w}*" for w in sorted(set(eq["tags"]) - set(t["tags"])))
            lines.append(
                f"[{t['name']}]({t['id']}) has exactly the same variables, so only the "
                f"question's words tell them apart. To get this one, use one of: {words}. "
                "A question that says neither gets an answer flagged as ambiguous."
            )
    lines += [
        "",
        "### Found by",
        "",
        "Questions with these words retrieve this equation: "
        + ", ".join(f"*{t}*" for t in eq["tags"])
        + ".",
        "",
        "### Source",
        "",
        f"{eq['source']}. License: {eq['license']}. "
        f"Entry confidence: {eq['confidence_in_entry']:g}.",
        "",
        f"[Edit this entry]({REPO}/blob/main/src/askphysics/data/equations.json) · "
        "[All equations](Equations)",
        "",
    ]
    return "\n".join(lines)


def equations_index(equations: list[dict[str, Any]]) -> str:
    """``Equations.md``: every equation, grouped by domain."""
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for eq in equations:
        by_domain[eq["domain"]].append(eq)
    lines = [
        f"Ask Physics knows {len(equations)} equations. Every one has a source and a "
        "license, and Noether checks that its units balance before it is allowed in.",
        "",
    ]
    for domain in [*DOMAINS, *sorted(set(by_domain) - set(DOMAINS))]:
        if domain not in by_domain:
            continue
        lines += [f"## {DOMAINS.get(domain, domain)}", ""]
        for eq in sorted(by_domain[domain], key=lambda e: e["name"].lower()):
            lines.append(f"- [{eq['name']}]({eq['id']}): `{eq['id']}`")
        lines.append("")
    return "\n".join(lines)


def constants_page(constants: list[dict[str, Any]]) -> str:
    """``Constants.md``: every physical constant Noether may plug in."""
    lines = [
        "Plans may use these constants without the question stating them.",
        "",
        "| Name | Symbol | Value | Unit | Source |",
        "|---|---|---|---|---|",
    ]
    for c in sorted(constants, key=lambda c: c["name"]):
        lines.append(
            f"| {c['name'].replace('_', ' ')} | `{c['symbol']}` | {c['value']:g} | "
            f"`{c['unit']}` | {_cell(c['source'])} |"
        )
    return "\n".join([*lines, ""])


def glossary_page(text: str) -> str:
    """``docs/GLOSSARY.md`` without its title, which the wiki shows from the page name."""
    return re.sub(r"\A# [^\n]*\n+", "", text)


def build(out: Path) -> list[str]:
    """Write every page under ``out``; returns the page names written."""
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for page in sorted(PAGES.glob("*.md")):
        shutil.copyfile(page, out / page.name)
    equations = json.loads((DATA / "equations.json").read_text(encoding="utf-8"))
    for eq in equations:
        (out / f"{eq['id']}.md").write_text(equation_page(eq, equations), encoding="utf-8")
    (out / "Equations.md").write_text(equations_index(equations), encoding="utf-8")
    constants = json.loads((DATA / "constants.json").read_text(encoding="utf-8"))
    (out / "Constants.md").write_text(constants_page(constants), encoding="utf-8")
    glossary = GLOSSARY.read_text(encoding="utf-8")
    (out / "Glossary.md").write_text(glossary_page(glossary), encoding="utf-8")
    return sorted(p.stem for p in out.glob("*.md"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, nargs="?", default=Path("build/wiki"))
    args = parser.parse_args()
    pages = build(args.out)
    print(f"{len(pages)} pages -> {args.out}")


if __name__ == "__main__":
    main()
