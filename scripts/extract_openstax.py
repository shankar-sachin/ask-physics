"""Extract the prose and exercises of OpenStax *Physics* (2020) (ADR-016).

    python scripts/extract_openstax.py [--source DIR] [--out third_party/openstax-physics]

Writes ``prose.jsonl`` (body paragraphs, for language-model training) and
``questions.jsonl`` (exercises, for the real-question eval and classify examples).

The book is CC BY 4.0 (its LICENSE file and collection metadata say so). Only prose
paragraphs are kept: no figures, captions, media, tables, exercises, display equations, or
teacher-support material (it quotes state standards that aren't OpenStax's to license).
Inline math is written out as text ("v = d / t"), an auto-numbered reference to a figure
or table reads "the figure" or "the table", and a paragraph that embeds an exercise or
starts mid-sentence (after a display equation) is skipped. Exercises are kept with their
answer options and solution text, except those that need a figure or table. The output
keeps the book's license and attribution; it is never relabelled MIT.

Without ``--source`` the repository is cloned at the pinned commit below.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from pathlib import Path

REPO = "https://github.com/openstax/osbooks-physics.git"
COMMIT = "dfdfd7a5356ecdd42e504de3df50d9153e33ea49"
COLLECTION = "collections/physics.collection.xml"

CNXML = "{http://cnx.rice.edu/cnxml}"
COLXML = "{http://cnx.rice.edu/collxml}"
MDML = "{http://cnx.rice.edu/mdml}"
MATHML = "{http://www.w3.org/1998/Math/MathML}"

# Containers whose paragraphs are not prose.
SKIP = {
    f"{CNXML}{tag}"
    for tag in ("figure", "table", "exercise", "equation", "media", "footnote", "list", "glossary")
}
# The preface is author bios and course marketing, not physics.
SKIP_CHAPTERS = ("Front matter",)
# The decoder only writes printable ASCII, so the prose is folded to it.
ASCII = str.maketrans(
    {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2014": " - ",
        "\u2013": "-", "\u2212": "-", "\u2026": "...", "\u00a0": " ", "\u00d7": "x",
        "\u00b7": "*", "\u2248": "~", "\u2264": "<=", "\u2265": ">=", "\u00b0": " degrees ",
        "\u03c0": "pi", "\u03a9": "ohm", "\u03bc": "mu", "\u00b5": "mu", "\u0394": "delta",
        "\u03b8": "theta", "\u03bb": "lambda", "\u03c9": "omega", "\u03c1": "rho",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma", "\u03b5": "epsilon",
        "\u03c4": "tau", "\u03c3": "sigma", "\u03a3": "sum", "\u2192": "->",
    }
)  # fmt: skip


def to_ascii(text: str) -> str:
    """Curly quotes, dashes, and Greek letters as ASCII; accents dropped (Schrodinger)."""
    folded = unicodedata.normalize("NFKD", text.translate(ASCII))
    return "".join(ch for ch in folded if ch.isascii() and not unicodedata.combining(ch))


# Notes for teachers, and the section objectives (a list, not prose).
SKIP_NOTES = ("teacher", "learning-objectives")
POWER_START = set("0123456789-+\u2212\u2013")
REFERENCES = {"Figure": "the figure", "Table": "the table"}
MIN_WORDS = 8


def math_text(node: ET.Element) -> str:
    """Inline MathML as plain text: "v = d / t", "x^2", "sqrt(2 g h)"."""
    tag = node.tag.removeprefix(MATHML)
    kids = [math_text(k) for k in node]
    if tag in ("mi", "mn", "mo", "mtext"):
        return (node.text or "").strip()
    if tag == "msup" and len(kids) == 2:
        return f"{kids[0]}^{kids[1].replace(' ', '')}"
    if tag == "msub" and len(kids) == 2:
        return f"{kids[0]}{kids[1]}"
    if tag == "msubsup" and len(kids) == 3:
        return f"{kids[0]}{kids[1]}^{kids[2].replace(' ', '')}"
    if tag == "mfrac" and len(kids) == 2:
        return f"{kids[0]} / {kids[1]}"
    if tag == "msqrt":
        return f"sqrt({' '.join(kids)})"
    return " ".join(k for k in kids if k)


def para_text(para: ET.Element, *, sentence: bool = True) -> str | None:
    """A paragraph as plain text, or None if it can't be read without its page.

    ``sentence`` also rejects text that starts mid-sentence; answer options ("$0.69",
    "zero") are read with it off.
    """
    parts: list[str] = []

    def walk(node: ET.Element) -> bool:
        if node.tag == f"{MATHML}math":
            parts.append(" " + math_text(node) + " ")
            parts.append(node.tail or "")
            return True
        if node.tag == f"{CNXML}link" and not "".join(node.itertext()).strip():
            target = node.attrib.get("target-id", "")
            kind = next((v for k, v in REFERENCES.items() if target.startswith(k)), None)
            if kind is None:
                return False  # an embedded exercise or an unknown reference
            parts.append(kind)
            parts.append(node.tail or "")
            return True
        # "10<sup>8</sup>" is 10^8 and "m/s<sup>2</sup>" is m/s^2, but "18<sup>th</sup>" is 18th.
        if node.tag == f"{CNXML}sup" and (node.text or "").strip()[:1] in POWER_START:
            parts.append("^")
        parts.append(node.text or "")
        for child in node:
            if not walk(child):
                return False
        if node is not para:
            parts.append(node.tail or "")
        return True

    if not walk(para):
        return None
    text = re.sub(r"\s+", " ", to_ascii("".join(parts))).strip()
    text = re.sub(r"\s+([.,;:!?)])", r"\1", text)
    starts_well = text[:1].isupper() or text[:1].isdigit() or text[:1] in ('"', "\u201c", "(")
    if not text or (sentence and not starts_well):
        return None  # starts mid-sentence, after a display equation
    return text


# Sections whose exercises are questions to answer: practice and end-of-chapter problems,
# the end-of-chapter conceptual questions, and the test prep. Labs, worked examples, and
# the performance tasks are not.
EXERCISE_SECTIONS = (
    "practice-problems", "problems", "check-understanding", "concept", "critical-thinking",
    "multiple-choice", "short-answer", "extended-response",
)  # fmt: skip


def exercise(node: ET.Element) -> dict[str, object] | None:
    """One exercise as {text, options, solution}, or None if it needs a figure or table."""
    problem = node.find(f"{CNXML}problem")
    if problem is None or any(problem.iter(f"{CNXML}figure")) or any(problem.iter(f"{CNXML}table")):
        return None
    paras = [para_text(p) for p in problem.findall(f"{CNXML}para")]
    if not paras or any(p is None for p in paras):
        return None
    options: list[str] = []
    for lst in problem.findall(f"{CNXML}list"):
        for item in lst.findall(f"{CNXML}item"):
            text = para_text(item, sentence=False)
            if not text:
                return None  # an option we can't read makes the question unanswerable
            options.append(text)
    solution = node.find(f"{CNXML}solution")
    answer = None
    if solution is not None:
        parts = [para_text(p, sentence=False) for p in solution.iter(f"{CNXML}para")]
        answer = " ".join(p for p in parts if p) or None
    return {"text": " ".join(p for p in paras if p), "options": options, "solution": answer}


def exercises(source: Path, collection: str) -> Iterator[dict[str, object]]:
    """Every exercise in ``EXERCISE_SECTIONS``, in reading order, with where it came from."""
    for chapter, module in modules(source, collection):
        path = source / "modules" / module / "index.cnxml"
        if chapter in SKIP_CHAPTERS or not path.exists():
            continue
        for section in ET.parse(path).getroot().iter(f"{CNXML}section"):
            kind = section.attrib.get("class", "")
            if kind not in EXERCISE_SECTIONS:
                continue
            for node in section.iter(f"{CNXML}exercise"):
                row = exercise(node)
                if row is not None:
                    yield {
                        "id": node.attrib["id"],
                        "module": module,
                        "chapter": to_ascii(chapter),
                        "kind": kind,
                        **row,
                    }


def clone(dest: Path, repo: str = REPO, commit: str = COMMIT) -> Path:
    """Check out ``repo`` at ``commit``: only the text (no images), so it stays small."""
    git = ["git", "-C", str(dest)]
    subprocess.run(["git", "init", "-q", str(dest)], check=True)
    subprocess.run(
        [*git, "sparse-checkout", "set", "--no-cone", "/LICENSE", "/collections/*",
         "/modules/*/index.cnxml"],
        check=True,
    )  # fmt: skip
    subprocess.run(
        [*git, "fetch", "-q", "--depth", "1", "--filter=blob:none", repo, commit], check=True
    )
    # The checkout fetches the files lazily, and git prints the server's progress for each
    # batch; keep it unless something fails.
    done = subprocess.run([*git, "checkout", "-q", "FETCH_HEAD"], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"git checkout of {repo} at {commit} failed:\n{done.stderr}")
    return dest


def verify_cc_by(source: Path, collections: list[str]) -> str:
    """The checkout's CC BY 4.0 license text, after checking every book is CC BY 4.0 too.

    Both must hold at the commit being read: OpenStax relicensed most books to CC BY-NC-SA
    in March 2026, and only versions published before that are CC BY (ADR-017).

    Raises:
        SystemExit: the LICENSE or a book's metadata is anything else.
    """
    license_text = (source / "LICENSE").read_text(encoding="utf-8")
    if not license_text.startswith("Attribution 4.0 International"):
        raise SystemExit(f"{source}: LICENSE is not CC BY 4.0; refusing to extract")
    for collection in collections:
        meta = ET.parse(source / collection).getroot().find(f"{COLXML}metadata")
        lic = meta.find(f"{MDML}license") if meta is not None else None
        url = lic.attrib.get("url", "") if lic is not None else ""
        if "/licenses/by/4.0" not in url:
            raise SystemExit(f"{collection}: book license is {url or 'missing'}, not CC BY 4.0")
    return license_text


def book(source: Path, collection: str) -> Iterator[tuple[str, str, str]]:
    """(chapter, module, paragraph) for one book's prose, in reading order."""
    for chapter, module in modules(source, collection):
        path = source / "modules" / module / "index.cnxml"
        if chapter in SKIP_CHAPTERS or not path.exists():
            continue
        for text in paragraphs(path):
            yield to_ascii(chapter), module, text


def modules(source: Path, collection: str = COLLECTION) -> Iterator[tuple[str, str]]:
    """(chapter title, module id) in reading order."""
    root = ET.parse(source / collection).getroot()

    def walk(node: ET.Element, chapter: str) -> Iterator[tuple[str, str]]:
        for child in node:
            if child.tag == f"{COLXML}module":
                yield chapter, child.attrib["document"]
            elif child.tag == f"{COLXML}subcollection":
                title = child.findtext(f"{MDML}title") or chapter
                content = child.find(f"{COLXML}content")
                if content is not None:
                    yield from walk(content, title)

    content = root.find(f"{COLXML}content")
    if content is not None:
        yield from walk(content, "Front matter")


def paragraphs(path: Path) -> Iterator[str]:
    """Body paragraphs of one module, as plain text."""
    content = ET.parse(path).getroot().find(f"{CNXML}content")
    if content is None:
        return

    def walk(node: ET.Element) -> Iterator[ET.Element]:
        for child in node:
            if child.tag in SKIP:
                continue
            if child.tag == f"{CNXML}note" and any(
                c in child.attrib.get("class", "") for c in SKIP_NOTES
            ):
                continue
            if child.tag == f"{CNXML}para":
                yield child
            else:
                yield from walk(child)

    for para in walk(content):
        text = para_text(para)
        if text is not None and len(text.split()) >= MIN_WORDS:
            yield text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="A local clone of osbooks-physics.")
    parser.add_argument("--out", type=Path, default=Path("third_party/openstax-physics"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        source = args.source or clone(Path(tmp) / "osbooks-physics")
        if (
            not (source / "LICENSE")
            .read_text(encoding="utf-8")
            .startswith("Attribution 4.0 International")
        ):
            raise SystemExit("the source LICENSE is not CC BY 4.0; refusing to extract")
        license_text = verify_cc_by(source, [COLLECTION])
        args.out.mkdir(parents=True, exist_ok=True)
        rows = 0
        words = 0
        with (args.out / "prose.jsonl").open("w", encoding="utf-8") as f:
            for chapter, module, text in book(source, COLLECTION):
                f.write(json.dumps({"module": module, "chapter": chapter, "text": text}) + "\n")
                rows += 1
                words += len(text.split())
        questions = 0
        with (args.out / "questions.jsonl").open("w", encoding="utf-8") as f:
            for row in exercises(source, COLLECTION):
                f.write(json.dumps(row) + "\n")
                questions += 1
        (args.out / "LICENSE").write_text(license_text, encoding="utf-8")
    print(f"{rows} paragraphs, {words} words -> {args.out / 'prose.jsonl'}")
    print(f"{questions} exercises -> {args.out / 'questions.jsonl'}")


if __name__ == "__main__":
    main()
