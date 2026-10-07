"""Public-domain books from Project Gutenberg for the prose corpus (ADR-017).

Used by ``scripts/build_corpus.py``. A book qualifies when it is English text and every
person the catalog lists for it (author, translator, editor) died in or before
``latest_death`` (1955 by default). That makes it public domain in the US, where Project
Gutenberg operates, and in countries whose copyright runs for the author's life plus 70
years. The Project Gutenberg header, footer, and every mention of the trademark are
removed, as its license asks of anyone redistributing the text without its name.
"""

from __future__ import annotations

import csv
import re
import time
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from extract_openstax import to_ascii

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
TEXT_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
USER_AGENT = "ask-physics-corpus/1.0 (+https://github.com/shankar-sachin/ask-physics)"
MIN_WORDS = 8

_START = re.compile(r"^\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
_END = re.compile(r"^\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M | re.I)
# "Faraday, Michael, 1791-1867", "Aristotle, 384 BCE-322 BCE", "Crookes, William, -1919".
_LIFESPAN = re.compile(r"(?:\d+\s*(?:BCE)?)?\s*-\s*(\d+)\s*(BCE)?")
_BRACKETED = re.compile(r"\[(?:Illustration|Footnote|Sidenote|Transcriber)[^\]]*\]", re.I)
_FOOTNOTE_MARK = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class Book:
    """One catalog entry."""

    id: int
    title: str
    authors: str
    locc: str
    subjects: str


def death_years(authors: str) -> list[int | None]:
    """The death year of each person in a catalog ``Authors`` field (None when unknown)."""
    years: list[int | None] = []
    for person in filter(None, (p.strip() for p in authors.split(";"))):
        match = _LIFESPAN.search(person)
        if match is None:
            years.append(None)
        else:
            year = int(match.group(1))
            years.append(-year if match.group(2) else year)
    return years


def public_domain(authors: str, latest_death: int) -> bool:
    """Whether everyone credited died in or before ``latest_death`` (and anyone is credited)."""
    years = death_years(authors)
    return bool(years) and all(y is not None and y <= latest_death for y in years)


def read_catalog(path: Path) -> list[Book]:
    """English text entries of ``pg_catalog.csv``."""
    books: list[Book] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("Type") != "Text" or row.get("Language") != "en":
                continue
            books.append(
                Book(
                    id=int(row["Text#"]),
                    title=" ".join(row["Title"].split()),
                    authors=row.get("Authors", ""),
                    locc=row.get("LoCC", ""),
                    subjects=row.get("Subjects", ""),
                )
            )
    return books


def select(
    books: Iterable[Book],
    loccs: Sequence[str],
    latest_death: int,
    exclude_subjects: Sequence[str] = (),
) -> list[Book]:
    """Public-domain books in the given Library of Congress classes, in ``loccs`` order.

    A book counts for the first class in ``loccs`` that one of its codes starts with, so
    putting the sciences first ("QC" physics, "QB" astronomy) ranks them ahead of
    literature ("PR", "PS"). Within a class, books keep catalog order.
    """
    ranked: list[tuple[int, int, Book]] = []
    for book in books:
        if any(s.lower() in book.subjects.lower() for s in exclude_subjects):
            continue
        if not public_domain(book.authors, latest_death):
            continue
        codes = [c.strip() for c in book.locc.split(";") if c.strip()]
        rank = next(
            (i for i, cls in enumerate(loccs) if any(c.startswith(cls) for c in codes)), None
        )
        if rank is not None:
            ranked.append((rank, book.id, book))
    return [book for _, _, book in sorted(ranked)]


def body(text: str) -> str | None:
    """The book's own text, without Project Gutenberg's header and footer.

    None when the markers are missing, or when the header says the eBook is copyrighted
    (a handful are, with the holder's permission; they are not ours to use).
    """
    start, end = _START.search(text), _END.search(text)
    if start is None or end is None or end.start() <= start.end():
        return None
    if re.search(r"\bcopyrighted\b", text[: start.start()], re.I):
        return None
    return text[start.end() : end.start()]


def paragraphs(text: str) -> Iterator[str]:
    """Prose paragraphs of a book body: unwrapped, ASCII, at least ``MIN_WORDS`` words.

    Skipped: headings (all capitals, "CHAPTER ..."), verse and tables (short lines),
    anything that names Project Gutenberg, and anything mostly not letters.
    """
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [line.strip() for line in block.strip().splitlines() if line.strip()]
        if not lines:
            continue
        if len(lines) >= 3 and sorted(len(line) for line in lines)[len(lines) // 2] < 40:
            continue  # verse, a table, or a list
        joined = _FOOTNOTE_MARK.sub("", _BRACKETED.sub("", " ".join(lines))).replace("_", "")
        joined = re.sub(r"\s+", " ", to_ascii(joined)).strip()
        if len(joined.split()) < MIN_WORDS or "gutenberg" in joined.lower():
            continue
        letters = sum(c.isalpha() for c in joined)
        if letters < 0.6 * len(joined.replace(" ", "")) or joined.isupper():
            continue
        if re.match(r"(?:CHAPTER|BOOK|PART|SECTION)\b", joined):
            continue
        yield joined


def fetch(url: str, dest: Path, delay: float = 2.0) -> Path:
    """Download ``url`` to ``dest`` unless it is already there, pausing ``delay`` seconds."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    time.sleep(delay)  # be a polite client of a volunteer-run site
    return dest
