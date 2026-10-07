"""``scripts/gutenberg.py``: which books count as public domain, and what text survives."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import gutenberg

CATALOG = """Text#,Type,Issued,Title,Language,Authors,Subjects,LoCC,Bookshelves
1,Text,2001-01-01,The Chemical History of a Candle,en,"Faraday, Michael, 1791-1867",Combustion,QD,
2,Text,2001-01-01,Relativity,en,"Einstein, Albert, 1879-1955; Lawson, Robert W., 1890-1960 [Translator]",Relativity,QC,
3,Text,2001-01-01,Pride and Prejudice,en,"Austen, Jane, 1775-1817",Fiction,PR,
4,Text,2001-01-01,Matter and Motion,en,"Maxwell, James Clerk, 1831-1879",Dynamics,QC,
5,Text,2001-01-01,Some Poems,en,"Poet, A., 1800-1850",Poetry,PR,
6,Text,2001-01-01,Anonymous Notes,en,Anonymous,Science,Q,
7,Sound,2001-01-01,An Audiobook,en,"Reader, A., 1800-1850",Physics,QC,
8,Text,2001-01-01,Physique,fr,"Auteur, Un, 1800-1850",Physics,QC,
"""


def test_death_years() -> None:
    assert gutenberg.death_years("Faraday, Michael, 1791-1867") == [1867]
    assert gutenberg.death_years("Aristotle, 384 BCE-322 BCE") == [-322]
    assert gutenberg.death_years("A, B, 1800-1850; C, D, 1890-1960 [Translator]") == [1850, 1960]
    assert gutenberg.death_years("Anonymous") == [None]


def test_everyone_credited_must_be_long_dead() -> None:
    assert gutenberg.public_domain("Faraday, Michael, 1791-1867", 1955)
    # Einstein died in 1955, but his translator in 1960: not public domain everywhere.
    assert not gutenberg.public_domain(
        "Einstein, Albert, 1879-1955; Lawson, Robert W., 1890-1960 [Translator]", 1955
    )
    assert not gutenberg.public_domain("Anonymous", 1955)
    assert not gutenberg.public_domain("", 1955)


def test_select_ranks_science_first(tmp_path: Path) -> None:
    (tmp_path / "c.csv").write_text(CATALOG)
    books = gutenberg.read_catalog(tmp_path / "c.csv")
    assert {b.id for b in books} == {1, 2, 3, 4, 5, 6}  # English text only
    picked = gutenberg.select(books, ["QC", "QD", "PR"], 1955, ["Poetry"])
    assert [b.id for b in picked] == [4, 1, 3]  # physics, chemistry, then literature


BOOK = """The Project Gutenberg eBook of A Test

*** START OF THE PROJECT GUTENBERG EBOOK A TEST ***

CHAPTER I.

The light that falls upon a body is partly reflected, and the rest enters
it, where it may be absorbed or pass through to the other side.

[Illustration: A prism]

When a ray of _white_ light passes through a prism[1], it is spread into a
band of colours, which Newton called the spectrum.

  Twinkle, twinkle,
  little star, how
  I wonder what
  you are up there

This file was produced by volunteers for Project Gutenberg in a long sentence here.

*** END OF THE PROJECT GUTENBERG EBOOK A TEST ***
License text that must not survive.
"""


def test_only_the_books_own_prose_survives() -> None:
    body = gutenberg.body(BOOK)
    assert body is not None and "License text" not in body
    assert list(gutenberg.paragraphs(body)) == [
        "The light that falls upon a body is partly reflected, and the rest enters it, where "
        "it may be absorbed or pass through to the other side.",
        "When a ray of white light passes through a prism, it is spread into a band of "
        "colours, which Newton called the spectrum.",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "No markers at all.",
        "This eBook is copyrighted.\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nx\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK X ***",
    ],
)
def test_unusable_files_are_skipped(text: str) -> None:
    assert gutenberg.body(text) is None
