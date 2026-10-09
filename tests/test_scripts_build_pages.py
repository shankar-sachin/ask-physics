"""``scripts/build_pages.py``: the docs pages built from ``docs/pages/`` and the database."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_pages

from askphysics.data.loader import DataStore


@pytest.fixture(scope="module")
def pages(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("pages")
    build_pages.build(out)
    return out


def test_every_equation_has_a_page(pages: Path, store: DataStore) -> None:
    index = (pages / "Equations.md").read_text()
    for eq in store.equations.values():
        page = (pages / f"{eq.id}.md").read_text()
        assert eq.latex in page and eq.source in page and eq.license in page
        assert all(f"`{v.symbol}`" in page for v in eq.variables)
        assert f"]({eq.id})" in index


def test_look_alikes_say_which_words_pick_them(pages: Path) -> None:
    page = (pages / "parallel_resistors.md").read_text()
    assert "[Two resistors in series](series_resistors)" in page
    assert "use one of: *parallel*." in page
    assert "Look-alikes" not in (pages / "kinetic_energy.md").read_text()


def test_every_page_link_resolves(pages: Path) -> None:
    names = {p.stem for p in pages.glob("*.md")}
    for page in pages.glob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text()):
            if not target.startswith(("http://", "https://")):
                assert target in names, f"{page.name} links to missing page {target}"


def test_constants_and_glossary_come_from_their_sources(pages: Path, store: DataStore) -> None:
    constants = (pages / "Constants.md").read_text()
    assert all(f"`{c.symbol}`" in constants for c in store.constants.values())
    glossary = (pages / "Glossary.md").read_text()
    assert not glossary.startswith("# ") and "**Constrained decoding.**" in glossary


def test_rebuilding_replaces_stale_pages(tmp_path: Path) -> None:
    (tmp_path / "removed_equation.md").write_text("old")
    pages = build_pages.build(tmp_path)
    assert "removed_equation" not in pages and "Home" in pages
    assert not (tmp_path / "removed_equation.md").exists()
