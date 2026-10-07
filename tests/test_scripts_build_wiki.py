"""``scripts/build_wiki.py``: the GitHub Wiki built from ``docs/wiki/`` and the database."""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_wiki

from askphysics.data.loader import DataStore


@pytest.fixture(scope="module")
def wiki(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("wiki")
    build_wiki.build(out)
    return out


def test_every_equation_has_a_page(wiki: Path, store: DataStore) -> None:
    index = (wiki / "Equations.md").read_text()
    for eq in store.equations.values():
        page = (wiki / f"{eq.id}.md").read_text()
        assert eq.latex in page and eq.source in page and eq.license in page
        assert all(f"`{v.symbol}`" in page for v in eq.variables)
        assert f"]({eq.id})" in index


def test_look_alikes_say_which_words_pick_them(wiki: Path) -> None:
    page = (wiki / "parallel_resistors.md").read_text()
    assert "[Two resistors in series](series_resistors)" in page
    assert "use one of: *parallel*." in page
    assert "Look-alikes" not in (wiki / "kinetic_energy.md").read_text()


def test_every_wiki_link_resolves(wiki: Path) -> None:
    pages = {p.stem for p in wiki.glob("*.md")}
    for page in wiki.glob("*.md"):
        for target in re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text()):
            if not target.startswith(("http://", "https://")):
                assert target in pages, f"{page.name} links to missing page {target}"


def test_constants_and_glossary_come_from_their_sources(wiki: Path, store: DataStore) -> None:
    constants = (wiki / "Constants.md").read_text()
    assert all(f"`{c.symbol}`" in constants for c in store.constants.values())
    glossary = (wiki / "Glossary.md").read_text()
    assert not glossary.startswith("# ") and "**Constrained decoding.**" in glossary


def test_rebuilding_replaces_stale_pages(tmp_path: Path) -> None:
    (tmp_path / "removed_equation.md").write_text("old")
    pages = build_wiki.build(tmp_path)
    assert "removed_equation" not in pages and "Home" in pages
    assert not (tmp_path / "removed_equation.md").exists()
