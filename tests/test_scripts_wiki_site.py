"""``scripts/wiki_site.py``: the wiki as static pages at askphysics.vercel.app/wiki/."""

import html
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import wiki_site

from askphysics.data.loader import DataStore


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site") / "wiki"
    wiki_site.build(out)
    return out


def test_markdown_subset_renders() -> None:
    page = wiki_site.markdown_to_html(
        "## Look-alikes\n\n"
        "Some **bold**, *italic*, `code`, and a [page](Equations) link.\n\n"
        "- one\n- two, wrapped\n  onto a second line\n\n"
        "1. first\n2. second\n\n"
        "| a | b |\n|---|---|\n| `x` | y \\| z |\n\n"
        "```math\nv^2 = v_0^2 + 2 a d\n```\n\n"
        "```bash\nask <this>\n```"
    )
    assert '<h3 id="look-alikes">Look-alikes</h3>' in page
    assert "<strong>bold</strong>" in page and "<em>italic</em>" in page
    assert '<a href="/wiki/Equations/">page</a>' in page
    assert "<li>two, wrapped onto a second line</li>" in page and "<ol>" in page
    assert "<td><code>x</code></td><td>y | z</td>" in page
    assert '<div class="math">v^2 = v_0^2 + 2 a d</div>' in page
    assert "<pre><code>ask &lt;this&gt;</code></pre>" in page


def test_links_and_escaping() -> None:
    assert wiki_site.inline("[Home](Home)") == '<a href="/wiki/">Home</a>'
    assert wiki_site.inline("**[Home](Home)**") == '<a href="/wiki/"><strong>Home</strong></a>'
    assert wiki_site.inline("[FAQ](FAQ#does-it)") == '<a href="/wiki/FAQ/#does-it">FAQ</a>'
    external = wiki_site.inline("[`docs/wiki/`](https://github.com/x)")
    assert external == '<a href="https://github.com/x" rel="noopener"><code>docs/wiki/</code></a>'
    # Nothing in the text can become markup, and a strange link target goes nowhere.
    assert "<script>" not in wiki_site.inline("<script>alert(1)</script>")
    assert wiki_site.inline("[x](javascript:void)") == '<a href="/wiki/">x</a>'


def test_every_equation_has_a_titled_page(site: Path, store: DataStore) -> None:
    assert (site / "index.html").exists() and (site / "wiki.css").exists()
    for eq in store.equations.values():
        page = (site / eq.id / "index.html").read_text()
        assert f"<h1>{html.escape(eq.name)}</h1>" in page, eq.id
        assert '<div class="math">' in page


def test_every_internal_link_resolves(site: Path) -> None:
    pages = {p.parent.name for p in site.glob("*/index.html")}
    for page in site.rglob("index.html"):
        text = page.read_text()
        assert "](" not in text, f"raw Markdown link left in {page}"
        for href in re.findall(r'href="/wiki/([^"#]*)', text):
            name = href.strip("/")
            assert not name or name in pages or name in {"wiki.css", "wiki.js"}, (page, href)


def test_the_answer_card_links_equations_to_the_wiki() -> None:
    app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text()
    assert "/wiki/${eq.id}/" in app and "/^[a-z0-9_]+$/.test(eq.id)" in app
