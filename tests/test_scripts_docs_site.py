"""``scripts/docs_site.py``: Ask Physics Docs, static pages at askphysics.vercel.app/docs/."""

import html
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import docs_site

from askphysics.data.loader import DataStore


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("site") / "docs"
    docs_site.build(out)
    return out


def test_markdown_subset_renders() -> None:
    page = docs_site.markdown_to_html(
        "## Look-alikes\n\n"
        "Some **bold**, *italic*, `code`, and a [page](Equations) link.\n\n"
        "- one\n- two, wrapped\n  onto a second line\n\n"
        "1. first\n2. second\n\n"
        "| a | b |\n|---|---|\n| `x` | y \\| z |\n\n"
        "```math\nv^2 = v_0^2 + 2 a d\n```\n\n"
        "```bash\nask <this>\n```\n\n"
        "> A **callout**\n> on two lines."
    )
    assert '<h3 id="look-alikes">Look-alikes</h3>' in page
    assert "<strong>bold</strong>" in page and "<em>italic</em>" in page
    assert '<a href="/docs/Equations/">page</a>' in page
    assert "<li>two, wrapped onto a second line</li>" in page and "<ol>" in page
    assert "<td><code>x</code></td><td>y | z</td>" in page
    assert '<div class="math">v^2 = v_0^2 + 2 a d</div>' in page
    assert "<pre><code>ask &lt;this&gt;</code></pre>" in page
    assert '<div class="callout"><p>A <strong>callout</strong> on two lines.</p></div>' in page


def test_links_and_escaping() -> None:
    assert docs_site.inline("[Home](Home)") == '<a href="/docs/">Home</a>'
    assert docs_site.inline("**[Home](Home)**") == '<a href="/docs/"><strong>Home</strong></a>'
    assert docs_site.inline("[FAQ](FAQ#does-it)") == '<a href="/docs/FAQ/#does-it">FAQ</a>'
    external = docs_site.inline("[`docs/docs/`](https://github.com/x)")
    assert external == '<a href="https://github.com/x" rel="noopener"><code>docs/docs/</code></a>'
    # Nothing in the text can become markup, and a strange link target goes nowhere.
    assert "<script>" not in docs_site.inline("<script>alert(1)</script>")
    assert docs_site.inline("[x](javascript:void)") == '<a href="/docs/">x</a>'


def test_every_equation_has_a_titled_page(site: Path, store: DataStore) -> None:
    assert (site / "index.html").exists() and (site / "docs.css").exists()
    for eq in store.equations.values():
        page = (site / eq.id / "index.html").read_text()
        assert f"<h1>{html.escape(eq.name)}</h1>" in page, eq.id
        assert '<p class="eyebrow">Equations</p>' in page, eq.id
        assert '<div class="math">' in page


def test_every_internal_link_resolves(site: Path) -> None:
    pages = {p.parent.name for p in site.glob("*/index.html")}
    for page in site.rglob("index.html"):
        text = page.read_text()
        assert "](" not in text, f"raw Markdown link left in {page}"
        for href in re.findall(r'href="/docs/([^"#]*)', text):
            name = href.strip("/")
            assert not name or name in pages or name in {"docs.css", "docs.js", "search.json"}, (
                page,
                href,
            )


def test_the_answer_card_links_equations_to_the_docs() -> None:
    app = (Path(__file__).resolve().parents[1] / "web" / "app.js").read_text()
    assert "/docs/${eq.id}/" in app and "/^[a-z0-9_]+$/.test(eq.id)" in app


def test_docs_layout(site: Path) -> None:
    home = (site / "index.html").read_text()
    assert "<title>Ask Physics Docs</title>" in home
    assert '<span class="brand-sub">Docs</span>' in home
    faq = (site / "FAQ" / "index.html").read_text()
    assert "<title>FAQ · Ask Physics Docs</title>" in faq
    # The FAQ's section is the current tab, and the FAQ the current sidebar link.
    assert '<a href="/docs/Getting-Started/" class="here" aria-current="true">Using it</a>' in faq
    assert '<a href="/docs/FAQ/" class="here" aria-current="page">FAQ</a>' in faq
    # A page that opens with a paragraph gets it as its summary; the FAQ opens with a question.
    assert '<p class="lead">' not in faq and 'class="docs-toc"' in faq
    assert '<p class="lead">' in (site / "Reading-an-Answer" / "index.html").read_text()
    assert (site / "FAQ" / "index.md").read_text().startswith("# FAQ\n")


def test_search_index_covers_every_page(site: Path) -> None:
    index = json.loads((site / "search.json").read_text())
    pages = {p.parent.name for p in site.glob("*/index.html")}
    assert {e["u"] for e in index} == {"/docs/"} | {f"/docs/{p}/" for p in pages}
    kin = next(e for e in index if e["u"] == "/docs/kin_v_squared/")
    assert kin["s"] == "Equations" and "Assumptions" in kin["h"]
    assert all(len(e["x"]) <= 400 and "](" not in e["x"] for e in index)
