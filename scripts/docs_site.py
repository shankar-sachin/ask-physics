"""Render the Ask Physics Wiki as static pages for the website (askphysics.vercel.app/wiki/).

    python3 scripts/wiki_site.py [build/site/wiki]

Builds the same Markdown pages as the GitHub Wiki (``scripts/build_wiki.py``) and writes
each one as ``<out>/<Page>/index.html``, with ``Home`` at ``<out>/index.html``, so the
answer card can link ``/wiki/kin_v_at/``. The Markdown is our own and simple, so a small
renderer here handles it: headings, paragraphs, lists, tables, code, ``math`` blocks,
emphasis, and links. Every piece of text is HTML-escaped; math is typeset in the browser
by KaTeX, and shows as plain LaTeX if KaTeX doesn't load.

Standard library only, Python 3.9+, because the Vercel build runs it as-is.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

import build_wiki

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_PAGE = re.compile(r"^[A-Za-z0-9_-]+$")
_BOLD_LINK = re.compile(r"\*\*\[([^\]]+)\]\(([^)\s]+)\)\*\*")


def page_url(target: str) -> str:
    """Where a wiki link points: ``Equations`` is ``/wiki/Equations/``, ``Home`` is ``/wiki/``."""
    if target.startswith(("http://", "https://", "mailto:")):
        return target
    name, _, anchor = target.partition("#")
    if not _PAGE.match(name):
        return "/wiki/"
    path = "/wiki/" if name == "Home" else f"/wiki/{name}/"
    return path + (f"#{anchor}" if anchor else "")


def _spans(text: str) -> str:
    """Code spans, bold, and italics, with everything else escaped."""
    pieces = _CODE.split(text)
    out = []
    for i, piece in enumerate(pieces):
        if i % 2:
            out.append(f"<code>{html.escape(piece)}</code>")
            continue
        escaped = html.escape(piece, quote=False)
        escaped = _BOLD.sub(r"<strong>\1</strong>", escaped)
        out.append(_ITALIC.sub(r"<em>\1</em>", escaped))
    return "".join(out)


def inline(text: str) -> str:
    """One line of Markdown as HTML. Links are found first, so a link's text may hold code."""
    # "**[Home](Home)**": bold around a whole link becomes bold link text.
    text = _BOLD_LINK.sub(r"[**\1**](\2)", text)
    out, last = [], 0
    for m in _LINK.finditer(text):
        out.append(_spans(text[last : m.start()]))
        href = html.escape(page_url(m.group(2)), quote=True)
        external = ' rel="noopener"' if href.startswith("http") else ""
        out.append(f'<a href="{href}"{external}>{_spans(m.group(1))}</a>')
        last = m.end()
    out.append(_spans(text[last:]))
    return "".join(out)


def slug(text: str) -> str:
    """GitHub's anchor for a heading: "Reading an answer" is ``reading-an-answer``."""
    return re.sub(r"[^a-z0-9 -]", "", text.lower()).strip().replace(" ", "-")


def _table(rows: list[str]) -> str:
    def cells(row: str) -> list[str]:
        return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", row.strip())[1:-1]]

    head, body = cells(rows[0]), [cells(r) for r in rows[2:]]
    out = ["<table><thead><tr>", *(f"<th>{inline(c)}</th>" for c in head), "</tr></thead><tbody>"]
    for row in body:
        out += ["<tr>", *(f"<td>{inline(c)}</td>" for c in row), "</tr>"]
    return "".join([*out, "</tbody></table>"])


def markdown_to_html(text: str) -> str:
    """Render the wiki's Markdown subset."""
    lines = text.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.startswith("```"):
            lang = line[3:].strip()
            body: list[str] = []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                body.append(lines[i])
                i += 1
            i += 1
            code = html.escape("\n".join(body))
            out.append(
                f'<div class="math">{code}</div>'
                if lang == "math"
                else f"<pre><code>{code}</code></pre>"
            )
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            level, title = len(heading.group(1)), heading.group(2).strip()
            out.append(f'<h{level + 1} id="{slug(title)}">{inline(title)}</h{level + 1}>')
            i += 1
            continue
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(_table(rows))
            continue
        bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            ordered = bullet.group(2)[0].isdigit()
            items: list[str] = []
            while i < len(lines):
                m = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", lines[i])
                if m:
                    items.append(m.group(3))
                elif lines[i].startswith("  ") and lines[i].strip() and items:
                    items[-1] += " " + lines[i].strip()  # a wrapped item
                else:
                    break
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(
                f"<{tag}>" + "".join(f"<li>{inline(item)}</li>" for item in items) + f"</{tag}>"
            )
            continue
        para = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(#{1,4}\s|```|\||\s*([-*]|\d+\.)\s)", lines[i])
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def title_of(name: str, equations: dict[str, str]) -> str:
    """A page's heading: the equation's name for an equation page, else the page name."""
    if name == "Home":
        return "Ask Physics Wiki"
    return equations.get(name) or name.replace("-", " ")


def render_page(name: str, body_md: str, sidebar_md: str, footer_md: str, title: str) -> str:
    """One complete HTML page."""
    title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} · Ask Physics</title>
  <meta name="theme-color" content="#07080f">
  <link rel="icon" href="/images/logo.svg" type="image/svg+xml">
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/wiki/wiki.css">
  <link rel="stylesheet" href="{KATEX}/katex.min.css" crossorigin="anonymous">
</head>
<body>
  <header class="nav">
    <a class="brand" href="/" aria-label="Ask Physics home">
      <img src="/images/logo.svg" alt="" width="34" height="34">
      <span>Ask <b>Physics</b></span>
    </a>
    <nav>
      <a href="/#ask">Try it</a>
      <a href="/wiki/">Wiki</a>
      <a href="/wiki/Equations/">Equations</a>
      <a href="https://github.com/shankar-sachin/ask-physics" class="gh">GitHub</a>
    </nav>
  </header>
  <div class="wiki">
    <aside class="wiki-side">{markdown_to_html(sidebar_md)}</aside>
    <main class="wiki-page">
      <h1>{title}</h1>
      {markdown_to_html(body_md)}
      <footer class="wiki-foot">{markdown_to_html(footer_md)}</footer>
    </main>
  </div>
  <script defer src="{KATEX}/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="/wiki/wiki.js"></script>
</body>
</html>
"""


def build(out: Path) -> list[str]:
    """Write every wiki page under ``out``; returns the page names."""
    with tempfile.TemporaryDirectory() as tmp:
        pages = Path(tmp)
        build_wiki.build(pages)
        sidebar = (pages / "_Sidebar.md").read_text(encoding="utf-8")
        footer = (pages / "_Footer.md").read_text(encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        names = sorted(p.stem for p in pages.glob("*.md") if not p.stem.startswith("_"))
        equations = {
            e["id"]: e["name"]
            for e in json.loads((build_wiki.DATA / "equations.json").read_text(encoding="utf-8"))
        }
        for name in names:
            dest = out if name == "Home" else out / name
            dest.mkdir(parents=True, exist_ok=True)
            body = (pages / f"{name}.md").read_text(encoding="utf-8")
            (dest / "index.html").write_text(
                render_page(name, body, sidebar, footer, title_of(name, equations)),
                encoding="utf-8",
            )
    for asset in ("wiki.css", "wiki.js"):
        shutil.copyfile(build_wiki.ROOT / "web" / asset, out / asset)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, nargs="?", default=Path("build/site/wiki"))
    args = parser.parse_args()
    names = build(args.out)
    print(f"{len(names)} wiki pages -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
