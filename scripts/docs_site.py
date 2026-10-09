"""Render Ask Physics Docs, the website's documentation (askphysics.vercel.app/docs/).

    python3 scripts/docs_site.py [build/site/docs]

Builds the Markdown pages with ``scripts/build_pages.py`` and writes
each one as ``<out>/<Page>/index.html`` (plus its Markdown as ``index.md``, for "Copy
page"), with ``Home`` at ``<out>/index.html``, so the answer card can link
``/docs/kin_v_at/``. Pages are grouped into the sidebar's sections, shown as tabs; a search
index (``search.json``) and an outline of each page come from the same Markdown.

The Markdown is our own and simple, so a small renderer here handles it: headings,
paragraphs, lists, tables, code, ``math`` blocks, ``>`` callouts, emphasis, and links.
Every piece of text is HTML-escaped; math is typeset in the browser by KaTeX, and shows
as plain LaTeX if KaTeX doesn't load.

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
from dataclasses import dataclass, field
from pathlib import Path

import build_pages

KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.11/dist"
# GitHub's mark (Octicons, MIT) and a download arrow, inline so they need no requests.
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" width="22" height="22" aria-hidden="true" fill="currentColor">'
    '<path d="'
    "M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1"
    ".13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.1"
    "5.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 ."
    "27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3."
    "06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-"
    "1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.0"
    "1 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"
    '"/></svg>'
)
DOWNLOAD_ICON = (
    '<svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true" fill="none" '
    'stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 4v11M7 10l5 5 5-5M5 20h14"/></svg>'
)
FONTS = (
    "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700"
    "&family=JetBrains+Mono:wght@400;600&family=Inter:wght@400;500;600&display=swap"
)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])")
_PAGE = re.compile(r"^[A-Za-z0-9_-]+$")
_BOLD_LINK = re.compile(r"\*\*\[([^\]]+)\]\(([^)\s]+)\)\*\*")


def page_url(target: str) -> str:
    """Where a docs link points: ``Equations`` is ``/docs/Equations/``, ``Home`` is ``/docs/``."""
    if target.startswith(("http://", "https://", "mailto:")):
        return target
    name, _, anchor = target.partition("#")
    if not _PAGE.match(name):
        return "/docs/"
    path = "/docs/" if name == "Home" else f"/docs/{name}/"
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
    """Render the docs' Markdown subset. A ``>`` quote becomes a callout box."""
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
        if line.startswith(">"):
            quote = []
            while i < len(lines) and lines[i].startswith(">"):
                quote.append(lines[i][1:].strip())
                i += 1
            out.append(f'<div class="callout">{markdown_to_html(chr(10).join(quote))}</div>')
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
            and not re.match(r"^(#{1,4}\s|```|\||>|\s*([-*]|\d+\.)\s)", lines[i])
        ):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


@dataclass
class Section:
    """A tab: a group of pages from the sidebar ("Using it"), in sidebar order."""

    name: str
    pages: list[tuple[str, str]] = field(default_factory=list)  # (page name, link text)
    groups: list[tuple[str, list[tuple[str, str]]]] = field(default_factory=list)


def sections(sidebar_md: str, equations: list[dict[str, str]]) -> list[Section]:
    """The tabs: Overview, then each bold group of ``_Sidebar.md``, then Equations, whose
    sidebar lists every equation by domain."""
    out = [Section("Overview", [("Home", "Overview")])]
    for line in sidebar_md.splitlines():
        group = re.fullmatch(r"\*\*([^*\[\]]+)\*\*", line.strip())
        item = re.match(r"^\s*[-*]\s+\[([^\]]+)\]\(([^)#\s]+)\)", line)
        if group:
            out.append(Section(group.group(1).strip()))
        elif item and len(out) > 1 and item.group(2) != "Equations":
            out[-1].pages.append((item.group(2), item.group(1)))
    eqs = Section("Equations", [("Equations", "All equations")])
    by_domain: dict[str, list[tuple[str, str]]] = {}
    for eq in sorted(equations, key=lambda e: e["name"].lower()):
        by_domain.setdefault(eq["domain"], []).append((eq["id"], eq["name"]))
    order = [*build_pages.DOMAINS, *sorted(set(by_domain) - set(build_pages.DOMAINS))]
    eqs.groups = [(build_pages.DOMAINS.get(d, d), by_domain[d]) for d in order if d in by_domain]
    tabs = [s for s in out if s.pages]
    # The overview's sidebar lists every guide, so the home page is a way in to all of them.
    tabs[0].groups = [(s.name, s.pages) for s in tabs[1:]]
    return [*tabs, eqs]


def title_of(name: str, equations: dict[str, str]) -> str:
    """A page's heading: the equation's name for an equation page, else the page name."""
    if name == "Home":
        return "Ask Physics Docs"
    return equations.get(name) or name.replace("-", " ")


def outline(body_html: str) -> list[tuple[int, str, str]]:
    """(level, anchor, text) for each section heading on a page, for "On this page"."""
    found = re.findall(r'<h([34]) id="([^"]*)">(.*?)</h\1>', body_html)
    return [(int(level), anchor, re.sub(r"<[^>]+>", "", text)) for level, anchor, text in found]


def _lead(body_html: str) -> str:
    """The first paragraph, when the page opens with one, styled as the page's summary."""
    return re.sub(r"\A<p>", '<p class="lead">', body_html, count=1)


def _nav_list(items: list[tuple[str, str]], current: str) -> str:
    links = []
    for page, text in items:
        here = ' class="here" aria-current="page"' if page == current else ""
        links.append(f'<li><a href="{page_url(page)}"{here}>{html.escape(text)}</a></li>')
    return "<ul>" + "".join(links) + "</ul>"


def _sidebar(section: Section, current: str) -> str:
    if section.groups:
        parts = [_nav_list(section.pages, current)]
        for group, items in section.groups:
            parts.append(f"<p>{html.escape(group)}</p>{_nav_list(items, current)}")
        return "".join(parts)
    return f"<p>{html.escape(section.name)}</p>{_nav_list(section.pages, current)}"


def _neighbours(section: Section, current: str) -> str:
    order = [*section.pages, *(item for _, items in section.groups for item in items)]
    names = [page for page, _ in order]
    if current not in names:
        return ""
    i = names.index(current)
    links = []
    if i > 0:
        page, text = order[i - 1]
        links.append(f'<a class="prev" href="{page_url(page)}"><small>Previous</small>'
                     f"{html.escape(text)}</a>")  # fmt: skip
    if i + 1 < len(order):
        page, text = order[i + 1]
        links.append(f'<a class="next" href="{page_url(page)}"><small>Next</small>'
                     f"{html.escape(text)}</a>")  # fmt: skip
    return f'<nav class="docs-pager">{"".join(links)}</nav>' if links else ""


def section_of(name: str, all_sections: list[Section]) -> Section:
    """The tab a page belongs to: the one listing it as its own page, else the equations."""
    for section in all_sections:
        if name in [page for page, _ in section.pages]:
            return section
    equations = all_sections[-1]
    if any(name == page for _, items in equations.groups for page, _ in items):
        return equations
    return all_sections[0]


def render_page(name: str, body_md: str, all_sections: list[Section], title: str) -> str:
    """One complete HTML page."""
    section = section_of(name, all_sections)
    if section.name == "Equations" and name != "Equations":  # the title names the equation
        body_md = re.sub(r"\A\*\*[^*\n]+\*\* · ", "", body_md)
    body = _lead(markdown_to_html(body_md))
    toc = outline(body)
    if toc and min(level for level, _, _ in toc) == 4:  # equation pages start at ###
        toc = [(level - 1, anchor, text) for level, anchor, text in toc]
    toc_html = ""
    if len(toc) >= 2:
        items = "".join(
            f'<li class="l{level}"><a href="#{html.escape(anchor)}">{html.escape(text)}</a></li>'
            for level, anchor, text in toc
        )
        toc_html = (
            '<nav class="docs-toc" aria-label="On this page">'
            f"<p>On this page</p><ul>{items}</ul></nav>"
        )
    tabs = "".join(
        f'<a href="{page_url(s.pages[0][0])}"'
        + (' class="here" aria-current="true"' if s is section else "")
        + f">{html.escape(s.name)}</a>"
        for s in all_sections
    )
    source = (
        f"{build_pages.REPO}/blob/main/src/askphysics/data/equations.json"
        if section.name == "Equations" and name != "Equations"
        else f"{build_pages.REPO}/tree/main/docs/pages"
    )
    eyebrow = "Welcome" if name == "Home" else section.name
    page_title = "Ask Physics Docs" if name == "Home" else f"{title} · Ask Physics Docs"
    title = html.escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <meta name="theme-color" content="#07080f">
  <link rel="icon" href="/images/logo.svg" type="image/svg+xml">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="{FONTS}" rel="stylesheet">
  <link rel="stylesheet" href="/styles.css">
  <link rel="stylesheet" href="/docs/docs.css">
  <link rel="stylesheet" href="{KATEX}/katex.min.css" crossorigin="anonymous">
</head>
<body class="docs-body">
  <header class="docs-top">
    <div class="docs-bar">
      <a class="brand" href="/docs/" aria-label="Ask Physics Docs home">
        <img src="/images/logo.svg" alt="" width="38" height="38">
        <span>Ask <b>Physics</b></span><span class="brand-sub">Docs</span>
      </a>
      <div class="docs-search" role="search">
        <input id="docs-q" type="search" placeholder="Search the docs"
          aria-label="Search the docs" autocomplete="off">
        <kbd>/</kbd>
        <ul id="docs-results" role="listbox" hidden></ul>
      </div>
      <nav class="docs-links">
        <a href="/#ask">Ask a question</a>
        <a href="/#install" class="install-btn"
          >{DOWNLOAD_ICON}<span class="label">Install</span></a>
        <a href="{build_pages.REPO}" class="gh" aria-label="Ask Physics on GitHub"
          title="GitHub">{GITHUB_ICON}</a>
      </nav>
    </div>
    <nav class="docs-tabs" aria-label="Sections">{tabs}</nav>
  </header>
  <div class="docs">
    <button class="docs-menu" type="button" aria-expanded="false"
      aria-controls="docs-side">Browse {html.escape(section.name)}</button>
    <aside class="docs-side" id="docs-side">{_sidebar(section, name)}</aside>
    <main class="docs-page">
      <p class="eyebrow">{html.escape(eyebrow)}</p>
      <div class="docs-head">
        <h1>{title}</h1>
        <button class="copy-page" type="button"
          data-src="{page_url(name)}index.md">Copy page</button>
      </div>
      {body}
      {_neighbours(section, name)}
      <footer class="docs-foot">Built from <a href="{source}" rel="noopener">the repository</a>
        by <code>scripts/docs_site.py</code>. To change a page, edit it there.</footer>
    </main>
    {toc_html}
  </div>
  <script defer src="{KATEX}/katex.min.js" crossorigin="anonymous"></script>
  <script defer src="/docs/docs.js"></script>
</body>
</html>
"""


def search_entry(name: str, title: str, section: str, body_md: str) -> dict[str, object]:
    """What search looks at for one page: its title, headings, and opening text."""
    plain = re.sub(r"```.*?```", " ", body_md, flags=re.S)
    plain = _LINK.sub(r"\1", plain)
    headings = re.findall(r"^#{1,4}\s+(.*)$", plain, flags=re.M)
    plain = re.sub(r"[#*`|>_-]+", " ", plain)
    return {
        "t": title,
        "u": page_url(name),
        "s": section,
        "h": headings,
        "x": re.sub(r"\s+", " ", plain).strip()[:400],
    }


def build(out: Path) -> list[str]:
    """Write every docs page under ``out``; returns the page names."""
    with tempfile.TemporaryDirectory() as tmp:
        pages = Path(tmp)
        build_pages.build(pages)
        sidebar = (pages / "_Sidebar.md").read_text(encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        names = sorted(p.stem for p in pages.glob("*.md") if not p.stem.startswith("_"))
        equations = json.loads((build_pages.DATA / "equations.json").read_text(encoding="utf-8"))
        eq_names = {e["id"]: e["name"] for e in equations}
        tabs = sections(sidebar, equations)
        index = []
        for name in names:
            dest = out if name == "Home" else out / name
            dest.mkdir(parents=True, exist_ok=True)
            body = (pages / f"{name}.md").read_text(encoding="utf-8")
            title = title_of(name, eq_names)
            (dest / "index.html").write_text(render_page(name, body, tabs, title), encoding="utf-8")
            (dest / "index.md").write_text(f"# {title}\n\n{body}", encoding="utf-8")
            index.append(search_entry(name, title, section_of(name, tabs).name, body))
        (out / "search.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    for asset in ("docs.css", "docs.js"):
        shutil.copyfile(build_pages.ROOT / "web" / asset, out / asset)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, nargs="?", default=Path("build/site/docs"))
    args = parser.parse_args()
    names = build(args.out)
    print(f"{len(names)} docs pages -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
