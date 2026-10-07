"""Build the prose corpus the Fermi models learn English from (ADR-017).

    python scripts/build_corpus.py [--out build/corpus] [--target-words 20000000]

Reads ``third_party/corpus/sources.json`` and writes, under ``--out``:

- ``prose.jsonl``: one paragraph per line (``source``, ``title``, ``text``), deduplicated.
- ``ATTRIBUTION.md``: every source with its license; CC BY needs this to travel with the
  text and with any model trained on it.
- ``LICENSE-CC-BY-4.0.txt``: the license the OpenStax text is under.
- ``sources.lock.json``: exactly what went in (commits, Gutenberg ids, file hashes).

OpenStax books are read at pinned commits from before OpenStax relicensed them to
CC BY-NC-SA, and the build stops if the LICENSE or any book's metadata at that commit
isn't CC BY 4.0. Project Gutenberg books are public domain (``scripts/gutenberg.py``);
science comes first, then literature, until the word target is reached. Paragraphs
containing slurs or swear words are dropped (the list is kept as hashes).

The corpus is about 130 MB, so it is built locally and never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import extract_openstax as openstax
import gutenberg

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "third_party" / "corpus" / "sources.json"
# sha256 prefixes of slurs and swear words; a paragraph containing one is dropped, so the
# models never learn to write them. Plain text stays out of the repository on purpose.
# Words with common innocent meanings (a surname, a crack in a shutter, a rooster, a
# donkey, Moby Dick, a bloody nose, a finger prick) are left off: they dropped real text.
BLOCKED = frozenset(
    {
        "120f6e5b4ea32f65", "5b3ae48be122f7ed", "08a841e996781e9e", "341d56384afc0f47",
        "c3de533e9b7fe63b", "268651b3ece98010", "98b52c4b6b7d1f48", "044eb98b18769b88",
        "eef3bd091670c344", "0ce875d620076b53", "cc02032349c833ac", "82159dd02870c13e",
        "6122fdf4ae1474fb", "6437ea7144483cfa", "be00f1af522eaa99", "448b303c89bef071",
        "8f5083e3e5c7dc89", "1e02eec4f1095143", "16ea09fc78ca83ca", "ad6cacf3af98d07b",
        "eda887476f1dffc9", "bb87adf8aa783887", "701033ae26411908", "22fc75e65a0e9d34",
        "333f7618092958c7", "12e6274e4309293e", "6ac3c336e4094835", "522788b65f01ccc0",
        "b690cdbe59f14b3b", "31506a8448a761a4", "bb61ef40814ce34c", "63333e4b5d1cde54",
        "2f5f6ce5ae30b54a", "d95b1fc7856571b0", "85fc17f7069acd39", "45fb7c3b72b6856c",
        "986c99004163a6c5", "b2a7d488cf78ce74", "30a989afc82c0a21", "d75a838dc758ba17",
        "5a9d9dcefb56e593", "245c20d383aea7a8", "e512a05583448f44", "a24efab8d957af0a",
        "e9143a29033d9145", "235e285376cb0d3b", "2d2332f38d6c0213", "5cb6ba4a9d88885c",
        "fb96c27642b4bd11", "b601d5e9e6c9804e", "0ebdc3317b75839f", "5921fb04bb6bfe65",
        "566f532d486c9477", "517ae3f73216da64", "f50c51ed2315dcf3", "26578569a290ecbe",
        "7bc671151cbfaee7", "b81d6dc25832a854", "df736257aa81e4a7", "2189c0ed714f0c54",
        "db10a7906121f4ea", "9ae315a94e428a7e", "937d56e49744d206", "c2c3b68b48832afd",
        "11cf8376a1575fdc", "bdb1cc258d6976aa", "dd92623b0a4b255f",

    }
)  # fmt: skip


def blocked(text: str) -> bool:
    """Whether ``text`` contains a word on the blocklist."""
    return any(
        hashlib.sha256(w.encode()).hexdigest()[:16] in BLOCKED
        for w in re.findall(r"[a-z]+", text.lower())
    )


_INITIALS = re.compile(r"\b[A-Z]\.,?\s")
_YEAR = re.compile(r"\((?:1[89]|20)\d\d[a-z]?\)")


def citation(text: str) -> bool:
    """Whether ``text`` looks like a reference-list entry: "Oyserman, D., Coon, H. (2002)."""
    return bool(_YEAR.search(text)) and len(_INITIALS.findall(text)) >= 2


def progress(message: str) -> None:
    """One line of progress on stderr, so a long, quiet step never looks frozen."""
    print(message, file=sys.stderr, flush=True)


def _key(text: str) -> str:
    return hashlib.sha256(re.sub(r"\W+", "", text.lower()).encode()).hexdigest()


class Corpus:
    """Paragraphs written so far, deduplicated, with per-source counts."""

    def __init__(self, path: Path) -> None:
        self.file = path.open("w", encoding="utf-8")
        self.seen: set[str] = set()
        self.words = 0
        self.counts: dict[str, dict[str, int]] = {}
        self.dropped = {"duplicate": 0, "blocked": 0, "citation": 0}

    def add(self, source: str, title: str, text: str) -> None:
        key = _key(text)
        if key in self.seen:
            self.dropped["duplicate"] += 1
            return
        if blocked(text):
            self.dropped["blocked"] += 1
            return
        if citation(text):
            self.dropped["citation"] += 1
            return
        self.seen.add(key)
        n = len(text.split())
        self.words += n
        stats = self.counts.setdefault(source, {"paragraphs": 0, "words": 0})
        stats["paragraphs"] += 1
        stats["words"] += n
        self.file.write(json.dumps({"source": source, "title": title, "text": text}) + "\n")

    def close(self) -> None:
        self.file.close()


def openstax_books(
    entries: list[dict[str, Any]], cache: Path
) -> Iterator[tuple[str, str, str, str]]:
    """(source id, title, paragraph, license text) for every pinned OpenStax book."""
    for i, entry in enumerate(entries, 1):
        name = entry["repo"].split("/")[-1]
        source = cache / "openstax" / f"{name}@{entry['commit'][:12]}"
        if not (source / "LICENSE").exists():
            progress(f"[OpenStax {i}/{len(entries)}] downloading {entry['repo']}")
            openstax.clone(source, f"https://github.com/{entry['repo']}.git", entry["commit"])
        collections = [f"collections/{b['collection']}" for b in entry["books"]]
        license_text = openstax.verify_cc_by(source, collections)
        for b, collection in zip(entry["books"], collections, strict=True):
            progress(f"[OpenStax {i}/{len(entries)}] reading {b['title']}")
            for _, _, text in openstax.book(source, collection):
                yield f"openstax:{b['slug']}", b["title"], text, license_text


def gutenberg_books(
    spec: dict[str, Any], cache: Path, catalog: Path | None, offline: bool, delay: float
) -> Iterator[tuple[gutenberg.Book, Path]]:
    """Selected public-domain books and their downloaded files, in selection order."""
    if catalog is None:
        catalog = cache / "gutenberg" / "pg_catalog.csv"
        if not offline and not catalog.exists():
            progress("[Gutenberg] downloading the catalog")
            gutenberg.fetch(gutenberg.CATALOG_URL, catalog, delay=0)
    if not catalog.exists():
        print("no Gutenberg catalog; skipping public-domain books", file=sys.stderr)
        return
    books = gutenberg.select(
        gutenberg.read_catalog(catalog),
        spec["loccs"],
        spec["latest_death"],
        spec.get("exclude_subjects", ()),
    )
    progress(f"[Gutenberg] {len(books):,} public-domain books qualify; taking them in order")
    for book in books:
        path = cache / "gutenberg" / f"pg{book.id}.txt"
        if not path.exists():
            if offline:
                continue
            progress(f"[Gutenberg] downloading #{book.id} {book.title[:70]}")
            try:
                gutenberg.fetch(gutenberg.TEXT_URL.format(id=book.id), path, delay=delay)
            except OSError as exc:
                print(f"skipping Gutenberg #{book.id}: {exc}", file=sys.stderr)
                continue
        yield book, path


def build(
    sources: Path,
    out: Path,
    cache: Path,
    *,
    target_words: int | None = None,
    catalog: Path | None = None,
    offline: bool = False,
    delay: float = 2.0,
) -> dict[str, Any]:
    """Build the corpus; returns the lock data written to ``sources.lock.json``."""
    spec = json.loads(sources.read_text(encoding="utf-8"))
    target = target_words or int(spec["target_words"])
    out.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(out / "prose.jsonl")
    lock: dict[str, Any] = {"openstax": [], "gutenberg": []}
    cc_by = ""
    try:
        for source, title, text, terms in openstax_books(spec["openstax"], cache):
            corpus.add(source, title, text)
            cc_by = terms
        progress(f"[OpenStax] done: {corpus.words:,} words")
        lock["openstax"] = [{"repo": e["repo"], "commit": e["commit"]} for e in spec["openstax"]]
        pd = spec.get("gutenberg")
        if pd:
            for book, path in gutenberg_books(pd, cache, catalog, offline, delay):
                if corpus.words >= target:
                    break
                text = body(path)
                if text is None:
                    continue
                kept = list(gutenberg.paragraphs(text))
                if sum(len(p.split()) for p in kept) < int(pd.get("min_book_words", 0)):
                    continue
                before = corpus.words
                for paragraph in kept:
                    corpus.add(f"gutenberg:{book.id}", book.title, paragraph)
                if corpus.words == before:
                    continue  # nothing new: every paragraph was already in
                progress(f"[Gutenberg] {corpus.words:,} of {target:,} words")
                lock["gutenberg"].append(
                    {
                        "id": book.id,
                        "title": book.title,
                        "authors": book.authors,
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
    finally:
        corpus.close()
    lock["words"] = corpus.words
    lock["dropped"] = corpus.dropped
    lock["sources"] = corpus.counts
    (out / "sources.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    if cc_by:
        (out / "LICENSE-CC-BY-4.0.txt").write_text(cc_by, encoding="utf-8")
    (out / "ATTRIBUTION.md").write_text(attribution(spec, lock), encoding="utf-8")
    return lock


def body(path: Path) -> str | None:
    """A downloaded Gutenberg file's own text, or None if it can't be used."""
    return gutenberg.body(path.read_text(encoding="utf-8", errors="replace"))


def attribution(spec: dict[str, Any], lock: dict[str, Any]) -> str:
    """ATTRIBUTION.md for the built corpus."""
    lines = [
        "# Attribution: the Ask Physics prose corpus",
        "",
        "`prose.jsonl` is built by `scripts/build_corpus.py` (ADR-017) from the sources",
        "below. It is not covered by Ask Physics' MIT license; each part keeps its own.",
        "",
        "## OpenStax textbooks (CC BY 4.0)",
        "",
        "Licensed under the Creative Commons Attribution 4.0 International license",
        "(<https://creativecommons.org/licenses/by/4.0/>, full text in",
        "`LICENSE-CC-BY-4.0.txt`). Each was read at the commit listed, a version",
        "published under CC BY 4.0 before OpenStax relicensed later versions. Published",
        "by OpenStax, Rice University; access for free at <https://openstax.org>.",
        "",
        "Changes: only prose paragraphs kept (figures, captions, tables, exercises,",
        "display equations, prefaces, and teacher material removed), inline math written",
        'as text, figure and table references written as "the figure" and "the table",',
        "text converted to ASCII, and repeated paragraphs removed.",
        "",
    ]
    for entry in spec["openstax"]:
        for b in entry["books"]:
            by = ", ".join(b["authors"]) if b["authors"] else "OpenStax"
            lines.append(
                f"- *{b['title']}*, by {by}. <https://openstax.org/details/books/{b['slug']}>."
                f" Source: `{entry['repo']}` at `{entry['commit']}`."
            )
    lines += [
        "",
        "## Public-domain books",
        "",
        "Public domain in the United States, and everyone credited died in or before",
        f"{spec.get('gutenberg', {}).get('latest_death', 1955)}. Texts obtained from Project",
        "Gutenberg; its header, footer, and trademark were removed, and the text was",
        "unwrapped, converted to ASCII, and stripped of verse, tables, and headings.",
        "",
    ]
    for g in lock["gutenberg"]:
        lines.append(f"- *{g['title']}*, by {g['authors'] or 'unknown'} (Gutenberg #{g['id']}).")
    lines += [
        "",
        "OpenStax, Rice University, and Project Gutenberg do not endorse Ask Physics;",
        "their names are used only to give credit.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--out", type=Path, default=Path("build/corpus"))
    parser.add_argument("--cache", type=Path, default=Path("build/corpus-cache"))
    parser.add_argument("--target-words", type=int, default=None)
    parser.add_argument("--catalog", type=Path, help="A local pg_catalog.csv.")
    parser.add_argument("--offline", action="store_true", help="Use only cached downloads.")
    parser.add_argument("--delay", type=float, default=2.0, help="Seconds between downloads.")
    args = parser.parse_args()
    lock = build(
        args.sources,
        args.out,
        args.cache,
        target_words=args.target_words,
        catalog=args.catalog,
        offline=args.offline,
        delay=args.delay,
    )
    print(
        f"{lock['words']:,} words from {len(lock['openstax'])} OpenStax repositories and "
        f"{len(lock['gutenberg'])} public-domain books -> {args.out / 'prose.jsonl'} "
        f"(dropped {lock['dropped']['duplicate']:,} duplicates, "
        f"{lock['dropped']['citation']:,} citations, "
        f"{lock['dropped']['blocked']:,} with slurs or swearing)"
    )


if __name__ == "__main__":
    main()
