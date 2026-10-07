"""``scripts/build_corpus.py`` end to end, offline, from a prepared cache."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_corpus

from test_scripts_gutenberg import BOOK, CATALOG

COMMIT = "a" * 40
COLLECTION = """<?xml version="1.0"?>
<col:collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml"
  xmlns:col="http://cnx.rice.edu/collxml">
  <metadata xmlns:md="http://cnx.rice.edu/mdml">
    <md:title>Fake Science</md:title>
    <md:license url="http://creativecommons.org/licenses/LICENSE/4.0/">x</md:license>
  </metadata>
  <col:content>
    <col:subcollection><md:title>Waves</md:title>
      <col:content><col:module document="m1"/></col:content>
    </col:subcollection>
  </col:content>
</col:collection>"""
MODULE = """<document xmlns="http://cnx.rice.edu/cnxml"><content>
<para>A wave carries energy from one place to another without carrying matter along.</para>
<para>A wave carries energy from one place to another without carrying matter along.</para>
<para>Smith, J., Jones, K., and Lee, M. (2002). Waves and their uses in modern science.</para>
<para>The zzqx word makes this otherwise ordinary sentence about sound waves go away.</para>
</content></document>"""


def _cache(root: Path, license_name: str = "by") -> Path:
    book = root / "openstax" / f"fake-book@{COMMIT[:12]}"
    (book / "collections").mkdir(parents=True)
    (book / "modules" / "m1").mkdir(parents=True)
    head = "Attribution 4.0 International" if license_name == "by" else "Attribution-NonCommercial"
    (book / "LICENSE").write_text(head + "\n...")
    (book / "collections" / "fake.collection.xml").write_text(
        COLLECTION.replace("LICENSE", license_name)
    )
    (book / "modules" / "m1" / "index.cnxml").write_text(MODULE)
    pg = root / "gutenberg"
    pg.mkdir()
    (pg / "pg_catalog.csv").write_text(CATALOG)
    for book_id in (1, 3, 4):
        (pg / f"pg{book_id}.txt").write_text(BOOK)
    return root


def _sources(path: Path, target: int) -> Path:
    path.write_text(
        json.dumps(
            {
                "target_words": target,
                "openstax": [
                    {
                        "repo": "openstax/fake-book",
                        "commit": COMMIT,
                        "books": [
                            {
                                "collection": "fake.collection.xml",
                                "title": "Fake Science",
                                "slug": "fake-science",
                                "authors": [],
                            }
                        ],
                    }
                ],
                "gutenberg": {
                    "latest_death": 1955,
                    "loccs": ["QC", "QD", "PR"],
                    "exclude_subjects": ["Poetry"],
                    "min_book_words": 10,
                },
            }
        )
    )
    return path


def test_builds_a_deduplicated_attributed_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        build_corpus, "BLOCKED", frozenset({hashlib.sha256(b"zzqx").hexdigest()[:16]})
    )
    cache = _cache(tmp_path / "cache")
    out = tmp_path / "out"
    lock = build_corpus.build(
        _sources(tmp_path / "s.json", 10**6), out, cache,
        catalog=cache / "gutenberg" / "pg_catalog.csv", offline=True, delay=0,
    )  # fmt: skip
    rows = [json.loads(line) for line in (out / "prose.jsonl").read_text().splitlines()]
    sources = [r["source"] for r in rows]
    assert (
        sources.count("openstax:fake-science") == 1
    )  # the duplicate, citation, and blocked one go
    # Every selected public-domain book is the same sample text, so only the first one adds.
    assert sources.count("gutenberg:4") == 2 and "gutenberg:1" not in sources
    assert lock["dropped"] == {"duplicate": 5, "blocked": 1, "citation": 1}
    assert [g["id"] for g in lock["gutenberg"]] == [4]  # only books that added text
    assert lock["openstax"] == [{"repo": "openstax/fake-book", "commit": COMMIT}]
    attribution = (out / "ATTRIBUTION.md").read_text()
    assert "*Fake Science*, by OpenStax" in attribution and COMMIT in attribution
    assert "Gutenberg #4" in attribution and "Gutenberg #1" not in attribution
    assert "CC BY 4.0" in attribution
    assert (out / "LICENSE-CC-BY-4.0.txt").read_text().startswith("Attribution 4.0")


def test_stops_adding_books_at_the_word_target(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache")
    lock = build_corpus.build(
        _sources(tmp_path / "s.json", 20), tmp_path / "out", cache,
        catalog=cache / "gutenberg" / "pg_catalog.csv", offline=True, delay=0,
    )  # fmt: skip
    assert lock["gutenberg"] == []  # OpenStax alone already passed 20 words


def test_refuses_a_pin_that_is_not_cc_by(tmp_path: Path) -> None:
    cache = _cache(tmp_path / "cache", license_name="by-nc-sa")
    with pytest.raises(SystemExit, match=r"not CC BY 4\.0"):
        build_corpus.build(
            _sources(tmp_path / "s.json", 10), tmp_path / "out", cache, offline=True, delay=0
        )


def test_the_manifest_pins_only_cc_by_books() -> None:
    spec = json.loads(build_corpus.SOURCES.read_text())
    assert len(spec["openstax"]) == 33
    assert sum(len(e["books"]) for e in spec["openstax"]) == 52
    assert all(len(e["commit"]) == 40 for e in spec["openstax"])
    titles = {b["title"] for e in spec["openstax"] for b in e["books"]}
    assert {"University Physics Volume 1", "U.S. History", "Writing Guide with Handbook"} <= titles
    never_cc_by = {"Calculus Volume 1", "Organic Chemistry", "Business Law I Essentials"}
    assert not titles & never_cc_by
    # Non-fiction ranks ahead of literature (P), which comes last.
    assert spec["gutenberg"]["loccs"][-1] == "P" and spec["gutenberg"]["loccs"][0] == "QC"
