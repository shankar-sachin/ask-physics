"""``scripts/extract_openstax.py`` on a tiny made-up book in the OpenStax layout."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COLLECTION = """<?xml version="1.0"?>
<col:collection xmlns="http://cnx.rice.edu/collxml" xmlns:md="http://cnx.rice.edu/mdml"
  xmlns:col="http://cnx.rice.edu/collxml">
  <metadata xmlns:md="http://cnx.rice.edu/mdml">
    <md:title>Physics</md:title>
    <md:license url="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution License</md:license>
  </metadata>
  <col:content>
    <col:module document="m1"/>
    <col:subcollection><md:title>Motion</md:title>
      <col:content><col:module document="m2"/></col:content>
    </col:subcollection>
  </col:content>
</col:collection>"""

PREFACE = """<document xmlns="http://cnx.rice.edu/cnxml"><content>
<para>The authors of this book have taught physics for many years in schools.</para>
</content></document>"""

CHAPTER = """<document xmlns="http://cnx.rice.edu/cnxml" xmlns:m="http://www.w3.org/1998/Math/MathML">
<content>
<para>Velocity is the rate at which an object changes its position over time.</para>
<para>The average speed is <m:math><m:mrow><m:mi>v</m:mi><m:mo>=</m:mo><m:mfrac><m:mi>d</m:mi><m:mi>t</m:mi></m:mfrac></m:mrow></m:math> for a trip of known length and duration.</para>
<para>A falling apple speeds up steadily, as shown in <link target-id="Figure_02_01_apple"/>.</para>
<para>Try this problem now <link class="os-embed" url="#ost/api/ex/k12phys-ch02-ex001"/> before you go on.</para>
<para>which follows from the definition of acceleration given just above in the text.</para>
<para>Too short to keep.</para>
<note class="os-teacher"><para>The student is expected to know the definition of science.</para></note>
<note class="misconception"><para>Speed and velocity are not the same thing — velocity has a direction.</para></note>
<figure><caption><para>A photo of a cheetah running fast across the open grassland.</para></caption></figure>
</content></document>"""


def test_extracts_only_clean_prose(tmp_path: Path) -> None:
    source = tmp_path / "book"
    (source / "collections").mkdir(parents=True)
    (source / "collections" / "physics.collection.xml").write_text(COLLECTION)
    for module, text in (("m1", PREFACE), ("m2", CHAPTER)):
        (source / "modules" / module).mkdir(parents=True)
        (source / "modules" / module / "index.cnxml").write_text(text)
    (source / "LICENSE").write_text("Attribution 4.0 International\n...")
    out = tmp_path / "out"
    subprocess.run(
        [sys.executable, "scripts/extract_openstax.py", "--source", str(source), "--out", str(out)],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in (out / "prose.jsonl").read_text().splitlines()]
    texts = [r["text"] for r in rows]
    assert texts == [
        "Velocity is the rate at which an object changes its position over time.",
        "The average speed is v = d / t for a trip of known length and duration.",
        "A falling apple speeds up steadily, as shown in the figure.",
        "Speed and velocity are not the same thing - velocity has a direction.",
    ]
    assert {r["chapter"] for r in rows} == {"Motion"}
    assert (out / "LICENSE").read_text().startswith("Attribution 4.0 International")


def test_refuses_a_source_that_is_not_cc_by(tmp_path: Path) -> None:
    source = tmp_path / "book"
    (source / "collections").mkdir(parents=True)
    (source / "LICENSE").write_text("Attribution-NonCommercial-ShareAlike 4.0 International")
    result = subprocess.run(
        [sys.executable, "scripts/extract_openstax.py", "--source", str(source),
         "--out", str(tmp_path / "out")],
        cwd=ROOT, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode != 0 and "not CC BY 4.0" in result.stderr
    assert not (tmp_path / "out" / "prose.jsonl").exists()


def test_refuses_a_book_whose_own_license_is_not_cc_by(tmp_path: Path) -> None:
    source = tmp_path / "book"
    (source / "collections").mkdir(parents=True)
    (source / "collections" / "physics.collection.xml").write_text(
        COLLECTION.replace("licenses/by/4.0/", "licenses/by-nc-sa/4.0/")
    )
    (source / "LICENSE").write_text("Attribution 4.0 International\n...")
    result = subprocess.run(
        [sys.executable, "scripts/extract_openstax.py", "--source", str(source),
         "--out", str(tmp_path / "out")],
        cwd=ROOT, capture_output=True, text=True,
    )  # fmt: skip
    assert result.returncode != 0 and "not CC BY 4.0" in result.stderr
