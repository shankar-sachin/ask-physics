"""Print one version's section of CHANGELOG.md, for the notes of a release page.

    python3 scripts/release_notes.py 0.4.0 [CHANGELOG.md]

Prints everything under the ``## [0.4.0] - DATE`` heading, up to the next ``## [`` heading,
without the heading itself and with blank lines trimmed from both ends. Exits 1 when the
version has no section or the section is empty. Standard library only.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HEADING = re.compile(r"^## \[(?P<name>[^\]]+)\]")


def extract(changelog: str, version: str) -> str:
    """The notes under ``## [version]``, or ``""`` when there is no such section."""
    lines: list[str] = []
    inside = False
    for line in changelog.splitlines():
        heading = HEADING.match(line)
        if heading:
            if inside:
                break
            inside = heading.group("name") == version
            continue
        if inside:
            lines.append(line.rstrip())
    return "\n".join(lines).strip("\n")


def main(argv: list[str]) -> int:
    if argv[:1] in (["-h"], ["--help"]):
        print(__doc__)
        return 0
    if not 1 <= len(argv) <= 2:
        print(__doc__, file=sys.stderr)
        return 2
    version = argv[0].removeprefix("v")
    path = Path(argv[1]) if len(argv) == 2 else Path(__file__).resolve().parents[1] / "CHANGELOG.md"
    notes = extract(path.read_text(encoding="utf-8"), version)
    if not notes.strip():
        print(f"error: no notes for {version} in {path}", file=sys.stderr)
        return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
