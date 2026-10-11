"""Regenerate the README screenshots from the real CLI output.

Runs askphysics commands through a recording Rich console, exports SVGs,
and renders them to PNGs with headless Chromium:

    make screenshots

Needs Node with Playwright and a Chromium for it (``npx playwright install
chromium`` if you don't have one). Training uses fermi-luna-1 on CPU, so it
takes a few seconds.
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

from rich.console import Console
from typer.testing import CliRunner

from askphysics import cli, devcli
from askphysics.ui import make_console

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
WIDTH = 100
FONT = "'DejaVu Sans Mono', 'Fira Code', Menlo, Consolas, monospace"


def capture(
    name: str, title: str, args: list[str], before: Callable[[], None] | None = None
) -> Path:
    # Not interactive: spinners vanish and progress bars print only their final frame.
    console: Console = make_console(
        record=True,
        width=WIDTH,
        force_terminal=True,
        force_interactive=False,
        color_system="truecolor",
        file=io.StringIO(),
    )
    # The title is the command shown, so it says which program draws it.
    module = devcli if title.startswith("askphysics-dev") else cli
    module.console = console
    if before:
        before()
    result = CliRunner().invoke(module.app, args)
    if result.exit_code not in (0, 1):
        raise SystemExit(f"{name} failed:\n{result.output}")
    svg = console.export_svg(title=title)
    svg = re.sub(r"@font-face\s*\{[^}]*\}", "", svg)  # no web fonts: render offline
    svg = re.sub(r"font-family:[^;]+;", f"font-family: {FONT};", svg)
    path = OUT / f"{name}.svg"
    path.write_text(svg, encoding="utf-8")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    svgs = [
        capture(
            "ask-answered",
            "askphysics ask",
            ["ask", "How fast does a falling object hit the ground if it is dropped from 20 m?"],
        ),
        capture("ask-refused", "askphysics ask", ["ask", "How much does the color blue weigh?"]),
        capture(
            "ask-partial",
            "askphysics ask",
            ["ask", "How many rubber ducks would it take to stop a freight train?"],
        ),
        capture("validate-data", "askphysics-dev validate-data", ["validate-data"]),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # short relative paths in the screenshots
        data, tok, models = Path("build/data"), Path("build/tokenizer.json"), Path("models")
        quiet = CliRunner()
        quiet.invoke(
            devcli.app,
            [
                "model",
                "build-data",
                "--out",
                str(data),
                "--examples",
                "1500",
                "--blocklist",
                str(ROOT / "evals" / "questions.yaml"),
            ],
        )
        quiet.invoke(
            devcli.app,
            [
                "model",
                "train-tokenizer",
                "--data",
                str(data),
                "--out",
                str(tok),
                "--vocab-size",
                "512",
            ],
        )
        svgs.append(
            capture(
                "model-train",
                "askphysics-dev model train",
                [
                    "model",
                    "train",
                    "--model",
                    "fermi-luna-1",
                    "--data",
                    str(data),
                    "--tokenizer",
                    str(tok),
                    "--out",
                    str(models / "fermi-luna-1"),
                    "--steps",
                    "300",
                    "--batch-size",
                    "16",
                    "--device",
                    "cpu",
                ],
            )
        )
        svgs.append(
            capture(
                "model-info",
                "askphysics-dev model info",
                ["model", "info", "--directory", str(models)],
            )
        )
        os.chdir(ROOT)
    env = dict(os.environ)
    env.setdefault(
        "NODE_PATH",
        subprocess.run(
            ["npm", "root", "-g"], capture_output=True, text=True, check=True
        ).stdout.strip(),
    )
    subprocess.run(
        ["node", str(ROOT / "scripts" / "svg2png.mjs"), str(OUT), *map(str, svgs)],
        check=True,
        env=env,
    )
    for svg in svgs:
        svg.unlink()


if __name__ == "__main__":
    sys.exit(main())
