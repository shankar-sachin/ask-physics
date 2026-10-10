"""The ``askphysics`` command: ``ask`` and ``version``, and nothing a user has no use for.

Maintainer commands live in ``askphysics.devcli`` (``askphysics-dev``, ADR-022).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import Annotated, cast

import typer
from rich.text import Text

from askphysics import __version__
from askphysics.config import Provider, Settings
from askphysics.errors import AskPhysicsError
from askphysics.pipeline import Pipeline
from askphysics.ui import answer_card, banner, make_console, tolerate_narrow_encodings

app = typer.Typer(
    name="askphysics",
    help="Ask a physics question. Our own language models read it; real math answers it.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
console = make_console()


@app.callback()
def main() -> None:
    tolerate_narrow_encodings(sys.stdout, sys.stderr)


def _fail(message: str) -> typer.Exit:
    console.print(Text("✗ ", style="bad") + Text(message))
    return typer.Exit(code=1)


@app.command()
def ask(
    words: Annotated[
        list[str],
        typer.Argument(help="The physics question. Quotes are optional.", metavar="QUESTION"),
    ],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the Answer as JSON instead of a card.")
    ] = False,
    llm: Annotated[
        str | None,
        typer.Option(help="Provider: auto (Fermi models if installed, else fake), fermi, or fake."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(help="Force one Fermi model for every stage, e.g. fermi-tellus-1."),
    ] = None,
) -> None:
    """Answer a physics question."""
    question = " ".join(words)
    try:
        settings = Settings.from_env()
        if llm is not None:
            settings = replace(settings, llm_provider=cast(Provider, llm))
        if model is not None:
            settings = replace(settings, model=model)
        _auto_pull(settings, json_output)
        pipeline = Pipeline.from_settings(settings)
        if json_output:
            answer = pipeline.run(question)
        else:
            with console.status("[muted]reading the question, then doing the math"):
                answer = pipeline.run(question)
    except AskPhysicsError as exc:
        raise _fail(str(exc)) from exc
    if json_output:
        typer.echo(answer.model_dump_json(indent=2))
    else:
        console.print(answer_card(answer, pipeline.data.equations))
        if settings.llm_provider == "auto" and pipeline.roster is None:
            console.print(_no_models_hint(), style="muted")


def _auto_pull(settings: Settings, quiet: bool) -> str | None:
    """Download published models that aren't installed yet, before the first answer (ADR-012).

    Returns why the download failed, or None when nothing failed or nothing was needed.
    Nothing is read from the network unless the manifest publishes a model that is missing,
    and a locally trained model of the same name is never replaced. The display goes to
    stdout on a terminal and to stderr otherwise (or with ``quiet``, for ``--json``), so
    scripted use keeps a clean stdout. A failure never raises: ``ask`` degrades as it does
    without models, and the next ``ask`` tries again.
    """
    if not settings.auto_pull or settings.llm_provider == "fake":
        return None
    from askphysics.files_view import PullView, human_size
    from askphysics.llm.routing import SOLEM, TELLUS
    from askphysics.lm.paths import default_model_dir
    from askphysics.lm.weights import missing_published, read_manifest
    from askphysics.lm.weights import pull as pull_models

    root = default_model_dir()
    try:
        manifest = read_manifest()
        names = missing_published(
            [settings.model] if settings.model else [TELLUS, SOLEM], root, manifest=manifest
        )
    except (AskPhysicsError, OSError, ValueError):
        return None  # an unreadable manifest is not worth stopping a question for
    if not names:
        return None
    out = console if console.is_terminal and not quiet else make_console(stderr=True)
    total = sum(f.size for n in names for f in manifest[n].files)
    short = ", ".join(n.removeprefix("fermi-").removesuffix("-1") for n in names)
    out.print(
        f"First question: downloading the Fermi models ({short}, {human_size(total)}) once. "
        "They are kept on this machine, so this won't happen again.",
        style="muted",
    )
    try:
        with PullView(out) as view:
            for name in names:
                view.start_model(name, [(f.name, f.size) for f in manifest[name].files])
                downloaded = pull_models([name], root, manifest=manifest, progress=view.update)
                pinned = [(f.name, f.size, f.sha256) for f in manifest[name].files]
                view.finish_model(name, pinned, downloaded[name], root / name)
    except (AskPhysicsError, OSError) as exc:
        reason = f"couldn't download the Fermi models: {exc}"
        out.print(Text("! ", style="warn") + Text(reason))
        out.print("Answering without them; the next question will try again.", style="muted")
        return reason
    return None


def _no_models_hint() -> str:
    """What to do when ``auto`` found no Fermi models and used the fake one."""
    from askphysics.lm.weights import read_manifest

    if read_manifest():
        return "No Fermi models are installed, so a stand-in answered. Run: askphysics model pull"
    return (
        "No Fermi models are installed, so a stand-in answered. Trained weights aren't "
        "published for this version yet; to train your own, see docs/TRAINING.md."
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    text = banner()
    text.append(f"\n  askphysics {__version__}", style="value")
    console.print(text)


if __name__ == "__main__":  # pragma: no cover
    app()
