"""The ``askphysics`` command: ``ask`` and ``version``, plus a hidden hook for the installers.

Maintainer commands live in ``askphysics.devcli`` (ADR-022).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.console import Console
from rich.text import Text

from askphysics import __version__
from askphysics.config import Provider, Settings
from askphysics.errors import AskPhysicsError
from askphysics.pipeline import Pipeline
from askphysics.ui import answer_card, banner, make_console, tolerate_narrow_encodings

if TYPE_CHECKING:
    from askphysics.lm.weights import PinnedModel

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
            console.print(_no_models_hint(settings), style="muted")


def _download_models(
    names: list[str], manifest: dict[str, PinnedModel], root: Path, out: Console
) -> None:
    """Download ``names`` into ``root`` with the pull display, each checked against the manifest.

    Raises:
        AskPhysicsError: a download failed or didn't match the manifest.
        OSError: the models directory isn't writable.
    """
    from askphysics.files_view import PullView
    from askphysics.lm.weights import pull as pull_models

    with PullView(out) as view:
        for name in names:
            view.start_model(name, [(f.name, f.size) for f in manifest[name].files])
            downloaded = pull_models([name], root, manifest=manifest, progress=view.update)
            pinned = [(f.name, f.size, f.sha256) for f in manifest[name].files]
            view.finish_model(name, pinned, downloaded[name], root / name)


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
    from askphysics.files_view import human_size
    from askphysics.llm.routing import SOLEM, TELLUS
    from askphysics.lm.paths import default_model_dir
    from askphysics.lm.weights import missing_published, read_manifest

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
        _download_models(names, manifest, root, out)
    except (AskPhysicsError, OSError) as exc:
        reason = f"couldn't download the Fermi models: {exc}"
        out.print(Text("! ", style="warn") + Text(reason))
        out.print("Answering without them; the next question will try again.", style="muted")
        return reason
    return None


def _no_models_hint(settings: Settings) -> str:
    """Why a stand-in answered, when ``auto`` found no Fermi models, and what happens next.

    Never names a command: nothing here is for the user to run.
    """
    from askphysics.lm.weights import read_manifest

    lead = "No Fermi models are installed, so a stand-in answered."
    if not read_manifest():
        return f"{lead} Trained models aren't published for this version yet."
    if settings.auto_pull:
        return f"{lead} They download when you ask again with a network connection."
    return (
        f"{lead} Automatic downloads are off (ASKPHYSICS_AUTO_PULL=0); remove that and ask again."
    )


@app.command("install-models", hidden=True)
def install_models() -> None:
    """Download the published tellus and solem models (run by the installers and Homebrew).

    Internal: not listed in ``--help`` and not for users to run. It is ``ask``'s first-question
    download, done up front: the same verified pull into ``$ASKPHYSICS_MODEL_DIR`` (default
    ``~/.cache/askphysics/models``), skipping models already installed. It succeeds quietly while
    nothing is published, and exits 1 when a download fails, so the caller can say the models
    will download on the first question instead.
    """
    from askphysics.llm.routing import SOLEM, TELLUS
    from askphysics.lm.paths import default_model_dir
    from askphysics.lm.weights import missing_published, read_manifest

    root = default_model_dir()
    try:
        manifest = read_manifest()
        names = missing_published([TELLUS, SOLEM], root, manifest=manifest)
    except (AskPhysicsError, OSError, ValueError) as exc:
        raise _fail(f"couldn't read the list of Fermi models: {exc}") from exc
    if not manifest:
        console.print("Trained models aren't published for this version yet.", style="muted")
        return
    if not names:
        console.print("The Fermi models are already installed.", style="muted")
        return
    try:
        _download_models(names, manifest, root, console)
    except (AskPhysicsError, OSError) as exc:
        raise _fail(f"couldn't download the Fermi models: {exc}") from exc


@app.command()
def version() -> None:
    """Print the installed version."""
    text = banner()
    text.append(f"\n  askphysics {__version__}", style="value")
    console.print(text)


if __name__ == "__main__":  # pragma: no cover
    app()
