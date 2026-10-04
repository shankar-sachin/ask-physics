"""Typer CLI: ``ask``, ``version``, ``validate-data``, and the ``model`` commands."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated, cast

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from askphysics import __version__
from askphysics.config import Provider, Settings
from askphysics.data.loader import load_all
from askphysics.errors import AskPhysicsError, DataValidationError
from askphysics.models import Answer
from askphysics.pipeline import Pipeline

app = typer.Typer(
    name="askphysics",
    help="Ask a physics question. Answers come from retrieved equations and symbolic math.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

_STATUS_STYLE = {"answered": "green", "degraded": "yellow", "refused": "red"}
_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def render_answer(answer: Answer) -> None:
    """Pretty-print an answer with Rich."""
    style = _STATUS_STYLE[answer.status]
    if answer.final_value is not None:
        headline = f"[bold]{answer.final_value:.6g} {escape(answer.unit or '')}[/bold]"
    else:
        headline = f"[bold]{answer.status.upper()}[/bold]"
    console.print(
        Panel(
            f"{headline}\n\n{escape(answer.explanation)}",
            title=f"[{style}]{answer.status}[/{style}] · {answer.category}",
            subtitle=escape(answer.question),
            expand=False,
        )
    )

    conf = answer.confidence
    cstyle = _CONFIDENCE_STYLE[conf.label]
    console.print(f"Confidence: [{cstyle}]{conf.label}[/{cstyle}] ({conf.score:.2f})")

    if answer.equations_used:
        table = Table("Equation id", "Name", title="Equations used", title_justify="left")
        for ref in answer.equations_used:
            table.add_row(ref.id, ref.name)
        console.print(table)
    for heading, items in (("Assumptions", answer.assumptions), ("Caveats", answer.caveats)):
        if items:
            console.print(f"[bold]{heading}[/bold]")
            for item in items:
                console.print(f"  - {escape(item)}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The physics question, in quotes.")],
    json_output: Annotated[
        bool, typer.Option("--json", help="Print the Answer as JSON instead of a panel.")
    ] = False,
    llm: Annotated[
        str | None,
        typer.Option(help="LLM provider: fake (default). The Fermi models arrive in v0.3."),
    ] = None,
) -> None:
    """Answer a physics question."""
    try:
        settings = Settings.from_env()
        if llm is not None:
            settings = replace(settings, llm_provider=cast(Provider, llm))
        answer = Pipeline.from_settings(settings).run(question)
    except AskPhysicsError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc
    if json_output:
        typer.echo(answer.model_dump_json(indent=2))
    else:
        render_answer(answer)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(f"askphysics {__version__}")


@app.command("validate-data")
def validate_data() -> None:
    """Validate every seed data file (schema, SymPy parse, units, cross-references)."""
    try:
        store = load_all()
    except DataValidationError as exc:
        console.print(f"[red]Data validation failed with {len(exc.problems)} problem(s):[/red]")
        for problem in exc.problems:
            console.print(f"  - {escape(problem)}")
        raise typer.Exit(code=1) from exc
    table = Table("File", "Entries", title="Seed data: all valid", title_justify="left")
    for name, count in store.summary().items():
        table.add_row(name, str(count))
    console.print(table)


model_app = typer.Typer(
    help="Build training data for, train, and inspect the Fermi models.", no_args_is_help=True
)
app.add_typer(model_app, name="model")


@model_app.command("build-data")
def build_data(
    out: Annotated[Path, typer.Option(help="Output directory for the JSONL shards.")] = Path(
        "build/data"
    ),
    examples: Annotated[int, typer.Option(min=1, help="Number of examples to generate.")] = 10_000,
    seed: Annotated[int, typer.Option(help="Seed; the same seed gives the same data.")] = 0,
    workers: Annotated[int, typer.Option(min=1, help="Parallel worker processes.")] = 1,
    blocklist: Annotated[
        Path, typer.Option(help="Eval questions to keep out of the data (leakage policy).")
    ] = Path("evals/questions.yaml"),
) -> None:
    """Generate training data from the equation database (solved by the symbolic machine)."""
    from askphysics.lm.factory import build_dataset, load_blocklist

    blocked = load_blocklist(blocklist)
    if not blocked:
        console.print(
            f"[yellow]Warning:[/yellow] no eval questions found at {escape(str(blocklist))}"
        )
    with console.status(f"Generating {examples:,} examples with {workers} worker(s)..."):
        manifest = build_dataset(out, examples, seed=seed, workers=workers, blocklist=blocked)
    table = Table(
        "Split / task", "Examples", title=f"Dataset written to {out}", title_justify="left"
    )
    for key, count in manifest["counts"].items():
        table.add_row(key, f"{count:,}")
    console.print(table)
    console.print(
        f"Dropped {manifest['dropped']:,} attempts (unsolvable or too close to an eval question)."
    )


if __name__ == "__main__":  # pragma: no cover
    app()
