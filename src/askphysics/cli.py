"""Typer CLI: ``askphysics ask``, ``askphysics version``, ``askphysics validate-data``."""

from __future__ import annotations

from dataclasses import replace
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
        typer.Option(help="LLM provider: fake (default, no API key) or anthropic."),
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


if __name__ == "__main__":  # pragma: no cover
    app()
