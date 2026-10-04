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


@model_app.command("train-tokenizer")
def train_tokenizer_cmd(
    data: Annotated[Path, typer.Option(help="Dataset directory from build-data.")] = Path(
        "build/data"
    ),
    out: Annotated[Path, typer.Option(help="Where to write tokenizer.json.")] = Path(
        "build/tokenizer.json"
    ),
    vocab_size: Annotated[int, typer.Option(min=262, help="Vocabulary size.")] = 8192,
    max_examples: Annotated[int, typer.Option(min=1, help="Training examples to read.")] = 50_000,
) -> None:
    """Train the byte-level BPE tokenizer on the dataset's training split."""
    from askphysics.lm.train import train_tokenizer

    with console.status(f"Learning up to {vocab_size:,} tokens from {escape(str(data))}..."):
        tokenizer = train_tokenizer(data, vocab_size, max_examples)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    console.print(f"Tokenizer with {tokenizer.vocab_size:,} tokens written to {escape(str(out))}")


@model_app.command("train")
def train_cmd(
    model: Annotated[str, typer.Option(help="Model preset, e.g. fermi-solem-1.")] = "fermi-luna-1",
    data: Annotated[Path, typer.Option(help="Dataset directory from build-data.")] = Path(
        "build/data"
    ),
    tokenizer: Annotated[Path, typer.Option(help="tokenizer.json from train-tokenizer.")] = Path(
        "build/tokenizer.json"
    ),
    out: Annotated[
        Path | None, typer.Option(help="Output directory (default: the installed models dir).")
    ] = None,
    steps: Annotated[int, typer.Option(min=1)] = 1000,
    batch_size: Annotated[int, typer.Option(min=1)] = 16,
    lr: Annotated[
        float | None, typer.Option(help="Peak learning rate (default per model).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="mps, cuda, or cpu (default: best).")] = None,
    seed: Annotated[int, typer.Option()] = 0,
    resume: Annotated[bool, typer.Option(help="Continue from a checkpoint in --out.")] = False,
) -> None:
    """Train a Fermi model from scratch on factory data."""
    from askphysics.lm.checkpoints import default_model_dir
    from askphysics.lm.config import get_config
    from askphysics.lm.tokenizer import Tokenizer
    from askphysics.lm.train import DEFAULT_LR, TrainConfig, train

    try:
        config = get_config(model)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {escape(str(exc.args[0]))}")
        raise typer.Exit(code=1) from exc
    out_dir = out or default_model_dir() / config.name
    cfg = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        lr=lr or DEFAULT_LR.get(config.name, 1e-3),
        warmup_steps=max(1, min(500, steps // 20)),
        eval_every=max(1, min(500, steps // 10)),
        checkpoint_every=max(1, min(2000, steps // 4)),
        log_every=max(1, min(100, steps // 50)),
        seed=seed,
        device=device,
    )
    console.print(
        f"Training [bold]{config.name}[/bold] ({config.num_parameters():,} params) "
        f"for {steps:,} steps, writing to {escape(str(out_dir))}"
    )

    def show(entry: dict[str, object]) -> None:
        if "val_loss" in entry:
            console.print(f"  step {entry['step']:>7}  [cyan]val loss {entry['val_loss']}[/cyan]")
        else:
            console.print(
                f"  step {entry['step']:>7}  loss {entry['loss']}  "
                f"lr {float(entry['lr']):.2e}  {entry['target_tokens_per_s']} tok/s"  # type: ignore[arg-type]
            )

    train(config, Tokenizer.load(tokenizer), data, out_dir, cfg, resume=resume, on_log=show)
    console.print(f"[green]Done.[/green] {config.name} saved to {escape(str(out_dir))}")


@model_app.command("info")
def info_cmd(
    directory: Annotated[
        Path | None, typer.Option(help="Models directory (default: the installed models dir).")
    ] = None,
) -> None:
    """List installed Fermi models."""
    import json as _json

    from askphysics.lm.checkpoints import CONFIG_FILE, WEIGHTS_FILE, default_model_dir
    from askphysics.lm.config import ModelConfig

    root = directory or default_model_dir()
    table = Table(
        "Model",
        "Params",
        "Size",
        "Last val loss",
        title=f"Fermi models in {root}",
        title_justify="left",
    )
    found = 0
    for path in sorted(root.glob("*")) if root.exists() else []:
        if not (path / CONFIG_FILE).exists() or not (path / WEIGHTS_FILE).exists():
            continue
        found += 1
        config = ModelConfig(**_json.loads((path / CONFIG_FILE).read_text()))
        size = (path / WEIGHTS_FILE).stat().st_size / 1e6
        val = "-"
        metrics = path / "metrics.jsonl"
        if metrics.exists():
            losses = [
                _json.loads(line)["val_loss"]
                for line in metrics.read_text().splitlines()
                if '"val_loss"' in line
            ]
            val = f"{losses[-1]:.4f}" if losses else "-"
        table.add_row(config.name, f"{config.num_parameters():,}", f"{size:.1f} MB", val)
    if found:
        console.print(table)
    else:
        console.print(
            f"No Fermi models installed in {escape(str(root))} yet. "
            "Train one with [bold]askphysics model train[/bold]."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
