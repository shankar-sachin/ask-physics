"""Typer CLI: ``ask``, ``version``, ``validate-data``, and the ``model`` commands."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, cast

import typer
from rich.table import Table
from rich.text import Text

from askphysics import __version__
from askphysics.config import Provider, Settings
from askphysics.data.loader import load_all
from askphysics.errors import AskPhysicsError, DataValidationError
from askphysics.pipeline import Pipeline
from askphysics.ui import (
    answer_card,
    banner,
    make_console,
    safe,
    tolerate_narrow_encodings,
    training_progress,
)

if TYPE_CHECKING:
    from askphysics.lm.evaluate import RealReport

# OpenStax Physics questions with project-written gold plans (ADR-016), in a checkout.
REAL_QUESTIONS = Path("third_party/openstax-physics/real_eval.jsonl")

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


@app.command("validate-data")
def validate_data() -> None:
    """Validate every seed data file (schema, SymPy parse, units, cross-references)."""
    try:
        store = load_all()
    except DataValidationError as exc:
        console.print(
            Text(f"✗ data validation failed with {len(exc.problems)} problem(s)", style="bad")
        )
        for problem in exc.problems:
            console.print(f"  [muted]•[/] {safe(problem)}")
        raise typer.Exit(code=1) from exc
    table = Table(
        title=Text("✓ seed data: all valid", style="ok"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("file")
    table.add_column("entries", justify="right", style="value")
    for name, count in store.summary().items():
        table.add_row(name.replace("_", " "), str(count))
    console.print(table)


model_app = typer.Typer(
    help="Build training data for, train, and inspect the Fermi models.",
    no_args_is_help=True,
    rich_markup_mode="rich",
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
    """Generate training data from the equation database (solved by Noether)."""
    from askphysics.lm.factory import build_dataset, load_blocklist

    blocked = load_blocklist(blocklist)
    if not blocked:
        console.print(f"[warn]![/] no eval questions found at {safe(str(blocklist))}")
    with console.status(f"[muted]generating {examples:,} examples with {workers} worker(s)"):
        manifest = build_dataset(out, examples, seed=seed, workers=workers, blocklist=blocked)
    table = Table(
        title=Text(f"✓ dataset written to {out}", style="ok"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("split / task")
    table.add_column("examples", justify="right", style="value")
    for key, count in manifest["counts"].items():
        table.add_row(key, f"{count:,}")
    console.print(table)
    console.print(
        f"[muted]Dropped {manifest['dropped']:,} attempts "
        "(unsolvable, or too close to an eval question).[/]"
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
    prose: Annotated[
        Path | None,
        typer.Option(
            help="Real prose to learn words from, e.g. third_party/openstax-physics/prose.jsonl."
        ),
    ] = None,
) -> None:
    """Train the byte-level BPE tokenizer on the dataset's training split."""
    from askphysics.lm.corpus import read_prose
    from askphysics.lm.train import train_tokenizer

    texts = read_prose(prose) if prose else []
    with console.status(f"[muted]learning up to {vocab_size:,} tokens from {safe(str(data))}"):
        tokenizer = train_tokenizer(data, vocab_size, max_examples, prose=texts)
    out.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(out)
    console.print(f"[ok]✓[/] tokenizer with {tokenizer.vocab_size:,} tokens → {safe(str(out))}")


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
    prose: Annotated[
        Path | None,
        typer.Option(help="Real prose (prose.jsonl) for a language-modeling stage (ADR-016)."),
    ] = None,
    prose_steps: Annotated[
        int, typer.Option(min=0, help="Steps of prose alone before the tasks.")
    ] = 0,
    prose_share: Annotated[
        float, typer.Option(min=0.0, max=1.0, help="Share of later batches that are prose.")
    ] = 0.0,
) -> None:
    """Train a Fermi model from scratch on factory data, optionally after real prose."""
    from askphysics.lm.checkpoints import default_model_dir
    from askphysics.lm.config import get_config
    from askphysics.lm.tokenizer import Tokenizer
    from askphysics.lm.train import DEFAULT_LR, TrainConfig, train

    try:
        config = get_config(model)
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from exc
    out_dir = out or default_model_dir() / config.name
    cfg = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        lr=lr or DEFAULT_LR.get(config.name, 1e-3),
        warmup_steps=max(1, min(500, steps // 20)),
        eval_every=max(1, min(500, steps // 10)),
        checkpoint_every=max(1, min(2000, steps // 4)),
        log_every=max(1, min(50, steps // 100)),
        seed=seed,
        device=device,
        prose_steps=prose_steps,
        prose_share=prose_share,
    )
    if (prose_steps or prose_share) and prose is None:
        raise _fail("--prose-steps and --prose-share need --prose")
    if prose_steps >= steps:
        raise _fail(
            f"--prose-steps ({prose_steps}) must be fewer than --steps ({steps}): prose-only "
            "steps come first, so the model would never train on the tasks. For example, "
            f"--steps {prose_steps + 3000} trains 3000 task steps after the prose."
        )
    from askphysics.lm.corpus import read_prose

    texts = read_prose(prose) if prose else []
    head = banner()
    head.append(f"\n  training {config.name}", style="value")
    head.append(
        f"  {config.num_parameters():,} params · {steps:,} steps · → {out_dir}", style="muted"
    )
    console.print(head)

    last_val: dict[str, Any] = {}
    with training_progress(console) as progress:
        task = progress.add_task(config.name, total=steps, loss="…", val="…", speed="")

        def show(entry: dict[str, Any]) -> None:
            if "val_loss" in entry:
                last_val.update(entry)
                progress.update(task, val=f"{entry['val_loss']:.3f}")
            else:
                progress.update(
                    task,
                    completed=entry["step"],
                    loss=f"{entry['loss']:.3f}",
                    speed=f"{entry['target_tokens_per_s']:,.0f} tok/s",
                )

        train(
            config,
            Tokenizer.load(tokenizer),
            data,
            out_dir,
            cfg,
            resume=resume,
            on_log=show,
            prose=texts,
        )
        progress.update(task, completed=steps)
    by_task = [
        f"{key.removeprefix('val_loss_')} {value:.3f}"
        for key, value in last_val.items()
        if key.startswith("val_loss_")
    ]
    if by_task:
        console.print("  val loss by task: " + " · ".join(by_task), style="muted")
    console.print(f"[ok]✓[/] {config.name} saved to {safe(str(out_dir))}")


@model_app.command("eval")
def eval_cmd(
    model: Annotated[str, typer.Option(help="Installed model to score.")] = "fermi-tellus-1",
    data: Annotated[Path, typer.Option(help="Dataset directory from build-data.")] = Path(
        "build/data"
    ),
    examples: Annotated[int, typer.Option(min=1, help="Examples per task (classify, plan).")] = 200,
    directory: Annotated[
        Path | None, typer.Option(help="Model directory (default: the installed models dir).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="mps, cuda, or cpu (default: best).")] = None,
    seed: Annotated[int, typer.Option()] = 0,
    attempts: Annotated[
        int, typer.Option(min=1, help="Plan attempts per question, as `ask` makes them.")
    ] = Settings().plan_attempts,
    real: Annotated[
        Path,
        typer.Option(help="Real textbook questions with gold plans; skipped if missing."),
    ] = REAL_QUESTIONS,
) -> None:
    """Score a model on held-out questions: right category, plans that compute right, and
    how often the answer `ask` would give is wrong while passing every check. Then ask the
    real textbook questions through the whole pipeline, the model doing every stage."""
    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn

    from askphysics.lm.checkpoints import default_model_dir, load_model
    from askphysics.lm.device import select_device
    from askphysics.lm.evaluate import (
        evaluate_real,
        evaluate_tasks,
        read_real_questions,
        sample_examples,
    )
    from askphysics.lm.factory import read_examples
    from askphysics.lm.generate import Decoder

    model_dir = directory or default_model_dir() / model
    try:
        loaded, tokenizer = load_model(model_dir, select_device(device))
    except AskPhysicsError as exc:
        raise _fail(str(exc)) from exc
    picked = sample_examples(read_examples(data / "val"), examples, seed)
    if not picked:
        raise _fail(f"no validation examples in {data / 'val'}; run build-data first")
    decoder = Decoder(loaded, tokenizer)
    store = load_all()
    with Progress(
        TextColumn(f"[brand]scoring {model}"),
        BarColumn(bar_width=32, complete_style="accent", finished_style="ok"),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("eval", total=len(picked))
        report = evaluate_tasks(
            decoder,
            picked,
            store,
            attempts=attempts,
            on_progress=lambda n: progress.update(task, completed=n),
        )
    real_report = None
    if real.exists():
        from askphysics.llm.base import Roster
        from askphysics.llm.fermi_client import FermiClient
        from askphysics.retrieval.keyword import KeywordRetriever

        questions = read_real_questions(real)
        client = FermiClient(model, store, directory=model_dir, decoder=decoder)
        pipeline = Pipeline(
            llm=client,
            retriever=KeywordRetriever(store.equations.values(), store.examples.values()),
            data=store,
            settings=Settings(),
            roster=Roster(classify=client, plan=(client,) * attempts, explain=client),
        )
        with Progress(
            TextColumn(f"[brand]asking {model} real questions"),
            BarColumn(bar_width=32, complete_style="accent", finished_style="ok"),
            MofNCompleteColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("real", total=len(questions))
            real_report = evaluate_real(
                pipeline.solve,
                questions,
                store,
                on_progress=lambda n: progress.update(task, completed=n),
            )
        report.real = asdict(real_report)
    (model_dir / "eval.json").write_text(report.to_json(), encoding="utf-8")

    table = Table(
        title=Text(f"{model} on held-out questions", style="brand"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("check")
    table.add_column("score", justify="right")
    rows = [
        (f"right category ({report.classify_examples} classify)", report.category_accuracy),
        (f"right equation ({report.plan_examples} plan)", report.equation_accuracy),
        ("right target", report.target_accuracy),
        ("right numbers and units", report.knowns_accuracy),
        ("[accent]valid plan (right answer)[/]", report.valid_plan_rate),
    ]
    for label, score in rows:
        table.add_row(label, f"{score:.1%}")
    table.add_section()
    tries = f"up to {attempts} tries, {report.mean_tries:.2f} on average"
    table.add_row(
        f"[accent]right answer as ask gives it[/] ({tries})", f"{report.routed_right_rate:.1%}"
    )
    table.add_row("wrong, but flagged or refused", f"{report.flagged_wrong_rate:.1%}")
    table.add_row("[bad]confidently wrong[/]", f"{report.confidently_wrong_rate:.1%}")
    console.print(table)
    for f in report.confidently_wrong[:5]:
        console.print(f"  [bad]confidently wrong:[/] {safe(f['question'])}")
        console.print(f"    [muted]expected[/] {safe(str(f['expected']))}")
        console.print(f"    [muted]got     [/] {safe(str(f['got']))}")
    for f in report.failures[:5]:
        console.print(f"  [muted]{f['task']} miss:[/] {safe(f['question'])}")
        console.print(f"    [muted]expected[/] {safe(str(f['expected']))}")
        console.print(f"    [muted]got     [/] {safe(str(f['got']))}")
    if real_report is not None:
        _show_real(model, real_report, real)
    console.print(f"[ok]✓[/] full report in {safe(str(model_dir / 'eval.json'))}")


def _show_real(model: str, report: RealReport, path: Path) -> None:
    table = Table(
        title=Text(f"{model} on {report.questions} real textbook questions", style="brand"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("check")
    table.add_column("score", justify="right")
    table.add_row("classified standard", f"{report.standard_rate:.1%}")
    table.add_row("right equations retrieved", f"{report.retrieved_rate:.1%}")
    table.add_row("[accent]right answer as ask gives it[/]", f"{report.right_rate:.1%}")
    table.add_row("no answer, or wrong but flagged", f"{report.flagged_wrong_rate:.1%}")
    table.add_row("[bad]confidently wrong[/]", f"{report.confidently_wrong_rate:.1%}")
    console.print(table)
    for miss in report.misses[:5]:
        console.print(f"  [muted]{safe(miss['stage'])}:[/] {safe(miss['question'])}")
        console.print(f"    [muted]expected[/] {safe(miss['expected'])}")
        console.print(f"    [muted]got     [/] {safe(miss['got'])}")
    console.print(f"  [muted]questions from {safe(str(path))} (OpenStax Physics, CC BY 4.0)[/]")


@model_app.command("pull")
def pull_cmd(
    model: Annotated[
        list[str] | None,
        typer.Option(help="A model to download (repeatable). Default: tellus and solem."),
    ] = None,
    all_models: Annotated[
        bool, typer.Option("--all", help="Also download celeste (about 240 MB).")
    ] = False,
    force: Annotated[
        bool, typer.Option(help="Replace a locally trained model of the same name.")
    ] = False,
    directory: Annotated[
        Path | None, typer.Option(help="Models directory (default: the installed models dir).")
    ] = None,
    if_published: Annotated[
        bool,
        typer.Option(help="Succeed quietly when nothing is published yet (for installers)."),
    ] = False,
) -> None:
    """Download the published Fermi models, checked against the pinned manifest (ADR-012)."""
    from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn

    from askphysics.llm.routing import CELESTE, SOLEM, TELLUS
    from askphysics.lm.paths import default_model_dir
    from askphysics.lm.weights import pull as pull_models
    from askphysics.lm.weights import read_manifest

    root = directory or default_model_dir()
    try:
        manifest = read_manifest()
    except AskPhysicsError as exc:
        raise _fail(str(exc)) from exc
    if not manifest:
        message = (
            "no trained weights are published for this version of askphysics yet; "
            "to train your own, see docs/TRAINING.md"
        )
        if if_published:
            console.print(message, style="muted")
            return
        raise _fail(message)
    names = list(model) if model else [n for n in (TELLUS, SOLEM) if n in manifest]
    if all_models and CELESTE not in names:
        names.append(CELESTE)
    with Progress(
        TextColumn("[brand]{task.description}"),
        BarColumn(bar_width=32, complete_style="accent", finished_style="ok"),
        DownloadColumn(),
        console=console,
    ) as progress:
        tasks: dict[str, Any] = {}

        def update(name: str, done: int, total: int) -> None:
            if name not in tasks:
                tasks[name] = progress.add_task(name, total=total)
            progress.update(tasks[name], completed=done)

        results: dict[str, bool] = {}
        for name in names:
            tasks.clear()
            progress.console.print(f"[label]{name}[/]")
            try:
                results |= pull_models(
                    [name], root, manifest=manifest, force=force, progress=update
                )
            except AskPhysicsError as exc:
                raise _fail(str(exc)) from exc
    for name, downloaded in results.items():
        state = "downloaded and verified" if downloaded else "already up to date"
        console.print(f"[ok]✓[/] {name} {state} in {safe(str(root / name))}")


@model_app.command("package")
def package_cmd(
    model: Annotated[str, typer.Option(help="Installed model to package.")],
    release: Annotated[
        str, typer.Option(help="GitHub release tag the assets will be uploaded to.")
    ],
    out: Annotated[Path, typer.Option(help="Where to write the release files.")] = Path(
        "build/release"
    ),
    attribution: Annotated[
        Path | None,
        typer.Option(help="ATTRIBUTION.md of the prose the model trained on, to ship with it."),
    ] = None,
    directory: Annotated[
        Path | None, typer.Option(help="Model directory (default: the installed models dir).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="mps, cuda, or cpu (default: best).")] = None,
    measure: Annotated[
        bool, typer.Option(help="Time a few questions through the pipeline for the card.")
    ] = True,
) -> None:
    """Maintainers: package a trained model for release (bf16 weights, model card, manifest)."""
    from askphysics.lm.checkpoints import default_model_dir
    from askphysics.lm.package import package
    from askphysics.lm.weights import manifest_path

    source = directory or default_model_dir() / model
    try:
        with console.status(f"[muted]packaging {model}"):
            entry = package(
                model,
                source,
                out,
                release,
                load_all(),
                attribution=attribution,
                device=device,
                measure=measure,
            )
    except AskPhysicsError as exc:
        raise _fail(str(exc)) from exc
    size = sum(f["size"] for f in entry["files"].values()) / 1e6
    console.print(f"[ok]✓[/] {model}: {len(entry['files'])} files, {size:.1f} MB")
    console.print(f"  upload everything in {safe(str(out / 'assets'))} to the release {release}")
    console.print(f"  then commit {safe(str(manifest_path()))}, which now pins them")


@model_app.command("info")
def info_cmd(
    directory: Annotated[
        Path | None, typer.Option(help="Models directory (default: the installed models dir).")
    ] = None,
) -> None:
    """List installed Fermi models."""
    from askphysics.lm.checkpoints import CONFIG_FILE, WEIGHTS_FILE, default_model_dir
    from askphysics.lm.config import ModelConfig

    root = directory or default_model_dir()
    table = Table(
        title=Text(f"Fermi models in {root}", style="brand"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("model")
    table.add_column("params", justify="right")
    table.add_column("size", justify="right")
    table.add_column("last val loss", justify="right")
    found = 0
    for path in sorted(root.glob("*")) if root.exists() else []:
        if not (path / CONFIG_FILE).exists() or not (path / WEIGHTS_FILE).exists():
            continue
        found += 1
        config = ModelConfig(**json.loads((path / CONFIG_FILE).read_text()))
        size = (path / WEIGHTS_FILE).stat().st_size / 1e6
        val = "-"
        metrics = path / "metrics.jsonl"
        if metrics.exists():
            losses = [
                json.loads(line)["val_loss"]
                for line in metrics.read_text().splitlines()
                if '"val_loss"' in line
            ]
            val = f"{losses[-1]:.4f}" if losses else "-"
        table.add_row(
            f"[accent]{config.name}[/]", f"{config.num_parameters():,}", f"{size:.1f} MB", val
        )
    if found:
        console.print(table)
    else:
        console.print(
            f"No Fermi models installed in {safe(str(root))} yet. "
            "Train one with [brand]askphysics model train[/]."
        )


if __name__ == "__main__":  # pragma: no cover
    app()
