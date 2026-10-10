"""Typer CLI: ``ask``, ``version``, ``validate-data``, and the ``model`` commands."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, cast

import typer
from rich.table import Table
from rich.text import Text

from askphysics import __version__
from askphysics.config import Provider, Settings
from askphysics.data.loader import load_all
from askphysics.errors import AskPhysicsError, ConfigError, DataValidationError
from askphysics.pipeline import Pipeline
from askphysics.ui import (
    answer_card,
    banner,
    make_console,
    safe,
    tolerate_narrow_encodings,
)

if TYPE_CHECKING:
    from askphysics.lm.evaluate import RealReport

# OpenStax Physics questions (ADR-016), in a checkout: all of them, and those with gold
# plans the project wrote, which are the real-question eval and never training data.
TEXTBOOK_QUESTIONS = Path("third_party/openstax-physics/questions.jsonl")
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
    textbook: Annotated[
        Path,
        typer.Option(help="Textbook questions to add as classify examples; skipped if missing."),
    ] = TEXTBOOK_QUESTIONS,
) -> None:
    """Generate training data from the equation database (solved by Noether), plus real
    textbook problems as classify examples (never the real-question eval's)."""
    from askphysics.lm.factory import build_dataset, load_blocklist, textbook_examples

    blocked = load_blocklist(blocklist)
    if not blocked:
        console.print(f"[warn]![/] no eval questions found at {safe(str(blocklist))}")
    extra = []
    if textbook.exists():
        held_out = set()
        if REAL_QUESTIONS.exists():
            lines = REAL_QUESTIONS.read_text(encoding="utf-8").splitlines()
            held_out = {json.loads(line)["id"] for line in lines if line.strip()}
        extra = textbook_examples(textbook, exclude=held_out, seed=seed)
    with console.status(f"[muted]generating {examples:,} examples with {workers} worker(s)"):
        manifest = build_dataset(
            out, examples, seed=seed, workers=workers, blocklist=blocked, extra=extra
        )
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
    if extra:
        console.print(
            f"[muted]Includes {len(extra):,} classify examples from {safe(str(textbook))} "
            "(OpenStax Physics, CC BY 4.0).[/]"
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
    grad_accum: Annotated[
        int,
        typer.Option(
            min=1,
            help="Micro-batches per optimizer step; the effective batch is batch-size x this. "
            "Use it when a full batch doesn't fit in memory.",
        ),
    ] = 1,
    lr: Annotated[
        float | None, typer.Option(help="Peak learning rate (default per model).")
    ] = None,
    device: Annotated[str | None, typer.Option(help="mps, cuda, or cpu (default: best).")] = None,
    precision: Annotated[
        str, typer.Option(help="auto (bf16 on mps/cuda, fp32 on cpu), bf16, or fp32.")
    ] = "auto",
    backend: Annotated[
        str,
        typer.Option(
            help="auto (MLX on an Apple Silicon Mac, torch elsewhere), mlx, or torch. See ADR-019."
        ),
    ] = "auto",
    mlx_cache_gb: Annotated[
        float,
        typer.Option(min=0.5, help="MLX only: most memory kept for reuse, in GB."),
    ] = 4.0,
    mlx_memory_gb: Annotated[
        float | None,
        typer.Option(
            min=1.0,
            help="MLX only: memory limit in GB (default: 70% of system RAM).",
        ),
    ] = None,
    checkpoint_blocks: Annotated[
        bool,
        typer.Option(
            help="Recompute each block's activations in the backward pass, instead of "
            "storing them. Less memory, more compute; works on both backends."
        ),
    ] = False,
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
    from askphysics.lm.backend import on_apple_silicon, resolve_backend
    from askphysics.lm.checkpoints import default_model_dir
    from askphysics.lm.config import get_config
    from askphysics.lm.tokenizer import Tokenizer
    from askphysics.lm.train import DEFAULT_LR, PRECISIONS, TrainConfig, train

    try:
        config = get_config(model)
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from exc
    out_dir = out or default_model_dir() / config.name
    if precision not in PRECISIONS:
        raise _fail(f"unknown --precision {precision!r}; choose from {', '.join(PRECISIONS)}")
    try:
        chosen = resolve_backend(backend, device)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    if chosen != "mlx" and mlx_memory_gb is not None:
        raise _fail("--mlx-memory-gb applies to the mlx backend only")
    cfg = TrainConfig(
        steps=steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr or DEFAULT_LR.get(config.name, 1e-3),
        warmup_steps=max(1, min(500, steps // 20)),
        eval_every=max(1, min(500, steps // 10)),
        checkpoint_every=max(1, min(2000, steps // 4)),
        log_every=max(1, min(50, steps // 100)),
        seed=seed,
        device=device,
        precision=precision,
        prose_steps=prose_steps,
        prose_share=prose_share,
        checkpoint_blocks=checkpoint_blocks,
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
        f"  {config.num_parameters():,} params · {steps:,} steps · {chosen} · → {out_dir}",
        style="muted",
    )
    console.print(head)
    if backend == "auto" and chosen == "torch" and on_apple_silicon():
        console.print(
            "  mlx is not installed, so this runs on torch, which is slower on a Mac and can "
            "use much more memory. Install it with: pip install -e '.[mlx]'",
            style="muted",
        )

    from askphysics.train_view import RunInfo, TrainingMonitor

    try:
        shown_device = _training_device(chosen, device)
    except ConfigError as exc:
        raise _fail(str(exc)) from exc
    info = RunInfo(
        name=config.name,
        backend=chosen,
        device=shown_device,
        steps=steps,
        out_dir=out_dir,
        has_prose=bool(texts),
        prose_steps=prose_steps,
        prose_share=prose_share,
    )
    with TrainingMonitor(console, info) as monitor:
        try:
            if chosen == "mlx":
                from askphysics.lm.mlx_train import train_mlx

                train_mlx(
                    config,
                    Tokenizer.load(tokenizer),
                    data,
                    out_dir,
                    cfg,
                    resume=resume,
                    on_log=monitor.on_log,
                    on_step=monitor.on_step,
                    prose=texts,
                    cache_limit_gb=mlx_cache_gb,
                    memory_limit_gb=mlx_memory_gb,
                )
            else:
                train(
                    config,
                    Tokenizer.load(tokenizer),
                    data,
                    out_dir,
                    cfg,
                    resume=resume,
                    on_log=monitor.on_log,
                    on_step=monitor.on_step,
                    prose=texts,
                )
        except ConfigError as exc:  # a refused resume, or a device the backend can't use
            raise _fail(str(exc)) from exc
    monitor.print_summary()


def _training_device(backend: str, device: str | None) -> str:
    """The device name the training view shows: what the run will use, not what was asked."""
    from askphysics.lm.backend import on_apple_silicon

    if backend == "torch":
        from askphysics.lm.device import select_device

        return select_device(device).type
    return device or ("mps" if on_apple_silicon() else "cpu")


@model_app.command("backend")
def backend_cmd(
    backend: Annotated[
        str, typer.Option(help="auto, mlx, or torch (as for model train).")
    ] = "auto",
    device: Annotated[str | None, typer.Option(help="mps, cuda, or cpu, if chosen.")] = None,
) -> None:
    """Print the training backend that model train would use here (for scripts)."""
    from askphysics.lm.backend import resolve_backend

    try:
        typer.echo(resolve_backend(backend, device))
    except ConfigError as exc:
        typer.echo(f"error: {exc}", err=True)  # stderr, so a script's $(...) still shows it
        raise typer.Exit(code=1) from exc


@model_app.command("bench")
def bench_cmd(
    model: Annotated[str, typer.Option(help="Model preset, e.g. fermi-solem-1.")] = "fermi-solem-1",
    batch_size: Annotated[int, typer.Option(min=1)] = 32,
    width: Annotated[
        int | None,
        typer.Option(
            help="Largest tokens per row, 8 to the model's context (default: the context). "
            "Times each training bucket width up to it: 64, 96, 128, and so on."
        ),
    ] = None,
    steps: Annotated[int, typer.Option(min=1, help="Timed steps per width per setting.")] = 3,
    warmup: Annotated[int, typer.Option(min=0, help="Untimed steps per width per setting.")] = 2,
    device: Annotated[str | None, typer.Option(help="Only bench this device.")] = None,
    plan_steps: Annotated[
        int, typer.Option(min=1, help="Training length to project hours for.")
    ] = 5000,
) -> None:
    """Time training steps on each device and precision and name the fastest."""
    from askphysics.lm.bench import (
        available_settings,
        bench,
        bench_widths,
        format_hours,
        projected_hours,
    )
    from askphysics.lm.config import get_config

    try:
        config = get_config(model)
    except KeyError as exc:
        raise _fail(str(exc.args[0])) from exc
    if width is None:
        width = config.context_length
    if not 8 <= width <= config.context_length:
        raise _fail(f"--width must be between 8 and {config.context_length} for {model}")
    settings = [s for s in available_settings() if device in (None, s[0])]
    if not settings:
        raise _fail(f"no benchmarkable setting for device {device!r}")
    widths = bench_widths(width)
    avg_width = sum(widths) / len(widths)
    width_list = ", ".join(str(w) for w in widths)
    console.print(
        f"  [muted]benchmarking {model}: batch {batch_size} x widths {width_list}, "
        f"{warmup} warmup + {steps} timed steps per width[/]"
    )
    results = bench(
        config, batch_size=batch_size, width=width, steps=steps, warmup=warmup, settings=settings
    )
    table = Table(
        title=Text(f"{model} training speed", style="brand"),
        title_justify="left",
        border_style="muted",
        header_style="label",
    )
    table.add_column("device")
    table.add_column("precision")
    table.add_column("s/step", justify="right")
    table.add_column("tokens/s", justify="right")
    table.add_column("GPU memory", justify="right")
    table.add_column(f"{plan_steps:,} steps", justify="right")
    for r in results:
        if r.error is not None:
            table.add_row(r.device, r.precision, "[bad]failed[/]", "", "", "")
            continue
        hours = projected_hours(r.tokens_per_s, plan_steps, batch_size, avg_width)
        gpu = "-" if r.gpu_mem_gb is None else f"{r.gpu_mem_gb:.1f} GB"
        table.add_row(
            r.device,
            r.precision,
            f"{r.seconds_per_step:.3f}",
            f"{r.tokens_per_s:,.0f}",
            gpu,
            format_hours(hours),
        )
    console.print(table)
    for r in results:
        if r.error is not None:
            console.print(f"  [bad]{r.device} {r.precision}:[/] {safe(r.error)}")
    ok = [r for r in results if r.error is None]
    if not ok:
        raise typer.Exit(1)
    best = max(ok, key=lambda r: r.tokens_per_s)
    console.print(
        f"[ok]fastest: {best.device} {best.precision} ({best.tokens_per_s:,.0f} tokens/s).[/] "
        f"Train with --device {best.device} --precision {best.precision}"
    )


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
    rescue_with: Annotated[
        str | None,
        typer.Option(help="Bigger installed model to retry the questions this one misses."),
    ] = None,
    rescue_directory: Annotated[
        Path | None,
        typer.Option(help="Rescue model directory (default: the installed models dir)."),
    ] = None,
) -> None:
    """Score a model on held-out questions: right category, plans that compute right, and
    how often the answer `ask` would give is wrong while passing every check. Then ask the
    real textbook questions through the whole pipeline, the model doing every stage.
    With --rescue-with, a bigger model also plans the misses, as ask's escalation does."""
    from collections import Counter

    from askphysics.eval_view import (
        EvalMonitor,
        print_results,
        report_sections,
        working,
    )
    from askphysics.lm.checkpoints import default_model_dir, load_model
    from askphysics.lm.device import select_device
    from askphysics.lm.evaluate import (
        EvalTick,
        evaluate_real,
        evaluate_tasks,
        read_real_questions,
        sample_examples,
    )
    from askphysics.lm.factory import read_examples
    from askphysics.lm.generate import Decoder

    model_dir = directory or default_model_dir() / model
    try:
        with working(console, f"loading {model}"):
            loaded, tokenizer = load_model(model_dir, select_device(device))
    except AskPhysicsError as exc:
        raise _fail(str(exc)) from exc
    picked = sample_examples(read_examples(data / "val"), examples, seed)
    if not picked:
        raise _fail(f"no validation examples in {data / 'val'}; run build-data first")
    decoder = Decoder(loaded, tokenizer)
    if rescue_directory is not None and rescue_with is None:
        raise _fail("--rescue-directory needs --rescue-with")
    rescue: Decoder | None = None
    if rescue_with is not None:
        rescue_dir = rescue_directory or default_model_dir() / rescue_with
        try:
            with working(console, f"loading {rescue_with}"):
                rescuer, rescue_tokenizer = load_model(rescue_dir, select_device(device))
        except AskPhysicsError as exc:
            raise _fail(str(exc)) from exc
        rescue = Decoder(rescuer, rescue_tokenizer)
    store = load_all()
    with EvalMonitor(
        console, model, Counter(e.task for e in picked), rescue_with=rescue_with
    ) as monitor:
        report = evaluate_tasks(
            decoder,
            picked,
            store,
            attempts=attempts,
            rescue=rescue,
            on_tick=monitor.on_tick,
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
        total = len(questions)
        with EvalMonitor(
            console,
            model,
            {"question": total},
            label="asking",
            noun="real textbook questions",
            stats=False,
        ) as asking:
            real_report = evaluate_real(
                pipeline.solve,
                questions,
                store,
                on_progress=lambda n: asking.on_tick(EvalTick("score", n, total, "question")),
            )
        report.real = asdict(real_report)
    (model_dir / "eval.json").write_text(report.to_json(), encoding="utf-8")

    console.print()
    sections, footer = report_sections(report, attempts=attempts, rescue_with=rescue_with)
    print_results(console, f"{model} on held-out questions", sections, footer)
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
    from askphysics.eval_view import print_results, real_sections

    console.print()
    print_results(
        console, f"{model} on {report.questions} real textbook questions", real_sections(report)
    )
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
    from askphysics.files_view import PullView
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
    with PullView(console) as view:
        for name in names:
            if name in manifest:
                files = [(f.name, f.size) for f in manifest[name].files]
                view.start_model(name, files)
            try:
                downloaded = pull_models(
                    [name], root, manifest=manifest, force=force, progress=view.update
                )
            except AskPhysicsError as exc:
                raise _fail(str(exc)) from exc
            pinned = [(f.name, f.size, f.sha256) for f in manifest[name].files]
            view.finish_model(name, pinned, downloaded[name], root / name)


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
