"""Package a trained Fermi model for release: bf16 weights, a model card, and pinned assets.

    askphysics model package --model fermi-solem-1 --release models-v0.4.0 \\
        --attribution build/corpus/ATTRIBUTION.md

Writes, under ``out``:

- ``<model>/``: the packaged model, an ordinary model directory (``model eval
  --directory`` scores exactly what ships);
- ``assets/``: the same files under release asset names, to upload to the GitHub release.

and adds the model to the weights manifest the package ships (ADR-012), which pins each
asset's URL, size, and sha256.
"""

from __future__ import annotations

import json
import shutil
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from askphysics.config import Settings
from askphysics.data.loader import DataStore
from askphysics.errors import ConfigError
from askphysics.lm.checkpoints import CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE, load_model
from askphysics.lm.config import ModelConfig
from askphysics.lm.weights import pin, write_manifest

CARD_FILE = "MODEL_CARD.md"
ATTRIBUTION_FILE = "ATTRIBUTION.md"
METRICS_FILE = "metrics.jsonl"
SUMMARY_FILE = "training_summary.json"
EVAL_FILE = "eval.json"

# Timed through the whole pipeline, as `askphysics ask` answers them.
BENCHMARK_QUESTIONS = (
    "How fast does a falling object hit the ground if it is dropped from 20 m?",
    "A 1200 kg car accelerates at 3 m/s^2. What net force acts on it?",
    "What current flows through a 220 ohm resistor across a 9 V battery?",
    "How much kinetic energy does a 0.145 kg baseball moving at 40 m/s carry?",
    "A wave has a frequency of 440 Hz and a wavelength of 0.78 m. How fast does it travel?",
)


@dataclass(frozen=True)
class Throughput:
    """Seconds per question through the pipeline, after the weights are loaded."""

    device: str
    median_s: float
    worst_s: float
    questions: int


def to_bf16(source: Path, dest: Path) -> None:
    """Copy a model directory with its weights stored in bfloat16, about half the size.

    The weights' metadata (config and task format) is kept, so ``load_model`` accepts the
    copy, and loading casts the weights back to float32.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with safe_open(str(source / WEIGHTS_FILE), framework="pt") as f:
        metadata = f.metadata() or {}
    state = {
        k: v.to(torch.bfloat16) if v.is_floating_point() else v
        for k, v in load_file(str(source / WEIGHTS_FILE)).items()
    }
    save_file(state, str(dest / WEIGHTS_FILE), metadata=metadata)
    for name in (CONFIG_FILE, TOKENIZER_FILE):
        shutil.copyfile(source / name, dest / name)


def measure_throughput(
    name: str,
    directory: Path,
    data: DataStore,
    device: str | None = None,
    questions: Sequence[str] = BENCHMARK_QUESTIONS,
) -> Throughput:
    """Time ``questions`` through the pipeline with this model doing every stage."""
    from askphysics.llm.base import Roster
    from askphysics.llm.fermi_client import FermiClient
    from askphysics.pipeline import Pipeline
    from askphysics.retrieval.keyword import KeywordRetriever

    settings = Settings()
    client = FermiClient(name, data, directory=directory, device=device)
    roster = Roster(classify=client, plan=(client,) * settings.plan_attempts, explain=client)
    pipeline = Pipeline(
        llm=client,
        retriever=KeywordRetriever(data.equations.values(), data.examples.values()),
        data=data,
        settings=settings,
        roster=roster,
    )
    pipeline.run(questions[0])  # load the weights and warm up
    times = []
    for question in questions:
        start = time.perf_counter()
        pipeline.run(question)
        times.append(time.perf_counter() - start)
    used = client.decoder.engine.device
    return Throughput(
        device=used,
        median_s=round(statistics.median(times), 2),
        worst_s=round(max(times), 2),
        questions=len(times),
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def model_card(
    name: str,
    config: ModelConfig,
    parameters: int,
    release: str,
    *,
    summary: dict[str, Any] | None,
    metrics: list[dict[str, Any]],
    report: dict[str, Any] | None,
    throughput: Throughput | None,
    attribution: bool,
) -> str:
    """The model card: what the model is, how it was trained, how well it does, and credit."""
    lines = [
        f"# {name}",
        "",
        "A Fermi model from [Ask Physics](https://github.com/shankar-sachin/ask-physics), "
        "trained from scratch. It reads physics questions, writes plans that name equations "
        "and copy values, and writes explanations. It never does arithmetic: Noether "
        "(SymPy and Pint) computes every number.",
        "",
        f"Release `{release}`. Weights stored in bfloat16 as safetensors. License: MIT.",
        "",
        "## Architecture",
        "",
        "| | |",
        "|---|---|",
        f"| Parameters | {parameters:,} |",
        f"| Layers | {config.n_layers} |",
        f"| Width (d_model) | {config.d_model} |",
        f"| Attention heads | {config.n_heads} |",
        f"| Context | {config.context_length} tokens |",
        f"| Vocabulary | {config.vocab_size:,} |",
        "",
        "A decoder-only transformer: tied embeddings, RMSNorm, rotary position embeddings, "
        "SwiGLU, no biases.",
        "",
        "## Training",
        "",
    ]
    if summary:
        train = summary.get("train", {})
        rows = [
            ("Steps", train.get("steps")),
            ("Batch size", train.get("batch_size")),
            ("Task examples", summary.get("train_examples")),
            ("Prose paragraphs", summary.get("prose_paragraphs") or None),
            ("Prose-only steps first", train.get("prose_steps") or None),
            ("Share of later batches on prose", train.get("prose_share") or None),
            ("Device", summary.get("device")),
        ]
        lines += ["| | |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |"
                  for k, v in rows if v is not None]  # fmt: skip
        lines.append("")
    speeds = [m["target_tokens_per_s"] for m in metrics if "target_tokens_per_s" in m]
    if speeds:
        lines += [f"Training throughput: {speeds[-1]:,.0f} target tokens per second.", ""]
    evals = [m for m in metrics if "val_loss" in m]
    if evals:
        picked = evals[:: max(1, len(evals) // 10)]
        if picked[-1] is not evals[-1]:
            picked.append(evals[-1])
        prose = any("val_loss_prose" in m for m in picked)
        lines += [
            "Validation loss over training:",
            "",
            "| Step | Validation loss |" + (" Prose validation loss |" if prose else ""),
            "|---:|---:|" + ("---:|" if prose else ""),
        ]
        for m in picked:
            row = f"| {m['step']:,} | {m['val_loss']:.4f} |"
            if prose:
                row += f" {m['val_loss_prose']:.4f} |" if "val_loss_prose" in m else " |"
            lines.append(row)
        lines.append("")
    lines += ["## Results", ""]
    if report:
        attempts = report.get("attempts", 1)
        lines += [
            "Held-out questions from templates the model never trained on "
            f"(`askphysics model eval`, {report['classify_examples']:,} classification and "
            f"{report['plan_examples']:,} planning questions):",
            "",
            "| Check | Score |",
            "|---|---:|",
            f"| Right category | {report['category_accuracy']:.1%} |",
            f"| Right equation | {report['equation_accuracy']:.1%} |",
            f"| Right target | {report['target_accuracy']:.1%} |",
            f"| Right numbers and units | {report['knowns_accuracy']:.1%} |",
            f"| Valid plan, right answer | {report['valid_plan_rate']:.1%} |",
            f"| Right answer after up to {attempts} tries | {report['routed_right_rate']:.1%} |",
            f"| Wrong, but flagged or refused | {report['flagged_wrong_rate']:.1%} |",
            f"| Confidently wrong | {report['confidently_wrong_rate']:.1%} |",
            "",
            "*Confidently wrong* is a wrong answer that passes every sanity check; the "
            "target is 0.1% or less.",
            "",
        ]
    else:
        lines += ["Not evaluated yet: run `askphysics model eval` before packaging.", ""]
    if throughput:
        lines += [
            f"Answering speed on {throughput.device}: {throughput.median_s} s per question "
            f"(median of {throughput.questions}; slowest {throughput.worst_s} s), through "
            "the whole pipeline with this model doing every stage.",
            "",
        ]
    lines += [
        "## Limits",
        "",
        "A small model trained from scratch understands phrasings close to its training "
        "data and stumbles on unusual ones. Constrained decoding keeps those stumbles safe: "
        "the answer degrades instead of inventing an equation or a number.",
        "",
        "## Data and credit",
        "",
        "Task examples are generated by the project's data factory from its own equation "
        "database and solved by Noether. No outside model wrote any training data.",
        "",
    ]
    if attribution:
        lines += [
            "The model also learned English from the prose corpus (ADR-017): OpenStax "
            "textbooks under CC BY 4.0 and public-domain books. Every source, its license, "
            f"and the changes made are listed in `{ATTRIBUTION_FILE}`, which ships with "
            "these weights. OpenStax and Rice University do not endorse Ask Physics.",
            "",
        ]
    return "\n".join(lines)


def package(
    name: str,
    source: Path,
    out: Path,
    release: str,
    data: DataStore,
    *,
    attribution: Path | None = None,
    device: str | None = None,
    measure: bool = True,
    manifest: Path | None = None,
) -> dict[str, Any]:
    """Package the model in ``source``; returns its new manifest entry.

    Raises:
        ConfigError: the model doesn't load, or ``attribution`` doesn't exist.
    """
    if attribution is not None and not attribution.is_file():
        raise ConfigError(f"attribution file {attribution} does not exist")
    model_dir, asset_dir = out / name, out / "assets"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    to_bf16(source, model_dir)
    model, _ = load_model(model_dir)  # proves the packaged copy loads
    throughput = measure_throughput(name, model_dir, data, device) if measure else None
    card = model_card(
        name,
        model.config,
        model.num_parameters(),
        release,
        summary=_read_json(source / SUMMARY_FILE),
        metrics=_metrics(source / METRICS_FILE),
        report=_read_json(source / EVAL_FILE),
        throughput=throughput,
        attribution=attribution is not None,
    )
    (model_dir / CARD_FILE).write_text(card, encoding="utf-8")
    if attribution is not None:
        shutil.copyfile(attribution, model_dir / ATTRIBUTION_FILE)

    assets = {
        WEIGHTS_FILE: f"{name}.safetensors",
        CONFIG_FILE: f"{name}.config.json",
        TOKENIZER_FILE: f"{name}.tokenizer.json",
        CARD_FILE: f"{name}.card.md",
    }
    if attribution is not None:
        assets[ATTRIBUTION_FILE] = f"{name}.attribution.md"
    asset_dir.mkdir(parents=True, exist_ok=True)
    for installed, asset in assets.items():
        shutil.copyfile(model_dir / installed, asset_dir / asset)
    entry = pin(asset_dir, name, release, assets)
    write_manifest({name: entry}, manifest)
    return entry


__all__ = ["Throughput", "measure_throughput", "model_card", "package", "to_bf16"]
