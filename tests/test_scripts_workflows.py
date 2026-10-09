"""The workflow scripts (``scripts/*.sh``), run for real where cheap and with DRY_RUN=1
otherwise, so no test builds data or trains anything."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_evals

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
WORKFLOWS = sorted(p for p in SCRIPTS.glob("*.sh") if p.name not in {"lib.sh", "build_site.sh"})


def _run(
    script: str, *args: str, tmp_path: Path, dry: bool = True
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "NO_CAFFEINATE": "1",
        "ASKPHYSICS_MODEL_DIR": str(tmp_path / "models"),
        "ASKPHYSICS_BACKUP_DIR": str(tmp_path / "backups"),
    }
    if dry:
        env["DRY_RUN"] = "1"
    return subprocess.run(
        ["sh", str(SCRIPTS / script), *args], env=env, capture_output=True, text=True, check=False
    )


def _commands(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line[2:] for line in result.stdout.splitlines() if line.startswith("+ ")]


@pytest.mark.parametrize("script", WORKFLOWS, ids=lambda p: p.name)
def test_every_script_parses_and_explains_itself(script: Path, tmp_path: Path) -> None:
    subprocess.run(["sh", "-n", str(script)], check=True)
    result = _run(script.name, "--help", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    first = result.stdout.splitlines()[0]
    assert first and not first.startswith("#") and not first.startswith("set ")


def test_train_backs_up_scores_both_and_compares(tmp_path: Path) -> None:
    (tmp_path / "models" / "fermi-solem-1").mkdir(parents=True)
    prose = tmp_path / "prose.jsonl"
    prose.write_text("{}\n")
    result = _run("train.sh", "fermi-solem-1", "--data", str(tmp_path / "data"),
                  "--tokenizer", str(tmp_path / "tok.json"), "--prose", str(prose),
                  tmp_path=tmp_path)  # fmt: skip
    assert result.returncode == 0, result.stderr
    steps = [c.split(" ")[0:3] for c in _commands(result)]
    assert steps == [
        ["mkdir", "-p", str(tmp_path / "backups")],
        ["cp", "-R", str(tmp_path / "models" / "fermi-solem-1")],
        ["rm", "-rf", str(tmp_path / "data")],
        ["askphysics", "model", "build-data"],
        ["askphysics", "model", "train-tokenizer"],
        ["askphysics", "model", "eval"],  # the backed-up model: the baseline
        ["askphysics", "model", "bench"],
        ["askphysics", "model", "train"],
        ["askphysics", "model", "eval"],  # the new one
        ["python3", str(SCRIPTS / "compare_evals.py"), steps[-1][2]],
    ]
    assert steps[-1][2].startswith(str(tmp_path / "backups" / "fermi-solem-1-"))
    train = next(c for c in _commands(result) if c.startswith("askphysics model train "))
    assert f"--prose {prose} --prose-steps 2000 --prose-share 0.1" in train
    assert "--steps 5000" in train and "--resume" not in train
    assert train.endswith("--precision auto")


def test_train_refuses_a_run_with_no_task_steps(tmp_path: Path) -> None:
    # --steps 3000 with --prose-steps 3000 once spent a whole night on prose alone.
    prose = tmp_path / "prose.jsonl"
    prose.write_text("{}\n")
    result = _run("train.sh", "fermi-solem-1", "--steps", "3000", "--prose", str(prose),
                  "--prose-steps", "3000", tmp_path=tmp_path)  # fmt: skip
    assert result.returncode != 0 and "never trains on the tasks" in result.stderr


def test_train_resume_skips_backup_and_rebuilds(tmp_path: Path) -> None:
    (tmp_path / "models" / "fermi-tellus-1").mkdir(parents=True)
    result = _run("train.sh", "fermi-tellus-1", "--resume", "--no-eval", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert _commands(result) == [
        "askphysics model bench --model fermi-tellus-1 --batch-size 32 --plan-steps 1500",
        "askphysics model train --model fermi-tellus-1 --data build/data --tokenizer "
        "build/tokenizer.json --steps 1500 --batch-size 32 --backend torch "
        "--precision auto --resume",
    ]
    refused = _run("train.sh", "fermi-tellus-1", "--resume", "--fresh-data", tmp_path=tmp_path)
    assert refused.returncode != 0 and "--resume can't rebuild" in refused.stderr


def test_train_precision_and_no_bench(tmp_path: Path) -> None:
    result = _run("train.sh", "fermi-tellus-1", "--precision", "fp32", "--no-bench", "--no-eval",
                  tmp_path=tmp_path)  # fmt: skip
    assert result.returncode == 0, result.stderr
    commands = _commands(result)
    assert not any(c.startswith("askphysics model bench") for c in commands)
    train = next(c for c in commands if c.startswith("askphysics model train "))
    assert train.endswith("--batch-size 32 --backend torch --precision fp32")


def test_train_grad_accum_is_passed_only_when_set(tmp_path: Path) -> None:
    flags = ["--no-bench", "--no-eval", "--no-prose"]
    result = _run("train.sh", "fermi-celeste-1", "--grad-accum", "2", *flags, tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    train = next(c for c in _commands(result) if c.startswith("askphysics model train "))
    assert "--grad-accum 2" in train
    assert train.index("--grad-accum 2") < train.index("--precision")
    plain = _run("train.sh", "fermi-celeste-1", *flags, tmp_path=tmp_path)
    train = next(c for c in _commands(plain) if c.startswith("askphysics model train "))
    assert "--grad-accum" not in train
    for bad in ("0", "-1", "two"):
        refused = _run("train.sh", "fermi-celeste-1", "--grad-accum", bad, *flags,
                       tmp_path=tmp_path)  # fmt: skip
        assert (
            refused.returncode != 0 and "--grad-accum must be a positive integer" in refused.stderr
        )


def test_train_rejects_unknown_precision(tmp_path: Path) -> None:
    result = _run("train.sh", "fermi-tellus-1", "--precision", "fp16", tmp_path=tmp_path)
    assert result.returncode != 0 and "precision" in result.stderr


def test_train_rejects_unknown_models_and_options(tmp_path: Path) -> None:
    assert _run("train.sh", "fermi-gpt-9", tmp_path=tmp_path).returncode != 0
    assert _run("train.sh", "fermi-solem-1", "--stepz", "3", tmp_path=tmp_path).returncode != 0


def test_backup_and_restore_round_trip(tmp_path: Path) -> None:
    model = tmp_path / "models" / "fermi-solem-1"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_text("good weights")
    assert _run("models.sh", "backup", "fermi-solem-1", "good", tmp_path=tmp_path,
                dry=False).returncode == 0  # fmt: skip
    (model / "model.safetensors").write_text("worse weights")
    result = _run("models.sh", "restore", "fermi-solem-1", "good", tmp_path=tmp_path, dry=False)
    assert result.returncode == 0, result.stderr
    assert (model / "model.safetensors").read_text() == "good weights"
    # What the restore replaced was backed up first.
    kept = [p for p in (tmp_path / "backups").iterdir() if p.name != "good"]
    assert [(p / "model.safetensors").read_text() for p in kept] == ["worse weights"]
    again = _run("models.sh", "backup", "fermi-solem-1", "good", tmp_path=tmp_path, dry=False)
    assert again.returncode != 0 and "not overwriting" in again.stderr


def test_restore_without_a_backup_fails(tmp_path: Path) -> None:
    result = _run("models.sh", "restore", "fermi-solem-1", tmp_path=tmp_path, dry=False)
    assert result.returncode != 0 and "no backup" in result.stderr


def test_check_runs_every_gate(tmp_path: Path) -> None:
    full = _commands(_run("check.sh", tmp_path=tmp_path))
    assert full == [
        "python -m ruff check .",
        "python -m ruff format --check .",
        "python -m mypy",
        "askphysics validate-data",
        "python -m pytest -q",
    ]
    fast = _commands(_run("check.sh", "--fast", "--conflicts", "origin/x", tmp_path=tmp_path))
    assert "python -m pytest -q" not in fast
    assert "git merge --no-commit --no-ff -q origin/x" in fast
    assert fast[-1] == "git merge --abort"


def test_eval_against_compares_on_the_same_questions(tmp_path: Path) -> None:
    result = _run("eval.sh", "fermi-solem-1", "--against", "/old", "--examples", "200",
                  tmp_path=tmp_path)  # fmt: skip
    commands = _commands(result)
    assert commands[0].startswith("askphysics model eval --model fermi-solem-1 --directory /old")
    assert all("--examples 200" in c for c in commands[:2])
    assert commands[-1].endswith(f"/old/eval.json {tmp_path}/models/fermi-solem-1/eval.json")


def test_release_packages_each_installed_model(tmp_path: Path) -> None:
    for name in ("fermi-tellus-1", "fermi-solem-1"):
        (tmp_path / "models" / name).mkdir(parents=True)
    result = _run("release_weights.sh", "models-v0.4.0", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    packaged = [c for c in _commands(result) if c.startswith("askphysics model package")]
    assert [c.split("--model ")[1].split(" ")[0] for c in packaged] == [
        "fermi-tellus-1",
        "fermi-solem-1",
    ]
    assert all("--release models-v0.4.0" in c for c in packaged)
    assert "build/release/assets/" in result.stdout
    nothing = tmp_path / "empty"
    nothing.mkdir()
    assert _run("release_weights.sh", "r", tmp_path=nothing).returncode != 0


def test_compare_evals_shows_changes_and_fixed_questions() -> None:
    old = {"valid_plan_rate": 0.989, "confidently_wrong_rate": 0.004, "plan_examples": 3000,
           "confidently_wrong": [{"question": "clay"}, {"question": "go-kart"}]}  # fmt: skip
    new = {"valid_plan_rate": 0.999, "confidently_wrong_rate": 0.001, "plan_examples": 3000,
           "confidently_wrong": [{"question": "boulders"}]}  # fmt: skip
    text = compare_evals.compare(old, new)
    assert "98.9%" in text and "99.9%" in text and "+1.0% better" in text
    assert "confidently wrong" in text and "-0.3% better" in text
    assert "No longer confidently wrong:\n  clay\n  go-kart" in text
    assert "Newly confidently wrong:\n  boulders" in text


def test_compare_evals_needs_both_reports(tmp_path: Path) -> None:
    report = tmp_path / "eval.json"
    report.write_text(json.dumps({}))
    assert compare_evals.main([str(report), str(tmp_path / "missing.json")]) == 1
    assert compare_evals.main([str(report)]) == 2


def test_train_backend_mlx_skips_the_torch_bench(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    result = _run("train.sh", "fermi-solem-1", "--backend", "mlx", "--device", "cpu",
                  "--checkpoint-blocks", "--no-eval", "--no-prose", tmp_path=tmp_path)  # fmt: skip
    assert result.returncode == 0, result.stderr
    commands = _commands(result)
    assert not any(c.startswith("askphysics model bench") for c in commands)
    train = next(c for c in commands if c.startswith("askphysics model train "))
    assert "--backend mlx" in train and "--checkpoint-blocks" in train
    assert "--device cpu" in train and train.index("--checkpoint-blocks") < train.index(
        "--precision"
    )
    assert "Skipping the torch speed check" in result.stdout


def test_train_checkpoint_blocks_needs_mlx(tmp_path: Path) -> None:
    refused = _run("train.sh", "fermi-tellus-1", "--backend", "torch", "--checkpoint-blocks",
                   "--no-bench", "--no-eval", tmp_path=tmp_path)  # fmt: skip
    assert refused.returncode != 0 and "needs the mlx backend" in refused.stderr
    unknown = _run("train.sh", "fermi-tellus-1", "--backend", "jax", tmp_path=tmp_path)
    assert unknown.returncode != 0 and "unknown --backend jax" in unknown.stderr
