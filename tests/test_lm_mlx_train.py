import json
from pathlib import Path

import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import numpy as np
import torch

from askphysics.errors import ConfigError
from askphysics.lm import train as torch_train
from askphysics.lm.checkpoints import load_model
from askphysics.lm.config import LUNA, ModelConfig
from askphysics.lm.factory import build_dataset
from askphysics.lm.mlx_model import FermiLM, export_params, import_params
from askphysics.lm.mlx_train import (
    EVAL_DTYPE,
    AdamW,
    _evaluate,
    _make_step,
    mean_gradients,
    mlx_dtype,
    optimizer_step,
    system_memory_gb,
    train_mlx,
)
from askphysics.lm.model import IGNORE_INDEX
from askphysics.lm.model import FermiLM as TorchFermiLM
from askphysics.lm.tokenizer import Tokenizer
from askphysics.lm.train import (
    STATE_FILE,
    TokenizedSet,
    TrainConfig,
    _optimizer_step,
    make_batch,
    numpy_batch,
    train,
    train_tokenizer,
)

TINY = ModelConfig(name="t", vocab_size=64, d_model=32, n_layers=2, n_heads=4, context_length=32)

# Small on purpose: mlx's CPU backend takes about a second per training step at these widths.
FAST = TrainConfig(
    steps=20,
    batch_size=4,
    lr=3e-3,
    warmup_steps=4,
    eval_every=10,
    eval_batches=1,
    checkpoint_every=1000,
    log_every=5,
    device="cpu",
)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("data")
    build_dataset(root, 150, seed=0)
    return root


@pytest.fixture(scope="module")
def tokenizer(dataset: Path) -> Tokenizer:
    return train_tokenizer(dataset, vocab_size=LUNA.vocab_size)


def _random_micro(
    rng: np.random.Generator, rows: int, width: int, vocab: int
) -> tuple[np.ndarray, np.ndarray]:
    """Inputs and labels for one micro-batch, with prompt positions ignored in the labels."""
    inputs = rng.integers(0, vocab, (rows, width))
    labels = rng.integers(0, vocab, (rows, width))
    labels[:, : width // 3] = IGNORE_INDEX
    return inputs, labels


def _as_mx(pair: tuple[np.ndarray, np.ndarray]) -> tuple[mx.array, mx.array]:
    return mx.array(pair[0].astype(np.int32)), mx.array(pair[1].astype(np.int32))


def _as_torch(pair: tuple[np.ndarray, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.from_numpy(pair[0].astype(np.int64)), torch.from_numpy(pair[1].astype(np.int64))


def _mlx_copy_of(torch_model: TorchFermiLM) -> FermiLM:
    model = FermiLM(torch_model.config)
    import_params(model, {k: v.detach().numpy() for k, v in torch_model.state_dict().items()})
    return model


def _params_close(model: FermiLM, other: FermiLM, atol: float) -> bool:
    a, b = export_params(model), export_params(other)
    return all(np.allclose(a[name], b[name], atol=atol) for name in a)


def test_one_step_matches_the_torch_trainer() -> None:
    # Same weights, same two micro-batches, three steps: AdamW with its decay mask, bias
    # correction, and gradient clipping must land on the same parameters as torch's.
    torch.manual_seed(0)
    torch_model = TorchFermiLM(TINY)
    mlx_model = _mlx_copy_of(torch_model)
    cfg = TrainConfig(lr=1e-2, weight_decay=0.1, grad_clip=0.5)  # clipping is active here
    torch_opt = torch_train._optimizer(torch_model, cfg)
    adam = AdamW(mlx_model, cfg.weight_decay)
    rng = np.random.default_rng(0)
    cpu = torch.device("cpu")
    for _ in range(3):
        micro = [_random_micro(rng, 2, 12, TINY.vocab_size) for _ in range(2)]
        expected = _optimizer_step(
            torch_model, torch_opt, [_as_torch(m) for m in micro], cfg, cpu, None
        )
        got = optimizer_step(
            mlx_model, adam, [_as_mx(m) for m in micro], mx.array(cfg.lr), cfg, mx.float32
        )
        assert float(got) == pytest.approx(float(expected), abs=1e-5)
    for name, value in export_params(mlx_model).items():
        assert np.allclose(value, torch_model.state_dict()[name].numpy(), atol=1e-4), name


def test_grad_accum_of_two_micro_batches_matches_one_batch_of_four() -> None:
    # The accumulated gradient must be the gradient of the big batch. Compared on the gradients
    # themselves: Adam's update divides by sqrt(v), which amplifies float noise on the
    # near-zero gradients, so parameters after a step are a poor place to look.
    gen = np.random.default_rng(1)
    a, b = _random_micro(gen, 2, 10, TINY.vocab_size), _random_micro(gen, 2, 10, TINY.vocab_size)
    torch.manual_seed(0)
    model = _mlx_copy_of(TorchFermiLM(TINY))
    big = (np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]))
    loss_big, grads_big = mean_gradients(model, [_as_mx(big)], mx.float32, False)
    loss_acc, grads_acc = mean_gradients(model, [_as_mx(a), _as_mx(b)], mx.float32, False)
    assert float(loss_big) == pytest.approx(float(loss_acc), abs=1e-6)
    assert grads_big.keys() == grads_acc.keys()
    for name in grads_big:
        assert np.allclose(np.array(grads_big[name]), np.array(grads_acc[name]), atol=1e-7), name


def test_block_checkpointing_gives_the_same_step() -> None:
    # Recomputing the blocks in the backward pass must not change the update.
    torch.manual_seed(0)
    base = _mlx_copy_of(TorchFermiLM(TINY))
    plain, checkpointed = FermiLM(TINY), FermiLM(TINY)
    import_params(plain, export_params(base))
    import_params(checkpointed, export_params(base))
    micro = [_as_mx(_random_micro(np.random.default_rng(2), 2, 12, TINY.vocab_size))]
    lr = mx.array(1e-2)
    loss_plain = optimizer_step(
        plain, AdamW(plain, 0.1), micro, lr, TrainConfig(lr=1e-2), mx.float32
    )
    loss_ckpt = optimizer_step(
        checkpointed,
        AdamW(checkpointed, 0.1),
        micro,
        lr,
        TrainConfig(lr=1e-2, checkpoint_blocks=True),
        mx.float32,
    )
    assert float(loss_plain) == pytest.approx(float(loss_ckpt), abs=1e-6)
    assert _params_close(plain, checkpointed, atol=1e-6)


def test_evaluation_matches_the_torch_trainers_loss() -> None:
    # The torch trainer scores validation in fp32 (its autocast covers training steps only).
    torch.manual_seed(0)
    torch_model = TorchFermiLM(TINY)
    mlx_model = _mlx_copy_of(torch_model)
    gen = np.random.default_rng(4)
    data = TokenizedSet()
    for _ in range(6):
        data.add(gen.integers(0, TINY.vocab_size, 20).tolist(), 6, "t")
    cfg = TrainConfig(batch_size=4, eval_batches=2)
    expected = torch_train.evaluate(torch_model, data, cfg, 0, torch.device("cpu"))
    # Within 1e-5: bf16 evaluation misses by about 4e-5 on this data, so this would catch it.
    assert _evaluate(mlx_model, data, cfg, 0, TINY.context_length, EVAL_DTYPE) == pytest.approx(
        expected, abs=1e-5
    )


def test_compiled_step_matches_the_eager_step() -> None:
    torch.manual_seed(0)
    base = _mlx_copy_of(TorchFermiLM(TINY))
    eager, compiled = FermiLM(TINY), FermiLM(TINY)
    import_params(eager, export_params(base))
    import_params(compiled, export_params(base))
    cfg = TrainConfig(lr=1e-2)
    adam_e, adam_c = AdamW(eager, 0.1), AdamW(compiled, 0.1)
    step_e = _make_step(eager, adam_e, cfg, mx.float32, compile_step=False)
    step_c = _make_step(compiled, adam_c, cfg, mx.float32, compile_step=True)
    rng = np.random.default_rng(3)
    for _ in range(3):
        micro = [_as_mx(_random_micro(rng, 2, 12, TINY.vocab_size))]
        loss_e = step_e(mx.array(cfg.lr), micro)
        loss_c = step_c(mx.array(cfg.lr), micro)
        mx.eval(loss_e, loss_c)
        assert float(loss_e) == pytest.approx(float(loss_c), abs=1e-5)
    assert _params_close(eager, compiled, atol=1e-5)


def test_bf16_forward_trains_with_finite_loss(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    cfg = TrainConfig(
        **{**FAST.__dict__, "steps": 3, "log_every": 3, "eval_every": 3, "precision": "bf16"}
    )
    metrics = train_mlx(LUNA, tokenizer, dataset, tmp_path / "bf16", cfg)
    logged = [m for m in metrics if "loss" in m]
    assert logged and all(np.isfinite(m["loss"]) for m in logged)


def test_mlx_training_lowers_loss_and_writes_the_torch_metric_keys(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "luna"
    metrics = train_mlx(LUNA, tokenizer, dataset, out, FAST)
    losses = [m["loss"] for m in metrics if "loss" in m]
    assert sum(losses[-2:]) / 2 < losses[0] * 0.95
    evals = [m for m in metrics if "val_loss" in m]
    assert evals and {"val_loss_classify", "val_loss_plan", "val_loss_explain"} <= evals[-1].keys()
    # The same metric keys as the torch trainer.
    torch_keys = {"step", "loss", "lr", "target_tokens_per_s", "mem_gb"}
    logged = [m for m in metrics if "loss" in m]
    assert all(torch_keys == set(m) for m in logged)
    lines = [json.loads(x) for x in (out / "metrics.jsonl").read_text().splitlines()]
    assert lines == metrics
    summary = json.loads((out / "training_summary.json").read_text())
    assert summary["backend"] == "mlx" and summary["config"] == "fermi-luna-1"
    assert summary["device"] == "cpu"  # torch names the Metal GPU "mps"; MLX on CPU is "cpu"
    assert json.loads((out / STATE_FILE).read_text())["backend"] == "mlx"


def test_saved_model_loads_in_torch_with_matching_logits(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "luna"
    train_mlx(LUNA, tokenizer, dataset, out, TrainConfig(**{**FAST.__dict__, "steps": 4}))
    torch_model, loaded_tok = load_model(out)
    assert loaded_tok.merges == tokenizer.merges
    assert torch_model.num_parameters() == LUNA.num_parameters()
    mlx_model = FermiLM(LUNA)
    from safetensors.numpy import load_file

    import_params(mlx_model, load_file(str(out / "model.safetensors")))
    ids = np.random.default_rng(5).integers(0, LUNA.vocab_size, (2, 14))
    with torch.no_grad():
        expected, _ = torch_model(torch.from_numpy(ids))
    got = np.array(mlx_model(mx.array(ids.astype(np.int32))))
    assert np.allclose(got, expected.numpy(), atol=1e-4)


def test_resume_continues_from_the_checkpoint(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "luna"
    train_mlx(LUNA, tokenizer, dataset, out, TrainConfig(**{**FAST.__dict__, "steps": 6}))
    assert json.loads((out / STATE_FILE).read_text())["step"] == 6
    more = TrainConfig(**{**FAST.__dict__, "steps": 12})
    metrics = train_mlx(LUNA, tokenizer, dataset, out, more, resume=True)
    assert min(m["step"] for m in metrics) > 6
    assert json.loads((out / STATE_FILE).read_text())["step"] == 12


def test_resume_refuses_a_torch_checkpoint(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "torch-run"
    train(LUNA, tokenizer, dataset, out, TrainConfig(**{**FAST.__dict__, "steps": 4}))
    with pytest.raises(ConfigError, match="torch backend"):
        train_mlx(LUNA, tokenizer, dataset, out, FAST, resume=True)


def test_resume_refuses_an_mlx_checkpoint(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    out = tmp_path / "mlx-run"
    train_mlx(LUNA, tokenizer, dataset, out, TrainConfig(**{**FAST.__dict__, "steps": 4}))
    with pytest.raises(ConfigError, match="mlx backend"):
        train(LUNA, tokenizer, dataset, out, FAST, resume=True)


def test_memory_limits_are_set_for_the_run_and_restored(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    before = mx.get_memory_limit()
    cfg = TrainConfig(**{**FAST.__dict__, "steps": 2, "log_every": 2, "eval_every": 2})
    train_mlx(
        LUNA, tokenizer, dataset, tmp_path / "limits", cfg, memory_limit_gb=8.0, cache_limit_gb=1.0
    )
    assert mx.get_memory_limit() == before
    assert system_memory_gb() > 1.0


def test_precision_names_map_to_compute_dtypes() -> None:
    assert mlx_dtype("fp32", on_gpu=True) == mx.float32
    assert mlx_dtype("bf16", on_gpu=False) == mx.bfloat16
    # auto follows the device, as torch's does: bf16 on the GPU, fp32 on the CPU.
    assert mlx_dtype("auto", on_gpu=True) == mx.bfloat16
    assert mlx_dtype("auto", on_gpu=False) == mx.float32
    with pytest.raises(ValueError, match="unknown precision"):
        mlx_dtype("fp16", on_gpu=True)


def test_cuda_is_refused_by_the_mlx_trainer(
    dataset: Path, tokenizer: Tokenizer, tmp_path: Path
) -> None:
    with pytest.raises(ConfigError, match="cuda"):
        train_mlx(LUNA, tokenizer, dataset, tmp_path / "x", TrainConfig(device="cuda"))


def test_batches_match_the_torch_batch_builder() -> None:
    # The MLX path builds numpy batches; they must be the torch path's batches exactly.
    rows = [([1, 2, 3, 4, 5], 2), ([6, 7], 1)]
    torch_inputs, torch_labels = make_batch(rows, 0, torch.device("cpu"), multiple=8)
    inputs, labels = numpy_batch(rows, 0, multiple=8)
    assert np.array_equal(inputs, torch_inputs.numpy())
    assert np.array_equal(labels, torch_labels.numpy())
