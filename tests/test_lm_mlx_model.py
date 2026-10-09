import pytest

pytest.importorskip("mlx.core")

import mlx.core as mx
import numpy as np
import torch
from torch.nn import functional

from askphysics.errors import ConfigError
from askphysics.lm.config import LUNA, ModelConfig
from askphysics.lm.mlx_model import (
    FermiLM,
    cross_entropy,
    export_params,
    import_params,
)
from askphysics.lm.model import IGNORE_INDEX
from askphysics.lm.model import FermiLM as TorchFermiLM

TINY = ModelConfig(name="t", vocab_size=64, d_model=32, n_layers=2, n_heads=4, context_length=32)


def _pair(config: ModelConfig) -> tuple[TorchFermiLM, FermiLM]:
    """A torch model and an MLX model holding the same weights."""
    torch.manual_seed(0)
    torch_model = TorchFermiLM(config).eval()
    mlx_model = FermiLM(config)
    weights = {k: v.detach().numpy() for k, v in torch_model.state_dict().items()}
    import_params(mlx_model, weights)
    return torch_model, mlx_model


@pytest.mark.parametrize("config", [TINY, LUNA], ids=["tiny", "luna"])
def test_parameter_names_and_shapes_match_torch(config: ModelConfig) -> None:
    torch_shapes = {k: tuple(v.shape) for k, v in TorchFermiLM(config).state_dict().items()}
    mlx_model = FermiLM(config)
    mlx_shapes = {k: v.shape for k, v in export_params(mlx_model).items()}
    assert mlx_shapes == torch_shapes
    assert mlx_model.num_parameters() == config.num_parameters()
    assert sum(int(np.prod(s)) for s in mlx_shapes.values()) == config.num_parameters()


def test_export_and_import_round_trip() -> None:
    a = FermiLM(TINY)
    b = FermiLM(TINY)
    import_params(b, export_params(a))
    for name, value in export_params(a).items():
        assert np.array_equal(export_params(b)[name], value)


def test_import_rejects_mismatched_weights() -> None:
    weights = export_params(FermiLM(TINY))
    weights.pop("norm.weight")
    with pytest.raises(ConfigError, match=r"norm\.weight"):
        import_params(FermiLM(TINY), weights)
    weights = export_params(FermiLM(TINY))
    weights["embed.weight"] = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ConfigError, match=r"embed\.weight"):
        import_params(FermiLM(TINY), weights)


def test_logits_match_torch_with_the_same_weights() -> None:
    torch_model, mlx_model = _pair(LUNA)
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, LUNA.vocab_size, (3, 17), generator=gen)
    with torch.no_grad():
        expected, _ = torch_model(ids)
    got = np.array(mlx_model(mx.array(ids.numpy())))
    assert got.shape == tuple(expected.shape)
    assert np.allclose(got, expected.numpy(), atol=1e-4)


def test_loss_matches_torch_and_ignores_prompt_tokens() -> None:
    torch_model, mlx_model = _pair(TINY)
    gen = torch.Generator().manual_seed(2)
    ids = torch.randint(0, TINY.vocab_size, (2, 12), generator=gen)
    targets = ids.clone()
    targets[:, :5] = IGNORE_INDEX
    targets[1, 9:] = IGNORE_INDEX
    with torch.no_grad():
        _, expected = torch_model(ids, targets)
    assert expected is not None
    got = float(mlx_model.loss(mx.array(ids.numpy()), mx.array(targets.numpy())))
    assert got == pytest.approx(float(expected), abs=1e-4)


def test_cross_entropy_matches_torch_mean_over_targets() -> None:
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(4, 6, 10, generator=gen)
    targets = torch.randint(0, 10, (4, 6), generator=gen)
    targets[0, :3] = IGNORE_INDEX
    expected = functional.cross_entropy(
        logits.reshape(-1, 10), targets.reshape(-1), ignore_index=-100
    )
    got = cross_entropy(mx.array(logits.numpy()), mx.array(targets.numpy()))
    assert float(got) == pytest.approx(float(expected), abs=1e-5)


def test_attention_is_causal() -> None:
    model = FermiLM(TINY)
    ids = mx.array(np.random.default_rng(0).integers(0, TINY.vocab_size, (1, 12)))
    changed = np.array(ids)
    changed[0, 8:] = (changed[0, 8:] + 1) % TINY.vocab_size
    a = np.array(model(ids))
    b = np.array(model(mx.array(changed)))
    assert np.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not np.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_bf16_forward_stays_close_to_fp32() -> None:
    _, model = _pair(LUNA)
    ids = mx.array(np.random.default_rng(4).integers(0, LUNA.vocab_size, (2, 16)))
    targets = mx.array(np.random.default_rng(5).integers(0, LUNA.vocab_size, (2, 16)))
    full = float(model.loss(ids, targets))
    half = float(model.loss(ids, targets, mx.bfloat16))
    assert np.isfinite(half) and abs(half - full) < 0.05
    assert model(ids, mx.bfloat16).dtype == mx.bfloat16
    assert model(ids).dtype == mx.float32


def test_context_length_is_enforced() -> None:
    model = FermiLM(TINY)
    with pytest.raises(ValueError, match="exceeds context"):
        model(mx.zeros((1, TINY.context_length + 1), dtype=mx.int32))
