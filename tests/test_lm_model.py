import pytest
import torch

from askphysics.lm.config import LUNA, PRESETS, ModelConfig
from askphysics.lm.model import IGNORE_INDEX, FermiLM, RMSNorm, apply_rope, rope_tables


@pytest.fixture
def luna() -> FermiLM:
    torch.manual_seed(0)
    return FermiLM(LUNA).eval()


@pytest.mark.parametrize("name", list(PRESETS))
def test_built_parameters_match_config(name: str) -> None:
    config = PRESETS[name]
    with torch.device("meta"):  # no memory allocated, so celeste is cheap to check
        model = FermiLM(config)
    assert model.num_parameters() == config.num_parameters()


def test_output_head_is_tied(luna: FermiLM) -> None:
    names = [n for n, _ in luna.named_parameters()]
    assert not any("head" in n for n in names)
    assert sum(1 for n in names if n == "embed.weight") == 1


def test_forward_shapes_and_loss(luna: FermiLM) -> None:
    ids = torch.randint(0, LUNA.vocab_size, (3, 17))
    targets = torch.randint(0, LUNA.vocab_size, (3, 17))
    logits, loss = luna(ids, targets)
    assert logits.shape == (3, 17, LUNA.vocab_size)
    assert loss is not None and loss.ndim == 0
    # A freshly initialized model is close to uniform over the vocabulary.
    assert abs(loss.item() - torch.log(torch.tensor(float(LUNA.vocab_size))).item()) < 0.5


def test_no_loss_without_targets(luna: FermiLM) -> None:
    _, loss = luna(torch.zeros(1, 4, dtype=torch.long))
    assert loss is None


def test_ignored_targets_do_not_count(luna: FermiLM) -> None:
    ids = torch.randint(0, LUNA.vocab_size, (1, 8))
    targets = ids.clone()
    _, full = luna(ids, targets)
    targets[0, :4] = IGNORE_INDEX
    _, partial = luna(ids, targets)
    assert full is not None and partial is not None
    assert not torch.isclose(full, partial)


def test_attention_is_causal(luna: FermiLM) -> None:
    ids = torch.randint(0, LUNA.vocab_size, (1, 12))
    changed = ids.clone()
    changed[0, 8:] = (changed[0, 8:] + 1) % LUNA.vocab_size
    a, _ = luna(ids)
    b, _ = luna(changed)
    assert torch.allclose(a[0, :8], b[0, :8], atol=1e-5)
    assert not torch.allclose(a[0, 8:], b[0, 8:], atol=1e-5)


def test_kv_cache_matches_full_forward(luna: FermiLM) -> None:
    ids = torch.randint(0, LUNA.vocab_size, (2, 20))
    full, _ = luna(ids)
    logits, past = luna.step(ids[:, :12])
    steps = [logits]
    for t in range(12, 20):
        logits, past = luna.step(ids[:, t : t + 1], past)
        steps.append(logits)
    assert torch.allclose(full, torch.cat(steps, dim=1), atol=1e-5)


def test_cache_requires_single_token_steps(luna: FermiLM) -> None:
    _, past = luna.step(torch.zeros(1, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="one token"):
        luna.step(torch.zeros(1, 2, dtype=torch.long), past)


def test_context_length_enforced(luna: FermiLM) -> None:
    with pytest.raises(ValueError, match="exceeds context"):
        luna(torch.zeros(1, LUNA.context_length + 1, dtype=torch.long))


def test_rope_preserves_norm() -> None:
    cos, sin = rope_tables(head_dim=16, length=32, theta=10_000.0)
    x = torch.randn(1, 2, 32, 16)
    rotated = apply_rope(x, cos, sin)
    assert torch.allclose(x.norm(dim=-1), rotated.norm(dim=-1), atol=1e-5)
    assert torch.allclose(rotated[:, :, 0], x[:, :, 0])  # position 0 is not rotated


def test_rope_offset_matches_slice() -> None:
    cos, sin = rope_tables(head_dim=8, length=16, theta=10_000.0)
    x = torch.randn(1, 1, 16, 8)
    assert torch.allclose(apply_rope(x, cos, sin)[:, :, 5:6], apply_rope(x[:, :, 5:6], cos, sin, 5))


def test_rmsnorm_normalizes() -> None:
    norm = RMSNorm(8)
    y = norm(torch.randn(4, 8) * 50)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4), atol=1e-3)


def test_overfits_a_single_sequence() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        name="tiny", vocab_size=64, d_model=32, n_layers=1, n_heads=2, context_length=32
    )
    model = FermiLM(config)
    ids = torch.randint(0, 64, (1, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    first = None
    for _ in range(60):
        _, loss = model(ids[:, :-1], ids[:, 1:])
        assert loss is not None
        first = first if first is not None else loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.2


def _small_model(checkpoint_blocks: bool) -> FermiLM:
    config = ModelConfig(
        name="t", vocab_size=64, d_model=16, n_layers=3, n_heads=2, context_length=12
    )
    torch.manual_seed(3)
    model = FermiLM(config)
    model.checkpoint_blocks = checkpoint_blocks
    return model


def _loss_and_grads(
    model: FermiLM, ids: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    model.zero_grad(set_to_none=True)
    logits, loss = model(ids, targets)
    assert loss is not None
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    return logits.detach().clone(), grads


def test_block_checkpointing_gives_the_same_logits_and_gradients() -> None:
    gen = torch.Generator().manual_seed(4)
    ids = torch.randint(0, 64, (2, 12), generator=gen)
    targets = torch.randint(0, 64, (2, 12), generator=gen)
    plain = _small_model(checkpoint_blocks=False)
    checkpointed = _small_model(checkpoint_blocks=True)
    checkpointed.load_state_dict(plain.state_dict())
    plain.train()
    checkpointed.train()
    logits_a, grads_a = _loss_and_grads(plain, ids, targets)
    logits_b, grads_b = _loss_and_grads(checkpointed, ids, targets)
    assert torch.allclose(logits_a, logits_b, atol=1e-6)
    assert grads_a.keys() == grads_b.keys() and len(grads_a) > 0
    for name in grads_a:
        assert torch.allclose(grads_a[name], grads_b[name], atol=1e-6), name


def test_block_checkpointing_applies_only_while_training(monkeypatch: pytest.MonkeyPatch) -> None:
    from askphysics.lm import model as model_module

    calls: list[int] = []
    real = model_module.checkpoint

    def counting(*args: object, **kwargs: object) -> object:
        calls.append(1)
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(model_module, "checkpoint", counting)
    ids = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    model = _small_model(checkpoint_blocks=True)

    model.train()
    model(ids, targets)
    assert len(calls) == model.config.n_layers  # one checkpoint per block, while training

    calls.clear()
    model.eval()
    model(ids, targets)  # evaluation keeps every activation
    with torch.no_grad():
        model.train()
        model(ids, targets)  # no gradients, so nothing to recompute
    past = None
    model.eval()
    for i in range(3):
        _, past = model.step(ids[:, i : i + 1], past)  # decoding with a KV cache
    assert calls == []

    calls.clear()
    model.train()
    model.checkpoint_blocks = False
    model(ids, targets)
    assert calls == []
