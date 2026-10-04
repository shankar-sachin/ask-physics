import pytest

from askphysics.lm.config import MAGNETAR, NANO, PRESETS, PULSAR, QUASAR, ModelConfig, get_config


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (PULSAR, 3_217_440),
        (QUASAR, 29_893_120),
        (MAGNETAR, 119_563_008),
        (NANO, 139_584),
    ],
)
def test_parameter_counts_are_pinned(config: ModelConfig, expected: int) -> None:
    # docs/MODELS.md promises ~3.2M / ~30M / ~120M; changing a preset must be deliberate.
    assert config.num_parameters() == expected


def test_family_shares_a_vocabulary() -> None:
    assert PULSAR.vocab_size == QUASAR.vocab_size == MAGNETAR.vocab_size == 8192


def test_family_names() -> None:
    assert set(PRESETS) == {"fermi-pulsar-1", "fermi-quasar-1", "fermi-magnetar-1", "fermi-nano"}


def test_mlp_hidden_is_about_eight_thirds() -> None:
    for config in PRESETS.values():
        assert config.mlp_hidden % config.mlp_multiple_of == 0
        assert 8 * config.d_model / 3 <= config.mlp_hidden < 8 * config.d_model / 3 + 64


def test_get_config() -> None:
    assert get_config("fermi-quasar-1") is QUASAR
    with pytest.raises(KeyError, match="presets"):
        get_config("fermi-blackhole-1")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 100, "n_heads": 3},  # not divisible
        {"d_model": 96, "n_heads": 32},  # odd head_dim
        {"d_model": 64, "n_heads": 4, "n_layers": 0},
    ],
)
def test_invalid_shapes_rejected(kwargs: dict[str, int]) -> None:
    base = {
        "name": "bad",
        "vocab_size": 300,
        "d_model": 64,
        "n_layers": 1,
        "n_heads": 4,
        "context_length": 16,
    }
    with pytest.raises(ValueError):
        ModelConfig(**{**base, **kwargs})  # type: ignore[arg-type]
