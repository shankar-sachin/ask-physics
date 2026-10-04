import pytest

from askphysics.config import Settings
from askphysics.errors import ConfigError


def test_defaults_use_fake_llm() -> None:
    s = Settings()
    assert s.llm_provider == "fake"
    assert s.top_k == 5
    assert s.temperature == 0.0
    assert s.model == "fermi-solem-1"


def test_from_env_overrides() -> None:
    env = {
        "ASKPHYSICS_LLM_PROVIDER": "fake",
        "ASKPHYSICS_MODEL": "fermi-celeste-1",
        "ASKPHYSICS_TOP_K": "3",
        "ASKPHYSICS_TEMPERATURE": "0.5",
    }
    s = Settings.from_env(env)
    assert (s.llm_provider, s.model, s.top_k, s.temperature) == ("fake", "fermi-celeste-1", 3, 0.5)


def test_from_env_with_nothing_set_gives_defaults() -> None:
    assert Settings.from_env({}) == Settings()


@pytest.mark.parametrize(
    "env",
    [
        {"ASKPHYSICS_TOP_K": "lots"},
        {"ASKPHYSICS_TOP_K": "0"},
        {"ASKPHYSICS_TEMPERATURE": "2"},
        {"ASKPHYSICS_LLM_PROVIDER": "skynet"},
    ],
)
def test_invalid_env_raises_config_error(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)
