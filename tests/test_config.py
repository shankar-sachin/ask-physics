import pytest

from askphysics.config import Settings
from askphysics.errors import ConfigError


def test_defaults_route_automatically() -> None:
    s = Settings()
    assert s.llm_provider == "auto"
    assert s.top_k == 5
    assert s.temperature == 0.0
    assert s.model is None
    assert (s.plan_attempts, s.escalations, s.device) == (5, 1, None)


def test_from_env_overrides() -> None:
    env = {
        "ASKPHYSICS_LLM_PROVIDER": "fake",
        "ASKPHYSICS_MODEL": "fermi-celeste-1",
        "ASKPHYSICS_TOP_K": "3",
        "ASKPHYSICS_TEMPERATURE": "0.5",
        "ASKPHYSICS_PLAN_ATTEMPTS": "3",
        "ASKPHYSICS_ESCALATIONS": "0",
        "ASKPHYSICS_DEVICE": "cpu",
    }
    s = Settings.from_env(env)
    assert (s.llm_provider, s.model, s.top_k, s.temperature) == ("fake", "fermi-celeste-1", 3, 0.5)
    assert (s.plan_attempts, s.escalations, s.device) == (3, 0, "cpu")


def test_from_env_with_nothing_set_gives_defaults() -> None:
    assert Settings.from_env({}) == Settings()


@pytest.mark.parametrize(
    "env",
    [
        {"ASKPHYSICS_TOP_K": "lots"},
        {"ASKPHYSICS_TOP_K": "0"},
        {"ASKPHYSICS_TEMPERATURE": "2"},
        {"ASKPHYSICS_LLM_PROVIDER": "skynet"},
        {"ASKPHYSICS_MODEL": "gpt-5"},
        {"ASKPHYSICS_PLAN_ATTEMPTS": "0"},
        {"ASKPHYSICS_ESCALATIONS": "-1"},
    ],
)
def test_invalid_env_raises_config_error(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        Settings.from_env(env)
