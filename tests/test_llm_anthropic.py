import pytest

from askphysics.errors import ConfigError
from askphysics.llm.anthropic_client import DEFAULT_MODEL, AnthropicClient
from askphysics.llm.base import LLMClient
from askphysics.models import Classification


def test_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_reads_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    client = AnthropicClient()
    assert client.api_key == "test-key"
    assert client.model == DEFAULT_MODEL
    assert isinstance(client, LLMClient)


def test_methods_are_stubs() -> None:
    client = AnthropicClient(api_key="test-key")
    with pytest.raises(NotImplementedError):
        client.complete_json(system="s", user="{}", schema=Classification)
    with pytest.raises(NotImplementedError):
        client.complete_text(system="s", user="{}")
