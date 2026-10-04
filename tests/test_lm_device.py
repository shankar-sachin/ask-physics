import pytest
import torch

from askphysics.errors import ConfigError
from askphysics.lm.device import select_device


def test_cpu_always_available() -> None:
    assert select_device("cpu") == torch.device("cpu")


def test_auto_selection_returns_a_device() -> None:
    assert select_device().type in {"mps", "cuda", "cpu"}


def test_unknown_device_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown device"):
        select_device("tpu")


def test_unavailable_device_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ConfigError, match="not available"):
        select_device("cuda")
