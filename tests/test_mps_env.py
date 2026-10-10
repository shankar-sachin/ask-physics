import os
import subprocess
import sys

import pytest

from askphysics.mps_env import (
    DEFAULT_HIGH,
    DEFAULT_LOW,
    HIGH_RATIO,
    LOW_RATIO,
    set_mps_watermark_defaults,
)


def test_defaults_fill_an_empty_environment() -> None:
    env: dict[str, str] = {}
    set_mps_watermark_defaults(env)
    assert env == {HIGH_RATIO: "0.7", LOW_RATIO: "0.5"}
    assert float(env[HIGH_RATIO]) == DEFAULT_HIGH
    assert float(env[LOW_RATIO]) == DEFAULT_LOW


def test_user_values_are_never_overridden() -> None:
    env = {HIGH_RATIO: "0.9", LOW_RATIO: "0.2"}
    set_mps_watermark_defaults(env)
    assert env == {HIGH_RATIO: "0.9", LOW_RATIO: "0.2"}


def test_a_low_high_ratio_pulls_the_low_ratio_down() -> None:
    env = {HIGH_RATIO: "0.3"}
    set_mps_watermark_defaults(env)
    assert env == {HIGH_RATIO: "0.3", LOW_RATIO: "0.3"}


def test_a_high_ratio_above_the_default_keeps_the_default_low() -> None:
    env = {HIGH_RATIO: "0.9"}
    set_mps_watermark_defaults(env)
    assert env[LOW_RATIO] == "0.5"


def test_an_unparsable_high_ratio_is_left_for_torch_to_reject() -> None:
    env = {HIGH_RATIO: "lots"}
    set_mps_watermark_defaults(env)
    assert env == {HIGH_RATIO: "lots", LOW_RATIO: "0.5"}


def test_low_ratio_alone_is_kept_even_above_the_default_high() -> None:
    env = {LOW_RATIO: "0.8"}
    set_mps_watermark_defaults(env)
    assert env == {HIGH_RATIO: "0.7", LOW_RATIO: "0.8"}


def test_importing_the_package_sets_the_defaults_before_torch() -> None:
    # A fresh interpreter, so this process's environment and imports don't matter. The package
    # import comes first, and torch sees the ratios the moment it is imported.
    env = {k: v for k, v in os.environ.items() if k not in (HIGH_RATIO, LOW_RATIO)}
    code = (
        "import os, askphysics\n"
        "assert 'torch' not in __import__('sys').modules\n"
        "import torch  # noqa: F401\n"
        f"print(os.environ[{HIGH_RATIO!r}], os.environ[{LOW_RATIO!r}])\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert done.stdout.split() == ["0.7", "0.5"]


@pytest.mark.parametrize("name", [HIGH_RATIO, LOW_RATIO])
def test_importing_the_package_keeps_a_value_the_user_set(name: str) -> None:
    env = {**os.environ, name: "0.42"}
    code = f"import os, askphysics; print(os.environ[{name!r}])"
    done = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert done.stdout.strip() == "0.42"
