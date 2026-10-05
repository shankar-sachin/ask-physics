import pytest

from askphysics.errors import ConfigError
from askphysics.llm.routing import CELESTE, SOLEM, TELLUS, Route, plan_route


def test_full_install_splits_and_escalates() -> None:
    route = plan_route([TELLUS, SOLEM, CELESTE])
    assert route == Route(classify=TELLUS, plan=(SOLEM,) * 5 + (CELESTE,), explain=SOLEM)
    assert route.models == [TELLUS, SOLEM, CELESTE]


def test_tellus_alone_runs_everything() -> None:
    assert plan_route([TELLUS]) == Route(classify=TELLUS, plan=(TELLUS,) * 5, explain=TELLUS)


def test_missing_models_are_skipped() -> None:
    route = plan_route([TELLUS, CELESTE], attempts=2)
    assert route == Route(classify=TELLUS, plan=(TELLUS, TELLUS, CELESTE), explain=TELLUS)
    route = plan_route([SOLEM])
    assert route == Route(classify=SOLEM, plan=(SOLEM,) * 5, explain=SOLEM)
    route = plan_route([CELESTE], escalations=3)  # celeste never escalates to itself
    assert route == Route(classify=CELESTE, plan=(CELESTE,) * 5, explain=CELESTE)


def test_attempts_and_escalations_are_settings() -> None:
    route = plan_route([TELLUS, SOLEM, CELESTE], attempts=1, escalations=0)
    assert route.plan == (SOLEM,)


def test_forced_model_handles_every_stage() -> None:
    route = plan_route([TELLUS, SOLEM, "fermi-luna-1"], forced="fermi-luna-1", attempts=2)
    assert route == Route(
        classify="fermi-luna-1", plan=("fermi-luna-1",) * 2, explain="fermi-luna-1"
    )


def test_luna_is_never_picked_unless_forced() -> None:
    with pytest.raises(ConfigError, match="no Fermi models"):
        plan_route(["fermi-luna-1"])
    with pytest.raises(ConfigError, match="not installed"):
        plan_route([TELLUS], forced=SOLEM)
