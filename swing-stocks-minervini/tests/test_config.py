import json

import pytest

from src.config import Config


def test_combined_mode_is_an_explicit_valid_runtime_mode(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")

    assert Config.from_env().simulation_mode == "both"


def test_default_slate_risk_floor_is_serialized_in_run_params(monkeypatch) -> None:
    monkeypatch.delenv("MIN_SLATE_RISK_UTILIZATION", raising=False)

    cfg = Config.from_env()

    assert cfg.min_slate_risk_utilization == 0.50
    assert json.loads(cfg.to_json())["min_slate_risk_utilization"] == 0.50


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01"])
def test_slate_risk_floor_must_be_a_fraction(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MIN_SLATE_RISK_UTILIZATION", value)

    with pytest.raises(ValueError, match="MIN_SLATE_RISK_UTILIZATION"):
        Config.from_env()


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("EXPOSURE_WINNERS_TO_STEP_UP", "EXPOSURE_WINNERS_TO_STEP_UP"),
        ("EXPOSURE_LOSSES_TO_RESET", "EXPOSURE_LOSSES_TO_RESET"),
    ],
)
def test_feedback_thresholds_must_be_positive(
    monkeypatch, name: str, message: str
) -> None:
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValueError, match=message):
        Config.from_env()
