import json

import pytest

from src.config import Config


def test_combined_mode_is_an_explicit_valid_runtime_mode(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")

    assert Config.from_env().simulation_mode == "both"


def test_portfolio_ranking_sensitivity_and_salt_are_serialized(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")
    monkeypatch.setenv("PORTFOLIO_RANKING_SENSITIVITY_ENABLE", "true")
    monkeypatch.setenv("NEUTRAL_RANK_SALT", "fixed-test-salt")

    cfg = Config.from_env()
    params = json.loads(cfg.to_json())

    assert cfg.portfolio_ranking_sensitivity_enable is True
    assert cfg.neutral_rank_salt == "fixed-test-salt"
    assert params["portfolio_ranking_sensitivity_enable"] is True
    assert params["neutral_rank_salt"] == "fixed-test-salt"


def test_portfolio_ranking_sensitivity_rejects_independent_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "independent")
    monkeypatch.setenv("PORTFOLIO_RANKING_SENSITIVITY_ENABLE", "true")

    with pytest.raises(ValueError, match="requires SIMULATION_MODE"):
        Config.from_env()


def test_default_run_label_identifies_v7_class_local_32salt(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RUN_LABEL", raising=False)

    assert (
        Config.from_env().run_label
        == "minervini_sepa_daily_v7_class_local_32salt"
    )


def test_v7_has_no_fundamental_entry_gate_configuration(monkeypatch) -> None:
    monkeypatch.setenv("BAD_FUNDAMENTALS_FILTER_ENABLE", "true")

    cfg = Config.from_env()

    assert not hasattr(cfg, "bad_fundamentals_filter_enable")
    assert "bad_fundamentals_filter_enable" not in json.loads(cfg.to_json())


def test_default_slate_risk_floor_is_serialized_in_run_params(monkeypatch) -> None:
    monkeypatch.delenv("MIN_SLATE_RISK_UTILIZATION", raising=False)

    cfg = Config.from_env()

    assert cfg.min_slate_risk_utilization == 0.50
    assert json.loads(cfg.to_json())["min_slate_risk_utilization"] == 0.50


def test_default_daily_order_limit_is_serialized_in_run_params(monkeypatch) -> None:
    monkeypatch.delenv("PORTFOLIO_MAX_DAILY_ORDERS", raising=False)

    cfg = Config.from_env()

    assert cfg.portfolio_max_daily_orders == 3
    assert json.loads(cfg.to_json())["portfolio_max_daily_orders"] == 3


@pytest.mark.parametrize("value", ["0", "-0.1", "1.01"])
def test_slate_risk_floor_must_be_a_fraction(monkeypatch, value: str) -> None:
    monkeypatch.setenv("MIN_SLATE_RISK_UTILIZATION", value)

    with pytest.raises(ValueError, match="MIN_SLATE_RISK_UTILIZATION"):
        Config.from_env()


@pytest.mark.parametrize("value", ["0", "-1", "4"])
def test_daily_order_limit_must_be_between_one_and_three(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("PORTFOLIO_MAX_DAILY_ORDERS", value)

    with pytest.raises(ValueError, match="PORTFOLIO_MAX_DAILY_ORDERS"):
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
