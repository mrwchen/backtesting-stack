import json

import pytest

from src.config import Config


def test_combined_mode_is_an_explicit_valid_runtime_mode(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")

    assert Config.from_env().simulation_mode == "both"


def test_portfolio_ranking_experiment_configuration_is_serialized(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")
    monkeypatch.setenv("PORTFOLIO_RANKING_EXPERIMENT_ENABLE", "true")
    monkeypatch.setenv("PORTFOLIO_RANKING_MODE", "QUALITY_ONLY")
    monkeypatch.setenv("PORTFOLIO_SETUP_TYPES", "FLAT_BASE, VCP")
    monkeypatch.setenv("NEUTRAL_RANK_SALT", "fixed-test-salt")

    cfg = Config.from_env()
    params = json.loads(cfg.to_json())

    assert cfg.portfolio_ranking_experiment_enable is True
    assert cfg.portfolio_ranking_mode == "quality_only"
    assert cfg.portfolio_setup_types == ("flat_base", "vcp")
    assert cfg.neutral_rank_salt == "fixed-test-salt"
    assert params["portfolio_ranking_experiment_enable"] is True
    assert params["portfolio_ranking_mode"] == "quality_only"
    assert params["portfolio_setup_types"] == ["flat_base", "vcp"]
    assert params["neutral_rank_salt"] == "fixed-test-salt"


def test_portfolio_ranking_experiment_rejects_independent_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "independent")
    monkeypatch.setenv("PORTFOLIO_RANKING_EXPERIMENT_ENABLE", "true")

    with pytest.raises(ValueError, match="requires SIMULATION_MODE"):
        Config.from_env()


def test_v8_ranking_defaults(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RUN_LABEL", raising=False)
    monkeypatch.delenv("PORTFOLIO_RANKING_MODE", raising=False)
    monkeypatch.delenv("PORTFOLIO_SETUP_TYPES", raising=False)
    monkeypatch.delenv("NEUTRAL_RANK_SALT", raising=False)

    cfg = Config.from_env()

    assert cfg.run_label == "minervini_sepa_daily_v8_ranking_experiment"
    assert cfg.portfolio_ranking_mode == "validated"
    assert cfg.portfolio_setup_types == ("flat_base", "vcp")
    assert cfg.neutral_rank_salt == "v8-bootstrap-00"


def test_v8_has_no_fundamental_entry_gate_configuration(monkeypatch) -> None:
    monkeypatch.setenv("BAD_FUNDAMENTALS_FILTER_ENABLE", "true")

    cfg = Config.from_env()

    assert not hasattr(cfg, "bad_fundamentals_filter_enable")
    assert "bad_fundamentals_filter_enable" not in json.loads(cfg.to_json())


def test_old_ranking_sensitivity_environment_variable_is_not_supported(
    monkeypatch,
) -> None:
    monkeypatch.setenv("PORTFOLIO_RANKING_SENSITIVITY_ENABLE", "true")

    cfg = Config.from_env()

    assert not hasattr(cfg, "portfolio_ranking_sensitivity_enable")


@pytest.mark.parametrize("value", ["fill_weighted", "unknown"])
def test_ranking_mode_must_be_a_v8_mode(monkeypatch, value: str) -> None:
    monkeypatch.setenv("PORTFOLIO_RANKING_MODE", value)

    with pytest.raises(ValueError, match="PORTFOLIO_RANKING_MODE"):
        Config.from_env()


@pytest.mark.parametrize(
    "value",
    ["power_play", "tight_shelf", "flat_base,power_play", "flat_base,flat_base"],
)
def test_portfolio_setup_types_are_restricted_and_unique(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("PORTFOLIO_SETUP_TYPES", value)

    with pytest.raises(ValueError, match="PORTFOLIO_SETUP_TYPES"):
        Config.from_env()


def test_portfolio_setup_types_must_not_be_empty(monkeypatch) -> None:
    monkeypatch.setenv("PORTFOLIO_SETUP_TYPES", ", ,")

    with pytest.raises(ValueError, match="PORTFOLIO_SETUP_TYPES"):
        Config.from_env()


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
