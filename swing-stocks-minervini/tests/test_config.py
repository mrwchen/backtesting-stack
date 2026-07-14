import json

import pytest

from src.config import Config


def test_combined_mode_is_an_explicit_valid_runtime_mode(monkeypatch) -> None:
    monkeypatch.setenv("SIMULATION_MODE", "both")

    assert Config.from_env().simulation_mode == "both"


def test_legacy_sensitivity_stage_is_not_part_of_v9(monkeypatch) -> None:
    monkeypatch.setenv("STAGE", "sensitivity")

    with pytest.raises(ValueError, match="unsupported STAGE"):
        Config.from_env()


def test_v9_forward_configuration_is_serialized(monkeypatch) -> None:
    monkeypatch.setenv("PORTFOLIO_RANKING_MODE", "RELATIVE_QUALITY")
    monkeypatch.setenv("PORTFOLIO_SETUP_TYPES", "FLAT_BASE")
    monkeypatch.setenv("NEUTRAL_RANK_SALT", "v9-relative-quality-shadow")

    cfg = Config.from_env()
    params = json.loads(cfg.to_json())

    assert cfg.start_date == "2020-01-02"
    assert cfg.forward_start_date == "2026-07-13"
    assert cfg.end_date is None
    assert cfg.force_close_at_end is False
    assert cfg.portfolio_ranking_mode == "relative_quality"
    assert cfg.portfolio_setup_types == ("flat_base",)
    assert cfg.neutral_rank_salt == "v9-relative-quality-shadow"
    assert params["forward_start_date"] == "2026-07-13"
    assert params["force_close_at_end"] is False
    assert params["portfolio_ranking_mode"] == "relative_quality"
    assert params["portfolio_setup_types"] == ["flat_base"]
    assert params["neutral_rank_salt"] == "v9-relative-quality-shadow"


def test_v8_matrix_flags_are_not_part_of_the_v9_contract(monkeypatch) -> None:
    monkeypatch.setenv("PORTFOLIO_RANKING_EXPERIMENT_ENABLE", "true")
    monkeypatch.setenv("PORTFOLIO_RANKING_SENSITIVITY_ENABLE", "true")

    cfg = Config.from_env()
    params = json.loads(cfg.to_json())

    assert not hasattr(cfg, "portfolio_ranking_experiment_enable")
    assert not hasattr(cfg, "portfolio_ranking_sensitivity_enable")
    assert "portfolio_ranking_experiment_enable" not in params
    assert "portfolio_ranking_sensitivity_enable" not in params


def test_v9_protocol_defaults(
    monkeypatch,
) -> None:
    monkeypatch.delenv("RUN_LABEL", raising=False)
    monkeypatch.delenv("START_DATE", raising=False)
    monkeypatch.delenv("FORWARD_START_DATE", raising=False)
    monkeypatch.delenv("END_DATE", raising=False)
    monkeypatch.delenv("FORCE_CLOSE_AT_END", raising=False)
    monkeypatch.delenv("PORTFOLIO_RANKING_MODE", raising=False)
    monkeypatch.delenv("PORTFOLIO_SETUP_TYPES", raising=False)
    monkeypatch.delenv("NEUTRAL_RANK_SALT", raising=False)

    cfg = Config.from_env()

    assert cfg.run_label == "minervini_sepa_daily_v9_forward_shadow"
    assert cfg.start_date == "2020-01-02"
    assert cfg.forward_start_date == "2026-07-13"
    assert cfg.end_date is None
    assert cfg.force_close_at_end is False
    assert cfg.portfolio_ranking_mode == "relative_quality"
    assert cfg.portfolio_setup_types == ("flat_base",)
    assert cfg.neutral_rank_salt == "v9-relative-quality-shadow"


def test_v9_has_no_fundamental_entry_gate_configuration(monkeypatch) -> None:
    monkeypatch.setenv("BAD_FUNDAMENTALS_FILTER_ENABLE", "true")

    cfg = Config.from_env()

    assert not hasattr(cfg, "bad_fundamentals_filter_enable")
    assert "bad_fundamentals_filter_enable" not in json.loads(cfg.to_json())


@pytest.mark.parametrize(
    "value", ["neutral", "quality_only", "validated", "fill_weighted", "unknown"]
)
def test_ranking_mode_must_be_a_v9_mode(monkeypatch, value: str) -> None:
    monkeypatch.setenv("PORTFOLIO_RANKING_MODE", value)

    with pytest.raises(ValueError, match="PORTFOLIO_RANKING_MODE"):
        Config.from_env()


@pytest.mark.parametrize("value", ["independent", "portfolio", "unknown"])
def test_external_simulation_mode_is_frozen_to_both(monkeypatch, value: str) -> None:
    monkeypatch.setenv("SIMULATION_MODE", value)

    with pytest.raises(ValueError, match="SIMULATION_MODE must be both"):
        Config.from_env()


def test_quality_shadow_salt_is_frozen(monkeypatch) -> None:
    monkeypatch.setenv("NEUTRAL_RANK_SALT", "custom-salt")

    with pytest.raises(ValueError, match="NEUTRAL_RANK_SALT is frozen"):
        Config.from_env()


@pytest.mark.parametrize(
    "value",
    [
        "vcp",
        "power_play",
        "tight_shelf",
        "flat_base,vcp",
        "flat_base,flat_base",
        ", ,",
    ],
)
def test_portfolio_setup_types_are_restricted_and_unique(
    monkeypatch, value: str
) -> None:
    monkeypatch.setenv("PORTFOLIO_SETUP_TYPES", value)

    with pytest.raises(ValueError, match="PORTFOLIO_SETUP_TYPES"):
        Config.from_env()


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("START_DATE", "2020-01-03", "START_DATE is frozen"),
        ("FORWARD_START_DATE", "2026-07-14", "FORWARD_START_DATE is frozen"),
        ("END_DATE", "2026-07-10", "END_DATE"),
        ("FORCE_CLOSE_AT_END", "true", "FORCE_CLOSE_AT_END"),
    ],
)
def test_v9_forward_boundary_is_frozen(
    monkeypatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
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
