from datetime import date

import pytest

from backtest_models.minervini import MODEL_VERSION
from src import forward_shadow


def test_forward_protocol_dates_are_frozen() -> None:
    assert MODEL_VERSION == "minervini_daily_v9"
    assert forward_shadow.HISTORY_START_DATE == date(2020, 1, 2)
    assert forward_shadow.FORWARD_START_DATE == date(2026, 7, 13)


def test_neutral_control_salts_are_frozen_unique_and_complete() -> None:
    salts = forward_shadow.NEUTRAL_CONTROL_SALTS

    assert len(salts) == 32
    assert len(set(salts)) == 32
    assert salts[0] == "v9-neutral-control-00"
    assert salts[-1] == "v9-neutral-control-31"
    assert forward_shadow.RELATIVE_QUALITY_SHADOW_SALT not in salts


def test_portfolio_cases_are_one_shadow_and_32_flat_base_controls() -> None:
    cases = forward_shadow.PORTFOLIO_CASES

    assert len(cases) == 33
    assert cases[0] is forward_shadow.RELATIVE_QUALITY_SHADOW_CASE
    assert cases[0].name == "relative_quality_flat_base_shadow"
    assert cases[0].role == "shadow"
    assert cases[0].ranking_mode == "relative_quality"
    assert cases[0].neutral_rank_salt == "v9-relative-quality-shadow"
    assert cases[1:] == forward_shadow.NEUTRAL_CONTROL_CASES
    assert {case.role for case in cases[1:]} == {"control"}
    assert {case.ranking_mode for case in cases[1:]} == {"neutral"}
    assert {case.setup_types for case in cases} == {("flat_base",)}
    assert [case.neutral_rank_salt for case in cases[1:]] == list(
        forward_shadow.NEUTRAL_CONTROL_SALTS
    )
    assert len({case.name for case in cases}) == 33


def test_control_summary_reports_median_adverse_decile_and_worst() -> None:
    results = [
        {
            "total_return": float(value),
            "cagr": float(value) / 10,
            "max_drawdown": float(value) / 100,
            "profit_factor": None if value == 0 else float(value),
            "avg_r_multiple": -float(value),
        }
        for value in range(32)
    ]

    summary = forward_shadow.summarize_controls(results)

    assert summary["total_return"] == {
        "count": 32,
        "missing_count": 0,
        "median": 15.5,
        "adverse_quantile": pytest.approx(3.1),
        "worst": 0.0,
    }
    assert summary["max_drawdown"]["adverse_quantile"] == pytest.approx(0.279)
    assert summary["max_drawdown"]["worst"] == pytest.approx(0.31)
    assert summary["avg_r_multiple"]["worst"] == -31.0
    assert summary["profit_factor"]["count"] == 31
    assert summary["profit_factor"]["missing_count"] == 1


def test_control_summary_requires_the_complete_frozen_panel() -> None:
    with pytest.raises(ValueError, match="expected 32"):
        forward_shadow.summarize_controls([])
