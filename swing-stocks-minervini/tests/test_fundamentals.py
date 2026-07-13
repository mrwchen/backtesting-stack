import numpy as np
import pandas as pd

from src.fundamentals import quarterly_flags, sponsorship_flags
from tests.util import make_cfg


def _grid():
    return pd.bdate_range("2023-01-02", "2025-12-31"), pd.Index(["AAA"])


def _events(rows):
    frame = pd.DataFrame(rows)
    frame["available_date"] = pd.to_datetime(frame["available_date"])
    frame["fiscal_period_end_date"] = pd.to_datetime(frame["fiscal_period_end_date"])
    return frame


def test_quarterly_growth_acceleration_margin_and_streak_are_point_in_time():
    dates, symbols = _grid()
    events = _events(
        [
            {
                "symbol": "AAA", "available_date": "2024-05-02",
                "fiscal_period_end_date": "2024-03-31", "diluted_eps": 1.20,
                "prior_year_diluted_eps": 1.00, "quarterly_revenue": 110.0,
                "prior_year_quarterly_revenue": 100.0,
                "quarterly_operating_margin": 0.12,
                "prior_year_quarterly_operating_margin": 0.10,
                "quarterly_net_margin": 0.08, "prior_year_quarterly_net_margin": 0.07,
            },
            {
                "symbol": "AAA", "available_date": "2024-08-02",
                "fiscal_period_end_date": "2024-06-30", "diluted_eps": 1.50,
                "prior_year_diluted_eps": 1.10, "quarterly_revenue": 130.0,
                "prior_year_quarterly_revenue": 110.0,
                "quarterly_operating_margin": 0.14,
                "prior_year_quarterly_operating_margin": 0.11,
                "quarterly_net_margin": 0.09, "prior_year_quarterly_net_margin": 0.075,
            },
        ]
    )
    result = quarterly_flags(
        events, dates, symbols,
        make_cfg(fundamentals_min_pass=4, quarterly_growth_streak_min=2),
    )

    assert not result["eps_pass"].loc[:"2024-05-01", "AAA"].any()
    assert result["fundamental_score"].loc[:"2024-05-01", "AAA"].isna().all()
    assert bool(result["eps_pass"].loc["2024-08-02", "AAA"])
    assert bool(result["revenue_pass"].loc["2024-08-02", "AAA"])
    assert bool(result["margin_pass"].loc["2024-08-02", "AAA"])
    assert bool(result["acceleration_pass"].loc["2024-08-02", "AAA"])
    assert result["growth_streak"].loc["2024-08-02", "AAA"] == 2
    assert bool(result["fundamentals_pass"].loc["2024-08-02", "AAA"])


def test_null_quarterly_event_resets_known_growth_state():
    dates, symbols = _grid()
    base = {
        "symbol": "AAA", "fiscal_period_end_date": "2024-03-31",
        "quarterly_revenue": 120.0, "prior_year_quarterly_revenue": 100.0,
        "quarterly_operating_margin": 0.12,
        "prior_year_quarterly_operating_margin": 0.10,
        "quarterly_net_margin": np.nan, "prior_year_quarterly_net_margin": np.nan,
    }
    events = _events(
        [
            {**base, "available_date": "2024-05-02", "diluted_eps": 1.3, "prior_year_diluted_eps": 1.0},
            {**base, "available_date": "2024-08-02", "fiscal_period_end_date": "2024-06-30",
             "diluted_eps": np.nan, "prior_year_diluted_eps": 1.0},
        ]
    )
    result = quarterly_flags(events, dates, symbols, make_cfg())
    assert bool(result["eps_pass"].loc["2024-08-01", "AAA"])
    assert not bool(result["eps_pass"].loc["2024-08-02", "AAA"])
    assert pd.isna(result["eps_yoy"].loc["2024-08-02", "AAA"])


def test_partial_fundamentals_report_quality_and_coverage_separately():
    dates, symbols = _grid()
    events = _events(
        [
            {
                "symbol": "AAA",
                "available_date": "2024-05-02",
                "fiscal_period_end_date": "2024-03-31",
                "diluted_eps": 1.3,
                "prior_year_diluted_eps": 1.0,
                "quarterly_revenue": 120.0,
                "prior_year_quarterly_revenue": 100.0,
                "quarterly_operating_margin": 0.12,
                "prior_year_quarterly_operating_margin": 0.10,
                "quarterly_net_margin": np.nan,
                "prior_year_quarterly_net_margin": np.nan,
            }
        ]
    )

    result = quarterly_flags(events, dates, symbols, make_cfg())
    day = pd.Timestamp("2024-05-02")

    # EPS, revenue and margin are three observed passes. Acceleration, streak
    # and four-quarter stability are not yet comparable and must not become
    # three implicit failures.
    assert result["fundamental_score"].loc[day, "AAA"] == 3.0
    assert result["fundamental_coverage"].loc[day, "AAA"] == 0.5
    assert pd.isna(result["fundamental_score"].loc[:"2024-05-01", "AAA"]).all()
    assert (result["fundamental_coverage"].loc[:"2024-05-01", "AAA"] == 0.0).all()


def test_quarterly_state_expires_after_configured_sessions():
    dates, symbols = _grid()
    events = _events(
        [{
            "symbol": "AAA", "available_date": "2024-05-02",
            "fiscal_period_end_date": "2024-03-31", "diluted_eps": 1.3,
            "prior_year_diluted_eps": 1.0, "quarterly_revenue": 120.0,
            "prior_year_quarterly_revenue": 100.0,
            "quarterly_operating_margin": 0.12,
            "prior_year_quarterly_operating_margin": 0.10,
            "quarterly_net_margin": 0.08, "prior_year_quarterly_net_margin": 0.07,
        }]
    )
    result = quarterly_flags(
        events, dates, symbols,
        make_cfg(quarterly_fundamental_stale_trading_days=5),
    )
    event_index = dates.get_loc(pd.Timestamp("2024-05-02"))
    assert bool(result["eps_pass"].iloc[event_index + 5, 0])
    assert not bool(result["eps_pass"].iloc[event_index + 6, 0])
    assert pd.isna(result["fundamental_score"].iloc[event_index + 6, 0])


def test_13f_sponsorship_uses_only_effective_date_and_cumulative_manager_count():
    dates, symbols = _grid()
    events = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "available_date": pd.to_datetime(["2024-05-16", "2024-05-17", "2024-08-15"]),
            "manager_count_delta": [6, 5, -2],
            "net_activity_delta": [6, 5, -2],
        }
    )
    result = sponsorship_flags(
        events, dates, symbols,
        make_cfg(institutional_min_managers=10, institutional_net_activity_min=0),
    )
    assert not bool(result["institutional_sponsorship_pass"].loc["2024-05-16", "AAA"])
    assert bool(result["institutional_sponsorship_pass"].loc["2024-05-17", "AAA"])
    assert result["institutional_manager_count"].loc["2024-08-15", "AAA"] == 9
    assert not bool(result["institutional_sponsorship_pass"].loc["2024-08-15", "AAA"])


def test_same_quarter_amendment_does_not_increment_growth_streak():
    dates, symbols = _grid()
    common = {
        "symbol": "AAA", "diluted_eps": 1.3, "prior_year_diluted_eps": 1.0,
        "quarterly_revenue": 120.0, "prior_year_quarterly_revenue": 100.0,
        "quarterly_operating_margin": 0.12,
        "prior_year_quarterly_operating_margin": 0.10,
        "quarterly_net_margin": 0.08, "prior_year_quarterly_net_margin": 0.07,
    }
    events = _events(
        [
            {**common, "available_date": "2024-05-02", "fiscal_period_end_date": "2024-03-31"},
            {**common, "available_date": "2024-05-10", "fiscal_period_end_date": "2024-03-31",
             "diluted_eps": 1.35},
        ]
    )
    result = quarterly_flags(events, dates, symbols, make_cfg())
    assert result["growth_streak"].loc["2024-05-10", "AAA"] == 1
    assert pd.isna(result["eps_acceleration"].loc["2024-05-10", "AAA"])
