from __future__ import annotations

import pandas as pd

from src.simulator import simulate
from tests.util import make_config


def test_simulator_enters_next_open_and_stops_out():
    dates = pd.bdate_range("2024-01-01", periods=5)
    prices = pd.DataFrame(
        {
            "symbol": ["AAA"] * 5,
            "date": dates,
            "open": [100, 100, 100, 94, 94],
            "high": [101, 102, 103, 95, 95],
            "low": [99, 99, 99, 93, 93],
            "close": [100, 101, 100, 94, 94],
            "volume": [1000] * 5,
        }
    )
    signals = pd.DataFrame(
        {
            "period_end_date": [dates[1].date()],
            "symbol": ["AAA"],
            "ibkr_industry": ["Software"],
            "ibkr_category": ["Application"],
        }
    )
    cfg = make_config(high_lookback_days=3)

    result = simulate(prices, signals, cfg, dates[0].date(), dates[-1].date())

    assert len(result.trades) == 1
    trade = result.trades.iloc[0]
    assert trade["entry_date"] == dates[2].date()
    assert trade["entry_price"] == 100.0
    assert trade["exit_reason"] == "stop_loss"
    assert trade["exit_price"] == 94.0
    assert trade["pnl"] == -60.0


def test_trailing_stop_activates_after_ten_percent_gain():
    dates = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame(
        {
            "symbol": ["AAA"] * 6,
            "date": dates,
            "open": [100, 100, 100, 111, 112, 108],
            "high": [101, 102, 111, 118, 113, 109],
            "low": [99, 99, 100, 110, 107, 106],
            "close": [100, 101, 110, 117, 108, 107],
            "volume": [1000] * 6,
        }
    )
    signals = pd.DataFrame(
        {
            "period_end_date": [dates[1].date()],
            "symbol": ["AAA"],
            "ibkr_industry": ["Software"],
            "ibkr_category": ["Application"],
        }
    )
    cfg = make_config(high_lookback_days=3)

    result = simulate(prices, signals, cfg, dates[0].date(), dates[-1].date())

    trade = result.trades.iloc[0]
    assert trade["exit_reason"] == "trailing_stop"
    assert trade["exit_price"] == 112.0
    assert trade["pnl"] == 120.0
