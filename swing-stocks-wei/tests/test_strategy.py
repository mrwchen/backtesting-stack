from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import compute_signals
from tests.util import make_config


def _prices(close_values, volume_values, feed="sip", market_cap_usd=3_000_000_000.0):
    dates = pd.bdate_range("2024-01-01", periods=len(close_values))
    high = np.asarray(close_values, dtype=float) * 1.01
    high[31] = 130.0
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "open": np.asarray(close_values, dtype=float),
            "high": high,
            "low": np.asarray(close_values, dtype=float) * 0.99,
            "close": np.asarray(close_values, dtype=float),
            "volume": np.asarray(volume_values, dtype=float),
            "alpaca_price_feed": feed,
            "market_cap_usd": market_cap_usd,
            "market_cap_currency": "USD",
        }
    )


def _multi_industry_prices():
    dates = pd.bdate_range("2023-01-02", periods=218)
    aaa_close = [100.0] * 210 + [99, 98, 97, 99, 102, 104, 106, 108]
    base = pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "open": aaa_close,
            "high": np.asarray(aaa_close, dtype=float) * 1.01,
            "low": np.asarray(aaa_close, dtype=float) * 0.99,
            "close": aaa_close,
            "volume": [1000.0] * 218,
            "alpaca_price_feed": "sip",
            "market_cap_usd": 3_000_000_000.0,
            "market_cap_currency": "USD",
        }
    )
    base.loc[214, "volume"] = 5000.0
    weak_members = []
    for symbol in ("AAB", "AAC"):
        close = [100.0] * 210 + [90.0] * 8
        weak_members.append(
            pd.DataFrame(
                {
                    "symbol": symbol,
                    "date": dates,
                    "open": close,
                    "high": np.asarray(close, dtype=float) * 1.01,
                    "low": np.asarray(close, dtype=float) * 0.99,
                    "close": close,
                    "volume": [1000.0] * 218,
                    "alpaca_price_feed": "sip",
                    "market_cap_usd": 3_000_000_000.0,
                    "market_cap_currency": "USD",
                }
            )
        )
    return pd.concat([base, *weak_members], ignore_index=True)


def test_signal_uses_recent_52w_high_ema_cross_and_volume():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    cfg = make_config()
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume),
        universe,
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["symbol"] == "AAA"
    assert signal["had_52w_high_last_10d"]
    assert signal["volume_sma50_pass"]
    assert signal["volume_feed_pass"]
    assert signal["volume_pass"]
    assert signal["market_cap_pass"]
    assert signal["planned_entry_market_cap_usd"] >= 2_000_000_000
    assert signal["planned_entry_date"] > signal["period_end_date"]


def test_signal_requires_volume_above_sma():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    cfg = make_config()
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume),
        universe,
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_can_ignore_volume_filter_when_disabled():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    cfg = make_config(volume_filter_enable=False)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume),
        universe,
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert len(signals) == 1
    assert not signals.iloc[0]["volume_sma50_pass"]
    assert signals.iloc[0]["volume_feed_pass"]
    assert signals.iloc[0]["volume_pass"]


def test_signal_requires_sip_volume_feed_even_when_volume_filter_disabled():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    cfg = make_config(volume_filter_enable=False)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume, feed="iex"),
        universe,
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_requires_min_market_cap_on_planned_entry_date():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    prices = _prices(close, volume)
    signal_idx = 35
    prices.loc[signal_idx + 1, "market_cap_usd"] = 1_999_999_999.0
    cfg = make_config()
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        prices,
        universe,
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_requires_industry_breadth_gate_when_enabled():
    cfg = make_config(
        ibkr_industry_breadth_filter_enable=True,
        ibkr_industry_breadth_min_symbols=3,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "AAB", "AAC"],
            "ibkr_industry": ["Software", "Software", "Software"],
            "ibkr_category": ["Application", "Application", "Application"],
        }
    )

    signals = compute_signals(
        _multi_industry_prices(),
        universe,
        cfg,
        pd.Timestamp("2023-01-02").date(),
        pd.Timestamp("2023-12-31").date(),
    )

    assert signals.empty
