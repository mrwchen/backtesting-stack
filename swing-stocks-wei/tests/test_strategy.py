from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy import compute_signals
from tests.util import make_config


def _fundamentals(prices, current_revenue=125_000_000.0, prior_revenue=100_000_000.0):
    rows = []
    for symbol, sub in prices.groupby("symbol"):
        current_date = pd.Timestamp(sub["date"].min())
        rows.extend(
            [
                {
                    "symbol": symbol,
                    "available_date": current_date - pd.Timedelta(days=365),
                    "revenue_ttm": prior_revenue,
                },
                {
                    "symbol": symbol,
                    "available_date": current_date,
                    "revenue_ttm": current_revenue,
                },
            ]
        )
    return pd.DataFrame(rows)


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


def _multi_category_prices():
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


def _already_above_at_52w_high_prices():
    dates = pd.bdate_range("2024-01-01", periods=36)
    close = [100.0] * 30 + [101, 102, 110, 111, 112, 113]
    high = [120.0] * 29 + [119.0, 119.0, 119.0, 130.0, 112.5, 113.5, 114.5]
    volume = [1000.0] * 36
    volume[32] = 5000.0
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "date": dates,
            "open": close,
            "high": high,
            "low": np.asarray(close, dtype=float) * 0.99,
            "close": close,
            "volume": volume,
            "alpaca_price_feed": "sip",
            "market_cap_usd": 3_000_000_000.0,
            "market_cap_currency": "USD",
        }
    )


def test_signal_uses_recent_52w_high_ema_cross_and_volume():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    cfg = make_config(max_entry_gap_pct=0.10)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume),
        universe,
        _fundamentals(_prices(close, volume)),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["symbol"] == "AAA"
    assert signal["had_52w_high_recent"]
    assert signal["pullback_pass"]
    assert signal["ema_cross_up"]
    assert not signal["ema_already_above_on_52w_high"]
    assert signal["volume_sma50_pass"]
    assert signal["volume_feed_pass"]
    assert signal["volume_pass"]
    assert signal["market_cap_pass"]
    assert signal["entry_ref_market_cap_usd"] >= 2_000_000_000
    assert signal["entry_gap_pass"]
    assert signal["planned_entry_date"] > signal["period_end_date"]


def test_signal_delays_entry_after_recent_ema_cross():
    close = [100.0] * 30 + [
        92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110, 112, 114, 116, 118, 120
    ]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    prices = _prices(close, volume)
    cfg = make_config(ema_cross_lookback_days=8)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        prices,
        universe,
        _fundamentals(prices),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert len(signals) == 1
    signal = signals.iloc[0]
    assert signal["ema_cross_up"]
    assert signal["ema_cross_recent"]
    assert signal["ema_cross_delay_days"] == 8
    assert signal["planned_entry_date"] == prices.loc[44, "date"].date()


def test_signal_rejects_52w_high_when_ema_already_above_without_reclaim_cross():
    prices = _already_above_at_52w_high_prices()
    cfg = make_config()
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )
    target_date = prices.loc[32, "date"].date()

    signals = compute_signals(
        prices,
        universe,
        _fundamentals(prices),
        cfg,
        target_date,
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


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
        _fundamentals(_prices(close, volume)),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_can_ignore_volume_filter_when_disabled():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    cfg = make_config(volume_filter_enable=False, max_entry_gap_pct=0.10)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        _prices(close, volume),
        universe,
        _fundamentals(_prices(close, volume)),
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
        _fundamentals(_prices(close, volume, feed="iex")),
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
    prices.loc[signal_idx, "market_cap_usd"] = 1_999_999_999.0
    cfg = make_config()
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        prices,
        universe,
        _fundamentals(prices),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_rejects_large_entry_gap_up():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    prices = _prices(close, volume)
    prices.loc[36, "open"] = prices.loc[35, "close"] * 1.05
    cfg = make_config(max_entry_gap_pct=0.02)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        prices,
        universe,
        _fundamentals(prices),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_requires_revenue_growth_above_threshold():
    close = [100.0] * 30 + [92, 90, 88, 89, 91, 94, 98, 103, 106, 108, 110]
    volume = [1000.0] * len(close)
    volume[35] = 5000.0
    prices = _prices(close, volume)
    cfg = make_config(revenue_yoy_min=0.20)
    universe = pd.DataFrame(
        {"symbol": ["AAA"], "ibkr_industry": ["Software"], "ibkr_category": ["Application"]}
    )

    signals = compute_signals(
        prices,
        universe,
        _fundamentals(prices, current_revenue=119_000_000.0, prior_revenue=100_000_000.0),
        cfg,
        pd.Timestamp("2024-01-01").date(),
        pd.Timestamp("2024-12-31").date(),
    )

    assert signals.empty


def test_signal_requires_category_breadth_gate_when_enabled():
    cfg = make_config(
        ibkr_category_breadth_filter_enable=True,
        ibkr_category_breadth_min_symbols=3,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "AAB", "AAC"],
            "ibkr_industry": ["Software", "Software", "Software"],
            "ibkr_category": ["Application", "Application", "Application"],
        }
    )

    signals = compute_signals(
        _multi_category_prices(),
        universe,
        _fundamentals(_multi_category_prices()),
        cfg,
        pd.Timestamp("2023-01-02").date(),
        pd.Timestamp("2023-12-31").date(),
    )

    assert signals.empty
