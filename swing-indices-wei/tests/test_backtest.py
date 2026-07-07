from datetime import date, timedelta

import numpy as np
import pytest

from src.backtest import run_backtest


def _days(n: int, start: date = date(2022, 1, 3)) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def test_always_long_equals_buy_and_hold_without_costs():
    days = _days(100)
    closes = 100.0 * np.cumprod(1 + np.sin(np.arange(100)) * 0.01)
    pos = np.ones(100, dtype=np.int8)
    r = run_backtest(days, closes, pos, days[0], cost_bps_per_side=0.0)
    assert r.total_return_pct == pytest.approx(r.bh_return_pct, abs=1e-9)
    assert r.days_invested_pct == 100.0
    assert len(r.trades) == 1 and r.trades[0].is_open


def test_always_flat_keeps_equity_at_one():
    days = _days(50)
    closes = np.linspace(100, 50, 50)
    pos = np.zeros(50, dtype=np.int8)
    r = run_backtest(days, closes, pos, days[0], cost_bps_per_side=5.0)
    assert r.total_return_pct == pytest.approx(0.0)
    assert r.max_drawdown_pct == pytest.approx(0.0)
    assert r.trades == []


def test_flat_period_skips_market_loss():
    days = _days(30)
    closes = np.concatenate([np.full(10, 100.0), np.linspace(100, 80, 10), np.full(10, 80.0)])
    pos = np.concatenate([np.ones(9), np.zeros(11), np.ones(10)]).astype(np.int8)
    r = run_backtest(days, closes, pos, days[0], cost_bps_per_side=0.0)
    # Strategy sat out the -20% leg entirely.
    assert r.total_return_pct == pytest.approx(0.0, abs=1e-6)
    assert r.bh_return_pct == pytest.approx(-20.0, abs=1e-6)
    assert len(r.trades) == 2
    assert r.trades[0].exit_date == days[9]
    assert r.trades[1].entry_date == days[20]


def test_costs_are_charged_per_side():
    days = _days(10)
    closes = np.full(10, 100.0)
    # one round trip: buy on day 0, sell on day 5
    pos = np.concatenate([np.ones(5), np.zeros(5)]).astype(np.int8)
    r = run_backtest(days, closes, pos, days[0], cost_bps_per_side=100.0)  # 1% per side
    assert r.total_return_pct == pytest.approx(((1 - 0.01) ** 2 - 1) * 100, abs=1e-6)


def test_trade_bookkeeping():
    days = _days(20)
    closes = np.linspace(100, 119, 20)
    pos = np.concatenate([np.ones(10), np.zeros(10)]).astype(np.int8)
    r = run_backtest(days, closes, pos, days[0], cost_bps_per_side=0.0)
    assert len(r.trades) == 1
    t = r.trades[0]
    assert not t.is_open
    assert t.entry_price == pytest.approx(closes[0])
    assert t.exit_price == pytest.approx(closes[10])
    assert t.gross_return_pct == pytest.approx((closes[10] / closes[0] - 1) * 100)


def test_start_date_inside_series_is_respected():
    days = _days(60)
    closes = np.linspace(100, 160, 60)
    pos = np.ones(60, dtype=np.int8)
    r = run_backtest(days, closes, pos, days[30], cost_bps_per_side=0.0)
    assert r.days[0] == days[30]
    assert r.bh_return_pct == pytest.approx((closes[-1] / closes[30] - 1) * 100)
