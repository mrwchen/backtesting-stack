"""Turn a daily position series into equity curves, trades and metrics.

Conventions:
- position[t] is decided at close t and earns the close-to-close return t -> t+1.
- Transaction costs: cost_bps_per_side is charged on every position change
  (0->1 buy, 1->0 sell), applied multiplicatively to the equity curve.
- Trades are reported gross (price-to-price); the equity curve is net of costs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np


@dataclass
class Trade:
    trade_no: int
    entry_date: date
    exit_date: date | None
    entry_price: float
    exit_price: float | None
    gross_return_pct: float | None
    holding_days: int | None
    is_open: bool


@dataclass
class BacktestResult:
    days: list[date]
    equity: np.ndarray
    bh_equity: np.ndarray
    trades: list[Trade] = field(default_factory=list)
    total_return_pct: float = 0.0
    bh_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bh_max_drawdown_pct: float = 0.0
    cagr_pct: float = 0.0
    days_invested_pct: float = 0.0


def _max_drawdown(equity: np.ndarray) -> float:
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def run_backtest(days: list[date], closes: np.ndarray, position: np.ndarray,
                 start_date: date, cost_bps_per_side: float) -> BacktestResult:
    """Evaluate from the first trading day >= start_date to the last bar."""
    start = next(i for i, d in enumerate(days) if d >= start_date)
    if start >= len(days) - 1:
        raise ValueError("evaluation window too short")

    ev_days = days[start:]
    c = closes[start:]
    pos = position[start:].astype(float)
    rets = c[1:] / c[:-1] - 1.0

    # daily equity: position held from close t to close t+1, costs on flips
    step = 1.0 + pos[:-1] * rets
    turnover = np.abs(np.diff(np.concatenate([[0.0], pos])))[:-1]
    step = step * (1.0 - cost_bps_per_side / 1e4 * turnover)
    equity = np.concatenate([[1.0], np.cumprod(step)])
    bh_equity = c / c[0]

    trades: list[Trade] = []
    entry_idx: int | None = None
    for t in range(len(pos)):
        if entry_idx is None and pos[t] == 1:
            entry_idx = t
        elif entry_idx is not None and pos[t] == 0:
            trades.append(_close_trade(len(trades) + 1, ev_days, c, entry_idx, t))
            entry_idx = None
    if entry_idx is not None:
        trades.append(_close_trade(len(trades) + 1, ev_days, c, entry_idx, len(pos) - 1,
                                   is_open=True))

    years = max((ev_days[-1] - ev_days[0]).days / 365.25, 1e-9)
    result = BacktestResult(
        days=ev_days,
        equity=equity,
        bh_equity=bh_equity,
        trades=trades,
        total_return_pct=float((equity[-1] - 1.0) * 100),
        bh_return_pct=float((bh_equity[-1] - 1.0) * 100),
        max_drawdown_pct=_max_drawdown(equity) * 100,
        bh_max_drawdown_pct=_max_drawdown(bh_equity) * 100,
        cagr_pct=float((equity[-1] ** (1.0 / years) - 1.0) * 100),
        days_invested_pct=float(np.mean(pos[:-1] == 1) * 100),
    )
    return result


def _close_trade(trade_no: int, days: list[date], closes: np.ndarray,
                 entry: int, exit_: int, is_open: bool = False) -> Trade:
    exit_price = float(closes[exit_])
    return Trade(
        trade_no=trade_no,
        entry_date=days[entry],
        exit_date=days[exit_],
        entry_price=float(closes[entry]),
        exit_price=exit_price,
        gross_return_pct=float((exit_price / closes[entry] - 1.0) * 100),
        holding_days=(days[exit_] - days[entry]).days,
        is_open=is_open,
    )
