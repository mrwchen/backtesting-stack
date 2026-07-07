"""Portfolio simulation: turn per-stock signals into one equity curve.

Mechanics (all fills at the close of the signal day):
- Exits first: a held stock whose signal flipped to flat is sold at today's
  close (only on days the stock actually traded).
- Entries second: signal flips flat->long AND the stress gate is OFF (no new
  entries while the market stress light is red). Candidates are ranked by
  category momentum ascending (most beaten-down first) and admitted while the
  total and per-category position limits allow.
- Position size: WEIGHT_<tier>_PCT of current portfolio equity at entry,
  capped by available cash. No rebalancing afterwards, no leverage, no shorts.
- Costs: cost_bps_per_side on every buy and sell notional.
- Benchmark: equal-weight daily-rebalanced index of the same universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .strategy import sizing_tier


@dataclass
class StockTrade:
    trade_no: int
    symbol: str
    category: str
    entry_date: date
    exit_date: date | None
    entry_price: float
    exit_price: float | None
    gross_return_pct: float | None
    holding_days: int | None
    weight_pct: float
    tier: str
    cat_mom_at_entry: float | None
    is_open: bool


@dataclass
class PortfolioResult:
    days: list[date]
    equity: np.ndarray
    bh_equity: np.ndarray
    n_positions: np.ndarray
    gross_exposure_pct: np.ndarray
    trades: list[StockTrade] = field(default_factory=list)
    total_return_pct: float = 0.0
    bh_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    bh_max_drawdown_pct: float = 0.0
    cagr_pct: float = 0.0
    avg_gross_exposure_pct: float = 0.0


@dataclass
class _Holding:
    shares: float
    trade: StockTrade


def _max_drawdown(equity: np.ndarray) -> float:
    return float((equity / np.maximum.accumulate(equity) - 1.0).min())


def _benchmark(closes: np.ndarray) -> np.ndarray:
    """Equal-weight daily-rebalanced index over the (ffilled) universe closes."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = closes[1:] / closes[:-1] - 1.0
    mean_rets = np.nanmean(rets, axis=1)
    mean_rets = np.nan_to_num(mean_rets, nan=0.0)
    return np.concatenate([[1.0], np.cumprod(1.0 + mean_rets)])


def run_portfolio(days: list[date], symbols: list[str], categories: dict[str, str],
                  closes: np.ndarray, fresh: np.ndarray, positions: np.ndarray,
                  stress_on: np.ndarray, cat_momentum: dict[str, np.ndarray],
                  weight_pct_by_tier: dict[str, float], deep_threshold: float,
                  max_positions: int, max_per_category: int,
                  cost_bps_per_side: float) -> PortfolioResult:
    """Simulate the portfolio over the evaluation window.

    Array shapes: closes/fresh/positions are (n_days, n_symbols); closes are
    forward-filled (NaN only before a stock's first bar); fresh marks days the
    stock actually traded. stress_on is (n_days,). cat_momentum maps category
    to a (n_days,) array (NaN where unknown).
    """
    n_days, n_sym = closes.shape
    cost = cost_bps_per_side / 1e4
    cash = 1.0
    held: dict[int, _Holding] = {}
    trades: list[StockTrade] = []
    equity = np.empty(n_days)
    n_pos = np.empty(n_days, dtype=int)
    exposure = np.empty(n_days)

    def mark_to_market(t: int) -> float:
        return cash + sum(h.shares * closes[t, s] for s, h in held.items())

    for t in range(n_days):
        # 1) exits at today's close
        for s in [s for s, h in held.items()
                  if fresh[t, s] and positions[t, s] == 0]:
            h = held.pop(s)
            price = closes[t, s]
            cash += h.shares * price * (1.0 - cost)
            tr = h.trade
            tr.exit_date = days[t]
            tr.exit_price = float(price)
            tr.gross_return_pct = float((price / tr.entry_price - 1.0) * 100)
            tr.holding_days = (days[t] - tr.entry_date).days

        # 2) entries at today's close (blocked while the stress light is red)
        if not stress_on[t]:
            flipped = [
                s for s in range(n_sym)
                if s not in held and fresh[t, s] and positions[t, s] == 1
                and (t == 0 or positions[t - 1, s] == 0)
            ]
            mom = {s: float(cat_momentum[categories[symbols[s]]][t])
                   for s in flipped}
            flipped.sort(key=lambda s: (np.nan_to_num(mom[s], nan=0.0), symbols[s]))
            cat_count: dict[str, int] = {}
            for h in held.values():
                cat_count[h.trade.category] = cat_count.get(h.trade.category, 0) + 1
            equity_now = mark_to_market(t)
            for s in flipped:
                if len(held) >= max_positions:
                    break
                cat = categories[symbols[s]]
                if cat_count.get(cat, 0) >= max_per_category:
                    continue
                tier = sizing_tier(mom[s], deep_threshold)
                weight = weight_pct_by_tier[tier] / 100.0
                budget = min(weight * equity_now, cash)
                if budget <= 0:
                    continue
                price = closes[t, s]
                shares = budget * (1.0 - cost) / price
                cash -= budget
                trade = StockTrade(
                    trade_no=len(trades) + 1, symbol=symbols[s], category=cat,
                    entry_date=days[t], exit_date=None,
                    entry_price=float(price), exit_price=None,
                    gross_return_pct=None, holding_days=None,
                    weight_pct=weight * 100.0, tier=tier,
                    cat_mom_at_entry=None if np.isnan(mom[s]) else mom[s],
                    is_open=False,
                )
                trades.append(trade)
                held[s] = _Holding(shares=shares, trade=trade)
                cat_count[cat] = cat_count.get(cat, 0) + 1

        equity[t] = mark_to_market(t)
        n_pos[t] = len(held)
        exposure[t] = (equity[t] - cash) / equity[t] * 100.0

    # mark remaining holdings as open trades at the last close
    for s, h in held.items():
        tr = h.trade
        price = closes[n_days - 1, s]
        tr.exit_date = days[-1]
        tr.exit_price = float(price)
        tr.gross_return_pct = float((price / tr.entry_price - 1.0) * 100)
        tr.holding_days = (days[-1] - tr.entry_date).days
        tr.is_open = True

    bh_equity = _benchmark(closes)
    years = max((days[-1] - days[0]).days / 365.25, 1e-9)
    return PortfolioResult(
        days=days,
        equity=equity,
        bh_equity=bh_equity,
        n_positions=n_pos,
        gross_exposure_pct=exposure,
        trades=trades,
        total_return_pct=float((equity[-1] - 1.0) * 100),
        bh_return_pct=float((bh_equity[-1] - 1.0) * 100),
        max_drawdown_pct=_max_drawdown(equity) * 100,
        bh_max_drawdown_pct=_max_drawdown(bh_equity) * 100,
        cagr_pct=float((equity[-1] ** (1.0 / years) - 1.0) * 100),
        avg_gross_exposure_pct=float(np.mean(exposure)),
    )
