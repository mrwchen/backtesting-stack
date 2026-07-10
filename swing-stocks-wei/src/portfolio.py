"""Portfolio simulation: turn per-stock signals into one equity curve.

Mechanics (all fills at the close of the signal day):
- Exits first: a held stock whose signal flipped to flat is sold at today's
  close (only on days the stock actually traded).
- Trimming second: a position that grew past trim_above_pct of equity is sold
  down to trim_target_pct (0 = disabled). Without it a single 5% entry can
  compound into a position that dominates equity and drawdown.
- Entries third: the per-stock signal flipped flat->long confirm_days trading
  days ago, stayed long since, AND the stress gate is OFF today (no new
  entries while the market stress light is red). Candidates are ranked by
  category momentum ascending (most beaten-down first), ties broken by
  per-stock momentum ascending; on mass-entry days after the gate opens whole
  categories share one momentum value, so without this tie-breaker admission
  degenerated to alphabetical order. Admitted while the total and per-category
  position limits allow.
- Position size: WEIGHT_<tier>_PCT of current portfolio equity at entry,
  capped by available cash. No rebalancing afterwards, no leverage, no shorts.
- Costs: cost_bps_per_side on every buy and sell notional (also on trims).
- Benchmark: equal-weight daily-rebalanced index of the same universe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .strategy import entry_candidates, sizing_tier


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
    target_weight_pct: float
    effective_weight_pct: float
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


def _benchmark(closes: np.ndarray, symbols: list[str], days: list[date]) -> np.ndarray:
    """Equal-weight daily-rebalanced index over the (ffilled) universe closes."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rets = closes[1:] / closes[:-1] - 1.0
    invalid = np.argwhere((rets > 9.0) | (rets < -0.9))
    if invalid.size:
        day_index, symbol_index = invalid[0]
        raise RuntimeError(
            "implausible benchmark return after continuity validation: "
            f"{symbols[int(symbol_index)]} {days[int(day_index) + 1]} "
            f"{rets[int(day_index), int(symbol_index)] * 100:+.2f}%"
        )
    mean_rets = np.nanmean(rets, axis=1)
    mean_rets = np.nan_to_num(mean_rets, nan=0.0)
    return np.concatenate([[1.0], np.cumprod(1.0 + mean_rets)])


def run_portfolio(days: list[date], symbols: list[str], categories: dict[str, str],
                  closes: np.ndarray, fresh: np.ndarray, positions: np.ndarray,
                  stress_on: np.ndarray, cat_momentum: dict[str, np.ndarray],
                  weight_pct_by_tier: dict[str, float], deep_threshold: float,
                  max_positions: int, max_per_category: int,
                  cost_bps_per_side: float, stock_mom: np.ndarray | None = None,
                  entry_confirm_days: int = 0, trim_above_pct: float = 0.0,
                  trim_target_pct: float = 0.0, sl_pct: float = 0.0,
                  time_stop_days: int = 0,
                  time_stop_min_ret_pct: float = 0.0,
                  reentry_cooldown_days: int = 0) -> PortfolioResult:
    """Simulate the portfolio over the evaluation window.

    Array shapes: closes/fresh/positions are (n_days, n_symbols); closes are
    forward-filled (NaN only before a stock's first bar); fresh marks days the
    stock actually traded. stress_on is (n_days,). cat_momentum maps category
    to a (n_days,) array (NaN where unknown). stock_mom is the per-stock
    momentum matrix used as entry tie-breaker (None = alphabetical fallback).

    Stop overlays (both evaluated on the close, 0 = disabled): sl_pct exits a
    position sl_pct% below entry (catastrophe stop, not a timing stop);
    time_stop_days exits a position that is still at or below
    time_stop_min_ret_pct% after that many trading days (dead-money recycling).
    A symbol stopped by the catastrophe stop is locked until its signal resets
    to flat, so the still-long signal cannot re-enter immediately. Time-stopped
    symbols behave the same by default; with reentry_cooldown_days > 0 they are
    only blocked for that many trading days and become entry candidates again
    while their signal is still long (recycled capital goes back to work
    instead of waiting for the next flat->long flip). A signal reset clears the
    cooldown, so re-entries after a reset go through the normal
    flip-plus-confirmation path.
    """
    n_days, n_sym = closes.shape
    cost = cost_bps_per_side / 1e4
    cand = entry_candidates(positions, entry_confirm_days)
    cash = 1.0
    held: dict[int, _Holding] = {}
    entry_row: dict[int, int] = {}
    locked = np.zeros(n_sym, dtype=bool)
    cooldown = np.full(n_sym, -1, dtype=int)  # re-eligible from this row on; -1 = none
    trades: list[StockTrade] = []
    equity = np.empty(n_days)
    n_pos = np.empty(n_days, dtype=int)
    exposure = np.empty(n_days)

    def mark_to_market(t: int) -> float:
        return cash + sum(h.shares * closes[t, s] for s, h in held.items())

    def sell_position(s: int, t: int) -> None:
        nonlocal cash
        h = held.pop(s)
        entry_row.pop(s, None)
        price = closes[t, s]
        cash += h.shares * price * (1.0 - cost)
        tr = h.trade
        tr.exit_date = days[t]
        tr.exit_price = float(price)
        tr.gross_return_pct = float((price / tr.entry_price - 1.0) * 100)
        tr.holding_days = (days[t] - tr.entry_date).days

    for t in range(n_days):
        # a stopped symbol stays blocked until its signal has reset to flat
        locked &= positions[t] != 0
        cooldown[positions[t] == 0] = -1

        # 1) signal exits at today's close
        for s in [s for s, h in held.items()
                  if fresh[t, s] and positions[t, s] == 0]:
            sell_position(s, t)

        # 1b) stop overlays
        if sl_pct > 0 or time_stop_days > 0:
            for s in list(held):
                if not fresh[t, s]:
                    continue
                ret = closes[t, s] / held[s].trade.entry_price - 1.0
                if sl_pct > 0 and ret <= -sl_pct / 100.0:
                    sell_position(s, t)
                    locked[s] = True
                elif (time_stop_days > 0
                        and t - entry_row[s] >= time_stop_days
                        and ret <= time_stop_min_ret_pct / 100.0):
                    sell_position(s, t)
                    if reentry_cooldown_days > 0:
                        cooldown[s] = t + reentry_cooldown_days
                    else:
                        locked[s] = True

        # 2) trim positions that outgrew their risk budget
        if trim_above_pct > 0:
            equity_now = mark_to_market(t)
            for s, h in held.items():
                if not fresh[t, s]:
                    continue
                value = h.shares * closes[t, s]
                if value > trim_above_pct / 100.0 * equity_now:
                    sell_value = value - trim_target_pct / 100.0 * equity_now
                    h.shares -= sell_value / closes[t, s]
                    cash += sell_value * (1.0 - cost)

        # 3) entries at today's close (blocked while the stress light is red)
        if not stress_on[t]:
            flipped = [
                s for s in range(n_sym)
                if s not in held and not locked[s] and fresh[t, s]
                and (cand[t, s] or 0 <= cooldown[s] <= t)
            ]
            mom = {s: float(cat_momentum[categories[symbols[s]]][t])
                   for s in flipped}
            flipped.sort(key=lambda s: (
                np.nan_to_num(mom[s], nan=0.0),
                np.nan_to_num(stock_mom[t, s], nan=0.0) if stock_mom is not None else 0.0,
                symbols[s]))
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
                    target_weight_pct=weight * 100.0,
                    effective_weight_pct=float(budget / equity_now * 100.0) if equity_now > 0 else 0.0,
                    tier=tier,
                    cat_mom_at_entry=None if np.isnan(mom[s]) else mom[s],
                    is_open=False,
                )
                trades.append(trade)
                held[s] = _Holding(shares=shares, trade=trade)
                entry_row[s] = t
                cooldown[s] = -1
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

    bh_equity = _benchmark(closes, symbols, days)
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
