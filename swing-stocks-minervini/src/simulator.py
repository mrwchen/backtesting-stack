"""Event-driven daily simulation: every triggered setup is traded independently.

There is deliberately NO portfolio logic — no cash constraint, no position
limit, no exposure cap and no compounding. Every stock whose setup triggers is
taken, so the results measure the strategy itself across the whole universe.
The only per-symbol rule: while a trade in a symbol is open, no second trade
is opened in the same symbol.

Entry:  stop-buy over the pivot, from the day after detection while the setup
        is valid. Fill = max(open, pivot) plus slippage; days that gap more
        than MAX_GAP_PCT over the pivot are skipped. When the market-breadth
        gate is off (market_on[t] == False) no new entries are taken; open
        positions keep running into their regular exits.
Exits:  initial stop (gap-aware, also checked on the entry bar itself —
        breakout and stop-out on the same day is assumed to resolve against
        us), partial profit at PARTIAL_AT_R with optional break-even stop,
        trailing exit on a close below the TRAIL_MA_DAYS MA (executed next
        open), forced close at the end of the simulation.
Sizing: fixed per trade — RISK_PCT of INITIAL_EQUITY per initial stop
        distance, capped at MAX_POSITION_PCT of INITIAL_EQUITY.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Position:
    position_id: int
    setup_id: int | None
    symbol: str
    col: int
    entry_idx: int
    entry_price: float
    stop: float
    initial_stop: float
    shares: int
    initial_shares: int
    pivot: float
    partial_done: bool = False
    exit_next_open: bool = False


@dataclass
class SimResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


def simulate(
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    open_m: pd.DataFrame,
    high_m: pd.DataFrame,
    low_m: pd.DataFrame,
    close_m: pd.DataFrame,
    setups: pd.DataFrame,
    cfg: Config,
    sim_start_idx: int = 0,
    market_on: np.ndarray | None = None,
) -> SimResult:
    o = open_m.to_numpy()
    h = high_m.to_numpy()
    lo = low_m.to_numpy()
    c = close_m.to_numpy()
    c_ffill = close_m.ffill().to_numpy()
    ma_trail = close_m.rolling(cfg.trail_ma_days, min_periods=cfg.trail_ma_days).mean().to_numpy()
    col_index = {s: i for i, s in enumerate(symbols)}

    setups = setups[setups["symbol"].isin(col_index)].copy()
    setups["start_idx"] = dates.searchsorted(pd.to_datetime(setups["detect_date"])) + 1
    setups["end_idx"] = dates.searchsorted(
        pd.to_datetime(setups["valid_until"]), side="right"
    ) - 1
    setups = setups.sort_values(["start_idx", "symbol"]).reset_index(drop=True)
    setup_rows = list(setups.itertuples(index=False))

    sizing_base = cfg.initial_equity
    realized_pnl = 0.0
    positions: dict[str, Position] = {}
    active_setups: list = []
    trades: list[dict] = []
    equity_rows: list[dict] = []
    next_setup = 0
    next_position_id = 1

    def record_trade(pos: Position, t: int, shares: int, price: float, leg: str, reason: str):
        nonlocal realized_pnl
        proceeds = shares * price * (1 - cfg.commission_pct)
        cost = shares * pos.entry_price * (1 + cfg.commission_pct)
        pnl = proceeds - cost
        realized_pnl += pnl
        r_unit = pos.entry_price - pos.initial_stop
        trades.append(
            {
                "position_id": pos.position_id,
                "setup_id": pos.setup_id,
                "symbol": pos.symbol,
                "leg": leg,
                "exit_reason": reason,
                "entry_date": dates[pos.entry_idx].date(),
                "entry_price": round(pos.entry_price, 4),
                "stop_price": round(pos.initial_stop, 4),
                "pivot": round(pos.pivot, 4),
                "shares": shares,
                "exit_date": dates[t].date(),
                "exit_price": round(price, 4),
                "pnl": round(pnl, 2),
                "r_multiple": round((price - pos.entry_price) / r_unit, 4) if r_unit > 0 else None,
                "holding_days": int(t - pos.entry_idx),
            }
        )

    for t in range(sim_start_idx, len(dates)):
        # ---- exits ---------------------------------------------------------
        for sym in list(positions):
            pos = positions[sym]
            col = pos.col
            day_open, day_high, day_low = o[t, col], h[t, col], lo[t, col]
            if np.isnan(day_open):
                continue  # no bar today, hold

            if pos.exit_next_open:
                record_trade(pos, t, pos.shares, day_open * (1 - cfg.slippage_pct), "final", "trail_ma")
                del positions[sym]
                continue

            if day_open <= pos.stop:
                record_trade(pos, t, pos.shares, day_open * (1 - cfg.slippage_pct), "final", "stop_gap")
                del positions[sym]
                continue
            if day_low <= pos.stop:
                record_trade(pos, t, pos.shares, pos.stop * (1 - cfg.slippage_pct), "final", "stop")
                del positions[sym]
                continue

            r_unit = pos.entry_price - pos.initial_stop
            target = pos.entry_price + cfg.partial_at_r * r_unit
            if not pos.partial_done and r_unit > 0 and day_high >= target:
                part = int(pos.shares * cfg.partial_fraction)
                if part >= 1:
                    fill = max(day_open, target) * (1 - cfg.slippage_pct)
                    record_trade(pos, t, part, fill, "partial", "partial_target")
                    pos.shares -= part
                pos.partial_done = True
                if cfg.breakeven_after_partial:
                    pos.stop = max(pos.stop, pos.entry_price)
                if pos.shares <= 0:
                    del positions[sym]
                    continue

            trail = ma_trail[t, col]
            if not np.isnan(trail) and c[t, col] < trail:
                pos.exit_next_open = True

        # ---- collect newly active setups ------------------------------------
        while next_setup < len(setup_rows) and setup_rows[next_setup].start_idx <= t:
            active_setups.append(setup_rows[next_setup])
            next_setup += 1
        active_setups = [s for s in active_setups if s.end_idx >= t]

        # ---- entries: every triggering setup is taken (unless the market
        # breadth gate is off; setups stay active until they expire) ------------
        consumed = []
        entries_allowed = market_on is None or bool(market_on[t])
        for s in active_setups if entries_allowed else []:
            if s.symbol in positions:
                continue
            col = col_index[s.symbol]
            day_open, day_high = o[t, col], h[t, col]
            if np.isnan(day_open) or np.isnan(day_high):
                continue
            pivot = float(s.pivot)
            if day_high <= pivot:
                continue
            if day_open > pivot * (1 + cfg.max_gap_pct):
                continue  # gapped too far; setup stays valid for later days

            fill = max(day_open, pivot) * (1 + cfg.slippage_pct)
            stop = float(s.stop_level)
            risk_per_share = fill - stop
            if risk_per_share <= 0:
                consumed.append(s)
                continue

            shares = int(
                min(
                    sizing_base * cfg.risk_pct / risk_per_share,
                    sizing_base * cfg.max_position_pct / fill,
                )
            )
            if shares < 1:
                continue

            new_position = Position(
                position_id=next_position_id,
                setup_id=getattr(s, "setup_id", None),
                symbol=s.symbol,
                col=col,
                entry_idx=t,
                entry_price=fill,
                stop=stop,
                initial_stop=stop,
                shares=shares,
                initial_shares=shares,
                pivot=pivot,
            )
            next_position_id += 1
            consumed.append(s)
            # entry-day stop: if the bar's low also breaches the stop, assume
            # the breakout came first and we were stopped out the same day
            if lo[t, col] <= stop:
                record_trade(
                    new_position, t, shares, stop * (1 - cfg.slippage_pct), "final", "stop_entry_day"
                )
            else:
                positions[s.symbol] = new_position
        for s in consumed:
            active_setups.remove(s)

        # ---- daily aggregate (research curve, not a cash-constrained account) --
        open_notional = 0.0
        unrealized = 0.0
        for p in positions.values():
            price = c_ffill[t, p.col]
            if np.isnan(price):
                continue
            open_notional += p.shares * price
            unrealized += p.shares * (price - p.entry_price)
        equity_rows.append(
            {
                "period_end_date": dates[t].date(),
                "equity": round(sizing_base + realized_pnl + unrealized, 2),
                "open_positions": len(positions),
                "exposure_pct": round(open_notional / sizing_base, 6),
            }
        )

    # ---- close remaining positions at the last close --------------------------
    last = len(dates) - 1
    for sym in list(positions):
        pos = positions[sym]
        price = c_ffill[last, pos.col]
        if np.isnan(price):
            price = pos.entry_price
        record_trade(pos, last, pos.shares, price, "final", "eod")
        del positions[sym]

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    return SimResult(trades=trades_df, equity=equity_df, metrics=_metrics(trades_df, equity_df, cfg))


def _metrics(trades: pd.DataFrame, equity: pd.DataFrame, cfg: Config) -> dict:
    initial = cfg.initial_equity
    if equity.empty:
        return {"initial_equity": initial, "final_equity": initial}
    final = float(equity["equity"].iloc[-1])
    eq = equity["equity"].astype(float)
    years = len(eq) / 252.0
    drawdown = 1.0 - eq / eq.cummax()

    metrics = {
        "initial_equity": initial,
        "final_equity": round(final, 2),
        "total_return": round(final / initial - 1.0, 6),
        "cagr": round((final / initial) ** (1.0 / years) - 1.0, 6) if years > 0 and final > 0 else None,
        "max_drawdown": round(float(drawdown.max()), 6),
        "avg_exposure": round(float(equity["exposure_pct"].astype(float).mean()), 6),
        "num_trade_legs": int(len(trades)),
        "num_positions": 0,
        "win_rate": None,
        "profit_factor": None,
        "avg_r_multiple": None,
    }
    if trades.empty:
        return metrics

    per_position = trades.groupby("position_id")["pnl"].sum()
    wins = (per_position > 0).sum()
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = trades.loc[trades["pnl"] < 0, "pnl"].sum()
    metrics.update(
        {
            "num_positions": int(len(per_position)),
            "win_rate": round(float(wins / len(per_position)), 6),
            "profit_factor": round(float(gross_profit / abs(gross_loss)), 6)
            if gross_loss < 0
            else None,
            "avg_r_multiple": round(float(trades["r_multiple"].dropna().mean()), 6)
            if trades["r_multiple"].notna().any()
            else None,
        }
    )
    return metrics
