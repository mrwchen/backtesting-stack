"""Independent per-symbol trade simulation for Wei pullback signals."""
from __future__ import annotations

from dataclasses import dataclass
from math import floor, isfinite

import numpy as np
import pandas as pd

from .config import Config
from .data_loader import pivot_prices


@dataclass(frozen=True)
class SimulationResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    metrics: dict


TRADE_COLUMNS = [
    "position_id",
    "symbol",
    "ibkr_industry",
    "ibkr_category",
    "signal_date",
    "entry_date",
    "exit_date",
    "exit_reason",
    "entry_price",
    "exit_price",
    "shares",
    "notional_usd",
    "initial_stop_price",
    "exit_stop_price",
    "max_price",
    "pnl",
    "return_pct",
    "r_multiple",
    "holding_days",
]

EQUITY_COLUMNS = [
    "period_end_date",
    "equity",
    "realized_pnl",
    "open_pnl",
    "open_positions",
    "gross_exposure_usd",
    "exposure_pct",
]


def _positive(value) -> bool:
    try:
        return value is not None and isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _shares(entry_price: float, cfg: Config) -> float:
    if cfg.allow_fractional_shares:
        return cfg.position_size_usd / entry_price
    return float(floor(cfg.position_size_usd / entry_price))


def simulate(prices: pd.DataFrame, signals: pd.DataFrame, cfg: Config, start, end) -> SimulationResult:
    trades = _simulate_trades(prices, signals, cfg)
    close_matrix = pivot_prices(prices, "close")
    window_dates = close_matrix.index[
        (close_matrix.index.date >= start) & (close_matrix.index.date <= end)
    ]
    equity = _build_equity_curve(close_matrix.loc[window_dates], trades, cfg)
    metrics = _compute_metrics(equity, trades, cfg, start, end)
    return SimulationResult(trades=trades, equity=equity, metrics=metrics)


def _simulate_trades(prices: pd.DataFrame, signals: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if prices.empty or signals.empty:
        return pd.DataFrame(columns=TRADE_COLUMNS)

    signal_groups = {
        symbol: sub.sort_values("period_end_date").reset_index(drop=True)
        for symbol, sub in signals.groupby("symbol", sort=True)
    }
    rows: list[dict] = []
    position_id = 1

    for symbol, symbol_signals in signal_groups.items():
        bars = prices[prices["symbol"] == symbol].sort_values("date").reset_index(drop=True)
        if bars.empty:
            continue
        date_to_idx = {ts.date(): idx for idx, ts in enumerate(pd.to_datetime(bars["date"]))}
        next_available_idx = 0

        for signal in symbol_signals.itertuples(index=False):
            signal_date = signal.period_end_date
            signal_idx = date_to_idx.get(signal_date)
            if signal_idx is None or signal_idx < next_available_idx:
                continue
            entry_idx = signal_idx + 1
            if entry_idx >= len(bars):
                continue
            entry_bar = bars.iloc[entry_idx]
            entry_price = float(entry_bar["open"])
            if not _positive(entry_price):
                continue
            shares = _shares(entry_price, cfg)
            if shares <= 0:
                continue

            trade = _run_trade(
                bars=bars,
                entry_idx=entry_idx,
                signal=signal,
                position_id=position_id,
                entry_price=entry_price,
                shares=shares,
                cfg=cfg,
            )
            rows.append(trade)
            position_id += 1
            next_available_idx = int(trade["_exit_idx"]) + 1

    if not rows:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    trades = pd.DataFrame(rows).drop(columns=["_exit_idx"])
    return trades[TRADE_COLUMNS].reset_index(drop=True)


def _run_trade(
    bars: pd.DataFrame,
    entry_idx: int,
    signal,
    position_id: int,
    entry_price: float,
    shares: float,
    cfg: Config,
) -> dict:
    initial_stop = entry_price * (1.0 - cfg.stop_loss_pct)
    active_stop = initial_stop
    max_price = entry_price
    exit_idx = len(bars) - 1
    exit_price = float(bars.iloc[-1]["close"])
    exit_reason = "end_of_data"

    for idx in range(entry_idx, len(bars)):
        bar = bars.iloc[idx]
        open_price = float(bar["open"])
        high_price = float(bar["high"])
        low_price = float(bar["low"])
        close_price = float(bar["close"])

        if _positive(low_price) and low_price <= active_stop:
            exit_idx = idx
            exit_price = open_price if _positive(open_price) and open_price <= active_stop else active_stop
            exit_reason = "trailing_stop" if active_stop > initial_stop else "stop_loss"
            break

        if _positive(high_price) and high_price > max_price:
            max_price = high_price
        if max_price >= entry_price * (1.0 + cfg.trailing_activate_pct):
            trailing_stop = max_price * (1.0 - cfg.trailing_loss_pct)
            if trailing_stop > active_stop:
                active_stop = trailing_stop

        if idx == len(bars) - 1 and _positive(close_price):
            exit_price = close_price

    entry_date = pd.Timestamp(bars.iloc[entry_idx]["date"]).date()
    exit_date = pd.Timestamp(bars.iloc[exit_idx]["date"]).date()
    pnl = (exit_price - entry_price) * shares
    notional = entry_price * shares
    initial_risk = max((entry_price - initial_stop) * shares, 1e-12)
    return {
        "_exit_idx": exit_idx,
        "position_id": position_id,
        "symbol": signal.symbol,
        "ibkr_industry": signal.ibkr_industry,
        "ibkr_category": signal.ibkr_category,
        "signal_date": signal.period_end_date,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_reason": exit_reason,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "shares": round(shares, 8),
        "notional_usd": round(notional, 2),
        "initial_stop_price": round(initial_stop, 4),
        "exit_stop_price": round(active_stop, 4),
        "max_price": round(max_price, 4),
        "pnl": round(pnl, 2),
        "return_pct": round((exit_price / entry_price) - 1.0, 6),
        "r_multiple": round(pnl / initial_risk, 6),
        "holding_days": int(exit_idx - entry_idx + 1),
    }


def _build_equity_curve(close_matrix: pd.DataFrame, trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if close_matrix.empty:
        return pd.DataFrame(columns=EQUITY_COLUMNS)
    if trades.empty:
        return pd.DataFrame(
            {
                "period_end_date": [ts.date() for ts in close_matrix.index],
                "equity": cfg.initial_equity,
                "realized_pnl": 0.0,
                "open_pnl": 0.0,
                "open_positions": 0,
                "gross_exposure_usd": 0.0,
                "exposure_pct": 0.0,
            }
        )

    trade_symbols = trades["symbol"].to_numpy()
    symbol_to_col = {symbol: idx for idx, symbol in enumerate(close_matrix.columns)}
    valid_trade_mask = np.array([symbol in symbol_to_col for symbol in trade_symbols], dtype=bool)
    t = trades.loc[valid_trade_mask].reset_index(drop=True)
    if t.empty:
        return pd.DataFrame(columns=EQUITY_COLUMNS)

    date_index = pd.Index([ts.date() for ts in close_matrix.index])
    entry_idx = date_index.get_indexer(pd.to_datetime(t["entry_date"]).dt.date)
    exit_idx = date_index.get_indexer(pd.to_datetime(t["exit_date"]).dt.date)
    valid_date_mask = (entry_idx >= 0) & (exit_idx >= 0)
    if not valid_date_mask.all():
        t = t.loc[valid_date_mask].reset_index(drop=True)
        entry_idx = entry_idx[valid_date_mask]
        exit_idx = exit_idx[valid_date_mask]
        if t.empty:
            return pd.DataFrame(columns=EQUITY_COLUMNS)
    col_idx = np.array([symbol_to_col[symbol] for symbol in t["symbol"]], dtype=int)
    entry_prices = t["entry_price"].to_numpy(dtype=float)
    shares = t["shares"].to_numpy(dtype=float)
    pnls = t["pnl"].to_numpy(dtype=float)
    notionals = t["notional_usd"].to_numpy(dtype=float)
    close_values = close_matrix.to_numpy(dtype=float)

    rows: list[dict] = []
    for idx, ts in enumerate(close_matrix.index):
        closed = exit_idx <= idx
        open_mask = (entry_idx <= idx) & (exit_idx > idx)
        realized_pnl = float(np.nansum(pnls[closed]))
        open_pnl = 0.0
        gross_exposure = 0.0
        if open_mask.any():
            close_prices = close_values[idx, col_idx[open_mask]]
            open_pnl = float(np.nansum((close_prices - entry_prices[open_mask]) * shares[open_mask]))
            gross_exposure = float(np.nansum(notionals[open_mask]))
        equity = cfg.initial_equity + realized_pnl + open_pnl
        exposure_pct = gross_exposure / equity if equity else 0.0
        rows.append(
            {
                "period_end_date": ts.date(),
                "equity": round(equity, 2),
                "realized_pnl": round(realized_pnl, 2),
                "open_pnl": round(open_pnl, 2),
                "open_positions": int(open_mask.sum()),
                "gross_exposure_usd": round(gross_exposure, 2),
                "exposure_pct": round(exposure_pct, 6),
            }
        )
    return pd.DataFrame(rows, columns=EQUITY_COLUMNS)


def _compute_metrics(equity: pd.DataFrame, trades: pd.DataFrame, cfg: Config, start, end) -> dict:
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else cfg.initial_equity
    total_return = (final_equity / cfg.initial_equity) - 1.0
    days = max((pd.Timestamp(end) - pd.Timestamp(start)).days, 1)
    cagr = (final_equity / cfg.initial_equity) ** (365.25 / days) - 1.0

    if equity.empty:
        max_drawdown = 0.0
    else:
        eq = equity["equity"].astype(float)
        max_drawdown = float((eq / eq.cummax() - 1.0).min())

    if trades.empty:
        return {
            "final_equity": final_equity,
            "total_pnl": 0.0,
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": max_drawdown,
            "win_rate": None,
            "profit_factor": None,
            "avg_r_multiple": None,
            "avg_holding_days": None,
        }

    pnl = trades["pnl"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    return {
        "final_equity": final_equity,
        "total_pnl": float(pnl.sum()),
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": profit_factor,
        "avg_r_multiple": float(trades["r_multiple"].astype(float).mean()),
        "avg_holding_days": float(trades["holding_days"].astype(float).mean()),
    }
