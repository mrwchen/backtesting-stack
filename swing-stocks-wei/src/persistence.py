"""Read/write helpers for the backtest_wei_* result tables."""
from __future__ import annotations

import pandas as pd

from . import db
from .config import Config
from .simulator import EQUITY_COLUMNS, TRADE_COLUMNS
from .strategy import SIGNAL_COLUMNS

RESULT_TABLES = (
    "backtest_wei_signals_daily",
    "backtest_wei_runs",
    "backtest_wei_trades",
    "backtest_wei_equity_daily",
)

SIGNALS_TABLE = "backtest_wei_signals_daily"
RUNS_TABLE = "backtest_wei_runs"
TRADES_TABLE = "backtest_wei_trades"
EQUITY_TABLE = "backtest_wei_equity_daily"


def write_signals(conn, run_id: int, signals: pd.DataFrame) -> None:
    if signals.empty:
        return
    out = signals.copy()
    out["run_id"] = run_id
    db.copy_df(conn, out, SIGNALS_TABLE, ["run_id", *SIGNAL_COLUMNS])


def create_run(conn, cfg: Config, start, end, signal_count: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {RUNS_TABLE}
                (strategy_version, config_fingerprint, run_label, start_date, end_date, warmup_calendar_days,
                 high_lookback_days, high_recent_days, ema_fast_days, ema_slow_days,
                 ema_cross_lookback_days, min_pullback_pct, max_entry_gap_pct,
                 atr_days, initial_stop_mode, atr_stop_multiple,
                 volume_sma_days, volume_filter_enable, position_size_usd, stop_loss_pct,
                 trailing_activate_pct, trailing_loss_pct, initial_equity,
                 allow_fractional_shares, min_price, min_market_cap_usd,
                 revenue_yoy_min, revenue_stale_trading_days,
                 ibkr_category_breadth_filter_enable,
                 ibkr_category_breadth_on_threshold, ibkr_category_breadth_off_threshold,
                 ibkr_category_breadth_min_symbols, signal_count, trade_count)
            VALUES
                (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                 %s, %s, %s, %s, %s, %s)
            RETURNING run_id
            """,
            (
                cfg.strategy_version,
                cfg.fingerprint(),
                cfg.run_label,
                start,
                end,
                cfg.warmup_calendar_days,
                cfg.high_lookback_days,
                cfg.high_recent_days,
                cfg.ema_fast_days,
                cfg.ema_slow_days,
                cfg.ema_cross_lookback_days,
                cfg.min_pullback_pct,
                cfg.max_entry_gap_pct,
                cfg.atr_days,
                cfg.initial_stop_mode,
                cfg.atr_stop_multiple,
                cfg.volume_sma_days,
                cfg.volume_filter_enable,
                cfg.position_size_usd,
                cfg.stop_loss_pct,
                cfg.trailing_activate_pct,
                cfg.trailing_loss_pct,
                cfg.initial_equity,
                cfg.allow_fractional_shares,
                cfg.min_price,
                cfg.min_market_cap_usd,
                cfg.revenue_yoy_min,
                cfg.revenue_stale_trading_days,
                cfg.ibkr_category_breadth_filter_enable,
                cfg.ibkr_category_breadth_on_threshold,
                cfg.ibkr_category_breadth_off_threshold,
                cfg.ibkr_category_breadth_min_symbols,
                signal_count,
                0,
            ),
        )
        run_id = int(cur.fetchone()[0])
    conn.commit()
    return run_id


def update_run_result(conn, run_id: int, metrics: dict, trade_count: int) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {RUNS_TABLE}
            SET trade_count = %s,
                final_equity = %s,
                total_pnl = %s,
                total_return = %s,
                cagr = %s,
                max_drawdown = %s,
                win_rate = %s,
                profit_factor = %s,
                avg_r_multiple = %s,
                avg_holding_days = %s
            WHERE run_id = %s
            """,
            (
                trade_count,
                metrics.get("final_equity"),
                metrics.get("total_pnl"),
                metrics.get("total_return"),
                metrics.get("cagr"),
                metrics.get("max_drawdown"),
                metrics.get("win_rate"),
                metrics.get("profit_factor"),
                metrics.get("avg_r_multiple"),
                metrics.get("avg_holding_days"),
                run_id,
            ),
        )
    conn.commit()


def write_trades(conn, run_id: int, trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    out = trades.copy()
    out["run_id"] = run_id
    columns = ["run_id", *TRADE_COLUMNS]
    db.copy_df(conn, out, TRADES_TABLE, columns)


def write_equity(conn, run_id: int, equity: pd.DataFrame) -> None:
    if equity.empty:
        return
    out = equity.copy()
    out["run_id"] = run_id
    columns = ["run_id", *EQUITY_COLUMNS]
    db.copy_df(conn, out, EQUITY_TABLE, columns)
