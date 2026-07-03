"""Read/write helpers for the backtesting_minervini_* result tables."""
from __future__ import annotations

import json
import logging

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

RS_TABLE = "backtesting_minervini_rs_daily"
SCREEN_TABLE = "backtesting_minervini_screen_daily"
SETUPS_TABLE = "backtesting_minervini_setups"
RUNS_TABLE = "backtesting_minervini_runs"
TRADES_TABLE = "backtesting_minervini_trades"
EQUITY_TABLE = "backtesting_minervini_equity_daily"

RS_COLUMNS = ["period_end_date", "symbol", "rs_raw", "rs_rating", "universe_size"]

SCREEN_COLUMNS = [
    "period_end_date", "symbol", "close", "rs_rating",
    "crit_price_above_ma150_200", "crit_ma150_above_ma200", "crit_ma200_rising",
    "crit_ma50_above_ma150_200", "crit_price_above_ma50", "crit_above_52w_low",
    "crit_near_52w_high", "crit_rs_rating", "trend_template_pass",
    "eps_pass", "revenue_pass", "margin_pass", "fundamentals_pass",
    "screen_pass", "eps_yoy", "revenue_yoy",
]

SETUP_COLUMNS = [
    "symbol", "sector", "industry", "detect_date", "pivot", "last_low", "stop_level",
    "base_start_date", "base_days", "n_contractions", "contraction_depths",
    "dryup_ratio", "close", "valid_until",
]

TRADE_COLUMNS = [
    "run_id", "position_id", "setup_id", "symbol", "sector", "industry",
    "leg", "exit_reason",
    "entry_date", "entry_price", "stop_price", "pivot", "shares",
    "exit_date", "exit_price", "pnl", "r_multiple", "holding_days",
]

EQUITY_COLUMNS = ["run_id", "period_end_date", "equity", "open_positions", "exposure_pct"]


def write_rs(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, RS_TABLE, "period_end_date", start, end)
    db.copy_df(conn, df, RS_TABLE, RS_COLUMNS)


def write_screen(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, SCREEN_TABLE, "period_end_date", start, end)
    db.copy_df(conn, df, SCREEN_TABLE, SCREEN_COLUMNS)


def write_setups(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, SETUPS_TABLE, "detect_date", start, end)
    if df.empty:
        log.warning("no setups detected in %s..%s", start, end)
        return
    df = df.copy()
    df["contraction_depths"] = df["contraction_depths"].map(json.dumps)
    db.copy_df(conn, df, SETUPS_TABLE, SETUP_COLUMNS)


def read_screen_pass_days(conn, start, end) -> pd.DataFrame:
    return db.read_df(
        conn,
        f"""SELECT symbol, period_end_date FROM {SCREEN_TABLE}
            WHERE screen_pass AND period_end_date BETWEEN %s AND %s""",
        (start, end),
    )


def read_setups(conn, start, end) -> pd.DataFrame:
    return db.read_df(
        conn,
        f"""SELECT setup_id, symbol, detect_date, pivot, last_low, stop_level, valid_until
            FROM {SETUPS_TABLE}
            WHERE detect_date BETWEEN %s AND %s
            ORDER BY detect_date, symbol""",
        (start, end),
    )


def create_run(conn, cfg: Config, metrics: dict, start, end) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {RUNS_TABLE}
                (run_label, start_date, end_date, params, initial_equity, final_equity,
                 total_return, cagr, max_drawdown, win_rate, profit_factor,
                 avg_r_multiple, num_positions, num_trade_legs, avg_exposure)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING run_id""",
            (
                cfg.run_label, start, end, cfg.to_json(),
                metrics.get("initial_equity"), metrics.get("final_equity"),
                metrics.get("total_return"), metrics.get("cagr"),
                metrics.get("max_drawdown"), metrics.get("win_rate"),
                metrics.get("profit_factor"), metrics.get("avg_r_multiple"),
                metrics.get("num_positions"), metrics.get("num_trade_legs"),
                metrics.get("avg_exposure"),
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    return run_id


def write_trades(conn, run_id: int, trades: pd.DataFrame) -> None:
    if trades.empty:
        return
    trades = trades.copy()
    trades["run_id"] = run_id
    db.copy_df(conn, trades, TRADES_TABLE, TRADE_COLUMNS)


def write_equity(conn, run_id: int, equity: pd.DataFrame) -> None:
    if equity.empty:
        return
    equity = equity.copy()
    equity["run_id"] = run_id
    db.copy_df(conn, equity, EQUITY_TABLE, EQUITY_COLUMNS)
