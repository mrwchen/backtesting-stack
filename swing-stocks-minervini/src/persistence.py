"""Read/write helpers for the backtesting_minervini_* result tables."""
from __future__ import annotations

import logging

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

RS_TABLE = "backtesting_minervini_rs_daily"
SCREEN_TABLE = "backtesting_minervini_screen_daily"
MARKET_TABLE = "backtesting_minervini_market_daily"
SETUPS_TABLE = "backtesting_minervini_setups"
RUNS_TABLE = "backtesting_minervini_runs"
BREAKOUT_EVENTS_TABLE = "backtesting_minervini_breakout_events"
TRADES_TABLE = "backtesting_minervini_trades"
EQUITY_TABLE = "backtesting_minervini_equity_daily"

RS_COLUMNS = ["period_end_date", "symbol", "rs_raw", "rs_rating", "universe_size"]

SCREEN_COLUMNS = [
    "period_end_date", "symbol", "ibkr_industry", "ibkr_category",
    "close", "rs_rating",
    "ibkr_industry_rs_rating", "ibkr_category_rs_rating",
    "stock_industry_rs_rating", "stock_category_rs_rating",
    "ibkr_industry_pass", "ibkr_category_pass",
    "stock_industry_pass", "stock_category_pass", "group_filter_pass",
    "ibkr_industry_breadth_pct", "ibkr_industry_breadth_on",
    "ibkr_industry_breadth_pass",
    "crit_price_above_ma150_200", "crit_ma150_above_ma200", "crit_ma200_rising",
    "crit_ma50_above_ma150_200", "crit_price_above_ma50", "crit_above_52w_low",
    "crit_near_52w_high", "crit_rs_rating", "trend_template_pass",
    "eps_pass", "revenue_pass", "margin_pass", "acceleration_pass",
    "streak_pass", "stability_pass", "fundamental_score", "fundamentals_pass",
    "institutional_manager_count", "institutional_net_activity",
    "institutional_sponsorship_pass", "screen_pass", "eps_yoy", "revenue_yoy",
    "eps_acceleration", "revenue_acceleration", "margin_delta", "growth_streak",
]

SETUP_COLUMNS = [
    "symbol", "ibkr_industry", "ibkr_category", "detect_date", "pivot", "last_low", "stop_level",
    "base_start_date", "base_days", "n_contractions", "base_count",
    "dryup_ratio", "vcp_score", "depth_quality_score", "final_tightness_score",
    "contraction_smoothness_score", "volume_dryup_score", "volume_slope_score",
    "tight_closes_score", "base_duration_score", "pivot_proximity_score",
    "overhead_supply_score", "prior_advance_score", "weekly_structure_score",
    "close", "valid_until",
]

TRADE_COLUMNS = [
    "run_id", "position_id", "setup_id", "symbol", "ibkr_industry", "ibkr_category",
    "leg", "exit_reason",
    "entry_date", "entry_price", "stop_price", "pivot", "shares",
    "exit_date", "exit_price", "pnl", "r_multiple", "holding_days",
    "regime_composite", "regime_label",
]

BREAKOUT_EVENT_COLUMNS = [
    "run_id", "setup_id", "symbol", "setup_detect_date", "breakout_date",
    "planned_entry_date", "pivot", "trigger_price", "breakout_open",
    "breakout_high", "breakout_low", "breakout_close", "breakout_volume",
    "average_volume_prior", "volume_history_sessions", "breakout_volume_ratio",
    "close_above_pivot", "volume_confirmed", "confirmation_pass",
    "entry_filled", "entry_date", "entry_price", "decision",
]

MARKET_COLUMNS = [
    "period_end_date", "primary_index", "primary_index_close",
    "primary_index_volume", "primary_index_return_pct", "market_breadth_pct",
    "breadth_confirmed", "market_status", "rally_attempt_day",
    "distribution_day", "distribution_days", "follow_through_day",
    "entry_exposure_cap", "market_on",
]

EQUITY_COLUMNS = [
    "run_id", "period_end_date", "equity", "open_positions", "exposure_pct",
    "feedback_exposure_level", "market_exposure_cap", "entry_exposure_limit",
]


def write_rs(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, RS_TABLE, "period_end_date", start, end)
    db.copy_df(conn, df, RS_TABLE, RS_COLUMNS)


def write_screen(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, SCREEN_TABLE, "period_end_date", start, end)
    db.copy_df(conn, df, SCREEN_TABLE, SCREEN_COLUMNS)


def write_market(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, MARKET_TABLE, "period_end_date", start, end)
    db.copy_df(conn, df, MARKET_TABLE, MARKET_COLUMNS)


def write_setups(conn, df: pd.DataFrame, start, end) -> None:
    db.delete_range(conn, SETUPS_TABLE, "detect_date", start, end)
    if df.empty:
        log.warning("no setups detected in %s..%s", start, end)
        return
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
        f"""SELECT st.setup_id, st.symbol, st.detect_date, st.pivot, st.last_low,
                   st.stop_level, st.base_days, st.n_contractions,
                   st.dryup_ratio, st.vcp_score, st.close, st.valid_until,
                   sc.rs_rating, sc.ibkr_industry_rs_rating,
                   sc.ibkr_category_rs_rating, sc.stock_industry_rs_rating,
                   sc.stock_category_rs_rating, sc.eps_yoy, sc.revenue_yoy
            FROM {SETUPS_TABLE} st
            LEFT JOIN {SCREEN_TABLE} sc
              ON sc.symbol = st.symbol
             AND sc.period_end_date = st.detect_date
            WHERE st.detect_date BETWEEN %s AND %s
            ORDER BY st.detect_date, st.symbol""",
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


def write_breakout_events(conn, run_id: int, events: pd.DataFrame) -> None:
    if events.empty:
        return
    events = events.copy()
    events["run_id"] = run_id
    db.copy_df(conn, events, BREAKOUT_EVENTS_TABLE, BREAKOUT_EVENT_COLUMNS)


def write_equity(conn, run_id: int, equity: pd.DataFrame) -> None:
    if equity.empty:
        return
    equity = equity.copy()
    equity["run_id"] = run_id
    db.copy_df(conn, equity, EQUITY_TABLE, EQUITY_COLUMNS)
