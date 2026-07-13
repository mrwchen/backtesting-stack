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
STAGE_STATE_TABLE = "backtesting_minervini_stage_state"

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
    "symbol", "setup_type", "ibkr_industry", "ibkr_category", "detect_date",
    "pivot", "last_low", "stop_level", "base_start_date", "base_days",
    "n_contractions", "contraction_depths", "base_count", "dryup_ratio",
    "setup_score", "prior_advance_pct", "final_tightness_pct",
    "structure_quality_score", "volume_dryup_score", "tightness_score",
    "pivot_proximity_score", "prior_advance_score", "close", "valid_until",
]

TRADE_COLUMNS = [
    "run_id", "position_id", "setup_id", "setup_type", "symbol",
    "ibkr_industry", "ibkr_category", "leg", "exit_reason",
    "entry_date", "entry_price", "stop_price", "pivot", "shares",
    "exit_date", "exit_price", "pnl", "r_multiple", "holding_days",
    "regime_composite", "regime_label",
]

BREAKOUT_EVENT_COLUMNS = [
    "run_id", "setup_id", "setup_type", "symbol", "setup_detect_date",
    "breakout_date", "pivot", "trigger_price", "entry_filled", "entry_date",
    "entry_price", "decision",
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


def write_stage_state(
    conn,
    *,
    stage: str,
    model_version: str,
    config_fingerprint: str,
    input_fingerprint: str,
    output_fingerprint: str,
    start,
    end,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {STAGE_STATE_TABLE}
                (stage, model_version, config_fingerprint, input_fingerprint,
                 output_fingerprint, start_date, end_date, updated_ts)
                VALUES (%s, %s, %s, %s, %s, %s, %s, now())
                ON CONFLICT (stage) DO UPDATE SET
                    model_version = EXCLUDED.model_version,
                    config_fingerprint = EXCLUDED.config_fingerprint,
                    input_fingerprint = EXCLUDED.input_fingerprint,
                    output_fingerprint = EXCLUDED.output_fingerprint,
                    start_date = EXCLUDED.start_date,
                    end_date = EXCLUDED.end_date,
                    updated_ts = now()""",
            (
                stage,
                model_version,
                config_fingerprint,
                input_fingerprint,
                output_fingerprint,
                start,
                end,
            ),
        )
    conn.commit()


def require_stage_state(
    conn,
    *,
    stage: str,
    model_version: str,
    config_fingerprint: str,
    input_fingerprint: str | None,
    start,
    end,
) -> str:
    state = db.read_df(
        conn,
        f"""SELECT model_version, config_fingerprint, input_fingerprint,
                   output_fingerprint, start_date, end_date
            FROM {STAGE_STATE_TABLE} WHERE stage = %s""",
        (stage,),
    )
    if state.empty:
        raise RuntimeError(f"missing {stage} stage state; run the prerequisite stage")
    row = state.iloc[0]
    matches = (
        row["model_version"] == model_version
        and row["config_fingerprint"] == config_fingerprint
        and row["start_date"] == start
        and row["end_date"] == end
        and (input_fingerprint is None or row["input_fingerprint"] == input_fingerprint)
    )
    if not matches:
        raise RuntimeError(
            f"{stage} stage state does not match the current model/configuration"
        )
    return str(row["output_fingerprint"])


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
    df = df.copy()
    df["contraction_depths"] = df["contraction_depths"].map(
        lambda values: "{" + ",".join(f"{float(value):.8g}" for value in values) + "}"
    )
    db.copy_df(conn, df, SETUPS_TABLE, SETUP_COLUMNS)


def read_screen_stage_output(conn, start, end) -> pd.DataFrame:
    columns = ", ".join(SCREEN_COLUMNS)
    return db.read_df(
        conn,
        f"""SELECT {columns} FROM {SCREEN_TABLE}
            WHERE screen_pass AND period_end_date BETWEEN %s AND %s
            ORDER BY period_end_date, symbol""",
        (start, end),
    )


def read_setups(conn, start, end) -> pd.DataFrame:
    return db.read_df(
        conn,
        f"""SELECT st.setup_id, st.symbol, st.setup_type, st.detect_date,
                   st.pivot, st.last_low,
                   st.stop_level, st.base_start_date, st.base_days,
                   st.n_contractions, st.contraction_depths, st.base_count,
                   st.dryup_ratio,
                   st.setup_score, st.prior_advance_pct,
                   st.final_tightness_pct, st.structure_quality_score,
                   st.volume_dryup_score, st.tightness_score,
                   st.pivot_proximity_score, st.prior_advance_score,
                   st.close, st.valid_until,
                   sc.rs_rating, sc.ibkr_industry_rs_rating,
                   sc.ibkr_category_rs_rating, sc.stock_industry_rs_rating,
                   sc.stock_category_rs_rating, sc.group_filter_pass,
                   sc.ibkr_industry_breadth_pass, sc.fundamental_score,
                   sc.fundamentals_pass, sc.institutional_manager_count,
                   sc.institutional_net_activity,
                   sc.institutional_sponsorship_pass, sc.eps_yoy, sc.revenue_yoy
            FROM {SETUPS_TABLE} st
            LEFT JOIN {SCREEN_TABLE} sc
              ON sc.symbol = st.symbol
             AND sc.period_end_date = st.detect_date
            WHERE st.detect_date BETWEEN %s AND %s
            ORDER BY st.detect_date, st.symbol""",
        (start, end),
    )


def create_run(
    conn,
    cfg: Config,
    metrics: dict,
    start,
    end,
    *,
    model_version: str,
    input_fingerprint: str,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""INSERT INTO {RUNS_TABLE}
                (run_label, model_version, input_fingerprint, start_date, end_date,
                 params, initial_equity, final_equity,
                 total_return, cagr, max_drawdown, win_rate, profit_factor,
                 avg_r_multiple, num_positions, num_trade_legs, avg_exposure)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING run_id""",
            (
                cfg.run_label, model_version, input_fingerprint, start, end,
                cfg.to_json(),
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
