"""Persist walk-forward optimizer runs, parameter sets, folds and top trades."""

from __future__ import annotations

import logging
import math
from datetime import datetime

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from zoneinfo import ZoneInfo

from . import config
from .config import OptimizerConfig, RunConfig
from .entities import ClosedTrade

log = logging.getLogger(__name__)

RUN_TABLE = "backtest2_nas100_hfmed_runs"
PARAMETER_TABLE = "backtest2_nas100_hfmed_parameter_sets"
FOLD_TABLE = "backtest2_nas100_hfmed_fold_results"
MC_TABLE = "backtest2_nas100_hfmed_monte_carlo"
TRADES_TABLE = "backtest2_nas100_hfmed_trades"

METRIC_KEYS = [
    "folds",
    "expected_folds",
    "total_trades",
    "total_return_pct",
    "mean_return_pct",
    "median_return_pct",
    "std_return_pct",
    "max_drawdown_pct",
    "profit_factor",
    "win_rate_pct",
    "profitable_folds_pct",
    "gross_profit_eur",
    "gross_loss_eur",
    "net_profit_eur",
    "avg_trade_pnl_eur",
    "signals_total",
    "ruined_folds",
]

SUMMARY_KEYS = [
    "total_trades",
    "winning_trades",
    "losing_trades",
    "breakeven_trades",
    "win_rate_pct",
    "profit_factor",
    "total_return_pct",
    "max_drawdown_pct",
    "avg_win_pct",
    "avg_loss_pct",
    "gross_profit_eur",
    "gross_loss_eur",
    "net_profit_eur",
    "avg_trade_pnl_eur",
    "final_equity",
]

MC_KEYS = [
    "mc_score_rank",
    "n_simulations",
    "base_final_equity_p05",
    "base_final_equity_p25",
    "base_final_equity_p50",
    "base_final_equity_p75",
    "base_final_equity_p95",
    "base_max_drawdown_p05",
    "base_max_drawdown_p25",
    "base_max_drawdown_p50",
    "base_max_drawdown_p75",
    "base_max_drawdown_p95",
    "base_total_return_p05",
    "base_total_return_p25",
    "base_total_return_p50",
    "base_total_return_p75",
    "base_total_return_p95",
    "base_prob_of_ruin_pct",
    "base_prob_profitable_pct",
    "base_worst_final_equity",
    "base_best_final_equity",
    "base_worst_max_drawdown_pct",
    "slip_final_equity_p05",
    "slip_final_equity_p25",
    "slip_final_equity_p50",
    "slip_final_equity_p75",
    "slip_final_equity_p95",
    "slip_max_drawdown_p05",
    "slip_max_drawdown_p25",
    "slip_max_drawdown_p50",
    "slip_max_drawdown_p75",
    "slip_max_drawdown_p95",
    "slip_total_return_p05",
    "slip_total_return_p25",
    "slip_total_return_p50",
    "slip_total_return_p75",
    "slip_total_return_p95",
    "slip_prob_of_ruin_pct",
    "slip_prob_profitable_pct",
    "slip_worst_final_equity",
    "slip_best_final_equity",
    "slip_worst_max_drawdown_pct",
    "seq_final_equity_p05",
    "seq_final_equity_p25",
    "seq_final_equity_p50",
    "seq_final_equity_p75",
    "seq_final_equity_p95",
    "seq_max_drawdown_p05",
    "seq_max_drawdown_p25",
    "seq_max_drawdown_p50",
    "seq_max_drawdown_p75",
    "seq_max_drawdown_p95",
    "seq_total_return_p05",
    "seq_total_return_p25",
    "seq_total_return_p50",
    "seq_total_return_p75",
    "seq_total_return_p95",
    "seq_prob_of_ruin_pct",
    "seq_prob_profitable_pct",
    "seq_worst_final_equity",
    "seq_best_final_equity",
    "seq_worst_max_drawdown_pct",
]


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(config.RESULT_SCHEMA), sql.Identifier(name))


def _run_label() -> str:
    return datetime.now(ZoneInfo(config.RUN_LABEL_TZ)).strftime("%Y-%m-%d %H:%M")


def _notes(cfg: RunConfig, mode: str) -> str:
    parts = [
        cfg.symbol,
        f"mode {mode}",
        f"source {cfg.source_table}",
        f"bar_seconds {cfg.bar_seconds}",
        f"median {cfg.median_quantile:g}",
        f"stop_mode {cfg.stop_mode}",
        f"account {cfg.account_profile}",
        "spread live_bid_ask",
    ]
    if config.RUN_NOTES_EXTRA:
        parts.append(config.RUN_NOTES_EXTRA)
    return " | ".join(parts)


def _db_scalar(value):
    if value is None:
        return None
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if hasattr(value, "item"):
        return value.item()
    return value


def _db_value(value):
    value = _db_scalar(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _db_round(value, digits: int):
    value = _db_value(value)
    if value is None:
        return None
    return round(float(value), digits)


def validate_schema(conn: psycopg2.extensions.connection) -> None:
    required = [RUN_TABLE, PARAMETER_TABLE, FOLD_TABLE, MC_TABLE, TRADES_TABLE]
    missing = []
    with conn.cursor() as cur:
        for table in required:
            cur.execute("SELECT to_regclass(%s)", (f"{config.RESULT_SCHEMA}.{table}",))
            if cur.fetchone()[0] is None:
                missing.append(f"{config.RESULT_SCHEMA}.{table}")
    if missing:
        raise RuntimeError(
            "Missing result tables: "
            + ", ".join(missing)
            + ". Run the init container so init/schema.sql creates the schema."
        )


def create_run(
    conn,
    cfg: RunConfig,
    opt: OptimizerConfig | None,
    mode: str,
    data_start_ts,
    data_end_ts,
    ticks_loaded: int,
    bars_built: int,
    folds_built: int,
) -> int:
    columns = {
        "run_label": _run_label(),
        "mode": mode,
        "status": "running",
        "notes": _notes(cfg, mode),
        "source_table": cfg.source_table,
        "symbol": cfg.symbol,
        "start_ts_utc": cfg.start_ts_utc,
        "end_ts_utc": cfg.end_ts_utc,
        "data_start_ts": data_start_ts,
        "data_end_ts": data_end_ts,
        "ticks_loaded": ticks_loaded,
        "bars_built": bars_built,
        "folds_built": folds_built,
        "bar_seconds": cfg.bar_seconds,
        "baseline_lookback_bars": cfg.lookback_bars,
        "min_lookback_bars": cfg.min_lookback_bars,
        "price_step": cfg.price_step,
        "median_quantile": cfg.median_quantile,
        "band_lower_quantile": cfg.band_lower_quantile,
        "band_upper_quantile": cfg.band_upper_quantile,
        "stop_mode": cfg.stop_mode,
        "baseline_stop_points": cfg.stop_points,
        "baseline_take_profit_points": cfg.take_profit_points,
        "baseline_min_profile_range_points": cfg.min_profile_range_points,
        "baseline_stop_profile_lower_quantile": cfg.stop_profile_lower_quantile,
        "baseline_stop_profile_upper_quantile": cfg.stop_profile_upper_quantile,
        "baseline_stop_profile_buffer_points": cfg.stop_profile_buffer_points,
        "baseline_min_stop_distance_points": cfg.min_stop_distance_points,
        "baseline_max_stop_distance_points": cfg.max_stop_distance_points,
        "account_profile": cfg.account_profile,
        "initial_equity": cfg.initial_equity,
        "account_currency": cfg.account_currency,
        "margin_requirement_pct": cfg.margin_requirement_pct,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "max_margin_pct": cfg.max_margin_pct,
        "contract_multiplier": cfg.contract_multiplier,
        "lot_size": cfg.lot_size,
        "eurusd_rate": cfg.eurusd_rate,
        "spread_points": cfg.spread_points,
        "slippage_points": cfg.slippage_points,
        "commission_per_unit": cfg.commission_per_unit,
        "monte_carlo_enabled": cfg.monte_carlo_enabled,
        "monte_carlo_simulations": cfg.monte_carlo_simulations,
        "mc_extra_slippage_points": cfg.mc_extra_slippage_points,
        "mc_block_size": cfg.mc_block_size,
        "mc_ruin_drawdown_pct": cfg.mc_ruin_drawdown_pct,
        "mc_random_seed": cfg.mc_random_seed,
        "wf_train_days": opt.train_days if opt else None,
        "wf_test_days": opt.test_days if opt else None,
        "wf_step_days": opt.step_days if opt else None,
        "wf_train_top_n_per_fold": opt.train_top_n_per_fold if opt else None,
        "optimizer_processes": opt.processes if opt else None,
        "stage1_max_parameter_sets": opt.stage1_max_parameter_sets if opt else None,
        "stage2_enabled": opt.stage2_enabled if opt else None,
        "stage2_seed_top_n": opt.stage2_seed_top_n if opt else None,
        "stage2_max_parameter_sets": opt.stage2_max_parameter_sets if opt else None,
        "mc_score_top_n": opt.mc_score_top_n if opt else None,
        "persist_top_trades_n": opt.persist_top_trades_n if opt else None,
        "min_oos_trades": opt.min_oos_trades if opt else None,
        "min_oos_profit_factor": opt.min_oos_profit_factor if opt else None,
        "max_oos_drawdown_pct": opt.max_oos_drawdown_pct if opt else None,
        "max_mc_ruin_pct": opt.max_mc_ruin_pct if opt else None,
    }
    cols = list(columns.keys())
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES ({ph}) RETURNING run_id").format(
        tbl=_table(RUN_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        ph=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    with conn.cursor() as cur:
        cur.execute(query, [_db_value(columns[col]) for col in cols])
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created optimizer run %d mode %s", run_id, mode)
    return run_id


def update_run_complete(
    conn,
    run_id: int,
    status: str,
    run_duration_seconds: float,
    stage1_parameter_sets: int,
    stage2_parameter_sets: int,
    best_parameter_set_id: int | None,
    best_score: float | None,
) -> None:
    fields = {
        "status": status,
        "run_duration_seconds": round(run_duration_seconds, 3),
        "stage1_parameter_sets": stage1_parameter_sets,
        "stage2_parameter_sets": stage2_parameter_sets,
        "best_parameter_set_id": best_parameter_set_id,
        "best_score": best_score,
    }
    assignments = sql.SQL(", ").join(
        sql.SQL("{} = {}").format(sql.Identifier(key), sql.Placeholder()) for key in fields
    )
    query = sql.SQL("UPDATE {tbl} SET {assignments} WHERE run_id = {run_id}").format(
        tbl=_table(RUN_TABLE),
        assignments=assignments,
        run_id=sql.Placeholder(),
    )
    with conn.cursor() as cur:
        cur.execute(query, [*(_db_value(value) for value in fields.values()), run_id])
    conn.commit()


def insert_parameter_sets(conn, run_id: int, aggregates: list[dict]) -> dict[str, int]:
    if not aggregates:
        return {}
    columns = [
        "run_id",
        "stage",
        "stage_rank",
        "parameter_hash",
        "parameter_label",
        "parameter_signature",
        "lookback_bars",
        "take_profit_points",
        "min_profile_range_points",
        "stop_profile_lower_quantile",
        "stop_profile_upper_quantile",
        "stop_profile_buffer_points",
        "min_stop_distance_points",
        "max_stop_distance_points",
        "pre_mc_score",
        "score",
        "mc_scored",
        "mc_prob_of_ruin_pct",
        "passed_pre_mc_filters",
        "passed_filters",
    ]
    columns.extend(f"train_{key}" for key in METRIC_KEYS)
    columns.extend(f"oos_{key}" for key in METRIC_KEYS)

    rows = []
    for item in aggregates:
        values = item["values"]
        row = [
            run_id,
            item["stage"],
            item["stage_rank"],
            item["parameter_hash"],
            item["parameter_label"],
            item["parameter_signature"],
            int(values["LOOKBACK_BARS"]),
            values["TAKE_PROFIT_POINTS"],
            values["MIN_PROFILE_RANGE_POINTS"],
            values["STOP_PROFILE_LOWER_QUANTILE"],
            values["STOP_PROFILE_UPPER_QUANTILE"],
            values["STOP_PROFILE_BUFFER_POINTS"],
            values["MIN_STOP_DISTANCE_POINTS"],
            values["MAX_STOP_DISTANCE_POINTS"],
            item["pre_mc_score"],
            item["score"],
            item["mc_scored"],
            item["mc_prob_of_ruin_pct"],
            item["passed_pre_mc_filters"],
            item["passed_filters"],
        ]
        row.extend(item.get(f"train_{key}") for key in METRIC_KEYS)
        row.extend(item.get(f"oos_{key}") for key in METRIC_KEYS)
        rows.append(tuple(_db_value(value) for value in row))

    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s RETURNING parameter_set_id, parameter_hash").format(
        tbl=_table(PARAMETER_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
    )
    with conn.cursor() as cur:
        returned = execute_values(cur, query.as_string(conn), rows, page_size=2000, fetch=True)
    conn.commit()
    log.info("Inserted parameter sets %d", len(returned))
    return {digest: parameter_set_id for parameter_set_id, digest in returned}


def insert_fold_results(conn, run_id: int, evaluations: list, parameter_ids: dict[str, int]) -> None:
    if not evaluations:
        return
    columns = [
        "run_id",
        "parameter_set_id",
        "stage",
        "fold_index",
        "window_role",
        "window_start",
        "window_end",
        "ticks_simulated",
        "bars_total",
        "signals_total",
        "long_signals",
        "short_signals",
        "rejected_missing_band",
        "rejected_band_too_narrow",
        "rejected_stop_too_small",
        "rejected_stop_too_large",
        "skipped_no_size",
        "ruined",
        "score",
    ]
    columns.extend(SUMMARY_KEYS)
    rows = []
    for item in evaluations:
        parameter_set_id = parameter_ids.get(item.parameter_hash)
        if parameter_set_id is None:
            continue
        row = [
            run_id,
            parameter_set_id,
            item.stage,
            item.fold_index,
            item.window_role,
            item.window_start,
            item.window_end,
            item.ticks_simulated,
            item.bars_total,
            item.signals_total,
            item.long_signals,
            item.short_signals,
            item.rejected_missing_band,
            item.rejected_band_too_narrow,
            item.rejected_stop_too_small,
            item.rejected_stop_too_large,
            item.skipped_no_size,
            item.ruined,
            item.score,
        ]
        row.extend(item.summary.get(key) for key in SUMMARY_KEYS)
        rows.append(tuple(_db_value(value) for value in row))
    if not rows:
        return
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s").format(
        tbl=_table(FOLD_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
    )
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), rows, page_size=2000)
    conn.commit()
    log.info("Inserted fold results %d", len(rows))


def insert_monte_carlo(conn, mc_by_hash: dict[str, dict], parameter_ids: dict[str, int]) -> None:
    if not mc_by_hash:
        return
    columns = ["parameter_set_id", *MC_KEYS]
    rows = []
    for digest, mc in mc_by_hash.items():
        parameter_set_id = parameter_ids.get(digest)
        if parameter_set_id is None:
            continue
        rows.append(tuple(_db_value(value) for value in [parameter_set_id, *(mc.get(key) for key in MC_KEYS)]))
    if not rows:
        return
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s").format(
        tbl=_table(MC_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
    )
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), rows, page_size=500)
    conn.commit()
    log.info("Inserted Monte-Carlo rows %d", len(rows))


def trade_rows(parameter_set_id: int, stage: str, fold_index: int, window_role: str, trades: list[ClosedTrade]) -> list[tuple]:
    rows = []
    for t in trades:
        rows.append(
            (
                parameter_set_id,
                stage,
                fold_index,
                window_role,
                _db_scalar(t.signal_ts),
                _db_scalar(t.entry_ts),
                _db_scalar(t.exit_ts),
                t.direction,
                _db_round(t.median_level, 4),
                _db_round(t.signal_mid, 4),
                _db_round(t.previous_mid, 4),
                _db_round(t.entry_bid, 4),
                _db_round(t.entry_ask, 4),
                _db_round(t.entry_price, 4),
                _db_round(t.exit_bid, 4),
                _db_round(t.exit_ask, 4),
                _db_round(t.exit_price, 4),
                _db_round(t.stop_price, 4),
                _db_round(t.take_profit_price, 4),
                _db_round(t.units, 8),
                _db_round(t.notional_eur, 2),
                _db_round(t.margin_used_eur, 2),
                _db_round(t.gross_pnl_eur, 2),
                _db_round(t.extra_costs_eur, 2),
                _db_round(t.pnl_eur, 2),
                _db_round(t.equity_before, 2),
                _db_round(t.equity_after, 2),
                _db_round(t.return_pct, 4),
                _db_round(t.price_pnl_points, 4),
                t.outcome_status,
                int(t.ticks_held),
                _db_round(t.seconds_held, 3),
            )
        )
    return rows


def insert_trade_rows(conn, rows: list[tuple]) -> None:
    if not rows:
        return
    columns = [
        "parameter_set_id",
        "stage",
        "fold_index",
        "window_role",
        "signal_ts",
        "entry_ts",
        "exit_ts",
        "direction",
        "median_level",
        "signal_mid",
        "previous_mid",
        "entry_bid",
        "entry_ask",
        "entry_price",
        "exit_bid",
        "exit_ask",
        "exit_price",
        "stop_price",
        "take_profit_price",
        "units",
        "notional_eur",
        "margin_used_eur",
        "gross_pnl_eur",
        "extra_costs_eur",
        "pnl_eur",
        "equity_before",
        "equity_after",
        "return_pct",
        "price_pnl_points",
        "outcome_status",
        "ticks_held",
        "seconds_held",
    ]
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES %s").format(
        tbl=_table(TRADES_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, columns)),
    )
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), rows, page_size=1000)
    conn.commit()
    log.info("Inserted top trades %d", len(rows))


def mark_top_trade_sets(conn, run_id: int, parameter_set_ids: list[int]) -> None:
    if not parameter_set_ids:
        return
    query = sql.SQL(
        "UPDATE {tbl} SET top_trades_persisted = TRUE WHERE run_id = %s AND parameter_set_id = ANY(%s)"
    ).format(tbl=_table(PARAMETER_TABLE))
    with conn.cursor() as cur:
        cur.execute(query, (run_id, parameter_set_ids))
    conn.commit()
