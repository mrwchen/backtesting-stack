"""Persist runs, trades and Monte-Carlo results."""

import logging
import math
from datetime import datetime

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from zoneinfo import ZoneInfo

from . import config
from .config import RunConfig
from .entities import ClosedTrade, SimulationResult

log = logging.getLogger(__name__)

RUN_TABLE = "backtest2_nas100_hfmed_runs"
TRADES_TABLE = "backtest2_nas100_hfmed_trades"
MC_TABLE = "backtest2_nas100_hfmed_monte_carlo"


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(config.RESULT_SCHEMA), sql.Identifier(name))


def _run_label() -> str:
    return datetime.now(ZoneInfo(config.RUN_LABEL_TZ)).strftime("%Y-%m-%d %H:%M")


def _notes() -> str:
    parts = [
        f"{config.SYMBOL}",
        f"source={config.SOURCE_TABLE}",
        f"bar_seconds={config.BAR_SECONDS}",
        f"lookback={config.LOOKBACK_BARS}",
        f"median=0.5",
        f"stop_mode={config.STOP_MODE}",
        f"entry_range=q0-q100",
        f"stop_range=q0-q100",
        f"band={config.BAND_LOWER_QUANTILE:g}-{config.BAND_UPPER_QUANTILE:g}",
        f"min_profile_range={config.MIN_PROFILE_RANGE_POINTS:g}",
        f"band_buffer={config.BAND_STOP_BUFFER_POINTS:g}",
        f"stop_limits={config.MIN_STOP_POINTS:g}-{config.MAX_STOP_POINTS:g}",
        f"fixed_stop={config.STOP_POINTS:g}",
        f"tp={config.TAKE_PROFIT_POINTS:g}",
        f"account={config.ACCOUNT_PROFILE}",
        "spread=live_bid_ask",
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
    required = [RUN_TABLE, TRADES_TABLE, MC_TABLE]
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


def create_run(conn, cfg: RunConfig, data_start_ts, data_end_ts, ticks_loaded: int, bars_built: int) -> int:
    columns = {
        "run_label": _run_label(),
        "notes": _notes(),
        "source_table": cfg.source_table,
        "symbol": cfg.symbol,
        "start_ts_utc": cfg.start_ts_utc,
        "end_ts_utc": cfg.end_ts_utc,
        "data_start_ts": data_start_ts,
        "data_end_ts": data_end_ts,
        "ticks_loaded": ticks_loaded,
        "bars_built": bars_built,
        "bar_seconds": cfg.bar_seconds,
        "lookback_bars": cfg.lookback_bars,
        "min_lookback_bars": cfg.min_lookback_bars,
        "price_step": cfg.price_step,
        "median_quantile": cfg.median_quantile,
        "stop_points": cfg.stop_points,
        "take_profit_points": cfg.take_profit_points,
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
    log.info("Created run_id %d (%s)", run_id, _notes())
    return run_id


def update_run_summary(conn, run_id: int, summary: dict, result: SimulationResult, run_duration_seconds: float) -> None:
    fields = {
        "run_duration_seconds": round(run_duration_seconds, 3),
        "ticks_simulated": result.ticks_simulated,
        "signals_total": result.signals_total,
        "long_signals": result.long_signals,
        "short_signals": result.short_signals,
        "skipped_signals_no_size": result.skipped_signals_no_size,
        "ruined": result.ruined,
        "final_equity": summary.get("final_equity"),
        "total_return_pct": summary.get("total_return_pct"),
        "total_trades": summary.get("total_trades"),
        "winning_trades": summary.get("winning_trades"),
        "losing_trades": summary.get("losing_trades"),
        "breakeven_trades": summary.get("breakeven_trades"),
        "win_rate_pct": summary.get("win_rate_pct"),
        "profit_factor": summary.get("profit_factor"),
        "max_drawdown_pct": summary.get("max_drawdown_pct"),
        "avg_win_pct": summary.get("avg_win_pct"),
        "avg_loss_pct": summary.get("avg_loss_pct"),
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


def write_trades(conn, run_id: int, trades: list[ClosedTrade]) -> None:
    if not trades:
        return
    rows = [
        (
            run_id,
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
        for t in trades
    ]
    query = sql.SQL(
        "INSERT INTO {tbl} (run_id, signal_ts, entry_ts, exit_ts, direction, median_level, "
        "signal_mid, previous_mid, entry_bid, entry_ask, entry_price, exit_bid, exit_ask, "
        "exit_price, stop_price, take_profit_price, units, notional_eur, margin_used_eur, "
        "gross_pnl_eur, extra_costs_eur, pnl_eur, equity_before, equity_after, return_pct, "
        "price_pnl_points, outcome_status, ticks_held, seconds_held) VALUES %s"
    ).format(tbl=_table(TRADES_TABLE))
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), rows, page_size=1000)
    conn.commit()
    log.info("Wrote %d trades for run_id %d", len(trades), run_id)


def write_monte_carlo(conn, run_id: int, mc: dict | None) -> None:
    if not mc:
        return
    payload = {"run_id": run_id, **mc}
    cols = list(payload.keys())
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES ({ph})").format(
        tbl=_table(MC_TABLE),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        ph=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    with conn.cursor() as cur:
        cur.execute(query, [_db_value(payload[col]) for col in cols])
    conn.commit()
    log.info("Wrote Monte-Carlo results for run_id %d", run_id)
