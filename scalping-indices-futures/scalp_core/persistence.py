"""Persist runs, trades and Monte-Carlo results to TimescaleDB."""

import logging
from datetime import datetime
from typing import Optional

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values
from zoneinfo import ZoneInfo

from . import config
from .config import RunConfig
from .entities import ClosedTrade

log = logging.getLogger(__name__)


def _table(name: str) -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(config.RESULT_SCHEMA), sql.Identifier(name))


def _run_label() -> str:
    return datetime.now(ZoneInfo(config.RUN_LABEL_TZ)).strftime("%Y-%m-%d %H:%M")


def _notes() -> str:
    parts = [
        f"{config.SYMBOL}/{config.BAR_SIZE}",
        f"price={config.PRICE_MODEL}",
        f"vol={config.VOL_MODEL}",
        f"decision={config.DECISION_MODEL}",
        f"regime_states={config.REGIME_STATES}",
        f"account={config.ACCOUNT_PROFILE}",
    ]
    if config.RUN_NOTES_EXTRA:
        parts.append(config.RUN_NOTES_EXTRA)
    return " | ".join(parts)


def create_run(conn, cfg: RunConfig, data_start_ts, data_end_ts, bars_total: int) -> int:
    columns = {
        "run_label": _run_label(),
        "notes": _notes(),
        "symbol": cfg.symbol,
        "bar_size": cfg.bar_size,
        "data_start_ts": data_start_ts,
        "data_end_ts": data_end_ts,
        "bars_total": bars_total,
        "price_model": cfg.price_model,
        "vol_model": cfg.vol_model,
        "decision_model": cfg.decision_model,
        "regime_states": cfg.regime_states,
        "regime_block_high_vol_state": cfg.regime_block_high_vol_state,
        "warmup_bars": cfg.warmup_bars,
        "train_window_bars": cfg.train_window_bars,
        "refit_every_bars": cfg.refit_every_bars,
        "prob_threshold": cfg.prob_threshold,
        "stop_vol_mult": cfg.stop_vol_mult,
        "tp_vol_mult": cfg.tp_vol_mult,
        "min_stop_pct": cfg.min_stop_pct,
        "max_stop_pct": cfg.max_stop_pct,
        "max_hold_bars": cfg.max_hold_bars,
        "allow_short": cfg.allow_short,
        "session_flat_time": cfg.session_flat_time,
        "session_tz": cfg.session_tz,
        "account_profile": cfg.account_profile,
        "initial_equity": cfg.initial_equity,
        "account_currency": cfg.account_currency,
        "margin_requirement_pct": cfg.margin_requirement_pct,
        "risk_per_trade_pct": cfg.risk_per_trade_pct,
        "max_margin_pct": cfg.max_margin_pct,
        "contract_multiplier": cfg.contract_multiplier,
        "eurusd_rate": cfg.eurusd_rate,
        "spread_bps": cfg.spread_bps,
        "slippage_bps": cfg.slippage_bps,
        "commission_per_unit": cfg.commission_per_unit,
    }
    cols = list(columns.keys())
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES ({ph}) RETURNING run_id").format(
        tbl=_table("backtest2_scalp_runs"),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        ph=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    with conn.cursor() as cur:
        cur.execute(query, [columns[k] for k in cols])
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created run_id %d (%s)", run_id, _notes())
    return run_id


def update_run_summary(conn, run_id: int, summary: dict, run_duration_seconds: float,
                       bars_simulated: int, ruined: bool) -> None:
    fields = {
        "run_duration_seconds": round(run_duration_seconds, 3),
        "bars_simulated": bars_simulated,
        "ruined": ruined,
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
        sql.SQL("{} = {}").format(sql.Identifier(k), sql.Placeholder()) for k in fields
    )
    query = sql.SQL("UPDATE {tbl} SET {a} WHERE run_id = {ph}").format(
        tbl=_table("backtest2_scalp_runs"), a=assignments, ph=sql.Placeholder(),
    )
    with conn.cursor() as cur:
        cur.execute(query, [*fields.values(), run_id])
    conn.commit()


def write_trades(conn, run_id: int, trades: list[ClosedTrade]) -> None:
    if not trades:
        return
    rows = [
        (
            run_id, t.intent_ts, t.entry_ts, t.entry_price, t.direction, t.units,
            round(t.notional, 2), round(t.margin_used, 2), t.regime_state, t.prob_up,
            t.sigma_pts, t.stop_price, t.take_profit_price, t.outcome_status, t.exit_ts,
            t.exit_price, t.bars_held, t.return_pct, round(t.pnl, 2), round(t.costs, 2),
            round(t.equity_before, 2), round(t.equity_after, 2),
        )
        for t in trades
    ]
    query = sql.SQL(
        "INSERT INTO {tbl} (run_id, intent_ts, entry_ts, entry_price, direction, units, "
        "notional_eur, margin_used_eur, regime_state, prob_up, sigma_pts, stop_price, "
        "take_profit_price, outcome_status, exit_ts, exit_price, bars_held, return_pct, "
        "pnl_eur, costs_eur, equity_before, equity_after) VALUES %s"
    ).format(tbl=_table("backtest2_scalp_trades"))
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), rows, page_size=500)
    conn.commit()
    log.info("Wrote %d trades for run_id %d", len(trades), run_id)


def write_monte_carlo(conn, run_id: int, mc: Optional[dict]) -> None:
    if not mc:
        return
    payload = {"run_id": run_id, **mc}
    cols = list(payload.keys())
    query = sql.SQL("INSERT INTO {tbl} ({cols}) VALUES ({ph})").format(
        tbl=_table("backtest2_scalp_monte_carlo"),
        cols=sql.SQL(", ").join(map(sql.Identifier, cols)),
        ph=sql.SQL(", ").join(sql.Placeholder() * len(cols)),
    )
    with conn.cursor() as cur:
        cur.execute(query, [payload[k] for k in cols])
    conn.commit()
    log.info("Wrote Monte-Carlo results for run_id %d", run_id)
