"""Persistence for backtest run metadata, trades, decisions, and summaries."""

import logging
import math
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import execute_values

from . import runtime
from .config import *
from .entities import AccountCurvePoint, ClosedTrade, DecisionEvent
from .policy import COMMON_POLICY

log = logging.getLogger(__name__)


def _cfg_or_none(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, None)

def create_run(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    notes: Optional[str] = None,
    reserved_run_id: Optional[int] = None,
) -> int:
    if reserved_run_id is not None and reserved_run_id <= 0:
        raise ValueError(f"Invalid reserved run id: {reserved_run_id!r}")

    run_notes = _build_run_notes(notes, cfg)
    run_id_column = "run_id, " if reserved_run_id is not None else ""
    run_id_placeholder = "%s, " if reserved_run_id is not None else ""
    params = (
        START_DATE, END_DATE, run_notes, RUN_LABEL_TZ, runtime.CURRENT_MODEL_FILE,
        ACCOUNT_PROFILE,
        INITIAL_EQUITY, RISK_PER_TRADE_PCT, MAX_OPEN_POSITIONS,
        MARGIN_REQUIREMENT_PCT, PS_MARGIN_STOP_OUT_LEVEL_PCT, PS_MIN_ENTRY_MARGIN_LEVEL_PCT,
        IBKR_LONG_INITIAL_MARGIN_PCT if ACCOUNT_PROFILE == "ibkr_acc" else None,
        IBKR_LONG_MAINTENANCE_MARGIN_PCT if ACCOUNT_PROFILE == "ibkr_acc" else None,
        IBKR_SHORT_INITIAL_MARGIN_PCT if ACCOUNT_PROFILE == "ibkr_acc" else None,
        IBKR_SHORT_MAINTENANCE_MARGIN_PCT if ACCOUNT_PROFILE == "ibkr_acc" else None,
        ALLOW_FRACTIONAL_SHARES, SPREAD_BPS, SLIPPAGE_BPS,
        COMMISSION_PER_ORDER_USD, COMMISSION_PER_SHARE_USD,
        COMMISSION_MIN_PER_ORDER_USD, COMMISSION_MAX_PCT,
        COMMISSION_BPS, MARGIN_FINANCING_RATE_PCT,
        PS_SHARE_CFD_ARR_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
        PS_SHARE_CFD_ADMIN_FEE_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
        PS_SHARE_CFD_SHORT_BORROW_RATE_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
        PS_SHARE_CFD_OVERNIGHT_DAY_COUNT if ACCOUNT_PROFILE == "ps_acc" else None,
        ENTRY_WINDOW_ENABLED, ENTRY_WINDOW_TZ, ENTRY_WINDOW_START, ENTRY_WINDOW_END,
        COMMON_POLICY.long_min_fundamental, COMMON_POLICY.short_max_fundamental, COMMON_POLICY.min_market_cap_m,
        _cfg_or_none(cfg, "long_min_pullback"),
        _cfg_or_none(cfg, "long_max_pullback"),
        _cfg_or_none(cfg, "long_ideal_pullback"),
        _cfg_or_none(cfg, "long_max_rsi"),
        _cfg_or_none(cfg, "short_min_bounce"),
        _cfg_or_none(cfg, "short_max_bounce"),
        _cfg_or_none(cfg, "short_ideal_bounce"),
        _cfg_or_none(cfg, "short_min_rsi"),
        _cfg_or_none(cfg, "short_max_rsi"),
        TAKE_PROFIT_MODE,
        EXECUTION_LONG_TAKE_PROFIT_PCT,
        EXECUTION_SHORT_TAKE_PROFIT_PCT,
        EXECUTION_LONG_TRAILING_ACTIVATION_PCT,
        EXECUTION_SHORT_TRAILING_ACTIVATION_PCT,
        EXECUTION_LONG_TRAILING_DISTANCE_PCT,
        EXECUTION_SHORT_TRAILING_DISTANCE_PCT,
        EXECUTION_LONG_MAX_HOLD_DAYS,
        EXECUTION_SHORT_MAX_HOLD_DAYS,
        COMMON_STOP_LOSS_ENABLED,
        COMMON_STOP_LOOKBACK_BARS,
        COMMON_STOP_BUFFER,
        COMMON_STOP_ATR_LOOKBACK_BARS,
        COMMON_STOP_ATR_MULT,
        COMMON_MIN_STOP_PCT,
        COMMON_MAX_STOP_PCT,
    )
    if reserved_run_id is not None:
        params = (reserved_run_id, *params)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {_result_table("backtest_runs")} (
                {run_id_column}start_date, end_date, notes, run_label, model_file,
                account_profile,
                initial_equity, risk_per_trade_pct, max_open_positions,
                ps_margin_requirement_pct, ps_margin_stop_out_level_pct, ps_min_entry_margin_level_pct,
                ibkr_long_initial_margin_pct, ibkr_long_maintenance_margin_pct,
                ibkr_short_initial_margin_pct, ibkr_short_maintenance_margin_pct,
                allow_fractional_shares, spread_bps, slippage_bps,
                commission_per_order_usd, commission_per_share_usd,
                commission_min_per_order_usd, commission_max_pct,
                commission_bps, margin_financing_rate_pct,
                ps_share_cfd_arr_pct, ps_share_cfd_admin_fee_pct,
                ps_share_cfd_short_borrow_rate_pct, ps_share_cfd_overnight_day_count,
                entry_window_enabled, entry_window_tz, entry_window_start, entry_window_end,
                long_min_fundamental, short_max_fundamental, min_market_cap_m,
                long_min_pullback, long_max_pullback, long_ideal_pullback, long_max_rsi,
                short_min_bounce, short_max_bounce, short_ideal_bounce, short_min_rsi, short_max_rsi,
                take_profit_mode,
                execution_long_take_profit_pct, execution_short_take_profit_pct,
                execution_long_trailing_activation_pct, execution_short_trailing_activation_pct,
                execution_long_trailing_distance_pct, execution_short_trailing_distance_pct,
                execution_long_max_hold_days, execution_short_max_hold_days,
                common_stop_loss_enabled, common_stop_lookback_bars, common_stop_buffer,
                common_stop_atr_lookback_bars, common_stop_atr_mult,
                common_min_stop_pct, common_max_stop_pct
            ) VALUES (
                {run_id_placeholder}%s, %s, %s, to_char(NOW() AT TIME ZONE %s, 'YYYY-MM-DD HH24:MI'), %s,
                %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s,
                %s, %s, %s
            ) RETURNING run_id
            """,
            params,
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created run %d model %s account profile %s", run_id, runtime.CURRENT_MODEL_FILE, ACCOUNT_PROFILE)
    return run_id


def reserve_run_ids(conn: psycopg2.extensions.connection, count: int) -> list[int]:
    if count < 0:
        raise ValueError(f"Invalid run id reservation count: {count!r}")
    if count == 0:
        return []

    table_name = f"{RESULT_SCHEMA}.backtest_runs"
    with conn.cursor() as cur:
        cur.execute("SELECT pg_get_serial_sequence(%s, 'run_id')", (table_name,))
        sequence_name = cur.fetchone()[0]
        if not sequence_name:
            raise RuntimeError(f"Could not find run_id sequence for {table_name}")
        cur.execute(
            "SELECT nextval(%s::regclass) FROM generate_series(1, %s)",
            (sequence_name, count),
        )
        run_ids = [int(row[0]) for row in cur.fetchall()]

    log.info("Reserved %d backtest run ids first %d last %d", count, run_ids[0], run_ids[-1])
    return run_ids


def delete_run_results(conn: psycopg2.extensions.connection, run_id: int) -> None:
    if run_id <= 0:
        raise ValueError(f"Invalid run id cleanup target: {run_id!r}")

    with conn.cursor() as cur:
        for table_name in (
            "backtest_monte_carlo",
            "backtest_account_curve",
            "backtest_decision_events",
            "backtest_trades",
            "backtest_runs",
        ):
            cur.execute(f"DELETE FROM {_result_table(table_name)} WHERE run_id = %s", (run_id,))
    conn.commit()
    log.warning("Deleted partial run results for retry run %d", run_id)


def write_trades(
    conn: psycopg2.extensions.connection,
    run_id: int,
    trades: list[ClosedTrade],
) -> None:
    if not trades:
        return
    rows = []
    for t in trades:
        p = t.position
        plan = p.plan
        rows.append((
            run_id,
            p.entry_date,
            p.symbol,
            p.exchange,
            p.cik,
            p.direction,
            p.world_regime_label or None,
            Decimal(str(round(p.world_regime_score, 2))) if p.world_regime_score else None,
            p.valuation_label or None,
            plan.sector or None,
            plan.industry or None,
            Decimal(str(round(plan.fundamental_score, 2))),
            Decimal(str(round(plan.intent_score, 4))),
            plan.intent_reason,
            Decimal(str(round(p.entry_price, 4))),
            Decimal(str(round(p.stop_loss, 4))),
            p.take_profit_mode,
            _decimal_or_none(p.take_profit, 4),
            _decimal_or_none(p.trailing_activation_price, 4),
            _decimal_or_none(p.trailing_distance_pct, 4),
            Decimal(str(round(p.position_size_usd, 2))),
            Decimal(str(round(p.shares, 6))),
            Decimal(str(round(p.margin_used, 2))),
            Decimal(str(round(p.maintenance_margin_used, 2))),
            Decimal(str(round(p.equity_before, 2))),
            t.outcome_status,
            Decimal(str(round(t.outcome_price, 4))),
            t.outcome_date,
            t.outcome_bars,
            t.trailing_activated,
            _decimal_or_none(t.trailing_stop, 4),
            Decimal(str(round(t.return_pct, 4))),
            _decimal_or_none(t.margin_hours_usd, 4),
            _decimal_or_none(t.return_per_margin_hour_pct, 8),
            Decimal(str(round(t.pnl_usd, 2))),
            Decimal(str(round(t.equity_after, 2))),
            p.entry_ts,
            t.trailing_activated_ts,
            t.exit_ts,
        ))

    query = """
        INSERT INTO {table} (
            run_id, intent_date, symbol, exchange, cik, direction,
            world_regime_label, world_regime_score,
            valuation_label, sector, industry,
            fundamental_score, intent_score, intent_reason,
            entry_price, stop_loss, take_profit_mode, take_profit,
            trailing_activation_price, trailing_distance_pct,
            position_size_usd, shares, margin_used, maintenance_margin_used, equity_before,
            outcome_status, outcome_price, outcome_date, outcome_bars,
            trailing_activated, trailing_stop, return_pct, margin_hours_usd, return_per_margin_hour_pct,
            pnl_usd, equity_after,
            entry_ts, trailing_activated_ts, exit_ts
        ) VALUES %s
        ON CONFLICT (run_id, intent_date, symbol, exchange, cik, direction, entry_ts) DO UPDATE SET
            world_regime_score = EXCLUDED.world_regime_score,
            outcome_status = EXCLUDED.outcome_status,
            outcome_price  = EXCLUDED.outcome_price,
            outcome_date   = EXCLUDED.outcome_date,
            outcome_bars   = EXCLUDED.outcome_bars,
            trailing_activated = EXCLUDED.trailing_activated,
            trailing_stop  = EXCLUDED.trailing_stop,
            return_pct     = EXCLUDED.return_pct,
            margin_hours_usd = EXCLUDED.margin_hours_usd,
            return_per_margin_hour_pct = EXCLUDED.return_per_margin_hour_pct,
            pnl_usd        = EXCLUDED.pnl_usd,
            equity_after   = EXCLUDED.equity_after,
            trailing_activated_ts = EXCLUDED.trailing_activated_ts,
            exit_ts        = EXCLUDED.exit_ts
    """.format(table=_result_table("backtest_trades"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=TRADE_INSERT_PAGE_SIZE)
    conn.commit()


def write_account_curve(
    conn: psycopg2.extensions.connection,
    run_id: int,
    points: list[AccountCurvePoint],
) -> None:
    if not points:
        return
    rows = [
        (
            p.run_id,
            p.ts,
            p.trade_date,
            p.seq_in_run,
            Decimal(str(round(p.balance_usd, 2))),
            Decimal(str(round(p.open_pnl_usd, 2))),
            Decimal(str(round(p.equity_usd, 2))),
            Decimal(str(round(p.initial_margin_usd, 2))),
            Decimal(str(round(p.maintenance_margin_usd, 2))),
            Decimal(str(round(p.available_funds_usd, 2))),
            Decimal(str(round(p.excess_liquidity_usd, 2))),
            p.open_positions,
            Decimal(str(round(p.realized_pnl_usd, 2))),
            p.closed_trades,
        )
        for p in points
    ]

    query = """
        INSERT INTO {table} (
            run_id, ts, trade_date, seq_in_run,
            balance_usd, open_pnl_usd, equity_usd,
            initial_margin_usd, maintenance_margin_usd,
            available_funds_usd, excess_liquidity_usd,
            open_positions, realized_pnl_usd, closed_trades
        ) VALUES %s
        ON CONFLICT (run_id, ts, seq_in_run) DO UPDATE SET
            trade_date       = EXCLUDED.trade_date,
            balance_usd      = EXCLUDED.balance_usd,
            open_pnl_usd     = EXCLUDED.open_pnl_usd,
            equity_usd       = EXCLUDED.equity_usd,
            initial_margin_usd = EXCLUDED.initial_margin_usd,
            maintenance_margin_usd = EXCLUDED.maintenance_margin_usd,
            available_funds_usd = EXCLUDED.available_funds_usd,
            excess_liquidity_usd = EXCLUDED.excess_liquidity_usd,
            open_positions   = EXCLUDED.open_positions,
            realized_pnl_usd = EXCLUDED.realized_pnl_usd,
            closed_trades    = EXCLUDED.closed_trades
    """.format(table=_result_table("backtest_account_curve"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=ACCOUNT_CURVE_INSERT_PAGE_SIZE)
    conn.commit()
    log.info("Wrote %d account-curve snapshots for run %d", len(points), run_id)


def _decimal_or_none(value: Optional[float], digits: int) -> Optional[Decimal]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return Decimal(str(round(float(value), digits)))


def _should_write_decision_event(event: DecisionEvent) -> bool:
    if DECISION_EVENT_MODE == "all":
        return True
    if DECISION_EVENT_MODE == "none":
        return False
    if event.decision_stage == "regime_risk":
        return True
    if DECISION_EVENT_MODE == "signals":
        if event.intent_passed or event.opened:
            return True
        return event.symbol is None and event.decision in {
            "skipped_day",
            "skipped_direction",
            "no_candidates",
        }
    if DECISION_EVENT_MODE == "summary":
        return event.opened or (
            event.symbol is None
            and event.decision in {"skipped_day", "skipped_direction", "no_candidates"}
        )
    raise ValueError(f"Unknown DECISION_EVENT_MODE: {DECISION_EVENT_MODE!r}")


def write_decision_events(
    conn: psycopg2.extensions.connection,
    events: list[DecisionEvent],
) -> None:
    if not events:
        return
    events = [event for event in events if _should_write_decision_event(event)]
    if not events:
        return

    rows = [
        (
            e.run_id,
            e.intent_date,
            e.as_of_ts,
            e.symbol,
            e.exchange,
            e.cik,
            e.direction,
            e.decision_stage,
            e.decision,
            e.reason_code,
            e.reason_text or None,
            e.intent_passed,
            e.opened,
            e.candidate_rank,
            e.intent_rank,
            e.world_regime_label or None,
            _decimal_or_none(e.world_regime_score, 2),
            e.valuation_label or None,
            e.sector or None,
            e.industry or None,
            _decimal_or_none(e.fundamental_score, 4),
            _decimal_or_none(e.mispricing_score, 4),
            _decimal_or_none(e.market_cap_m, 2),
            e.bar_count,
            e.min_bars,
            _decimal_or_none(e.intent_score, 4),
            e.intent_reason or None,
            e.entry_ts,
            _decimal_or_none(e.entry_price, 4),
            _decimal_or_none(e.stop_loss, 4),
            _decimal_or_none(e.take_profit, 4),
            _decimal_or_none(e.trailing_activation_price, 4),
            _decimal_or_none(e.trailing_distance_pct, 4),
            e.open_positions,
            e.max_open_positions,
            _decimal_or_none(e.account_equity, 2),
            _decimal_or_none(e.initial_margin, 2),
            _decimal_or_none(e.maintenance_margin, 2),
            _decimal_or_none(e.available_funds, 2),
            _decimal_or_none(e.excess_liquidity, 2),
            _decimal_or_none(e.required_initial_margin, 2),
            _decimal_or_none(e.required_maintenance_margin, 2),
            _decimal_or_none(e.available_funds_after, 2),
            _decimal_or_none(e.excess_liquidity_after, 2),
            _decimal_or_none(e.position_size_usd, 2),
            _decimal_or_none(e.shares, 6),
        )
        for e in events
    ]

    query = """
        INSERT INTO {table} (
            run_id, intent_date, as_of_ts, symbol, exchange, cik, direction,
            decision_stage, decision, reason_code, reason_text,
            intent_passed, opened, candidate_rank, intent_rank,
            world_regime_label, world_regime_score,
            valuation_label,
            sector, industry, fundamental_score, mispricing_score, market_cap_m,
            bar_count, min_bars, intent_score, intent_reason,
            entry_ts, entry_price, stop_loss, take_profit,
            trailing_activation_price, trailing_distance_pct,
            open_positions, max_open_positions,
            account_equity, initial_margin, maintenance_margin,
            available_funds, excess_liquidity,
            required_initial_margin, required_maintenance_margin,
            available_funds_after, excess_liquidity_after,
            position_size_usd, shares
        ) VALUES %s
    """.format(table=_result_table("backtest_decision_events"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=DECISION_EVENT_INSERT_PAGE_SIZE)
    conn.commit()


def update_run_summary(
    conn: psycopg2.extensions.connection,
    run_id: int,
    trades: list[ClosedTrade],
    final_equity: float,
    max_drawdown_pct: Optional[float] = None,
) -> None:
    wins      = [t for t in trades if t.pnl_usd > 0]
    losses    = [t for t in trades if t.pnl_usd < 0]
    breakevens= [t for t in trades if t.pnl_usd == 0]
    expired   = [t for t in trades if "MAX_HOLD" in t.outcome_status]

    win_rate  = len(wins) / len(trades) * 100.0 if trades else 0.0
    total_ret = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0
    avg_ret   = sum(t.return_pct for t in trades) / len(trades) if trades else 0.0
    avg_win   = sum(t.return_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t.return_pct for t in losses) / len(losses) if losses else 0.0
    margin_hours_usd = sum(t.margin_hours_usd for t in trades if math.isfinite(t.margin_hours_usd))
    total_pnl = sum(t.pnl_usd for t in trades)
    return_per_margin_hour_pct = (
        total_pnl / margin_hours_usd * 100.0 if margin_hours_usd > 0.0 else None
    )

    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss   = abs(sum(t.pnl_usd for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    if max_drawdown_pct is None:
        equity_series = [INITIAL_EQUITY] + [t.equity_after for t in trades]
        peak = equity_series[0]
        max_dd = 0.0
        for eq in equity_series:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100.0
            if dd > max_dd:
                max_dd = dd
    else:
        max_dd = max_drawdown_pct

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {_result_table("backtest_runs")} SET
                final_equity     = %s,
                total_trades     = %s,
                winning_trades   = %s,
                losing_trades    = %s,
                breakeven_trades = %s,
                expired_trades   = %s,
                win_rate_pct     = %s,
                total_return_pct = %s,
                margin_hours_usd = %s,
                return_per_margin_hour_pct = %s,
                max_drawdown_pct = %s,
                avg_return_pct   = %s,
                avg_win_pct      = %s,
                avg_loss_pct     = %s,
                profit_factor    = %s
            WHERE run_id = %s
            """,
            (
                round(final_equity, 2),
                len(trades), len(wins), len(losses), len(breakevens), len(expired),
                round(win_rate, 2), round(total_ret, 2),
                round(margin_hours_usd, 4),
                round(return_per_margin_hour_pct, 8) if return_per_margin_hour_pct is not None else None,
                round(max_dd, 2),
                round(avg_ret, 4), round(avg_win, 4), round(avg_loss, 4),
                round(profit_factor, 4) if profit_factor else None,
                run_id,
            ),
        )
    conn.commit()


def update_run_duration(
    conn: psycopg2.extensions.connection,
    run_id: int,
    duration_seconds: float,
) -> None:
    if not math.isfinite(duration_seconds) or duration_seconds < 0:
        raise ValueError(f"Invalid run duration seconds: {duration_seconds!r}")

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {_result_table("backtest_runs")} SET
                run_duration_seconds = %s
            WHERE run_id = %s
            """,
            (round(duration_seconds, 3), run_id),
        )
    conn.commit()
    log.info("Updated run %d duration %.3f seconds", run_id, duration_seconds)
