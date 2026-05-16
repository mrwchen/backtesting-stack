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

log = logging.getLogger(__name__)

def create_run(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    long_max_hold_days: float,
    short_max_hold_days: float,
    tp1_close_ratio: float,
    notes: Optional[str] = None,
) -> int:
    run_notes = _build_run_notes(notes)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {_result_table("backtest_runs")} (
                start_date, end_date, notes, run_label, model_file,
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
                long_max_score, short_min_score,
                long_min_fundamental, short_max_fundamental, min_market_cap_m,
                long_min_pullback, long_max_pullback, long_ideal_pullback, long_max_rsi,
                short_min_bounce, short_max_bounce, short_ideal_bounce, short_min_rsi, short_max_rsi,
                long_sl_buffer, short_sl_buffer,
                long_tp1_pct, long_tp2_pct, short_tp1_pct, short_tp2_pct,
                long_max_hold_days, short_max_hold_days,
                tp1_close_ratio
            ) VALUES (
                %s, %s, %s, to_char(NOW() AT TIME ZONE %s, 'YYYY-MM-DD HH24:MI'), %s,
                %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s
            ) RETURNING run_id
            """,
            (
                START_DATE, END_DATE, run_notes, RUN_LABEL_TZ, runtime.CURRENT_MODEL_FILE,
                ACCOUNT_PROFILE,
                INITIAL_EQUITY, RISK_PER_TRADE_PCT, MAX_OPEN_POSITIONS,
                MARGIN_REQUIREMENT_PCT, PS_MARGIN_STOP_OUT_LEVEL_PCT, PS_MIN_ENTRY_MARGIN_LEVEL_PCT,
                IBKR_LONG_INITIAL_MARGIN_PCT, IBKR_LONG_MAINTENANCE_MARGIN_PCT,
                IBKR_SHORT_INITIAL_MARGIN_PCT, IBKR_SHORT_MAINTENANCE_MARGIN_PCT,
                ALLOW_FRACTIONAL_SHARES, SPREAD_BPS, SLIPPAGE_BPS,
                COMMISSION_PER_ORDER_USD, COMMISSION_PER_SHARE_USD,
                COMMISSION_MIN_PER_ORDER_USD, COMMISSION_MAX_PCT,
                COMMISSION_BPS, MARGIN_FINANCING_RATE_PCT,
                PS_SHARE_CFD_ARR_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
                PS_SHARE_CFD_ADMIN_FEE_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
                PS_SHARE_CFD_SHORT_BORROW_RATE_PCT if ACCOUNT_PROFILE == "ps_acc" else None,
                PS_SHARE_CFD_OVERNIGHT_DAY_COUNT if ACCOUNT_PROFILE == "ps_acc" else None,
                ENTRY_WINDOW_ENABLED, ENTRY_WINDOW_TZ, ENTRY_WINDOW_START, ENTRY_WINDOW_END,
                cfg.long_max_score, cfg.short_min_score,
                cfg.long_min_fundamental, cfg.short_max_fundamental, MIN_MARKET_CAP_M,
                cfg.long_min_pullback, cfg.long_max_pullback, cfg.long_ideal_pullback, cfg.long_max_rsi,
                cfg.short_min_bounce, cfg.short_max_bounce, cfg.short_ideal_bounce, cfg.short_min_rsi, cfg.short_max_rsi,
                cfg.long_sl_buffer, cfg.short_sl_buffer,
                cfg.long_tp1_pct, cfg.long_tp2_pct, cfg.short_tp1_pct, cfg.short_tp2_pct,
                long_max_hold_days, short_max_hold_days,
                tp1_close_ratio,
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created run_id=%d model_file=%s account_profile=%s", run_id, runtime.CURRENT_MODEL_FILE, ACCOUNT_PROFILE)
    return run_id


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
        s = p.signal
        rows.append((
            run_id,
            p.entry_date,
            p.symbol,
            p.direction,
            p.world_regime_label or None,
            Decimal(str(round(p.world_regime_score, 2))) if p.world_regime_score else None,
            p.valuation_label or None,
            Decimal(str(round(s.fundamental_score, 2))),
            Decimal(str(round(s.entry_score, 4))),
            Decimal(str(round(s.combined_score, 4))),
            Decimal(str(round(p.entry_price, 4))),
            Decimal(str(round(p.stop_loss, 4))),
            Decimal(str(round(p.take_profit_1, 4))),
            Decimal(str(round(p.take_profit_2, 4))),
            Decimal(str(round(s.pullback_pct, 2))),
            Decimal(str(round(s.rsi_1h, 2))),
            Decimal(str(round(s.volume_ratio, 3))),
            s.entry_reason,
            Decimal(str(round(p.position_size_usd, 2))),
            Decimal(str(round(p.shares, 6))),
            Decimal(str(round(p.margin_used, 2))),
            Decimal(str(round(p.maintenance_margin_used, 2))),
            Decimal(str(round(p.equity_before, 2))),
            t.outcome_status,
            Decimal(str(round(t.outcome_price, 4))),
            t.outcome_date,
            t.outcome_bars,
            t.tp1_hit,
            Decimal(str(round(t.return_pct, 4))),
            Decimal(str(round(t.pnl_usd, 2))),
            Decimal(str(round(t.equity_after, 2))),
            p.entry_ts,
            t.tp1_exit_ts,
            t.exit_ts,
        ))

    query = """
        INSERT INTO {table} (
            run_id, signal_date, symbol, direction,
            world_regime_label, world_regime_score, valuation_label,
            fundamental_score, entry_score, combined_score,
            entry_price, stop_loss, take_profit_1, take_profit_2,
            pullback_pct, rsi_1h, volume_ratio, entry_reason,
            position_size_usd, shares, margin_used, maintenance_margin_used, equity_before,
            outcome_status, outcome_price, outcome_date, outcome_bars,
            tp1_hit, return_pct, pnl_usd, equity_after,
            entry_ts, tp1_exit_ts, exit_ts
        ) VALUES %s
        ON CONFLICT (run_id, signal_date, symbol) DO UPDATE SET
            world_regime_score = EXCLUDED.world_regime_score,
            outcome_status = EXCLUDED.outcome_status,
            outcome_price  = EXCLUDED.outcome_price,
            pnl_usd        = EXCLUDED.pnl_usd,
            equity_after   = EXCLUDED.equity_after,
            entry_ts       = EXCLUDED.entry_ts,
            tp1_exit_ts    = EXCLUDED.tp1_exit_ts,
            exit_ts        = EXCLUDED.exit_ts
    """.format(table=_result_table("backtest_trades"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=200)
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
        execute_values(cur, query, rows, page_size=500)
    conn.commit()
    log.info("Wrote %d account-curve snapshots for run_id=%d", len(points), run_id)


def _decimal_or_none(value: Optional[float], digits: int) -> Optional[Decimal]:
    if value is None:
        return None
    if not math.isfinite(float(value)):
        return None
    return Decimal(str(round(float(value), digits)))


def write_decision_events(
    conn: psycopg2.extensions.connection,
    events: list[DecisionEvent],
) -> None:
    if not events:
        return

    rows = [
        (
            e.run_id,
            e.signal_date,
            e.as_of_ts,
            e.symbol,
            e.direction,
            e.decision_stage,
            e.decision,
            e.reason_code,
            e.reason_text or None,
            e.signal_passed,
            e.opened,
            e.candidate_rank,
            e.signal_rank,
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
            e.entry_ts,
            _decimal_or_none(e.entry_price, 4),
            _decimal_or_none(e.stop_loss, 4),
            _decimal_or_none(e.take_profit_1, 4),
            _decimal_or_none(e.take_profit_2, 4),
            _decimal_or_none(e.pullback_pct, 2),
            _decimal_or_none(e.rsi_1h, 2),
            _decimal_or_none(e.volume_ratio, 3),
            _decimal_or_none(e.entry_score, 4),
            _decimal_or_none(e.combined_score, 4),
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
            run_id, signal_date, as_of_ts, symbol, direction,
            decision_stage, decision, reason_code, reason_text,
            signal_passed, opened, candidate_rank, signal_rank,
            world_regime_label, world_regime_score, valuation_label,
            sector, industry, fundamental_score, mispricing_score, market_cap_m,
            bar_count, min_bars, entry_ts, entry_price, stop_loss,
            take_profit_1, take_profit_2, pullback_pct, rsi_1h, volume_ratio,
            entry_score, combined_score, open_positions, max_open_positions,
            account_equity, initial_margin, maintenance_margin,
            available_funds, excess_liquidity,
            required_initial_margin, required_maintenance_margin,
            available_funds_after, excess_liquidity_after,
            position_size_usd, shares
        ) VALUES %s
    """.format(table=_result_table("backtest_decision_events"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=500)
    conn.commit()


def update_run_summary(
    conn: psycopg2.extensions.connection,
    run_id: int,
    trades: list[ClosedTrade],
    final_equity: float,
) -> None:
    if not trades:
        return

    wins      = [t for t in trades if t.pnl_usd > 0]
    losses    = [t for t in trades if t.pnl_usd < 0]
    breakevens= [t for t in trades if t.pnl_usd == 0]
    expired   = [t for t in trades if "MAX_HOLD" in t.outcome_status]

    win_rate  = len(wins) / len(trades) * 100.0 if trades else 0.0
    total_ret = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0
    avg_ret   = sum(t.return_pct for t in trades) / len(trades) if trades else 0.0
    avg_win   = sum(t.return_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t.return_pct for t in losses) / len(losses) if losses else 0.0

    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss   = abs(sum(t.pnl_usd for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    # Max drawdown (peak-to-trough on equity_after series)
    equity_series = [INITIAL_EQUITY] + [t.equity_after for t in trades]
    peak = equity_series[0]
    max_dd = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

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
                round(win_rate, 2), round(total_ret, 2), round(max_dd, 2),
                round(avg_ret, 4), round(avg_win, 4), round(avg_loss, 4),
                round(profit_factor, 4) if profit_factor else None,
                run_id,
            ),
        )
    conn.commit()
