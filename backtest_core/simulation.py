"""Core point-in-time portfolio simulation loop."""

import logging
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2

from backtest_shared import Signal, SignalEvaluation
from . import runtime
from .broker import (
    _account_snapshot_values,
    _active_maintenance_margin_used,
    _active_margin_used,
    _enforce_account_margin_liquidation,
    _make_trade,
    _margin_level_pct,
    _pnl_long,
    _pnl_short,
    _remove_position_by_identity,
    calc_position,
    initial_stop_cash_risk,
)
from .config import *
from .entities import AccountCurvePoint, ClosedTrade, DecisionEvent, OpenPosition, PortfolioEvent
from .market_data import (
    _day_close_ts,
    _day_signal_cutoff_ts,
    _ensure_utc_ts,
    get_bars_range,
    get_bars_range_through,
    get_candidates,
    get_next_bar_open,
    get_trading_days,
    get_world_regime,
    _is_stop_loss_active,
    _is_in_sl_tp_window,
    load_recent_bars_for_identities,
    log_cache_stats,
    preload_identity_bars,
    preload_candidate_timelines,
)
from .model_loader import get_model_module
from .monte_carlo import run_monte_carlo
from .policy import (
    candidate_policy_kwargs,
    direction_filter_negative_earnings,
    direction_max_positions,
    direction_risk_multiplier,
    regime_exposure_for_score,
)
from .persistence import (
    create_run,
    update_run_duration,
    update_run_summary,
    write_account_curve,
    write_decision_events,
    write_trades,
)
from .trade_levels import apply_trade_levels, common_stop_required_lookback

log = logging.getLogger(__name__)

DIRECTIONS = ("LONG", "SHORT")


def _bar_lookback_limit(model: Any, cfg: Any) -> int:
    required_bar_lookback = getattr(model, "required_bar_lookback", None)
    common_stop_lookback = common_stop_required_lookback()
    if callable(required_bar_lookback):
        return max(1, int(cfg.min_bars), int(required_bar_lookback(cfg)), common_stop_lookback)
    return max(1, int(cfg.min_bars) + int(cfg.price_lookback_bars), common_stop_lookback)


def _direction_open_count(open_positions: list[OpenPosition], direction: str) -> int:
    return sum(1 for pos in open_positions if pos.direction == direction)


def _signal_event_key(signal: Signal) -> tuple[str, tuple[str, str, int]]:
    return (signal.direction, signal.identity_key)


def _max_drawdown_pct_from_equity(equity_values: list[float]) -> float:
    if not equity_values:
        return 0.0
    peak = equity_values[0]
    max_dd = 0.0
    for eq in equity_values:
        if eq > peak:
            peak = eq
        if peak <= 0:
            continue
        dd = (peak - eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _direction_report(trades: list[ClosedTrade], direction: str) -> dict:
    direction_trades = [t for t in trades if t.position.direction == direction]
    wins = [t for t in direction_trades if t.pnl_usd > 0]
    losses = [t for t in direction_trades if t.pnl_usd < 0]
    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))

    r_values = []
    for trade in direction_trades:
        risk_usd = initial_stop_cash_risk(trade.position)
        if risk_usd > 0.0:
            r_values.append(trade.pnl_usd / risk_usd)

    return {
        "trades": len(direction_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": len(wins) / len(direction_trades) * 100.0 if direction_trades else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "avg_r": sum(r_values) / len(r_values) if r_values else None,
        "pnl_usd": sum(t.pnl_usd for t in direction_trades),
    }


def _format_optional(value: Optional[float], fmt: str) -> str:
    return fmt % value if value is not None else "N/A"


def _long_stop_fill_price(stop_price: float, bar_open: object) -> float:
    return min(stop_price, float(bar_open))


def _short_stop_fill_price(stop_price: float, bar_open: object) -> float:
    return max(stop_price, float(bar_open))


def _middle_low_reaches(open_: float, close: float, low: float, level: float) -> bool:
    return low <= level and low < min(open_, close)


def _middle_high_reaches(open_: float, close: float, high: float, level: float) -> bool:
    return high >= level and high > max(open_, close)


def _long_stop_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    price: float,
    bar_date: date,
    total_bars: int,
    tp1_hit: bool,
    tp1_price: Optional[float],
    equity: float,
    ts: datetime,
    tp1_exit_ts: Optional[datetime],
) -> ClosedTrade:
    if tp1_hit:
        pnl = _pnl_long(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
        status = "HIT_TP1_THEN_BE"
    else:
        pnl = _pnl_long(pos, price, price, split_exits=False)
        status = "HIT_SL"
    return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)


def _short_stop_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    price: float,
    bar_date: date,
    total_bars: int,
    tp1_hit: bool,
    tp1_price: Optional[float],
    equity: float,
    ts: datetime,
    tp1_exit_ts: Optional[datetime],
) -> ClosedTrade:
    if tp1_hit:
        pnl = _pnl_short(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
        status = "HIT_TP1_THEN_BE"
    else:
        pnl = _pnl_short(pos, price, price, split_exits=False)
        status = "HIT_SL"
    return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)


def _long_tp2_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    bar_date: date,
    total_bars: int,
    tp1_price: Optional[float],
    equity: float,
    ts: datetime,
    tp1_exit_ts: Optional[datetime],
) -> ClosedTrade:
    price = pos.take_profit_2
    pnl = _pnl_long(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
    return _make_trade(conn, pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)


def _short_tp2_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    bar_date: date,
    total_bars: int,
    tp1_price: Optional[float],
    equity: float,
    ts: datetime,
    tp1_exit_ts: Optional[datetime],
) -> ClosedTrade:
    price = pos.take_profit_2
    pnl = _pnl_short(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
    return _make_trade(conn, pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)


def _simulate_long_intrabar(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    ts: datetime,
    bar_date: date,
    total_bars: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    stop_loss_active: bool,
    sl_tp_active: bool,
    tp1_hit: bool,
    tp1_price: Optional[float],
    tp1_exit_ts: Optional[datetime],
    effective_sl: float,
    equity: float,
) -> tuple[Optional[ClosedTrade], bool, Optional[float], Optional[datetime], float]:
    # Open is known to be first. Favourable gaps can reach TP before any unknown low.
    if stop_loss_active and open_ <= effective_sl:
        price = _long_stop_fill_price(effective_sl, open_)
        return _long_stop_trade(conn, pos, price, bar_date, total_bars, tp1_hit, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if sl_tp_active:
        if tp1_hit:
            if open_ >= pos.take_profit_2:
                return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        elif open_ >= pos.take_profit_1:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            if open_ >= pos.take_profit_2:
                return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    # High/low ordering between open and close is unknown. Resolve conflicts with SL first.
    if tp1_hit:
        stop_mid = stop_loss_active and _middle_low_reaches(open_, close, low, effective_sl)
        tp2_mid = sl_tp_active and _middle_high_reaches(open_, close, high, pos.take_profit_2)
        if stop_mid:
            return _long_stop_trade(conn, pos, effective_sl, bar_date, total_bars, True, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        if tp2_mid:
            return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
    else:
        stop_mid = stop_loss_active and _middle_low_reaches(open_, close, low, effective_sl)
        tp1_mid = sl_tp_active and _middle_high_reaches(open_, close, high, pos.take_profit_1)
        if stop_mid:
            return _long_stop_trade(conn, pos, effective_sl, bar_date, total_bars, False, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        if tp1_mid:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            be_mid = stop_loss_active and _middle_low_reaches(open_, close, low, effective_sl)
            tp2_mid = sl_tp_active and _middle_high_reaches(open_, close, high, pos.take_profit_2)
            if be_mid:
                return _long_stop_trade(conn, pos, effective_sl, bar_date, total_bars, True, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
            if tp2_mid:
                return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    # Close is known to be last.
    if stop_loss_active and close <= effective_sl:
        return _long_stop_trade(conn, pos, effective_sl, bar_date, total_bars, tp1_hit, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if sl_tp_active:
        if tp1_hit:
            if close >= pos.take_profit_2:
                return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        elif close >= pos.take_profit_1:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            if close >= pos.take_profit_2:
                return _long_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    return None, tp1_hit, tp1_price, tp1_exit_ts, effective_sl


def _simulate_short_intrabar(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    ts: datetime,
    bar_date: date,
    total_bars: int,
    open_: float,
    high: float,
    low: float,
    close: float,
    stop_loss_active: bool,
    sl_tp_active: bool,
    tp1_hit: bool,
    tp1_price: Optional[float],
    tp1_exit_ts: Optional[datetime],
    effective_sl: float,
    equity: float,
) -> tuple[Optional[ClosedTrade], bool, Optional[float], Optional[datetime], float]:
    if stop_loss_active and open_ >= effective_sl:
        price = _short_stop_fill_price(effective_sl, open_)
        return _short_stop_trade(conn, pos, price, bar_date, total_bars, tp1_hit, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if sl_tp_active:
        if tp1_hit:
            if open_ <= pos.take_profit_2:
                return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        elif open_ <= pos.take_profit_1:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            if open_ <= pos.take_profit_2:
                return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if tp1_hit:
        stop_mid = stop_loss_active and _middle_high_reaches(open_, close, high, effective_sl)
        tp2_mid = sl_tp_active and _middle_low_reaches(open_, close, low, pos.take_profit_2)
        if stop_mid:
            return _short_stop_trade(conn, pos, effective_sl, bar_date, total_bars, True, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        if tp2_mid:
            return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
    else:
        stop_mid = stop_loss_active and _middle_high_reaches(open_, close, high, effective_sl)
        tp1_mid = sl_tp_active and _middle_low_reaches(open_, close, low, pos.take_profit_1)
        if stop_mid:
            return _short_stop_trade(conn, pos, effective_sl, bar_date, total_bars, False, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        if tp1_mid:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            be_mid = stop_loss_active and _middle_high_reaches(open_, close, high, effective_sl)
            tp2_mid = sl_tp_active and _middle_low_reaches(open_, close, low, pos.take_profit_2)
            if be_mid:
                return _short_stop_trade(conn, pos, effective_sl, bar_date, total_bars, True, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
            if tp2_mid:
                return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if stop_loss_active and close >= effective_sl:
        return _short_stop_trade(conn, pos, effective_sl, bar_date, total_bars, tp1_hit, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    if sl_tp_active:
        if tp1_hit:
            if close <= pos.take_profit_2:
                return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl
        elif close <= pos.take_profit_1:
            tp1_hit = True
            tp1_price = pos.take_profit_1
            tp1_exit_ts = ts
            effective_sl = pos.entry_price
            if close <= pos.take_profit_2:
                return _short_tp2_trade(conn, pos, bar_date, total_bars, tp1_price, equity, ts, tp1_exit_ts), tp1_hit, tp1_price, tp1_exit_ts, effective_sl

    return None, tp1_hit, tp1_price, tp1_exit_ts, effective_sl


def simulate_outcome(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    up_to_ts: datetime,
    equity: float,
) -> Optional[ClosedTrade]:
    """
    Check whether pos has closed by up_to_ts.
    Returns ClosedTrade if closed, None if still open.

    TP logic: position is split 50/50 between TP1 and TP2.
    After TP1 hit, SL moves to entry (breakeven).

    Incremental: each call only scans bars newer than pos.last_bar_ts and
    resumes from the TP1/SL state stored on pos, making the loop O(N total)
    across all daily calls rather than O(N²).
    """
    up_to_ts = _ensure_utc_ts(up_to_ts)
    after_ts = pos.last_bar_ts if pos.last_bar_ts is not None else pos.entry_ts - timedelta(microseconds=1)
    bars = get_bars_range_through(conn, pos.identity_key, after_ts, up_to_ts)
    if not bars:
        return None

    tp1_hit = pos.tp1_hit
    tp1_price = pos.tp1_price
    tp1_exit_ts = pos.tp1_exit_ts
    effective_sl = pos.effective_sl
    is_long = pos.direction == "LONG"

    for bar_idx, (ts, open_, high, low, close) in enumerate(bars):
        bar_date = ts.date() if hasattr(ts, "date") else ts
        total_bars = pos.bars_processed + bar_idx + 1
        sl_tp_active = _is_in_sl_tp_window(ts)
        stop_loss_active = _is_stop_loss_active(ts)

        if is_long:
            trade, tp1_hit, tp1_price, tp1_exit_ts, effective_sl = _simulate_long_intrabar(
                conn,
                pos,
                ts,
                bar_date,
                total_bars,
                float(open_),
                float(high),
                float(low),
                float(close),
                stop_loss_active,
                sl_tp_active,
                tp1_hit,
                tp1_price,
                tp1_exit_ts,
                effective_sl,
                equity,
            )
        else:
            trade, tp1_hit, tp1_price, tp1_exit_ts, effective_sl = _simulate_short_intrabar(
                conn,
                pos,
                ts,
                bar_date,
                total_bars,
                float(open_),
                float(high),
                float(low),
                float(close),
                stop_loss_active,
                sl_tp_active,
                tp1_hit,
                tp1_price,
                tp1_exit_ts,
                effective_sl,
                equity,
            )
        if trade is not None:
            return trade

        if ts >= pos.valid_until:
            price = float(close)
            if is_long:
                pnl = _pnl_long(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            else:
                pnl = _pnl_short(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            status = "MAX_HOLD_TP1" if tp1_hit else "MAX_HOLD"
            return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

    # Still open — persist incremental state for the next day's call
    pos.tp1_hit = tp1_hit
    pos.tp1_price = tp1_price
    pos.tp1_exit_ts = tp1_exit_ts
    pos.effective_sl = effective_sl
    pos.last_bar_ts = bars[-1][0]
    pos.bars_processed += len(bars)
    return None

def run_backtest(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    notes: Optional[str] = None,
) -> tuple[int, dict]:
    run_started = _time.perf_counter()
    run_id = create_run(conn, cfg, notes)

    equity: float = INITIAL_EQUITY
    open_positions: list[OpenPosition] = []
    closed_trades: list[ClosedTrade] = []
    account_curve: list[AccountCurvePoint] = []
    account_curve_seq = 0
    decision_event_buffer: list[DecisionEvent] = []

    def flush_decision_events(force: bool = False) -> None:
        if not decision_event_buffer:
            return
        if not force and len(decision_event_buffer) < DECISION_EVENT_FLUSH_BATCH_SIZE:
            return
        events_to_write = list(decision_event_buffer)
        decision_event_buffer.clear()
        write_decision_events(conn, events_to_write)

    def buffer_decision_events(events: list[DecisionEvent]) -> None:
        if not events:
            return
        decision_event_buffer.extend(events)
        flush_decision_events()

    def record_account_curve(as_of_ts: datetime, active_positions: list[OpenPosition]) -> None:
        nonlocal account_curve_seq
        account_curve_seq += 1
        as_of_ts = _ensure_utc_ts(as_of_ts)
        snapshot = _account_snapshot_values(
            conn,
            active_positions,
            equity,
            as_of_ts,
        )
        account_curve.append(AccountCurvePoint(
            run_id=run_id,
            ts=as_of_ts,
            trade_date=as_of_ts.date(),
            seq_in_run=account_curve_seq,
            balance_usd=round(equity, 2),
            open_pnl_usd=round(snapshot.open_pnl, 2),
            equity_usd=round(snapshot.equity_with_loan_value, 2),
            initial_margin_usd=round(snapshot.initial_margin, 2),
            maintenance_margin_usd=round(snapshot.maintenance_margin, 2),
            available_funds_usd=round(snapshot.available_funds, 2),
            excess_liquidity_usd=round(snapshot.excess_liquidity, 2),
            open_positions=len(active_positions),
            realized_pnl_usd=round(equity - INITIAL_EQUITY, 2),
            closed_trades=len(closed_trades),
        ))

    def apply_position_events_through(
        positions: list[OpenPosition],
        end_ts: datetime,
    ) -> tuple[list[OpenPosition], int, float]:
        nonlocal equity
        end_ts = _ensure_utc_ts(end_ts)
        portfolio_events: list[PortfolioEvent] = []
        closed_count = 0
        realized_pnl = 0.0
        if positions:
            preload_identity_bars(
                conn,
                [pos.identity_key for pos in positions],
                end_ts,
                batch_size=BAR_CACHE_BATCH_SIZE,
                log_batches=False,
            )

        for pos in positions:
            before_tp1_hit = pos.tp1_hit
            before_tp1_price = pos.tp1_price
            before_tp1_exit_ts = pos.tp1_exit_ts
            before_effective_sl = pos.effective_sl
            trade = simulate_outcome(conn, pos, end_ts, equity)
            if trade is not None:
                close_ts = trade.exit_ts or end_ts
                if not before_tp1_hit and trade.tp1_hit and trade.tp1_exit_ts and trade.tp1_exit_ts < close_ts:
                    portfolio_events.append(PortfolioEvent(
                        ts=trade.tp1_exit_ts,
                        priority=0,
                        kind="tp1",
                        position=pos,
                    ))
                portfolio_events.append(PortfolioEvent(
                    ts=close_ts,
                    priority=1,
                    kind="close",
                    position=pos,
                    trade=trade,
                ))
                continue

            if not before_tp1_hit and pos.tp1_hit:
                tp1_event_ts = pos.tp1_exit_ts or end_ts
                pos.tp1_hit = before_tp1_hit
                pos.tp1_price = before_tp1_price
                pos.tp1_exit_ts = before_tp1_exit_ts
                pos.effective_sl = before_effective_sl
                portfolio_events.append(PortfolioEvent(
                    ts=tp1_event_ts,
                    priority=0,
                    kind="tp1",
                    position=pos,
                ))

        active_positions = list(positions)
        for event in sorted(portfolio_events, key=lambda e: (_ensure_utc_ts(e.ts), e.priority)):
            if event.kind == "tp1":
                event.position.tp1_hit = True
                event.position.tp1_price = event.position.take_profit_1
                event.position.tp1_exit_ts = _ensure_utc_ts(event.ts)
                event.position.effective_sl = event.position.entry_price
                record_account_curve(event.ts, active_positions)
                continue

            if event.kind == "close" and event.trade is not None:
                _remove_position_by_identity(active_positions, event.position)
                event.trade.equity_after = round(equity + event.trade.pnl_usd, 2)
                equity = event.trade.equity_after
                closed_trades.append(event.trade)
                closed_count += 1
                realized_pnl += event.trade.pnl_usd
                log.debug("Closed %-6s %s %s pnl %.0f balance %.0f",
                          event.position.symbol, event.position.direction, event.trade.outcome_status,
                          event.trade.pnl_usd, equity)
                record_account_curve(event.ts, active_positions)

        liquidation_trades, equity = _enforce_account_margin_liquidation(
            conn,
            active_positions,
            equity,
            end_ts,
        )
        if liquidation_trades:
            closed_trades.extend(liquidation_trades)
            closed_count += len(liquidation_trades)
            realized_pnl += sum(t.pnl_usd for t in liquidation_trades)
            record_account_curve(end_ts, active_positions)

        return active_positions, closed_count, realized_pnl

    trading_days = get_trading_days(conn, START_DATE, END_DATE)
    log.info("Trading days to simulate: %d (%s → %s)", len(trading_days), START_DATE, END_DATE)
    if trading_days:
        preload_candidate_timelines(
            conn,
            DIRECTIONS,
            **candidate_policy_kwargs(),
            source_table=SOURCE_FUNDAMENTAL_SCORES_TABLE,
            as_of_date=trading_days[0],
            as_of_ts=_day_signal_cutoff_ts(trading_days[0]),
            pepperstone_table=PS_TRADABLE_SYMBOLS_TABLE,
            required_currency="USD" if REQUIRE_USD_FUNDAMENTALS else None,
            allow_rebuilt_historical_fundamentals=ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS,
            filter_negative_earnings_by_direction={
                direction: direction_filter_negative_earnings(direction)
                for direction in DIRECTIONS
            },
            ibkr_margin_table=IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
        )
    record_account_curve(datetime.combine(START_DATE, datetime.min.time(), tzinfo=timezone.utc), open_positions)

    # Diagnostic counters
    days_no_regime = 0
    days_no_active_budget = 0
    days_no_candidates = 0
    days_no_signals    = 0
    days_with_signals  = 0

    for day_idx, day in enumerate(trading_days, start=1):
        log_progress_today = day_idx == 1 or day_idx == len(trading_days) or day_idx % PROGRESS_LOG_EVERY_DAYS == 0
        if log_progress_today:
            log.info(
                "Day %d/%d %s starting model %s open positions %d closed trades %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                len(open_positions),
                len(closed_trades),
            )
            log_cache_stats(f"day_start {day_idx}/{len(trading_days)} {day}")

        # ── 1. Apply open-position state changes up to entry decision time ──
        closed_today = 0
        day_pnl = 0.0
        day_end_ts = _day_signal_cutoff_ts(day)
        day_close_ts = _day_close_ts(day)
        open_positions, closed_before_entry, pnl_before_entry = apply_position_events_through(open_positions, day_end_ts)
        closed_today += closed_before_entry
        day_pnl += pnl_before_entry

        # ── 2. Generate signals for today ───────────────────────────────────
        regime = get_world_regime(conn, source_table=SOURCE_WORLD_REGIME_TABLE, as_of_date=day)
        if not regime:
            days_no_regime += 1
            buffer_decision_events([DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="no_regime",
                reason_text="No world-regime row was available for this trading day.",
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(open_positions, day_close_ts)
            closed_today += closed_after_entry
            day_pnl += pnl_after_entry
            record_account_curve(day_close_ts, open_positions)
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s no regime, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        regime_bucket, regime_exposure = regime_exposure_for_score(regime.score)
        if log_progress_today:
            log.info(
                "Regime exposure day %d/%d %s model %s bucket %s score %.1f long risk %.2f short risk %.2f max long %d max short %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                regime_bucket,
                regime.score,
                direction_risk_multiplier(regime_exposure, "LONG"),
                direction_risk_multiplier(regime_exposure, "SHORT"),
                direction_max_positions(regime_exposure, "LONG"),
                direction_max_positions(regime_exposure, "SHORT"),
            )

        if all(
            direction_risk_multiplier(regime_exposure, direction) <= 0.0
            and direction_max_positions(regime_exposure, direction) <= 0
            for direction in DIRECTIONS
        ):
            days_no_active_budget += 1
            buffer_decision_events([DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="no_regime_exposure_budget",
                reason_text=f"Regime bucket {regime_bucket} assigned zero risk and zero max positions to both directions.",
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(open_positions, day_close_ts)
            closed_today += closed_after_entry
            day_pnl += pnl_after_entry
            record_account_curve(day_close_ts, open_positions)
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s regime bucket %s had no exposure budget, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, regime_bucket, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        model = get_model_module()
        signals_by_direction: dict[str, list[Signal]] = {direction: [] for direction in DIRECTIONS}
        signals: list[Signal] = []
        decision_events: list[DecisionEvent] = []
        signal_events: dict[tuple[str, tuple[str, str, int]], DecisionEvent] = {}
        skipped_no_bars = 0
        total_candidates = 0
        candidate_counts: dict[str, int] = {direction: 0 for direction in DIRECTIONS}
        signal_counts: dict[str, int] = {direction: 0 for direction in DIRECTIONS}

        for direction in DIRECTIONS:
            direction_risk = direction_risk_multiplier(regime_exposure, direction)
            direction_cap = direction_max_positions(regime_exposure, direction)
            if direction_risk <= 0.0 and direction_cap <= 0:
                decision_events.append(DecisionEvent(
                    run_id=run_id,
                    signal_date=day,
                    as_of_ts=day_end_ts,
                    symbol=None,
                    exchange=None,
                    cik=None,
                    direction=direction,
                    decision_stage="regime_filter",
                    decision="skipped_direction",
                    reason_code="regime_direction_disabled",
                    reason_text=f"Regime bucket {regime_bucket} assigned zero risk and zero max {direction.lower()} positions.",
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    open_positions=len(open_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                ))
                continue

            if log_progress_today:
                log.info(
                    "Candidate query starting day %d/%d %s model %s direction %s bucket %s cutoff %s",
                    day_idx,
                    len(trading_days),
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    regime_bucket,
                    day_end_ts,
                )
            candidate_started = _time.perf_counter()
            candidates = get_candidates(
                conn,
                direction,
                **candidate_policy_kwargs(),
                source_table=SOURCE_FUNDAMENTAL_SCORES_TABLE,
                as_of_date=day,
                as_of_ts=day_end_ts,
                pepperstone_table=PS_TRADABLE_SYMBOLS_TABLE,
                required_currency="USD" if REQUIRE_USD_FUNDAMENTALS else None,
                allow_rebuilt_historical_fundamentals=ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS,
                filter_negative_earnings=direction_filter_negative_earnings(direction),
                ibkr_margin_table=IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
            )
            candidate_elapsed = _time.perf_counter() - candidate_started
            candidate_counts[direction] = len(candidates)
            total_candidates += len(candidates)
            if log_progress_today or candidate_elapsed >= 5.0:
                log.info(
                    "Candidate query complete day %d/%d %s model %s direction %s bucket %s found %d candidates in %.1f s",
                    day_idx,
                    len(trading_days),
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    regime_bucket,
                    len(candidates),
                    candidate_elapsed,
                )

            if not candidates:
                decision_events.append(DecisionEvent(
                    run_id=run_id,
                    signal_date=day,
                    as_of_ts=day_end_ts,
                    symbol=None,
                    exchange=None,
                    cik=None,
                    direction=direction,
                    decision_stage="candidate_filter",
                    decision="no_candidates",
                    reason_code="no_candidates_after_fundamental_filters",
                    reason_text="No symbols passed the point-in-time fundamental, currency, market-cap and broker filters.",
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    open_positions=len(open_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                ))
                continue

            candidate_identities = [fundamental.identity_key for fundamental in candidates]
            bar_lookback_limit = _bar_lookback_limit(model, cfg)
            bar_load_started = _time.perf_counter()
            recent_bars_by_identity = load_recent_bars_for_identities(
                conn,
                candidate_identities,
                bar_lookback_limit,
                day_end_ts,
                batch_size=BAR_CACHE_BATCH_SIZE,
                log_batches=log_progress_today,
            )
            loaded_bar_rows = sum(len(bars) for bars in recent_bars_by_identity.values())
            bar_load_elapsed = _time.perf_counter() - bar_load_started
            if log_progress_today or bar_load_elapsed >= 5.0:
                log.info(
                    "Recent bar load complete day %d/%d %s model %s direction %s loaded %d rows for %d candidates limit %d through %s in %.1f s",
                    day_idx,
                    len(trading_days),
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    loaded_bar_rows,
                    len(candidate_identities),
                    bar_lookback_limit,
                    day_end_ts,
                    bar_load_elapsed,
                )

            compute_fn = model.compute_long_signal if direction == "LONG" else model.compute_short_signal
            evaluate_fn = getattr(
                model,
                "evaluate_long_signal" if direction == "LONG" else "evaluate_short_signal",
                None,
            )

            for candidate_rank, fundamental in enumerate(candidates, start=1):
                bars = recent_bars_by_identity.get(fundamental.identity_key, [])
                if len(bars) < cfg.min_bars:
                    skipped_no_bars += 1
                    decision_events.append(DecisionEvent(
                        run_id=run_id,
                        signal_date=day,
                        as_of_ts=day_end_ts,
                        symbol=fundamental.symbol,
                        exchange=fundamental.exchange,
                        cik=fundamental.cik,
                        direction=direction,
                        decision_stage="bar_load",
                        decision="rejected",
                        reason_code="insufficient_bars",
                        reason_text=f"Only {len(bars)} cached 1h bars were available; model requires at least {cfg.min_bars}.",
                        candidate_rank=candidate_rank,
                        world_regime_label=regime.label,
                        world_regime_score=regime.score,
                        valuation_label=fundamental.valuation_label,
                        sector=fundamental.sector,
                        industry=fundamental.industry,
                        fundamental_score=fundamental.composite_score,
                        mispricing_score=fundamental.mispricing_score,
                        market_cap_m=fundamental.market_cap_m,
                        bar_count=len(bars),
                        min_bars=cfg.min_bars,
                        open_positions=len(open_positions),
                        max_open_positions=MAX_OPEN_POSITIONS,
                        account_equity=equity,
                    ))
                    continue
                if evaluate_fn is not None:
                    evaluation = evaluate_fn(
                        bars,
                        fundamental,
                        datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                        cfg,
                    )
                    signal = evaluation.signal
                else:
                    signal = compute_fn(
                        bars,
                        fundamental,
                        datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                        cfg,
                    )
                    evaluation = SignalEvaluation(
                        signal=signal,
                        decision="signal" if signal else "rejected",
                        reason_code="signal_passed" if signal else "no_signal",
                        reason_text=signal.entry_reason if signal else "Model returned no signal without a detailed reason.",
                    )
                if signal:
                    signal.exchange = fundamental.exchange
                    signal.cik = fundamental.cik
                    next_entry = get_next_bar_open(conn, fundamental.identity_key, bars[-1].ts)
                    if next_entry is None:
                        signal = None
                        evaluation.signal = None
                        evaluation.decision = "rejected"
                        evaluation.reason_code = "next_entry_bar_missing"
                        evaluation.reason_text = (
                            f"No 1h bar after signal bar {bars[-1].ts} was available for next-bar-open entry."
                        )
                        evaluation.entry_price = None
                        evaluation.stop_loss = None
                        evaluation.take_profit_1 = None
                        evaluation.take_profit_2 = None
                    else:
                        entry_ts, entry_open = next_entry
                        trade_levels = apply_trade_levels(
                            signal,
                            bars,
                            cfg,
                            _ensure_utc_ts(entry_ts),
                            entry_open,
                        )
                        if not trade_levels.accepted:
                            signal = None
                            evaluation.signal = None
                            evaluation.decision = "rejected"
                            evaluation.reason_code = trade_levels.reason_code
                            evaluation.reason_text = trade_levels.reason_text
                            evaluation.entry_price = None
                            evaluation.stop_loss = None
                            evaluation.take_profit_1 = None
                            evaluation.take_profit_2 = None
                        else:
                            evaluation.entry_price = signal.entry_price
                            evaluation.stop_loss = signal.stop_loss
                            evaluation.take_profit_1 = signal.take_profit_1
                            evaluation.take_profit_2 = signal.take_profit_2
                            signals.append(signal)
                            signals_by_direction[direction].append(signal)
                event = DecisionEvent(
                    run_id=run_id,
                    signal_date=day,
                    as_of_ts=day_end_ts,
                    symbol=fundamental.symbol,
                    exchange=fundamental.exchange,
                    cik=fundamental.cik,
                    direction=direction,
                    decision_stage="signal_eval",
                    decision="signal" if signal else "rejected",
                    reason_code=evaluation.reason_code,
                    reason_text=evaluation.reason_text,
                    signal_passed=bool(signal),
                    candidate_rank=candidate_rank,
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    valuation_label=fundamental.valuation_label,
                    sector=fundamental.sector,
                    industry=fundamental.industry,
                    fundamental_score=fundamental.composite_score,
                    mispricing_score=fundamental.mispricing_score,
                    market_cap_m=fundamental.market_cap_m,
                    bar_count=len(bars),
                    min_bars=cfg.min_bars,
                    entry_ts=signal.entry_ts if signal else None,
                    entry_price=evaluation.entry_price if evaluation.entry_price is not None else (signal.entry_price if signal else None),
                    stop_loss=evaluation.stop_loss if evaluation.stop_loss is not None else (signal.stop_loss if signal else None),
                    take_profit_1=evaluation.take_profit_1 if evaluation.take_profit_1 is not None else (signal.take_profit_1 if signal else None),
                    take_profit_2=evaluation.take_profit_2 if evaluation.take_profit_2 is not None else (signal.take_profit_2 if signal else None),
                    pullback_pct=evaluation.pullback_pct if evaluation.pullback_pct is not None else (signal.pullback_pct if signal else None),
                    rsi_1h=evaluation.rsi_1h if evaluation.rsi_1h is not None else (signal.rsi_1h if signal else None),
                    volume_ratio=evaluation.volume_ratio if evaluation.volume_ratio is not None else (signal.volume_ratio if signal else None),
                    entry_score=evaluation.entry_score if evaluation.entry_score is not None else (signal.entry_score if signal else None),
                    combined_score=evaluation.combined_score if evaluation.combined_score is not None else (signal.combined_score if signal else None),
                    open_positions=len(open_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                )
                decision_events.append(event)
                if signal:
                    signal_events[_signal_event_key(signal)] = event

        for direction, direction_signals in signals_by_direction.items():
            direction_signals.sort(key=lambda s: s.combined_score, reverse=True)
            signal_counts[direction] = len(direction_signals)
            for signal_rank, signal in enumerate(direction_signals, start=1):
                event = signal_events.get(_signal_event_key(signal))
                if event:
                    event.signal_rank = signal_rank

        if total_candidates == 0:
            days_no_candidates += 1
        if signals:
            days_with_signals += 1
        else:
            days_no_signals += 1

        # ── 3. Open new positions ────────────────────────────────────────────
        open_identities = {p.identity_key for p in open_positions}
        if SECTOR_DIVERSIFICATION_ENABLED:
            open_sectors: set[str] = {p.signal.sector for p in open_positions if p.signal.sector}
            open_sector_industries: set[tuple[str, str]] = {
                (p.signal.sector, p.signal.industry)
                for p in open_positions
                if p.signal.sector
            }

            def _sector_tier(s: Signal) -> int:
                if not s.sector or s.sector not in open_sectors:
                    return 0  # new sector preferred
                if (s.sector, s.industry) not in open_sector_industries:
                    return 1  # same sector, different industry
                return 2      # same sector and industry

            for direction, direction_signals in signals_by_direction.items():
                direction_signals.sort(key=lambda s: (_sector_tier(s), -s.combined_score))
                for signal_rank, signal in enumerate(direction_signals, start=1):
                    event = signal_events.get(_signal_event_key(signal))
                    if event:
                        event.signal_rank = signal_rank

        direction_order = sorted(
            DIRECTIONS,
            key=lambda d: (
                direction_risk_multiplier(regime_exposure, d),
                direction_max_positions(regime_exposure, d),
            ),
            reverse=True,
        )
        signals_to_open: list[Signal] = []
        for direction in direction_order:
            signals_to_open.extend(signals_by_direction[direction])

        opened_today = 0
        account_snapshot_today = _account_snapshot_values(conn, open_positions, equity, day_end_ts)
        account_equity_today = account_snapshot_today.equity_with_loan_value
        initial_margin = sum(_active_margin_used(p) for p in open_positions)
        maintenance_margin = sum(_active_maintenance_margin_used(p) for p in open_positions)
        direction_open_counts = {
            direction: _direction_open_count(open_positions, direction)
            for direction in DIRECTIONS
        }

        for signal in signals_to_open:
            event = signal_events.get(_signal_event_key(signal))
            direction_risk = direction_risk_multiplier(regime_exposure, signal.direction)
            direction_cap = direction_max_positions(regime_exposure, signal.direction)
            available_funds = account_equity_today - initial_margin
            excess_liquidity = account_equity_today - maintenance_margin
            if event:
                event.open_positions = len(open_positions)
                event.max_open_positions = MAX_OPEN_POSITIONS
                event.account_equity = account_equity_today
                event.initial_margin = initial_margin
                event.maintenance_margin = maintenance_margin
                event.available_funds = available_funds
                event.excess_liquidity = excess_liquidity

            if direction_risk <= 0.0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "regime_direction_risk_zero"
                    event.reason_text = (
                        f"Regime bucket {regime_bucket} assigned zero {signal.direction.lower()} risk."
                    )
                continue
            if direction_open_counts[signal.direction] >= direction_cap:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "max_direction_positions_reached"
                    event.reason_text = (
                        f"Regime bucket {regime_bucket} allows {direction_cap} open "
                        f"{signal.direction.lower()} positions; this limit was already reached."
                    )
                continue
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "max_open_positions_reached"
                    event.reason_text = f"Maximum open positions {MAX_OPEN_POSITIONS} was already reached."
                continue
            if signal.identity_key in open_identities:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "instrument_already_open"
                    event.reason_text = "Instrument already had an open position."
                continue

            if account_equity_today <= 0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "account_equity_non_positive"
                    event.reason_text = "Account equity was not positive at decision time."
                continue

            initial_margin_used, maintenance_margin_used, shares, position_size_usd = calc_position(
                conn,
                signal,
                account_equity_today,
                direction_risk,
            )
            if event:
                event.required_initial_margin = initial_margin_used
                event.required_maintenance_margin = maintenance_margin_used
                event.position_size_usd = position_size_usd
                event.shares = shares
            if position_size_usd <= 0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "position_size_non_positive"
                    event.reason_text = "Position sizing produced a non-positive position size."
                continue

            initial_margin_after = initial_margin + initial_margin_used
            maintenance_margin_after = maintenance_margin + maintenance_margin_used
            available_funds_after = account_equity_today - initial_margin_after
            excess_liquidity_after = account_equity_today - maintenance_margin_after
            if event:
                event.available_funds_after = available_funds_after
                event.excess_liquidity_after = excess_liquidity_after

            if ACCOUNT_PROFILE == "ps_acc":
                margin_level_after = _margin_level_pct(account_equity_today, initial_margin_after)
                if margin_level_after <= PS_MARGIN_STOP_OUT_LEVEL_PCT:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "margin_level_stop_out_guard"
                        event.reason_text = (
                            f"Margin level after entry would be {margin_level_after:.2f}%, "
                            f"at or below Pepperstone stop-out level {PS_MARGIN_STOP_OUT_LEVEL_PCT:.2f}%."
                        )
                    continue
                if margin_level_after < PS_MIN_ENTRY_MARGIN_LEVEL_PCT:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "margin_level_entry_guard"
                        event.reason_text = (
                            f"Margin level after entry would be {margin_level_after:.2f}%, "
                            f"below configured Pepperstone backtest minimum {PS_MIN_ENTRY_MARGIN_LEVEL_PCT:.2f}%."
                        )
                    continue
            elif ACCOUNT_PROFILE == "ibkr_acc":
                if available_funds_after < 0:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "available_funds_insufficient"
                        event.reason_text = (
                            f"Available Funds after entry would be {available_funds_after:.2f}, "
                            "below zero."
                        )
                    continue
                if excess_liquidity_after <= 0:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "excess_liquidity_non_positive_guard"
                        event.reason_text = (
                            f"Excess Liquidity after entry would be {excess_liquidity_after:.2f}, "
                            "at or below zero."
                        )
                    continue

            open_positions.append(OpenPosition(
                symbol=signal.symbol,
                exchange=signal.exchange,
                cik=signal.cik,
                direction=signal.direction,
                entry_date=day,
                entry_ts=signal.entry_ts or day_end_ts,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                effective_sl=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                valid_until=(signal.entry_ts or day_end_ts) + timedelta(
                    days=cfg.long_max_hold_days if signal.direction == "LONG" else cfg.short_max_hold_days
                ),
                tp1_close_ratio=cfg.tp1_close_ratio,
                shares=shares,
                position_size_usd=position_size_usd,
                margin_used=initial_margin_used,
                maintenance_margin_used=maintenance_margin_used,
                equity_before=account_equity_today,
                signal=signal,
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                valuation_label=signal.valuation_label,
            ))
            open_identities.add(signal.identity_key)
            initial_margin += initial_margin_used
            maintenance_margin += maintenance_margin_used
            direction_open_counts[signal.direction] += 1
            opened_today += 1
            record_account_curve(signal.entry_ts or day_end_ts, open_positions)
            if event:
                event.decision_stage = "order_open"
                event.decision = "opened"
                event.reason_code = "opened"
                event.reason_text = "Signal passed portfolio checks and a simulated position was opened."
                event.opened = True
            log.debug("Opened %-6s %s entry %.2f stop %.2f margin %.0f equity %.0f",
                      signal.symbol, signal.direction, signal.entry_price,
                      signal.stop_loss, initial_margin_used, equity)

        # ── 4. Apply post-entry position state changes through day close ────
        open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(open_positions, day_close_ts)
        closed_today += closed_after_entry
        day_pnl += pnl_after_entry
        record_account_curve(day_close_ts, open_positions)

        buffer_decision_events(decision_events)

        if log_progress_today or opened_today > 0:
            log.info(
                "Progress %d/%d %s model %s bucket %s regime %.1f, candidates long %d short %d, signals long %d short %d, skipped no bars %d, opened %d, closed today %d, day pnl %.0f, open %d, equity %.0f, closed total %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                regime_bucket,
                regime.score,
                candidate_counts["LONG"],
                candidate_counts["SHORT"],
                signal_counts["LONG"],
                signal_counts["SHORT"],
                skipped_no_bars,
                opened_today,
                closed_today,
                day_pnl,
                len(open_positions),
                equity,
                len(closed_trades),
            )

    log.info(
        "Day breakdown no regime %d, no active budget %d, no candidates %d, no signals %d, with signals %d",
        days_no_regime, days_no_active_budget, days_no_candidates, days_no_signals, days_with_signals,
    )
    log_cache_stats("before_force_close")

    # ── 5. Force-close remaining open positions at last available price ──────
    last_day = trading_days[-1] if trading_days else END_DATE
    if open_positions:
        preload_identity_bars(
            conn,
            [pos.identity_key for pos in open_positions],
            _day_close_ts(last_day),
            batch_size=BAR_CACHE_BATCH_SIZE,
            log_batches=True,
        )
    for pos in list(open_positions):
        bars = get_bars_range(conn, pos.identity_key, pos.entry_ts, last_day)
        last_price = float(bars[-1][4]) if bars else pos.entry_price
        if pos.direction == "LONG":
            if pos.tp1_hit:
                pnl = _pnl_long(pos, pos.tp1_price or pos.take_profit_1, last_price, split_exits=True)
            else:
                pnl = _pnl_long(pos, last_price, last_price, split_exits=False)
        else:
            if pos.tp1_hit:
                pnl = _pnl_short(pos, pos.tp1_price or pos.take_profit_1, last_price, split_exits=True)
            else:
                pnl = _pnl_short(pos, last_price, last_price, split_exits=False)
        trade = _make_trade(
            conn,
            pos,
            "FORCE_CLOSED",
            last_price,
            last_day,
            len(bars) if bars else 0,
            pos.tp1_hit,
            pnl,
            equity,
            _day_close_ts(last_day),
            pos.tp1_exit_ts,
        )
        _remove_position_by_identity(open_positions, pos)
        trade.equity_after = round(equity + trade.pnl_usd, 2)
        equity = trade.equity_after
        closed_trades.append(trade)
        record_account_curve(trade.exit_ts or _day_close_ts(last_day), open_positions)

    # ── 6. Persist results ───────────────────────────────────────────────────
    log.info("Writing %d trades and %d account snapshots for run %d", len(closed_trades), len(account_curve), run_id)

    flush_decision_events(force=True)
    max_dd = _max_drawdown_pct_from_equity([point.equity_usd for point in account_curve])

    # Patch world_regime_label into rows (stored on signal, pass through)
    # (already embedded in entry_reason; trade write accesses signal directly)
    write_account_curve(conn, run_id, account_curve)
    write_trades(conn, run_id, closed_trades)
    update_run_summary(conn, run_id, closed_trades, equity, max_drawdown_pct=max_dd)
    if MONTE_CARLO_ENABLED:
        run_monte_carlo(conn, run_id, closed_trades, INITIAL_EQUITY, MONTE_CARLO_SIMULATIONS)

    run_duration_seconds = _time.perf_counter() - run_started
    update_run_duration(conn, run_id, run_duration_seconds)

    n_trades = len(closed_trades)
    n_wins = sum(1 for t in closed_trades if t.pnl_usd > 0)
    n_losses = sum(1 for t in closed_trades if t.pnl_usd < 0)
    gross_profit = sum(t.pnl_usd for t in closed_trades if t.pnl_usd > 0)
    gross_loss = abs(sum(t.pnl_usd for t in closed_trades if t.pnl_usd < 0))
    total_return = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0
    margin_hours_usd = sum(t.margin_hours_usd for t in closed_trades)
    return_per_margin_hour_pct = (
        sum(t.pnl_usd for t in closed_trades) / margin_hours_usd * 100.0
        if margin_hours_usd > 0.0
        else None
    )

    direction_reports = {
        direction: _direction_report(closed_trades, direction)
        for direction in DIRECTIONS
    }

    summary = {
        "run_id": run_id,
        "total_trades": n_trades,
        "win_rate_pct": n_wins / n_trades * 100.0 if n_trades else 0.0,
        "total_return_pct": total_return,
        "margin_hours_usd": margin_hours_usd,
        "return_per_margin_hour_pct": return_per_margin_hour_pct,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
        "run_duration_seconds": run_duration_seconds,
    }
    for direction, report in direction_reports.items():
        prefix = direction.lower()
        summary[f"{prefix}_trades"] = report["trades"]
        summary[f"{prefix}_win_rate_pct"] = report["win_rate_pct"]
        summary[f"{prefix}_profit_factor"] = report["profit_factor"]
        summary[f"{prefix}_avg_r"] = report["avg_r"]
        summary[f"{prefix}_pnl_usd"] = report["pnl_usd"]

    log.info(
        "Run %d complete trades %d wins %d final equity %.0f return %.1f%% duration %.1f seconds",
        run_id, n_trades, n_wins, equity, total_return, run_duration_seconds,
    )
    for direction in DIRECTIONS:
        report = direction_reports[direction]
        log.info(
            "Run %d direction %s trades %d wins %d losses %d win rate %.1f%% profit factor %s avg R %s pnl %.0f",
            run_id,
            direction,
            report["trades"],
            report["wins"],
            report["losses"],
            report["win_rate_pct"],
            _format_optional(report["profit_factor"], "%.3f"),
            _format_optional(report["avg_r"], "%.3f"),
            report["pnl_usd"],
        )
    return run_id, summary
