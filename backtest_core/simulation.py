"""Core point-in-time portfolio simulation loop."""

import logging
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2

from backtest_shared import InstrumentKey, IntentEvaluation, TradeIntent, TradePlan, fundamental_base_score
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
from .entities import AccountCurvePoint, ClosedTrade, DecisionEvent, OpenPosition
from .market_data import (
    _day_close_ts,
    _day_signal_cutoff_ts,
    _ensure_utc_ts,
    get_bars_range,
    get_bars_range_through,
    get_candidates,
    get_trading_days,
    get_world_regime,
    _is_in_entry_window,
    _is_stop_loss_active,
    _is_in_sl_tp_window,
    latest_expected_signal_bar_start_ts,
    load_next_bar_opens,
    load_recent_bars_for_identities,
    log_cache_stats,
    preload_identity_bars,
    preload_candidate_timelines,
    signal_bar_close_decisions_for_day,
)
from .model_loader import get_model_module
from .monte_carlo import run_monte_carlo
from .policy import (
    candidate_policy_kwargs,
    direction_filter_negative_earnings,
    direction_max_positions,
    direction_risk_multiplier,
    regime_exposure_for_label,
)
from .persistence import (
    create_run,
    update_run_duration,
    update_run_summary,
    write_account_curve,
    write_decision_events,
    write_trades,
)
from .shock_overlay import (
    apply_shock_overlay,
    should_evaluate_disabled_direction,
    risk_off_long_sleeve_risk,
    shock_stress_direction_cap,
    shock_stress_plan_block_reason,
    shock_stress_portfolio_block_reason,
    shock_stress_sector_limit,
)
from .trade_levels import (
    build_trade_plan,
    common_stop_required_lookback,
    execution_max_hold_days,
    validate_intent_for_candidate,
)

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


def _same_direction_sector_open_count(open_positions: list[OpenPosition], plan: TradePlan) -> int:
    sector = str(plan.sector or "").strip().lower()
    if not sector:
        return 0
    return sum(
        1
        for pos in open_positions
        if pos.direction == plan.direction
        and str(pos.plan.sector or "").strip().lower() == sector
    )


def _candidate_score_kwargs(cfg: Any) -> dict:
    return {
        "fundamental_score_mode": getattr(cfg, "fundamental_score_mode", "peer"),
        "fundamental_peer_weight": getattr(cfg, "fundamental_peer_weight", 1.0),
        "fundamental_abs_weight": getattr(cfg, "fundamental_abs_weight", 0.0),
        "long_min_absolute_score": getattr(cfg, "long_min_absolute_score", None),
        "short_max_absolute_score": getattr(cfg, "short_max_absolute_score", None),
    }


def _model_fundamental_score(fundamental: Any, cfg: Any) -> float:
    return fundamental_base_score(
        fundamental,
        getattr(cfg, "fundamental_score_mode", "peer"),
        getattr(cfg, "fundamental_peer_weight", 1.0),
        getattr(cfg, "fundamental_abs_weight", 0.0),
    )


def _signal_bar_recency_rejection(
    bars: list[Any],
    as_of_ts: datetime,
    trading_days: list[date],
) -> tuple[str, str] | None:
    if not bars:
        return None
    latest_signal_bar_start_ts = _ensure_utc_ts(bars[-1].ts)
    latest_complete_ts = latest_signal_bar_start_ts + timedelta(hours=1)
    as_of_ts = _ensure_utc_ts(as_of_ts)
    if latest_complete_ts > as_of_ts:
        return (
            "latest_signal_bar_not_complete",
            f"Latest 1h signal bar completes at {latest_complete_ts}, after decision timestamp {as_of_ts}.",
        )
    expected_signal_bar_start_ts = latest_expected_signal_bar_start_ts(as_of_ts, trading_days)
    if expected_signal_bar_start_ts is None or latest_signal_bar_start_ts >= expected_signal_bar_start_ts:
        return None

    max_staleness = timedelta(hours=SIGNAL_BAR_MAX_STALENESS_HOURS)
    staleness = as_of_ts - latest_complete_ts
    if staleness > max_staleness:
        return (
            "latest_signal_bar_too_stale",
            (
                f"Latest complete 1h signal bar ended at {latest_complete_ts}, "
                f"{staleness} before decision timestamp {as_of_ts}; latest expected signal bar start "
                f"was {expected_signal_bar_start_ts}; max allowed during an active signal session is {max_staleness}."
            ),
        )
    return None


def _plan_event_key(plan: TradePlan) -> tuple[str, tuple[str, str, int]]:
    return (plan.direction, plan.identity_key)


def _copy_plan_shock_to_event(event: DecisionEvent, plan: TradePlan) -> None:
    event.dominant_shock_type = plan.dominant_shock_type
    event.max_shock_type_score = plan.max_shock_type_score
    event.defensive_risk_off_score = plan.defensive_risk_off_score
    event.energy_commodity_shock_score = plan.energy_commodity_shock_score
    event.rates_inflation_usd_shock_score = plan.rates_inflation_usd_shock_score
    event.credit_banking_stress_score = plan.credit_banking_stress_score
    event.policy_geopolitical_score = plan.policy_geopolitical_score
    event.shock_sector_bias = plan.shock_sector_bias
    event.shock_score_delta = plan.shock_score_delta
    event.shock_risk_multiplier = plan.shock_risk_multiplier
    event.shock_base_intent_score = plan.shock_base_intent_score


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


def _exit_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    status: str,
    price: float,
    bar_date: date,
    total_bars: int,
    equity: float,
    ts: datetime,
) -> ClosedTrade:
    pnl = _pnl_long(pos, price) if pos.direction == "LONG" else _pnl_short(pos, price)
    return _make_trade(conn, pos, status, price, bar_date, total_bars, pnl, equity, ts)


def _long_stop_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    price: float,
    bar_date: date,
    total_bars: int,
    equity: float,
    ts: datetime,
) -> ClosedTrade:
    status = "HIT_TRAILING_STOP" if pos.trailing_activated else "HIT_SL"
    return _exit_trade(conn, pos, status, price, bar_date, total_bars, equity, ts)


def _short_stop_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    price: float,
    bar_date: date,
    total_bars: int,
    equity: float,
    ts: datetime,
) -> ClosedTrade:
    status = "HIT_TRAILING_STOP" if pos.trailing_activated else "HIT_SL"
    return _exit_trade(conn, pos, status, price, bar_date, total_bars, equity, ts)


def _long_take_profit_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    bar_date: date,
    total_bars: int,
    equity: float,
    ts: datetime,
) -> ClosedTrade:
    price = pos.take_profit
    if price is None:
        raise ValueError("Fixed long take-profit was requested without a take_profit level")
    return _exit_trade(conn, pos, "HIT_TP", price, bar_date, total_bars, equity, ts)


def _short_take_profit_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    bar_date: date,
    total_bars: int,
    equity: float,
    ts: datetime,
) -> ClosedTrade:
    price = pos.take_profit
    if price is None:
        raise ValueError("Fixed short take-profit was requested without a take_profit level")
    return _exit_trade(conn, pos, "HIT_TP", price, bar_date, total_bars, equity, ts)


def _simulate_position_bar(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    equity: float,
) -> Optional[ClosedTrade]:
    ts = _ensure_utc_ts(ts)
    bar_date = ts.date()
    total_bars = pos.bars_processed + 1
    stop_loss_active = _is_stop_loss_active(ts, conn, pos.identity_key)
    sl_tp_active = _is_in_sl_tp_window(ts, conn, pos.identity_key)
    is_long = pos.direction == "LONG"

    if is_long:
        trade = _simulate_long_intrabar(
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
            equity,
        )
    else:
        trade = _simulate_short_intrabar(
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
            equity,
        )
    if trade is not None:
        return trade

    if ts >= pos.valid_until:
        price = float(close)
        pnl = _pnl_long(pos, price) if is_long else _pnl_short(pos, price)
        return _make_trade(conn, pos, "MAX_HOLD", price, bar_date, total_bars, pnl, equity, ts)

    pos.last_bar_ts = ts
    pos.bars_processed = total_bars
    return None


def _activate_long_trailing(pos: OpenPosition, reference_price: float, ts: datetime) -> float:
    distance = pos.trailing_distance_pct or 0.0
    pos.trailing_activated = True
    pos.trailing_activated_ts = pos.trailing_activated_ts or _ensure_utc_ts(ts)
    pos.trailing_reference_price = max(reference_price, pos.trailing_reference_price or reference_price)
    pos.trailing_stop = pos.trailing_reference_price * (1.0 - distance)
    pos.effective_sl = max(pos.effective_sl, pos.trailing_stop)
    return pos.effective_sl


def _activate_short_trailing(pos: OpenPosition, reference_price: float, ts: datetime) -> float:
    distance = pos.trailing_distance_pct or 0.0
    pos.trailing_activated = True
    pos.trailing_activated_ts = pos.trailing_activated_ts or _ensure_utc_ts(ts)
    pos.trailing_reference_price = min(reference_price, pos.trailing_reference_price or reference_price)
    pos.trailing_stop = pos.trailing_reference_price * (1.0 + distance)
    pos.effective_sl = min(pos.effective_sl, pos.trailing_stop)
    return pos.effective_sl


def _update_long_trailing(pos: OpenPosition, high: float) -> float:
    if not pos.trailing_activated:
        return pos.effective_sl
    distance = pos.trailing_distance_pct or 0.0
    reference = max(high, pos.trailing_reference_price or high)
    pos.trailing_reference_price = reference
    pos.trailing_stop = reference * (1.0 - distance)
    pos.effective_sl = max(pos.effective_sl, pos.trailing_stop)
    return pos.effective_sl


def _update_short_trailing(pos: OpenPosition, low: float) -> float:
    if not pos.trailing_activated:
        return pos.effective_sl
    distance = pos.trailing_distance_pct or 0.0
    reference = min(low, pos.trailing_reference_price or low)
    pos.trailing_reference_price = reference
    pos.trailing_stop = reference * (1.0 + distance)
    pos.effective_sl = min(pos.effective_sl, pos.trailing_stop)
    return pos.effective_sl


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
    equity: float,
) -> Optional[ClosedTrade]:
    # Open is known to be first. Favourable gaps can reach TP before any unknown low.
    if stop_loss_active and open_ <= pos.effective_sl:
        price = _long_stop_fill_price(pos.effective_sl, open_)
        return _long_stop_trade(conn, pos, price, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None and open_ >= pos.take_profit:
        return _long_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing" and not pos.trailing_activated:
        activation = pos.trailing_activation_price
        if activation is not None and open_ >= activation:
            _activate_long_trailing(pos, open_, ts)

    # High/low ordering between open and close is unknown. Resolve conflicts with SL first.
    stop_mid = stop_loss_active and _middle_low_reaches(open_, close, low, pos.effective_sl)
    if stop_mid:
        return _long_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None:
        tp_mid = _middle_high_reaches(open_, close, high, pos.take_profit)
        if tp_mid:
            return _long_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing":
        activation = pos.trailing_activation_price
        if not pos.trailing_activated and activation is not None and _middle_high_reaches(open_, close, high, activation):
            _activate_long_trailing(pos, high, ts)
        elif pos.trailing_activated:
            _update_long_trailing(pos, high)
        if stop_loss_active and pos.trailing_activated and _middle_low_reaches(open_, close, low, pos.effective_sl):
            return _long_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    # Close is known to be last.
    if stop_loss_active and close <= pos.effective_sl:
        return _long_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None and close >= pos.take_profit:
        return _long_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing":
        activation = pos.trailing_activation_price
        if not pos.trailing_activated and activation is not None and close >= activation:
            _activate_long_trailing(pos, close, ts)

    return None


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
    equity: float,
) -> Optional[ClosedTrade]:
    if stop_loss_active and open_ >= pos.effective_sl:
        price = _short_stop_fill_price(pos.effective_sl, open_)
        return _short_stop_trade(conn, pos, price, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None and open_ <= pos.take_profit:
        return _short_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing" and not pos.trailing_activated:
        activation = pos.trailing_activation_price
        if activation is not None and open_ <= activation:
            _activate_short_trailing(pos, open_, ts)

    stop_mid = stop_loss_active and _middle_high_reaches(open_, close, high, pos.effective_sl)
    if stop_mid:
        return _short_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None:
        tp_mid = _middle_low_reaches(open_, close, low, pos.take_profit)
        if tp_mid:
            return _short_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing":
        activation = pos.trailing_activation_price
        if not pos.trailing_activated and activation is not None and _middle_low_reaches(open_, close, low, activation):
            _activate_short_trailing(pos, low, ts)
        elif pos.trailing_activated:
            _update_short_trailing(pos, low)
        if stop_loss_active and pos.trailing_activated and _middle_high_reaches(open_, close, high, pos.effective_sl):
            return _short_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    if stop_loss_active and close >= pos.effective_sl:
        return _short_stop_trade(conn, pos, pos.effective_sl, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "fixed" and pos.take_profit is not None and close <= pos.take_profit:
        return _short_take_profit_trade(conn, pos, bar_date, total_bars, equity, ts)

    if sl_tp_active and pos.take_profit_mode == "trailing":
        activation = pos.trailing_activation_price
        if not pos.trailing_activated and activation is not None and close <= activation:
            _activate_short_trailing(pos, close, ts)

    return None


def simulate_outcome(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    up_to_ts: datetime,
    equity: float,
) -> Optional[ClosedTrade]:
    """
    Check whether pos has closed by up_to_ts.
    Returns ClosedTrade if closed, None if still open.

    Take-profit logic is either a full fixed TP or an activated trailing stop.

    Incremental: each call only scans bars newer than pos.last_bar_ts and
    resumes from the stop/trailing state stored on pos, making the loop O(N total)
    across all decision calls rather than O(N²).
    """
    up_to_ts = _ensure_utc_ts(up_to_ts)
    after_ts = pos.last_bar_ts if pos.last_bar_ts is not None else pos.entry_ts - timedelta(microseconds=1)
    bars = get_bars_range_through(conn, pos.identity_key, after_ts, up_to_ts)
    if not bars:
        return None

    for ts, open_, high, low, close in bars:
        trade = _simulate_position_bar(conn, pos, ts, open_, high, low, close, equity)
        if trade is not None:
            return trade
    return None

def run_backtest(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    notes: Optional[str] = None,
    reserved_run_id: Optional[int] = None,
) -> tuple[int, dict]:
    run_started = _time.perf_counter()
    run_id = create_run(conn, cfg, notes, reserved_run_id=reserved_run_id)

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
        closed_identity_blocklist: Optional[set[InstrumentKey]] = None,
    ) -> tuple[list[OpenPosition], int, float]:
        nonlocal equity
        end_ts = _ensure_utc_ts(end_ts)
        active_positions = list(positions)
        bars_by_position: dict[int, list] = {}
        bar_index_by_position: dict[int, int] = {}
        closed_count = 0
        realized_pnl = 0.0
        closed_identity_blocklist = closed_identity_blocklist if closed_identity_blocklist is not None else set()

        if active_positions:
            preload_identity_bars(
                conn,
                [pos.identity_key for pos in active_positions],
                end_ts,
                batch_size=BAR_CACHE_BATCH_SIZE,
                log_batches=False,
            )

        def position_is_active(position: OpenPosition) -> bool:
            return any(current is position for current in active_positions)

        def ensure_position_bars(position: OpenPosition) -> None:
            key = id(position)
            if key in bars_by_position:
                return
            after_ts = (
                position.last_bar_ts
                if position.last_bar_ts is not None
                else position.entry_ts - timedelta(microseconds=1)
            )
            bars_by_position[key] = get_bars_range_through(conn, position.identity_key, after_ts, end_ts)
            bar_index_by_position[key] = 0

        while True:
            for position in active_positions:
                ensure_position_bars(position)

            next_ts: Optional[datetime] = None
            for position in active_positions:
                key = id(position)
                bars = bars_by_position.get(key, [])
                idx = bar_index_by_position.get(key, 0)
                if idx >= len(bars):
                    continue
                bar_ts = _ensure_utc_ts(bars[idx][0])
                if next_ts is None or bar_ts < next_ts:
                    next_ts = bar_ts

            if next_ts is None:
                break

            close_events: list[tuple[OpenPosition, ClosedTrade]] = []
            for position in list(active_positions):
                key = id(position)
                bars = bars_by_position.get(key, [])
                idx = bar_index_by_position.get(key, 0)
                if idx >= len(bars):
                    continue
                ts, open_, high, low, close = bars[idx]
                ts = _ensure_utc_ts(ts)
                if ts != next_ts:
                    continue
                bar_index_by_position[key] = idx + 1
                trade = _simulate_position_bar(conn, position, ts, open_, high, low, close, equity)
                if trade is not None:
                    close_events.append((position, trade))

            if not close_events:
                continue

            for position, trade in close_events:
                if not position_is_active(position):
                    continue
                close_ts = _ensure_utc_ts(trade.exit_ts or next_ts)
                _remove_position_by_identity(active_positions, position)
                trade.equity_after = round(equity + trade.pnl_usd, 2)
                equity = trade.equity_after
                closed_trades.append(trade)
                closed_count += 1
                realized_pnl += trade.pnl_usd
                closed_identity_blocklist.add(position.identity_key)
                log.debug("Closed %-6s %s %s pnl %.0f balance %.0f",
                          position.symbol, position.direction, trade.outcome_status,
                          trade.pnl_usd, equity)
                record_account_curve(close_ts, active_positions)

            liquidation_trades, equity = _enforce_account_margin_liquidation(
                conn,
                active_positions,
                equity,
                next_ts,
            )
            if liquidation_trades:
                closed_trades.extend(liquidation_trades)
                closed_count += len(liquidation_trades)
                realized_pnl += sum(t.pnl_usd for t in liquidation_trades)
                for trade in liquidation_trades:
                    closed_identity_blocklist.add(trade.position.identity_key)
                record_account_curve(next_ts, active_positions)

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

    def build_intent_plans_for_as_of(
        day: date,
        as_of_ts: datetime,
        regime: Any,
        regime_label: str,
        regime_exposure: dict,
        model: Any,
        active_positions: list[OpenPosition],
        *,
        log_progress_today: bool = False,
        context_label: str = "signal_bar_close",
        entry_after_ts: Optional[datetime] = None,
        earliest_entry_ts: Optional[datetime] = None,
        latest_entry_ts: Optional[datetime] = None,
        entry_after_label: str = "decision timestamp",
    ) -> tuple[
        dict[str, list[TradePlan]],
        dict[tuple[str, tuple[str, str, int]], DecisionEvent],
        list[DecisionEvent],
        dict,
        dict[tuple[str, tuple[str, str, int]], dict[str, Any]],
    ]:
        as_of_ts = _ensure_utc_ts(as_of_ts)
        entry_after_ts = _ensure_utc_ts(entry_after_ts) if entry_after_ts is not None else as_of_ts
        earliest_entry_ts = _ensure_utc_ts(earliest_entry_ts) if earliest_entry_ts is not None else None
        latest_entry_ts = _ensure_utc_ts(latest_entry_ts) if latest_entry_ts is not None else None
        plans_by_direction: dict[str, list[TradePlan]] = {direction: [] for direction in DIRECTIONS}
        plans: list[TradePlan] = []
        decision_events: list[DecisionEvent] = []
        plan_events: dict[tuple[str, tuple[str, str, int]], DecisionEvent] = {}
        plan_entry_contexts: dict[tuple[str, tuple[str, str, int]], dict[str, Any]] = {}
        skipped_no_bars = 0
        total_candidates = 0
        candidate_counts: dict[str, int] = {direction: 0 for direction in DIRECTIONS}
        intent_counts: dict[str, int] = {direction: 0 for direction in DIRECTIONS}

        for direction in DIRECTIONS:
            direction_risk = direction_risk_multiplier(regime_exposure, direction)
            direction_cap = direction_max_positions(regime_exposure, direction)
            if (
                direction_risk <= 0.0
                and direction_cap <= 0
                and not should_evaluate_disabled_direction(regime_label, direction)
            ):
                decision_events.append(DecisionEvent(
                    run_id=run_id,
                    intent_date=day,
                    as_of_ts=as_of_ts,
                    symbol=None,
                    exchange=None,
                    cik=None,
                    direction=direction,
                    decision_stage="regime_filter",
                    decision="skipped_direction",
                    reason_code="regime_direction_disabled",
                    reason_text=f"Regime label {regime_label} assigned zero risk and zero max {direction.lower()} positions.",
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    open_positions=len(active_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                ))
                continue

            if log_progress_today:
                log.info(
                    "Candidate query starting context %s day %s model %s direction %s regime label %s cutoff %s",
                    context_label,
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    regime_label,
                    as_of_ts,
                )
            candidate_started = _time.perf_counter()
            candidates = get_candidates(
                conn,
                direction,
                **candidate_policy_kwargs(),
                **candidate_score_kwargs,
                source_table=SOURCE_FUNDAMENTAL_SCORES_TABLE,
                as_of_date=day,
                as_of_ts=as_of_ts,
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
                    "Candidate query complete context %s day %s model %s direction %s regime label %s found %d candidates in %.1f s",
                    context_label,
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    regime_label,
                    len(candidates),
                    candidate_elapsed,
                )

            if not candidates:
                decision_events.append(DecisionEvent(
                    run_id=run_id,
                    intent_date=day,
                    as_of_ts=as_of_ts,
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
                    open_positions=len(active_positions),
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
                as_of_ts,
                batch_size=BAR_CACHE_BATCH_SIZE,
                log_batches=log_progress_today,
            )
            loaded_bar_rows = sum(len(bars) for bars in recent_bars_by_identity.values())
            bar_load_elapsed = _time.perf_counter() - bar_load_started
            if log_progress_today or bar_load_elapsed >= 5.0:
                log.info(
                    "Recent bar load complete context %s day %s model %s direction %s loaded %d rows for %d candidates limit %d through %s in %.1f s",
                    context_label,
                    day,
                    runtime.CURRENT_MODEL_FILE,
                    direction,
                    loaded_bar_rows,
                    len(candidate_identities),
                    bar_lookback_limit,
                    as_of_ts,
                    bar_load_elapsed,
                )

            evaluate_fn = model.evaluate_long_intent if direction == "LONG" else model.evaluate_short_intent
            pending_intents: list[dict[str, Any]] = []

            for candidate_rank, fundamental in enumerate(candidates, start=1):
                bars = recent_bars_by_identity.get(fundamental.identity_key, [])
                if len(bars) < cfg.min_bars:
                    skipped_no_bars += 1
                    decision_events.append(DecisionEvent(
                        run_id=run_id,
                        intent_date=day,
                        as_of_ts=as_of_ts,
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
                        fundamental_score=_model_fundamental_score(fundamental, cfg),
                        mispricing_score=fundamental.mispricing_score,
                        market_cap_m=fundamental.market_cap_m,
                        bar_count=len(bars),
                        min_bars=cfg.min_bars,
                        open_positions=len(active_positions),
                        max_open_positions=MAX_OPEN_POSITIONS,
                        account_equity=equity,
                    ))
                    continue

                staleness_rejection = _signal_bar_recency_rejection(bars, as_of_ts, trading_days)
                if staleness_rejection is not None:
                    reason_code, reason_text = staleness_rejection
                    decision_events.append(DecisionEvent(
                        run_id=run_id,
                        intent_date=day,
                        as_of_ts=as_of_ts,
                        symbol=fundamental.symbol,
                        exchange=fundamental.exchange,
                        cik=fundamental.cik,
                        direction=direction,
                        decision_stage="bar_load",
                        decision="rejected",
                        reason_code=reason_code,
                        reason_text=reason_text,
                        candidate_rank=candidate_rank,
                        world_regime_label=regime.label,
                        world_regime_score=regime.score,
                        valuation_label=fundamental.valuation_label,
                        sector=fundamental.sector,
                        industry=fundamental.industry,
                        fundamental_score=_model_fundamental_score(fundamental, cfg),
                        mispricing_score=fundamental.mispricing_score,
                        market_cap_m=fundamental.market_cap_m,
                        bar_count=len(bars),
                        min_bars=cfg.min_bars,
                        open_positions=len(active_positions),
                        max_open_positions=MAX_OPEN_POSITIONS,
                        account_equity=equity,
                    ))
                    continue

                evaluation = evaluate_fn(bars, fundamental, as_of_ts, cfg)
                if not isinstance(evaluation, IntentEvaluation):
                    evaluation = IntentEvaluation(
                        intent=None,
                        decision="rejected",
                        reason_code="invalid_model_evaluation",
                        reason_text="Model did not return an IntentEvaluation.",
                    )
                intent = evaluation.intent
                plan: TradePlan | None = None
                if intent and not isinstance(intent, TradeIntent):
                    intent = None
                    evaluation.intent = None
                    evaluation.decision = "rejected"
                    evaluation.reason_code = "invalid_model_intent"
                    evaluation.reason_text = "Model did not return a TradeIntent."
                if intent:
                    intent_check = validate_intent_for_candidate(intent, fundamental, direction)
                    if not intent_check.accepted:
                        intent = None
                        evaluation.intent = None
                        evaluation.decision = "rejected"
                        evaluation.reason_code = intent_check.reason_code
                        evaluation.reason_text = intent_check.reason_text
                if intent:
                    pending_intents.append({
                        "candidate_rank": candidate_rank,
                        "fundamental": fundamental,
                        "bars": bars,
                        "evaluation": evaluation,
                        "intent": intent,
                        "after_ts": entry_after_ts,
                        "missing_reason_text": (
                            f"No 1h bar after {entry_after_label} {entry_after_ts} was available for next-bar-open entry."
                        ),
                    })
                    continue

                event = DecisionEvent(
                    run_id=run_id,
                    intent_date=day,
                    as_of_ts=as_of_ts,
                    symbol=fundamental.symbol,
                    exchange=fundamental.exchange,
                    cik=fundamental.cik,
                    direction=direction,
                    decision_stage="intent_eval",
                    decision="rejected",
                    reason_code=evaluation.reason_code,
                    reason_text=evaluation.reason_text,
                    intent_passed=False,
                    candidate_rank=candidate_rank,
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    valuation_label=fundamental.valuation_label,
                    sector=fundamental.sector,
                    industry=fundamental.industry,
                    fundamental_score=_model_fundamental_score(fundamental, cfg),
                    mispricing_score=fundamental.mispricing_score,
                    market_cap_m=fundamental.market_cap_m,
                    bar_count=len(bars),
                    min_bars=cfg.min_bars,
                    intent_score=None,
                    intent_reason="",
                    open_positions=len(active_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                )
                decision_events.append(event)

            next_bar_opens: dict[tuple[InstrumentKey, datetime], tuple[datetime, float]] = {}
            if pending_intents:
                next_open_started = _time.perf_counter()
                next_bar_opens = load_next_bar_opens(
                    conn,
                    [
                        (pending["fundamental"].identity_key, pending["after_ts"])
                        for pending in pending_intents
                    ],
                    batch_size=BAR_CACHE_BATCH_SIZE,
                )
                next_open_elapsed = _time.perf_counter() - next_open_started
                if log_progress_today or next_open_elapsed >= 5.0:
                    log.info(
                        "Next entry open batch complete context %s day %s model %s direction %s intents %d found %d in %.1f s",
                        context_label,
                        day,
                        runtime.CURRENT_MODEL_FILE,
                        direction,
                        len(pending_intents),
                        len(next_bar_opens),
                        next_open_elapsed,
                    )

            for pending in pending_intents:
                candidate_rank = pending["candidate_rank"]
                fundamental = pending["fundamental"]
                bars = pending["bars"]
                evaluation = pending["evaluation"]
                intent = pending["intent"]
                after_ts = pending["after_ts"]
                plan: TradePlan | None = None
                next_entry = next_bar_opens.get((fundamental.identity_key, after_ts))
                if next_entry is None:
                    intent = None
                    evaluation.intent = None
                    evaluation.decision = "rejected"
                    evaluation.reason_code = "next_entry_bar_missing"
                    evaluation.reason_text = pending["missing_reason_text"]
                else:
                    entry_ts, entry_open = next_entry
                    entry_ts = _ensure_utc_ts(entry_ts)
                    if earliest_entry_ts is not None and entry_ts < earliest_entry_ts:
                        intent = None
                        evaluation.intent = None
                        evaluation.decision = "rejected"
                        evaluation.reason_code = "next_entry_before_decision_timestamp"
                        evaluation.reason_text = (
                            f"Next entry bar {entry_ts} is before earliest allowed entry timestamp {earliest_entry_ts}."
                        )
                    elif latest_entry_ts is not None and entry_ts > latest_entry_ts:
                        intent = None
                        evaluation.intent = None
                        evaluation.decision = "rejected"
                        evaluation.reason_code = "next_entry_after_decision_timestamp"
                        evaluation.reason_text = (
                            f"Next entry bar {entry_ts} is after latest allowed entry timestamp {latest_entry_ts}."
                        )
                    else:
                        trade_plan_result = build_trade_plan(
                            intent,
                            fundamental,
                            bars,
                            entry_ts,
                            entry_open,
                        )
                        if not trade_plan_result.accepted:
                            intent = None
                            evaluation.intent = None
                            evaluation.decision = "rejected"
                            evaluation.reason_code = trade_plan_result.reason_code
                            evaluation.reason_text = trade_plan_result.reason_text
                        else:
                            plan = trade_plan_result.plan
                            if plan is None:
                                intent = None
                                evaluation.intent = None
                                evaluation.decision = "rejected"
                                evaluation.reason_code = "execution_plan_missing"
                                evaluation.reason_text = "Execution risk engine accepted the intent without returning a trade plan."
                            else:
                                plan.fundamental_score = _model_fundamental_score(fundamental, cfg)
                                apply_shock_overlay(plan, fundamental, regime)
                                plans.append(plan)
                                plans_by_direction[direction].append(plan)

                event = DecisionEvent(
                    run_id=run_id,
                    intent_date=day,
                    as_of_ts=as_of_ts,
                    symbol=fundamental.symbol,
                    exchange=fundamental.exchange,
                    cik=fundamental.cik,
                    direction=direction,
                    decision_stage="intent_eval",
                    decision="intent" if plan else "rejected",
                    reason_code=evaluation.reason_code,
                    reason_text=evaluation.reason_text,
                    intent_passed=bool(plan),
                    candidate_rank=candidate_rank,
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    valuation_label=fundamental.valuation_label,
                    sector=fundamental.sector,
                    industry=fundamental.industry,
                    fundamental_score=_model_fundamental_score(fundamental, cfg),
                    mispricing_score=fundamental.mispricing_score,
                    market_cap_m=fundamental.market_cap_m,
                    bar_count=len(bars),
                    min_bars=cfg.min_bars,
                    intent_score=plan.intent_score if plan else (intent.score if intent else None),
                    intent_reason=plan.intent_reason if plan else (intent.reason if intent else ""),
                    entry_ts=plan.entry_ts if plan else None,
                    entry_price=plan.entry_price if plan else None,
                    stop_loss=plan.stop_loss if plan else None,
                    take_profit=plan.take_profit if plan else None,
                    trailing_activation_price=plan.trailing_activation_price if plan else None,
                    trailing_distance_pct=plan.trailing_distance_pct if plan else None,
                    open_positions=len(active_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                )
                decision_events.append(event)
                if plan:
                    _copy_plan_shock_to_event(event, plan)
                    plan_key = _plan_event_key(plan)
                    plan_events[plan_key] = event
                    plan_entry_contexts[plan_key] = {
                        "intent": intent,
                        "fundamental": fundamental,
                        "bars": bars,
                    }

        for direction, direction_plans in plans_by_direction.items():
            direction_plans.sort(key=lambda plan: plan.intent_score, reverse=True)
            intent_counts[direction] = len(direction_plans)
            for intent_rank, plan in enumerate(direction_plans, start=1):
                event = plan_events.get(_plan_event_key(plan))
                if event:
                    event.intent_rank = intent_rank

        stats = {
            "total_candidates": total_candidates,
            "plans": len(plans),
            "skipped_no_bars": skipped_no_bars,
            "candidate_counts": candidate_counts,
            "intent_counts": intent_counts,
        }
        return plans_by_direction, plan_events, decision_events, stats, plan_entry_contexts

    def open_ranked_plans(
        day: date,
        as_of_ts: datetime,
        regime: Any,
        regime_label: str,
        regime_exposure: dict,
        plans_by_direction: dict[str, list[TradePlan]],
        plan_events: dict[tuple[str, tuple[str, str, int]], DecisionEvent],
        active_positions: list[OpenPosition],
        blocked_identities: set[InstrumentKey],
        *,
        require_entry_window: bool = False,
        entry_deadline_ts: Optional[datetime] = None,
        day_start_equity: Optional[float] = None,
    ) -> int:
        if SECTOR_DIVERSIFICATION_ENABLED:
            open_sectors: set[str] = {p.plan.sector for p in active_positions if p.plan.sector}
            open_sector_industries: set[tuple[str, str]] = {
                (p.plan.sector, p.plan.industry)
                for p in active_positions
                if p.plan.sector
            }

            def _sector_tier(plan: TradePlan) -> int:
                if not plan.sector or plan.sector not in open_sectors:
                    return 0
                if (plan.sector, plan.industry) not in open_sector_industries:
                    return 1
                return 2

            for direction, direction_plans in plans_by_direction.items():
                direction_plans.sort(key=lambda plan: (_sector_tier(plan), -plan.intent_score))
                for intent_rank, plan in enumerate(direction_plans, start=1):
                    event = plan_events.get(_plan_event_key(plan))
                    if event:
                        event.intent_rank = intent_rank

        direction_order = sorted(
            DIRECTIONS,
            key=lambda d: (
                direction_risk_multiplier(regime_exposure, d),
                direction_max_positions(regime_exposure, d),
            ),
            reverse=True,
        )
        plans_to_open: list[TradePlan] = []
        for direction in direction_order:
            plans_to_open.extend(plans_by_direction[direction])

        opened_count = 0
        open_identities = {p.identity_key for p in active_positions}
        entry_deadline_ts = _ensure_utc_ts(entry_deadline_ts) if entry_deadline_ts is not None else None

        for plan in plans_to_open:
            event = plan_events.get(_plan_event_key(plan))
            plan_entry_ts = _ensure_utc_ts(plan.entry_ts or as_of_ts)
            snapshot = _account_snapshot_values(conn, active_positions, equity, plan_entry_ts)
            account_equity_current = snapshot.equity_with_loan_value
            guard_day_start_equity = day_start_equity if day_start_equity is not None else account_equity_current
            initial_margin = sum(_active_margin_used(p) for p in active_positions)
            maintenance_margin = sum(_active_maintenance_margin_used(p) for p in active_positions)
            available_funds = account_equity_current - initial_margin
            excess_liquidity = account_equity_current - maintenance_margin
            direction_risk = direction_risk_multiplier(regime_exposure, plan.direction)
            direction_cap = direction_max_positions(regime_exposure, plan.direction)
            sleeve_risk = risk_off_long_sleeve_risk(plan, regime_label)
            if sleeve_risk is not None and direction_risk <= 0.0:
                direction_risk = sleeve_risk
                direction_cap = max(direction_cap, SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS)
            else:
                direction_risk *= plan.shock_risk_multiplier
            direction_cap = shock_stress_direction_cap(regime, plan.direction, direction_cap)
            if event:
                event.open_positions = len(active_positions)
                event.max_open_positions = MAX_OPEN_POSITIONS
                event.account_equity = account_equity_current
                event.initial_margin = initial_margin
                event.maintenance_margin = maintenance_margin
                event.available_funds = available_funds
                event.excess_liquidity = excess_liquidity

            if direction_risk <= 0.0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "regime_direction_risk_zero"
                    event.reason_text = f"Regime label {regime_label} assigned zero {plan.direction.lower()} risk."
                continue
            stress_plan_block = shock_stress_plan_block_reason(plan, regime)
            if stress_plan_block is not None:
                reason_code, reason_text = stress_plan_block
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = reason_code
                    event.reason_text = reason_text
                continue
            stress_portfolio_block = shock_stress_portfolio_block_reason(
                regime,
                account_equity_current,
                guard_day_start_equity,
                snapshot.open_pnl,
            )
            if stress_portfolio_block is not None:
                reason_code, reason_text = stress_portfolio_block
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = reason_code
                    event.reason_text = reason_text
                continue
            if _direction_open_count(active_positions, plan.direction) >= direction_cap:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "max_direction_positions_reached"
                    event.reason_text = (
                        f"Regime label {regime_label} allows {direction_cap} open "
                        f"{plan.direction.lower()} positions; this limit was already reached."
                    )
                continue
            if len(active_positions) >= MAX_OPEN_POSITIONS:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "max_open_positions_reached"
                    event.reason_text = f"Maximum open positions {MAX_OPEN_POSITIONS} was already reached."
                continue
            sector_limit = shock_stress_sector_limit(regime)
            if sector_limit is not None and _same_direction_sector_open_count(active_positions, plan) >= sector_limit:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "shock_stress_sector_cap_reached"
                    event.reason_text = (
                        f"max_shock_type_score stress guard allows {sector_limit} open "
                        f"{plan.direction.lower()} positions in sector {plan.sector or '-'}."
                    )
                continue
            if plan.identity_key in open_identities:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "instrument_already_open"
                    event.reason_text = "Instrument already had an open position."
                continue
            if plan.identity_key in blocked_identities:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "instrument_already_used_today"
                    event.reason_text = "Instrument was already opened or closed on this trading day."
                continue
            if require_entry_window and not _is_in_entry_window(plan_entry_ts, conn, plan.identity_key):
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "entry_outside_entry_window"
                    event.reason_text = f"Next entry bar {plan_entry_ts} is outside the configured entry window."
                continue
            if entry_deadline_ts is not None and plan_entry_ts > entry_deadline_ts:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "entry_after_deadline"
                    event.reason_text = f"Next entry bar {plan_entry_ts} is after entry deadline {entry_deadline_ts}."
                continue
            if account_equity_current <= 0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "account_equity_non_positive"
                    event.reason_text = "Account equity was not positive at decision time."
                continue

            initial_margin_used, maintenance_margin_used, shares, position_size_usd = calc_position(
                conn,
                plan,
                account_equity_current,
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
            available_funds_after = account_equity_current - initial_margin_after
            excess_liquidity_after = account_equity_current - maintenance_margin_after
            if event:
                event.available_funds_after = available_funds_after
                event.excess_liquidity_after = excess_liquidity_after

            if ACCOUNT_PROFILE == "ps_acc":
                margin_level_after = _margin_level_pct(account_equity_current, initial_margin_after)
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

            active_positions.append(OpenPosition(
                symbol=plan.symbol,
                exchange=plan.exchange,
                cik=plan.cik,
                direction=plan.direction,
                entry_date=day,
                entry_ts=plan_entry_ts,
                entry_price=plan.entry_price,
                stop_loss=plan.stop_loss,
                effective_sl=plan.stop_loss,
                take_profit_mode=plan.take_profit_mode,
                take_profit=plan.take_profit,
                trailing_activation_price=plan.trailing_activation_price,
                trailing_distance_pct=plan.trailing_distance_pct,
                valid_until=plan_entry_ts + timedelta(days=execution_max_hold_days(plan.direction)),
                shares=shares,
                position_size_usd=position_size_usd,
                margin_used=initial_margin_used,
                maintenance_margin_used=maintenance_margin_used,
                equity_before=account_equity_current,
                plan=plan,
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                valuation_label=plan.valuation_label,
            ))
            open_identities.add(plan.identity_key)
            blocked_identities.add(plan.identity_key)
            opened_count += 1
            record_account_curve(plan_entry_ts, active_positions)
            if event:
                event.decision_stage = "order_open"
                event.decision = "opened"
                event.reason_code = "opened"
                event.reason_text = "Intent passed portfolio checks and a simulated position was opened."
                event.opened = True
            log.debug("Opened %-6s %s entry %.2f stop %.2f margin %.0f equity %.0f",
                      plan.symbol, plan.direction, plan.entry_price,
                      plan.stop_loss, initial_margin_used, equity)

        return opened_count

    trading_days = get_trading_days(conn, START_DATE, END_DATE)
    log.info("Trading days to simulate: %d (%s → %s)", len(trading_days), START_DATE, END_DATE)
    candidate_score_kwargs = _candidate_score_kwargs(cfg)
    if trading_days:
        first_signal_decisions = signal_bar_close_decisions_for_day(trading_days[0])
        preload_as_of_ts = (
            first_signal_decisions[0][1]
            if first_signal_decisions
            else datetime.combine(trading_days[0], datetime.min.time(), tzinfo=timezone.utc)
        )
        preload_candidate_timelines(
            conn,
            DIRECTIONS,
            **candidate_policy_kwargs(),
            **candidate_score_kwargs,
            source_table=SOURCE_FUNDAMENTAL_SCORES_TABLE,
            as_of_date=trading_days[0],
            as_of_ts=preload_as_of_ts,
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
    days_no_intents    = 0
    days_with_intents  = 0

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

        # ── 1. Initialize day state ────────────────────────────────────────
        closed_today = 0
        day_pnl = 0.0
        opened_today = 0
        used_identities_today: set[InstrumentKey] = set()
        entry_deadline_ts = _day_signal_cutoff_ts(day)
        day_close_ts = _day_close_ts(day)
        day_start_ts = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        signal_decision_points = signal_bar_close_decisions_for_day(day)
        day_decision_ts = signal_decision_points[0][1] if signal_decision_points else day_start_ts
        day_start_equity = _account_snapshot_values(conn, open_positions, equity, day_start_ts).equity_with_loan_value

        # ── 2. Generate model intents and central execution plans ───────────
        regime_as_of_date = day - timedelta(days=1)
        regime = get_world_regime(conn, source_table=SOURCE_WORLD_REGIME_TABLE, as_of_date=regime_as_of_date)
        if not regime:
            days_no_regime += 1
            buffer_decision_events([DecisionEvent(
                run_id=run_id,
                intent_date=day,
                as_of_ts=day_decision_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="no_regime",
                reason_text=f"No world-regime row was available as of {regime_as_of_date}.",
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(
                open_positions,
                day_close_ts,
            )
            closed_today += closed_after_entry
            day_pnl += pnl_after_entry
            record_account_curve(day_close_ts, open_positions)
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s no regime, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        regime_label, regime_exposure = regime_exposure_for_label(regime.label)
        if log_progress_today:
            log.info(
                "Regime exposure day %d/%d %s model %s label %s score %.1f long risk %.2f short risk %.2f max long %d max short %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                regime_label,
                regime.score,
                direction_risk_multiplier(regime_exposure, "LONG"),
                direction_risk_multiplier(regime_exposure, "SHORT"),
                direction_max_positions(regime_exposure, "LONG"),
                direction_max_positions(regime_exposure, "SHORT"),
            )

        if all(
            direction_risk_multiplier(regime_exposure, direction) <= 0.0
            and direction_max_positions(regime_exposure, direction) <= 0
            and not should_evaluate_disabled_direction(regime_label, direction)
            for direction in DIRECTIONS
        ):
            days_no_active_budget += 1
            buffer_decision_events([DecisionEvent(
                run_id=run_id,
                intent_date=day,
                as_of_ts=day_decision_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="no_regime_exposure_budget",
                reason_text=f"Regime label {regime_label} assigned zero risk and zero max positions to both directions.",
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(
                open_positions,
                day_close_ts,
            )
            closed_today += closed_after_entry
            day_pnl += pnl_after_entry
            record_account_curve(day_close_ts, open_positions)
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s regime label %s had no exposure budget, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, regime_label, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        model = get_model_module()

        candidate_counts = {direction: 0 for direction in DIRECTIONS}
        intent_counts = {direction: 0 for direction in DIRECTIONS}
        skipped_no_bars = 0
        total_candidates = 0
        plans_count = 0
        decisions_processed = 0

        valid_signal_decision_points = [
            (signal_bar_start_ts, signal_decision_ts)
            for signal_bar_start_ts, signal_decision_ts in signal_decision_points
            if signal_decision_ts <= day_close_ts
        ]

        if not valid_signal_decision_points:
            buffer_decision_events([DecisionEvent(
                run_id=run_id,
                intent_date=day,
                as_of_ts=day_decision_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="signal_schedule",
                decision="skipped_day",
                reason_code="no_signal_bar_close_decisions",
                reason_text="No signal-bar-close decision timestamps were configured for this trading day.",
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])

        for signal_bar_start_ts, signal_decision_ts in valid_signal_decision_points:
            decisions_processed += 1
            open_positions, closed_before_decision, pnl_before_decision = apply_position_events_through(
                open_positions,
                signal_decision_ts,
                closed_identity_blocklist=used_identities_today,
            )
            closed_today += closed_before_decision
            day_pnl += pnl_before_decision

            (
                signal_plans_by_direction,
                signal_plan_events,
                signal_decision_events,
                signal_stats,
                _signal_plan_contexts,
            ) = build_intent_plans_for_as_of(
                day,
                signal_decision_ts,
                regime,
                regime_label,
                regime_exposure,
                model,
                open_positions,
                log_progress_today=log_progress_today,
                context_label="signal_bar_close",
                entry_after_ts=signal_bar_start_ts,
                earliest_entry_ts=signal_decision_ts,
                latest_entry_ts=signal_decision_ts,
                entry_after_label="closed signal bar start",
            )

            for direction in DIRECTIONS:
                candidate_counts[direction] += signal_stats["candidate_counts"][direction]
                intent_counts[direction] += signal_stats["intent_counts"][direction]
            total_candidates += signal_stats["total_candidates"]
            skipped_no_bars += signal_stats["skipped_no_bars"]
            plans_count += signal_stats["plans"]

            signal_entry_deadline_ts = min(entry_deadline_ts, signal_decision_ts)
            opened_today += open_ranked_plans(
                day,
                signal_decision_ts,
                regime,
                regime_label,
                regime_exposure,
                signal_plans_by_direction,
                signal_plan_events,
                open_positions,
                used_identities_today,
                require_entry_window=True,
                entry_deadline_ts=signal_entry_deadline_ts,
                day_start_equity=day_start_equity,
            )
            buffer_decision_events(signal_decision_events)

        if total_candidates == 0:
            days_no_candidates += 1
        if plans_count:
            days_with_intents += 1
        else:
            days_no_intents += 1

        open_positions, closed_after_entry, pnl_after_entry = apply_position_events_through(
            open_positions,
            day_close_ts,
            closed_identity_blocklist=used_identities_today,
        )
        closed_today += closed_after_entry
        day_pnl += pnl_after_entry
        record_account_curve(day_close_ts, open_positions)

        if log_progress_today or opened_today > 0:
            log.info(
                "Progress %d/%d %s model %s signal decisions %d regime label %s score %.1f, candidates long %d short %d, intents long %d short %d, skipped no bars %d, opened %d, closed today %d, day pnl %.0f, open %d, equity %.0f, closed total %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                decisions_processed,
                regime_label,
                regime.score,
                candidate_counts["LONG"],
                candidate_counts["SHORT"],
                intent_counts["LONG"],
                intent_counts["SHORT"],
                skipped_no_bars,
                opened_today,
                closed_today,
                day_pnl,
                len(open_positions),
                equity,
                len(closed_trades),
            )
    log.info(
        "Day breakdown no regime %d, no active budget %d, no candidates %d, no intents %d, with intents %d",
        days_no_regime, days_no_active_budget, days_no_candidates, days_no_intents, days_with_intents,
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
            pnl = _pnl_long(pos, last_price)
        else:
            pnl = _pnl_short(pos, last_price)
        trade = _make_trade(
            conn,
            pos,
            "FORCE_CLOSED",
            last_price,
            last_day,
            len(bars) if bars else 0,
            pnl,
            equity,
            _day_close_ts(last_day),
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

    # Trade rows read world-regime and intent context from each open position.
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
