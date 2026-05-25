"""Broker/account mechanics: costs, margin, financing, and sizing."""

import logging
import math
from bisect import bisect_left, bisect_right
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2

from backtest_shared import TradePlan
from .config import *
from .entities import AccountMarginSnapshot, ClosedTrade, OpenPosition
from .ibkr_margin import get_ibkr_margin_requirement
from .market_data import _day_close_ts, _ensure_utc_ts, _load_identity_bars_through

log = logging.getLogger(__name__)
_PEPPERSTONE_ROLLOVER_ZONE = ZoneInfo("America/New_York")

def _pnl_long(pos: OpenPosition, exit_price: float) -> float:
    entry = _buy_fill(pos.entry_price)
    exit_fill = _sell_fill(exit_price)
    gross = pos.shares * (exit_fill - entry)
    costs = _entry_cost(pos.shares, entry) + _exit_cost(pos.shares, exit_fill)
    return gross - costs


def _pnl_short(pos: OpenPosition, exit_price: float) -> float:
    entry = _sell_fill(pos.entry_price)
    exit_fill = _buy_fill(exit_price)
    gross = pos.shares * (entry - exit_fill)
    costs = _entry_cost(pos.shares, entry) + _exit_cost(pos.shares, exit_fill)
    return gross - costs


def _buy_fill(mid_price: float) -> float:
    return mid_price * (1.0 + _execution_bps() / 10000.0)


def _sell_fill(mid_price: float) -> float:
    return mid_price * (1.0 - _execution_bps() / 10000.0)


def _execution_bps() -> float:
    return SPREAD_BPS * 0.5 + SLIPPAGE_BPS


def _entry_cost(shares: float, fill_price: float) -> float:
    return _order_cost(shares, fill_price)


def _exit_cost(shares: float, fill_price: float) -> float:
    return _order_cost(shares, fill_price)


def _order_cost(shares: float, fill_price: float) -> float:
    notional = abs(shares * fill_price)
    if abs(shares) <= 0.0 or notional <= 0.0:
        return 0.0
    cost = (
        COMMISSION_PER_ORDER_USD
        + abs(shares) * COMMISSION_PER_SHARE_USD
        + notional * COMMISSION_BPS / 10000.0
    )
    if COMMISSION_MIN_PER_ORDER_USD > 0:
        cost = max(cost, COMMISSION_MIN_PER_ORDER_USD)
    if COMMISSION_MAX_PCT > 0 and notional > 0:
        cost = min(cost, notional * COMMISSION_MAX_PCT / 100.0)
    return cost


def _active_margin_used(pos: OpenPosition) -> float:
    return pos.margin_used


def _active_maintenance_margin_used(pos: OpenPosition) -> float:
    return pos.maintenance_margin_used


def _initial_margin_pct(direction: str) -> float:
    if ACCOUNT_PROFILE == "ibkr_acc":
        raise RuntimeError("IBKR margin must be read from symbol margin requirements")
    return float(MARGIN_REQUIREMENT_PCT)


def _maintenance_margin_pct(direction: str) -> float:
    if ACCOUNT_PROFILE == "ibkr_acc":
        raise RuntimeError("IBKR margin must be read from symbol margin requirements")
    return float(MARGIN_REQUIREMENT_PCT)


def _margin_level_pct(account_equity: float, used_margin: float) -> float:
    if used_margin <= 0.0:
        return math.inf
    return account_equity / used_margin * 100.0


def _latest_close_price(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_date: date,
) -> float:
    as_of_ts = _day_close_ts(as_of_date)
    timestamps, bars = _load_identity_bars_through(conn, pos.identity_key, as_of_ts)
    idx = bisect_right(timestamps, as_of_ts) - 1
    if idx < 0:
        return pos.entry_price
    return float(bars[idx].close)


def _latest_close_price_at(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_ts: datetime,
) -> float:
    as_of_ts = _ensure_utc_ts(as_of_ts)
    if as_of_ts <= _ensure_utc_ts(pos.entry_ts):
        return pos.entry_price
    timestamps, bars = _load_identity_bars_through(conn, pos.identity_key, as_of_ts)
    idx = bisect_left(timestamps, as_of_ts) - 1
    if idx < 0:
        return pos.entry_price
    return float(bars[idx].close)


def _open_position_mark_to_market_pnl(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    mark_price: float,
    as_of_date: date,
) -> float:
    if pos.direction == "LONG":
        pnl = _pnl_long(pos, mark_price)
    else:
        pnl = _pnl_short(pos, mark_price)
    return pnl - _financing_cost(conn, pos, as_of_date, _day_close_ts(as_of_date))


def _open_position_mark_to_market_pnl_at(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    mark_price: float,
    as_of_ts: datetime,
) -> float:
    as_of_ts = _ensure_utc_ts(as_of_ts)
    if pos.direction == "LONG":
        pnl = _pnl_long(pos, mark_price)
    else:
        pnl = _pnl_short(pos, mark_price)
    return pnl - _financing_cost(conn, pos, as_of_ts.date(), as_of_ts)


def _account_snapshot_values(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    balance: float,
    as_of_ts: datetime,
) -> AccountMarginSnapshot:
    open_pnl = 0.0
    for pos in open_positions:
        mark_price = _latest_close_price_at(conn, pos, as_of_ts)
        open_pnl += _open_position_mark_to_market_pnl_at(conn, pos, mark_price, as_of_ts)
    initial_margin = sum(_active_margin_used(pos) for pos in open_positions)
    maintenance_margin = sum(_active_maintenance_margin_used(pos) for pos in open_positions)
    equity_with_loan_value = balance + open_pnl
    return AccountMarginSnapshot(
        open_pnl=open_pnl,
        equity_with_loan_value=equity_with_loan_value,
        initial_margin=initial_margin,
        maintenance_margin=maintenance_margin,
        available_funds=equity_with_loan_value - initial_margin,
        excess_liquidity=equity_with_loan_value - maintenance_margin,
    )


def _mark_to_market_close_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    status: str,
    mark_price: float,
    realized_equity: float,
    exit_ts: datetime,
) -> ClosedTrade:
    exit_ts = _ensure_utc_ts(exit_ts)
    if pos.direction == "LONG":
        pnl = _pnl_long(pos, mark_price)
    else:
        pnl = _pnl_short(pos, mark_price)
    return _make_trade(
        conn,
        pos,
        status,
        mark_price,
        exit_ts.date(),
        pos.bars_processed,
        pnl,
        realized_equity,
        exit_ts,
    )


def _position_stop_out_rank(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_ts: datetime,
) -> tuple[float, float, str, str, int]:
    mark_price = _latest_close_price_at(conn, pos, as_of_ts)
    pnl = _open_position_mark_to_market_pnl_at(conn, pos, mark_price, as_of_ts)
    return (pnl, -_active_margin_used(pos), pos.symbol, pos.exchange, pos.cik)


def _position_ibkr_liquidation_rank(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_ts: datetime,
) -> tuple[float, float, str, str, int]:
    mark_price = _latest_close_price_at(conn, pos, as_of_ts)
    pnl = _open_position_mark_to_market_pnl_at(conn, pos, mark_price, as_of_ts)
    return (pnl, -_active_maintenance_margin_used(pos), pos.symbol, pos.exchange, pos.cik)


def _enforce_pepperstone_margin_stop_out(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    realized_equity: float,
    as_of_ts: datetime,
) -> tuple[list[ClosedTrade], float]:
    if ACCOUNT_PROFILE != "ps_acc" or not open_positions:
        return [], realized_equity

    as_of_ts = _ensure_utc_ts(as_of_ts)
    stop_out_trades: list[ClosedTrade] = []
    active_positions = open_positions

    while active_positions:
        snapshot = _account_snapshot_values(
            conn,
            active_positions,
            realized_equity,
            as_of_ts,
        )
        margin_level = _margin_level_pct(snapshot.equity_with_loan_value, snapshot.initial_margin)
        if margin_level > PS_MARGIN_STOP_OUT_LEVEL_PCT:
            break

        position = min(
            active_positions,
            key=lambda p: _position_stop_out_rank(conn, p, as_of_ts),
        )
        mark_price = _latest_close_price_at(conn, position, as_of_ts)
        trade = _mark_to_market_close_trade(
            conn,
            position,
            "MARGIN_STOP_OUT",
            mark_price,
            realized_equity,
            as_of_ts,
        )
        _remove_position_by_identity(active_positions, position)
        realized_equity = round(realized_equity + trade.pnl_usd, 2)
        trade.equity_after = realized_equity
        stop_out_trades.append(trade)
        log.warning(
            "Pepperstone margin stop-out %s margin level %.2f%% threshold %.2f%% pnl %.2f balance %.2f",
            position.symbol,
            margin_level,
            PS_MARGIN_STOP_OUT_LEVEL_PCT,
            trade.pnl_usd,
            realized_equity,
        )

    return stop_out_trades, realized_equity


def _enforce_ibkr_excess_liquidity_liquidation(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    realized_equity: float,
    as_of_ts: datetime,
) -> tuple[list[ClosedTrade], float]:
    if ACCOUNT_PROFILE != "ibkr_acc" or not open_positions:
        return [], realized_equity

    as_of_ts = _ensure_utc_ts(as_of_ts)
    liquidation_trades: list[ClosedTrade] = []
    active_positions = open_positions

    while active_positions:
        snapshot = _account_snapshot_values(conn, active_positions, realized_equity, as_of_ts)
        if snapshot.excess_liquidity > 0.0:
            break

        position = min(
            active_positions,
            key=lambda p: _position_ibkr_liquidation_rank(conn, p, as_of_ts),
        )
        mark_price = _latest_close_price_at(conn, position, as_of_ts)
        trade = _mark_to_market_close_trade(
            conn,
            position,
            "IBKR_MARGIN_LIQUIDATION",
            mark_price,
            realized_equity,
            as_of_ts,
        )
        _remove_position_by_identity(active_positions, position)
        realized_equity = round(realized_equity + trade.pnl_usd, 2)
        trade.equity_after = realized_equity
        liquidation_trades.append(trade)
        log.warning(
            "IBKR margin liquidation %s excess liquidity %.2f maintenance margin %.2f pnl %.2f balance %.2f",
            position.symbol,
            snapshot.excess_liquidity,
            snapshot.maintenance_margin,
            trade.pnl_usd,
            realized_equity,
        )

    return liquidation_trades, realized_equity


def _enforce_account_margin_liquidation(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    realized_equity: float,
    as_of_ts: datetime,
) -> tuple[list[ClosedTrade], float]:
    if ACCOUNT_PROFILE == "ps_acc":
        return _enforce_pepperstone_margin_stop_out(conn, open_positions, realized_equity, as_of_ts)
    if ACCOUNT_PROFILE == "ibkr_acc":
        return _enforce_ibkr_excess_liquidity_liquidation(conn, open_positions, realized_equity, as_of_ts)
    return [], realized_equity


def _remove_position_by_identity(open_positions: list[OpenPosition], position: OpenPosition) -> None:
    for idx, current in enumerate(open_positions):
        if current is position:
            del open_positions[idx]
            return


def _account_equity(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    realized_equity: float,
    as_of_date: date,
) -> float:
    open_pnl = 0.0
    for pos in open_positions:
        open_pnl += _open_position_mark_to_market_pnl(conn, pos, _latest_close_price(conn, pos, as_of_date), as_of_date)
    return realized_equity + open_pnl


def _financing_days(start_ts: datetime, end_ts: datetime) -> float:
    return max(0.0, (end_ts - start_ts).total_seconds() / 86400.0)


def _pepperstone_rollover_ts(local_day: date) -> datetime:
    local_ts = datetime(
        local_day.year,
        local_day.month,
        local_day.day,
        17,
        0,
        0,
        tzinfo=_PEPPERSTONE_ROLLOVER_ZONE,
    )
    return local_ts.astimezone(timezone.utc)


def _pepperstone_rollover_multiplier(local_day: date) -> float:
    return 3.0 if local_day.weekday() == 4 else 1.0


def _pepperstone_rollovers_between(start_ts: datetime, end_ts: datetime) -> list[tuple[datetime, float]]:
    start_ts = _ensure_utc_ts(start_ts)
    end_ts = _ensure_utc_ts(end_ts)
    if end_ts <= start_ts:
        return []

    start_day = start_ts.astimezone(_PEPPERSTONE_ROLLOVER_ZONE).date()
    end_day = end_ts.astimezone(_PEPPERSTONE_ROLLOVER_ZONE).date()
    days = (end_day - start_day).days
    rollovers: list[tuple[datetime, float]] = []
    for day_offset in range(days + 1):
        local_day = start_day + timedelta(days=day_offset)
        if local_day.weekday() >= 5:
            continue
        rollover_ts = _pepperstone_rollover_ts(local_day)
        if start_ts < rollover_ts <= end_ts:
            rollovers.append((rollover_ts, _pepperstone_rollover_multiplier(local_day)))
    return rollovers


def _active_shares_at_rollover(
    pos: OpenPosition,
) -> float:
    return pos.shares


def _pepperstone_share_cfd_overnight_cost(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    end_ts: datetime,
) -> float:
    if ACCOUNT_PROFILE != "ps_acc":
        return 0.0

    if pos.direction == "LONG":
        annual_rate_pct = PS_SHARE_CFD_ADMIN_FEE_PCT + PS_SHARE_CFD_ARR_PCT
    else:
        annual_rate_pct = (
            PS_SHARE_CFD_ADMIN_FEE_PCT
            - PS_SHARE_CFD_ARR_PCT
            + PS_SHARE_CFD_SHORT_BORROW_RATE_PCT
        )

    total = 0.0
    for rollover_ts, multiplier in _pepperstone_rollovers_between(pos.entry_ts, end_ts):
        active_shares = _active_shares_at_rollover(pos)
        if active_shares <= 0.0:
            continue
        mark_price = _latest_close_price_at(conn, pos, rollover_ts)
        active_notional = abs(active_shares * mark_price)
        total += active_notional * annual_rate_pct / 100.0 * multiplier / PS_SHARE_CFD_OVERNIGHT_DAY_COUNT
    return total


def _generic_margin_financing_cost(
    pos: OpenPosition,
    end_ts: datetime,
) -> float:
    borrowed_notional = max(0.0, pos.position_size_usd - pos.margin_used)
    if borrowed_notional <= 0.0:
        return 0.0

    end_ts = _ensure_utc_ts(end_ts)
    total_days = max(1.0, _financing_days(pos.entry_ts, end_ts))
    return borrowed_notional * MARGIN_FINANCING_RATE_PCT / 100.0 * total_days / 365.0


def _financing_cost(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    outcome_date: date,
    exit_ts: Optional[datetime] = None,
) -> float:
    end_ts = exit_ts or _day_close_ts(outcome_date)
    if ACCOUNT_PROFILE == "ps_acc":
        return _pepperstone_share_cfd_overnight_cost(conn, pos, end_ts)
    return _generic_margin_financing_cost(pos, end_ts)


def _hours_between(start_ts: datetime, end_ts: datetime) -> float:
    start_ts = _ensure_utc_ts(start_ts)
    end_ts = _ensure_utc_ts(end_ts)
    return max(0.0, (end_ts - start_ts).total_seconds() / 3600.0)


def _margin_hours_usd(
    pos: OpenPosition,
    outcome_date: date,
    exit_ts: Optional[datetime],
) -> float:
    margin_used = max(0.0, pos.margin_used)
    if margin_used <= 0.0:
        return 0.0

    entry_ts = _ensure_utc_ts(pos.entry_ts)
    end_ts = _ensure_utc_ts(exit_ts or _day_close_ts(outcome_date))
    if end_ts <= entry_ts:
        return 0.0
    return margin_used * _hours_between(entry_ts, end_ts)


def _make_trade(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    status: str,
    outcome_price: float,
    outcome_date: date,
    outcome_bars: int,
    pnl: float,
    equity: float,
    exit_ts: datetime = None,
) -> ClosedTrade:
    pnl -= _financing_cost(conn, pos, outcome_date, exit_ts)
    return_pct = pnl / pos.position_size_usd * 100.0 if pos.position_size_usd else 0.0
    margin_hours_usd = _margin_hours_usd(pos, outcome_date, exit_ts)
    return_per_margin_hour_pct = pnl / margin_hours_usd * 100.0 if margin_hours_usd > 0.0 else None
    return ClosedTrade(
        position=pos,
        outcome_status=status,
        outcome_price=outcome_price,
        outcome_date=outcome_date,
        outcome_bars=outcome_bars,
        trailing_activated=pos.trailing_activated,
        return_pct=round(return_pct, 4),
        margin_hours_usd=round(margin_hours_usd, 4),
        return_per_margin_hour_pct=round(return_per_margin_hour_pct, 8) if return_per_margin_hour_pct is not None else None,
        pnl_usd=round(pnl, 2),
        equity_after=round(equity + pnl, 2),
        exit_ts=exit_ts,
        trailing_stop=pos.trailing_stop,
        trailing_activated_ts=pos.trailing_activated_ts,
    )


# ── Position sizing ───────────────────────────────────────────────────────────

def _stop_loss_cash_risk(
    shares: float,
    entry_fill: float,
    stop_fill: float,
    direction: str,
) -> float:
    if shares <= 0.0:
        return 0.0
    if direction == "LONG":
        gross_loss = shares * (entry_fill - stop_fill)
    else:
        gross_loss = shares * (stop_fill - entry_fill)
    return gross_loss + _entry_cost(shares, entry_fill) + _exit_cost(shares, stop_fill)


def initial_stop_cash_risk(pos: OpenPosition) -> float:
    if pos.direction == "LONG":
        entry_fill = _buy_fill(pos.entry_price)
        stop_fill = _sell_fill(pos.stop_loss)
    else:
        entry_fill = _sell_fill(pos.entry_price)
        stop_fill = _buy_fill(pos.stop_loss)
    return _stop_loss_cash_risk(pos.shares, entry_fill, stop_fill, pos.direction)


def _max_shares_for_stop_risk(
    risk_usd: float,
    entry_fill: float,
    stop_fill: float,
    direction: str,
) -> float:
    def risk_for(shares: float) -> float:
        return _stop_loss_cash_risk(shares, entry_fill, stop_fill, direction)

    high = 1.0
    while risk_for(high) <= risk_usd:
        high *= 2.0
        if high > 1e12:
            break

    if ALLOW_FRACTIONAL_SHARES:
        low = 0.0
        for _ in range(80):
            mid = (low + high) / 2.0
            if risk_for(mid) <= risk_usd:
                low = mid
            else:
                high = mid
        return low

    low_i = 0
    high_i = max(1, int(math.ceil(high)))
    while low_i + 1 < high_i:
        mid = (low_i + high_i) // 2
        if risk_for(float(mid)) <= risk_usd:
            low_i = mid
        else:
            high_i = mid
    return float(low_i)


def calc_position(
    conn: psycopg2.extensions.connection,
    plan: TradePlan,
    equity: float,
    risk_multiplier: float,
) -> tuple[float, float, float, float]:
    """Return (initial_margin_used, maintenance_margin_used, shares, position_size_usd)."""
    risk_usd = equity * RISK_PER_TRADE_PCT * max(0.0, risk_multiplier) / 100.0
    if plan.direction == "LONG":
        entry_fill = _buy_fill(plan.entry_price)
        stop_fill = _sell_fill(plan.stop_loss)
        loss_per_share_before_commission = entry_fill - stop_fill
    else:
        entry_fill = _sell_fill(plan.entry_price)
        stop_fill = _buy_fill(plan.stop_loss)
        loss_per_share_before_commission = stop_fill - entry_fill

    if loss_per_share_before_commission <= 0 or risk_usd <= 0:
        return 0.0, 0.0, 0.0, 0.0

    shares = _max_shares_for_stop_risk(risk_usd, entry_fill, stop_fill, plan.direction)
    if shares <= 0.0 or (not ALLOW_FRACTIONAL_SHARES and shares < 1.0):
        return 0.0, 0.0, 0.0, 0.0
    position_size_usd = abs(shares * entry_fill)
    if ACCOUNT_PROFILE == "ibkr_acc":
        margin_requirement = get_ibkr_margin_requirement(conn, plan.symbol, plan.direction)
        initial_margin_used = position_size_usd * margin_requirement.initial_margin_pct / 100.0
        maintenance_margin_used = position_size_usd * margin_requirement.maintenance_margin_pct / 100.0
    else:
        initial_margin_used = position_size_usd * _initial_margin_pct(plan.direction) / 100.0
        maintenance_margin_used = position_size_usd * _maintenance_margin_pct(plan.direction) / 100.0
    return initial_margin_used, maintenance_margin_used, shares, position_size_usd
