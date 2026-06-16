"""Pepperstone-style CFD account model for tick-level NAS100 backtests."""

import math

from . import config
from .entities import Sizing


def usd_to_eur(usd: float) -> float:
    rate = config.EURUSD_RATE if config.EURUSD_RATE > 0 else 1.0
    return usd / rate


def _round_down_to_lot_size(units: float) -> float:
    lot_size = config.LOT_SIZE
    if units < lot_size:
        return 0.0
    return round(math.floor((units + 1e-12) / lot_size) * lot_size, 10)


def size_position(equity_eur: float, entry_price: float, stop_points: float) -> Sizing:
    """Size a CFD position so the configured fixed stop costs the risk budget."""
    mult = config.CONTRACT_MULTIPLIER
    if stop_points <= 0 or entry_price <= 0 or equity_eur <= 0:
        return Sizing(0.0, 0.0, 0.0, 0.0)

    risk_budget_eur = equity_eur * (config.RISK_PER_TRADE_PCT / 100.0)
    risk_per_unit_eur = usd_to_eur(stop_points * mult)
    if risk_per_unit_eur <= 0:
        return Sizing(0.0, 0.0, 0.0, 0.0)
    units = risk_budget_eur / risk_per_unit_eur

    notional_eur = usd_to_eur(units * entry_price * mult)
    margin_used_eur = notional_eur * (config.MARGIN_REQUIREMENT_PCT / 100.0)
    max_margin_eur = equity_eur * (config.MAX_MARGIN_PCT / 100.0)
    if margin_used_eur > max_margin_eur and margin_used_eur > 0:
        units *= max_margin_eur / margin_used_eur

    units = _round_down_to_lot_size(units)
    if units <= 0:
        return Sizing(0.0, 0.0, 0.0, 0.0)

    notional_eur = usd_to_eur(units * entry_price * mult)
    margin_used_eur = notional_eur * (config.MARGIN_REQUIREMENT_PCT / 100.0)
    risk_eur = units * risk_per_unit_eur
    return Sizing(units=units, notional_eur=notional_eur, margin_used_eur=margin_used_eur, risk_eur=risk_eur)


def gross_pnl(units: float, entry_price: float, exit_price: float, direction: str) -> float:
    mult = config.CONTRACT_MULTIPLIER
    sign = 1.0 if direction == "LONG" else -1.0
    return usd_to_eur(units * mult * (exit_price - entry_price) * sign)


def extra_round_trip_costs(units: float) -> float:
    """Extra configured costs; live tick bid/ask spread is already in fill prices."""
    mult = config.CONTRACT_MULTIPLIER
    point_costs = usd_to_eur(units * (config.SPREAD_POINTS + config.SLIPPAGE_POINTS) * mult)
    commission = config.COMMISSION_PER_UNIT * units * 2.0
    return point_costs + commission

