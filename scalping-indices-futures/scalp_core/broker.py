"""PS_ACC broker model — CFD-style continuous sizing, margin, costs, PnL.

PS_ACC = Pepperstone-style index CFD account:
  * continuous (fractional) units — no whole-contract constraint;
  * margin = MARGIN_REQUIREMENT_PCT of notional (default 5%);
  * size derived from RISK_PER_TRADE_PCT of equity divided by the stop distance;
  * instrument is priced in USD; equity is in EUR, converted via EURUSD_RATE
    (USD per 1 EUR; default 1.0 = no conversion).
"""

from dataclasses import dataclass

from . import config


@dataclass(frozen=True)
class Sizing:
    units: float            # continuous units (>= 0)
    notional_eur: float
    margin_used_eur: float
    risk_eur: float         # intended cash risk at the stop


def _usd_to_eur(usd: float) -> float:
    rate = config.EURUSD_RATE if config.EURUSD_RATE > 0 else 1.0
    return usd / rate


def size_position(equity_eur: float, entry_price: float, stop_pct: float) -> Sizing:
    """Size a CFD position so the stop loss costs ~RISK_PER_TRADE_PCT of equity.

    stop_pct is the stop distance as a percent of entry price.
    """
    mult = config.CONTRACT_MULTIPLIER
    stop_pts = entry_price * (stop_pct / 100.0)
    if stop_pts <= 0 or entry_price <= 0 or equity_eur <= 0:
        return Sizing(0.0, 0.0, 0.0, 0.0)

    risk_eur = equity_eur * (config.RISK_PER_TRADE_PCT / 100.0)
    risk_per_unit_eur = _usd_to_eur(stop_pts * mult)
    if risk_per_unit_eur <= 0:
        return Sizing(0.0, 0.0, 0.0, 0.0)
    units = risk_eur / risk_per_unit_eur

    notional_eur = _usd_to_eur(units * entry_price * mult)
    margin_used_eur = notional_eur * (config.MARGIN_REQUIREMENT_PCT / 100.0)

    # Cap by maximum margin budget; scale units down proportionally if needed.
    max_margin_eur = equity_eur * (config.MAX_MARGIN_PCT / 100.0)
    if margin_used_eur > max_margin_eur and margin_used_eur > 0:
        scale = max_margin_eur / margin_used_eur
        units *= scale
        notional_eur *= scale
        margin_used_eur *= scale
        risk_eur *= scale

    return Sizing(units=units, notional_eur=notional_eur, margin_used_eur=margin_used_eur, risk_eur=risk_eur)


def round_trip_costs(units: float, entry_price: float, exit_price: float) -> float:
    """Spread + slippage (bps of notional, both legs) + per-unit commission (both legs)."""
    mult = config.CONTRACT_MULTIPLIER
    bps = (config.SPREAD_BPS + config.SLIPPAGE_BPS) / 10000.0
    entry_notional_eur = _usd_to_eur(units * entry_price * mult)
    exit_notional_eur = _usd_to_eur(units * exit_price * mult)
    spread_slip = (entry_notional_eur + exit_notional_eur) * bps
    commission = config.COMMISSION_PER_UNIT * units * 2.0
    return spread_slip + commission


def gross_pnl(units: float, entry_price: float, exit_price: float, direction: str) -> float:
    """Gross PnL in EUR before costs."""
    mult = config.CONTRACT_MULTIPLIER
    sign = 1.0 if direction == "LONG" else -1.0
    gross_usd = units * mult * (exit_price - entry_price) * sign
    return _usd_to_eur(gross_usd)
