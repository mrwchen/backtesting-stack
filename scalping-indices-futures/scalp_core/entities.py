"""Domain entities shared across the engine."""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ClosedTrade:
    """One completed round-trip trade."""

    intent_ts: datetime          # bar close at which the signal fired
    entry_ts: datetime
    entry_price: float
    direction: str               # "LONG" | "SHORT"
    units: float                 # continuous CFD units (always positive)
    notional: float              # units * entry_price * multiplier (account currency)
    margin_used: float
    regime_state: int
    prob_up: float
    sigma_pts: float             # forecast volatility in price points
    stop_price: float
    take_profit_price: float
    outcome_status: str          # HIT_TP | HIT_SL | MAX_HOLD | SESSION_FLAT
    exit_ts: datetime
    exit_price: float
    bars_held: int
    return_pct: float            # on margin (equity-relevant return)
    pnl: float                   # account currency
    costs: float                 # total transaction costs (account currency)
    equity_before: float
    equity_after: float
