"""Domain entities shared across the engine."""

from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Optional


@dataclass
class DecisionTrace:
    """One per-bar decision snapshot for model diagnostics."""

    ts: datetime
    session_date: date
    local_time: time
    close_price: float
    in_entry_window: bool
    decision_action: str
    decision_reason: str
    direction: Optional[str]
    prob_long_win: Optional[float]
    prob_short_win: Optional[float]
    selected_trade_prob: Optional[float]
    expected_net_r: Optional[float]
    expected_long_r: Optional[float]
    expected_short_r: Optional[float]
    regime_state: Optional[int]
    high_vol_state: Optional[bool]
    sigma_pts: Optional[float]
    atr_pts: Optional[float]
    stop_pct: Optional[float]
    tp_pct: Optional[float]


@dataclass
class ClosedTrade:
    """One completed round-trip trade."""

    intent_ts: datetime          # bar close at which the signal fired
    entry_ts: datetime
    entry_price: float
    direction: str               # "LONG" | "SHORT"
    units: float                 # CFD units rounded to lot size (always positive)
    notional: float              # units * entry_price * multiplier (account currency)
    margin_used: float
    regime_state: int
    prob_long_win: float
    prob_short_win: float
    selected_trade_prob: float
    expected_net_r: float
    decision_reason: str
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
