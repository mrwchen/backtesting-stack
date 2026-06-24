"""Domain entities for the NAS100 hit-frequency median backtest."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Sizing:
    units: float
    notional_eur: float
    margin_used_eur: float
    risk_eur: float


@dataclass(slots=True)
class ClosedTrade:
    signal_ts: datetime
    entry_ts: datetime
    exit_ts: datetime
    direction: str
    entry_session: str
    cross_quantile: float
    cross_level: float
    profile_low: float
    profile_high: float
    profile_range: float
    cross_price_range_position_pct: float
    entry_price_range_position_pct: float
    range_position_deviation_pct: float
    median_level: float
    signal_mid: float
    previous_mid: float
    entry_bid: float
    entry_ask: float
    entry_price: float
    exit_bid: float
    exit_ask: float
    exit_price: float
    stop_price: float
    take_profit_price: float
    units: float
    notional_eur: float
    margin_used_eur: float
    gross_pnl_eur: float
    extra_costs_eur: float
    pnl_eur: float
    equity_before: float
    equity_after: float
    return_pct: float
    price_pnl_points: float
    outcome_status: str
    ticks_held: int
    seconds_held: float
    realized_risk_eur: float
    realized_risk_pct: float
    margin_capped: bool


@dataclass(slots=True)
class SimulationResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    initial_equity: float = 0.0
    final_equity: float = 0.0
    ticks_total: int = 0
    ticks_simulated: int = 0
    bars_total: int = 0
    signals_total: int = 0
    long_signals: int = 0
    short_signals: int = 0
    rejected_signals_missing_band: int = 0
    rejected_signals_band_too_narrow: int = 0
    rejected_signals_price_range_position: int = 0
    rejected_signals_stop_too_small: int = 0
    rejected_signals_stop_too_large: int = 0
    skipped_signals_no_size: int = 0
    ruined: bool = False
