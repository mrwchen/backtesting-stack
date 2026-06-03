"""Domain data structures for the generic backtester."""

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from backtest_shared import InstrumentKey, TradePlan, instrument_key

@dataclass
class OpenPosition:
    symbol: str
    exchange: str
    cik: int
    direction: str
    entry_date: date
    entry_ts: datetime
    entry_price: float
    stop_loss: float
    effective_sl: float
    take_profit_mode: str
    take_profit: Optional[float]
    trailing_activation_price: Optional[float]
    trailing_distance_pct: Optional[float]
    valid_until: datetime
    shares: float
    position_size_usd: float
    margin_used: float
    maintenance_margin_used: float
    equity_before: float
    plan: TradePlan
    world_regime_label: str = ""
    world_regime_score: float = 0.0
    valuation_label: str = ""
    # incremental simulate_outcome state — updated in-place on each day's pass
    trailing_activated: bool = False
    trailing_reference_price: Optional[float] = None
    trailing_stop: Optional[float] = None
    trailing_activated_ts: Optional[datetime] = None
    last_bar_ts: Optional[datetime] = None
    bars_processed: int = 0

    @property
    def identity_key(self) -> InstrumentKey:
        return instrument_key(self.symbol, self.exchange, self.cik)


@dataclass
class ClosedTrade:
    position: OpenPosition
    outcome_status: str        # HIT_TP | HIT_TRAILING_STOP | HIT_SL | MAX_HOLD | FORCE_CLOSED | MARGIN_STOP_OUT | IBKR_MARGIN_LIQUIDATION
    outcome_price: float
    outcome_date: date
    outcome_bars: int
    trailing_activated: bool
    return_pct: float
    margin_hours_usd: float
    return_per_margin_hour_pct: Optional[float]
    pnl_usd: float
    equity_after: float
    exit_ts: datetime = None
    trailing_stop: Optional[float] = None
    trailing_activated_ts: Optional[datetime] = None


@dataclass
class AccountCurvePoint:
    run_id: int
    ts: datetime
    trade_date: date
    seq_in_run: int
    balance_usd: float
    open_pnl_usd: float
    equity_usd: float
    initial_margin_usd: float
    maintenance_margin_usd: float
    available_funds_usd: float
    excess_liquidity_usd: float
    open_positions: int
    realized_pnl_usd: float
    closed_trades: int


@dataclass(frozen=True)
class AccountMarginSnapshot:
    open_pnl: float
    equity_with_loan_value: float
    initial_margin: float
    maintenance_margin: float
    available_funds: float
    excess_liquidity: float


@dataclass(frozen=True)
class PortfolioEvent:
    ts: datetime
    priority: int
    kind: str
    position: OpenPosition
    trade: Optional[ClosedTrade] = None


@dataclass
class DecisionEvent:
    run_id: int
    intent_date: date
    as_of_ts: Optional[datetime]
    symbol: Optional[str]
    exchange: Optional[str]
    cik: Optional[int]
    direction: Optional[str]
    decision_stage: str
    decision: str
    reason_code: str
    reason_text: str = ""
    intent_passed: bool = False
    opened: bool = False
    candidate_rank: Optional[int] = None
    intent_rank: Optional[int] = None
    world_regime_label: str = ""
    world_regime_score: Optional[float] = None
    valuation_label: str = ""
    sector: str = ""
    industry: str = ""
    fundamental_score: Optional[float] = None
    mispricing_score: Optional[float] = None
    market_cap_m: Optional[float] = None
    bar_count: Optional[int] = None
    min_bars: Optional[int] = None
    intent_score: Optional[float] = None
    intent_reason: str = ""
    entry_ts: Optional[datetime] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_activation_price: Optional[float] = None
    trailing_distance_pct: Optional[float] = None
    open_positions: Optional[int] = None
    max_open_positions: Optional[int] = None
    account_equity: Optional[float] = None
    initial_margin: Optional[float] = None
    maintenance_margin: Optional[float] = None
    available_funds: Optional[float] = None
    excess_liquidity: Optional[float] = None
    required_initial_margin: Optional[float] = None
    required_maintenance_margin: Optional[float] = None
    available_funds_after: Optional[float] = None
    excess_liquidity_after: Optional[float] = None
    position_size_usd: Optional[float] = None
    shares: Optional[float] = None
