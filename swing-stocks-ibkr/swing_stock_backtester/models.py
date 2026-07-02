from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class StockIdentity:
    symbol: str
    exchange: str
    cik: int


@dataclass(frozen=True)
class EarningsEvent:
    earnings_date: date
    announcement_ts: datetime | None
    announcement_time_type: str
    source: str
    known_as_of_ts: datetime | None
    is_confirmed: bool
    surprise_pct: float | None


@dataclass
class Trade:
    strategy_name: str
    strategy_version: str
    identity: StockIdentity
    trade_number: int
    signal_date: date
    entry_date: date
    exit_date: date
    entry_price: float
    exit_price: float
    stop_price: float | None
    gross_return_pct: float
    net_return_pct: float
    holding_days: int
    exit_reason: str
    signal_score: float | None
    quality_score: float | None
    momentum_score: float | None
    entry_condition: str
    fundamental_asof_date: date | None
    earnings_event_date: date | None
    earnings_known_asof_ts: datetime | None


@dataclass
class StrategyResult:
    strategy_name: str
    strategy_version: str
    identity: StockIdentity
    status: str
    first_trade_date: date | None = None
    last_trade_date: date | None = None
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    flat_count: int = 0
    avg_return_pct: float | None = None
    median_return_pct: float | None = None
    best_return_pct: float | None = None
    worst_return_pct: float | None = None
    total_compounded_return_pct: float | None = None
    max_drawdown_pct: float | None = None
    profit_factor: float | None = None
    expectancy_pct: float | None = None
    avg_holding_days: float | None = None
    exposure_days: int = 0
    signal_count: int = 0
    skipped_signal_count: int = 0
    error_text: str | None = None
    trades: list[Trade] = field(default_factory=list)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SymbolBacktestResult:
    identity: StockIdentity
    status: str
    results: list[StrategyResult]
    error_text: str | None = None
