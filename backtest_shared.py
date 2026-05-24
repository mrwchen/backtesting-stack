"""Public API exposed to pluggable backtest model files."""

import math
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

InstrumentKey = tuple[str, str, int]


def instrument_key(symbol: str, exchange: str, cik: int) -> InstrumentKey:
    return (str(symbol).strip().upper(), str(exchange).strip().upper(), int(cik))


@dataclass(frozen=True)
class WorldRegime:
    day: object
    label: str
    score: float


@dataclass(frozen=True)
class FundamentalRow:
    symbol: str
    exchange: str
    cik: int
    composite_score: float
    sector: str
    industry: str
    composite_score_abs: float | None = None
    valuation_label: str = ""
    mispricing_score: float | None = None
    negative_earnings_flag: bool = False
    high_leverage_flag: bool = False
    market_cap_m: float | None = None
    long_eligible: bool = False
    short_eligible: bool = False
    relative_absolute_divergence: str = ""
    long_block_reason: str = ""
    short_block_reason: str = ""

    @property
    def identity_key(self) -> InstrumentKey:
        return instrument_key(self.symbol, self.exchange, self.cik)


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    direction: str
    score: float
    reason: str

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        direction = str(self.direction).strip().upper()
        reason = str(self.reason).strip()
        score = float(self.score)
        if not symbol:
            raise ValueError("TradeIntent.symbol is required")
        if direction not in {"LONG", "SHORT"}:
            raise ValueError("TradeIntent.direction must be LONG or SHORT")
        if not math.isfinite(score):
            raise ValueError("TradeIntent.score must be finite")
        if not reason:
            raise ValueError("TradeIntent.reason is required")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "reason", reason)


@dataclass
class TradePlan:
    symbol: str
    direction: str
    fundamental_score: float
    intent_score: float
    intent_reason: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    valuation_label: str = ""
    sector: str = ""
    industry: str = ""
    entry_ts: Optional[datetime] = None
    exchange: str = ""
    cik: int = 0

    @property
    def identity_key(self) -> InstrumentKey:
        return instrument_key(self.symbol, self.exchange, self.cik)


@dataclass
class IntentEvaluation:
    intent: Optional[TradeIntent]
    decision: str
    reason_code: str
    reason_text: str


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_list(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name, ",".join(default))
    return [x.strip() for x in raw.split(",") if x.strip()]


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
