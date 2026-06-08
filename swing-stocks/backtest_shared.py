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
    dominant_shock_type: str = ""
    max_shock_type_score: float | None = None
    defensive_risk_off_score: float | None = None
    energy_commodity_shock_score: float | None = None
    rates_inflation_usd_shock_score: float | None = None
    credit_banking_stress_score: float | None = None
    policy_geopolitical_score: float | None = None
    tech_stress_shock_score: float | None = None
    precious_metals_score: float | None = None
    industrial_metals_score: float | None = None
    metals_mining_shock_score: float | None = None
    metals_mining_subtype: str = ""


@dataclass(frozen=True)
class CandidateRow:
    symbol: str
    exchange: str
    cik: int
    broker_eligibility_bypassed: bool = False

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
    intent_score: float
    intent_reason: str
    entry_price: float
    stop_loss: float
    take_profit_mode: str
    take_profit: Optional[float]
    trailing_activation_price: Optional[float]
    trailing_distance_pct: Optional[float]
    entry_ts: Optional[datetime] = None
    exchange: str = ""
    cik: int = 0
    broker_eligibility_bypassed: bool = False
    allow_multiple_positions_per_instrument: bool = False

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


def env_optional_float(name: str, default: Optional[float]) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return None
    return float(text)


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    return text or default


def env_list(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name, ",".join(default))
    return [x.strip() for x in raw.split(",") if x.strip()]


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 50.0
    gain_sum = 0.0
    loss_sum = 0.0
    for i in range(1, period + 1):
        delta = closes[i] - closes[i - 1]
        if delta >= 0.0:
            gain_sum += delta
        else:
            loss_sum -= delta
    avg_gain = gain_sum / period
    avg_loss = loss_sum / period
    for i in range(period + 1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gain = delta if delta > 0.0 else 0.0
        loss = -delta if delta < 0.0 else 0.0
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
