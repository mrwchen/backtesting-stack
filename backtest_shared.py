"""Public API exposed to pluggable backtest model files."""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

@dataclass(frozen=True)
class WorldRegime:
    day: object
    label: str
    score: float


@dataclass(frozen=True)
class FundamentalRow:
    isin: str
    symbol: str
    composite_score: float
    sector: str
    industry: str
    valuation_label: str = ""
    mispricing_score: float | None = None
    negative_earnings_flag: bool = False
    high_leverage_flag: bool = False
    market_cap_m: float | None = None


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Signal:
    symbol: str
    direction: str
    fundamental_score: float
    entry_score: float
    combined_score: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    pullback_pct: float
    rsi_1h: float
    volume_ratio: float
    entry_reason: str
    valuation_label: str = ""
    sector: str = ""
    industry: str = ""
    entry_ts: Optional[datetime] = None
    isin: str = ""


@dataclass
class SignalEvaluation:
    signal: Optional[Signal]
    decision: str
    reason_code: str
    reason_text: str
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit_1: Optional[float] = None
    take_profit_2: Optional[float] = None
    pullback_pct: Optional[float] = None
    rsi_1h: Optional[float] = None
    volume_ratio: Optional[float] = None
    entry_score: Optional[float] = None
    combined_score: Optional[float] = None


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
