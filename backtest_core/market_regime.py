"""Fast market-regime and portfolio drawdown overlays."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import psycopg2
from psycopg2 import sql

from .config import (
    MARKET_REGIME_ELEVATED_DRAWDOWN_PCT,
    MARKET_REGIME_EXTREME_DRAWDOWN_PCT,
    MARKET_REGIME_GENERATE_HEDGE_ENABLED,
    MARKET_REGIME_GUARD_ENABLED,
    MARKET_REGIME_HEDGE_MIN_TIER,
    MARKET_REGIME_HIGH_DRAWDOWN_PCT,
    MARKET_REGIME_LONG_RISK_MULTIPLIER_ELEVATED,
    MARKET_REGIME_LONG_RISK_MULTIPLIER_EXTREME,
    MARKET_REGIME_LONG_RISK_MULTIPLIER_HIGH,
    MARKET_REGIME_LOOKBACK_DAYS,
    MARKET_REGIME_MA_CONFIRM_ENABLED,
    MARKET_REGIME_MA_CONFIRM_TIER,
    MARKET_REGIME_MA_LONG_DAYS,
    MARKET_REGIME_MA_SHORT_DAYS,
    MARKET_REGIME_MAX_LONG_POSITIONS_ELEVATED,
    MARKET_REGIME_MAX_LONG_POSITIONS_EXTREME,
    MARKET_REGIME_MAX_LONG_POSITIONS_HIGH,
    MARKET_REGIME_MIN_HISTORY_DAYS,
    MARKET_REGIME_SHORT_HEDGE_ENABLED,
    MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_ELEVATED,
    MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_EXTREME,
    MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_HIGH,
    MARKET_REGIME_SHORT_HEDGE_MIN_TIER,
    MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_ELEVATED,
    MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_EXTREME,
    MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_HIGH,
    MARKET_REGIME_SYMBOL,
    MARKET_REGIME_TZ,
    PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER_ENABLED,
    PORTFOLIO_DRAWDOWN_EXTREME_PCT,
    PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_EXTREME,
    PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_STRESS,
    PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_WARN,
    PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_EXTREME,
    PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_STRESS,
    PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_WARN,
    PORTFOLIO_DRAWDOWN_STRESS_PCT,
    PORTFOLIO_DRAWDOWN_WARN_PCT,
    SOURCE_MARKET_DATA_1H_TABLE,
)
from .market_data import _ensure_utc_ts
from .sql_utils import relation_identifier

NORMAL = 0
ELEVATED = 1
HIGH_STRESS = 2
EXTREME_STRESS = 3

_STATE_BY_TIER = {
    NORMAL: "NORMAL",
    ELEVATED: "ELEVATED",
    HIGH_STRESS: "HIGH_STRESS",
    EXTREME_STRESS: "EXTREME_STRESS",
}

_PORTFOLIO_STATE_BY_TIER = {
    NORMAL: "NORMAL",
    ELEVATED: "WARN",
    HIGH_STRESS: "STRESS",
    EXTREME_STRESS: "EXTREME",
}

_MARKET_REGIME_CACHE: dict[tuple[str, str, str, datetime], "MarketRegimeSnapshot"] = {}


@dataclass(frozen=True)
class MarketRegimeSnapshot:
    enabled: bool
    symbol: str
    as_of_ts: datetime
    state: str
    tier: int
    history_days: int
    close: float | None
    peak_close: float | None
    drawdown_pct: float | None
    ma_short: float | None
    ma_long: float | None
    reason: str


@dataclass(frozen=True)
class PortfolioDrawdownSnapshot:
    enabled: bool
    state: str
    tier: int
    equity: float
    peak_equity: float
    drawdown_pct: float
    reason: str


def _copy_exposure(exposure: dict[str, Any]) -> dict[str, Any]:
    return {
        "long_risk_multiplier": float(exposure.get("long_risk_multiplier", 0.0)),
        "short_risk_multiplier": float(exposure.get("short_risk_multiplier", 0.0)),
        "max_long_positions": int(exposure.get("max_long_positions", 0)),
        "max_short_positions": int(exposure.get("max_short_positions", 0)),
    }


def _tier_float(tier: int, elevated: float, high: float, extreme: float, normal: float = 1.0) -> float:
    if tier >= EXTREME_STRESS:
        return extreme
    if tier >= HIGH_STRESS:
        return high
    if tier >= ELEVATED:
        return elevated
    return normal


def _tier_int(tier: int, elevated: int, high: int, extreme: int, normal: int) -> int:
    if tier >= EXTREME_STRESS:
        return extreme
    if tier >= HIGH_STRESS:
        return high
    if tier >= ELEVATED:
        return elevated
    return normal


def _tier_from_drawdown(abs_drawdown_pct: float, elevated: float, high: float, extreme: float) -> int:
    if abs_drawdown_pct >= extreme:
        return EXTREME_STRESS
    if abs_drawdown_pct >= high:
        return HIGH_STRESS
    if abs_drawdown_pct >= elevated:
        return ELEVATED
    return NORMAL


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def get_market_regime_snapshot(
    conn: psycopg2.extensions.connection,
    as_of_ts: datetime,
) -> MarketRegimeSnapshot:
    as_of_ts = _ensure_utc_ts(as_of_ts)
    symbol = MARKET_REGIME_SYMBOL.strip().upper()
    cache_key = (SOURCE_MARKET_DATA_1H_TABLE, symbol, MARKET_REGIME_TZ, as_of_ts)
    cached = _MARKET_REGIME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if not MARKET_REGIME_GUARD_ENABLED or not symbol:
        snapshot = MarketRegimeSnapshot(
            enabled=False,
            symbol=symbol,
            as_of_ts=as_of_ts,
            state="DISABLED",
            tier=NORMAL,
            history_days=0,
            close=None,
            peak_close=None,
            drawdown_pct=None,
            ma_short=None,
            ma_long=None,
            reason="Market-regime guard disabled.",
        )
        _MARKET_REGIME_CACHE[cache_key] = snapshot
        return snapshot

    lookback_days = max(MARKET_REGIME_LOOKBACK_DAYS, MARKET_REGIME_MA_LONG_DAYS, MARKET_REGIME_MIN_HISTORY_DAYS)
    start_ts = as_of_ts - timedelta(days=max(90, lookback_days * 3))
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH daily AS (
                    SELECT DISTINCT ON ((ts AT TIME ZONE %s)::date)
                        (ts AT TIME ZONE %s)::date AS local_day,
                        close::double precision AS close,
                        ts
                    FROM {}
                    WHERE symbol = %s
                      AND ts >= %s
                      AND ts <= %s
                    ORDER BY (ts AT TIME ZONE %s)::date, ts DESC
                )
                SELECT local_day, close
                FROM daily
                ORDER BY local_day
                """
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (MARKET_REGIME_TZ, MARKET_REGIME_TZ, symbol, start_ts, as_of_ts, MARKET_REGIME_TZ),
        )
        rows = cur.fetchall()

    closes = [float(row[1]) for row in rows if row[1] is not None]
    history_days = len(closes)
    if history_days < MARKET_REGIME_MIN_HISTORY_DAYS:
        snapshot = MarketRegimeSnapshot(
            enabled=True,
            symbol=symbol,
            as_of_ts=as_of_ts,
            state="INSUFFICIENT_HISTORY",
            tier=NORMAL,
            history_days=history_days,
            close=closes[-1] if closes else None,
            peak_close=max(closes) if closes else None,
            drawdown_pct=None,
            ma_short=None,
            ma_long=None,
            reason=f"Only {history_days} daily closes available; {MARKET_REGIME_MIN_HISTORY_DAYS} required.",
        )
        _MARKET_REGIME_CACHE[cache_key] = snapshot
        return snapshot

    window = closes[-MARKET_REGIME_LOOKBACK_DAYS:]
    close = closes[-1]
    peak_close = max(window)
    drawdown_pct = (close / peak_close - 1.0) * 100.0 if peak_close > 0.0 else 0.0
    tier = _tier_from_drawdown(
        abs(min(0.0, drawdown_pct)),
        MARKET_REGIME_ELEVATED_DRAWDOWN_PCT,
        MARKET_REGIME_HIGH_DRAWDOWN_PCT,
        MARKET_REGIME_EXTREME_DRAWDOWN_PCT,
    )
    reason_parts = [f"{symbol} {MARKET_REGIME_LOOKBACK_DAYS}d drawdown {drawdown_pct:.2f}%"]

    ma_short = _average(closes[-MARKET_REGIME_MA_SHORT_DAYS:]) if history_days >= MARKET_REGIME_MA_SHORT_DAYS else None
    ma_long = _average(closes[-MARKET_REGIME_MA_LONG_DAYS:]) if history_days >= MARKET_REGIME_MA_LONG_DAYS else None
    if (
        MARKET_REGIME_MA_CONFIRM_ENABLED
        and ma_short is not None
        and ma_long is not None
        and close < ma_long
        and ma_short < ma_long
        and MARKET_REGIME_MA_CONFIRM_TIER > tier
    ):
        tier = MARKET_REGIME_MA_CONFIRM_TIER
        reason_parts.append(
            f"MA confirmation close {close:.2f} below long MA {ma_long:.2f} and short MA {ma_short:.2f} below long MA"
        )

    snapshot = MarketRegimeSnapshot(
        enabled=True,
        symbol=symbol,
        as_of_ts=as_of_ts,
        state=_STATE_BY_TIER[tier],
        tier=tier,
        history_days=history_days,
        close=close,
        peak_close=peak_close,
        drawdown_pct=drawdown_pct,
        ma_short=ma_short,
        ma_long=ma_long,
        reason="; ".join(reason_parts),
    )
    _MARKET_REGIME_CACHE[cache_key] = snapshot
    return snapshot


def apply_market_regime_exposure_overlay(
    exposure: dict[str, Any],
    snapshot: MarketRegimeSnapshot,
) -> dict[str, Any]:
    adjusted = _copy_exposure(exposure)
    if not snapshot.enabled or snapshot.tier <= NORMAL:
        return adjusted

    long_multiplier = _tier_float(
        snapshot.tier,
        MARKET_REGIME_LONG_RISK_MULTIPLIER_ELEVATED,
        MARKET_REGIME_LONG_RISK_MULTIPLIER_HIGH,
        MARKET_REGIME_LONG_RISK_MULTIPLIER_EXTREME,
    )
    long_cap = _tier_int(
        snapshot.tier,
        MARKET_REGIME_MAX_LONG_POSITIONS_ELEVATED,
        MARKET_REGIME_MAX_LONG_POSITIONS_HIGH,
        MARKET_REGIME_MAX_LONG_POSITIONS_EXTREME,
        adjusted["max_long_positions"],
    )
    adjusted["long_risk_multiplier"] *= long_multiplier
    adjusted["max_long_positions"] = min(adjusted["max_long_positions"], long_cap)

    if MARKET_REGIME_SHORT_HEDGE_ENABLED and snapshot.tier >= MARKET_REGIME_SHORT_HEDGE_MIN_TIER:
        short_risk = _tier_float(
            snapshot.tier,
            MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_ELEVATED,
            MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_HIGH,
            MARKET_REGIME_SHORT_HEDGE_RISK_MULTIPLIER_EXTREME,
            0.0,
        )
        short_cap = _tier_int(
            snapshot.tier,
            MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_ELEVATED,
            MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_HIGH,
            MARKET_REGIME_SHORT_HEDGE_MAX_POSITIONS_EXTREME,
            0,
        )
        adjusted["short_risk_multiplier"] = max(adjusted["short_risk_multiplier"], short_risk)
        adjusted["max_short_positions"] = max(adjusted["max_short_positions"], short_cap)

    return adjusted


def market_regime_generated_hedge_active(snapshot: MarketRegimeSnapshot) -> bool:
    return (
        MARKET_REGIME_GENERATE_HEDGE_ENABLED
        and snapshot.enabled
        and snapshot.tier >= MARKET_REGIME_HEDGE_MIN_TIER
    )


def get_portfolio_drawdown_snapshot(
    equity: float,
    peak_equity: float,
) -> PortfolioDrawdownSnapshot:
    equity = float(equity)
    peak_equity = max(float(peak_equity), equity)
    if not PORTFOLIO_DRAWDOWN_CIRCUIT_BREAKER_ENABLED or peak_equity <= 0.0:
        return PortfolioDrawdownSnapshot(
            enabled=False,
            state="DISABLED",
            tier=NORMAL,
            equity=equity,
            peak_equity=peak_equity,
            drawdown_pct=0.0,
            reason="Portfolio drawdown circuit breaker disabled.",
        )

    drawdown_pct = (equity / peak_equity - 1.0) * 100.0
    tier = _tier_from_drawdown(
        abs(min(0.0, drawdown_pct)),
        PORTFOLIO_DRAWDOWN_WARN_PCT,
        PORTFOLIO_DRAWDOWN_STRESS_PCT,
        PORTFOLIO_DRAWDOWN_EXTREME_PCT,
    )
    return PortfolioDrawdownSnapshot(
        enabled=True,
        state=_PORTFOLIO_STATE_BY_TIER[tier],
        tier=tier,
        equity=equity,
        peak_equity=peak_equity,
        drawdown_pct=drawdown_pct,
        reason=f"Portfolio drawdown {drawdown_pct:.2f}% from peak equity {peak_equity:.2f}.",
    )


def apply_portfolio_drawdown_exposure_overlay(
    exposure: dict[str, Any],
    snapshot: PortfolioDrawdownSnapshot,
) -> dict[str, Any]:
    adjusted = _copy_exposure(exposure)
    if not snapshot.enabled or snapshot.tier <= NORMAL:
        return adjusted

    long_multiplier = _tier_float(
        snapshot.tier,
        PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_WARN,
        PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_STRESS,
        PORTFOLIO_DRAWDOWN_LONG_RISK_MULTIPLIER_EXTREME,
    )
    long_cap = _tier_int(
        snapshot.tier,
        PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_WARN,
        PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_STRESS,
        PORTFOLIO_DRAWDOWN_MAX_LONG_POSITIONS_EXTREME,
        adjusted["max_long_positions"],
    )
    adjusted["long_risk_multiplier"] *= long_multiplier
    adjusted["max_long_positions"] = min(adjusted["max_long_positions"], long_cap)
    return adjusted
