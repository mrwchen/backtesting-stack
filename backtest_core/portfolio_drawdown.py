"""Portfolio drawdown exposure overlay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
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
)

NORMAL = 0
ELEVATED = 1
HIGH_STRESS = 2
EXTREME_STRESS = 3

_PORTFOLIO_STATE_BY_TIER = {
    NORMAL: "NORMAL",
    ELEVATED: "WARN",
    HIGH_STRESS: "STRESS",
    EXTREME_STRESS: "EXTREME",
}


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
