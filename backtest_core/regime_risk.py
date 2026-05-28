"""Hysteresis state for regime-change risk management."""

from __future__ import annotations

from dataclasses import dataclass

from backtest_shared import WorldRegime

from .config import (
    REGIME_RISK_ELEVATED_CONFIRM_DAYS,
    REGIME_RISK_ELEVATED_EXIT_SCORE,
    REGIME_RISK_EXTREME_CONFIRM_DAYS,
    REGIME_RISK_HIGH_CONFIRM_DAYS,
    REGIME_RISK_HIGH_EXIT_SCORE,
    REGIME_RISK_NEUTRAL_IS_ELEVATED,
    REGIME_RISK_RECOVERY_CONFIRM_DAYS,
    SHOCK_STRESS_GUARD_ELEVATED_SCORE,
    SHOCK_STRESS_GUARD_EXTREME_SCORE,
    SHOCK_STRESS_GUARD_HIGH_SCORE,
)
from .shock_overlay import shock_stress_score


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


@dataclass(frozen=True)
class RegimeRiskSnapshot:
    state: str
    tier: int
    raw_tier: int
    shock_score: float
    regime_label: str
    reason: str
    elevated_count: int
    high_count: int
    extreme_count: int
    recovery_count: int


def _label_tier(label: str) -> int:
    normalized = str(label or "").strip().upper()
    if normalized == "RISK-OFF":
        return EXTREME_STRESS
    if normalized == "DEFENSIVE":
        return HIGH_STRESS
    if normalized == "NEUTRAL" and REGIME_RISK_NEUTRAL_IS_ELEVATED:
        return ELEVATED
    return NORMAL


def _score_tier(regime: WorldRegime) -> int:
    shock_score = shock_stress_score(regime)
    if shock_score >= SHOCK_STRESS_GUARD_EXTREME_SCORE:
        return EXTREME_STRESS
    if shock_score >= SHOCK_STRESS_GUARD_HIGH_SCORE:
        return HIGH_STRESS
    if shock_score >= SHOCK_STRESS_GUARD_ELEVATED_SCORE:
        return ELEVATED
    return NORMAL


def classify_regime_risk_tier(label: str, regime: WorldRegime) -> tuple[int, str, float]:
    shock_score = shock_stress_score(regime)
    label_based = _label_tier(label)
    score_based = _score_tier(regime)
    raw_tier = max(label_based, score_based)
    if raw_tier == score_based and score_based > label_based:
        reason = f"shock_score {shock_score:.2f} reached {_STATE_BY_TIER[score_based]}"
    elif label_based > NORMAL:
        reason = f"regime_label {label} mapped to {_STATE_BY_TIER[label_based]}"
    else:
        reason = f"regime_label {label} and shock_score {shock_score:.2f} mapped to NORMAL"
    return raw_tier, reason, shock_score


class RegimeRiskTracker:
    """Promote quickly on confirmed stress, demote slowly after stable recovery."""

    def __init__(self) -> None:
        self.tier = NORMAL
        self.elevated_count = 0
        self.high_count = 0
        self.extreme_count = 0
        self.recovery_count = 0

    def update(self, label: str, regime: WorldRegime) -> RegimeRiskSnapshot:
        raw_tier, reason, shock_score = classify_regime_risk_tier(label, regime)

        self.elevated_count = self.elevated_count + 1 if raw_tier >= ELEVATED else 0
        self.high_count = self.high_count + 1 if raw_tier >= HIGH_STRESS else 0
        self.extreme_count = self.extreme_count + 1 if raw_tier >= EXTREME_STRESS else 0

        promoted = False
        if self.extreme_count >= REGIME_RISK_EXTREME_CONFIRM_DAYS and self.tier < EXTREME_STRESS:
            self.tier = EXTREME_STRESS
            promoted = True
        elif self.high_count >= REGIME_RISK_HIGH_CONFIRM_DAYS and self.tier < HIGH_STRESS:
            self.tier = HIGH_STRESS
            promoted = True
        elif self.elevated_count >= REGIME_RISK_ELEVATED_CONFIRM_DAYS and self.tier < ELEVATED:
            self.tier = ELEVATED
            promoted = True

        state = _STATE_BY_TIER[self.tier]
        if promoted:
            self.recovery_count = 0
            reason = f"{reason}; confirmed after hysteresis"
        elif self.tier > NORMAL and raw_tier < self.tier and self._recovery_allowed(raw_tier, shock_score):
            self.recovery_count += 1
            if self.recovery_count >= REGIME_RISK_RECOVERY_CONFIRM_DAYS:
                previous_tier = self.tier
                self.tier -= 1
                self.recovery_count = 0
                state = "RECOVERY" if self.tier == NORMAL else _STATE_BY_TIER[self.tier]
                reason = (
                    f"{reason}; demoted from {_STATE_BY_TIER[previous_tier]} after "
                    f"{REGIME_RISK_RECOVERY_CONFIRM_DAYS} recovery days"
                )
            else:
                reason = (
                    f"{reason}; waiting for recovery confirmation "
                    f"{self.recovery_count}/{REGIME_RISK_RECOVERY_CONFIRM_DAYS}"
                )
        elif raw_tier >= self.tier:
            self.recovery_count = 0

        return RegimeRiskSnapshot(
            state=state,
            tier=self.tier,
            raw_tier=raw_tier,
            shock_score=shock_score,
            regime_label=label,
            reason=reason,
            elevated_count=self.elevated_count,
            high_count=self.high_count,
            extreme_count=self.extreme_count,
            recovery_count=self.recovery_count,
        )

    @staticmethod
    def _recovery_allowed(raw_tier: int, shock_score: float) -> bool:
        if raw_tier >= HIGH_STRESS:
            return False
        if raw_tier == ELEVATED:
            return shock_score < REGIME_RISK_HIGH_EXIT_SCORE
        return shock_score < REGIME_RISK_ELEVATED_EXIT_SCORE
