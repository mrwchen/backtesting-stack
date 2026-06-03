"""Hysteresis state for regime-change risk management."""

from __future__ import annotations

from dataclasses import dataclass

from .config import (
    REGIME_RISK_ELEVATED_CONFIRM_DAYS,
    REGIME_RISK_EXTREME_CONFIRM_DAYS,
    REGIME_RISK_HIGH_CONFIRM_DAYS,
    REGIME_RISK_NEUTRAL_IS_ELEVATED,
    REGIME_RISK_RECOVERY_CONFIRM_DAYS,
)


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


def classify_regime_risk_tier(label: str) -> tuple[int, str]:
    label_based = _label_tier(label)
    reason = f"regime_label {label} mapped to {_STATE_BY_TIER[label_based]}"
    return label_based, reason


class RegimeRiskTracker:
    """Promote quickly on confirmed stress, demote slowly after stable recovery."""

    def __init__(self) -> None:
        self.tier = NORMAL
        self.elevated_count = 0
        self.high_count = 0
        self.extreme_count = 0
        self.recovery_count = 0

    def update(self, label: str) -> RegimeRiskSnapshot:
        raw_tier, reason = classify_regime_risk_tier(label)

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
        elif self.tier > NORMAL and raw_tier < self.tier:
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
            regime_label=label,
            reason=reason,
            elevated_count=self.elevated_count,
            high_count=self.high_count,
            extreme_count=self.extreme_count,
            recovery_count=self.recovery_count,
        )
