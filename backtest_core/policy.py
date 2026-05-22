"""Common portfolio and eligibility policy for all backtest models."""

from dataclasses import dataclass

from .config import (
    COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE,
    COMMON_FILTER_NEGATIVE_EARNINGS_LONG,
    COMMON_FILTER_NEGATIVE_EARNINGS_SHORT,
    COMMON_LONG_LABEL_BLOCKLIST,
    COMMON_LONG_MIN_FUNDAMENTAL,
    COMMON_MIN_MARKET_CAP_M,
    COMMON_SHORT_LABEL_BLOCKLIST,
    COMMON_SHORT_MAX_FUNDAMENTAL,
    REGIME_EXPOSURE_BUCKETS,
    REGIME_LONG_MAX_SCORE,
    REGIME_SHORT_MIN_SCORE,
    REGIME_STRONG_RISK_OFF_MIN_SCORE,
    REGIME_STRONG_RISK_ON_MAX_SCORE,
)


@dataclass(frozen=True)
class CommonPolicy:
    regime_strong_risk_on_max_score: float
    regime_long_max_score: float
    regime_short_min_score: float
    regime_strong_risk_off_min_score: float
    long_min_fundamental: float
    short_max_fundamental: float
    long_label_blocklist: tuple[str, ...]
    short_label_blocklist: tuple[str, ...]
    min_market_cap_m: float
    filter_high_leverage: bool
    filter_negative_earnings_long: bool
    filter_negative_earnings_short: bool


COMMON_POLICY = CommonPolicy(
    regime_strong_risk_on_max_score=REGIME_STRONG_RISK_ON_MAX_SCORE,
    regime_long_max_score=REGIME_LONG_MAX_SCORE,
    regime_short_min_score=REGIME_SHORT_MIN_SCORE,
    regime_strong_risk_off_min_score=REGIME_STRONG_RISK_OFF_MIN_SCORE,
    long_min_fundamental=COMMON_LONG_MIN_FUNDAMENTAL,
    short_max_fundamental=COMMON_SHORT_MAX_FUNDAMENTAL,
    long_label_blocklist=tuple(COMMON_LONG_LABEL_BLOCKLIST),
    short_label_blocklist=tuple(COMMON_SHORT_LABEL_BLOCKLIST),
    min_market_cap_m=COMMON_MIN_MARKET_CAP_M,
    filter_high_leverage=COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE,
    filter_negative_earnings_long=COMMON_FILTER_NEGATIVE_EARNINGS_LONG,
    filter_negative_earnings_short=COMMON_FILTER_NEGATIVE_EARNINGS_SHORT,
)


def regime_exposure_for_score(score: float, policy: CommonPolicy = COMMON_POLICY) -> tuple[str, dict]:
    if score < policy.regime_strong_risk_on_max_score:
        bucket_name = "strong_risk_on"
    elif score < policy.regime_long_max_score:
        bucket_name = "risk_on"
    elif score < policy.regime_short_min_score:
        bucket_name = "neutral"
    elif score < policy.regime_strong_risk_off_min_score:
        bucket_name = "risk_off"
    else:
        bucket_name = "strong_risk_off"
    return bucket_name, REGIME_EXPOSURE_BUCKETS[bucket_name]


def direction_risk_multiplier(exposure: dict, direction: str) -> float:
    return float(exposure[f"{direction.lower()}_risk_multiplier"])


def direction_max_positions(exposure: dict, direction: str) -> int:
    return int(exposure[f"max_{direction.lower()}_positions"])


def direction_filter_negative_earnings(direction: str, policy: CommonPolicy = COMMON_POLICY) -> bool:
    if direction == "LONG":
        return policy.filter_negative_earnings_long
    return policy.filter_negative_earnings_short


def candidate_policy_kwargs(policy: CommonPolicy = COMMON_POLICY) -> dict:
    return {
        "long_min_fundamental": policy.long_min_fundamental,
        "short_max_fundamental": policy.short_max_fundamental,
        "min_market_cap_m": policy.min_market_cap_m,
        "long_label_blocklist": list(policy.long_label_blocklist) or None,
        "short_label_blocklist": list(policy.short_label_blocklist) or None,
        "filter_high_leverage": policy.filter_high_leverage,
    }
