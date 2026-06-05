"""Common portfolio and eligibility policy for all backtest models."""

from dataclasses import dataclass

from .config import (
    COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE,
    COMMON_FILTER_NEGATIVE_EARNINGS_LONG,
    COMMON_FILTER_NEGATIVE_EARNINGS_SHORT,
    COMMON_FILTER_SCORER_ELIGIBILITY,
    COMMON_REQUIRE_UPCOMING_EARNINGS_DATE,
    COMMON_LONG_LABEL_BLOCKLIST,
    COMMON_LONG_MIN_FUNDAMENTAL,
    COMMON_MIN_MARKET_CAP_M,
    COMMON_SHORT_LABEL_BLOCKLIST,
    COMMON_SHORT_MAX_FUNDAMENTAL,
    MAX_OPEN_POSITIONS,
)


@dataclass(frozen=True)
class CommonPolicy:
    long_min_fundamental: float
    short_max_fundamental: float
    long_label_blocklist: tuple[str, ...]
    short_label_blocklist: tuple[str, ...]
    min_market_cap_m: float
    filter_high_leverage: bool
    filter_negative_earnings_long: bool
    filter_negative_earnings_short: bool
    filter_scorer_eligibility: bool
    require_upcoming_earnings_date: bool


COMMON_POLICY = CommonPolicy(
    long_min_fundamental=COMMON_LONG_MIN_FUNDAMENTAL,
    short_max_fundamental=COMMON_SHORT_MAX_FUNDAMENTAL,
    long_label_blocklist=tuple(COMMON_LONG_LABEL_BLOCKLIST),
    short_label_blocklist=tuple(COMMON_SHORT_LABEL_BLOCKLIST),
    min_market_cap_m=COMMON_MIN_MARKET_CAP_M,
    filter_high_leverage=COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE,
    filter_negative_earnings_long=COMMON_FILTER_NEGATIVE_EARNINGS_LONG,
    filter_negative_earnings_short=COMMON_FILTER_NEGATIVE_EARNINGS_SHORT,
    filter_scorer_eligibility=COMMON_FILTER_SCORER_ELIGIBILITY,
    require_upcoming_earnings_date=COMMON_REQUIRE_UPCOMING_EARNINGS_DATE,
)


def direction_risk_multiplier(exposure: dict, direction: str) -> float:
    return float(exposure[f"{direction.lower()}_risk_multiplier"])


def direction_max_positions(exposure: dict, direction: str) -> int:
    return int(exposure[f"max_{direction.lower()}_positions"])


def exposure_max_total_positions(exposure: dict) -> int:
    return int(exposure.get("max_total_positions", MAX_OPEN_POSITIONS))


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
        "filter_scorer_eligibility": policy.filter_scorer_eligibility,
        "require_upcoming_earnings_date": policy.require_upcoming_earnings_date,
    }
