"""Environment-backed configuration for one backtest process."""

import dataclasses
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from backtest_shared import env_bool, env_float, env_int, env_list, env_str
from . import runtime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

START_DATE             = date.fromisoformat(os.getenv("START_DATE", "2023-01-01"))
END_DATE               = date.fromisoformat(os.getenv("END_DATE", str(date.today())))
if env_bool("ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS", False):
    raise ValueError(
        "ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS=true is disabled; backtests must use point-in-time data_available_at guards."
    )
ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS = False
ACCOUNT_PROFILE_REQUEST = os.getenv("ACCOUNT_PROFILE", "ps_acc").strip().lower()
ACCOUNT_PROFILE        = ACCOUNT_PROFILE_REQUEST
INITIAL_EQUITY         = float(os.getenv("INITIAL_EQUITY_USD", "100000.0"))
RISK_PER_TRADE_PCT     = float(os.getenv("RISK_PER_TRADE_EQUITY_PCT", "2.0"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
MAX_POSITION_OPENS_PER_DAY = env_int("MAX_POSITION_OPENS_PER_DAY", MAX_OPEN_POSITIONS)
MAX_POSITION_OPENS_PER_HOUR = env_int("MAX_POSITION_OPENS_PER_HOUR", MAX_OPEN_POSITIONS)
if MAX_POSITION_OPENS_PER_DAY < 0:
    raise ValueError("MAX_POSITION_OPENS_PER_DAY must be >= 0")
if MAX_POSITION_OPENS_PER_HOUR < 0:
    raise ValueError("MAX_POSITION_OPENS_PER_HOUR must be >= 0")


def _default_position_cap(ratio: float) -> int:
    return max(0, min(MAX_OPEN_POSITIONS, int(round(MAX_OPEN_POSITIONS * ratio))))


def _regime_risk_multiplier(env_key: str, default: float) -> float:
    value = env_float(env_key, default)
    if value < 0.0:
        raise ValueError(f"{env_key} must be >= 0")
    return value


def _regime_position_cap(env_key: str, default: int) -> int:
    value = env_int(env_key, default)
    if value < 0:
        raise ValueError(f"{env_key} must be >= 0")
    return value


REGIME_EXPOSURE_BY_LABEL = {
    "RISK-ON": {
        "long_risk_multiplier": _regime_risk_multiplier("REGIME_RISK_ON_LONG_RISK_MULTIPLIER", 1.0),
        "short_risk_multiplier": _regime_risk_multiplier("REGIME_RISK_ON_SHORT_RISK_MULTIPLIER", 0.0),
        "max_long_positions": _regime_position_cap("REGIME_RISK_ON_MAX_LONG_POSITIONS", MAX_OPEN_POSITIONS),
        "max_short_positions": _regime_position_cap("REGIME_RISK_ON_MAX_SHORT_POSITIONS", 0),
    },
    "CONSTRUCTIVE": {
        "long_risk_multiplier": _regime_risk_multiplier("REGIME_CONSTRUCTIVE_LONG_RISK_MULTIPLIER", 1.0),
        "short_risk_multiplier": _regime_risk_multiplier("REGIME_CONSTRUCTIVE_SHORT_RISK_MULTIPLIER", 0.0),
        "max_long_positions": _regime_position_cap("REGIME_CONSTRUCTIVE_MAX_LONG_POSITIONS", MAX_OPEN_POSITIONS),
        "max_short_positions": _regime_position_cap("REGIME_CONSTRUCTIVE_MAX_SHORT_POSITIONS", 0),
    },
    "NEUTRAL": {
        "long_risk_multiplier": _regime_risk_multiplier("REGIME_NEUTRAL_LONG_RISK_MULTIPLIER", 0.40),
        "short_risk_multiplier": _regime_risk_multiplier("REGIME_NEUTRAL_SHORT_RISK_MULTIPLIER", 0.60),
        "max_long_positions": _regime_position_cap("REGIME_NEUTRAL_MAX_LONG_POSITIONS", _default_position_cap(0.3)),
        "max_short_positions": _regime_position_cap("REGIME_NEUTRAL_MAX_SHORT_POSITIONS", _default_position_cap(0.3)),
    },
    "DEFENSIVE": {
        "long_risk_multiplier": _regime_risk_multiplier("REGIME_DEFENSIVE_LONG_RISK_MULTIPLIER", 0.25),
        "short_risk_multiplier": _regime_risk_multiplier("REGIME_DEFENSIVE_SHORT_RISK_MULTIPLIER", 0.75),
        "max_long_positions": _regime_position_cap("REGIME_DEFENSIVE_MAX_LONG_POSITIONS", MAX_OPEN_POSITIONS - _default_position_cap(0.8)),
        "max_short_positions": _regime_position_cap("REGIME_DEFENSIVE_MAX_SHORT_POSITIONS", _default_position_cap(0.8)),
    },
    "RISK-OFF": {
        "long_risk_multiplier": _regime_risk_multiplier("REGIME_RISK_OFF_LONG_RISK_MULTIPLIER", 0.0),
        "short_risk_multiplier": _regime_risk_multiplier("REGIME_RISK_OFF_SHORT_RISK_MULTIPLIER", 1.0),
        "max_long_positions": _regime_position_cap("REGIME_RISK_OFF_MAX_LONG_POSITIONS", 0),
        "max_short_positions": _regime_position_cap("REGIME_RISK_OFF_MAX_SHORT_POSITIONS", MAX_OPEN_POSITIONS),
    },
}

ACCOUNT_PROFILE_DEFAULTS = {
    # Pepperstone EU retail US Share/ETF CFDs:
    # 5:1 share leverage => 20% margin; 0.02 USD/share per side, 0.02 USD minimum;
    # direct underlying exchange prices without an extra Pepperstone spread mark-up.
    # Share CFD overnight funding is charged at the 5pm New York rollover on full
    # active notional, with Friday carrying the weekend financing.
    "ps_acc": {
        "margin_requirement_pct": 20.0,
        "commission_per_order_usd": 0.0,
        "commission_per_share_usd": 0.02,
        "commission_min_per_order_usd": 0.02,
        "commission_max_pct": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 1.0,
        "share_cfd_arr_pct": 5.0,
        "share_cfd_admin_fee_pct": 2.5,
        "share_cfd_short_borrow_rate_pct": 0.0,
        "share_cfd_overnight_day_count": 360.0,
        "allow_fractional_shares": False,
    },
    # IBKR Pro Tiered US stocks:
    # Europe retail securities account approximation:
    # available funds uses initial margin; excess liquidity uses maintenance margin.
    # first tier commission and configured first-tier USD margin loan rate.
    "ibkr_acc": {
        "long_initial_margin_pct": 50.0,
        "long_maintenance_margin_pct": 25.0,
        "short_initial_margin_pct": 50.0,
        "short_maintenance_margin_pct": 50.0,
        "commission_per_order_usd": 0.0,
        "commission_per_share_usd": 0.0035,
        "commission_min_per_order_usd": 0.35,
        "commission_max_pct": 1.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 1.0,
        "margin_financing_rate_pct": 5.14,
        "allow_fractional_shares": True,
    },
}
if ACCOUNT_PROFILE_REQUEST == "all":
    ACCOUNT_PROFILE = "ps_acc"
elif ACCOUNT_PROFILE_REQUEST not in ACCOUNT_PROFILE_DEFAULTS:
    raise ValueError("ACCOUNT_PROFILE must be one of: ps_acc, ibkr_acc, all")
_ACC = ACCOUNT_PROFILE_DEFAULTS[ACCOUNT_PROFILE]
_ACC_ENV_PREFIX = {
    "ps_acc": "PS",
    "ibkr_acc": "IBKR",
}[ACCOUNT_PROFILE]

def _account_float(env_key: str, default_key: str) -> float:
    prefixed_key = f"{_ACC_ENV_PREFIX}_{env_key}"
    raw = os.getenv(prefixed_key)
    if raw is None:
        raw = str(_ACC[default_key])
    return float(raw)

def _account_bool(env_key: str, default_key: str) -> bool:
    prefixed_key = f"{_ACC_ENV_PREFIX}_{env_key}"
    raw = os.getenv(prefixed_key)
    if raw is None:
        return bool(_ACC[default_key])
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _account_setting(env_key: str, default: str) -> str:
    raw = os.getenv(f"{_ACC_ENV_PREFIX}_{env_key}")
    if raw is None:
        raw = os.getenv(env_key)
    return default if raw is None else raw

def _account_setting_bool(env_key: str, default: bool) -> bool:
    raw = os.getenv(f"{_ACC_ENV_PREFIX}_{env_key}")
    if raw is None:
        raw = os.getenv(env_key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

def _parse_window_setting(env_key: str, value: str) -> tuple[str, str]:
    start, sep, end = value.partition("-")
    if not sep or not start.strip() or not end.strip():
        raise ValueError(f"{env_key} must use HH:MM-HH:MM format")
    return start.strip(), end.strip()

def _account_window_setting(env_key: str) -> tuple[str, str]:
    prefixed_key = f"{_ACC_ENV_PREFIX}_{env_key}"
    raw = os.getenv(prefixed_key)
    if raw is None:
        raise ValueError(f"{prefixed_key} is required and must use HH:MM-HH:MM format")
    return _parse_window_setting(prefixed_key, raw)

def _account_window_setting_default(env_key: str, default: str) -> tuple[str, str]:
    prefixed_key = f"{_ACC_ENV_PREFIX}_{env_key}"
    raw = os.getenv(prefixed_key)
    if raw is None:
        raw = os.getenv(env_key, default)
    return _parse_window_setting(prefixed_key if os.getenv(prefixed_key) is not None else env_key, raw)

MARGIN_REQUIREMENT_PCT = _account_float("MARGIN_REQUIREMENT_PCT", "margin_requirement_pct") if "margin_requirement_pct" in _ACC else None
IBKR_LONG_INITIAL_MARGIN_PCT = _account_float("LONG_INITIAL_MARGIN_PCT", "long_initial_margin_pct") if ACCOUNT_PROFILE == "ibkr_acc" else None
IBKR_LONG_MAINTENANCE_MARGIN_PCT = _account_float("LONG_MAINTENANCE_MARGIN_PCT", "long_maintenance_margin_pct") if ACCOUNT_PROFILE == "ibkr_acc" else None
IBKR_SHORT_INITIAL_MARGIN_PCT = _account_float("SHORT_INITIAL_MARGIN_PCT", "short_initial_margin_pct") if ACCOUNT_PROFILE == "ibkr_acc" else None
IBKR_SHORT_MAINTENANCE_MARGIN_PCT = _account_float("SHORT_MAINTENANCE_MARGIN_PCT", "short_maintenance_margin_pct") if ACCOUNT_PROFILE == "ibkr_acc" else None
COMMISSION_PER_ORDER_USD = _account_float("COMMISSION_PER_ORDER_USD", "commission_per_order_usd")
COMMISSION_PER_SHARE_USD = _account_float("COMMISSION_PER_SHARE_USD", "commission_per_share_usd")
COMMISSION_MIN_PER_ORDER_USD = _account_float("COMMISSION_MIN_PER_ORDER_USD", "commission_min_per_order_usd")
COMMISSION_MAX_PCT = _account_float("COMMISSION_MAX_PCT", "commission_max_pct")
COMMISSION_BPS        = _account_float("COMMISSION_BPS", "commission_bps")
SPREAD_BPS            = _account_float("SPREAD_BPS", "spread_bps")
SLIPPAGE_BPS          = _account_float("SLIPPAGE_BPS", "slippage_bps")
MARGIN_FINANCING_RATE_PCT = _account_float("MARGIN_FINANCING_RATE_PCT", "margin_financing_rate_pct") if "margin_financing_rate_pct" in _ACC else 0.0
ALLOW_FRACTIONAL_SHARES = _account_bool("ALLOW_FRACTIONAL_SHARES", "allow_fractional_shares")
PS_SHARE_CFD_ARR_PCT = float(os.getenv("PS_SHARE_CFD_ARR_PCT", str(_ACC.get("share_cfd_arr_pct", 0.0)))) if ACCOUNT_PROFILE == "ps_acc" else 0.0
PS_SHARE_CFD_ADMIN_FEE_PCT = float(os.getenv("PS_SHARE_CFD_ADMIN_FEE_PCT", str(_ACC.get("share_cfd_admin_fee_pct", 0.0)))) if ACCOUNT_PROFILE == "ps_acc" else 0.0
PS_SHARE_CFD_SHORT_BORROW_RATE_PCT = float(os.getenv("PS_SHARE_CFD_SHORT_BORROW_RATE_PCT", str(_ACC.get("share_cfd_short_borrow_rate_pct", 0.0)))) if ACCOUNT_PROFILE == "ps_acc" else 0.0
PS_SHARE_CFD_OVERNIGHT_DAY_COUNT = float(os.getenv("PS_SHARE_CFD_OVERNIGHT_DAY_COUNT", str(_ACC.get("share_cfd_overnight_day_count", 360.0)))) if ACCOUNT_PROFILE == "ps_acc" else 360.0
PS_MARGIN_STOP_OUT_LEVEL_PCT = float(os.getenv("PS_MARGIN_STOP_OUT_LEVEL_PCT", "50.0"))
PS_MIN_ENTRY_MARGIN_LEVEL_PCT = float(os.getenv("PS_MIN_ENTRY_MARGIN_LEVEL_PCT", "100.0"))
if PS_MARGIN_STOP_OUT_LEVEL_PCT < 0.0:
    raise ValueError("PS_MARGIN_STOP_OUT_LEVEL_PCT must be >= 0")
if PS_MIN_ENTRY_MARGIN_LEVEL_PCT < 0.0:
    raise ValueError("PS_MIN_ENTRY_MARGIN_LEVEL_PCT must be >= 0")
if ACCOUNT_PROFILE == "ps_acc":
    for _name, _value in {
        "PS_SHARE_CFD_OVERNIGHT_DAY_COUNT": PS_SHARE_CFD_OVERNIGHT_DAY_COUNT,
    }.items():
        if _value <= 0.0:
            raise ValueError(f"{_name} must be > 0")
for _name, _value in {
    "IBKR_LONG_INITIAL_MARGIN_PCT": IBKR_LONG_INITIAL_MARGIN_PCT,
    "IBKR_LONG_MAINTENANCE_MARGIN_PCT": IBKR_LONG_MAINTENANCE_MARGIN_PCT,
    "IBKR_SHORT_INITIAL_MARGIN_PCT": IBKR_SHORT_INITIAL_MARGIN_PCT,
    "IBKR_SHORT_MAINTENANCE_MARGIN_PCT": IBKR_SHORT_MAINTENANCE_MARGIN_PCT,
}.items():
    if _value is not None and _value < 0.0:
        raise ValueError(f"{_name} must be >= 0")
RUN_NOTES_EXTRA        = os.getenv("RUN_NOTES_EXTRA", "")
RUN_LABEL_TZ           = os.getenv("RUN_LABEL_TZ", "Europe/Berlin")
PROGRESS_LOG_EVERY_DAYS = max(1, int(os.getenv("PROGRESS_LOG_EVERY_DAYS", "25")))
DECISION_EVENT_FLUSH_BATCH_SIZE = max(1, env_int("DECISION_EVENT_FLUSH_BATCH_SIZE", 5000))
DECISION_EVENT_MODE = os.getenv("DECISION_EVENT_MODE", "all").strip().lower()
if DECISION_EVENT_MODE not in {"all", "signals", "summary", "none"}:
    raise ValueError("DECISION_EVENT_MODE must be one of: all, signals, summary, none")
TRADE_INSERT_PAGE_SIZE = max(1, env_int("TRADE_INSERT_PAGE_SIZE", 1000))
ACCOUNT_CURVE_INSERT_PAGE_SIZE = max(1, env_int("ACCOUNT_CURVE_INSERT_PAGE_SIZE", 2000))
DECISION_EVENT_INSERT_PAGE_SIZE = max(1, env_int("DECISION_EVENT_INSERT_PAGE_SIZE", 2500))
BAR_CACHE_WARMUP_DAYS = max(1, int(os.getenv("BAR_CACHE_WARMUP_DAYS", "120")))
BAR_CACHE_BATCH_SIZE = max(1, int(os.getenv("BAR_CACHE_BATCH_SIZE", "100")))
BAR_CACHE_MAX_MIB = max(128, env_int("BAR_CACHE_MAX_MIB", 1024))
SIGNAL_BAR_MAX_STALENESS_HOURS = env_float("SIGNAL_BAR_MAX_STALENESS_HOURS", 2.0)
if SIGNAL_BAR_MAX_STALENESS_HOURS < 0.0:
    raise ValueError("SIGNAL_BAR_MAX_STALENESS_HOURS must be >= 0")
SIGNAL_BAR_CACHE_ENABLED = env_bool("SIGNAL_BAR_CACHE_ENABLED", True)
SIGNAL_BAR_CACHE_MAX_MIB = max(128, env_int("SIGNAL_BAR_CACHE_MAX_MIB", 2048))
CANDIDATE_TIMELINE_CACHE_ENABLED = env_bool("CANDIDATE_TIMELINE_CACHE_ENABLED", True)
CANDIDATE_TIMELINE_CACHE_MAX_MIB = max(128, env_int("CANDIDATE_TIMELINE_CACHE_MAX_MIB", 1024))
CANDIDATE_TIMELINE_CURSOR_ITERSIZE = max(1000, env_int("CANDIDATE_TIMELINE_CURSOR_ITERSIZE", 10000))
CANDIDATE_TIMELINE_SHARED_CACHE_DIR = os.getenv(
    "CANDIDATE_TIMELINE_SHARED_CACHE_DIR",
    "/tmp/backtest_candidate_timeline_cache",
)
MONTE_CARLO_ENABLED       = os.getenv("MONTE_CARLO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
MONTE_CARLO_SIMULATIONS = max(0, int(os.getenv("MONTE_CARLO_SIMULATIONS", "2000")))
COMMON_LONG_MIN_FUNDAMENTAL = env_float("COMMON_LONG_MIN_FUNDAMENTAL", 62.0)
COMMON_SHORT_MAX_FUNDAMENTAL = env_float("COMMON_SHORT_MAX_FUNDAMENTAL", 42.0)
COMMON_LONG_LABEL_BLOCKLIST = env_list("COMMON_LONG_LABEL_BLOCKLIST", ["value_trap", "overvalued", "overvalued_weak"])
COMMON_SHORT_LABEL_BLOCKLIST = env_list("COMMON_SHORT_LABEL_BLOCKLIST", ["deep_value", "quality_value", "compounder"])
COMMON_MIN_MARKET_CAP_M = float(os.getenv("COMMON_MIN_MARKET_CAP_USD_M", "1000.0"))
COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE = env_bool("COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE", True)
COMMON_FILTER_NEGATIVE_EARNINGS_LONG = env_bool("COMMON_FILTER_NEGATIVE_EARNINGS_LONG", False)
COMMON_FILTER_NEGATIVE_EARNINGS_SHORT = env_bool("COMMON_FILTER_NEGATIVE_EARNINGS_SHORT", False)
for _name, _value in {
    "COMMON_LONG_MIN_FUNDAMENTAL": COMMON_LONG_MIN_FUNDAMENTAL,
    "COMMON_SHORT_MAX_FUNDAMENTAL": COMMON_SHORT_MAX_FUNDAMENTAL,
}.items():
    if _value < 0.0 or _value > 100.0:
        raise ValueError(f"{_name} must be between 0 and 100")
if COMMON_MIN_MARKET_CAP_M < 0.0:
    raise ValueError("COMMON_MIN_MARKET_CAP_USD_M must be >= 0")
SECTOR_DIVERSIFICATION_ENABLED = env_bool("SECTOR_DIVERSIFICATION_ENABLED", False)

SHOCK_OVERLAY_ALLOWED_MODES = {"off", "score_only", "risk_only", "score_and_risk", "full"}
SHOCK_OVERLAY_MODE = env_str("SHOCK_OVERLAY_MODE", "off").lower()
if SHOCK_OVERLAY_MODE not in SHOCK_OVERLAY_ALLOWED_MODES:
    raise ValueError(
        "SHOCK_OVERLAY_MODE must be one of: off, score_only, risk_only, score_and_risk, full"
    )
SHOCK_OVERLAY_ACTIVE = SHOCK_OVERLAY_MODE != "off"
SHOCK_OVERLAY_POLICY_FILE = env_str(
    "SHOCK_OVERLAY_POLICY_FILE",
    "backtest_policy_configs/shock_overlay_policy.xlsx",
)
SHOCK_OVERLAY_SECTOR_BIAS_SHEET = env_str("SHOCK_OVERLAY_SECTOR_BIAS_SHEET", "sector_bias")
SHOCK_OVERLAY_SPECIAL_RULES_SHEET = env_str("SHOCK_OVERLAY_SPECIAL_RULES_SHEET", "special_rules")
SHOCK_OVERLAY_MIN_SHOCK_SCORE = env_float("SHOCK_OVERLAY_MIN_SHOCK_SCORE", 55.0)
SHOCK_OVERLAY_FULL_SHOCK_SCORE = env_float("SHOCK_OVERLAY_FULL_SHOCK_SCORE", 80.0)
SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA = env_float("SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA", 0.75)
SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT = env_float("SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT", 30.0)
SHOCK_OVERLAY_MAX_RISK_CUT_PCT = env_float("SHOCK_OVERLAY_MAX_RISK_CUT_PCT", 30.0)
SHOCK_OVERLAY_ALLOW_NEW_INTENTS = env_bool("SHOCK_OVERLAY_ALLOW_NEW_INTENTS", False)
SHOCK_OVERLAY_BLOCK_LONG_LABELS = tuple(
    label.lower() for label in env_list("SHOCK_OVERLAY_BLOCK_LONG_LABELS", ["insufficient_data", "value_trap"])
)
SHOCK_OVERLAY_BLOCK_SHORT_LABELS = tuple(
    label.lower() for label in env_list("SHOCK_OVERLAY_BLOCK_SHORT_LABELS", [])
)
SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_HIGH_LEVERAGE_CREDIT = env_bool(
    "SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_HIGH_LEVERAGE_CREDIT",
    True,
)
SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_NEGATIVE_EARNINGS = env_bool(
    "SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_NEGATIVE_EARNINGS",
    True,
)
SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_ENABLED = env_bool(
    "SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_ENABLED",
    False,
)
SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS = env_int(
    "SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS",
    1,
)
SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS = env_float(
    "SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS",
    0.60,
)
SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE = env_float(
    "SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE",
    8.0,
)
SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER = env_float(
    "SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER",
    0.15,
)
SHOCK_STRESS_GUARD_ENABLED = env_bool("SHOCK_STRESS_GUARD_ENABLED", False)
SHOCK_STRESS_GUARD_ELEVATED_SCORE = env_float("SHOCK_STRESS_GUARD_ELEVATED_SCORE", 60.0)
SHOCK_STRESS_GUARD_HIGH_SCORE = env_float("SHOCK_STRESS_GUARD_HIGH_SCORE", 70.0)
SHOCK_STRESS_GUARD_EXTREME_SCORE = env_float("SHOCK_STRESS_GUARD_EXTREME_SCORE", 80.0)
SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED = env_float("SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED", 0.85)
SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH = env_float("SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH", 0.65)
SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME = env_float("SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME", 0.35)
SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED = env_int("SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED", _default_position_cap(0.60))
SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH = env_int("SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH", _default_position_cap(0.40))
SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME = env_int("SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME", _default_position_cap(0.20))
SHOCK_STRESS_SECTOR_CAP_ENABLED = env_bool("SHOCK_STRESS_SECTOR_CAP_ENABLED", False)
SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED = env_int("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED", 2)
SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH = env_int("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH", 1)
SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME = env_int("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME", 1)
SHOCK_STRESS_BLOCK_NEGATIVE_BIAS_LONGS = env_bool("SHOCK_STRESS_BLOCK_NEGATIVE_BIAS_LONGS", True)
SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE = env_float("SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE", 70.0)
SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT = env_float("SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT", 1.0)
SHOCK_STRESS_PORTFOLIO_GUARD_ENABLED = env_bool("SHOCK_STRESS_PORTFOLIO_GUARD_ENABLED", False)
SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT = env_float("SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT", 2.0)
SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT = env_float("SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT", 2.0)

REGIME_RISK_MANAGEMENT_ENABLED = env_bool("REGIME_RISK_MANAGEMENT_ENABLED", False)
REGIME_RISK_NEUTRAL_IS_ELEVATED = env_bool("REGIME_RISK_NEUTRAL_IS_ELEVATED", True)
REGIME_RISK_ELEVATED_CONFIRM_DAYS = max(1, env_int("REGIME_RISK_ELEVATED_CONFIRM_DAYS", 2))
REGIME_RISK_HIGH_CONFIRM_DAYS = max(1, env_int("REGIME_RISK_HIGH_CONFIRM_DAYS", 2))
REGIME_RISK_EXTREME_CONFIRM_DAYS = max(1, env_int("REGIME_RISK_EXTREME_CONFIRM_DAYS", 1))
REGIME_RISK_RECOVERY_CONFIRM_DAYS = max(1, env_int("REGIME_RISK_RECOVERY_CONFIRM_DAYS", 3))
REGIME_RISK_ELEVATED_EXIT_SCORE = env_float("REGIME_RISK_ELEVATED_EXIT_SCORE", 55.0)
REGIME_RISK_HIGH_EXIT_SCORE = env_float("REGIME_RISK_HIGH_EXIT_SCORE", 65.0)
REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY = env_float("REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY", 0.30)
REGIME_RISK_POSITION_COOLDOWN_DAYS = max(0, env_int("REGIME_RISK_POSITION_COOLDOWN_DAYS", 2))
REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT = env_float("REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT", 3.0)
REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT = env_float("REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT", 2.0)
REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT = env_float("REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT", 1.0)
REGIME_RISK_HIGH_MAX_LONG_POSITIONS = env_int("REGIME_RISK_HIGH_MAX_LONG_POSITIONS", _default_position_cap(0.40))
REGIME_RISK_EXTREME_MAX_LONG_POSITIONS = env_int("REGIME_RISK_EXTREME_MAX_LONG_POSITIONS", _default_position_cap(0.20))
REGIME_RISK_HIGH_CLOSE_EXCESS_LONGS = env_bool("REGIME_RISK_HIGH_CLOSE_EXCESS_LONGS", True)
REGIME_RISK_RISK_OFF_CLOSE_LONGS = env_bool("REGIME_RISK_RISK_OFF_CLOSE_LONGS", True)
REGIME_RISK_HIGH_CLOSE_NEGATIVE_BIAS_LONGS = env_bool("REGIME_RISK_HIGH_CLOSE_NEGATIVE_BIAS_LONGS", True)
REGIME_RISK_EXTREME_CLOSE_NEGATIVE_BIAS_LONGS = env_bool("REGIME_RISK_EXTREME_CLOSE_NEGATIVE_BIAS_LONGS", True)
REGIME_RISK_HIGH_CLOSE_VALUATION_LABELS = {
    label.lower() for label in env_list("REGIME_RISK_HIGH_CLOSE_VALUATION_LABELS", ["deep_value", "undervalued", "value_trap"])
}
REGIME_RISK_EXTREME_CLOSE_VALUATION_LABELS = {
    label.lower() for label in env_list("REGIME_RISK_EXTREME_CLOSE_VALUATION_LABELS", ["deep_value", "undervalued", "value_trap"])
}
WORLD_REGIME_SHOCK_FIELDS_ACTIVE = SHOCK_OVERLAY_ACTIVE or SHOCK_STRESS_GUARD_ENABLED or REGIME_RISK_MANAGEMENT_ENABLED
if not (0.0 <= SHOCK_OVERLAY_MIN_SHOCK_SCORE <= 100.0):
    raise ValueError("SHOCK_OVERLAY_MIN_SHOCK_SCORE must be between 0 and 100")
if not (0.0 <= SHOCK_OVERLAY_FULL_SHOCK_SCORE <= 100.0):
    raise ValueError("SHOCK_OVERLAY_FULL_SHOCK_SCORE must be between 0 and 100")
if SHOCK_OVERLAY_FULL_SHOCK_SCORE <= SHOCK_OVERLAY_MIN_SHOCK_SCORE:
    raise ValueError("SHOCK_OVERLAY_FULL_SHOCK_SCORE must be greater than SHOCK_OVERLAY_MIN_SHOCK_SCORE")
if SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA < 0.0:
    raise ValueError("SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA must be >= 0")
if SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT < 0.0:
    raise ValueError("SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT must be >= 0")
if SHOCK_OVERLAY_MAX_RISK_CUT_PCT < 0.0:
    raise ValueError("SHOCK_OVERLAY_MAX_RISK_CUT_PCT must be >= 0")
if SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS < 0:
    raise ValueError("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS must be >= 0")
if not (-1.0 <= SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS <= 1.0):
    raise ValueError("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS must be between -1 and 1")
if SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE < 0.0:
    raise ValueError("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE must be >= 0")
if SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER < 0.0:
    raise ValueError("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER must be >= 0")
if not (0.0 <= SHOCK_STRESS_GUARD_ELEVATED_SCORE < SHOCK_STRESS_GUARD_HIGH_SCORE < SHOCK_STRESS_GUARD_EXTREME_SCORE <= 100.0):
    raise ValueError(
        "SHOCK_STRESS_GUARD scores must satisfy 0 <= ELEVATED < HIGH < EXTREME <= 100"
    )
for _name, _value in {
    "SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED": SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED,
    "SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH": SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH,
    "SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME": SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME,
    "SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT": SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT,
}.items():
    if _value < 0.0:
        raise ValueError(f"{_name} must be >= 0")
for _name, _value in {
    "SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED": SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED,
    "SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH": SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH,
    "SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME": SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME,
    "SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED": SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED,
    "SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH": SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH,
    "SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME": SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME,
}.items():
    if _value < 0:
        raise ValueError(f"{_name} must be >= 0")
for _name, _value in {
    "SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE": SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE,
    "SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT": SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT,
    "SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT": SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT,
}.items():
    if not (0.0 <= _value <= 100.0):
        raise ValueError(f"{_name} must be between 0 and 100")
if not (0.0 <= REGIME_RISK_ELEVATED_EXIT_SCORE < REGIME_RISK_HIGH_EXIT_SCORE <= 100.0):
    raise ValueError(
        "REGIME_RISK exit scores must satisfy 0 <= ELEVATED_EXIT_SCORE < HIGH_EXIT_SCORE <= 100"
    )
if not (0.0 < REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY <= 1.0):
    raise ValueError("REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY must be > 0 and <= 1")
for _name, _value in {
    "REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT": REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT,
    "REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT": REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT,
    "REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT": REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT,
}.items():
    if not (0.0 < _value <= 100.0):
        raise ValueError(f"{_name} must be > 0 and <= 100")
for _name, _value in {
    "REGIME_RISK_HIGH_MAX_LONG_POSITIONS": REGIME_RISK_HIGH_MAX_LONG_POSITIONS,
    "REGIME_RISK_EXTREME_MAX_LONG_POSITIONS": REGIME_RISK_EXTREME_MAX_LONG_POSITIONS,
}.items():
    if _value < 0:
        raise ValueError(f"{_name} must be >= 0")

def _execution_pct(env_key: str, default_pct: float) -> float:
    value = env_float(env_key, default_pct)
    if value < 0.0:
        raise ValueError(f"{env_key} must be >= 0")
    if value > 99.9999:
        raise ValueError(f"{env_key} must be <= 99.9999 because *_PCT values are percent points")
    if 0.0 < value < 0.5:
        raise ValueError(
            f"{env_key} uses percent points now: use 3.0 for 3%, not 0.03. "
            "Values below 0.5 look like old decimal-ratio input."
        )
    return value


TAKE_PROFIT_MODE = os.getenv("TAKE_PROFIT_MODE", "fixed").strip().lower()
EXECUTION_LONG_TAKE_PROFIT_PCT = _execution_pct("EXECUTION_LONG_TAKE_PROFIT_PCT", 5.5)
EXECUTION_SHORT_TAKE_PROFIT_PCT = _execution_pct("EXECUTION_SHORT_TAKE_PROFIT_PCT", 6.0)
EXECUTION_LONG_TRAILING_ACTIVATION_PCT = _execution_pct("EXECUTION_LONG_TRAILING_ACTIVATION_PCT", 4.0)
EXECUTION_SHORT_TRAILING_ACTIVATION_PCT = _execution_pct("EXECUTION_SHORT_TRAILING_ACTIVATION_PCT", 4.0)
EXECUTION_LONG_TRAILING_DISTANCE_PCT = _execution_pct("EXECUTION_LONG_TRAILING_DISTANCE_PCT", 3.0)
EXECUTION_SHORT_TRAILING_DISTANCE_PCT = _execution_pct("EXECUTION_SHORT_TRAILING_DISTANCE_PCT", 3.0)
EXECUTION_LONG_TAKE_PROFIT_RATIO = EXECUTION_LONG_TAKE_PROFIT_PCT / 100.0
EXECUTION_SHORT_TAKE_PROFIT_RATIO = EXECUTION_SHORT_TAKE_PROFIT_PCT / 100.0
EXECUTION_LONG_TRAILING_ACTIVATION_RATIO = EXECUTION_LONG_TRAILING_ACTIVATION_PCT / 100.0
EXECUTION_SHORT_TRAILING_ACTIVATION_RATIO = EXECUTION_SHORT_TRAILING_ACTIVATION_PCT / 100.0
EXECUTION_LONG_TRAILING_DISTANCE_RATIO = EXECUTION_LONG_TRAILING_DISTANCE_PCT / 100.0
EXECUTION_SHORT_TRAILING_DISTANCE_RATIO = EXECUTION_SHORT_TRAILING_DISTANCE_PCT / 100.0
EXECUTION_LONG_MAX_HOLD_DAYS = env_float("EXECUTION_LONG_MAX_HOLD_DAYS", 12.0)
EXECUTION_SHORT_MAX_HOLD_DAYS = env_float("EXECUTION_SHORT_MAX_HOLD_DAYS", 5.0)
if TAKE_PROFIT_MODE not in {"fixed", "trailing"}:
    raise ValueError("TAKE_PROFIT_MODE must be one of: fixed, trailing")
for _name, _value in {
    "EXECUTION_LONG_MAX_HOLD_DAYS": EXECUTION_LONG_MAX_HOLD_DAYS,
    "EXECUTION_SHORT_MAX_HOLD_DAYS": EXECUTION_SHORT_MAX_HOLD_DAYS,
}.items():
    if _value < 0.0:
        raise ValueError(f"{_name} must be >= 0")
if TAKE_PROFIT_MODE == "fixed" and EXECUTION_LONG_TAKE_PROFIT_PCT <= 0.0:
    raise ValueError("EXECUTION_LONG_TAKE_PROFIT_PCT must be > 0 in fixed mode")
if TAKE_PROFIT_MODE == "fixed" and EXECUTION_SHORT_TAKE_PROFIT_PCT <= 0.0:
    raise ValueError("EXECUTION_SHORT_TAKE_PROFIT_PCT must be > 0 in fixed mode")
if TAKE_PROFIT_MODE == "trailing":
    for _name, _value in {
        "EXECUTION_LONG_TRAILING_ACTIVATION_PCT": EXECUTION_LONG_TRAILING_ACTIVATION_PCT,
        "EXECUTION_SHORT_TRAILING_ACTIVATION_PCT": EXECUTION_SHORT_TRAILING_ACTIVATION_PCT,
        "EXECUTION_LONG_TRAILING_DISTANCE_PCT": EXECUTION_LONG_TRAILING_DISTANCE_PCT,
        "EXECUTION_SHORT_TRAILING_DISTANCE_PCT": EXECUTION_SHORT_TRAILING_DISTANCE_PCT,
    }.items():
        if _value <= 0.0:
            raise ValueError(f"{_name} must be > 0 in trailing mode")

COMMON_STOP_LOSS_ENABLED = env_bool("COMMON_STOP_LOSS_ENABLED", True)
COMMON_STOP_LOOKBACK_BARS = max(1, env_int("COMMON_STOP_LOOKBACK_BARS", 14))
COMMON_STOP_BUFFER = env_float("COMMON_STOP_BUFFER_RATIO", 0.007)
COMMON_STOP_ATR_LOOKBACK_BARS = max(1, env_int("COMMON_STOP_ATR_LOOKBACK_BARS", 14))
COMMON_STOP_ATR_MULT = env_float("COMMON_STOP_ATR_MULT", 1.5)
COMMON_MIN_STOP_PCT = env_float("COMMON_MIN_STOP_PCT", 2.5)
COMMON_MAX_STOP_PCT = env_float("COMMON_MAX_STOP_PCT", 11.0)
for _name, _value in {
    "COMMON_STOP_BUFFER_RATIO": COMMON_STOP_BUFFER,
    "COMMON_STOP_ATR_MULT": COMMON_STOP_ATR_MULT,
    "COMMON_MIN_STOP_PCT": COMMON_MIN_STOP_PCT,
    "COMMON_MAX_STOP_PCT": COMMON_MAX_STOP_PCT,
}.items():
    if _value < 0.0:
        raise ValueError(f"{_name} must be >= 0")
if COMMON_MAX_STOP_PCT > 0.0 and COMMON_MIN_STOP_PCT > COMMON_MAX_STOP_PCT:
    raise ValueError("COMMON_MIN_STOP_PCT must be <= COMMON_MAX_STOP_PCT")

GRID_SEARCH_ENABLED = os.getenv("GRID_SEARCH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
MODEL_SELECTION = os.getenv("MODEL_SELECTION", "single").strip().lower()
MODEL_FILE = os.getenv("MODEL_FILE", "pullback_bounce_fundamental_v1.py").strip()
runtime.CURRENT_MODEL_FILE = MODEL_FILE
MODEL_FILES = env_list("MODEL_FILES", [])
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "backtest_models")).strip()
MODEL_CONFIG_DIR = os.getenv("MODEL_CONFIG_DIR", str(PROJECT_ROOT / "backtest_model_configs")).strip()
MODEL_CONFIG_REQUIRED = env_bool("MODEL_CONFIG_REQUIRED", True)
MODEL_PARALLELISM = max(1, env_int("MODEL_PARALLELISM", 2))
MODEL_FAILURE_MODE = os.getenv("MODEL_FAILURE_MODE", "fail_fast").strip().lower()
BACKTEST_PARALLEL_CHILD = env_bool("BACKTEST_PARALLEL_CHILD", False)
BACKTEST_SHARED_TIMELINE_PREBUILDER = env_bool("BACKTEST_SHARED_TIMELINE_PREBUILDER", False)
RESULT_SCHEMA = os.getenv("RESULT_SCHEMA", "public").strip() or "public"
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", RESULT_SCHEMA):
    raise ValueError(f"Invalid RESULT_SCHEMA: {RESULT_SCHEMA!r}")


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _result_table(name: str) -> str:
    return f"{_qident(RESULT_SCHEMA)}.{_qident(name)}"


def _format_run_note_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, datetime):
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, dict):
        return ", ".join(
            f"{key}:{_format_run_note_value(val)}"
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        )
    if isinstance(value, (list, tuple, set)):
        values = sorted(value, key=str) if isinstance(value, set) else value
        return ", ".join(_format_run_note_value(item) for item in values)
    return str(value).replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")


def _model_config_note_pairs(cfg) -> list[tuple[str, object]]:
    if cfg is None:
        return []
    if dataclasses.is_dataclass(cfg):
        items = dataclasses.asdict(cfg).items()
    elif hasattr(cfg, "__dict__"):
        items = vars(cfg).items()
    else:
        return [("MODEL_CONFIG", cfg)]
    return [
        (f"MODEL_PARAM_{name.upper()}", value)
        for name, value in items
        if not name.startswith("_")
    ]


def _build_run_notes(notes: Optional[str], cfg=None) -> str:
    created_local = datetime.now(ZoneInfo(RUN_LABEL_TZ))
    suffix = (notes if notes is not None else RUN_NOTES_EXTRA).strip()
    pairs: list[tuple[str, object]] = [
        ("RUN_CREATED_LOCAL", f"{created_local:%Y-%m-%d %H:%M}"),
        ("RUN_LABEL_TZ", RUN_LABEL_TZ),
        ("START_DATE", START_DATE),
        ("END_DATE", END_DATE),
        ("MODEL_FILE", runtime.CURRENT_MODEL_FILE),
        ("ACCOUNT_PROFILE", ACCOUNT_PROFILE),
        ("MODEL_SELECTION", MODEL_SELECTION),
        ("GRID_SEARCH_ENABLED", GRID_SEARCH_ENABLED),
        ("INITIAL_EQUITY_USD", INITIAL_EQUITY),
        ("RISK_PER_TRADE_EQUITY_PCT", RISK_PER_TRADE_PCT),
        ("MAX_OPEN_POSITIONS", MAX_OPEN_POSITIONS),
        ("MAX_POSITION_OPENS_PER_DAY", MAX_POSITION_OPENS_PER_DAY),
        ("MAX_POSITION_OPENS_PER_HOUR", MAX_POSITION_OPENS_PER_HOUR),
    ]
    if suffix:
        pairs.append(("RUN_NOTE", suffix))

    pairs.extend(_model_config_note_pairs(cfg))

    pairs.extend([
        ("SOURCE_MARKET_DATA_1H_TABLE", SOURCE_MARKET_DATA_1H_TABLE),
        ("SOURCE_FUNDAMENTAL_SCORES_TABLE", SOURCE_FUNDAMENTAL_SCORES_TABLE),
        ("SOURCE_WORLD_REGIME_TABLE", SOURCE_WORLD_REGIME_TABLE),
        ("REQUIRE_USD_FUNDAMENTALS", REQUIRE_USD_FUNDAMENTALS),
        ("ALLOW_FRACTIONAL_SHARES", ALLOW_FRACTIONAL_SHARES),
        ("SPREAD_BPS", SPREAD_BPS),
        ("SLIPPAGE_BPS", SLIPPAGE_BPS),
        ("COMMISSION_PER_ORDER_USD", COMMISSION_PER_ORDER_USD),
        ("COMMISSION_PER_SHARE_USD", COMMISSION_PER_SHARE_USD),
        ("COMMISSION_MIN_PER_ORDER_USD", COMMISSION_MIN_PER_ORDER_USD),
        ("COMMISSION_MAX_PCT", COMMISSION_MAX_PCT),
        ("COMMISSION_BPS", COMMISSION_BPS),
    ])

    if ACCOUNT_PROFILE == "ps_acc":
        pairs.extend([
            ("PS_MARGIN_REQUIREMENT_PCT", MARGIN_REQUIREMENT_PCT),
            ("PS_MARGIN_STOP_OUT_LEVEL_PCT", PS_MARGIN_STOP_OUT_LEVEL_PCT),
            ("PS_MIN_ENTRY_MARGIN_LEVEL_PCT", PS_MIN_ENTRY_MARGIN_LEVEL_PCT),
            ("PS_SHARE_CFD_ARR_PCT", PS_SHARE_CFD_ARR_PCT),
            ("PS_SHARE_CFD_ADMIN_FEE_PCT", PS_SHARE_CFD_ADMIN_FEE_PCT),
            ("PS_SHARE_CFD_SHORT_BORROW_RATE_PCT", PS_SHARE_CFD_SHORT_BORROW_RATE_PCT),
            ("PS_SHARE_CFD_OVERNIGHT_DAY_COUNT", PS_SHARE_CFD_OVERNIGHT_DAY_COUNT),
            ("PS_TRADABLE_SYMBOLS_TABLE", PS_TRADABLE_SYMBOLS_TABLE),
            ("PS_24_ENTRY_SL_TP_ACTIVE", PS_24_ENTRY_SL_TP_ACTIVE),
        ])
    else:
        pairs.extend([
            ("IBKR_LONG_INITIAL_MARGIN_PCT", IBKR_LONG_INITIAL_MARGIN_PCT),
            ("IBKR_LONG_MAINTENANCE_MARGIN_PCT", IBKR_LONG_MAINTENANCE_MARGIN_PCT),
            ("IBKR_SHORT_INITIAL_MARGIN_PCT", IBKR_SHORT_INITIAL_MARGIN_PCT),
            ("IBKR_SHORT_MAINTENANCE_MARGIN_PCT", IBKR_SHORT_MAINTENANCE_MARGIN_PCT),
            ("IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE", IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE),
            ("MARGIN_FINANCING_RATE_PCT", MARGIN_FINANCING_RATE_PCT),
        ])

    pairs.extend([
        ("ENTRY_WINDOW_ENABLED", ENTRY_WINDOW_ENABLED),
        ("ENTRY_WINDOW_TZ", ENTRY_WINDOW_TZ),
        ("ENTRY_WINDOW_START", ENTRY_WINDOW_START),
        ("ENTRY_WINDOW_END", ENTRY_WINDOW_END),
        ("SL_TP_WINDOW_TZ", SL_TP_WINDOW_TZ),
        ("SL_TP_WINDOW_START", SL_TP_WINDOW_START),
        ("SL_TP_WINDOW_END", SL_TP_WINDOW_END),
        ("COMMON_LONG_MIN_FUNDAMENTAL", COMMON_LONG_MIN_FUNDAMENTAL),
        ("COMMON_SHORT_MAX_FUNDAMENTAL", COMMON_SHORT_MAX_FUNDAMENTAL),
        ("COMMON_LONG_LABEL_BLOCKLIST", COMMON_LONG_LABEL_BLOCKLIST),
        ("COMMON_SHORT_LABEL_BLOCKLIST", COMMON_SHORT_LABEL_BLOCKLIST),
        ("COMMON_MIN_MARKET_CAP_USD_M", COMMON_MIN_MARKET_CAP_M),
        ("COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE", COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE),
        ("COMMON_FILTER_NEGATIVE_EARNINGS_LONG", COMMON_FILTER_NEGATIVE_EARNINGS_LONG),
        ("COMMON_FILTER_NEGATIVE_EARNINGS_SHORT", COMMON_FILTER_NEGATIVE_EARNINGS_SHORT),
        ("SECTOR_DIVERSIFICATION_ENABLED", SECTOR_DIVERSIFICATION_ENABLED),
        ("TAKE_PROFIT_MODE", TAKE_PROFIT_MODE),
        ("EXECUTION_LONG_TAKE_PROFIT_PCT", EXECUTION_LONG_TAKE_PROFIT_PCT),
        ("EXECUTION_SHORT_TAKE_PROFIT_PCT", EXECUTION_SHORT_TAKE_PROFIT_PCT),
        ("EXECUTION_LONG_TRAILING_ACTIVATION_PCT", EXECUTION_LONG_TRAILING_ACTIVATION_PCT),
        ("EXECUTION_SHORT_TRAILING_ACTIVATION_PCT", EXECUTION_SHORT_TRAILING_ACTIVATION_PCT),
        ("EXECUTION_LONG_TRAILING_DISTANCE_PCT", EXECUTION_LONG_TRAILING_DISTANCE_PCT),
        ("EXECUTION_SHORT_TRAILING_DISTANCE_PCT", EXECUTION_SHORT_TRAILING_DISTANCE_PCT),
        ("EXECUTION_LONG_MAX_HOLD_DAYS", EXECUTION_LONG_MAX_HOLD_DAYS),
        ("EXECUTION_SHORT_MAX_HOLD_DAYS", EXECUTION_SHORT_MAX_HOLD_DAYS),
        ("COMMON_STOP_LOSS_ENABLED", COMMON_STOP_LOSS_ENABLED),
        ("COMMON_STOP_LOOKBACK_BARS", COMMON_STOP_LOOKBACK_BARS),
        ("COMMON_STOP_BUFFER_RATIO", COMMON_STOP_BUFFER),
        ("COMMON_STOP_ATR_LOOKBACK_BARS", COMMON_STOP_ATR_LOOKBACK_BARS),
        ("COMMON_STOP_ATR_MULT", COMMON_STOP_ATR_MULT),
        ("COMMON_MIN_STOP_PCT", COMMON_MIN_STOP_PCT),
        ("COMMON_MAX_STOP_PCT", COMMON_MAX_STOP_PCT),
        ("SHOCK_OVERLAY_MODE", SHOCK_OVERLAY_MODE),
        ("SHOCK_OVERLAY_POLICY_FILE", SHOCK_OVERLAY_POLICY_FILE),
        ("SHOCK_OVERLAY_SECTOR_BIAS_SHEET", SHOCK_OVERLAY_SECTOR_BIAS_SHEET),
        ("SHOCK_OVERLAY_SPECIAL_RULES_SHEET", SHOCK_OVERLAY_SPECIAL_RULES_SHEET),
        ("SHOCK_OVERLAY_MIN_SHOCK_SCORE", SHOCK_OVERLAY_MIN_SHOCK_SCORE),
        ("SHOCK_OVERLAY_FULL_SHOCK_SCORE", SHOCK_OVERLAY_FULL_SHOCK_SCORE),
        ("SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA", SHOCK_OVERLAY_MAX_INTENT_SCORE_DELTA),
        ("SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT", SHOCK_OVERLAY_MAX_RISK_UPLIFT_PCT),
        ("SHOCK_OVERLAY_MAX_RISK_CUT_PCT", SHOCK_OVERLAY_MAX_RISK_CUT_PCT),
        ("SHOCK_OVERLAY_ALLOW_NEW_INTENTS", SHOCK_OVERLAY_ALLOW_NEW_INTENTS),
        ("SHOCK_OVERLAY_BLOCK_LONG_LABELS", SHOCK_OVERLAY_BLOCK_LONG_LABELS),
        ("SHOCK_OVERLAY_BLOCK_SHORT_LABELS", SHOCK_OVERLAY_BLOCK_SHORT_LABELS),
        (
            "SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_HIGH_LEVERAGE_CREDIT",
            SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_HIGH_LEVERAGE_CREDIT,
        ),
        ("SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_NEGATIVE_EARNINGS", SHOCK_OVERLAY_DISABLE_LONG_BOOST_ON_NEGATIVE_EARNINGS),
        ("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_ENABLED", SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_ENABLED),
        ("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS", SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MAX_POSITIONS),
        ("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS", SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_BIAS),
        ("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE", SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_MIN_INTENT_SCORE),
        ("SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER", SHOCK_OVERLAY_RISK_OFF_LONG_SLEEVE_RISK_MULTIPLIER),
        ("SHOCK_STRESS_GUARD_ENABLED", SHOCK_STRESS_GUARD_ENABLED),
        ("SHOCK_STRESS_GUARD_ELEVATED_SCORE", SHOCK_STRESS_GUARD_ELEVATED_SCORE),
        ("SHOCK_STRESS_GUARD_HIGH_SCORE", SHOCK_STRESS_GUARD_HIGH_SCORE),
        ("SHOCK_STRESS_GUARD_EXTREME_SCORE", SHOCK_STRESS_GUARD_EXTREME_SCORE),
        ("SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED", SHOCK_STRESS_LONG_RISK_MULTIPLIER_ELEVATED),
        ("SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH", SHOCK_STRESS_LONG_RISK_MULTIPLIER_HIGH),
        ("SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME", SHOCK_STRESS_LONG_RISK_MULTIPLIER_EXTREME),
        ("SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED", SHOCK_STRESS_MAX_LONG_POSITIONS_ELEVATED),
        ("SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH", SHOCK_STRESS_MAX_LONG_POSITIONS_HIGH),
        ("SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME", SHOCK_STRESS_MAX_LONG_POSITIONS_EXTREME),
        ("SHOCK_STRESS_SECTOR_CAP_ENABLED", SHOCK_STRESS_SECTOR_CAP_ENABLED),
        ("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED", SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_ELEVATED),
        ("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH", SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_HIGH),
        ("SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME", SHOCK_STRESS_MAX_POSITIONS_PER_SECTOR_EXTREME),
        ("SHOCK_STRESS_BLOCK_NEGATIVE_BIAS_LONGS", SHOCK_STRESS_BLOCK_NEGATIVE_BIAS_LONGS),
        ("SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE", SHOCK_STRESS_CREDIT_STRESS_MIN_SCORE),
        ("SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT", SHOCK_STRESS_SHORT_MAX_RISK_MULTIPLIER_CREDIT),
        ("SHOCK_STRESS_PORTFOLIO_GUARD_ENABLED", SHOCK_STRESS_PORTFOLIO_GUARD_ENABLED),
        ("SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT", SHOCK_STRESS_PORTFOLIO_DAILY_LOSS_LIMIT_PCT),
        ("SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT", SHOCK_STRESS_PORTFOLIO_OPEN_LOSS_LIMIT_PCT),
        ("REGIME_RISK_MANAGEMENT_ENABLED", REGIME_RISK_MANAGEMENT_ENABLED),
        ("REGIME_RISK_NEUTRAL_IS_ELEVATED", REGIME_RISK_NEUTRAL_IS_ELEVATED),
        ("REGIME_RISK_ELEVATED_CONFIRM_DAYS", REGIME_RISK_ELEVATED_CONFIRM_DAYS),
        ("REGIME_RISK_HIGH_CONFIRM_DAYS", REGIME_RISK_HIGH_CONFIRM_DAYS),
        ("REGIME_RISK_EXTREME_CONFIRM_DAYS", REGIME_RISK_EXTREME_CONFIRM_DAYS),
        ("REGIME_RISK_RECOVERY_CONFIRM_DAYS", REGIME_RISK_RECOVERY_CONFIRM_DAYS),
        ("REGIME_RISK_ELEVATED_EXIT_SCORE", REGIME_RISK_ELEVATED_EXIT_SCORE),
        ("REGIME_RISK_HIGH_EXIT_SCORE", REGIME_RISK_HIGH_EXIT_SCORE),
        ("REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY", REGIME_RISK_MAX_CLOSE_FRACTION_PER_DAY),
        ("REGIME_RISK_POSITION_COOLDOWN_DAYS", REGIME_RISK_POSITION_COOLDOWN_DAYS),
        ("REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT", REGIME_RISK_ELEVATED_LONG_MAX_STOP_DISTANCE_PCT),
        ("REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT", REGIME_RISK_HIGH_LONG_MAX_STOP_DISTANCE_PCT),
        ("REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT", REGIME_RISK_EXTREME_LONG_MAX_STOP_DISTANCE_PCT),
        ("REGIME_RISK_HIGH_MAX_LONG_POSITIONS", REGIME_RISK_HIGH_MAX_LONG_POSITIONS),
        ("REGIME_RISK_EXTREME_MAX_LONG_POSITIONS", REGIME_RISK_EXTREME_MAX_LONG_POSITIONS),
        ("REGIME_RISK_HIGH_CLOSE_EXCESS_LONGS", REGIME_RISK_HIGH_CLOSE_EXCESS_LONGS),
        ("REGIME_RISK_RISK_OFF_CLOSE_LONGS", REGIME_RISK_RISK_OFF_CLOSE_LONGS),
        ("REGIME_RISK_HIGH_CLOSE_NEGATIVE_BIAS_LONGS", REGIME_RISK_HIGH_CLOSE_NEGATIVE_BIAS_LONGS),
        ("REGIME_RISK_EXTREME_CLOSE_NEGATIVE_BIAS_LONGS", REGIME_RISK_EXTREME_CLOSE_NEGATIVE_BIAS_LONGS),
        ("REGIME_RISK_HIGH_CLOSE_VALUATION_LABELS", REGIME_RISK_HIGH_CLOSE_VALUATION_LABELS),
        ("REGIME_RISK_EXTREME_CLOSE_VALUATION_LABELS", REGIME_RISK_EXTREME_CLOSE_VALUATION_LABELS),
        ("MONTE_CARLO_ENABLED", MONTE_CARLO_ENABLED),
        ("MONTE_CARLO_SIMULATIONS", MONTE_CARLO_SIMULATIONS),
        ("DECISION_EVENT_MODE", DECISION_EVENT_MODE),
    ])

    for regime_label, exposure in REGIME_EXPOSURE_BY_LABEL.items():
        prefix = "REGIME_" + regime_label.replace("-", "_")
        pairs.extend([
            (f"{prefix}_LONG_RISK_MULTIPLIER", exposure["long_risk_multiplier"]),
            (f"{prefix}_SHORT_RISK_MULTIPLIER", exposure["short_risk_multiplier"]),
            (f"{prefix}_MAX_LONG_POSITIONS", exposure["max_long_positions"]),
            (f"{prefix}_MAX_SHORT_POSITIONS", exposure["max_short_positions"]),
        ])

    return "\n".join(f"{name}={_format_run_note_value(value)}" for name, value in pairs)


def _parse_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})


def _parse_hold_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
ENTRY_WINDOW_ENABLED = _account_setting_bool("ENTRY_WINDOW_ENABLED", True)
ENTRY_WINDOW_TZ = _account_setting("ENTRY_WINDOW_TZ", "America/New_York")
ENTRY_WINDOW_START, ENTRY_WINDOW_END = _account_window_setting("ENTRY_WINDOW")
SL_TP_WINDOW_TZ = _account_setting("SL_TP_WINDOW_TZ", "America/New_York")
SL_TP_WINDOW_START, SL_TP_WINDOW_END = _account_window_setting("SL_TP_WINDOW")

SOURCE_MARKET_DATA_1H_TABLE = os.getenv("SOURCE_MARKET_DATA_1H_TABLE", "alpaca_market_data_1h")
SOURCE_FUNDAMENTAL_SCORES_TABLE = os.getenv("SOURCE_FUNDAMENTAL_SCORES_TABLE", "stock_scorer_fundamental_scores")
SOURCE_WORLD_REGIME_TABLE = os.getenv("SOURCE_WORLD_REGIME_TABLE", "world_regime_daily_scores_mv")
PS_TRADABLE_SYMBOLS_TABLE = os.getenv("PS_TRADABLE_SYMBOLS_TABLE", "public.pepperstone_data")
PS_24_ENTRY_SL_TP_ACTIVE = env_bool("PS_24_ENTRY_SL_TP_ACTIVE", False) if ACCOUNT_PROFILE == "ps_acc" else False
IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE = os.getenv(
    "IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE",
    "public.ibkr_symbol_margin_requirements",
)
REQUIRE_USD_FUNDAMENTALS = os.getenv("REQUIRE_USD_FUNDAMENTALS", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

DB_CONNECT_RETRIES       = int(os.getenv("DB_CONNECT_RETRIES", "5"))
DB_CONNECT_RETRY_DELAY_SECONDS = float(os.getenv("DB_CONNECT_RETRY_DELAY_SECONDS", "5.0"))
DB_STATEMENT_TIMEOUT_MS = max(0, int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "60000")))
DB_LOCK_TIMEOUT_MS = max(0, int(os.getenv("DB_LOCK_TIMEOUT_MS", "5000")))
DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS = max(0, int(os.getenv("DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS", "60000")))

DB = {
    "host":            os.getenv("PGHOST", "timescaledb"),
    "port":            int(os.getenv("PGPORT", "5432")),
    "dbname":          os.getenv("PGDATABASE", "postgres"),
    "user":            os.getenv("PGUSER", "market-data-account"),
    "password":        os.getenv("PGPASSWORD", "market-data-account-pw"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10")),
    "application_name": os.getenv("PGAPPNAME", "backtest_runner"),
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}

__all__ = [name for name in globals() if not name.startswith("__")]
