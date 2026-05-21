"""Environment-backed configuration for one backtest process."""

import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from backtest_shared import env_bool, env_float, env_int, env_list
from . import runtime

PROJECT_ROOT = Path(__file__).resolve().parent.parent

START_DATE             = date.fromisoformat(os.getenv("START_DATE", "2023-01-01"))
END_DATE               = date.fromisoformat(os.getenv("END_DATE", str(date.today())))
if env_bool("ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS", False):
    raise ValueError(
        "ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS=true is disabled; backtests must use point-in-time data_available_at guards."
    )
ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS = False
ACCOUNT_PROFILE        = os.getenv("ACCOUNT_PROFILE", "ps_acc").strip().lower()
INITIAL_EQUITY         = float(os.getenv("INITIAL_EQUITY", "100000.0"))
RISK_PER_TRADE_PCT     = float(os.getenv("RISK_PER_TRADE_PCT", "2.0"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS", "5"))

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
if ACCOUNT_PROFILE not in ACCOUNT_PROFILE_DEFAULTS:
    raise ValueError("ACCOUNT_PROFILE must be one of: ps_acc, ibkr_acc")
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
BAR_CACHE_WARMUP_DAYS = max(1, int(os.getenv("BAR_CACHE_WARMUP_DAYS", "120")))
BAR_CACHE_BATCH_SIZE = max(1, int(os.getenv("BAR_CACHE_BATCH_SIZE", "100")))
SIGNAL_BAR_CACHE_ENABLED = env_bool("SIGNAL_BAR_CACHE_ENABLED", True)
SIGNAL_BAR_CACHE_MAX_MIB = max(128, env_int("SIGNAL_BAR_CACHE_MAX_MIB", 2048))
MONTE_CARLO_ENABLED       = os.getenv("MONTE_CARLO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
N_MONTE_CARLO_SIMULATIONS = max(0, int(os.getenv("N_MONTE_CARLO_SIMULATIONS", "2000")))
MIN_MARKET_CAP_M = float(os.getenv("MIN_MARKET_CAP_M", "1000.0"))
FILTER_FUNDAMENTAL_HIGH_LEVERAGE = env_bool("FILTER_FUNDAMENTAL_HIGH_LEVERAGE", True)
FILTER_NEGATIVE_EARNINGS_LONG = env_bool("FILTER_NEGATIVE_EARNINGS_LONG", False)
FILTER_NEGATIVE_EARNINGS_SHORT = env_bool("FILTER_NEGATIVE_EARNINGS_SHORT", False)
LONG_MAX_HOLD_DAYS = max(0.0, float(os.getenv("LONG_MAX_HOLD_DAYS", "5.0")))
SHORT_MAX_HOLD_DAYS = max(0.0, float(os.getenv("SHORT_MAX_HOLD_DAYS", "5.0")))
TP1_CLOSE_RATIO = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_RATIO", "0.5"))))
SECTOR_DIVERSIFICATION_ENABLED = env_bool("SECTOR_DIVERSIFICATION_ENABLED", False)

GRID_SEARCH_ENABLED = os.getenv("GRID_SEARCH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
MODEL_SELECTION = os.getenv("MODEL_SELECTION", "single").strip().lower()
MODEL_FILE = os.getenv("MODEL_FILE", "pullback_bounce_fundamental_v1.py").strip()
runtime.CURRENT_MODEL_FILE = MODEL_FILE
MODEL_FILES = env_list("MODEL_FILES", [])
MODEL_DIR = os.getenv("MODEL_DIR", str(PROJECT_ROOT / "backtest_models")).strip()
MODEL_PARALLELISM = max(1, env_int("MODEL_PARALLELISM", 2))
MODEL_FAILURE_MODE = os.getenv("MODEL_FAILURE_MODE", "fail_fast").strip().lower()
BACKTEST_PARALLEL_CHILD = env_bool("BACKTEST_PARALLEL_CHILD", False)
RESULT_SCHEMA = os.getenv("RESULT_SCHEMA", "public").strip() or "public"
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", RESULT_SCHEMA):
    raise ValueError(f"Invalid RESULT_SCHEMA: {RESULT_SCHEMA!r}")


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _result_table(name: str) -> str:
    return f"{_qident(RESULT_SCHEMA)}.{_qident(name)}"


def _build_run_notes(notes: Optional[str]) -> str:
    created_local = datetime.now(ZoneInfo(RUN_LABEL_TZ))
    prefix = (
        f"{created_local:%Y-%m-%d %H:%M} | "
        f"{runtime.CURRENT_MODEL_FILE} | "
        f"{ACCOUNT_PROFILE} | "
        f"{START_DATE}->{END_DATE}"
    )
    suffix = (notes if notes is not None else RUN_NOTES_EXTRA).strip()
    return f"{prefix} | {suffix}" if suffix else prefix


def _parse_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})


def _parse_hold_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
ENTRY_WINDOW_ENABLED = os.getenv("ENTRY_WINDOW_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
ENTRY_WINDOW_TZ = os.getenv("ENTRY_WINDOW_TZ", "America/New_York")
ENTRY_WINDOW_START = os.getenv("ENTRY_WINDOW_START", "06:30")
ENTRY_WINDOW_END = os.getenv("ENTRY_WINDOW_END", "19:00")
SL_TP_WINDOW_TZ = os.getenv("SL_TP_WINDOW_TZ", "America/New_York")
SL_TP_WINDOW_START = os.getenv("SL_TP_WINDOW_START", "09:30")
SL_TP_WINDOW_END = os.getenv("SL_TP_WINDOW_END", "16:00")
STOP_LOSS_RTH_ONLY = env_bool("STOP_LOSS_RTH_ONLY", False)
STOP_LOSS_RTH_TZ = os.getenv("STOP_LOSS_RTH_TZ", "America/New_York")
STOP_LOSS_RTH_START = os.getenv("STOP_LOSS_RTH_START", "09:30")
STOP_LOSS_RTH_END = os.getenv("STOP_LOSS_RTH_END", "16:00")

SOURCE_1H           = os.getenv("SOURCE_1H", "alpaca_market_data_1h")
SOURCE_FUNDAMENTAL  = os.getenv("SOURCE_FUNDAMENTAL", "stocks_analysis_fundamental_scores")
SOURCE_WORLD_REGIME = os.getenv("SOURCE_WORLD_REGIME", "world_regime_daily_scores_mv")
PEPPERSTONE_TABLE   = os.getenv("PEPPERSTONE_TABLE", "public.pepperstone_data")
IBKR_MARGIN_REQUIREMENTS_TABLE = os.getenv("IBKR_MARGIN_REQUIREMENTS_TABLE", "public.ibkr_symbol_margin_requirements")
REQUIRE_USD_FUNDAMENTALS = os.getenv("REQUIRE_USD_FUNDAMENTALS", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

DB_CONNECT_RETRIES       = int(os.getenv("DB_CONNECT_RETRIES", "5"))
DB_CONNECT_RETRY_DELAY_S = float(os.getenv("DB_CONNECT_RETRY_DELAY_S", "5.0"))
DB_STATEMENT_TIMEOUT_MS = max(0, int(os.getenv("DB_STATEMENT_TIMEOUT_MS", "60000")))
DB_LOCK_TIMEOUT_MS = max(0, int(os.getenv("DB_LOCK_TIMEOUT_MS", "5000")))
DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS = max(0, int(os.getenv("DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS", "60000")))

DB = {
    "host":            os.getenv("PGHOST", "timescaledb"),
    "port":            int(os.getenv("PGPORT", "5432")),
    "dbname":          os.getenv("PGDATABASE", "postgres"),
    "user":            os.getenv("PGUSER", "market-data-account"),
    "password":        os.getenv("PGPASSWORD", "market-data-account-pw"),
    "connect_timeout": int(os.getenv("CONNECT_TIMEOUT_SECONDS", "10")),
    "application_name": os.getenv("PGAPPNAME", "backtest_runner"),
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}

__all__ = [name for name in globals() if not name.startswith("__")]
