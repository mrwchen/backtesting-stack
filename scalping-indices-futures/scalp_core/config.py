"""Environment-backed configuration for one scalping backtest run.

All tunables come from environment variables (set in docker-compose.yml). The four
layer switches the user asked for are:
    PRICE_MODEL     = kalman | state_space
    VOL_MODEL       = garch  | egarch
    DECISION_MODEL  = bayes  | logistic
plus the regime layer (always HMM) and the Monte-Carlo risk layer.
"""

import os
import re
from dataclasses import dataclass
from datetime import date
from typing import Optional


# ── env helpers (same contract as swing-stocks/backtest_shared) ────────────────

def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    return text or default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _env_date(name: str) -> Optional[date]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return date.fromisoformat(raw.strip())


def _one_of(name: str, default: str, choices: set[str]) -> str:
    value = env_str(name, default).lower()
    if value not in choices:
        raise ValueError(f"{name}={value!r} invalid; expected one of {sorted(choices)}")
    return value


# ── data source ────────────────────────────────────────────────────────────────

SOURCE_TABLE = env_str("SOURCE_TABLE", "ibkr_market_data")
SYMBOL = env_str("SYMBOL", "NQ")
BAR_SIZE = env_str("BAR_SIZE", "1 min")
START_DATE = _env_date("START_DATE")  # None -> auto full available range
END_DATE = _env_date("END_DATE")

# ── layer switches ──────────────────────────────────────────────────────────────

PRICE_MODEL = _one_of("PRICE_MODEL", "kalman", {"kalman", "state_space"})
VOL_MODEL = _one_of("VOL_MODEL", "garch", {"garch", "egarch"})
DECISION_MODEL = _one_of("DECISION_MODEL", "bayes", {"bayes", "logistic"})

# Kalman local-linear-trend noise multipliers (relative to training diff variance,
# so the filter is self-scaling across instruments of different price levels).
KF_OBS_NOISE_MULT = env_float("KF_OBS_NOISE_MULT", 1.0)
KF_LEVEL_NOISE_MULT = env_float("KF_LEVEL_NOISE_MULT", 0.1)
KF_TREND_NOISE_MULT = env_float("KF_TREND_NOISE_MULT", 0.001)

# ── feature windows (inputs to the decision layer + ATR for stops) ──────────────

RSI_PERIOD = max(2, env_int("RSI_PERIOD", 14))
ROLL_VOL_BARS = max(2, env_int("ROLL_VOL_BARS", 20))
MOMENTUM_BARS = max(1, env_int("MOMENTUM_BARS", 10))
ATR_BARS = max(2, env_int("ATR_BARS", 14))

# ── layer 1: regime (HMM) ───────────────────────────────────────────────────────

REGIME_STATES = max(2, env_int("REGIME_STATES", 3))
REGIME_BLOCK_HIGH_VOL_STATE = env_bool("REGIME_BLOCK_HIGH_VOL_STATE", False)
HMM_N_ITER = max(10, env_int("HMM_N_ITER", 100))
HMM_COVARIANCE_TYPE = _one_of("HMM_COVARIANCE_TYPE", "diag", {"diag", "full", "tied", "spherical"})

# ── layer 3: volatility model detail (GARCH/EGARCH order + error distribution) ──

GARCH_P = max(1, env_int("GARCH_P", 1))
GARCH_Q = max(1, env_int("GARCH_Q", 1))
GARCH_DIST = _one_of("GARCH_DIST", "normal", {"normal", "studentst", "skewt", "ged"})

# ── layer 4: decision model detail ──────────────────────────────────────────────

LOGISTIC_C = env_float("LOGISTIC_C", 1.0)
MIN_TRAIN_ROWS = max(20, env_int("MIN_TRAIN_ROWS", 50))

# ── walk-forward fitting ────────────────────────────────────────────────────────

WARMUP_BARS = max(200, env_int("WARMUP_BARS", 1500))
TRAIN_WINDOW_BARS = env_int("TRAIN_WINDOW_BARS", 0)  # 0 = expanding window
REFIT_EVERY_BARS = max(1, env_int("REFIT_EVERY_BARS", 250))

# ── layer 4: decision threshold ─────────────────────────────────────────────────

PROB_THRESHOLD = env_float("PROB_THRESHOLD", 0.55)

# ── stop-loss / take-profit logic ───────────────────────────────────────────────
# STOP_MODE chooses the distance basis: "vol" = GARCH/EGARCH sigma, "atr" = ATR.
# TP_MODE chooses the exit:            "fixed" = hard take-profit, "trailing" = trailing stop.
STOP_MODE = _one_of("STOP_MODE", "vol", {"vol", "atr"})
TP_MODE = _one_of("TP_MODE", "fixed", {"fixed", "trailing"})
# vol-basis multipliers (used when STOP_MODE=vol)
STOP_VOL_MULT = env_float("STOP_VOL_MULT", 2.0)
TP_VOL_MULT = env_float("TP_VOL_MULT", 3.0)
# atr-basis multipliers (used when STOP_MODE=atr)
STOP_ATR_MULT = env_float("STOP_ATR_MULT", 2.0)
TP_ATR_MULT = env_float("TP_ATR_MULT", 3.0)
# trailing-stop multipliers (applied to the same basis unit, used when TP_MODE=trailing)
TRAILING_ACTIVATION_MULT = env_float("TRAILING_ACTIVATION_MULT", 2.0)
TRAILING_DISTANCE_MULT = env_float("TRAILING_DISTANCE_MULT", 1.5)
# clamp on the stop distance (percent of price) — bounds sizing in both modes
MIN_STOP_PCT = env_float("MIN_STOP_PCT", 0.05)
MAX_STOP_PCT = env_float("MAX_STOP_PCT", 0.6)
MAX_HOLD_BARS = max(1, env_int("MAX_HOLD_BARS", 60))
ALLOW_SHORT = env_bool("ALLOW_SHORT", True)

# ── trade management ────────────────────────────────────────────────────────────
# Cooldown after an exit before a new entry is allowed; tie-break when SL and TP
# would both trigger inside the same bar.
REENTRY_COOLDOWN_BARS = max(0, env_int("REENTRY_COOLDOWN_BARS", 0))
INTRABAR_FILL_PRIORITY = _one_of("INTRABAR_FILL_PRIORITY", "stop", {"stop", "tp"})

# ── session handling (intraday-only, flat at cutoff) ────────────────────────────

ENTRY_START_TIME = env_str("ENTRY_START_TIME", "09:30")
ENTRY_END_TIME = env_str("ENTRY_END_TIME", "15:45")
SESSION_FLAT_TIME = env_str("SESSION_FLAT_TIME", "16:55")
SESSION_TZ = env_str("SESSION_TZ", "America/New_York")

# ── account profile: PS_ACC ─────────────────────────────────────────────────────

ACCOUNT_PROFILE = env_str("ACCOUNT_PROFILE", "PS_ACC").upper()
INITIAL_EQUITY = env_float("INITIAL_EQUITY", 5000.0)
ACCOUNT_CURRENCY = env_str("ACCOUNT_CURRENCY", "EUR")
MARGIN_REQUIREMENT_PCT = env_float("MARGIN_REQUIREMENT_PCT", 5.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.5)
MAX_MARGIN_PCT = env_float("MAX_MARGIN_PCT", 45.0)
CONTRACT_MULTIPLIER = env_float("CONTRACT_MULTIPLIER", 1.0)
EURUSD_RATE = env_float("EURUSD_RATE", 1.0)  # USD price -> EUR equity; 1.0 = no conversion

# ── costs ───────────────────────────────────────────────────────────────────────

SPREAD_POINTS = env_float("SPREAD_POINTS", 1.2)
SLIPPAGE_POINTS = env_float("SLIPPAGE_POINTS", 0.3)
SPREAD_BPS = env_float("SPREAD_BPS", 0.0)
SLIPPAGE_BPS = env_float("SLIPPAGE_BPS", 0.0)
COMMISSION_PER_UNIT = env_float("COMMISSION_PER_UNIT", 0.0)

# ── layer 5: Monte-Carlo risk ───────────────────────────────────────────────────

MONTE_CARLO_ENABLED = env_bool("MONTE_CARLO_ENABLED", True)
MONTE_CARLO_SIMULATIONS = max(0, env_int("MONTE_CARLO_SIMULATIONS", 2000))
MC_EXTRA_SLIPPAGE_POINTS = env_float("MC_EXTRA_SLIPPAGE_POINTS", 0.5)
MC_EXTRA_SLIPPAGE_BPS = env_float("MC_EXTRA_SLIPPAGE_BPS", 0.0)
MC_BLOCK_SIZE = max(1, env_int("MC_BLOCK_SIZE", 5))
MC_RUIN_DRAWDOWN_PCT = env_float("MC_RUIN_DRAWDOWN_PCT", 50.0)
MC_RANDOM_SEED = env_int("MC_RANDOM_SEED", 12345)  # fixed seed = reproducible MC

# ── run metadata ────────────────────────────────────────────────────────────────

RUN_LABEL_TZ = env_str("RUN_LABEL_TZ", "Europe/Berlin")
RUN_NOTES_EXTRA = env_str("RUN_NOTES_EXTRA", "")

# ── result schema & DB ──────────────────────────────────────────────────────────

RESULT_SCHEMA = env_str("RESULT_SCHEMA", "public")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", RESULT_SCHEMA):
    raise ValueError(f"Invalid RESULT_SCHEMA: {RESULT_SCHEMA!r}")

DB_CONNECT_RETRIES = env_int("DB_CONNECT_RETRIES", 5)
DB_CONNECT_RETRY_DELAY_SECONDS = env_float("DB_CONNECT_RETRY_DELAY_SECONDS", 5.0)
DB_STATEMENT_TIMEOUT_MS = env_int("DB_STATEMENT_TIMEOUT_MS", 120000)
DB_LOCK_TIMEOUT_MS = env_int("DB_LOCK_TIMEOUT_MS", 5000)
DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS = env_int("DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS", 120000)

DB = {
    "host": os.getenv("PGHOST", "timescaledb"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "postgres"),
    "user": os.getenv("PGUSER", "market-data-account"),
    "password": os.getenv("PGPASSWORD", "market-data-account-pw"),
    "connect_timeout": int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "10")),
    "application_name": os.getenv("PGAPPNAME", "scalp_backtest_runner"),
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}


@dataclass(frozen=True)
class RunConfig:
    """Immutable snapshot of the active configuration, persisted with the run."""

    symbol: str
    bar_size: str
    price_model: str
    vol_model: str
    decision_model: str
    rsi_period: int
    roll_vol_bars: int
    momentum_bars: int
    atr_bars: int
    regime_states: int
    regime_block_high_vol_state: bool
    hmm_n_iter: int
    hmm_covariance_type: str
    garch_p: int
    garch_q: int
    garch_dist: str
    logistic_c: float
    min_train_rows: int
    warmup_bars: int
    train_window_bars: int
    refit_every_bars: int
    prob_threshold: float
    stop_mode: str
    tp_mode: str
    stop_vol_mult: float
    tp_vol_mult: float
    stop_atr_mult: float
    tp_atr_mult: float
    trailing_activation_mult: float
    trailing_distance_mult: float
    min_stop_pct: float
    max_stop_pct: float
    max_hold_bars: int
    allow_short: bool
    reentry_cooldown_bars: int
    intrabar_fill_priority: str
    session_flat_time: str
    session_tz: str
    account_profile: str
    initial_equity: float
    account_currency: str
    margin_requirement_pct: float
    risk_per_trade_pct: float
    max_margin_pct: float
    contract_multiplier: float
    eurusd_rate: float
    spread_points: float
    slippage_points: float
    spread_bps: float
    slippage_bps: float
    commission_per_unit: float
    mc_extra_slippage_points: float
    mc_extra_slippage_bps: float
    mc_random_seed: int


def active_run_config() -> RunConfig:
    return RunConfig(
        symbol=SYMBOL,
        bar_size=BAR_SIZE,
        price_model=PRICE_MODEL,
        vol_model=VOL_MODEL,
        decision_model=DECISION_MODEL,
        rsi_period=RSI_PERIOD,
        roll_vol_bars=ROLL_VOL_BARS,
        momentum_bars=MOMENTUM_BARS,
        atr_bars=ATR_BARS,
        regime_states=REGIME_STATES,
        regime_block_high_vol_state=REGIME_BLOCK_HIGH_VOL_STATE,
        hmm_n_iter=HMM_N_ITER,
        hmm_covariance_type=HMM_COVARIANCE_TYPE,
        garch_p=GARCH_P,
        garch_q=GARCH_Q,
        garch_dist=GARCH_DIST,
        logistic_c=LOGISTIC_C,
        min_train_rows=MIN_TRAIN_ROWS,
        warmup_bars=WARMUP_BARS,
        train_window_bars=TRAIN_WINDOW_BARS,
        refit_every_bars=REFIT_EVERY_BARS,
        prob_threshold=PROB_THRESHOLD,
        stop_mode=STOP_MODE,
        tp_mode=TP_MODE,
        stop_vol_mult=STOP_VOL_MULT,
        tp_vol_mult=TP_VOL_MULT,
        stop_atr_mult=STOP_ATR_MULT,
        tp_atr_mult=TP_ATR_MULT,
        trailing_activation_mult=TRAILING_ACTIVATION_MULT,
        trailing_distance_mult=TRAILING_DISTANCE_MULT,
        min_stop_pct=MIN_STOP_PCT,
        max_stop_pct=MAX_STOP_PCT,
        max_hold_bars=MAX_HOLD_BARS,
        allow_short=ALLOW_SHORT,
        reentry_cooldown_bars=REENTRY_COOLDOWN_BARS,
        intrabar_fill_priority=INTRABAR_FILL_PRIORITY,
        session_flat_time=SESSION_FLAT_TIME,
        session_tz=SESSION_TZ,
        account_profile=ACCOUNT_PROFILE,
        initial_equity=INITIAL_EQUITY,
        account_currency=ACCOUNT_CURRENCY,
        margin_requirement_pct=MARGIN_REQUIREMENT_PCT,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        max_margin_pct=MAX_MARGIN_PCT,
        contract_multiplier=CONTRACT_MULTIPLIER,
        eurusd_rate=EURUSD_RATE,
        spread_points=SPREAD_POINTS,
        slippage_points=SLIPPAGE_POINTS,
        spread_bps=SPREAD_BPS,
        slippage_bps=SLIPPAGE_BPS,
        commission_per_unit=COMMISSION_PER_UNIT,
        mc_extra_slippage_points=MC_EXTRA_SLIPPAGE_POINTS,
        mc_extra_slippage_bps=MC_EXTRA_SLIPPAGE_BPS,
        mc_random_seed=MC_RANDOM_SEED,
    )
