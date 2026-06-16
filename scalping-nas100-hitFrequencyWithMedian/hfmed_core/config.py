"""Environment-backed configuration for the NAS100 hit-frequency median backtest."""

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    return text if text else default


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


def env_float_alias(name: str, legacy_name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is not None and raw.strip():
        return float(raw)
    return env_float(legacy_name, default)


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _one_of(name: str, default: str, choices: set[str]) -> str:
    value = env_str(name, default).lower()
    if value not in choices:
        raise ValueError(f"{name}={value!r} invalid; expected one of {sorted(choices)}")
    return value


def env_optional_ts_utc(name: str) -> Optional[datetime]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    text = raw.strip()
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset or Z suffix")
    return parsed.astimezone(timezone.utc)


def _validate_identifier_path(value: str, name: str) -> None:
    parts = value.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"{name} must be table or schema.table, got {value!r}")
    for part in parts:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
            raise ValueError(f"{name} contains invalid identifier part {part!r}")


# Data source.
SOURCE_TABLE = env_str("SOURCE_TABLE", "public.pepperstone_ticks_data")
_validate_identifier_path(SOURCE_TABLE, "SOURCE_TABLE")
SYMBOL = env_str("SYMBOL", "NAS100")
START_TS_UTC = env_optional_ts_utc("START_TS_UTC")
END_TS_UTC = env_optional_ts_utc("END_TS_UTC")
if START_TS_UTC is not None and END_TS_UTC is not None and END_TS_UTC <= START_TS_UTC:
    raise ValueError("END_TS_UTC must be greater than START_TS_UTC")

# Hit-frequency profile.
BAR_SECONDS = max(1, env_int("BAR_SECONDS", 10))
LOOKBACK_BARS = max(1, env_int("LOOKBACK_BARS", 10))
MIN_LOOKBACK_BARS = max(1, env_int("MIN_LOOKBACK_BARS", LOOKBACK_BARS))
if MIN_LOOKBACK_BARS > LOOKBACK_BARS:
    raise ValueError("MIN_LOOKBACK_BARS must be <= LOOKBACK_BARS")
PRICE_STEP = env_float("PRICE_STEP", 1.0)
if PRICE_STEP <= 0:
    raise ValueError("PRICE_STEP must be positive")
MEDIAN_QUANTILE = 0.5
BAND_LOWER_QUANTILE = 0.45
BAND_UPPER_QUANTILE = 0.55

# Trade rules.
STOP_MODE = _one_of("STOP_MODE", "band", {"fixed", "band"})
STOP_POINTS = env_float("STOP_POINTS", 10.0)
TAKE_PROFIT_POINTS = env_float("TAKE_PROFIT_POINTS", 10.0)
if STOP_POINTS <= 0 or TAKE_PROFIT_POINTS <= 0:
    raise ValueError("STOP_POINTS and TAKE_PROFIT_POINTS must be positive")
MIN_PROFILE_RANGE_POINTS = env_float("MIN_PROFILE_RANGE_POINTS", env_float("MIN_BAND_POINTS", 40.0))
BAND_STOP_BUFFER_POINTS = env_float("BAND_STOP_BUFFER_POINTS", 0.5)
MIN_STOP_DISTANCE_POINTS = env_float_alias("MIN_STOP_DISTANCE_POINTS", "MIN_STOP_POINTS", 12.0)
MAX_STOP_DISTANCE_POINTS = env_float_alias("MAX_STOP_DISTANCE_POINTS", "MAX_STOP_POINTS", 20.0)
if MIN_PROFILE_RANGE_POINTS < 0:
    raise ValueError("MIN_PROFILE_RANGE_POINTS must be >= 0")
if BAND_STOP_BUFFER_POINTS < 0:
    raise ValueError("BAND_STOP_BUFFER_POINTS must be >= 0")
if MIN_STOP_DISTANCE_POINTS <= 0 or MAX_STOP_DISTANCE_POINTS <= MIN_STOP_DISTANCE_POINTS:
    raise ValueError("MAX_STOP_DISTANCE_POINTS must be greater than MIN_STOP_DISTANCE_POINTS")

# Account profile: PS_ACC.
ACCOUNT_PROFILE = env_str("ACCOUNT_PROFILE", "PS_ACC").upper()
INITIAL_EQUITY = env_float("INITIAL_EQUITY", 5000.0)
ACCOUNT_CURRENCY = env_str("ACCOUNT_CURRENCY", "EUR")
MARGIN_REQUIREMENT_PCT = env_float("MARGIN_REQUIREMENT_PCT", 5.0)
RISK_PER_TRADE_PCT = env_float("RISK_PER_TRADE_PCT", 1.5)
MAX_MARGIN_PCT = env_float("MAX_MARGIN_PCT", 45.0)
CONTRACT_MULTIPLIER = env_float("CONTRACT_MULTIPLIER", 1.0)
LOT_SIZE = env_float("LOT_SIZE", 0.1)
if LOT_SIZE <= 0:
    raise ValueError("LOT_SIZE must be positive")
EURUSD_RATE = env_float("EURUSD_RATE", 1.0)

# Extra costs. The live bid/ask spread from ticks is already included in fills.
SPREAD_POINTS = env_float("SPREAD_POINTS", 0.0)
SLIPPAGE_POINTS = env_float("SLIPPAGE_POINTS", 0.0)
COMMISSION_PER_UNIT = env_float("COMMISSION_PER_UNIT", 0.0)

# Monte-Carlo risk.
MONTE_CARLO_ENABLED = env_bool("MONTE_CARLO_ENABLED", True)
MONTE_CARLO_SIMULATIONS = max(0, env_int("MONTE_CARLO_SIMULATIONS", 2000))
MC_EXTRA_SLIPPAGE_POINTS = env_float("MC_EXTRA_SLIPPAGE_POINTS", 0.5)
MC_BLOCK_SIZE = max(1, env_int("MC_BLOCK_SIZE", 5))
MC_RUIN_DRAWDOWN_PCT = env_float("MC_RUIN_DRAWDOWN_PCT", 50.0)
MC_RANDOM_SEED = env_int("MC_RANDOM_SEED", 12345)

# Run metadata.
RUN_LABEL_TZ = env_str("RUN_LABEL_TZ", "Europe/Berlin")
RUN_NOTES_EXTRA = env_str("RUN_NOTES_EXTRA", "")

# Result schema and DB.
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
    "application_name": os.getenv("PGAPPNAME", "nas100_hfmed_backtest_runner"),
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}


@dataclass(frozen=True)
class RunConfig:
    source_table: str
    symbol: str
    start_ts_utc: Optional[datetime]
    end_ts_utc: Optional[datetime]
    bar_seconds: int
    lookback_bars: int
    min_lookback_bars: int
    price_step: float
    median_quantile: float
    band_lower_quantile: float
    band_upper_quantile: float
    stop_mode: str
    stop_points: float
    take_profit_points: float
    min_profile_range_points: float
    band_stop_buffer_points: float
    min_stop_distance_points: float
    max_stop_distance_points: float
    account_profile: str
    initial_equity: float
    account_currency: str
    margin_requirement_pct: float
    risk_per_trade_pct: float
    max_margin_pct: float
    contract_multiplier: float
    lot_size: float
    eurusd_rate: float
    spread_points: float
    slippage_points: float
    commission_per_unit: float
    monte_carlo_enabled: bool
    monte_carlo_simulations: int
    mc_extra_slippage_points: float
    mc_block_size: int
    mc_ruin_drawdown_pct: float
    mc_random_seed: int


def active_run_config() -> RunConfig:
    return RunConfig(
        source_table=SOURCE_TABLE,
        symbol=SYMBOL,
        start_ts_utc=START_TS_UTC,
        end_ts_utc=END_TS_UTC,
        bar_seconds=BAR_SECONDS,
        lookback_bars=LOOKBACK_BARS,
        min_lookback_bars=MIN_LOOKBACK_BARS,
        price_step=PRICE_STEP,
        median_quantile=MEDIAN_QUANTILE,
        band_lower_quantile=BAND_LOWER_QUANTILE,
        band_upper_quantile=BAND_UPPER_QUANTILE,
        stop_mode=STOP_MODE,
        stop_points=STOP_POINTS,
        take_profit_points=TAKE_PROFIT_POINTS,
        min_profile_range_points=MIN_PROFILE_RANGE_POINTS,
        band_stop_buffer_points=BAND_STOP_BUFFER_POINTS,
        min_stop_distance_points=MIN_STOP_DISTANCE_POINTS,
        max_stop_distance_points=MAX_STOP_DISTANCE_POINTS,
        account_profile=ACCOUNT_PROFILE,
        initial_equity=INITIAL_EQUITY,
        account_currency=ACCOUNT_CURRENCY,
        margin_requirement_pct=MARGIN_REQUIREMENT_PCT,
        risk_per_trade_pct=RISK_PER_TRADE_PCT,
        max_margin_pct=MAX_MARGIN_PCT,
        contract_multiplier=CONTRACT_MULTIPLIER,
        lot_size=LOT_SIZE,
        eurusd_rate=EURUSD_RATE,
        spread_points=SPREAD_POINTS,
        slippage_points=SLIPPAGE_POINTS,
        commission_per_unit=COMMISSION_PER_UNIT,
        monte_carlo_enabled=MONTE_CARLO_ENABLED,
        monte_carlo_simulations=MONTE_CARLO_SIMULATIONS,
        mc_extra_slippage_points=MC_EXTRA_SLIPPAGE_POINTS,
        mc_block_size=MC_BLOCK_SIZE,
        mc_ruin_drawdown_pct=MC_RUIN_DRAWDOWN_PCT,
        mc_random_seed=MC_RANDOM_SEED,
    )
