"""Environment-backed configuration for the NAS100 hit-frequency median backtest."""

import math
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo


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


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def env_int_list(name: str, default: str = "") -> tuple[int, ...]:
    raw = os.getenv(name)
    text = default if raw is None else raw.strip()
    if not text:
        return ()
    values: list[int] = []
    seen: set[int] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"{name} values must be positive integers")
        if value not in seen:
            values.append(value)
            seen.add(value)
    return tuple(values)


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


def _validate_timezone(value: str, name: str) -> None:
    try:
        ZoneInfo(value)
    except Exception as exc:
        raise ValueError(f"{name} must be a valid IANA timezone, got {value!r}") from exc


# Runner mode.
RUN_MODE = _one_of("RUN_MODE", "single", {"single", "walk_forward"})

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
MIN_LOOKBACK_BARS = max(1, env_int("MIN_LOOKBACK_BARS", 30))
LOOKBACK_BARS = MIN_LOOKBACK_BARS
PROFILE_MAX_LOOKBACK_SECONDS = env_optional_int("PROFILE_MAX_LOOKBACK_SECONDS")
if PROFILE_MAX_LOOKBACK_SECONDS is not None and PROFILE_MAX_LOOKBACK_SECONDS <= 0:
    raise ValueError("PROFILE_MAX_LOOKBACK_SECONDS must be positive when set")
PRICE_STEP = env_float("PRICE_STEP", 1.0)
if PRICE_STEP <= 0:
    raise ValueError("PRICE_STEP must be positive")
MEDIAN_QUANTILE = 0.5
BAND_LOWER_QUANTILE = 0.45
BAND_UPPER_QUANTILE = 0.55
LONG_CROSS_QUANTILE = MEDIAN_QUANTILE
SHORT_CROSS_QUANTILE = MEDIAN_QUANTILE
ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT = 0.0

# Trade rules.
STOP_MODE = _one_of("STOP_MODE", "band", {"fixed", "band"})
FIXED_STOP_POINTS = env_float("FIXED_STOP_POINTS", 10.0)
if FIXED_STOP_POINTS <= 0:
    raise ValueError("FIXED_STOP_POINTS must be positive")
ALL_STOP_MODES_TAKE_PROFIT_BPS = 1.0
BAND_STOP_MIN_PROFILE_RANGE_BPS = 0.0
BAND_STOP_PROFILE_LOWER_QUANTILE = 0.0
BAND_STOP_PROFILE_UPPER_QUANTILE = 1.0
BAND_STOP_PROFILE_BUFFER_POINTS = 0.0
BAND_STOP_MIN_DISTANCE_BPS = 1.0
BAND_STOP_MAX_DISTANCE_BPS = 2.0

# Entry session switches. Session boundaries are interpreted in SESSION_TIMEZONE.
SESSION_TIMEZONE = env_str("SESSION_TIMEZONE", "America/New_York")
_validate_timezone(SESSION_TIMEZONE, "SESSION_TIMEZONE")
SESSION_ASIA_EARLY_ENABLED = env_bool("SESSION_ASIA_EARLY_ENABLED", True)
SESSION_ASIA_LATE_ENABLED = env_bool("SESSION_ASIA_LATE_ENABLED", True)
SESSION_LONDON_OPEN_ENABLED = env_bool("SESSION_LONDON_OPEN_ENABLED", True)
SESSION_PRE_MARKET_EARLY_ENABLED = env_bool("SESSION_PRE_MARKET_EARLY_ENABLED", True)
SESSION_PRE_MARKET_ACTIVE_ENABLED = env_bool("SESSION_PRE_MARKET_ACTIVE_ENABLED", True)
SESSION_PRE_MARKET_MACRO_ENABLED = env_bool("SESSION_PRE_MARKET_MACRO_ENABLED", True)
SESSION_NY_OPEN_IMPULSE_ENABLED = env_bool("SESSION_NY_OPEN_IMPULSE_ENABLED", True)
SESSION_NY_MORNING_ENABLED = env_bool("SESSION_NY_MORNING_ENABLED", True)
SESSION_NY_MIDDAY_ENABLED = env_bool("SESSION_NY_MIDDAY_ENABLED", True)
SESSION_NY_LATE_ENABLED = env_bool("SESSION_NY_LATE_ENABLED", True)
SESSION_NY_POWER_HOUR_ENABLED = env_bool("SESSION_NY_POWER_HOUR_ENABLED", True)
SESSION_AFTER_CLOSE_SHOCK_ENABLED = env_bool("SESSION_AFTER_CLOSE_SHOCK_ENABLED", True)
SESSION_AFTER_HOURS_LATE_ENABLED = env_bool("SESSION_AFTER_HOURS_LATE_ENABLED", True)

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

# Walk-forward optimizer.
SINGLE_PARAMETER_PATH = env_str("SINGLE_PARAMETER_PATH", "single_parameter.ini")
PARAMETER_GRID_PATH = env_str("PARAMETER_GRID_PATH", "parameter_grid.ini")
WF_TRAIN_DAYS = max(1, env_int("WF_TRAIN_DAYS", 60))
WF_TEST_DAYS = max(1, env_int("WF_TEST_DAYS", 20))
WF_STEP_DAYS = max(1, env_int("WF_STEP_DAYS", 20))
WF_MATRIX_ENABLED = env_bool("WF_MATRIX_ENABLED", False)
WF_MATRIX_TRAIN_DAYS = env_int_list("WF_MATRIX_TRAIN_DAYS")
WF_MATRIX_TEST_DAYS = env_int_list("WF_MATRIX_TEST_DAYS")
if WF_MATRIX_ENABLED and (not WF_MATRIX_TRAIN_DAYS or not WF_MATRIX_TEST_DAYS):
    raise ValueError("WF_MATRIX_TRAIN_DAYS and WF_MATRIX_TEST_DAYS must be set when WF_MATRIX_ENABLED=true")
# Finalist OOS validation. A fixed candidate set (per-session train top-N union
# global train top-N, capped at MAX_SETS) is evaluated OOS in *every* fold, so the
# per-candidate oos_* columns have consistent, comparable full-fold coverage.
WF_FINALIST_ENABLED = env_bool("WF_FINALIST_ENABLED", True)
WF_FINALIST_TOP_N_PER_SESSION = max(0, env_int("WF_FINALIST_TOP_N_PER_SESSION", 0))
WF_FINALIST_GLOBAL_TOP_N = max(0, env_int("WF_FINALIST_GLOBAL_TOP_N", 0))
WF_FINALIST_MAX_SETS = max(0, env_int("WF_FINALIST_MAX_SETS", 2000))
WF_FINALIST_KEEP_TRADES = env_bool("WF_FINALIST_KEEP_TRADES", False)
WF_FINALIST_PERSIST_FOLD_RESULTS = env_bool("WF_FINALIST_PERSIST_FOLD_RESULTS", True)
OPTIMIZER_PROCESSES = max(1, env_int("OPTIMIZER_PROCESSES", 1))
OPTIMIZER_PROCESS_CHUNK_SIZE = max(1, env_int("OPTIMIZER_PROCESS_CHUNK_SIZE", 32))
OPTIMIZER_PROFILE_CACHE_SIZE = max(0, env_int("OPTIMIZER_PROFILE_CACHE_SIZE", 4))
OPTIMIZER_PROGRESS_LOG_EVERY = max(1, env_int("OPTIMIZER_PROGRESS_LOG_EVERY", 5000))
OPTIMIZER_PROGRESS_LOG_SECONDS = max(1, env_int("OPTIMIZER_PROGRESS_LOG_SECONDS", 60))
OPTIMIZER_SAMPLING_SEED = env_int("OPTIMIZER_SAMPLING_SEED", 12345)
STAGE1_MAX_PARAMETER_SETS = max(0, env_int("STAGE1_MAX_PARAMETER_SETS", 0))
STAGE2_ENABLED = env_bool("STAGE2_ENABLED", True)
STAGE2_SEED_TOP_N = max(1, env_int("STAGE2_SEED_TOP_N", 20))
STAGE2_MAX_PARAMETER_SETS = max(0, env_int("STAGE2_MAX_PARAMETER_SETS", 0))
MC_SCORE_TOP_N = max(0, env_int("MC_SCORE_TOP_N", 100))
PERSIST_TOP_TRADES_N = max(0, env_int("PERSIST_TOP_TRADES_N", 20))
MIN_OOS_TRADES = max(0, env_int("MIN_OOS_TRADES", 100))
MIN_OOS_PROFIT_FACTOR = env_float("MIN_OOS_PROFIT_FACTOR", 1.1)
MAX_OOS_DRAWDOWN_PCT = env_float("MAX_OOS_DRAWDOWN_PCT", 25.0)
MAX_MC_RUIN_PCT = env_float("MAX_MC_RUIN_PCT", 5.0)
SESSION_SELECTOR_MIN_TRADES_FLOOR = max(1, env_int("SESSION_SELECTOR_MIN_TRADES_FLOOR", 20))
SESSION_SELECTOR_MIN_TRADES_PER_TRAIN_DAY = max(0.0, env_float("SESSION_SELECTOR_MIN_TRADES_PER_TRAIN_DAY", 0.0))
SESSION_SELECTOR_LCB_Z = max(0.0, env_float("SESSION_SELECTOR_LCB_Z", 1.0))
SESSION_SELECTOR_TOP_N = max(1, env_int("SESSION_SELECTOR_TOP_N", 25))
SESSION_SELECTOR_PLATEAU_WEIGHT = min(1.0, max(0.0, env_float("SESSION_SELECTOR_PLATEAU_WEIGHT", 0.35)))
SESSION_SELECTOR_NEIGHBOR_DISTANCE = max(0.0, env_float("SESSION_SELECTOR_NEIGHBOR_DISTANCE", 1.25))
SESSION_SELECTOR_PREVIOUS_KEEP_SCORE_TOLERANCE = max(0.0, env_float("SESSION_SELECTOR_PREVIOUS_KEEP_SCORE_TOLERANCE", 2.0))

# Result schema and DB.
RESULT_SCHEMA = env_str("RESULT_SCHEMA", "public")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", RESULT_SCHEMA):
    raise ValueError(f"Invalid RESULT_SCHEMA: {RESULT_SCHEMA!r}")

TRADE_HISTORY_SCHEMA = env_str("TRADE_HISTORY_SCHEMA", "public")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", TRADE_HISTORY_SCHEMA):
    raise ValueError(f"Invalid TRADE_HISTORY_SCHEMA: {TRADE_HISTORY_SCHEMA!r}")
TRADE_HISTORY_TABLE = env_str("TRADE_HISTORY_TABLE", "trade_history")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", TRADE_HISTORY_TABLE):
    raise ValueError(f"Invalid TRADE_HISTORY_TABLE: {TRADE_HISTORY_TABLE!r}")
TRADE_HISTORY_ACCOUNT_NUMBER = env_str("TRADE_HISTORY_ACCOUNT_NUMBER", "000001")
TRADE_HISTORY_ACCOUNT_TYPE = env_str("TRADE_HISTORY_ACCOUNT_TYPE", "backtesting")
TRADE_HISTORY_BROKER_NAME = env_str("TRADE_HISTORY_BROKER_NAME", "backtest")
TRADE_HISTORY_CHANNEL = env_str("TRADE_HISTORY_CHANNEL", "nas100_hfmed_single")
TRADE_HISTORY_LABEL = env_str("TRADE_HISTORY_LABEL", "nas100_hfmed_single")
TRADE_HISTORY_PERSIST_ENABLED = env_bool("TRADE_HISTORY_PERSIST_ENABLED", False)

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
    profile_max_lookback_seconds: Optional[int]
    price_step: float
    median_quantile: float
    band_lower_quantile: float
    band_upper_quantile: float
    long_cross_quantile: float
    short_cross_quantile: float
    entry_price_range_position_max_deviation_pct: float
    stop_mode: str
    stop_points: float
    take_profit_bps: float
    min_profile_range_bps: float
    stop_profile_lower_quantile: float
    stop_profile_upper_quantile: float
    stop_profile_buffer_points: float
    min_stop_distance_bps: float
    max_stop_distance_bps: float
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
    session_timezone: str
    session_asia_early_enabled: bool
    session_asia_late_enabled: bool
    session_london_open_enabled: bool
    session_pre_market_early_enabled: bool
    session_pre_market_active_enabled: bool
    session_pre_market_macro_enabled: bool
    session_ny_open_impulse_enabled: bool
    session_ny_morning_enabled: bool
    session_ny_midday_enabled: bool
    session_ny_late_enabled: bool
    session_ny_power_hour_enabled: bool
    session_after_close_shock_enabled: bool
    session_after_hours_late_enabled: bool
    monte_carlo_enabled: bool
    monte_carlo_simulations: int
    mc_extra_slippage_points: float
    mc_block_size: int
    mc_ruin_drawdown_pct: float
    mc_random_seed: int


@dataclass(frozen=True)
class OptimizerConfig:
    single_parameter_path: str
    parameter_grid_path: str
    train_days: int
    test_days: int
    step_days: int
    finalist_enabled: bool
    finalist_top_n_per_session: int
    finalist_global_top_n: int
    finalist_max_sets: int
    finalist_keep_trades: bool
    finalist_persist_fold_results: bool
    processes: int
    process_chunk_size: int
    profile_cache_size: int
    progress_log_every: int
    progress_log_seconds: int
    sampling_seed: int
    stage1_max_parameter_sets: int
    stage2_enabled: bool
    stage2_seed_top_n: int
    stage2_max_parameter_sets: int
    mc_score_top_n: int
    persist_top_trades_n: int
    min_oos_trades: int
    min_oos_profit_factor: float
    max_oos_drawdown_pct: float
    max_mc_ruin_pct: float
    session_selector_min_trades_floor: int
    session_selector_min_trades_per_train_day: float
    session_selector_lcb_z: float
    session_selector_top_n: int
    session_selector_plateau_weight: float
    session_selector_neighbor_distance: float
    session_selector_previous_keep_score_tolerance: float


def active_run_config() -> RunConfig:
    return RunConfig(
        source_table=SOURCE_TABLE,
        symbol=SYMBOL,
        start_ts_utc=START_TS_UTC,
        end_ts_utc=END_TS_UTC,
        bar_seconds=BAR_SECONDS,
        lookback_bars=LOOKBACK_BARS,
        min_lookback_bars=MIN_LOOKBACK_BARS,
        profile_max_lookback_seconds=PROFILE_MAX_LOOKBACK_SECONDS,
        price_step=PRICE_STEP,
        median_quantile=MEDIAN_QUANTILE,
        band_lower_quantile=BAND_LOWER_QUANTILE,
        band_upper_quantile=BAND_UPPER_QUANTILE,
        long_cross_quantile=LONG_CROSS_QUANTILE,
        short_cross_quantile=SHORT_CROSS_QUANTILE,
        entry_price_range_position_max_deviation_pct=ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT,
        stop_mode=STOP_MODE,
        stop_points=FIXED_STOP_POINTS,
        take_profit_bps=ALL_STOP_MODES_TAKE_PROFIT_BPS,
        min_profile_range_bps=BAND_STOP_MIN_PROFILE_RANGE_BPS,
        stop_profile_lower_quantile=BAND_STOP_PROFILE_LOWER_QUANTILE,
        stop_profile_upper_quantile=BAND_STOP_PROFILE_UPPER_QUANTILE,
        stop_profile_buffer_points=BAND_STOP_PROFILE_BUFFER_POINTS,
        min_stop_distance_bps=BAND_STOP_MIN_DISTANCE_BPS,
        max_stop_distance_bps=BAND_STOP_MAX_DISTANCE_BPS,
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
        session_timezone=SESSION_TIMEZONE,
        session_asia_early_enabled=SESSION_ASIA_EARLY_ENABLED,
        session_asia_late_enabled=SESSION_ASIA_LATE_ENABLED,
        session_london_open_enabled=SESSION_LONDON_OPEN_ENABLED,
        session_pre_market_early_enabled=SESSION_PRE_MARKET_EARLY_ENABLED,
        session_pre_market_active_enabled=SESSION_PRE_MARKET_ACTIVE_ENABLED,
        session_pre_market_macro_enabled=SESSION_PRE_MARKET_MACRO_ENABLED,
        session_ny_open_impulse_enabled=SESSION_NY_OPEN_IMPULSE_ENABLED,
        session_ny_morning_enabled=SESSION_NY_MORNING_ENABLED,
        session_ny_midday_enabled=SESSION_NY_MIDDAY_ENABLED,
        session_ny_late_enabled=SESSION_NY_LATE_ENABLED,
        session_ny_power_hour_enabled=SESSION_NY_POWER_HOUR_ENABLED,
        session_after_close_shock_enabled=SESSION_AFTER_CLOSE_SHOCK_ENABLED,
        session_after_hours_late_enabled=SESSION_AFTER_HOURS_LATE_ENABLED,
        monte_carlo_enabled=MONTE_CARLO_ENABLED,
        monte_carlo_simulations=MONTE_CARLO_SIMULATIONS,
        mc_extra_slippage_points=MC_EXTRA_SLIPPAGE_POINTS,
        mc_block_size=MC_BLOCK_SIZE,
        mc_ruin_drawdown_pct=MC_RUIN_DRAWDOWN_PCT,
        mc_random_seed=MC_RANDOM_SEED,
    )


def active_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        single_parameter_path=SINGLE_PARAMETER_PATH,
        parameter_grid_path=PARAMETER_GRID_PATH,
        train_days=WF_TRAIN_DAYS,
        test_days=WF_TEST_DAYS,
        step_days=WF_STEP_DAYS,
        finalist_enabled=WF_FINALIST_ENABLED,
        finalist_top_n_per_session=WF_FINALIST_TOP_N_PER_SESSION,
        finalist_global_top_n=WF_FINALIST_GLOBAL_TOP_N,
        finalist_max_sets=WF_FINALIST_MAX_SETS,
        finalist_keep_trades=WF_FINALIST_KEEP_TRADES,
        finalist_persist_fold_results=WF_FINALIST_PERSIST_FOLD_RESULTS,
        processes=OPTIMIZER_PROCESSES,
        process_chunk_size=OPTIMIZER_PROCESS_CHUNK_SIZE,
        profile_cache_size=OPTIMIZER_PROFILE_CACHE_SIZE,
        progress_log_every=OPTIMIZER_PROGRESS_LOG_EVERY,
        progress_log_seconds=OPTIMIZER_PROGRESS_LOG_SECONDS,
        sampling_seed=OPTIMIZER_SAMPLING_SEED,
        stage1_max_parameter_sets=STAGE1_MAX_PARAMETER_SETS,
        stage2_enabled=STAGE2_ENABLED,
        stage2_seed_top_n=STAGE2_SEED_TOP_N,
        stage2_max_parameter_sets=STAGE2_MAX_PARAMETER_SETS,
        mc_score_top_n=MC_SCORE_TOP_N,
        persist_top_trades_n=PERSIST_TOP_TRADES_N,
        min_oos_trades=MIN_OOS_TRADES,
        min_oos_profit_factor=MIN_OOS_PROFIT_FACTOR,
        max_oos_drawdown_pct=MAX_OOS_DRAWDOWN_PCT,
        max_mc_ruin_pct=MAX_MC_RUIN_PCT,
        session_selector_min_trades_floor=SESSION_SELECTOR_MIN_TRADES_FLOOR,
        session_selector_min_trades_per_train_day=SESSION_SELECTOR_MIN_TRADES_PER_TRAIN_DAY,
        session_selector_lcb_z=SESSION_SELECTOR_LCB_Z,
        session_selector_top_n=SESSION_SELECTOR_TOP_N,
        session_selector_plateau_weight=SESSION_SELECTOR_PLATEAU_WEIGHT,
        session_selector_neighbor_distance=SESSION_SELECTOR_NEIGHBOR_DISTANCE,
        session_selector_previous_keep_score_tolerance=SESSION_SELECTOR_PREVIOUS_KEEP_SCORE_TOLERANCE,
    )


def active_optimizer_configs() -> list[OptimizerConfig]:
    base = active_optimizer_config()
    if not WF_MATRIX_ENABLED:
        return [base]
    return build_optimizer_config_matrix(base, WF_MATRIX_TRAIN_DAYS, WF_MATRIX_TEST_DAYS)


def build_optimizer_config_matrix(
    base: OptimizerConfig,
    train_days_values: tuple[int, ...],
    test_days_values: tuple[int, ...],
) -> list[OptimizerConfig]:
    configs: list[OptimizerConfig] = []
    for train_days in train_days_values:
        for test_days in test_days_values:
            configs.append(replace(base, train_days=train_days, test_days=test_days, step_days=test_days))
    return configs


def effective_session_selector_min_trades(opt: OptimizerConfig) -> int:
    floor = max(1, int(opt.session_selector_min_trades_floor))
    scaled = math.ceil(
        max(0.0, float(opt.session_selector_min_trades_per_train_day)) * max(1, int(opt.train_days))
    )
    return max(floor, scaled)


def apply_parameter_values(base: RunConfig, values: dict[str, float | int]) -> RunConfig:
    fields = {
        "lookback_bars": int(values["LOOKBACK_BARS"]),
        "long_cross_quantile": float(values["LONG_CROSS_QUANTILE"]),
        "short_cross_quantile": float(values["SHORT_CROSS_QUANTILE"]),
        "entry_price_range_position_max_deviation_pct": float(values["ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT"]),
        "take_profit_bps": float(values["ALL_STOP_MODES_TAKE_PROFIT_BPS"]),
        "min_profile_range_bps": float(values["BAND_STOP_MIN_PROFILE_RANGE_BPS"]),
        "stop_profile_lower_quantile": float(values["BAND_STOP_PROFILE_LOWER_QUANTILE"]),
        "stop_profile_upper_quantile": float(values["BAND_STOP_PROFILE_UPPER_QUANTILE"]),
        "stop_profile_buffer_points": float(values["BAND_STOP_PROFILE_BUFFER_POINTS"]),
        "min_stop_distance_bps": float(values["BAND_STOP_MIN_DISTANCE_BPS"]),
        "max_stop_distance_bps": float(values["BAND_STOP_MAX_DISTANCE_BPS"]),
    }
    fields["min_lookback_bars"] = min(base.min_lookback_bars, fields["lookback_bars"])
    if fields["lookback_bars"] < 1:
        raise ValueError("LOOKBACK_BARS must be positive")
    if not 0.0 <= fields["long_cross_quantile"] <= 1.0:
        raise ValueError("LONG_CROSS_QUANTILE must be between 0.0 and 1.0")
    if not 0.0 <= fields["short_cross_quantile"] <= 1.0:
        raise ValueError("SHORT_CROSS_QUANTILE must be between 0.0 and 1.0")
    if fields["entry_price_range_position_max_deviation_pct"] < 0:
        raise ValueError("ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT must be >= 0")
    if fields["take_profit_bps"] <= 0:
        raise ValueError("ALL_STOP_MODES_TAKE_PROFIT_BPS must be positive")
    if fields["min_profile_range_bps"] < 0:
        raise ValueError("BAND_STOP_MIN_PROFILE_RANGE_BPS must be >= 0")
    if not 0.0 <= fields["stop_profile_lower_quantile"] < fields["stop_profile_upper_quantile"] <= 1.0:
        raise ValueError("BAND_STOP_PROFILE quantiles must satisfy 0 <= lower < upper <= 1")
    if fields["stop_profile_buffer_points"] < 0:
        raise ValueError("BAND_STOP_PROFILE_BUFFER_POINTS must be >= 0")
    if fields["min_stop_distance_bps"] <= 0:
        raise ValueError("BAND_STOP_MIN_DISTANCE_BPS must be positive")
    if fields["max_stop_distance_bps"] <= fields["min_stop_distance_bps"]:
        raise ValueError("BAND_STOP_MAX_DISTANCE_BPS must be greater than BAND_STOP_MIN_DISTANCE_BPS")
    return replace(base, **fields)


SESSION_CONFIG_FIELDS = (
    ("session_asia_early_enabled", "asia_early_2000_0000"),
    ("session_asia_late_enabled", "asia_late_0000_0300"),
    ("session_london_open_enabled", "london_open_0300_0400"),
    ("session_pre_market_early_enabled", "pre_market_early_0400_0700"),
    ("session_pre_market_active_enabled", "pre_market_active_0700_0830"),
    ("session_pre_market_macro_enabled", "pre_market_macro_0830_0930"),
    ("session_ny_open_impulse_enabled", "ny_open_impulse_0930_1000"),
    ("session_ny_morning_enabled", "ny_morning_1000_1130"),
    ("session_ny_midday_enabled", "ny_midday_1130_1400"),
    ("session_ny_late_enabled", "ny_late_1400_1500"),
    ("session_ny_power_hour_enabled", "ny_power_hour_1500_1600"),
    ("session_after_close_shock_enabled", "after_close_shock_1600_1700"),
    ("session_after_hours_late_enabled", "after_hours_late_1700_2000"),
)


SESSION_ENABLE_FIELDS = (
    ("session_asia_early_enabled", "asia_early"),
    ("session_asia_late_enabled", "asia_late"),
    ("session_london_open_enabled", "london_open"),
    ("session_pre_market_early_enabled", "pre_market_early"),
    ("session_pre_market_active_enabled", "pre_market_active"),
    ("session_pre_market_macro_enabled", "pre_market_macro"),
    ("session_ny_open_impulse_enabled", "ny_open_impulse"),
    ("session_ny_morning_enabled", "ny_morning"),
    ("session_ny_midday_enabled", "ny_midday"),
    ("session_ny_late_enabled", "ny_late"),
    ("session_ny_power_hour_enabled", "ny_power_hour"),
    ("session_after_close_shock_enabled", "after_close_shock"),
    ("session_after_hours_late_enabled", "after_hours_late"),
)


def enabled_session_keys(cfg: RunConfig) -> list[str]:
    return [session_type for field, session_type in SESSION_ENABLE_FIELDS if getattr(cfg, field)]


def enabled_session_labels(cfg: RunConfig) -> list[str]:
    return [label for field, label in SESSION_CONFIG_FIELDS if getattr(cfg, field)]


def session_filter_summary(cfg: RunConfig) -> str:
    labels = enabled_session_labels(cfg)
    if len(labels) == len(SESSION_CONFIG_FIELDS):
        return f"all {cfg.session_timezone}"
    if not labels:
        return f"none {cfg.session_timezone}"
    return f"{','.join(labels)} {cfg.session_timezone}"
