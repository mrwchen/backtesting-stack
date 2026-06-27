"""Environment-backed configuration for the NAS100 range analysis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
import re
from typing import Optional


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    text = raw.strip()
    return text if text else default


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def env_optional_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    return int(raw)


def env_optional_ts_utc(name: str) -> Optional[datetime]:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    parsed = datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must include an explicit UTC offset or Z suffix")
    return parsed.astimezone(timezone.utc)


def _validate_identifier(value: str, name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{name} contains invalid identifier {value!r}")


def _validate_identifier_path(value: str, name: str) -> None:
    parts = value.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"{name} must be table or schema.table, got {value!r}")
    for part in parts:
        _validate_identifier(part, name)


@dataclass(frozen=True)
class AnalysisConfig:
    source_table: str
    symbol: str
    start_ts_utc: Optional[datetime]
    end_ts_utc: Optional[datetime]
    bar_seconds: int
    min_lookback_bars: int
    lookback_start: int
    lookback_end: int
    lookback_step: int
    profile_max_lookback_seconds: Optional[int]
    price_step: float
    result_schema: str
    result_table: str
    weekly_result_table: str
    copy_batch_rows: int
    analysis_processes: int

    @property
    def lookback_values(self) -> tuple[int, ...]:
        return tuple(range(self.lookback_start, self.lookback_end + 1, self.lookback_step))


SOURCE_TABLE = env_str("SOURCE_TABLE", "public.pepperstone_ticks_data")
_validate_identifier_path(SOURCE_TABLE, "SOURCE_TABLE")
SYMBOL = env_str("SYMBOL", "NAS100")
START_TS_UTC = env_optional_ts_utc("START_TS_UTC")
END_TS_UTC = env_optional_ts_utc("END_TS_UTC")
if START_TS_UTC is not None and END_TS_UTC is not None and END_TS_UTC <= START_TS_UTC:
    raise ValueError("END_TS_UTC must be greater than START_TS_UTC")

BAR_SECONDS = max(1, env_int("BAR_SECONDS", 5))
MIN_LOOKBACK_BARS = max(1, env_int("MIN_LOOKBACK_BARS", 30))
LOOKBACK_START = max(1, env_int("LOOKBACK_START", 30))
LOOKBACK_END = max(LOOKBACK_START, env_int("LOOKBACK_END", 260))
LOOKBACK_STEP = max(1, env_int("LOOKBACK_STEP", 10))
if not tuple(range(LOOKBACK_START, LOOKBACK_END + 1, LOOKBACK_STEP)):
    raise ValueError("Lookback range must produce at least one value")

PROFILE_MAX_LOOKBACK_SECONDS = env_optional_int("PROFILE_MAX_LOOKBACK_SECONDS")
if PROFILE_MAX_LOOKBACK_SECONDS is not None and PROFILE_MAX_LOOKBACK_SECONDS <= 0:
    raise ValueError("PROFILE_MAX_LOOKBACK_SECONDS must be positive when set")
PRICE_STEP = env_float("PRICE_STEP", 1.0)
if PRICE_STEP <= 0:
    raise ValueError("PRICE_STEP must be positive")

RESULT_SCHEMA = env_str("RESULT_SCHEMA", "public")
_validate_identifier(RESULT_SCHEMA, "RESULT_SCHEMA")
RESULT_TABLE = env_str("RESULT_TABLE", "backtest2_nas100_hfmed_range_analysis")
_validate_identifier(RESULT_TABLE, "RESULT_TABLE")
WEEKLY_RESULT_TABLE = env_str(
    "WEEKLY_RESULT_TABLE",
    "backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana",
)
_validate_identifier(WEEKLY_RESULT_TABLE, "WEEKLY_RESULT_TABLE")

COPY_BATCH_ROWS = max(1, env_int("COPY_BATCH_ROWS", 20_000))
ANALYSIS_PROCESSES = max(1, env_int("ANALYSIS_PROCESSES", 4))

DB_CONNECT_RETRIES = max(1, env_int("DB_CONNECT_RETRIES", 5))
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
    "application_name": os.getenv("PGAPPNAME", "nas100_hfmed_range_analysis"),
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}


def active_analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        source_table=SOURCE_TABLE,
        symbol=SYMBOL,
        start_ts_utc=START_TS_UTC,
        end_ts_utc=END_TS_UTC,
        bar_seconds=BAR_SECONDS,
        min_lookback_bars=MIN_LOOKBACK_BARS,
        lookback_start=LOOKBACK_START,
        lookback_end=LOOKBACK_END,
        lookback_step=LOOKBACK_STEP,
        profile_max_lookback_seconds=PROFILE_MAX_LOOKBACK_SECONDS,
        price_step=PRICE_STEP,
        result_schema=RESULT_SCHEMA,
        result_table=RESULT_TABLE,
        weekly_result_table=WEEKLY_RESULT_TABLE,
        copy_batch_rows=COPY_BATCH_ROWS,
        analysis_processes=ANALYSIS_PROCESSES,
    )
