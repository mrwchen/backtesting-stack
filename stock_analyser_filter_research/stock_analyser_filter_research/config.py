from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from math import isfinite
import os
import re


_QUALIFIED_IDENTIFIER = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*$"
)
_TARGET_PREFIX = "stock_analyser_filter_research_"


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_date(name: str, default: str, *, optional: bool = False) -> date | None:
    raw = os.getenv(name, default).strip()
    if optional and not raw:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD") from exc


def _table_name(name: str, value: str) -> str:
    if not _QUALIFIED_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be an unquoted PostgreSQL table name")
    return value


@dataclass(frozen=True)
class Config:
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str
    pg_app_name: str
    db_connect_timeout_seconds: int
    db_statement_timeout_ms: int
    source_table: str
    signal_result_table: str
    rule_result_table: str
    log_level: str

    signal_start_date: date
    signal_end_date: date | None
    discovery_end_date: date
    validation_end_date: date

    weak_5d_max_gain_pct: float
    strong_5d_min_gain_pct: float
    deep_loss_5d_max_loss_pct: float

    quantile_count: int
    max_rule_conditions: int
    min_candidate_match_pct: float
    max_candidate_match_pct: float
    min_strong_retention_pct: float
    min_matched_label_coverage_pct: float
    min_bad_lift: float
    min_selection_score_improvement: float

    max_workers: int
    worker_identity_batch_size: int
    db_fetch_batch_size: int
    db_copy_batch_size: int

    @classmethod
    def from_env(cls) -> "Config":
        cpu_default = min(4, max(1, (os.cpu_count() or 2) - 1))
        cfg = cls(
            pg_host=os.getenv("PGHOST", "timescaledb"),
            pg_port=_env_int("PGPORT", 5432),
            pg_database=os.getenv("PGDATABASE", "postgres"),
            pg_user=os.getenv("PGUSER", "market-data-account"),
            pg_password=os.getenv("PGPASSWORD", "market-data-account-pw"),
            pg_app_name=os.getenv(
                "PGAPPNAME", "stock_analyser_filter_research"
            ),
            db_connect_timeout_seconds=_env_int(
                "DB_CONNECT_TIMEOUT_SECONDS", 15
            ),
            db_statement_timeout_ms=_env_int("DB_STATEMENT_TIMEOUT_MS", 0),
            source_table=_table_name(
                "SOURCE_TABLE",
                os.getenv(
                    "SOURCE_TABLE", "stock_analyser_trend_template_daily"
                ),
            ),
            signal_result_table=_table_name(
                "SIGNAL_RESULT_TABLE",
                os.getenv(
                    "SIGNAL_RESULT_TABLE",
                    "stock_analyser_filter_research_signal_results",
                ),
            ),
            rule_result_table=_table_name(
                "RULE_RESULT_TABLE",
                os.getenv(
                    "RULE_RESULT_TABLE",
                    "stock_analyser_filter_research_rule_results",
                ),
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            signal_start_date=_env_date(
                "SIGNAL_START_DATE", "2016-01-01"
            ),
            signal_end_date=_env_date(
                "SIGNAL_END_DATE", "", optional=True
            ),
            discovery_end_date=_env_date(
                "DISCOVERY_END_DATE", "2022-12-31"
            ),
            validation_end_date=_env_date(
                "VALIDATION_END_DATE", "2024-12-31"
            ),
            weak_5d_max_gain_pct=_env_float(
                "WEAK_5D_MAX_GAIN_PCT", 2.0
            ),
            strong_5d_min_gain_pct=_env_float(
                "STRONG_5D_MIN_GAIN_PCT", 5.0
            ),
            deep_loss_5d_max_loss_pct=_env_float(
                "DEEP_LOSS_5D_MAX_LOSS_PCT", -5.0
            ),
            quantile_count=_env_int("QUANTILE_COUNT", 10),
            max_rule_conditions=_env_int("MAX_RULE_CONDITIONS", 3),
            min_candidate_match_pct=_env_float(
                "MIN_CANDIDATE_MATCH_PCT", 0.01
            ),
            max_candidate_match_pct=_env_float(
                "MAX_CANDIDATE_MATCH_PCT", 0.35
            ),
            min_strong_retention_pct=_env_float(
                "MIN_STRONG_RETENTION_PCT", 0.90
            ),
            min_matched_label_coverage_pct=_env_float(
                "MIN_MATCHED_LABEL_COVERAGE_PCT", 0.90
            ),
            min_bad_lift=_env_float("MIN_BAD_LIFT", 1.05),
            min_selection_score_improvement=_env_float(
                "MIN_SELECTION_SCORE_IMPROVEMENT", 0.01
            ),
            max_workers=_env_int("MAX_WORKERS", cpu_default),
            worker_identity_batch_size=_env_int(
                "WORKER_IDENTITY_BATCH_SIZE", 16
            ),
            db_fetch_batch_size=_env_int("DB_FETCH_BATCH_SIZE", 10_000),
            db_copy_batch_size=_env_int("DB_COPY_BATCH_SIZE", 5_000),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.pg_app_name.strip():
            raise ValueError("PGAPPNAME must not be empty")
        if self.db_connect_timeout_seconds < 1:
            raise ValueError("DB_CONNECT_TIMEOUT_SECONDS must be >= 1")
        if self.db_statement_timeout_ms < 0:
            raise ValueError("DB_STATEMENT_TIMEOUT_MS must be >= 0")
        if self.signal_start_date is None:
            raise ValueError("SIGNAL_START_DATE is required")
        if self.discovery_end_date is None or self.validation_end_date is None:
            raise ValueError("split end dates are required")
        if not (
            self.signal_start_date
            <= self.discovery_end_date
            < self.validation_end_date
        ):
            raise ValueError(
                "dates must satisfy SIGNAL_START_DATE <= DISCOVERY_END_DATE "
                "< VALIDATION_END_DATE"
            )
        if (
            self.signal_end_date is not None
            and self.signal_end_date <= self.validation_end_date
        ):
            raise ValueError(
                "SIGNAL_END_DATE must be empty or after VALIDATION_END_DATE"
            )
        float_values = {
            "WEAK_5D_MAX_GAIN_PCT": self.weak_5d_max_gain_pct,
            "STRONG_5D_MIN_GAIN_PCT": self.strong_5d_min_gain_pct,
            "DEEP_LOSS_5D_MAX_LOSS_PCT": self.deep_loss_5d_max_loss_pct,
            "MIN_CANDIDATE_MATCH_PCT": self.min_candidate_match_pct,
            "MAX_CANDIDATE_MATCH_PCT": self.max_candidate_match_pct,
            "MIN_STRONG_RETENTION_PCT": self.min_strong_retention_pct,
            "MIN_MATCHED_LABEL_COVERAGE_PCT": (
                self.min_matched_label_coverage_pct
            ),
            "MIN_BAD_LIFT": self.min_bad_lift,
            "MIN_SELECTION_SCORE_IMPROVEMENT": (
                self.min_selection_score_improvement
            ),
        }
        for name, value in float_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.weak_5d_max_gain_pct < 0:
            raise ValueError("WEAK_5D_MAX_GAIN_PCT must be >= 0")
        if self.strong_5d_min_gain_pct <= self.weak_5d_max_gain_pct:
            raise ValueError(
                "STRONG_5D_MIN_GAIN_PCT must exceed WEAK_5D_MAX_GAIN_PCT"
            )
        if self.deep_loss_5d_max_loss_pct >= 0:
            raise ValueError("DEEP_LOSS_5D_MAX_LOSS_PCT must be negative")
        if not 4 <= self.quantile_count <= 20:
            raise ValueError("QUANTILE_COUNT must be between 4 and 20")
        if not 1 <= self.max_rule_conditions <= 3:
            raise ValueError("MAX_RULE_CONDITIONS must be between 1 and 3")
        if not (
            0
            < self.min_candidate_match_pct
            < self.max_candidate_match_pct
            < 1
        ):
            raise ValueError(
                "candidate match percentages must satisfy 0 < MIN < MAX < 1"
            )
        if not 0 < self.min_strong_retention_pct <= 1:
            raise ValueError("MIN_STRONG_RETENTION_PCT must be in (0, 1]")
        if not 0 < self.min_matched_label_coverage_pct <= 1:
            raise ValueError(
                "MIN_MATCHED_LABEL_COVERAGE_PCT must be in (0, 1]"
            )
        if self.min_bad_lift < 1:
            raise ValueError("MIN_BAD_LIFT must be >= 1")
        if self.min_selection_score_improvement < 0:
            raise ValueError(
                "MIN_SELECTION_SCORE_IMPROVEMENT must be non-negative"
            )
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS must be >= 1")
        if self.worker_identity_batch_size < 1:
            raise ValueError("WORKER_IDENTITY_BATCH_SIZE must be >= 1")
        if self.db_fetch_batch_size < 1 or self.db_copy_batch_size < 1:
            raise ValueError("database batch sizes must be >= 1")
        for name, table in (
            ("SIGNAL_RESULT_TABLE", self.signal_result_table),
            ("RULE_RESULT_TABLE", self.rule_result_table),
        ):
            if not table.split(".")[-1].startswith(_TARGET_PREFIX):
                raise ValueError(f"{name} must start with {_TARGET_PREFIX}")
        if self.signal_result_table == self.rule_result_table:
            raise ValueError("result table names must be distinct")
