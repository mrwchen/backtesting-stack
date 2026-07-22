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
    if value != value.lower():
        raise ValueError(f"{name} must use lowercase PostgreSQL identifiers")
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
    fundamental_snapshot_table: str
    quarterly_fundamental_event_table: str
    earnings_event_table: str
    market_metrics_table: str
    world_market_observation_table: str
    security_master_current_table: str
    signal_result_table: str
    early_cut_result_table: str
    rule_result_table: str
    log_level: str

    signal_start_date: date
    signal_end_date: date | None
    discovery_end_date: date
    validation_end_date: date
    holdout_cutoff_date: date

    weak_5d_max_gain_pct: float
    strong_5d_min_gain_pct: float
    deep_loss_5d_max_loss_pct: float
    hard_stop_5d_max_loss_pct: float
    stagnant_5d_max_return_pct: float
    continuation_winner_min_return_pct: float
    terminal_stagnant_5d_max_return_pct: float
    terminal_winner_5d_min_return_pct: float

    quantile_count: int
    max_conditions_per_objective: int
    rule_search_beam_width: int
    walk_forward_first_year: int
    min_walk_forward_folds: int
    min_stable_fold_fraction: float
    min_fold_objective_lift: float
    min_fold_sample_count: int
    min_fold_objective_count: int
    min_fold_protected_retention_pct: float
    max_fold_match_pct: float
    min_candidate_match_pct: float
    max_candidate_match_pct: float
    min_protected_retention_pct: float
    min_matched_label_coverage_pct: float
    min_objective_lift: float
    min_selection_score_improvement: float
    permutation_trial_count: int
    max_stat_permutation_p_value: float
    permutation_random_seed: int
    min_holdout_sample_count: int
    taxonomy_backcast_industry_min_members: int
    taxonomy_backcast_category_min_members: int
    taxonomy_backcast_subcategory_min_members: int

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
            pg_app_name=os.getenv("PGAPPNAME", "stock_analyser_filter_research"),
            db_connect_timeout_seconds=_env_int("DB_CONNECT_TIMEOUT_SECONDS", 15),
            db_statement_timeout_ms=_env_int("DB_STATEMENT_TIMEOUT_MS", 0),
            source_table=_table_name(
                "SOURCE_TABLE",
                os.getenv("SOURCE_TABLE", "stock_analyser_trend_template_daily"),
            ),
            fundamental_snapshot_table=_table_name(
                "FUNDAMENTAL_SNAPSHOT_TABLE",
                os.getenv(
                    "FUNDAMENTAL_SNAPSHOT_TABLE",
                    "stock_core_sec_fundamentals_asof_daily",
                ),
            ),
            quarterly_fundamental_event_table=_table_name(
                "QUARTERLY_FUNDAMENTAL_EVENT_TABLE",
                os.getenv(
                    "QUARTERLY_FUNDAMENTAL_EVENT_TABLE",
                    "stock_core_sec_quarterly_fundamental_events",
                ),
            ),
            earnings_event_table=_table_name(
                "EARNINGS_EVENT_TABLE",
                os.getenv(
                    "EARNINGS_EVENT_TABLE",
                    "stock_core_earnings_calendar_events",
                ),
            ),
            market_metrics_table=_table_name(
                "MARKET_METRICS_TABLE",
                os.getenv(
                    "MARKET_METRICS_TABLE",
                    "stock_core_market_metrics_daily",
                ),
            ),
            world_market_observation_table=_table_name(
                "WORLD_MARKET_OBSERVATION_TABLE",
                os.getenv(
                    "WORLD_MARKET_OBSERVATION_TABLE",
                    "world_regime_observations",
                ),
            ),
            security_master_current_table=_table_name(
                "SECURITY_MASTER_CURRENT_TABLE",
                os.getenv(
                    "SECURITY_MASTER_CURRENT_TABLE",
                    "stock_core_security_master_current",
                ),
            ),
            signal_result_table=_table_name(
                "SIGNAL_RESULT_TABLE",
                os.getenv(
                    "SIGNAL_RESULT_TABLE",
                    "stock_analyser_filter_research_signal_results",
                ),
            ),
            early_cut_result_table=_table_name(
                "EARLY_CUT_RESULT_TABLE",
                os.getenv(
                    "EARLY_CUT_RESULT_TABLE",
                    "stock_analyser_filter_research_early_cut_results",
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
            signal_start_date=_env_date("SIGNAL_START_DATE", "2016-01-01"),
            signal_end_date=_env_date("SIGNAL_END_DATE", "", optional=True),
            discovery_end_date=_env_date("DISCOVERY_END_DATE", "2022-12-31"),
            validation_end_date=_env_date("VALIDATION_END_DATE", "2024-12-31"),
            holdout_cutoff_date=_env_date("HOLDOUT_CUTOFF_DATE", "2026-07-20"),
            weak_5d_max_gain_pct=_env_float("WEAK_5D_MAX_GAIN_PCT", 2.0),
            strong_5d_min_gain_pct=_env_float("STRONG_5D_MIN_GAIN_PCT", 5.0),
            deep_loss_5d_max_loss_pct=_env_float("DEEP_LOSS_5D_MAX_LOSS_PCT", -5.0),
            hard_stop_5d_max_loss_pct=_env_float(
                "HARD_STOP_5D_MAX_LOSS_PCT", -10.0
            ),
            stagnant_5d_max_return_pct=_env_float(
                "STAGNANT_5D_MAX_RETURN_PCT", 0.0
            ),
            continuation_winner_min_return_pct=_env_float(
                "CONTINUATION_WINNER_MIN_RETURN_PCT", 5.0
            ),
            terminal_stagnant_5d_max_return_pct=_env_float(
                "TERMINAL_STAGNANT_5D_MAX_RETURN_PCT", 1.0
            ),
            terminal_winner_5d_min_return_pct=_env_float(
                "TERMINAL_WINNER_5D_MIN_RETURN_PCT", 3.0
            ),
            quantile_count=_env_int("QUANTILE_COUNT", 10),
            max_conditions_per_objective=_env_int("MAX_CONDITIONS_PER_OBJECTIVE", 3),
            rule_search_beam_width=_env_int("RULE_SEARCH_BEAM_WIDTH", 30),
            walk_forward_first_year=_env_int("WALK_FORWARD_FIRST_YEAR", 2020),
            min_walk_forward_folds=_env_int("MIN_WALK_FORWARD_FOLDS", 4),
            min_stable_fold_fraction=_env_float("MIN_STABLE_FOLD_FRACTION", 0.75),
            min_fold_objective_lift=_env_float("MIN_FOLD_OBJECTIVE_LIFT", 1.0),
            min_fold_sample_count=_env_int("MIN_FOLD_SAMPLE_COUNT", 200),
            min_fold_objective_count=_env_int("MIN_FOLD_OBJECTIVE_COUNT", 20),
            min_fold_protected_retention_pct=_env_float(
                "MIN_FOLD_PROTECTED_RETENTION_PCT", 0.90
            ),
            max_fold_match_pct=_env_float("MAX_FOLD_MATCH_PCT", 0.40),
            min_candidate_match_pct=_env_float("MIN_CANDIDATE_MATCH_PCT", 0.01),
            max_candidate_match_pct=_env_float("MAX_CANDIDATE_MATCH_PCT", 0.35),
            min_protected_retention_pct=_env_float("MIN_PROTECTED_RETENTION_PCT", 0.92),
            min_matched_label_coverage_pct=_env_float(
                "MIN_MATCHED_LABEL_COVERAGE_PCT", 0.90
            ),
            min_objective_lift=_env_float("MIN_OBJECTIVE_LIFT", 1.05),
            min_selection_score_improvement=_env_float(
                "MIN_SELECTION_SCORE_IMPROVEMENT", 0.01
            ),
            permutation_trial_count=_env_int("PERMUTATION_TRIAL_COUNT", 999),
            max_stat_permutation_p_value=_env_float(
                "MAX_STAT_PERMUTATION_P_VALUE", 0.05
            ),
            permutation_random_seed=_env_int("PERMUTATION_RANDOM_SEED", 1729),
            min_holdout_sample_count=_env_int("MIN_HOLDOUT_SAMPLE_COUNT", 500),
            taxonomy_backcast_industry_min_members=_env_int(
                "TAXONOMY_BACKCAST_INDUSTRY_MIN_MEMBERS", 20
            ),
            taxonomy_backcast_category_min_members=_env_int(
                "TAXONOMY_BACKCAST_CATEGORY_MIN_MEMBERS", 10
            ),
            taxonomy_backcast_subcategory_min_members=_env_int(
                "TAXONOMY_BACKCAST_SUBCATEGORY_MIN_MEMBERS", 5
            ),
            max_workers=_env_int("MAX_WORKERS", cpu_default),
            worker_identity_batch_size=_env_int("WORKER_IDENTITY_BATCH_SIZE", 16),
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
        if not (
            self.signal_start_date
            <= self.discovery_end_date
            < self.validation_end_date
            < self.holdout_cutoff_date
        ):
            raise ValueError(
                "dates must satisfy SIGNAL_START_DATE <= DISCOVERY_END_DATE "
                "< VALIDATION_END_DATE < HOLDOUT_CUTOFF_DATE"
            )
        if (
            self.signal_end_date is not None
            and self.signal_end_date < self.signal_start_date
        ):
            raise ValueError(
                "SIGNAL_END_DATE must be empty or not precede SIGNAL_START_DATE"
            )

        float_values = {
            "WEAK_5D_MAX_GAIN_PCT": self.weak_5d_max_gain_pct,
            "STRONG_5D_MIN_GAIN_PCT": self.strong_5d_min_gain_pct,
            "DEEP_LOSS_5D_MAX_LOSS_PCT": self.deep_loss_5d_max_loss_pct,
            "HARD_STOP_5D_MAX_LOSS_PCT": self.hard_stop_5d_max_loss_pct,
            "STAGNANT_5D_MAX_RETURN_PCT": self.stagnant_5d_max_return_pct,
            "CONTINUATION_WINNER_MIN_RETURN_PCT": (
                self.continuation_winner_min_return_pct
            ),
            "TERMINAL_STAGNANT_5D_MAX_RETURN_PCT": (
                self.terminal_stagnant_5d_max_return_pct
            ),
            "TERMINAL_WINNER_5D_MIN_RETURN_PCT": (
                self.terminal_winner_5d_min_return_pct
            ),
            "MIN_STABLE_FOLD_FRACTION": self.min_stable_fold_fraction,
            "MIN_FOLD_OBJECTIVE_LIFT": self.min_fold_objective_lift,
            "MIN_FOLD_PROTECTED_RETENTION_PCT": (self.min_fold_protected_retention_pct),
            "MAX_FOLD_MATCH_PCT": self.max_fold_match_pct,
            "MIN_CANDIDATE_MATCH_PCT": self.min_candidate_match_pct,
            "MAX_CANDIDATE_MATCH_PCT": self.max_candidate_match_pct,
            "MIN_PROTECTED_RETENTION_PCT": (self.min_protected_retention_pct),
            "MIN_MATCHED_LABEL_COVERAGE_PCT": (self.min_matched_label_coverage_pct),
            "MIN_OBJECTIVE_LIFT": self.min_objective_lift,
            "MIN_SELECTION_SCORE_IMPROVEMENT": (
                self.min_selection_score_improvement
            ),
            "MAX_STAT_PERMUTATION_P_VALUE": self.max_stat_permutation_p_value,
        }
        for name, value in float_values.items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.weak_5d_max_gain_pct <= 0:
            raise ValueError("WEAK_5D_MAX_GAIN_PCT must be > 0")
        if self.strong_5d_min_gain_pct <= self.weak_5d_max_gain_pct:
            raise ValueError("STRONG_5D_MIN_GAIN_PCT must exceed WEAK_5D_MAX_GAIN_PCT")
        if self.deep_loss_5d_max_loss_pct >= 0:
            raise ValueError("DEEP_LOSS_5D_MAX_LOSS_PCT must be negative")
        if self.hard_stop_5d_max_loss_pct >= self.deep_loss_5d_max_loss_pct:
            raise ValueError(
                "HARD_STOP_5D_MAX_LOSS_PCT must be more negative than "
                "DEEP_LOSS_5D_MAX_LOSS_PCT"
            )
        if self.continuation_winner_min_return_pct <= 0:
            raise ValueError("CONTINUATION_WINNER_MIN_RETURN_PCT must be > 0")
        if (
            self.terminal_winner_5d_min_return_pct
            <= self.terminal_stagnant_5d_max_return_pct
        ):
            raise ValueError(
                "TERMINAL_WINNER_5D_MIN_RETURN_PCT must exceed "
                "TERMINAL_STAGNANT_5D_MAX_RETURN_PCT"
            )
        if not 4 <= self.quantile_count <= 20:
            raise ValueError("QUANTILE_COUNT must be between 4 and 20")
        if not 1 <= self.max_conditions_per_objective <= 3:
            raise ValueError("MAX_CONDITIONS_PER_OBJECTIVE must be between 1 and 3")
        if self.rule_search_beam_width < 1:
            raise ValueError("RULE_SEARCH_BEAM_WIDTH must be >= 1")
        if not (
            self.signal_start_date.year
            < self.walk_forward_first_year
            <= self.validation_end_date.year
        ):
            raise ValueError(
                "WALK_FORWARD_FIRST_YEAR must be after SIGNAL_START_DATE year "
                "and no later than VALIDATION_END_DATE year"
            )
        last_fold_year = self.validation_end_date.year
        if self.signal_end_date is not None:
            last_fold_year = min(last_fold_year, self.signal_end_date.year)
        available_folds = last_fold_year - self.walk_forward_first_year + 1
        if not 1 <= self.min_walk_forward_folds <= available_folds:
            raise ValueError(
                "MIN_WALK_FORWARD_FOLDS exceeds configured development years"
            )
        if self.min_fold_sample_count < 1 or self.min_fold_objective_count < 1:
            raise ValueError("fold sample/count minimums must be positive")
        if self.min_holdout_sample_count < 1:
            raise ValueError("MIN_HOLDOUT_SAMPLE_COUNT must be positive")
        for name, value in (
            (
                "TAXONOMY_BACKCAST_INDUSTRY_MIN_MEMBERS",
                self.taxonomy_backcast_industry_min_members,
            ),
            (
                "TAXONOMY_BACKCAST_CATEGORY_MIN_MEMBERS",
                self.taxonomy_backcast_category_min_members,
            ),
            (
                "TAXONOMY_BACKCAST_SUBCATEGORY_MIN_MEMBERS",
                self.taxonomy_backcast_subcategory_min_members,
            ),
        ):
            if value < 2:
                raise ValueError(f"{name} must be >= 2")
        for name, value in (
            ("MIN_STABLE_FOLD_FRACTION", self.min_stable_fold_fraction),
            (
                "MIN_FOLD_PROTECTED_RETENTION_PCT",
                self.min_fold_protected_retention_pct,
            ),
            ("MAX_FOLD_MATCH_PCT", self.max_fold_match_pct),
            (
                "MIN_PROTECTED_RETENTION_PCT",
                self.min_protected_retention_pct,
            ),
            (
                "MIN_MATCHED_LABEL_COVERAGE_PCT",
                self.min_matched_label_coverage_pct,
            ),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not (0 < self.min_candidate_match_pct < self.max_candidate_match_pct < 1):
            raise ValueError(
                "candidate match percentages must satisfy 0 < MIN < MAX < 1"
            )
        if self.min_objective_lift < 1 or self.min_fold_objective_lift < 1:
            raise ValueError("objective lift minimums must be >= 1")
        if not 0 <= self.min_selection_score_improvement <= 1:
            raise ValueError(
                "MIN_SELECTION_SCORE_IMPROVEMENT must be between 0 and 1"
            )
        if self.permutation_trial_count < 19:
            raise ValueError("PERMUTATION_TRIAL_COUNT must be >= 19")
        if not 0 < self.max_stat_permutation_p_value <= 1:
            raise ValueError(
                "MAX_STAT_PERMUTATION_P_VALUE must be in the interval (0, 1]"
            )
        if self.permutation_random_seed < 0:
            raise ValueError("PERMUTATION_RANDOM_SEED must be >= 0")
        if self.max_workers < 1:
            raise ValueError("MAX_WORKERS must be >= 1")
        if self.worker_identity_batch_size < 1:
            raise ValueError("WORKER_IDENTITY_BATCH_SIZE must be >= 1")
        if self.db_fetch_batch_size < 1 or self.db_copy_batch_size < 1:
            raise ValueError("database batch sizes must be >= 1")

        target_tables = (
            ("SIGNAL_RESULT_TABLE", self.signal_result_table),
            ("EARLY_CUT_RESULT_TABLE", self.early_cut_result_table),
            ("RULE_RESULT_TABLE", self.rule_result_table),
        )
        source_tables = (
            ("SOURCE_TABLE", self.source_table),
            ("FUNDAMENTAL_SNAPSHOT_TABLE", self.fundamental_snapshot_table),
            (
                "QUARTERLY_FUNDAMENTAL_EVENT_TABLE",
                self.quarterly_fundamental_event_table,
            ),
            ("EARNINGS_EVENT_TABLE", self.earnings_event_table),
            ("MARKET_METRICS_TABLE", self.market_metrics_table),
            (
                "WORLD_MARKET_OBSERVATION_TABLE",
                self.world_market_observation_table,
            ),
            (
                "SECURITY_MASTER_CURRENT_TABLE",
                self.security_master_current_table,
            ),
        )
        for name, table in source_tables:
            _table_name(name, table)
        for name, table in target_tables:
            _table_name(name, table)
            if not table.split(".")[-1].startswith(_TARGET_PREFIX):
                raise ValueError(f"{name} must start with {_TARGET_PREFIX}")
        normalized_tables = {
            (
                table.rsplit(".", 1)[0] if "." in table else "public",
                table.rsplit(".", 1)[-1],
            )
            for _, table in target_tables
        }
        if len(normalized_tables) != len(target_tables):
            raise ValueError("result table names must be distinct")
