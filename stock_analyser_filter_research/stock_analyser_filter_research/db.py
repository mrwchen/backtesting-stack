from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import date, timedelta
from numbers import Integral
from typing import Any, TypeAlias
from uuid import uuid4

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import extensions, sql

from .config import Config
from .contracts import (
    CALCULATION_SOURCE_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_INTEGER_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS,
    CURRENT_TAXONOMY_SOURCE_COLUMNS,
    EARLY_CUT_BOOLEAN_COLUMNS,
    EARLY_CUT_COLUMNS,
    EARLY_CUT_INTEGER_COLUMNS,
    EARNINGS_EVENT_SOURCE_COLUMNS,
    FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
    IDENTITY_COLUMNS,
    MARKET_METRIC_SOURCE_COLUMNS,
    QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
    RULE_BOOLEAN_COLUMNS,
    RULE_COLUMNS,
    RULE_INTEGER_COLUMNS,
    SIGNAL_BOOLEAN_COLUMNS,
    SIGNAL_COLUMNS,
    SIGNAL_INTEGER_COLUMNS,
    SOURCE_BOOLEAN_COLUMNS,
    SOURCE_COLUMNS,
    SOURCE_INTEGER_COLUMNS,
    WORLD_MARKET_SOURCE_COLUMNS,
)


# This is a session-level lock. It deliberately differs from the lock used by
# stock_analyser, because both programs are allowed to run at the same time.
ADVISORY_LOCK_KEY = 7_321_904_823
EXPECTED_SIGNAL_PRIMARY_KEY = ("signal_date", "symbol", "exchange", "cik")
EXPECTED_EARLY_CUT_PRIMARY_KEY = (
    "signal_date",
    "symbol",
    "exchange",
    "cik",
    "landmark_day",
)
EXPECTED_RULE_PRIMARY_KEY = ("result_id",)
EXPECTED_FUNDAMENTAL_SNAPSHOT_PRIMARY_KEY = (
    "symbol",
    "exchange",
    "cik",
    "period_end_date",
)
EXPECTED_QUARTERLY_FUNDAMENTAL_EVENT_PRIMARY_KEY = (
    "symbol",
    "exchange",
    "cik",
    "accession_number",
)
EXPECTED_EARNINGS_EVENT_PRIMARY_KEY = (
    "symbol",
    "exchange",
    "cik",
    "source",
    "source_event_id",
)
EXPECTED_MARKET_METRIC_PRIMARY_KEY = (
    "symbol",
    "exchange",
    "cik",
    "period_end_date",
)
EXPECTED_CHUNK_INTERVAL = timedelta(days=365)
COPY_NULL = r"\N"

StockIdentity: TypeAlias = tuple[str, str, int]
IdentityWork: TypeAlias = tuple[str, str, int, int]
StockSignalKey: TypeAlias = tuple[date, str, str, int]
ColumnContract: TypeAlias = tuple[str, bool, int | None, int | None]


def _add_column_contracts(
    contracts: dict[str, ColumnContract],
    columns: Sequence[str] | set[str],
    udt_name: str,
    *,
    nullable: bool,
    precision: int | None = None,
    scale: int | None = None,
) -> None:
    overlap = set(columns).intersection(contracts)
    if overlap:
        raise AssertionError(
            "duplicate database column contracts: " + ", ".join(sorted(overlap))
        )
    for column in columns:
        contracts[column] = (udt_name, nullable, precision, scale)


def _source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(contracts, ("period_end_date",), "date", nullable=False)
    _add_column_contracts(
        contracts, ("symbol", "exchange", "currency"), "text", nullable=False
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(
        contracts, ("price_continuity_segment",), "int4", nullable=False
    )
    _add_column_contracts(contracts, ("adjusted_volume",), "int8", nullable=True)
    _add_column_contracts(contracts, ("rs_rating",), "int2", nullable=True)
    _add_column_contracts(contracts, SOURCE_BOOLEAN_COLUMNS, "bool", nullable=False)
    _add_column_contracts(
        contracts,
        {
            "adjusted_volume_sma21_prior",
            "adjusted_volume_vs_sma21_prior_ratio",
            "adjusted_volume_sma50_prior",
            "adjusted_volume_vs_sma50_prior_ratio",
            "daily_traded_notional_sma21_prior_usd",
            "daily_traded_notional_vs_sma21_prior_ratio",
            "daily_traded_notional_sma50_prior_usd",
            "daily_traded_notional_vs_sma50_prior_ratio",
            "dollar_volume_63d",
        },
        "numeric",
        nullable=True,
        precision=30,
        scale=8,
    )
    _add_column_contracts(
        contracts,
        ("daily_traded_notional_usd",),
        "numeric",
        nullable=True,
        precision=24,
        scale=2,
    )
    _add_column_contracts(
        contracts,
        ("rs_raw",),
        "numeric",
        nullable=True,
        precision=24,
        scale=10,
    )
    remaining = set(SOURCE_COLUMNS) - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=20,
        scale=8,
    )
    if set(contracts) != set(SOURCE_COLUMNS):
        raise AssertionError("source database column contract is incomplete")
    return contracts


def _fundamental_snapshot_source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts, ("symbol", "exchange"), "text", nullable=False
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(contracts, ("period_end_date",), "date", nullable=False)
    _add_column_contracts(
        contracts,
        ("sec_fundamental_currency",),
        "text",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("sec_latest_period_end_date",),
        "date",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("sec_data_available_at",),
        "timestamptz",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        {
            "sec_revenue_ttm",
            "sec_operating_income_ttm",
            "sec_net_income_ttm",
            "sec_operating_cashflow_ttm",
            "sec_capex_ttm",
            "sec_free_cashflow_ttm",
            "sec_share_based_compensation_ttm",
            "sec_free_cashflow_sbc_adjusted_ttm",
            "sec_research_and_development_ttm",
            "sec_selling_general_and_admin_ttm",
            "sec_interest_expense_ttm",
            "sec_depreciation_and_amortization_ttm",
            "sec_assets",
            "sec_stockholders_equity",
            "sec_cash_and_equivalents",
            "sec_total_debt",
            "sec_current_assets",
            "sec_current_liabilities",
            "sec_inventory",
            "sec_accounts_receivable",
            "sec_accounts_payable",
            "sec_goodwill",
            "sec_intangible_assets",
            "sec_property_plant_equipment",
            "sec_shares_outstanding",
            "sec_weighted_avg_shares_diluted",
            "sec_common_stock_repurchased_ttm",
        },
        "int8",
        nullable=True,
    )
    remaining = set(FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS) - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=18,
        scale=6,
    )
    if set(contracts) != set(FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS):
        raise AssertionError("fundamental snapshot source contract is incomplete")
    return contracts


def _quarterly_fundamental_event_source_column_contracts(
) -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts,
        ("symbol", "exchange", "accession_number"),
        "text",
        nullable=False,
    )
    _add_column_contracts(contracts, ("currency",), "text", nullable=True)
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(
        contracts,
        ("effective_date", "fiscal_period_end_date"),
        "date",
        nullable=False,
    )
    _add_column_contracts(
        contracts, ("accepted_at",), "timestamptz", nullable=True
    )
    _add_column_contracts(
        contracts,
        ("diluted_eps", "prior_year_diluted_eps"),
        "numeric",
        nullable=True,
        precision=18,
        scale=6,
    )
    _add_column_contracts(
        contracts,
        (
            "quarterly_revenue",
            "prior_year_quarterly_revenue",
            "quarterly_operating_income",
            "prior_year_quarterly_operating_income",
            "quarterly_net_income",
            "prior_year_quarterly_net_income",
        ),
        "numeric",
        nullable=True,
        precision=28,
        scale=2,
    )
    remaining = set(QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS) - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=18,
        scale=8,
    )
    if set(contracts) != set(QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS):
        raise AssertionError("quarterly fundamental event source contract is incomplete")
    return contracts


def _earnings_event_source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts,
        (
            "symbol",
            "exchange",
            "announcement_time_type",
            "source",
            "source_event_id",
        ),
        "text",
        nullable=False,
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(contracts, ("earnings_date",), "date", nullable=False)
    _add_column_contracts(
        contracts, ("announcement_ts",), "timestamptz", nullable=True
    )
    _add_column_contracts(
        contracts, ("known_as_of_ts",), "timestamptz", nullable=False
    )
    _add_column_contracts(contracts, ("is_confirmed",), "bool", nullable=False)
    if set(contracts) != set(EARNINGS_EVENT_SOURCE_COLUMNS):
        raise AssertionError("earnings event source contract is incomplete")
    return contracts


def _market_metric_source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts, ("symbol", "exchange"), "text", nullable=False
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(contracts, ("period_end_date",), "date", nullable=False)
    _add_column_contracts(
        contracts,
        ("market_cap", "raw_volume", "shares_outstanding"),
        "int8",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("market_cap_currency", "shares_outstanding_source"),
        "text",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("adjusted_open",),
        "numeric",
        nullable=True,
        precision=15,
        scale=4,
    )
    _add_column_contracts(
        contracts,
        ("shares_outstanding_staleness_days",),
        "int4",
        nullable=True,
    )
    if set(contracts) != set(MARKET_METRIC_SOURCE_COLUMNS):
        raise AssertionError("market metric source contract is incomplete")
    return contracts


def _world_market_source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts, ("source", "series_id"), "text", nullable=False
    )
    _add_column_contracts(
        contracts,
        ("observation_time", "available_at", "asof_known_at"),
        "timestamptz",
        nullable=False,
    )
    _add_column_contracts(contracts, ("value",), "float8", nullable=False)
    _add_column_contracts(
        contracts,
        ("is_revision_prone", "is_final"),
        "bool",
        nullable=False,
    )
    _add_column_contracts(
        contracts, ("source_local_date",), "date", nullable=False
    )
    if set(contracts) != set(WORLD_MARKET_SOURCE_COLUMNS):
        raise AssertionError("world market source contract is incomplete")
    return contracts


def _current_taxonomy_source_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts, ("symbol", "exchange"), "text", nullable=False
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(
        contracts,
        ("ibkr_industry", "ibkr_category", "ibkr_subcategory"),
        "text",
        nullable=True,
    )
    if set(contracts) != set(CURRENT_TAXONOMY_SOURCE_COLUMNS):
        raise AssertionError("current taxonomy source contract is incomplete")
    return contracts


def _signal_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(
        contracts, ("signal_date", "previous_session_date"), "date", nullable=False
    )
    _add_column_contracts(
        contracts,
        tuple(
            f"forward_{day}d_label_end_date"
            for day in (5, 10, 20, 30, 40, 60, 90)
        ),
        "date",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "symbol",
            "exchange",
            "currency",
            "trigger_criteria",
            "analysis_split",
            "filter_decision",
        ),
        "text",
        nullable=False,
    )
    _add_column_contracts(
        contracts,
        (
            "gain_loss_order_5d",
            "weak_matched_rule_ids",
            "loss_first_matched_rule_ids",
            "matched_rule_ids",
            "confirmation_matched_rule_ids",
            "confirmation_reason",
            "exclusion_reason",
        ),
        "text",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
        "text",
        nullable=True,
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(
        contracts, ("price_continuity_segment",), "int4", nullable=False
    )
    _add_column_contracts(
        contracts,
        ("trigger_count", "previous_criteria_pass_count"),
        "int2",
        nullable=False,
    )
    _add_column_contracts(contracts, ("adjusted_volume",), "int8", nullable=True)
    _add_column_contracts(
        contracts,
        (
            "rs_rating",
            "prior_7_of_8_count_10d",
            "prior_peak_age_40_sessions",
            "prior_trough_age_40_sessions",
            "prior_distribution_day_count_20",
            "prior_churning_day_count_20",
            "prior_failed_breakout_count_20",
        ),
        "int2",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("sessions_since_previous_pass", "market_cap_shares_staleness_days"),
        "int4",
        nullable=True,
    )
    _add_column_contracts(
        contracts, ("market_cap_usd",), "int8", nullable=True
    )
    _add_column_contracts(
        contracts,
        CURRENT_TAXONOMY_BACKCAST_INTEGER_COLUMNS,
        "int4",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "weak_5d",
            "strong_5d",
            "deep_loss_5d",
            "bad_5d",
            "loss_first_5d",
            "strong_first_5d",
            "terminal_stagnant_5d",
            "terminal_winner_5d",
            "stagnant_5d",
            "hard_stop_10pct_5d",
            "terminal_nonpositive_20d",
            "terminal_winner_20d",
            "terminal_nonpositive_30d",
            "terminal_winner_30d",
            "runner_60d",
            "runner_90d",
            "late_strong_10d",
            "late_strong_20d",
        ),
        "bool",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "include_weak_filter",
            "include_loss_first_filter",
            "include_final",
            "strong_confirmation",
        ),
        "bool",
        nullable=False,
    )
    _add_column_contracts(
        contracts,
        (
            "first_gain_1pct_day",
            "first_gain_2pct_day",
            "first_gain_3pct_day",
            "first_gain_5pct_day",
            "first_loss_5pct_day",
            "first_loss_10pct_day",
        ),
        "int2",
        nullable=True,
    )
    _add_column_contracts(
        contracts, ("shares_outstanding",), "int8", nullable=True
    )
    _add_column_contracts(
        contracts, ("prior_history_sessions",), "int4", nullable=True
    )
    _add_column_contracts(
        contracts,
        {
            "adjusted_volume_sma21_prior",
            "adjusted_volume_vs_sma21_prior_ratio",
            "adjusted_volume_vs_sma7_prior_ratio",
            "adjusted_volume_vs_sma14_prior_ratio",
            "adjusted_volume_sma50_prior",
            "adjusted_volume_vs_sma50_prior_ratio",
            "adjusted_volume_vs_sma100_prior_ratio",
            "daily_traded_notional_sma21_prior_usd",
            "daily_traded_notional_vs_sma21_prior_ratio",
            "daily_traded_notional_vs_sma7_prior_ratio",
            "daily_traded_notional_vs_sma14_prior_ratio",
            "daily_traded_notional_sma50_prior_usd",
            "daily_traded_notional_vs_sma50_prior_ratio",
            "daily_traded_notional_vs_sma100_prior_ratio",
            "dollar_volume_63d",
        },
        "numeric",
        nullable=True,
        precision=30,
        scale=8,
    )
    _add_column_contracts(
        contracts,
        ("daily_traded_notional_usd",),
        "numeric",
        nullable=True,
        precision=24,
        scale=2,
    )
    _add_column_contracts(
        contracts,
        ("rs_raw",),
        "numeric",
        nullable=True,
        precision=24,
        scale=10,
    )
    remaining = set(SIGNAL_COLUMNS) - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=20,
        scale=8,
    )
    if set(contracts) != set(SIGNAL_COLUMNS):
        raise AssertionError("signal database column contract is incomplete")
    return contracts


def _early_cut_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(contracts, ("signal_date",), "date", nullable=False)
    _add_column_contracts(
        contracts,
        (
            "landmark_date",
            "effective_session_date",
            "horizon_end_date",
            "day20_end_date",
            "day40_end_date",
            "day60_end_date",
            "day90_end_date",
        ),
        "date",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "symbol",
            "exchange",
            "currency",
            "decision_stage",
            "analysis_split",
            "cut_decision",
            "management_decision",
        ),
        "text",
        nullable=False,
    )
    _add_column_contracts(
        contracts,
        (
            "continuation_outcome",
            "stagnation_matched_rule_ids",
            "loss_matched_rule_ids",
            "matched_rule_ids",
            "cut_reason",
            "management_matched_rule_ids",
            "management_reason",
        ),
        "text",
        nullable=True,
    )
    _add_column_contracts(contracts, ("cik",), "int8", nullable=False)
    _add_column_contracts(
        contracts, ("price_continuity_segment",), "int4", nullable=False
    )
    _add_column_contracts(contracts, ("landmark_day",), "int2", nullable=False)
    _add_column_contracts(
        contracts, ("prior_policy_cut_day",), "int2", nullable=True
    )
    _add_column_contracts(
        contracts, ("landmark_adjusted_volume",), "int8", nullable=True
    )
    _add_column_contracts(
        contracts,
        (
            "landmark_rs_rating",
            "landmark_criteria_pass_count",
            "first_gain_2pct_day_so_far",
            "first_gain_5pct_day_so_far",
            "first_loss_5pct_day_so_far",
            "first_loss_10pct_day_so_far",
            "future_first_gain_2pct_day",
            "future_first_gain_5pct_day",
            "future_first_loss_5pct_day",
        ),
        "int2",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "landmark_observed",
            "same_continuity_segment",
            "eligible_at_landmark",
            "active_at_landmark",
            "full_outcome_available",
            "include_stagnation_filter",
            "include_loss_filter",
            "include_final",
            "management_include_final",
        ),
        "bool",
        nullable=False,
    )
    nullable_boolean_columns = set(EARLY_CUT_BOOLEAN_COLUMNS) - {
        "landmark_observed",
        "same_continuity_segment",
        "eligible_at_landmark",
        "active_at_landmark",
        "full_outcome_available",
        "include_stagnation_filter",
        "include_loss_filter",
        "include_final",
        "management_include_final",
    }
    _add_column_contracts(contracts, nullable_boolean_columns, "bool", nullable=True)
    _add_column_contracts(
        contracts,
        {
            "landmark_volume_vs_sma21_prior_ratio",
            "landmark_volume_vs_sma7_prior_ratio",
            "landmark_volume_vs_sma14_prior_ratio",
            "landmark_volume_vs_sma50_prior_ratio",
            "landmark_volume_vs_sma100_prior_ratio",
            "landmark_notional_vs_sma21_prior_ratio",
            "landmark_notional_vs_sma7_prior_ratio",
            "landmark_notional_vs_sma14_prior_ratio",
            "landmark_notional_vs_sma50_prior_ratio",
            "landmark_notional_vs_sma100_prior_ratio",
            "mean_volume_since_signal_vs_prior21_ratio",
            "mean_notional_since_signal_vs_prior21_ratio",
        },
        "numeric",
        nullable=True,
        precision=30,
        scale=8,
    )
    _add_column_contracts(
        contracts,
        ("landmark_daily_traded_notional_usd",),
        "numeric",
        nullable=True,
        precision=24,
        scale=2,
    )
    remaining = set(EARLY_CUT_COLUMNS) - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=20,
        scale=8,
    )
    if set(contracts) != set(EARLY_CUT_COLUMNS):
        raise AssertionError("early-cut database column contract is incomplete")
    return contracts


def _rule_column_contracts() -> dict[str, ColumnContract]:
    contracts: dict[str, ColumnContract] = {}
    _add_column_contracts(contracts, ("result_id",), "int8", nullable=False)
    _add_column_contracts(
        contracts,
        (
            "rule_id",
            "result_kind",
            "decision_family",
            "objective",
            "protected_outcome",
            "feature_group",
            "rule_text",
            "evaluation_scope",
        ),
        "text",
        nullable=False,
    )
    _add_column_contracts(
        contracts,
        ("feature_name", "operator", "pattern_name", "pattern_match_mode"),
        "text",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        ("period_start", "period_end", "threshold_fit_end_date"),
        "date",
        nullable=True,
    )
    _add_column_contracts(
        contracts, ("is_selected", "is_final_filter"), "bool", nullable=False
    )
    _add_column_contracts(
        contracts,
        (
            "passes_holdout",
            "passes_development_gates",
            "passes_stability_gates",
            "passes_multiple_testing",
        ),
        "bool",
        nullable=True,
    )
    _add_column_contracts(
        contracts,
        (
            "landmark_day",
            "selection_order",
            "scope_year",
            "pattern_total_clause_count",
            "pattern_required_clause_count",
            "pattern_score_window_sessions",
        ),
        "int2",
        nullable=True,
    )
    _add_column_contracts(contracts, ("component_count",), "int2", nullable=False)
    _add_column_contracts(
        contracts,
        ("multiple_testing_candidate_count", "permutation_trial_count"),
        "int4",
        nullable=True,
    )
    count_columns = set(RULE_INTEGER_COLUMNS) - {
        "landmark_day",
        "selection_order",
        "scope_year",
        "component_count",
        "pattern_total_clause_count",
        "pattern_required_clause_count",
        "pattern_score_window_sessions",
        "multiple_testing_candidate_count",
        "permutation_trial_count",
    }
    _add_column_contracts(contracts, count_columns, "int4", nullable=False)
    _add_column_contracts(
        contracts,
        ("quantile_value", "threshold_value"),
        "numeric",
        nullable=True,
        precision=30,
        scale=10,
    )
    _add_column_contracts(
        contracts,
        ("pattern_score_threshold_pct",),
        "numeric",
        nullable=True,
        precision=20,
        scale=8,
    )
    remaining = {"result_id", *RULE_COLUMNS} - set(contracts)
    _add_column_contracts(
        contracts,
        remaining,
        "numeric",
        nullable=True,
        precision=18,
        scale=8,
    )
    if set(contracts) != {"result_id", *RULE_COLUMNS}:
        raise AssertionError("rule database column contract is incomplete")
    return contracts


SOURCE_COLUMN_CONTRACTS = _source_column_contracts()
FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMN_CONTRACTS = (
    _fundamental_snapshot_source_column_contracts()
)
QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMN_CONTRACTS = (
    _quarterly_fundamental_event_source_column_contracts()
)
EARNINGS_EVENT_SOURCE_COLUMN_CONTRACTS = _earnings_event_source_column_contracts()
MARKET_METRIC_SOURCE_COLUMN_CONTRACTS = _market_metric_source_column_contracts()
WORLD_MARKET_SOURCE_COLUMN_CONTRACTS = _world_market_source_column_contracts()
CURRENT_TAXONOMY_SOURCE_COLUMN_CONTRACTS = (
    _current_taxonomy_source_column_contracts()
)
SIGNAL_COLUMN_CONTRACTS = _signal_column_contracts()
EARLY_CUT_COLUMN_CONTRACTS = _early_cut_column_contracts()
RULE_COLUMN_CONTRACTS = _rule_column_contracts()


def _qualified_identifier(name: str) -> sql.Composed:
    """Quote a Config-validated, optionally schema-qualified identifier."""

    return sql.SQL(".").join(sql.Identifier(part) for part in name.split("."))


def _schema_and_table(name: str) -> tuple[str, str]:
    parts = name.split(".")
    if len(parts) == 1:
        return "public", parts[0]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"invalid qualified table name: {name!r}")


def _application_name(cfg: Config, app_suffix: str | None) -> str:
    suffix = app_suffix.strip() if app_suffix is not None else ""
    name = cfg.pg_app_name if not suffix else f"{cfg.pg_app_name}:{suffix}"
    if "\x00" in name:
        raise ValueError("PostgreSQL application_name must not contain NUL")
    # PostgreSQL application_name is limited to NAMEDATALEN - 1 bytes. Avoid
    # letting a multibyte character be cut in half.
    return name.encode("utf-8")[:63].decode("utf-8", errors="ignore")


@contextmanager
def connect(
    cfg: Config, app_suffix: str | None = None
) -> Iterator[extensions.connection]:
    """Open one process-owned connection with an explicit application name."""

    connection = psycopg2.connect(
        host=cfg.pg_host,
        port=cfg.pg_port,
        dbname=cfg.pg_database,
        user=cfg.pg_user,
        password=cfg.pg_password,
        application_name=_application_name(cfg, app_suffix),
        connect_timeout=cfg.db_connect_timeout_seconds,
        options=f"-c statement_timeout={cfg.db_statement_timeout_ms}",
    )
    try:
        yield connection
    finally:
        connection.close()


def _transaction_status(connection: extensions.connection) -> int:
    return int(connection.get_transaction_status())


@contextmanager
def advisory_lock(connection: extensions.connection) -> Iterator[None]:
    """Prevent two complete research rebuilds from running concurrently."""

    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        acquired = bool(cursor.fetchone()[0])
    if not acquired:
        connection.rollback()
        raise RuntimeError(
            "another stock_analyser_filter_research run is already active"
        )

    try:
        yield
    finally:
        # A failed statement leaves PostgreSQL unable to execute the unlock
        # query until the failed transaction is rolled back. Session locks are
        # intentionally unaffected by this rollback.
        if _transaction_status(connection) != extensions.TRANSACTION_STATUS_IDLE:
            connection.rollback()
        with connection.cursor() as cursor:
            cursor.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))
            released = bool(cursor.fetchone()[0])
        if not released:
            raise RuntimeError("research advisory lock was unexpectedly absent")


def begin_exported_snapshot(connection: extensions.connection) -> str:
    """Start and export the main process's repeatable, read-only snapshot.

    Schema validation consists only of reads, so any transaction it opened can
    be discarded safely before the snapshot starts. The caller must keep this
    transaction open until all workers have finished, then call commit().
    """

    if _transaction_status(connection) != extensions.TRANSACTION_STATUS_IDLE:
        connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cursor.execute("SELECT pg_export_snapshot()")
        snapshot_id = str(cursor.fetchone()[0])
    if not snapshot_id:
        connection.rollback()
        raise RuntimeError("PostgreSQL returned an empty exported snapshot id")
    return snapshot_id


def import_snapshot(connection: extensions.connection, snapshot_id: str) -> None:
    """Start a worker transaction on the main process's exported snapshot."""

    if not snapshot_id or "\x00" in snapshot_id:
        raise ValueError("snapshot_id must not be empty and must not contain NUL")
    if _transaction_status(connection) != extensions.TRANSACTION_STATUS_IDLE:
        connection.rollback()
    with connection.cursor() as cursor:
        cursor.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        # This must be the first command after BEGIN that establishes snapshot
        # state. Binding the value also avoids treating it as SQL syntax.
        cursor.execute("SET TRANSACTION SNAPSHOT %s", (snapshot_id,))


def _column_names(connection: extensions.connection, table_name: str) -> set[str]:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, relation_name),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def _column_definitions(
    connection: extensions.connection, table_name: str
) -> dict[str, ColumnContract]:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                column_name,
                udt_name,
                is_nullable,
                numeric_precision,
                numeric_scale
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, relation_name),
        )
        rows = cursor.fetchall()
    return {
        str(column_name): (
            str(udt_name),
            str(is_nullable).upper() == "YES",
            int(numeric_precision) if numeric_precision is not None else None,
            int(numeric_scale) if numeric_scale is not None else None,
        )
        for (
            column_name,
            udt_name,
            is_nullable,
            numeric_precision,
            numeric_scale,
        ) in rows
    }


def _column_contract_text(contract: ColumnContract) -> str:
    udt_name, nullable, precision, scale = contract
    type_text = f"numeric({precision},{scale})" if udt_name == "numeric" else udt_name
    return type_text + (" NULL" if nullable else " NOT NULL")


def _validate_column_definitions(
    connection: extensions.connection,
    table_name: str,
    expected: Mapping[str, ColumnContract],
) -> None:
    actual = _column_definitions(connection, table_name)
    mismatches: list[str] = []
    for column, expected_contract in expected.items():
        actual_contract = actual.get(column)
        if actual_contract is None:
            # Missing-column reporting is handled by the preceding set check.
            continue
        expected_udt, expected_nullable, expected_precision, expected_scale = (
            expected_contract
        )
        actual_udt, actual_nullable, actual_precision, actual_scale = actual_contract
        type_matches = actual_udt == expected_udt
        if expected_udt == "numeric":
            type_matches = type_matches and (
                actual_precision == expected_precision
                and actual_scale == expected_scale
            )
        if not type_matches or actual_nullable != expected_nullable:
            mismatches.append(
                f"{column}: expected "
                f"{_column_contract_text(expected_contract)}, found "
                f"{_column_contract_text(actual_contract)}"
            )
    if mismatches:
        raise RuntimeError(
            f"table {table_name} has incompatible column definitions ("
            + "; ".join(sorted(mismatches))
            + ")"
        )


def _validate_required_columns(
    connection: extensions.connection,
    table_name: str,
    required_columns: Sequence[str],
    *,
    reject_extra: bool,
) -> None:
    actual = _column_names(connection, table_name)
    required = set(required_columns)
    missing = sorted(required - actual)
    extra = sorted(actual - required) if reject_extra else []
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if extra:
            details.append("unexpected: " + ", ".join(extra))
        raise RuntimeError(
            f"table {table_name} has incompatible columns (" + "; ".join(details) + ")"
        )


def _primary_key(connection: extensions.connection, table_name: str) -> tuple[str, ...]:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
              ON kcu.constraint_schema = tc.constraint_schema
             AND kcu.constraint_name = tc.constraint_name
             AND kcu.table_schema = tc.table_schema
             AND kcu.table_name = tc.table_name
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s
              AND tc.table_name = %s
            ORDER BY kcu.ordinal_position
            """,
            (schema_name, relation_name),
        )
        return tuple(str(row[0]) for row in cursor.fetchall())


def _validate_primary_key(
    connection: extensions.connection,
    table_name: str,
    expected: tuple[str, ...],
) -> None:
    actual = _primary_key(connection, table_name)
    if actual != expected:
        raise RuntimeError(
            f"table {table_name} requires primary key {expected!r}, "
            f"found {actual!r}"
        )


def _validate_hypertable(
    connection: extensions.connection,
    table_name: str,
    time_column: str,
) -> None:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT compression_enabled
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = %s AND hypertable_name = %s
            """,
            (schema_name, relation_name),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"target table {table_name} is not a hypertable")
        if bool(row[0]):
            raise RuntimeError(
                f"target hypertable {table_name} must not use compression"
            )

        cursor.execute(
            """
            SELECT column_name, time_interval
            FROM timescaledb_information.dimensions
            WHERE hypertable_schema = %s AND hypertable_name = %s
            ORDER BY dimension_number
            """,
            (schema_name, relation_name),
        )
        dimensions = cursor.fetchall()
    if dimensions != [(time_column, EXPECTED_CHUNK_INTERVAL)]:
        raise RuntimeError(
            f"target hypertable {table_name} requires one {time_column} time "
            f"dimension with a 365-day chunk interval, found {dimensions!r}"
        )


def _validate_rule_is_regular_table(
    connection: extensions.connection, table_name: str
) -> None:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM timescaledb_information.hypertables
                WHERE hypertable_schema = %s AND hypertable_name = %s
            )
            """,
            (schema_name, relation_name),
        )
        is_hypertable = bool(cursor.fetchone()[0])
    if is_hypertable:
        raise RuntimeError(f"rule target {table_name} must be a regular table")


def _validate_rule_result_id_default(
    connection: extensions.connection, table_name: str
) -> None:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT is_nullable, column_default, is_identity
            FROM information_schema.columns
            WHERE table_schema = %s
              AND table_name = %s
              AND column_name = 'result_id'
            """,
            (schema_name, relation_name),
        )
        row = cursor.fetchone()
    if row is None:
        raise RuntimeError(f"rule target {table_name} has no result_id column")
    is_nullable, column_default, is_identity = row
    generated = str(is_identity).upper() == "YES" or (
        column_default is not None
        and str(column_default).lstrip().lower().startswith("nextval(")
    )
    if str(is_nullable).upper() != "NO" or not generated:
        raise RuntimeError(
            f"rule target {table_name}.result_id must be non-null and generated"
        )


def validate_schema(connection: extensions.connection, cfg: Config) -> None:
    """Validate the complete read/write contract without creating any object."""

    _validate_required_columns(
        connection,
        cfg.source_table,
        SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.fundamental_snapshot_table,
        FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.quarterly_fundamental_event_table,
        QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.earnings_event_table,
        EARNINGS_EVENT_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.market_metrics_table,
        MARKET_METRIC_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.world_market_observation_table,
        WORLD_MARKET_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.security_master_current_table,
        CURRENT_TAXONOMY_SOURCE_COLUMNS,
        reject_extra=False,
    )
    _validate_required_columns(
        connection,
        cfg.signal_result_table,
        SIGNAL_COLUMNS,
        reject_extra=True,
    )
    _validate_required_columns(
        connection,
        cfg.early_cut_result_table,
        EARLY_CUT_COLUMNS,
        reject_extra=True,
    )
    _validate_required_columns(
        connection,
        cfg.rule_result_table,
        ("result_id", *RULE_COLUMNS),
        reject_extra=True,
    )
    _validate_column_definitions(connection, cfg.source_table, SOURCE_COLUMN_CONTRACTS)
    _validate_column_definitions(
        connection,
        cfg.fundamental_snapshot_table,
        FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection,
        cfg.quarterly_fundamental_event_table,
        QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection,
        cfg.earnings_event_table,
        EARNINGS_EVENT_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection,
        cfg.market_metrics_table,
        MARKET_METRIC_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection,
        cfg.world_market_observation_table,
        WORLD_MARKET_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection,
        cfg.security_master_current_table,
        CURRENT_TAXONOMY_SOURCE_COLUMN_CONTRACTS,
    )
    _validate_column_definitions(
        connection, cfg.signal_result_table, SIGNAL_COLUMN_CONTRACTS
    )
    _validate_column_definitions(
        connection, cfg.early_cut_result_table, EARLY_CUT_COLUMN_CONTRACTS
    )
    _validate_column_definitions(
        connection, cfg.rule_result_table, RULE_COLUMN_CONTRACTS
    )
    _validate_primary_key(
        connection,
        cfg.signal_result_table,
        EXPECTED_SIGNAL_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.fundamental_snapshot_table,
        EXPECTED_FUNDAMENTAL_SNAPSHOT_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.quarterly_fundamental_event_table,
        EXPECTED_QUARTERLY_FUNDAMENTAL_EVENT_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.earnings_event_table,
        EXPECTED_EARNINGS_EVENT_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.market_metrics_table,
        EXPECTED_MARKET_METRIC_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.security_master_current_table,
        IDENTITY_COLUMNS,
    )
    _validate_primary_key(
        connection,
        cfg.early_cut_result_table,
        EXPECTED_EARLY_CUT_PRIMARY_KEY,
    )
    _validate_primary_key(
        connection,
        cfg.rule_result_table,
        EXPECTED_RULE_PRIMARY_KEY,
    )
    _validate_hypertable(connection, cfg.signal_result_table, "signal_date")
    _validate_hypertable(connection, cfg.early_cut_result_table, "signal_date")
    _validate_rule_is_regular_table(connection, cfg.rule_result_table)
    _validate_rule_result_id_default(connection, cfg.rule_result_table)


def _target_empty(connection: extensions.connection, table_name: str) -> bool:
    statement = sql.SQL("SELECT NOT EXISTS (SELECT 1 FROM {} LIMIT 1)").format(
        _qualified_identifier(table_name)
    )
    with connection.cursor() as cursor:
        cursor.execute(statement)
        return bool(cursor.fetchone()[0])


def assert_targets_empty(connection: extensions.connection, cfg: Config) -> None:
    """Refuse an implicit append/rebuild over prior research output."""

    nonempty = [
        table_name
        for table_name in (
            cfg.signal_result_table,
            cfg.early_cut_result_table,
            cfg.rule_result_table,
        )
        if not _target_empty(connection, table_name)
    ]
    if nonempty:
        raise RuntimeError(
            "result tables must be empty for a full research rebuild: "
            + ", ".join(nonempty)
            + "; run the Init-SQL service explicitly with "
            "DROP_ALL_STOCK_ANALYSER_FILTER_RESEARCH_TABLES_ON_START=true"
        )


def load_trading_dates(
    connection: extensions.connection, cfg: Config
) -> pd.DatetimeIndex:
    """Load every available session needed for lookbacks and forward labels."""

    statement = sql.SQL(
        "SELECT DISTINCT period_end_date FROM {} ORDER BY period_end_date"
    ).format(_qualified_identifier(cfg.source_table))
    with connection.cursor() as cursor:
        cursor.execute(statement)
        values = [row[0] for row in cursor.fetchall()]
    return pd.DatetimeIndex(values).normalize()


def load_identity_work(
    connection: extensions.connection, cfg: Config
) -> list[IdentityWork]:
    """Return relevant identities and their actual source-row workload.

    Identities without a passing day in the requested signal period cannot
    produce an event, so they are omitted. Every available row for a relevant
    identity is counted and later loaded, preserving causal lookbacks and the
    post-signal D+90 observation window even when SIGNAL_END_DATE is configured.
    """

    statement = sql.SQL(
        "SELECT source.symbol, source.exchange, source.cik, count(*)::bigint "
        "FROM {} AS source "
        "GROUP BY source.symbol, source.exchange, source.cik "
        "HAVING bool_or(source.trend_template_pass "
        "AND source.period_end_date >= %s "
        "AND (%s::date IS NULL OR source.period_end_date <= %s)) "
        "ORDER BY source.symbol, source.exchange, source.cik"
    ).format(_qualified_identifier(cfg.source_table))
    parameters = [
        cfg.signal_start_date,
        cfg.signal_end_date,
        cfg.signal_end_date,
    ]
    with connection.cursor() as cursor:
        cursor.execute(statement, parameters)
        rows = cursor.fetchall()
    return [
        (str(symbol), str(exchange), int(cik), int(row_count))
        for symbol, exchange, cik, row_count in rows
    ]


def _normalize_source_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.loc[:, CALCULATION_SOURCE_COLUMNS].copy()
    frame["period_end_date"] = pd.to_datetime(
        frame["period_end_date"], errors="raise"
    ).dt.normalize()

    for column in ("symbol", "exchange", "currency"):
        if frame[column].isna().any():
            raise RuntimeError(f"source column {column} unexpectedly contains null")
        frame[column] = frame[column].astype("string")

    for column in SOURCE_BOOLEAN_COLUMNS:
        if frame[column].isna().any():
            raise RuntimeError(f"source column {column} unexpectedly contains null")
        try:
            frame[column] = frame[column].astype("boolean")
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"source column {column} is not boolean") from exc

    for column in SOURCE_INTEGER_COLUMNS:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise RuntimeError(f"source column {column} contains invalid integers")
        finite = numeric.dropna().to_numpy(dtype=float)
        if finite.size and (
            not np.isfinite(finite).all()
            or not np.equal(finite, np.floor(finite)).all()
        ):
            raise RuntimeError(f"source column {column} contains invalid integers")
        try:
            frame[column] = numeric.astype("Int64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(f"source column {column} exceeds int64") from exc

    non_numeric = {
        "period_end_date",
        "symbol",
        "exchange",
        "currency",
        *SOURCE_BOOLEAN_COLUMNS,
        *SOURCE_INTEGER_COLUMNS,
    }
    for column in set(CALCULATION_SOURCE_COLUMNS) - non_numeric:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        if invalid.any():
            raise RuntimeError(f"source column {column} contains invalid numerics")
        frame[column] = numeric.astype("float64").replace([np.inf, -np.inf], np.nan)

    return frame


def _validate_identities(
    identities: Sequence[StockIdentity],
) -> tuple[list[str], list[str], list[int]]:
    normalized: list[StockIdentity] = []
    for identity in identities:
        if len(identity) != 3:
            raise ValueError("each stock identity must contain symbol, exchange, cik")
        symbol, exchange, cik = identity
        if not isinstance(symbol, str) or not symbol or "\x00" in symbol:
            raise ValueError("identity symbol must be a non-empty string")
        if not isinstance(exchange, str) or not exchange or "\x00" in exchange:
            raise ValueError("identity exchange must be a non-empty string")
        if not isinstance(cik, Integral) or isinstance(cik, bool):
            raise ValueError("identity cik must be an integer")
        normalized.append((symbol, exchange, int(cik)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("identity batch contains duplicates")
    return (
        [item[0] for item in normalized],
        [item[1] for item in normalized],
        [item[2] for item in normalized],
    )


def load_source_batch(
    connection: extensions.connection,
    cfg: Config,
    identities: Sequence[StockIdentity],
) -> pd.DataFrame:
    """Stream all available history for one bounded batch of identities."""

    if not identities:
        return pd.DataFrame(columns=CALCULATION_SOURCE_COLUMNS)
    symbols, exchanges, ciks = _validate_identities(identities)
    selected_columns = sql.SQL(", ").join(
        sql.SQL("source.{}").format(sql.Identifier(column)) for column in SOURCE_COLUMNS
    )
    selected_columns = selected_columns + sql.SQL(", market.adjusted_open")
    statement = sql.SQL(
        "SELECT {} FROM {} AS source "
        "LEFT JOIN {} AS market ON market.period_end_date = source.period_end_date "
        "AND market.symbol = source.symbol "
        "AND market.exchange = source.exchange "
        "AND market.cik = source.cik "
        "JOIN unnest(%s::text[], %s::text[], %s::bigint[]) "
        "AS selected(symbol, exchange, cik) "
        "ON source.symbol = selected.symbol "
        "AND source.exchange = selected.exchange "
        "AND source.cik = selected.cik "
        "ORDER BY source.symbol, source.exchange, source.cik, "
        "source.period_end_date"
    ).format(
        selected_columns,
        _qualified_identifier(cfg.source_table),
        _qualified_identifier(cfg.market_metrics_table),
    )
    parameters: list[Any] = [symbols, exchanges, ciks]
    frames: list[pd.DataFrame] = []
    cursor_name = f"safr_source_{uuid4().hex[:20]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(statement, parameters)
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            raw = pd.DataFrame.from_records(
                rows, columns=CALCULATION_SOURCE_COLUMNS
            )
            frames.append(_normalize_source_frame(raw))
    if not frames:
        return pd.DataFrame(columns=CALCULATION_SOURCE_COLUMNS)
    return pd.concat(frames, ignore_index=True).loc[:, CALCULATION_SOURCE_COLUMNS]


def _normalize_required_text(
    frame: pd.DataFrame, columns: Sequence[str], source_name: str
) -> None:
    for column in columns:
        if frame[column].isna().any():
            raise RuntimeError(
                f"{source_name} column {column} unexpectedly contains null"
            )
        frame[column] = frame[column].astype("string")


def _normalize_cik(frame: pd.DataFrame, source_name: str) -> None:
    original = frame["cik"]
    numeric = pd.to_numeric(original, errors="coerce")
    values = numeric.dropna().to_numpy(dtype=float)
    if (
        original.isna().any()
        or numeric.isna().any()
        or (values.size and not np.isfinite(values).all())
        or (values.size and not np.equal(values, np.floor(values)).all())
    ):
        raise RuntimeError(f"{source_name} column cik contains invalid integers")
    try:
        frame["cik"] = numeric.astype("Int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{source_name} column cik exceeds int64") from exc


def _normalize_optional_currency(frame: pd.DataFrame, column: str) -> None:
    currency = frame[column].astype("string").str.strip().str.upper()
    frame[column] = currency.mask(currency.eq(""), pd.NA)


def _normalize_numeric_columns(
    frame: pd.DataFrame, columns: Sequence[str], source_name: str
) -> None:
    for column in columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise RuntimeError(
                f"{source_name} column {column} contains invalid numerics"
            )
        frame[column] = numeric.astype("float64").replace([np.inf, -np.inf], np.nan)


def _normalize_fundamental_snapshot_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "fundamental snapshot source"
    frame = frame.loc[:, FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS].copy()
    _normalize_required_text(frame, ("symbol", "exchange"), source_name)
    _normalize_cik(frame, source_name)
    frame["period_end_date"] = pd.to_datetime(
        frame["period_end_date"], errors="raise"
    ).dt.normalize()
    frame["sec_latest_period_end_date"] = pd.to_datetime(
        frame["sec_latest_period_end_date"], errors="coerce"
    ).dt.normalize()
    frame["sec_data_available_at"] = pd.to_datetime(
        frame["sec_data_available_at"], errors="coerce", utc=True
    )
    _normalize_optional_currency(frame, "sec_fundamental_currency")
    non_numeric = {
        *IDENTITY_COLUMNS,
        "period_end_date",
        "sec_fundamental_currency",
        "sec_latest_period_end_date",
        "sec_data_available_at",
    }
    _normalize_numeric_columns(
        frame,
        tuple(set(FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS) - non_numeric),
        source_name,
    )
    return frame.loc[:, FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS]


def _normalize_quarterly_fundamental_event_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    source_name = "quarterly fundamental event source"
    frame = frame.loc[:, QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS].copy()
    _normalize_required_text(
        frame, ("symbol", "exchange", "accession_number"), source_name
    )
    _normalize_cik(frame, source_name)
    for column in ("effective_date", "fiscal_period_end_date"):
        frame[column] = pd.to_datetime(frame[column], errors="raise").dt.normalize()
    frame["accepted_at"] = pd.to_datetime(
        frame["accepted_at"], errors="coerce", utc=True
    )
    _normalize_optional_currency(frame, "currency")
    non_numeric = {
        *IDENTITY_COLUMNS,
        "accession_number",
        "accepted_at",
        "effective_date",
        "fiscal_period_end_date",
        "currency",
    }
    _normalize_numeric_columns(
        frame,
        tuple(set(QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS) - non_numeric),
        source_name,
    )
    return frame.loc[:, QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS]


def _normalize_earnings_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "earnings event source"
    frame = frame.loc[:, EARNINGS_EVENT_SOURCE_COLUMNS].copy()
    _normalize_required_text(
        frame,
        (
            "symbol",
            "exchange",
            "announcement_time_type",
            "source",
            "source_event_id",
        ),
        source_name,
    )
    _normalize_cik(frame, source_name)
    frame["earnings_date"] = pd.to_datetime(
        frame["earnings_date"], errors="raise"
    ).dt.normalize()
    frame["announcement_ts"] = pd.to_datetime(
        frame["announcement_ts"], errors="coerce", utc=True
    )
    frame["known_as_of_ts"] = pd.to_datetime(
        frame["known_as_of_ts"], errors="raise", utc=True
    )
    try:
        frame["is_confirmed"] = frame["is_confirmed"].astype("boolean")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("earnings event source is_confirmed is not boolean") from exc
    return frame.loc[:, EARNINGS_EVENT_SOURCE_COLUMNS]


def _load_fundamental_identity_batch(
    connection: extensions.connection,
    cfg: Config,
    identities: Sequence[StockIdentity],
    *,
    table_name: str,
    columns: Sequence[str],
    order_columns: Sequence[str],
    cursor_prefix: str,
    normalize: Callable[[pd.DataFrame], pd.DataFrame],
    additional_where: sql.Composable | None = None,
) -> pd.DataFrame:
    if not identities:
        return pd.DataFrame(columns=columns)
    symbols, exchanges, ciks = _validate_identities(identities)
    selected_columns = sql.SQL(", ").join(
        sql.SQL("source.{}").format(sql.Identifier(column)) for column in columns
    )
    ordering = sql.SQL(", ").join(
        sql.SQL("source.{}").format(sql.Identifier(column))
        for column in order_columns
    )
    where_clause = additional_where or sql.SQL("")
    statement = sql.SQL(
        "SELECT {} FROM {} AS source "
        "JOIN unnest(%s::text[], %s::text[], %s::bigint[]) "
        "AS selected(symbol, exchange, cik) "
        "ON source.symbol = selected.symbol "
        "AND source.exchange = selected.exchange "
        "AND source.cik = selected.cik "
        "{} ORDER BY {}"
    ).format(
        selected_columns,
        _qualified_identifier(table_name),
        where_clause,
        ordering,
    )
    parameters: list[Any] = [symbols, exchanges, ciks]
    frames: list[pd.DataFrame] = []
    cursor_name = f"{cursor_prefix}_{uuid4().hex[:20]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(statement, parameters)
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            raw = pd.DataFrame.from_records(rows, columns=columns)
            frames.append(normalize(raw))
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True).loc[:, columns]


def load_fundamental_snapshot_batch(
    connection: extensions.connection,
    cfg: Config,
    identities: Sequence[StockIdentity],
) -> pd.DataFrame:
    """Load SEC snapshots for a bounded identity batch in the shared DB snapshot."""

    return _load_fundamental_identity_batch(
        connection,
        cfg,
        identities,
        table_name=cfg.fundamental_snapshot_table,
        columns=FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
        order_columns=(*IDENTITY_COLUMNS, "period_end_date"),
        cursor_prefix="safr_fund_snapshot",
        normalize=_normalize_fundamental_snapshot_frame,
    )


def load_quarterly_fundamental_event_batch(
    connection: extensions.connection,
    cfg: Config,
    identities: Sequence[StockIdentity],
) -> pd.DataFrame:
    """Load SEC quarterly events for a bounded identity batch in the DB snapshot."""

    return _load_fundamental_identity_batch(
        connection,
        cfg,
        identities,
        table_name=cfg.quarterly_fundamental_event_table,
        columns=QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
        order_columns=(*IDENTITY_COLUMNS, "fiscal_period_end_date", "accepted_at"),
        cursor_prefix="safr_fund_event",
        normalize=_normalize_quarterly_fundamental_event_frame,
    )


def load_earnings_event_batch(
    connection: extensions.connection,
    cfg: Config,
    identities: Sequence[StockIdentity],
) -> pd.DataFrame:
    """Load earnings events; causal enrichment accepts SEC events only."""

    return _load_fundamental_identity_batch(
        connection,
        cfg,
        identities,
        table_name=cfg.earnings_event_table,
        columns=EARNINGS_EVENT_SOURCE_COLUMNS,
        order_columns=(*IDENTITY_COLUMNS, "earnings_date", "known_as_of_ts"),
        cursor_prefix="safr_earnings_event",
        normalize=_normalize_earnings_event_frame,
        additional_where=sql.SQL(
            "WHERE source.source = 'sec_8k_item_2_02' "
            "AND source.is_confirmed = true"
        ),
    )


def _normalize_nullable_integer_column(
    frame: pd.DataFrame, column: str, source_name: str
) -> None:
    original = frame[column]
    numeric = pd.to_numeric(original, errors="coerce")
    values = numeric.dropna().to_numpy(dtype=float)
    if (
        (original.notna() & numeric.isna()).any()
        or (values.size and not np.isfinite(values).all())
        or (values.size and not np.equal(values, np.floor(values)).all())
    ):
        raise RuntimeError(f"{source_name} column {column} contains invalid integers")
    try:
        frame[column] = numeric.astype("Int64")
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"{source_name} column {column} exceeds int64") from exc


def _normalize_market_metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "market metric source"
    frame = frame.loc[:, MARKET_METRIC_SOURCE_COLUMNS].copy()
    _normalize_required_text(frame, ("symbol", "exchange"), source_name)
    _normalize_cik(frame, source_name)
    frame["period_end_date"] = pd.to_datetime(
        frame["period_end_date"], errors="raise"
    ).dt.normalize()
    _normalize_optional_currency(frame, "market_cap_currency")
    _normalize_nullable_integer_column(frame, "market_cap", source_name)
    _normalize_nullable_integer_column(frame, "raw_volume", source_name)
    _normalize_nullable_integer_column(frame, "shares_outstanding", source_name)
    _normalize_nullable_integer_column(
        frame, "shares_outstanding_staleness_days", source_name
    )
    _normalize_numeric_columns(frame, ("adjusted_open",), source_name)
    source_values = frame["shares_outstanding_source"].astype("string").str.strip()
    frame["shares_outstanding_source"] = source_values.mask(
        source_values.eq(""), pd.NA
    )
    return frame.loc[:, MARKET_METRIC_SOURCE_COLUMNS]


def _validate_signal_keys(
    signal_keys: Sequence[StockSignalKey],
) -> tuple[list[date], list[str], list[str], list[int]]:
    normalized: list[StockSignalKey] = []
    for key in signal_keys:
        if len(key) != 4:
            raise ValueError(
                "each signal key must contain signal_date, symbol, exchange, cik"
            )
        signal_date, symbol, exchange, cik = key
        if pd.isna(signal_date):
            raise ValueError("signal key date must be a valid date")
        try:
            normalized_date = pd.Timestamp(signal_date).date()
        except (TypeError, ValueError) as exc:
            raise ValueError("signal key date must be a valid date") from exc
        if not isinstance(symbol, str) or not symbol or "\x00" in symbol:
            raise ValueError("signal key symbol must be a non-empty string")
        if not isinstance(exchange, str) or not exchange or "\x00" in exchange:
            raise ValueError("signal key exchange must be a non-empty string")
        if not isinstance(cik, Integral) or isinstance(cik, bool):
            raise ValueError("signal key cik must be an integer")
        normalized.append((normalized_date, symbol, exchange, int(cik)))
    if len(set(normalized)) != len(normalized):
        raise ValueError("signal key batch contains duplicates")
    return (
        [item[0] for item in normalized],
        [item[1] for item in normalized],
        [item[2] for item in normalized],
        [item[3] for item in normalized],
    )


def load_market_metrics_for_signals(
    connection: extensions.connection,
    cfg: Config,
    signal_keys: Sequence[StockSignalKey],
) -> pd.DataFrame:
    """Load only exact signal-day market metrics from the shared DB snapshot."""

    if not signal_keys:
        return pd.DataFrame(columns=MARKET_METRIC_SOURCE_COLUMNS)
    dates, symbols, exchanges, ciks = _validate_signal_keys(signal_keys)
    selected_columns = sql.SQL(", ").join(
        sql.SQL("source.{}").format(sql.Identifier(column))
        for column in MARKET_METRIC_SOURCE_COLUMNS
    )
    statement = sql.SQL(
        "SELECT {} FROM {} AS source "
        "JOIN unnest(%s::date[], %s::text[], %s::text[], %s::bigint[]) "
        "AS selected(period_end_date, symbol, exchange, cik) "
        "ON source.period_end_date = selected.period_end_date "
        "AND source.symbol = selected.symbol "
        "AND source.exchange = selected.exchange "
        "AND source.cik = selected.cik "
        "ORDER BY source.symbol, source.exchange, source.cik, "
        "source.period_end_date"
    ).format(selected_columns, _qualified_identifier(cfg.market_metrics_table))
    parameters: list[Any] = [dates, symbols, exchanges, ciks]
    frames: list[pd.DataFrame] = []
    cursor_name = f"safr_market_metric_{uuid4().hex[:20]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(statement, parameters)
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            raw = pd.DataFrame.from_records(
                rows, columns=MARKET_METRIC_SOURCE_COLUMNS
            )
            frames.append(_normalize_market_metric_frame(raw))
    if not frames:
        return pd.DataFrame(columns=MARKET_METRIC_SOURCE_COLUMNS)
    result = pd.concat(frames, ignore_index=True).loc[
        :, MARKET_METRIC_SOURCE_COLUMNS
    ]
    if result.duplicated(["period_end_date", *IDENTITY_COLUMNS]).any():
        raise RuntimeError("market metric source returned duplicate signal keys")
    return result


MARKET_BREADTH_COLUMNS = (
    "market_date",
    "market_breadth_above_ma50_ratio",
    "market_breadth_above_ma150_ratio",
    "market_breadth_above_ma200_ratio",
    "market_breadth_trend_template_ratio",
    "market_breadth_rs70_ratio",
    "market_breadth_rs90_ratio",
    "market_advancer_ratio",
    "market_median_daily_return_pct",
)


def load_market_breadth_daily(
    connection: extensions.connection, cfg: Config
) -> pd.DataFrame:
    """Aggregate causal same-close cross-sectional breadth once per run."""

    statement = sql.SQL(
        "SELECT period_end_date AS market_date, "
        "(avg((adjusted_close > ma50)::int) "
        "FILTER (WHERE adjusted_close IS NOT NULL AND ma50 IS NOT NULL))::double precision, "
        "(avg((adjusted_close > ma150)::int) "
        "FILTER (WHERE adjusted_close IS NOT NULL AND ma150 IS NOT NULL))::double precision, "
        "(avg((adjusted_close > ma200)::int) "
        "FILTER (WHERE adjusted_close IS NOT NULL AND ma200 IS NOT NULL))::double precision, "
        "avg(trend_template_pass::int)::double precision, "
        "(avg((rs_rating >= 70)::int) "
        "FILTER (WHERE rs_rating IS NOT NULL))::double precision, "
        "(avg((rs_rating >= 90)::int) "
        "FILTER (WHERE rs_rating IS NOT NULL))::double precision, "
        "(avg((daily_price_change_pct > 0)::int) "
        "FILTER (WHERE daily_price_change_pct IS NOT NULL))::double precision, "
        "percentile_cont(0.5) WITHIN GROUP (ORDER BY daily_price_change_pct) "
        "FILTER (WHERE daily_price_change_pct IS NOT NULL) "
        "FROM {} WHERE period_end_date >= %s "
        "AND (%s::date IS NULL OR period_end_date <= %s) "
        "GROUP BY period_end_date ORDER BY period_end_date"
    ).format(_qualified_identifier(cfg.source_table))
    with connection.cursor() as cursor:
        cursor.execute(
            statement,
            (cfg.signal_start_date, cfg.signal_end_date, cfg.signal_end_date),
        )
        rows = cursor.fetchall()
    frame = pd.DataFrame.from_records(rows, columns=MARKET_BREADTH_COLUMNS)
    if frame.empty:
        return frame
    frame["market_date"] = pd.to_datetime(
        frame["market_date"], errors="raise"
    ).dt.normalize()
    _normalize_numeric_columns(
        frame, MARKET_BREADTH_COLUMNS[1:], "market breadth source"
    )
    return frame.loc[:, MARKET_BREADTH_COLUMNS]


def load_current_taxonomy_backcast(
    connection: extensions.connection, cfg: Config
) -> pd.DataFrame:
    """Load the current IBKR hierarchy used only for backcast diagnostics."""

    selected = sql.SQL(", ").join(
        sql.Identifier(column) for column in CURRENT_TAXONOMY_SOURCE_COLUMNS
    )
    statement = sql.SQL("SELECT {} FROM {} ORDER BY symbol, exchange, cik").format(
        selected,
        _qualified_identifier(cfg.security_master_current_table),
    )
    with connection.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
    frame = pd.DataFrame.from_records(rows, columns=CURRENT_TAXONOMY_SOURCE_COLUMNS)
    if frame.empty:
        return frame
    _normalize_required_text(frame, ("symbol", "exchange"), "current taxonomy")
    _normalize_cik(frame, "current taxonomy")
    for column in ("ibkr_industry", "ibkr_category", "ibkr_subcategory"):
        values = frame[column].astype("string").str.strip()
        frame[column] = values.mask(values.eq(""), pd.NA)
    if frame.duplicated(list(IDENTITY_COLUMNS)).any():
        raise RuntimeError("current taxonomy contains duplicate stock identities")
    return frame.loc[:, CURRENT_TAXONOMY_SOURCE_COLUMNS]


def _taxonomy_group_aggregate_sql() -> sql.SQL:
    return sql.SQL(
        """
        SELECT
            period_end_date,
            CASE
                WHEN GROUPING(ibkr_category) = 1 THEN 'industry'
                WHEN GROUPING(ibkr_subcategory) = 1 THEN 'category'
                ELSE 'subcategory'
            END::text,
            ibkr_industry,
            ibkr_category,
            ibkr_subcategory,
            count(adjusted_close)::bigint,
            count(return_21d_pct)::bigint,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY return_21d_pct)
                FILTER (WHERE return_21d_pct IS NOT NULL),
            count(return_63d_pct)::bigint,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY return_63d_pct)
                FILTER (WHERE return_63d_pct IS NOT NULL),
            count(return_126d_pct)::bigint,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY return_126d_pct)
                FILTER (WHERE return_126d_pct IS NOT NULL),
            count(return_252d_pct)::bigint,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY return_252d_pct)
                FILTER (WHERE return_252d_pct IS NOT NULL),
            count(rs_raw)::bigint,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY rs_raw)
                FILTER (WHERE rs_raw IS NOT NULL),
            count(*) FILTER (
                WHERE adjusted_close IS NOT NULL AND ma50 IS NOT NULL
            )::bigint,
            avg((adjusted_close > ma50)::int) FILTER (
                WHERE adjusted_close IS NOT NULL AND ma50 IS NOT NULL
            )::double precision,
            count(*) FILTER (
                WHERE adjusted_close IS NOT NULL AND ma200 IS NOT NULL
            )::bigint,
            avg((adjusted_close > ma200)::int) FILTER (
                WHERE adjusted_close IS NOT NULL AND ma200 IS NOT NULL
            )::double precision,
            count(*) FILTER (
                WHERE adjusted_high IS NOT NULL AND high_52w IS NOT NULL
            )::bigint,
            count(*) FILTER (
                WHERE adjusted_high IS NOT NULL AND high_52w IS NOT NULL
                  AND adjusted_high >= high_52w
            )::bigint,
            avg((adjusted_high >= high_52w)::int) FILTER (
                WHERE adjusted_high IS NOT NULL AND high_52w IS NOT NULL
            )::double precision,
            count(rs_rating)::bigint,
            avg((rs_rating >= 70)::int) FILTER (
                WHERE rs_rating IS NOT NULL
            )::double precision,
            avg((rs_rating >= 90)::int) FILTER (
                WHERE rs_rating IS NOT NULL
            )::double precision,
            count(trend_template_pass)::bigint,
            avg(trend_template_pass::int)::double precision,
            count(*) FILTER (WHERE is_new_8of8_signal)::bigint,
            avg(is_new_8of8_signal::int)::double precision
        FROM member_features
        WHERE ibkr_industry IS NOT NULL
        GROUP BY GROUPING SETS (
            (period_end_date, ibkr_industry),
            (period_end_date, ibkr_industry, ibkr_category),
            (
                period_end_date, ibkr_industry, ibkr_category,
                ibkr_subcategory
            )
        )
        HAVING
            GROUPING(ibkr_category) = 1
            OR (
                GROUPING(ibkr_subcategory) = 1
                AND ibkr_category IS NOT NULL
            )
            OR (
                GROUPING(ibkr_subcategory) = 0
                AND ibkr_category IS NOT NULL
                AND ibkr_subcategory IS NOT NULL
            )
        """
    )


def load_current_taxonomy_backcast_group_context(
    connection: extensions.connection, cfg: Config
) -> pd.DataFrame:
    """Aggregate daily current-taxonomy group data for diagnostic backcasting."""

    aggregates = _taxonomy_group_aggregate_sql()
    statement = sql.SQL(
        """
        WITH classified AS (
            SELECT
                source.period_end_date,
                source.symbol,
                source.exchange,
                source.cik,
                source.price_continuity_segment,
                source.adjusted_close,
                source.adjusted_high,
                source.ma50,
                source.ma200,
                source.high_52w,
                source.rs_raw,
                source.rs_rating,
                source.trend_template_pass,
                NULLIF(TRIM(taxonomy.ibkr_industry), '') AS ibkr_industry,
                NULLIF(TRIM(taxonomy.ibkr_category), '') AS ibkr_category,
                NULLIF(TRIM(taxonomy.ibkr_subcategory), '') AS ibkr_subcategory,
                lag(source.adjusted_close, 21) OVER identity_segment AS close_21d_ago,
                lag(source.adjusted_close, 63) OVER identity_segment AS close_63d_ago,
                lag(source.adjusted_close, 126) OVER identity_segment AS close_126d_ago,
                lag(source.adjusted_close, 252) OVER identity_segment AS close_252d_ago,
                lag(source.trend_template_pass, 1)
                    OVER identity_segment AS prior_trend_template_pass
            FROM {source_table} AS source
            JOIN {taxonomy_table} AS taxonomy
              ON taxonomy.symbol = source.symbol
             AND taxonomy.exchange = source.exchange
             AND taxonomy.cik = source.cik
            WINDOW identity_segment AS (
                PARTITION BY source.symbol, source.exchange, source.cik,
                             source.price_continuity_segment
                ORDER BY source.period_end_date
            )
        ),
        member_features AS (
            SELECT
                *,
                CASE WHEN adjusted_close > 0 AND close_21d_ago > 0
                     THEN (adjusted_close / close_21d_ago - 1.0) * 100.0 END
                    AS return_21d_pct,
                CASE WHEN adjusted_close > 0 AND close_63d_ago > 0
                     THEN (adjusted_close / close_63d_ago - 1.0) * 100.0 END
                    AS return_63d_pct,
                CASE WHEN adjusted_close > 0 AND close_126d_ago > 0
                     THEN (adjusted_close / close_126d_ago - 1.0) * 100.0 END
                    AS return_126d_pct,
                CASE WHEN adjusted_close > 0 AND close_252d_ago > 0
                     THEN (adjusted_close / close_252d_ago - 1.0) * 100.0 END
                    AS return_252d_pct,
                COALESCE(
                    trend_template_pass IS TRUE
                    AND prior_trend_template_pass IS FALSE,
                    FALSE
                ) AS is_new_8of8_signal
            FROM classified
        ),
        grouped AS (
            {aggregates}
        )
        SELECT * FROM grouped
        WHERE period_end_date >= %s
          AND (%s::date IS NULL OR period_end_date <= %s)
        ORDER BY period_end_date, 2, 3, 4, 5
        """
    ).format(
        source_table=_qualified_identifier(cfg.source_table),
        taxonomy_table=_qualified_identifier(cfg.security_master_current_table),
        aggregates=aggregates,
    )
    parameters = (cfg.signal_start_date, cfg.signal_end_date, cfg.signal_end_date)
    frames: list[pd.DataFrame] = []
    cursor_name = f"safr_taxonomy_group_{uuid4().hex[:16]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(statement, parameters)
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            frames.append(
                pd.DataFrame.from_records(
                    rows,
                    columns=CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS,
                )
            )
    if not frames:
        return pd.DataFrame(
            columns=CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS
        )
    frame = pd.concat(frames, ignore_index=True)
    frame["market_date"] = pd.to_datetime(
        frame["market_date"], errors="raise"
    ).dt.normalize()
    for column in (
        "taxonomy_level",
        "ibkr_industry",
        "ibkr_category",
        "ibkr_subcategory",
    ):
        frame[column] = frame[column].astype("string")
    numeric = tuple(
        column
        for column in CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS
        if column
        not in {
            "market_date",
            "taxonomy_level",
            "ibkr_industry",
            "ibkr_category",
            "ibkr_subcategory",
        }
    )
    _normalize_numeric_columns(frame, numeric, "current taxonomy group context")
    return frame.loc[:, CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS]


def _stock_group_rank_sql(
    level: str,
    group_columns: tuple[str, ...],
) -> sql.SQL:
    partition = sql.SQL(", ").join(
        (sql.Identifier("period_end_date"),)
        + tuple(sql.Identifier(column) for column in group_columns)
    )
    tie_partition = sql.SQL(", ").join(
        (sql.Identifier("period_end_date"),)
        + tuple(sql.Identifier(column) for column in group_columns)
        + (sql.Identifier("rs_raw"),)
    )
    nonnull = sql.SQL(" AND ").join(
        sql.SQL("{} IS NOT NULL").format(sql.Identifier(column))
        for column in group_columns
    )
    return sql.SQL(
        """
        CASE
            WHEN rs_raw IS NOT NULL AND {nonnull}
             AND count(rs_raw) OVER (PARTITION BY {partition}) >= %s
            THEN (
                rank() OVER (
                    PARTITION BY {partition} ORDER BY rs_raw NULLS LAST
                )
                + (
                    count(*) OVER (PARTITION BY {tie_partition}) - 1
                ) / 2.0
            ) / count(rs_raw) OVER (PARTITION BY {partition})
        END AS {alias}
        """
    ).format(
        nonnull=nonnull,
        partition=partition,
        tie_partition=tie_partition,
        alias=sql.Identifier(
            f"ibkr_taxbc_{level}_stock_rs_raw_pct_rank"
        ),
    )


def load_current_taxonomy_backcast_member_ranks(
    connection: extensions.connection, cfg: Config
) -> pd.DataFrame:
    """Return current-taxonomy stock-in-group RS ranks for signal rows only."""

    ranks = sql.SQL(", ").join(
        (
            _stock_group_rank_sql("industry", ("ibkr_industry",)),
            _stock_group_rank_sql(
                "category", ("ibkr_industry", "ibkr_category")
            ),
            _stock_group_rank_sql(
                "subcategory",
                ("ibkr_industry", "ibkr_category", "ibkr_subcategory"),
            ),
        )
    )
    statement = sql.SQL(
        """
        WITH classified AS (
            SELECT
                source.period_end_date,
                source.symbol,
                source.exchange,
                source.cik,
                source.price_continuity_segment,
                source.rs_raw,
                source.trend_template_pass,
                lag(source.trend_template_pass, 1)
                    OVER identity_segment AS prior_trend_template_pass,
                NULLIF(TRIM(taxonomy.ibkr_industry), '') AS ibkr_industry,
                NULLIF(TRIM(taxonomy.ibkr_category), '') AS ibkr_category,
                NULLIF(TRIM(taxonomy.ibkr_subcategory), '') AS ibkr_subcategory
            FROM {source_table} AS source
            JOIN {taxonomy_table} AS taxonomy
              ON taxonomy.symbol = source.symbol
             AND taxonomy.exchange = source.exchange
             AND taxonomy.cik = source.cik
            WINDOW identity_segment AS (
                PARTITION BY source.symbol, source.exchange, source.cik,
                             source.price_continuity_segment
                ORDER BY source.period_end_date
            )
        ),
        ranked AS (
            SELECT classified.*, {ranks}
            FROM classified
        )
        SELECT period_end_date, symbol, exchange, cik,
               ibkr_taxbc_industry_stock_rs_raw_pct_rank,
               ibkr_taxbc_category_stock_rs_raw_pct_rank,
               ibkr_taxbc_subcategory_stock_rs_raw_pct_rank
        FROM ranked
        WHERE trend_template_pass IS TRUE
          AND prior_trend_template_pass IS FALSE
          AND period_end_date >= %s
          AND (%s::date IS NULL OR period_end_date <= %s)
        ORDER BY period_end_date, symbol, exchange, cik
        """
    ).format(
        source_table=_qualified_identifier(cfg.source_table),
        taxonomy_table=_qualified_identifier(cfg.security_master_current_table),
        ranks=ranks,
    )
    parameters = (
        cfg.taxonomy_backcast_industry_min_members,
        cfg.taxonomy_backcast_category_min_members,
        cfg.taxonomy_backcast_subcategory_min_members,
        cfg.signal_start_date,
        cfg.signal_end_date,
        cfg.signal_end_date,
    )
    frames: list[pd.DataFrame] = []
    cursor_name = f"safr_taxonomy_rank_{uuid4().hex[:16]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(statement, parameters)
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            frames.append(
                pd.DataFrame.from_records(
                    rows,
                    columns=CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS,
                )
            )
    if not frames:
        return pd.DataFrame(columns=CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS)
    frame = pd.concat(frames, ignore_index=True)
    frame["signal_date"] = pd.to_datetime(
        frame["signal_date"], errors="raise"
    ).dt.normalize()
    _normalize_required_text(frame, ("symbol", "exchange"), "taxonomy ranks")
    _normalize_cik(frame, "taxonomy ranks")
    _normalize_numeric_columns(
        frame,
        CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS[4:],
        "taxonomy ranks",
    )
    if frame.duplicated(["signal_date", *IDENTITY_COLUMNS]).any():
        raise RuntimeError("current taxonomy member ranks contain duplicate signals")
    return frame.loc[:, CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS]


def load_world_market_observations(
    connection: extensions.connection, cfg: Config
) -> pd.DataFrame:
    """Load only pre-approved non-revision market/index series from the DB."""

    selected = (
        ("twelve_data", "SPY"),
        ("twelve_data", "QQQ"),
        ("twelve_data", "IWM"),
        ("twelve_data", "DIA"),
        ("cboe_vix", "VIX"),
        ("cboe_vix", "VXN"),
        ("cboe_vix", "VVIX"),
        ("cboe_vix", "SKEW"),
        ("cboe_vix", "VIX9D"),
        ("cboe_vix", "VIX3M"),
    )
    sources = [item[0] for item in selected]
    series_ids = [item[1] for item in selected]
    selected_columns = sql.SQL(", ").join(
        sql.SQL("source.{}").format(sql.Identifier(column))
        for column in WORLD_MARKET_SOURCE_COLUMNS
    )
    statement = sql.SQL(
        "SELECT {} FROM {} AS source "
        "JOIN unnest(%s::text[], %s::text[]) AS selected(source, series_id) "
        "ON source.source = selected.source "
        "AND source.series_id = selected.series_id "
        "WHERE source.is_revision_prone = false "
        "AND source.is_final = true "
        "AND source.observation_time >= %s::date - INTERVAL '400 days' "
        "AND (%s::date IS NULL OR source.observation_time < %s::date + INTERVAL '2 days') "
        "ORDER BY source.source, source.series_id, source.asof_known_at, "
        "source.observation_time"
    ).format(
        selected_columns,
        _qualified_identifier(cfg.world_market_observation_table),
    )
    frames: list[pd.DataFrame] = []
    cursor_name = f"safr_world_market_{uuid4().hex[:20]}"
    with connection.cursor(name=cursor_name) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(
            statement,
            (
                sources,
                series_ids,
                cfg.signal_start_date,
                cfg.signal_end_date,
                cfg.signal_end_date,
            ),
        )
        while True:
            rows = cursor.fetchmany(cfg.db_fetch_batch_size)
            if not rows:
                break
            frames.append(pd.DataFrame.from_records(rows, columns=WORLD_MARKET_SOURCE_COLUMNS))
    if not frames:
        return pd.DataFrame(columns=WORLD_MARKET_SOURCE_COLUMNS)
    frame = pd.concat(frames, ignore_index=True).loc[:, WORLD_MARKET_SOURCE_COLUMNS]
    for column in ("source", "series_id"):
        if frame[column].isna().any():
            raise RuntimeError(f"world market source column {column} contains null")
        frame[column] = frame[column].astype("string")
    for column in ("observation_time", "available_at", "asof_known_at"):
        frame[column] = pd.to_datetime(frame[column], errors="raise", utc=True)
    frame["source_local_date"] = pd.to_datetime(
        frame["source_local_date"], errors="raise"
    ).dt.normalize()
    _normalize_numeric_columns(frame, ("value",), "world market source")
    for column in ("is_revision_prone", "is_final"):
        frame[column] = frame[column].astype("boolean")
    return frame


def _normalize_boolean_value(value: Any, column: str) -> Any:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral) and not isinstance(value, bool) and value in (0, 1):
        return bool(value)
    raise ValueError(f"COPY boolean column {column} contains {value!r}")


def _copy_csv_batch(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    integer_columns: Sequence[str] = (),
    boolean_columns: Sequence[str] = (),
) -> str:
    # COPY does not use the DataFrame index. A fresh positional index also
    # prevents pandas extension arrays from aligning a sliced batch back to
    # labels from the full result frame.
    batch = frame.loc[:, columns].copy().reset_index(drop=True)

    for column in set(integer_columns).intersection(columns):
        original = batch[column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & numeric.isna()
        values = numeric.dropna().to_numpy(dtype=float)
        if invalid.any() or (
            values.size
            and (
                not np.isfinite(values).all()
                or not np.equal(values, np.floor(values)).all()
            )
        ):
            raise ValueError(
                f"COPY integer column {column} contains fractions or invalid values"
            )
        try:
            batch[column] = numeric.astype("Int64")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"COPY integer column {column} exceeds int64") from exc

    for column in set(boolean_columns).intersection(columns):
        batch[column] = pd.array(
            [_normalize_boolean_value(value, column) for value in batch[column].array],
            dtype="boolean",
        )

    # Do not assign a mixed numeric DataFrame back through ``.loc`` here.
    # pandas 3.0.x can route that assignment through nullable integer arrays
    # and perform label-based access with a positional np.int64, which raised
    # the production COPY error ``KeyError np.int64(...)``. Integer columns
    # were already validated above; normalize every remaining numeric column
    # independently through a plain NumPy array.
    integer_column_set = set(integer_columns)
    for column in batch.select_dtypes(include=[np.number]).columns:
        if column in integer_column_set:
            continue
        values = batch[column].to_numpy(
            dtype=float,
            na_value=np.nan,
            copy=True,
        )
        values[~np.isfinite(values)] = np.nan
        batch[column] = values

    # PostgreSQL text values cannot contain NUL. An unquoted real text value
    # equal to COPY_NULL would also be interpreted silently as SQL NULL by
    # COPY CSV, so reject that ambiguity before opening the stream.
    for column in batch.select_dtypes(include=["object", "string", "category"]).columns:
        values = batch[column].dropna().astype(str)
        if values.str.contains("\x00", regex=False).any():
            raise ValueError(f"COPY text column {column} contains NUL")
        if values.eq(COPY_NULL).any():
            raise ValueError(
                f"COPY text column {column} equals reserved NULL token "
                f"{COPY_NULL!r}"
            )

    return batch.to_csv(
        None,
        columns=list(columns),
        header=False,
        index=False,
        na_rep=COPY_NULL,
        date_format="%Y-%m-%d",
        lineterminator="\n",
    )


class _CopyCsvStream:
    def __init__(
        self,
        frame: pd.DataFrame,
        columns: Sequence[str],
        batch_size: int,
        *,
        integer_columns: Sequence[str] = (),
        boolean_columns: Sequence[str] = (),
    ) -> None:
        if batch_size < 1:
            raise ValueError("COPY batch_size must be >= 1")
        self._frame = frame
        self._columns = tuple(columns)
        self._batch_size = batch_size
        self._integer_columns = tuple(integer_columns)
        self._boolean_columns = tuple(boolean_columns)
        self._next_row = 0
        self._buffer = ""

    def _fill_buffer(self) -> bool:
        if self._next_row >= len(self._frame):
            return False
        start = self._next_row
        end = min(self._next_row + self._batch_size, len(self._frame))
        try:
            self._buffer = _copy_csv_batch(
                self._frame.iloc[start:end],
                self._columns,
                integer_columns=self._integer_columns,
                boolean_columns=self._boolean_columns,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to serialize COPY rows {start} through {end - 1}"
            ) from exc
        self._next_row = end
        return True

    def read(self, size: int = -1) -> str:
        if size == 0:
            return ""
        parts: list[str] = []
        remaining = size
        while remaining < 0 or remaining > 0:
            if not self._buffer and not self._fill_buffer():
                break
            if remaining < 0:
                parts.append(self._buffer)
                self._buffer = ""
                continue
            take = min(remaining, len(self._buffer))
            parts.append(self._buffer[:take])
            self._buffer = self._buffer[take:]
            remaining -= take
        return "".join(parts)


def _validate_result_frame(
    frame: pd.DataFrame, columns: Sequence[str], name: str
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} frame is missing columns: " + ", ".join(missing))


def _copy_frame(
    connection: extensions.connection,
    table_name: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
    batch_size: int,
    *,
    integer_columns: Sequence[str],
    boolean_columns: Sequence[str],
) -> None:
    if frame.empty:
        return
    statement = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT CSV, NULL {})").format(
        _qualified_identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        sql.Literal(COPY_NULL),
    )
    stream = _CopyCsvStream(
        frame,
        columns,
        batch_size,
        integer_columns=integer_columns,
        boolean_columns=boolean_columns,
    )
    with connection.cursor() as cursor:
        cursor.copy_expert(statement.as_string(connection), stream, size=1024 * 1024)


def write_results_atomic(
    connection: extensions.connection,
    cfg: Config,
    signals: pd.DataFrame,
    early_cuts: pd.DataFrame,
    rules: pd.DataFrame,
) -> tuple[int, int, int]:
    """COPY all three result sets in one new read/write transaction."""

    _validate_result_frame(signals, SIGNAL_COLUMNS, "signal result")
    _validate_result_frame(early_cuts, EARLY_CUT_COLUMNS, "early-cut result")
    _validate_result_frame(rules, RULE_COLUMNS, "rule result")
    if (
        not signals.empty
        and signals.duplicated(list(EXPECTED_SIGNAL_PRIMARY_KEY)).any()
    ):
        raise ValueError("signal result frame contains duplicate primary keys")
    if (
        not early_cuts.empty
        and early_cuts.duplicated(list(EXPECTED_EARLY_CUT_PRIMARY_KEY)).any()
    ):
        raise ValueError("early-cut result frame contains duplicate primary keys")

    if _transaction_status(connection) != extensions.TRANSACTION_STATUS_IDLE:
        raise RuntimeError(
            "result COPY requires an idle connection; commit the exported "
            "source snapshot first"
        )

    try:
        with connection.cursor() as cursor:
            cursor.execute("BEGIN TRANSACTION READ WRITE")
        # Recheck after the long worker phase and inside the same transaction as
        # all three COPY statements. All program instances cooperate via the session
        # advisory lock, so this also closes the rebuild race.
        assert_targets_empty(connection, cfg)
        _copy_frame(
            connection,
            cfg.signal_result_table,
            signals,
            SIGNAL_COLUMNS,
            cfg.db_copy_batch_size,
            integer_columns=SIGNAL_INTEGER_COLUMNS,
            boolean_columns=SIGNAL_BOOLEAN_COLUMNS,
        )
        _copy_frame(
            connection,
            cfg.early_cut_result_table,
            early_cuts,
            EARLY_CUT_COLUMNS,
            cfg.db_copy_batch_size,
            integer_columns=EARLY_CUT_INTEGER_COLUMNS,
            boolean_columns=EARLY_CUT_BOOLEAN_COLUMNS,
        )
        _copy_frame(
            connection,
            cfg.rule_result_table,
            rules,
            RULE_COLUMNS,
            cfg.db_copy_batch_size,
            integer_columns=RULE_INTEGER_COLUMNS,
            boolean_columns=RULE_BOOLEAN_COLUMNS,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return len(signals), len(early_cuts), len(rules)
