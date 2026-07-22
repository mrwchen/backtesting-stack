from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    CRITERION_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_CHANGE_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_FEATURE_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_INTEGER_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_LEVEL_SPECS,
    CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS,
    CURRENT_TAXONOMY_BACKCAST_METRIC_SUFFIXES,
    CURRENT_TAXONOMY_BACKCAST_UNIT_INTERVAL_COLUMNS,
    CURRENT_TAXONOMY_SOURCE_COLUMNS,
    EARLY_CUT_COLUMNS,
    EARLY_CUT_BOOLEAN_COLUMNS,
    EARLY_CUT_INTEGER_COLUMNS,
    EARLY_CUT_LANDMARK_DAYS,
    EARLY_CONFIRMATION_RELATIVE_FEATURE_COLUMNS,
    MANAGEMENT_LANDMARK_DAYS,
    POSITION_LANDMARK_DAYS,
    EARLY_GLOBAL_MARKET_FEATURE_COLUMNS,
    EARNINGS_EVENT_FEATURE_COLUMNS,
    EARNINGS_EVENT_SOURCE_COLUMNS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
    IDENTITY_COLUMNS,
    MARKET_CAP_FEATURE_COLUMNS,
    MARKET_METRIC_SOURCE_COLUMNS,
    NOTIONAL_SOFT_PATTERN_BASE_NAMES,
    NOTIONAL_SOFT_PATTERN_SPECS,
    QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
    SIGNAL_COLUMNS,
    SIGNAL_BOOLEAN_COLUMNS,
    SIGNAL_INTEGER_COLUMNS,
    SOFT_PATTERN_FEATURE_COLUMNS,
    SOFT_PATTERN_SPECS,
    SUPPLY_DEMAND_FEATURE_COLUMNS,
    TECHNICAL_V3_FEATURE_COLUMNS,
    GLOBAL_MARKET_FEATURE_COLUMNS,
    WORLD_MARKET_SOURCE_COLUMNS,
    SOURCE_BOOLEAN_COLUMNS,
    SOURCE_COLUMNS,
)


_NEW_YORK = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class CalculationBatchResult:
    """One bounded worker result, before any research rules are fitted."""

    signals: pd.DataFrame
    early_cut: pd.DataFrame

    @property
    def early_cut_results(self) -> pd.DataFrame:
        """Descriptive alias for callers that prefer the table-oriented name."""

        return self.early_cut


def empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def empty_early_cut_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EARLY_CUT_COLUMNS)


def empty_calculation_batch() -> CalculationBatchResult:
    return CalculationBatchResult(
        signals=empty_signal_frame(),
        early_cut=empty_early_cut_frame(),
    )


_SIGNAL_TEXT_COLUMNS = (
    "symbol",
    "exchange",
    "currency",
    "trigger_criteria",
    "gain_loss_order_5d",
    "analysis_split",
    "weak_matched_rule_ids",
    "loss_first_matched_rule_ids",
    "matched_rule_ids",
    "confirmation_matched_rule_ids",
    "confirmation_reason",
    "filter_decision",
    "exclusion_reason",
    *CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
)

_EARLY_CUT_TEXT_COLUMNS = (
    "symbol",
    "exchange",
    "currency",
    "decision_stage",
    "continuation_outcome",
    "early_gain_loss_order_to_day20",
    "analysis_split",
    "stagnation_matched_rule_ids",
    "loss_matched_rule_ids",
    "matched_rule_ids",
    "cut_decision",
    "cut_reason",
    "early_confirmation_matched_rule_ids",
    "early_confirmation_decision",
    "early_confirmation_reason",
    "management_matched_rule_ids",
    "management_decision",
    "management_reason",
)

_SIGNAL_DATE_COLUMNS = (
    "signal_date",
    "previous_session_date",
    *(f"forward_{day}d_label_end_date" for day in (5, 10, 20, 30, 40, 60, 90)),
)

_EARLY_CUT_DATE_COLUMNS = (
    "signal_date",
    "landmark_date",
    "effective_session_date",
    "horizon_end_date",
    "day20_end_date",
    "day40_end_date",
    "day60_end_date",
    "day90_end_date",
)


def _normalize_result_dtypes(
    frame: pd.DataFrame,
    *,
    integer_columns: tuple[str, ...],
    boolean_columns: tuple[str, ...],
    text_columns: tuple[str, ...],
    date_columns: tuple[str, ...],
) -> pd.DataFrame:
    """Make worker partitions concatenate without null-sentinel drift."""

    result = frame.copy()
    for column in integer_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(
            "Int64"
        )
    for column in boolean_columns:
        result[column] = result[column].astype("boolean")
    for column in text_columns:
        result[column] = result[column].astype("string")
    for column in date_columns:
        result[column] = pd.to_datetime(result[column], errors="raise")
    typed = {
        *integer_columns,
        *boolean_columns,
        *text_columns,
        *date_columns,
    }
    for column in result.columns:
        if column not in typed:
            result[column] = pd.to_numeric(
                result[column], errors="raise"
            ).astype(float)
    return result


def _signal_decision_timestamp(signal_date: Any) -> pd.Timestamp:
    """Return the causal information boundary: 16:00 America/New_York."""

    normalized = pd.Timestamp(signal_date)
    if pd.isna(normalized):
        raise ValueError("signal date must not be null")
    local_close = datetime.combine(
        normalized.date(), time(hour=16), tzinfo=_NEW_YORK
    )
    return pd.Timestamp(local_close.astimezone(_UTC))


def _require_columns(
    frame: pd.DataFrame, required: tuple[str, ...], source_name: str
) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing columns: " + ", ".join(missing))


def _normalize_fundamental_identities(
    frame: pd.DataFrame, source_name: str
) -> pd.DataFrame:
    result = frame.copy()
    if result.loc[:, list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError(f"{source_name} identity columns must not be null")
    result["symbol"] = result["symbol"].astype("string")
    result["exchange"] = result["exchange"].astype("string")
    cik = pd.to_numeric(result["cik"], errors="coerce")
    finite = cik.dropna().to_numpy(dtype=float)
    if (
        cik.isna().any()
        or (finite.size and not np.isfinite(finite).all())
        or (finite.size and not np.equal(finite, np.floor(finite)).all())
    ):
        raise ValueError(f"{source_name} cik values must be integers")
    result["cik"] = cik.astype("Int64")
    return result


def _normalize_optional_currency_values(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip().str.upper()
    return values.mask(values.eq(""), pd.NA)


def _normalize_fundamental_numerics(
    frame: pd.DataFrame, columns: tuple[str, ...], source_name: str
) -> None:
    for column in columns:
        original = frame[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"{source_name} column {column} is not numeric")
        frame[column] = numeric.astype(float).replace([np.inf, -np.inf], np.nan)


def _prepare_fundamental_snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "fundamental snapshot data"
    _require_columns(frame, FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS, source_name)
    if frame.empty:
        return pd.DataFrame(columns=FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS)
    result = _normalize_fundamental_identities(
        frame.loc[:, FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS], source_name
    )
    result["period_end_date"] = pd.to_datetime(
        result["period_end_date"], errors="raise"
    ).dt.normalize()
    result["sec_latest_period_end_date"] = pd.to_datetime(
        result["sec_latest_period_end_date"], errors="raise"
    ).dt.normalize()
    result["sec_data_available_at"] = pd.to_datetime(
        result["sec_data_available_at"], errors="raise", utc=True
    )
    result["sec_fundamental_currency"] = _normalize_optional_currency_values(
        result["sec_fundamental_currency"]
    )
    numeric_columns = tuple(
        column
        for column in FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS
        if column
        not in {
            *IDENTITY_COLUMNS,
            "period_end_date",
            "sec_latest_period_end_date",
            "sec_data_available_at",
            "sec_fundamental_currency",
        }
    )
    _normalize_fundamental_numerics(result, numeric_columns, source_name)
    if result.duplicated([*IDENTITY_COLUMNS, "period_end_date"]).any():
        raise ValueError("fundamental snapshot data contains duplicate identity/date rows")
    return result


def _prepare_quarterly_fundamental_events(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "quarterly fundamental event data"
    _require_columns(frame, QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS, source_name)
    if frame.empty:
        return pd.DataFrame(columns=QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS)
    result = _normalize_fundamental_identities(
        frame.loc[:, QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS], source_name
    )
    if result["accession_number"].isna().any():
        raise ValueError("quarterly fundamental event accession number must not be null")
    result["accession_number"] = result["accession_number"].astype("string")
    for column in ("effective_date", "fiscal_period_end_date"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.normalize()
    result["accepted_at"] = pd.to_datetime(
        result["accepted_at"], errors="raise", utc=True
    )
    result["currency"] = _normalize_optional_currency_values(result["currency"])
    numeric_columns = tuple(
        column
        for column in QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS
        if column
        not in {
            *IDENTITY_COLUMNS,
            "accession_number",
            "accepted_at",
            "effective_date",
            "fiscal_period_end_date",
            "currency",
        }
    )
    _normalize_fundamental_numerics(result, numeric_columns, source_name)
    if result.duplicated([*IDENTITY_COLUMNS, "accession_number"]).any():
        raise ValueError(
            "quarterly fundamental event data contains duplicate identity/accession rows"
        )
    return result


def _prepare_earnings_events(frame: pd.DataFrame) -> pd.DataFrame:
    source_name = "earnings event data"
    _require_columns(frame, EARNINGS_EVENT_SOURCE_COLUMNS, source_name)
    if frame.empty:
        return pd.DataFrame(columns=EARNINGS_EVENT_SOURCE_COLUMNS)
    result = _normalize_fundamental_identities(
        frame.loc[:, EARNINGS_EVENT_SOURCE_COLUMNS], source_name
    )
    for column in (
        "source",
        "source_event_id",
    ):
        if result[column].isna().any():
            raise ValueError(f"earnings event {column} must not be null")
        result[column] = result[column].astype("string")
    result["announcement_time_type"] = result[
        "announcement_time_type"
    ].astype("string")
    result["earnings_date"] = pd.to_datetime(
        result["earnings_date"], errors="raise"
    ).dt.normalize()
    result["announcement_ts"] = pd.to_datetime(
        result["announcement_ts"], errors="coerce", utc=True
    )
    result["known_as_of_ts"] = pd.to_datetime(
        result["known_as_of_ts"], errors="raise", utc=True
    )
    result["is_confirmed"] = result["is_confirmed"].astype("boolean")
    if result.duplicated(
        [*IDENTITY_COLUMNS, "source", "source_event_id"]
    ).any():
        raise ValueError("earnings event data contains duplicate source events")
    return result


def _identity_groups(frame: pd.DataFrame) -> dict[tuple[str, str, int], pd.DataFrame]:
    return {
        (str(symbol), str(exchange), int(cik)): group.reset_index(drop=True)
        for (symbol, exchange, cik), group in frame.groupby(
            list(IDENTITY_COLUMNS), sort=False, observed=True
        )
    }


def _finite_scalar(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return np.nan
    return numeric if np.isfinite(numeric) else np.nan


def _safe_scalar_divide(
    numerator: Any,
    denominator: Any,
    *,
    positive_denominator: bool = True,
) -> float:
    numerator_value = _finite_scalar(numerator)
    denominator_value = _finite_scalar(denominator)
    denominator_valid = (
        denominator_value > 0 if positive_denominator else denominator_value != 0
    )
    if not np.isfinite(numerator_value) or not denominator_valid:
        return np.nan
    return float(numerator_value / denominator_value)


def _verified_currency(value: Any) -> bool:
    return bool(pd.notna(value) and str(value).strip())


def _assign_snapshot_features(
    output: pd.DataFrame,
    output_index: Any,
    signal_date: pd.Timestamp,
    decision_at: pd.Timestamp,
    rows: pd.DataFrame,
) -> None:
    eligible = rows.loc[
        rows["period_end_date"].le(signal_date)
        & rows["sec_latest_period_end_date"].notna()
        & rows["sec_latest_period_end_date"].le(signal_date)
        & rows["sec_data_available_at"].notna()
        & rows["sec_data_available_at"].le(decision_at)
    ]
    if eligible.empty:
        return
    selected = eligible.sort_values(
        [
            "sec_latest_period_end_date",
            "sec_data_available_at",
            "period_end_date",
        ],
        kind="mergesort",
    ).iloc[-1]
    available_at = pd.Timestamp(selected["sec_data_available_at"])
    report_date = pd.Timestamp(selected["sec_latest_period_end_date"])
    output.at[output_index, "fundamental_snapshot_age_days"] = (
        decision_at - available_at
    ).total_seconds() / 86_400.0
    output.at[output_index, "fundamental_report_age_days"] = float(
        (signal_date - report_date).days
    )
    if not _verified_currency(selected["sec_fundamental_currency"]):
        return

    direct = {
        "fundamental_gross_margin_ttm_ratio": "sec_gross_margin_ttm",
        "fundamental_operating_margin_ttm_ratio": "sec_operating_margin_ttm",
        "fundamental_net_margin_ttm_ratio": "sec_net_margin_ttm",
        "fundamental_fcf_margin_ttm_ratio": "sec_fcf_margin_ttm",
        "fundamental_fcf_sbc_adjusted_margin_ttm_ratio": (
            "sec_fcf_sbc_adjusted_margin_ttm"
        ),
        "fundamental_debt_to_capital_ratio": "sec_debt_to_capital",
        "fundamental_cash_to_assets_ratio": "sec_cash_to_assets",
        "fundamental_current_ratio": "sec_current_ratio",
        "fundamental_accruals_ratio": "sec_accruals_ratio",
    }
    for target, source in direct.items():
        output.at[output_index, target] = _finite_scalar(selected[source])
    revenue = _finite_scalar(selected["sec_revenue_ttm"])
    sbc = _finite_scalar(selected["sec_share_based_compensation_ttm"])
    if np.isfinite(revenue) and revenue > 0 and np.isfinite(sbc):
        output.at[output_index, "fundamental_sbc_to_revenue_ttm_ratio"] = (
            sbc / revenue
        )

    direct_ratios = {
        "fundamental_operating_cashflow_margin_ttm_ratio": (
            "sec_operating_cashflow_ttm",
            "sec_revenue_ttm",
        ),
        "fundamental_rd_to_revenue_ttm_ratio": (
            "sec_research_and_development_ttm",
            "sec_revenue_ttm",
        ),
        "fundamental_sga_to_revenue_ttm_ratio": (
            "sec_selling_general_and_admin_ttm",
            "sec_revenue_ttm",
        ),
        "fundamental_capex_to_revenue_ttm_ratio": (
            "sec_capex_ttm",
            "sec_revenue_ttm",
        ),
        "fundamental_da_to_revenue_ttm_ratio": (
            "sec_depreciation_and_amortization_ttm",
            "sec_revenue_ttm",
        ),
        "fundamental_roe_ttm_ratio": (
            "sec_net_income_ttm",
            "sec_stockholders_equity",
        ),
        "fundamental_roa_ttm_ratio": ("sec_net_income_ttm", "sec_assets"),
        "fundamental_cash_conversion_ttm_ratio": (
            "sec_operating_cashflow_ttm",
            "sec_net_income_ttm",
        ),
        "fundamental_fcf_conversion_ttm_ratio": (
            "sec_free_cashflow_ttm",
            "sec_net_income_ttm",
        ),
        "fundamental_fcf_sbc_adjusted_conversion_ttm_ratio": (
            "sec_free_cashflow_sbc_adjusted_ttm",
            "sec_net_income_ttm",
        ),
        "fundamental_interest_coverage_ttm_ratio": (
            "sec_operating_income_ttm",
            "sec_interest_expense_ttm",
        ),
        "fundamental_debt_to_assets_ratio": ("sec_total_debt", "sec_assets"),
        "fundamental_inventory_to_revenue_ttm_ratio": (
            "sec_inventory",
            "sec_revenue_ttm",
        ),
        "fundamental_receivables_to_revenue_ttm_ratio": (
            "sec_accounts_receivable",
            "sec_revenue_ttm",
        ),
        "fundamental_payables_to_revenue_ttm_ratio": (
            "sec_accounts_payable",
            "sec_revenue_ttm",
        ),
        "fundamental_asset_turnover_ttm_ratio": ("sec_revenue_ttm", "sec_assets"),
        "fundamental_diluted_share_pressure_ratio": (
            "sec_weighted_avg_shares_diluted",
            "sec_shares_outstanding",
        ),
        "fundamental_buyback_to_revenue_ttm_ratio": (
            "sec_common_stock_repurchased_ttm",
            "sec_revenue_ttm",
        ),
    }
    for target, (numerator, denominator) in direct_ratios.items():
        output.at[output_index, target] = _safe_scalar_divide(
            selected[numerator], selected[denominator]
        )
    diluted_share_ratio = _finite_scalar(
        output.at[output_index, "fundamental_diluted_share_pressure_ratio"]
    )
    if np.isfinite(diluted_share_ratio):
        output.at[
            output_index, "fundamental_diluted_share_pressure_ratio"
        ] = diluted_share_ratio - 1.0

    assets = _finite_scalar(selected["sec_assets"])
    debt = _finite_scalar(selected["sec_total_debt"])
    cash = _finite_scalar(selected["sec_cash_and_equivalents"])
    net_debt = debt - cash if np.isfinite(debt) and np.isfinite(cash) else np.nan
    if np.isfinite(net_debt) and np.isfinite(assets) and assets > 0:
        output.at[output_index, "fundamental_net_debt_to_assets_ratio"] = (
            net_debt / assets
        )
    operating_income = _finite_scalar(selected["sec_operating_income_ttm"])
    if np.isfinite(debt) and np.isfinite(operating_income) and operating_income > 0:
        output.at[
            output_index, "fundamental_debt_to_operating_income_ttm_ratio"
        ] = debt / operating_income
    fcf = _finite_scalar(selected["sec_free_cashflow_ttm"])
    if np.isfinite(net_debt) and np.isfinite(fcf) and fcf > 0:
        output.at[output_index, "fundamental_net_debt_to_fcf_ttm_ratio"] = (
            net_debt / fcf
        )
    current_assets = _finite_scalar(selected["sec_current_assets"])
    current_liabilities = _finite_scalar(selected["sec_current_liabilities"])
    inventory = _finite_scalar(selected["sec_inventory"])
    if np.isfinite(current_liabilities) and current_liabilities > 0:
        if np.isfinite(current_assets) and np.isfinite(inventory):
            output.at[output_index, "fundamental_quick_ratio"] = (
                current_assets - inventory
            ) / current_liabilities
    if np.isfinite(assets) and assets > 0:
        if np.isfinite(current_assets) and np.isfinite(current_liabilities):
            output.at[
                output_index, "fundamental_working_capital_to_assets_ratio"
            ] = (current_assets - current_liabilities) / assets
        goodwill = _finite_scalar(selected["sec_goodwill"])
        intangibles = _finite_scalar(selected["sec_intangible_assets"])
        if np.isfinite(goodwill) and np.isfinite(intangibles):
            output.at[
                output_index,
                "fundamental_goodwill_intangibles_to_assets_ratio",
            ] = (goodwill + intangibles) / assets

    selected_shares = _finite_scalar(selected["sec_shares_outstanding"])
    older = eligible.loc[
        eligible["period_end_date"].le(signal_date - pd.Timedelta(days=365))
        & eligible["sec_shares_outstanding"].notna()
    ]
    if not older.empty and np.isfinite(selected_shares) and selected_shares > 0:
        older_row = older.sort_values(
            ["period_end_date", "sec_data_available_at"], kind="mergesort"
        ).iloc[-1]
        older_shares = _finite_scalar(older_row["sec_shares_outstanding"])
        if np.isfinite(older_shares) and older_shares > 0:
            output.at[
                output_index, "fundamental_sec_shares_change_1y_ratio"
            ] = selected_shares / older_shares - 1.0


def _assign_quarterly_features(
    output: pd.DataFrame,
    output_index: Any,
    signal_date: pd.Timestamp,
    decision_at: pd.Timestamp,
    rows: pd.DataFrame,
) -> None:
    eligible = rows.loc[
        rows["effective_date"].le(signal_date)
        & rows["fiscal_period_end_date"].le(signal_date)
        & rows["accepted_at"].notna()
        & rows["accepted_at"].le(decision_at)
    ]
    if eligible.empty:
        return
    ordered = eligible.sort_values(
        ["fiscal_period_end_date", "accepted_at", "accession_number"],
        kind="mergesort",
    ).drop_duplicates("fiscal_period_end_date", keep="last")
    selected = ordered.iloc[-1]
    accepted_at = pd.Timestamp(selected["accepted_at"])
    quarter_date = pd.Timestamp(selected["fiscal_period_end_date"])
    output.at[output_index, "fundamental_quarter_filing_age_days"] = (
        decision_at - accepted_at
    ).total_seconds() / 86_400.0
    output.at[output_index, "fundamental_quarter_age_days"] = float(
        (signal_date - quarter_date).days
    )
    if not _verified_currency(selected["currency"]):
        return

    current_revenue = _finite_scalar(selected["quarterly_revenue"])
    prior_revenue = _finite_scalar(selected["prior_year_quarterly_revenue"])
    if np.isfinite(current_revenue) and np.isfinite(prior_revenue) and prior_revenue != 0:
        output.at[
            output_index, "fundamental_quarterly_revenue_yoy_growth_ratio"
        ] = (current_revenue - prior_revenue) / abs(prior_revenue)

    current_eps = _finite_scalar(selected["diluted_eps"])
    prior_eps = _finite_scalar(selected["prior_year_diluted_eps"])
    eps_scale = abs(current_eps) + abs(prior_eps)
    if np.isfinite(eps_scale) and eps_scale > 0:
        output.at[output_index, "fundamental_quarterly_eps_yoy_change_ratio"] = (
            current_eps - prior_eps
        ) / eps_scale
    if np.isfinite(current_eps) and np.isfinite(prior_eps):
        if prior_eps > 0:
            output.at[
                output_index, "fundamental_quarterly_eps_yoy_growth_ratio"
            ] = current_eps / prior_eps - 1.0
        output.at[output_index, "fundamental_quarterly_loss_to_profit"] = float(
            prior_eps <= 0 < current_eps
        )

    for target, change_target, current_column, prior_column in (
        (
            "fundamental_quarterly_operating_margin_ratio",
            "fundamental_quarterly_operating_margin_yoy_change",
            "quarterly_operating_margin",
            "prior_year_quarterly_operating_margin",
        ),
        (
            "fundamental_quarterly_net_margin_ratio",
            "fundamental_quarterly_net_margin_yoy_change",
            "quarterly_net_margin",
            "prior_year_quarterly_net_margin",
        ),
    ):
        current = _finite_scalar(selected[current_column])
        prior = _finite_scalar(selected[prior_column])
        output.at[output_index, target] = current
        if np.isfinite(current) and np.isfinite(prior):
            output.at[output_index, change_target] = current - prior

    recent = ordered.tail(5).copy()
    if len(recent) >= 2:
        revenue_growth = (
            (
                _numeric(recent["quarterly_revenue"])
                - _numeric(recent["prior_year_quarterly_revenue"])
            )
            / _numeric(recent["prior_year_quarterly_revenue"]).abs()
        ).where(_numeric(recent["prior_year_quarterly_revenue"]).ne(0))
        eps_growth = (
            _numeric(recent["diluted_eps"])
            / _numeric(recent["prior_year_diluted_eps"])
            - 1.0
        ).where(_numeric(recent["prior_year_diluted_eps"]).gt(0))
        latest_revenue_growth = _finite_scalar(revenue_growth.iloc[-1])
        previous_revenue_growth = _finite_scalar(revenue_growth.iloc[-2])
        if np.isfinite(latest_revenue_growth) and np.isfinite(
            previous_revenue_growth
        ):
            output.at[
                output_index, "fundamental_quarterly_revenue_growth_acceleration"
            ] = latest_revenue_growth - previous_revenue_growth
        latest_eps_growth = _finite_scalar(eps_growth.iloc[-1])
        previous_eps_growth = _finite_scalar(eps_growth.iloc[-2])
        if np.isfinite(latest_eps_growth) and np.isfinite(previous_eps_growth):
            output.at[
                output_index, "fundamental_quarterly_eps_growth_acceleration"
            ] = latest_eps_growth - previous_eps_growth

        previous_quarter = recent.iloc[-2]
        previous_quarter_revenue = _finite_scalar(
            previous_quarter["quarterly_revenue"]
        )
        if (
            np.isfinite(current_revenue)
            and np.isfinite(previous_quarter_revenue)
            and previous_quarter_revenue != 0
        ):
            output.at[
                output_index,
                "fundamental_quarterly_revenue_sequential_growth_ratio",
            ] = (current_revenue - previous_quarter_revenue) / abs(
                previous_quarter_revenue
            )
        previous_quarter_eps = _finite_scalar(previous_quarter["diluted_eps"])
        sequential_eps_scale = abs(current_eps) + abs(previous_quarter_eps)
        if np.isfinite(sequential_eps_scale) and sequential_eps_scale > 0:
            output.at[
                output_index, "fundamental_quarterly_eps_sequential_change_ratio"
            ] = (current_eps - previous_quarter_eps) / sequential_eps_scale

        for target, values in (
            (
                "fundamental_quarterly_operating_margin_acceleration",
                _numeric(recent["quarterly_operating_margin"])
                - _numeric(recent["prior_year_quarterly_operating_margin"]),
            ),
            (
                "fundamental_quarterly_net_margin_acceleration",
                _numeric(recent["quarterly_net_margin"])
                - _numeric(recent["prior_year_quarterly_net_margin"]),
            ),
        ):
            latest = _finite_scalar(values.iloc[-1])
            previous = _finite_scalar(values.iloc[-2])
            if np.isfinite(latest) and np.isfinite(previous):
                output.at[output_index, target] = latest - previous

        revenue_streak_values = revenue_growth.tail(4)
        if len(revenue_streak_values) == 4 and revenue_streak_values.notna().all():
            output.at[output_index, "fundamental_revenue_growth_streak_4q"] = float(
                revenue_streak_values.gt(0).sum()
            )
        eps_streak_values = eps_growth.tail(4)
        if len(eps_streak_values) == 4 and eps_streak_values.notna().all():
            output.at[output_index, "fundamental_eps_growth_streak_4q"] = float(
                eps_streak_values.gt(0).sum()
            )


def _assign_earnings_event_features(
    output: pd.DataFrame,
    output_index: Any,
    signal_date: pd.Timestamp,
    decision_at: pd.Timestamp,
    rows: pd.DataFrame,
) -> None:
    # Historical yfinance events are current snapshots and therefore excluded.
    eligible = rows.loc[
        rows["source"].eq("sec_8k_item_2_02")
        & rows["is_confirmed"].eq(True).fillna(False)
        & rows["known_as_of_ts"].le(decision_at)
    ].copy()
    if eligible.empty:
        return
    eligible["effective_at"] = eligible["announcement_ts"].fillna(
        eligible["known_as_of_ts"]
    )
    eligible = eligible.loc[eligible["effective_at"].le(decision_at)]
    if eligible.empty:
        return
    selected = eligible.sort_values(
        ["effective_at", "known_as_of_ts", "source_event_id"], kind="mergesort"
    ).iloc[-1]
    effective_at = pd.Timestamp(selected["effective_at"])
    age_days = (decision_at - effective_at).total_seconds() / 86_400.0
    local_event_date = effective_at.tz_convert(_NEW_YORK).normalize().tz_localize(None)
    output.at[output_index, "earnings_event_age_days"] = age_days
    output.at[output_index, "earnings_event_on_signal_day"] = float(
        local_event_date == signal_date
    )
    output.at[output_index, "earnings_event_within_5d"] = float(age_days <= 5.0)
    output.at[output_index, "earnings_event_within_21d"] = float(age_days <= 21.0)


def enrich_signal_fundamentals(
    signals: pd.DataFrame,
    fundamental_snapshots: pd.DataFrame,
    quarterly_fundamental_events: pd.DataFrame,
    earnings_events: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach only SEC information public by the signal-day 16:00 ET boundary."""

    output = signals.copy()
    for column in FUNDAMENTAL_FEATURE_COLUMNS:
        output[column] = np.nan
    for column in EARNINGS_EVENT_FEATURE_COLUMNS:
        output[column] = np.nan
    if output.empty:
        return output.loc[:, SIGNAL_COLUMNS]

    snapshots = _prepare_fundamental_snapshots(fundamental_snapshots)
    events = _prepare_quarterly_fundamental_events(quarterly_fundamental_events)
    prepared_earnings = _prepare_earnings_events(
        pd.DataFrame(columns=EARNINGS_EVENT_SOURCE_COLUMNS)
        if earnings_events is None
        else earnings_events
    )
    snapshot_groups = _identity_groups(snapshots) if not snapshots.empty else {}
    event_groups = _identity_groups(events) if not events.empty else {}
    earnings_groups = (
        _identity_groups(prepared_earnings) if not prepared_earnings.empty else {}
    )

    for output_index, signal in output.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"]).normalize()
        decision_at = _signal_decision_timestamp(signal_date)
        identity = (
            str(signal["symbol"]),
            str(signal["exchange"]),
            int(signal["cik"]),
        )
        snapshot_rows = snapshot_groups.get(identity)
        if snapshot_rows is not None:
            _assign_snapshot_features(
                output, output_index, signal_date, decision_at, snapshot_rows
            )
        event_rows = event_groups.get(identity)
        if event_rows is not None:
            _assign_quarterly_features(
                output, output_index, signal_date, decision_at, event_rows
            )
        earnings_rows = earnings_groups.get(identity)
        if earnings_rows is not None:
            _assign_earnings_event_features(
                output, output_index, signal_date, decision_at, earnings_rows
            )
    return output.loc[:, SIGNAL_COLUMNS]


def enrich_signal_market_metrics(
    signals: pd.DataFrame,
    market_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Attach exact signal-close USD market cap without forward-filled rows."""

    output = signals.copy()
    for column in MARKET_CAP_FEATURE_COLUMNS:
        output[column] = np.nan
    for column in SUPPLY_DEMAND_FEATURE_COLUMNS:
        output[column] = np.nan
    if output.empty:
        return output.loc[:, SIGNAL_COLUMNS]

    source_name = "market metric data"
    _require_columns(market_metrics, MARKET_METRIC_SOURCE_COLUMNS, source_name)
    if market_metrics.empty:
        output["market_cap_usd"] = output["market_cap_usd"].astype("Int64")
        output["market_cap_shares_staleness_days"] = output[
            "market_cap_shares_staleness_days"
        ].astype("Int64")
        output["shares_outstanding"] = output["shares_outstanding"].astype("Int64")
        return output.loc[:, SIGNAL_COLUMNS]

    metrics = _normalize_fundamental_identities(
        market_metrics.loc[:, MARKET_METRIC_SOURCE_COLUMNS], source_name
    )
    metrics["period_end_date"] = pd.to_datetime(
        metrics["period_end_date"], errors="raise"
    ).dt.normalize()
    metrics["market_cap_currency"] = _normalize_optional_currency_values(
        metrics["market_cap_currency"]
    )
    metrics["shares_outstanding_source"] = metrics[
        "shares_outstanding_source"
    ].astype("string").str.strip().mask(lambda values: values.eq(""), pd.NA)
    _normalize_fundamental_numerics(
        metrics,
        (
            "market_cap",
            "adjusted_open",
            "raw_volume",
            "shares_outstanding",
            "shares_outstanding_staleness_days",
        ),
        source_name,
    )
    for column in (
        "market_cap",
        "shares_outstanding",
        "shares_outstanding_staleness_days",
    ):
        values = pd.to_numeric(metrics[column], errors="coerce").dropna()
        if not values.eq(np.floor(values)).all():
            raise ValueError(f"market metric data column {column} must be integral")
    if metrics.duplicated(["period_end_date", *IDENTITY_COLUMNS]).any():
        raise ValueError("market metric data contains duplicate identity/date rows")
    negative_staleness = pd.to_numeric(
        metrics["shares_outstanding_staleness_days"], errors="coerce"
    ).dropna().lt(0)
    if negative_staleness.any():
        raise ValueError("market metric share-count staleness must not be negative")

    lookup = {
        (
            pd.Timestamp(row.period_end_date).normalize(),
            str(row.symbol),
            str(row.exchange),
            int(row.cik),
        ): row
        for row in metrics.itertuples(index=False)
    }
    for output_index, signal in output.iterrows():
        key = (
            pd.Timestamp(signal["signal_date"]).normalize(),
            str(signal["symbol"]),
            str(signal["exchange"]),
            int(signal["cik"]),
        )
        metric = lookup.get(key)
        if metric is None:
            continue
        currency = metric.market_cap_currency
        market_cap = _finite_scalar(metric.market_cap)
        staleness = _finite_scalar(metric.shares_outstanding_staleness_days)
        if (
            pd.notna(currency)
            and str(currency).strip().upper() == "USD"
            and np.isfinite(market_cap)
            and market_cap > 0
        ):
            output.at[output_index, "market_cap_usd"] = int(round(market_cap))
            output.at[output_index, "log_market_cap_usd"] = float(
                np.log(market_cap)
            )
            if np.isfinite(staleness) and staleness >= 0:
                output.at[
                    output_index, "market_cap_shares_staleness_days"
                ] = int(round(staleness))
        adjusted_open = _finite_scalar(metric.adjusted_open)
        previous_close = _finite_scalar(signal.get("prior_adjusted_close"))
        signal_close = _finite_scalar(signal.get("adjusted_close"))
        if np.isfinite(adjusted_open) and adjusted_open > 0:
            output.at[output_index, "signal_adjusted_open"] = adjusted_open
            if np.isfinite(previous_close) and previous_close > 0:
                output.at[output_index, "signal_gap_pct"] = (
                    adjusted_open / previous_close - 1.0
                ) * 100.0
            if np.isfinite(signal_close) and signal_close > 0:
                output.at[output_index, "signal_intraday_return_pct"] = (
                    signal_close / adjusted_open - 1.0
                ) * 100.0
        shares = _finite_scalar(metric.shares_outstanding)
        source = metric.shares_outstanding_source
        if (
            np.isfinite(shares)
            and shares > 0
            and np.isfinite(staleness)
            and staleness >= 0
            and pd.notna(source)
            and str(source).startswith("sec_")
        ):
            output.at[output_index, "shares_outstanding"] = int(round(shares))
            output.at[output_index, "log_shares_outstanding"] = float(np.log(shares))
            raw_volume = _finite_scalar(metric.raw_volume)
            if np.isfinite(raw_volume) and raw_volume >= 0:
                output.at[output_index, "signal_turnover_ratio"] = (
                    raw_volume / shares
                )

    output["market_cap_usd"] = pd.to_numeric(
        output["market_cap_usd"], errors="coerce"
    ).astype("Int64")
    output["market_cap_shares_staleness_days"] = pd.to_numeric(
        output["market_cap_shares_staleness_days"], errors="coerce"
    ).astype("Int64")
    output["shares_outstanding"] = pd.to_numeric(
        output["shares_outstanding"], errors="coerce"
    ).astype("Int64")
    return output.loc[:, SIGNAL_COLUMNS]


_WORLD_SERIES_TO_FIELD = {
    ("twelve_data", "SPY"): "_market_spy_level",
    ("twelve_data", "QQQ"): "_market_qqq_level",
    ("twelve_data", "IWM"): "_market_iwm_level",
    ("twelve_data", "DIA"): "_market_dia_level",
    ("cboe_vix", "VIX"): "market_vix_level",
    ("cboe_vix", "VXN"): "market_vxn_level",
    ("cboe_vix", "VVIX"): "market_vvix_level",
    ("cboe_vix", "SKEW"): "market_skew_level",
    ("cboe_vix", "VIX9D"): "_market_vix9d_level",
    ("cboe_vix", "VIX3M"): "_market_vix3m_level",
}


def build_global_market_context(
    trading_dates: pd.DatetimeIndex,
    market_breadth: pd.DataFrame,
    world_market_observations: pd.DataFrame,
) -> pd.DataFrame:
    """Build one point-in-time market row for every global trading session."""

    dates = pd.DatetimeIndex(trading_dates).normalize()
    context = pd.DataFrame({"market_date": dates})
    # Daily benchmark and volatility observations on the source VM are marked
    # available 30-45 minutes after the regular close. Entry/early decisions
    # take effect no earlier than the next session, so 17:00 New York is both
    # causal and consistent with the same-close stock/breadth features.
    context["decision_at"] = [
        _signal_decision_timestamp(value) + pd.Timedelta(hours=1)
        for value in context["market_date"]
    ]

    breadth_columns = (
        "market_breadth_above_ma50_ratio",
        "market_breadth_above_ma150_ratio",
        "market_breadth_above_ma200_ratio",
        "market_breadth_trend_template_ratio",
        "market_breadth_rs70_ratio",
        "market_breadth_rs90_ratio",
        "market_advancer_ratio",
        "market_median_daily_return_pct",
    )
    for column in breadth_columns:
        context[column] = np.nan
    if not market_breadth.empty:
        breadth = market_breadth.copy()
        breadth["market_date"] = pd.to_datetime(
            breadth["market_date"], errors="raise"
        ).dt.normalize()
        if breadth.duplicated("market_date").any():
            raise ValueError("market breadth contains duplicate dates")
        breadth = breadth.set_index("market_date")
        for column in breadth_columns:
            context[column] = context["market_date"].map(breadth[column])
    context["market_breadth_above_ma50_change_5d"] = (
        context["market_breadth_above_ma50_ratio"]
        - context["market_breadth_above_ma50_ratio"].shift(5)
    )
    context["market_breadth_above_ma200_change_21d"] = (
        context["market_breadth_above_ma200_ratio"]
        - context["market_breadth_above_ma200_ratio"].shift(21)
    )

    for field in _WORLD_SERIES_TO_FIELD.values():
        context[field] = np.nan
    if not world_market_observations.empty:
        _require_columns(
            world_market_observations,
            WORLD_MARKET_SOURCE_COLUMNS,
            "world market observations",
        )
        world = world_market_observations.loc[:, WORLD_MARKET_SOURCE_COLUMNS].copy()
        for column in ("observation_time", "available_at", "asof_known_at"):
            world[column] = pd.to_datetime(world[column], errors="raise", utc=True)
        world["known_at"] = world[["available_at", "asof_known_at"]].max(axis=1)
        decisions = context.loc[:, ["decision_at"]].sort_values("decision_at")
        for key, target in _WORLD_SERIES_TO_FIELD.items():
            source, series_id = key
            observations = world.loc[
                world["source"].eq(source) & world["series_id"].eq(series_id),
                ["known_at", "observation_time", "value"],
            ].copy()
            if observations.empty:
                continue
            observations = observations.sort_values(
                ["known_at", "observation_time"], kind="mergesort"
            ).drop_duplicates("known_at", keep="last")
            aligned = pd.merge_asof(
                decisions,
                observations,
                left_on="decision_at",
                right_on="known_at",
                direction="backward",
                allow_exact_matches=True,
            )
            context[target] = pd.to_numeric(
                aligned["value"], errors="coerce"
            ).to_numpy(dtype=float)

    for ticker in ("spy", "qqq", "iwm"):
        level = context[f"_market_{ticker}_level"]
        for window in (5, 21, 63):
            context[f"market_{ticker}_prior_return_{window}d_pct"] = (
                level.shift(1) / level.shift(window + 1) - 1.0
            ) * 100.0
    dia = context["_market_dia_level"]
    context["market_dia_prior_return_21d_pct"] = (
        dia.shift(1) / dia.shift(22) - 1.0
    ) * 100.0
    context["market_vix9d_to_vix_ratio"] = (
        context["_market_vix9d_level"] / context["market_vix_level"]
    ).where(context["market_vix_level"].gt(0))
    context["market_vix_to_vix3m_ratio"] = (
        context["market_vix_level"] / context["_market_vix3m_level"]
    ).where(context["_market_vix3m_level"].gt(0))
    return context


def enrich_global_features(
    signals: pd.DataFrame,
    early_cuts: pd.DataFrame,
    market_context: pd.DataFrame,
) -> CalculationBatchResult:
    """Attach market context and signal-date cross-sectional leadership ranks."""

    output = signals.copy()
    early = early_cuts.copy()
    for column in GLOBAL_MARKET_FEATURE_COLUMNS:
        output[column] = np.nan
    for column in EARLY_GLOBAL_MARKET_FEATURE_COLUMNS:
        early[column] = np.nan
    for column in EARLY_CONFIRMATION_RELATIVE_FEATURE_COLUMNS:
        early[column] = np.nan
    if output.empty:
        return CalculationBatchResult(
            output.loc[:, SIGNAL_COLUMNS], early.loc[:, EARLY_CUT_COLUMNS]
        )

    context = market_context.copy()
    context["market_date"] = pd.to_datetime(
        context["market_date"], errors="raise"
    ).dt.normalize()
    if context.duplicated("market_date").any():
        raise ValueError("global market context contains duplicate dates")
    context_lookup = context.set_index("market_date")

    signal_dates = pd.to_datetime(output["signal_date"], errors="raise").dt.normalize()
    direct_signal_context = tuple(
        column
        for column in GLOBAL_MARKET_FEATURE_COLUMNS
        if not column.startswith("cross_sectional_")
        and not column.startswith("relative_return_")
    )
    for column in direct_signal_context:
        if column in context_lookup:
            output[column] = signal_dates.map(context_lookup[column])

    for window in (21, 63, 126, 252):
        source_column = f"prior_return_{window}d_pct"
        target_column = f"cross_sectional_rs_{window}d_pct_rank"
        output[target_column] = output.groupby(
            signal_dates, sort=False, observed=True
        )[source_column].rank(method="average", pct=True)
    for benchmark in ("spy", "qqq", "iwm"):
        for window in (21, 63):
            output[
                f"relative_return_vs_{benchmark}_{window}d_pct_points"
            ] = _numeric(output[f"prior_return_{window}d_pct"]) - _numeric(
                output[f"market_{benchmark}_prior_return_{window}d_pct"]
            )

    if not early.empty:
        landmark_dates = pd.to_datetime(
            early["landmark_date"], errors="coerce"
        ).dt.normalize()
        for column in EARLY_GLOBAL_MARKET_FEATURE_COLUMNS:
            if column in context_lookup:
                early[column] = landmark_dates.map(context_lookup[column])
        signal_dates_early = pd.to_datetime(
            early["signal_date"], errors="raise"
        ).dt.normalize()
        for benchmark in ("spy", "qqq", "iwm"):
            level_column = f"_market_{benchmark}_level"
            landmark_level = landmark_dates.map(context_lookup[level_column])
            signal_level = signal_dates_early.map(context_lookup[level_column])
            market_return = (
                (landmark_level / signal_level - 1.0) * 100.0
            ).where(signal_level.gt(0))
            early[f"market_{benchmark}_return_since_signal_pct"] = market_return
            early[
                f"relative_return_vs_{benchmark}_since_signal_pct_points"
            ] = _numeric(early["close_return_from_signal_pct"]) - market_return

        early_landmarks = pd.to_numeric(
            early["landmark_day"], errors="coerce"
        ).isin(EARLY_CUT_LANDMARK_DAYS)
        early_group = [landmark_dates.loc[early_landmarks], early.loc[early_landmarks, "landmark_day"]]
        close_rank = _numeric(
            early.loc[early_landmarks, "close_return_from_signal_pct"]
        ).groupby(early_group, sort=False, observed=True).rank(
            method="average", pct=True
        )
        spy_relative = _numeric(
            early.loc[
                early_landmarks,
                "relative_return_vs_spy_since_signal_pct_points",
            ]
        )
        spy_rank = spy_relative.groupby(
            early_group, sort=False, observed=True
        ).rank(method="average", pct=True)
        early.loc[
            early_landmarks,
            "cross_sectional_close_return_since_signal_pct_rank",
        ] = close_rank
        early.loc[
            early_landmarks,
            "cross_sectional_relative_return_vs_spy_since_signal_pct_rank",
        ] = spy_rank
        relative_ranks = [close_rank, spy_rank]
        for benchmark in ("qqq", "iwm"):
            relative = _numeric(
                early.loc[
                    early_landmarks,
                    f"relative_return_vs_{benchmark}_since_signal_pct_points",
                ]
            )
            relative_ranks.append(
                relative.groupby(early_group, sort=False, observed=True).rank(
                    method="average", pct=True
                )
            )
        early.loc[early_landmarks, "early_relative_strength_score"] = (
            pd.concat(relative_ranks, axis=1).mean(axis=1, skipna=True) * 100.0
        )

    return CalculationBatchResult(
        output.loc[:, SIGNAL_COLUMNS], early.loc[:, EARLY_CUT_COLUMNS]
    )


def build_current_taxonomy_backcast_context(
    raw: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    """Build ranked and changing daily group metrics from current taxonomy.

    Price, RS and breadth inputs are causal as of each market close.  Only the
    taxonomy labels are a current-snapshot backcast, so all resulting features
    remain diagnostic-only in research.
    """

    context_metric_columns = tuple(
        suffix
        for suffix in CURRENT_TAXONOMY_BACKCAST_METRIC_SUFFIXES
        if suffix != "stock_rs_raw_pct_rank"
    )
    required = set(CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS)
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(
            "taxonomy group context is missing columns: " + ", ".join(missing)
        )
    if raw.empty:
        result = raw.loc[:, CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS].copy()
        for suffix in context_metric_columns:
            if suffix not in result:
                result[suffix] = np.nan
        return result.loc[
            :,
            [
                "market_date",
                "taxonomy_level",
                "ibkr_industry",
                "ibkr_category",
                "ibkr_subcategory",
                *context_metric_columns,
            ],
        ]
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")

    result = raw.loc[:, CURRENT_TAXONOMY_BACKCAST_GROUP_CONTEXT_COLUMNS].copy()
    result["market_date"] = pd.to_datetime(
        result["market_date"], errors="raise"
    ).dt.normalize()
    allowed_levels = {level for level, _labels in CURRENT_TAXONOMY_BACKCAST_LEVEL_SPECS}
    levels = result["taxonomy_level"].astype("string")
    unexpected = sorted(set(levels.dropna().astype(str)) - allowed_levels)
    if levels.isna().any() or unexpected:
        raise ValueError(
            "taxonomy group context contains invalid levels"
            + (": " + ", ".join(unexpected) if unexpected else "")
        )
    result["taxonomy_level"] = levels
    for column in ("ibkr_industry", "ibkr_category", "ibkr_subcategory"):
        values = result[column].astype("string").str.strip()
        result[column] = values.mask(values.eq(""), pd.NA)

    hierarchy_valid = (
        result["ibkr_category"].isna() | result["ibkr_industry"].notna()
    ) & (
        result["ibkr_subcategory"].isna() | result["ibkr_category"].notna()
    )
    if not hierarchy_valid.all():
        raise ValueError("taxonomy group context violates hierarchy ordering")
    key_columns = (
        "market_date",
        "taxonomy_level",
        "ibkr_industry",
        "ibkr_category",
        "ibkr_subcategory",
    )
    if result.duplicated(list(key_columns)).any():
        raise ValueError("taxonomy group context contains duplicate group dates")

    numeric_columns = tuple(required - set(key_columns))
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce").astype(float)
    level_minimums = {
        "industry": cfg.taxonomy_backcast_industry_min_members,
        "category": cfg.taxonomy_backcast_category_min_members,
        "subcategory": cfg.taxonomy_backcast_subcategory_min_members,
    }
    eligible_count_by_metric = {
        **{
            f"group_median_return_{window}d_pct": (
                f"return_{window}d_eligible_count"
            )
            for window in (21, 63, 126, 252)
        },
        "group_rs_raw_median": "rs_raw_eligible_count",
        "above_ma50_ratio": "ma50_eligible_count",
        "above_ma200_ratio": "ma200_eligible_count",
        "new_52w_high_count": "new_52w_high_eligible_count",
        "new_52w_high_ratio": "new_52w_high_eligible_count",
        "rs70_ratio": "rs_rating_eligible_count",
        "rs90_ratio": "rs_rating_eligible_count",
        "trend_template_ratio": "trend_template_eligible_count",
        "new_8of8_signal_count": "group_member_count",
        "new_8of8_signal_ratio": "group_member_count",
    }
    for level, minimum in level_minimums.items():
        in_level = result["taxonomy_level"].eq(level)
        for metric, count_column in eligible_count_by_metric.items():
            result.loc[
                in_level & result[count_column].lt(minimum), metric
            ] = np.nan

    result["leadership_breadth_score"] = result[
        ["rs70_ratio", "rs90_ratio", "trend_template_ratio"]
    ].mean(axis=1, skipna=False)
    for source, target in (
        *(
            (
                f"group_median_return_{window}d_pct",
                f"group_return_{window}d_pct_rank",
            )
            for window in (21, 63, 126, 252)
        ),
        ("group_rs_raw_median", "group_rs_raw_pct_rank"),
    ):
        result[target] = result.groupby(
            ["market_date", "taxonomy_level"],
            sort=False,
            observed=True,
            dropna=False,
        )[source].rank(method="average", pct=True)

    ordinal_lookup = pd.Series(
        np.arange(len(trading_dates), dtype=np.int64),
        index=pd.DatetimeIndex(trading_dates).normalize(),
    )
    result["_session_ordinal"] = result["market_date"].map(ordinal_lookup)
    if result["_session_ordinal"].isna().any():
        raise ValueError("taxonomy group context contains non-trading dates")
    group_keys = [
        "taxonomy_level",
        "ibkr_industry",
        "ibkr_category",
        "ibkr_subcategory",
    ]
    result = result.sort_values(
        [*group_keys, "market_date"], kind="stable", na_position="first"
    )
    grouped = result.groupby(
        group_keys, sort=False, observed=True, dropna=False
    )
    for source in (
        "above_ma50_ratio",
        "above_ma200_ratio",
        "new_52w_high_ratio",
        "leadership_breadth_score",
    ):
        for sessions in (5, 21):
            lagged = grouped[source].shift(sessions)
            lagged_ordinal = grouped["_session_ordinal"].shift(sessions)
            exact_lag = result["_session_ordinal"].sub(lagged_ordinal).eq(sessions)
            result[f"{source}_change_{sessions}d"] = (
                result[source] - lagged
            ).where(exact_lag)
    result = result.drop(columns="_session_ordinal").reset_index(drop=True)
    return result.loc[
        :,
        [
            "market_date",
            "taxonomy_level",
            "ibkr_industry",
            "ibkr_category",
            "ibkr_subcategory",
            *context_metric_columns,
        ],
    ]


def enrich_current_taxonomy_backcast_features(
    signals: pd.DataFrame,
    taxonomy: pd.DataFrame,
    group_context: pd.DataFrame,
    member_ranks: pd.DataFrame,
) -> pd.DataFrame:
    """Attach diagnostic current-taxonomy hierarchy metrics to signal rows."""

    output = signals.drop(
        columns=[
            *CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
            *CURRENT_TAXONOMY_BACKCAST_FEATURE_COLUMNS,
        ],
        errors="ignore",
    ).copy()
    output["signal_date"] = pd.to_datetime(
        output["signal_date"], errors="raise"
    ).dt.normalize()
    missing_taxonomy = sorted(set(CURRENT_TAXONOMY_SOURCE_COLUMNS) - set(taxonomy))
    if missing_taxonomy:
        raise ValueError(
            "current taxonomy is missing columns: " + ", ".join(missing_taxonomy)
        )
    taxonomy_values = taxonomy.loc[:, CURRENT_TAXONOMY_SOURCE_COLUMNS].copy()
    if taxonomy_values.duplicated(list(IDENTITY_COLUMNS)).any():
        raise ValueError("current taxonomy contains duplicate identities")
    for source, target in zip(
        ("ibkr_industry", "ibkr_category", "ibkr_subcategory"),
        CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
    ):
        values = taxonomy_values[source].astype("string").str.strip()
        taxonomy_values[target] = values.mask(values.eq(""), pd.NA)
    taxonomy_values = taxonomy_values.drop(
        columns=["ibkr_industry", "ibkr_category", "ibkr_subcategory"]
    )
    output = output.merge(
        taxonomy_values,
        on=list(IDENTITY_COLUMNS),
        how="left",
        validate="many_to_one",
        sort=False,
    )
    labels = CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS
    hierarchy_valid = (output[labels[1]].isna() | output[labels[0]].notna()) & (
        output[labels[2]].isna() | output[labels[1]].notna()
    )
    if not hierarchy_valid.all():
        raise ValueError("current taxonomy violates Industry/Category/Subcategory")

    context_metric_columns = tuple(
        suffix
        for suffix in CURRENT_TAXONOMY_BACKCAST_METRIC_SUFFIXES
        if suffix != "stock_rs_raw_pct_rank"
    )
    label_mapping = {
        "ibkr_industry": labels[0],
        "ibkr_category": labels[1],
        "ibkr_subcategory": labels[2],
    }
    for level, source_label_columns in CURRENT_TAXONOMY_BACKCAST_LEVEL_SPECS:
        context = group_context.loc[
            group_context["taxonomy_level"].eq(level),
            [
                "market_date",
                *source_label_columns,
                *context_metric_columns,
            ],
        ].copy()
        rename = {
            "market_date": "signal_date",
            **{column: label_mapping[column] for column in source_label_columns},
            **{
                suffix: f"ibkr_taxbc_{level}_{suffix}"
                for suffix in context_metric_columns
            },
        }
        context = context.rename(columns=rename)
        join_columns = [
            "signal_date",
            *(label_mapping[column] for column in source_label_columns),
        ]
        output = output.merge(
            context,
            on=join_columns,
            how="left",
            validate="many_to_one",
            sort=False,
        )

    required_ranks = set(CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS)
    missing_ranks = sorted(required_ranks - set(member_ranks.columns))
    if missing_ranks:
        raise ValueError(
            "current taxonomy member ranks are missing columns: "
            + ", ".join(missing_ranks)
        )
    ranks = member_ranks.loc[:, CURRENT_TAXONOMY_BACKCAST_MEMBER_RANK_COLUMNS]
    ranks = ranks.copy()
    ranks["signal_date"] = pd.to_datetime(
        ranks["signal_date"], errors="raise"
    ).dt.normalize()
    if ranks.duplicated(["signal_date", *IDENTITY_COLUMNS]).any():
        raise ValueError("current taxonomy member ranks contain duplicate signals")
    output = output.merge(
        ranks,
        on=["signal_date", *IDENTITY_COLUMNS],
        how="left",
        validate="one_to_one",
        sort=False,
    )

    for column in CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS:
        output[column] = output[column].astype("string")
    for column in CURRENT_TAXONOMY_BACKCAST_INTEGER_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("Int64")
    numeric_columns = (
        set(CURRENT_TAXONOMY_BACKCAST_FEATURE_COLUMNS)
        - set(CURRENT_TAXONOMY_BACKCAST_INTEGER_COLUMNS)
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype(float)
    for column in CURRENT_TAXONOMY_BACKCAST_UNIT_INTERVAL_COLUMNS:
        values = output[column].dropna()
        if ((values < -1e-12) | (values > 1.0 + 1e-12)).any():
            raise ValueError(f"taxonomy ratio/rank {column} is outside [0, 1]")
    for column in CURRENT_TAXONOMY_BACKCAST_CHANGE_COLUMNS:
        values = output[column].dropna()
        if ((values < -1.0 - 1e-12) | (values > 1.0 + 1e-12)).any():
            raise ValueError(f"taxonomy change {column} is outside [-1, 1]")
    missing_output = sorted(set(SIGNAL_COLUMNS) - set(output.columns))
    if missing_output:
        raise AssertionError(
            "taxonomy enrichment lost signal columns: " + ", ".join(missing_output)
        )
    return output.loc[:, SIGNAL_COLUMNS]


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _rolling_in_segment(
    values: pd.Series,
    segment: pd.Series,
    window: int,
    operation: str,
    *,
    offset: int = 0,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for segment_id in segment.dropna().unique():
        in_segment = segment.eq(segment_id)
        isolated = values.where(in_segment).shift(offset)
        rolled = getattr(isolated.rolling(window, min_periods=window), operation)()
        result.loc[in_segment] = rolled.loc[in_segment]
    return result


def _lag_in_segment(values: pd.Series, segment: pd.Series, periods: int) -> pd.Series:
    shifted = values.shift(periods)
    return shifted.where(segment.eq(segment.shift(periods)))


def _complete_prior_window(
    values: pd.Series, segment: pd.Series, window: int
) -> pd.Series:
    return _rolling_in_segment(values, segment, window, "count", offset=1).eq(window)


def _pct_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        ((numerator / denominator) - 1.0)
        .mul(100.0)
        .where(numerator.notna() & denominator.gt(0))
    )


def _prior_return(close: pd.Series, segment: pd.Series, sessions: int) -> pd.Series:
    end = _lag_in_segment(close, segment, 1)
    start = _lag_in_segment(close, segment, sessions + 1)
    complete = _complete_prior_window(close, segment, sessions + 1)
    return _pct_ratio(end, start).where(complete)


def _prior_correlation(
    left: pd.Series,
    right: pd.Series,
    segment: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for segment_id in segment.dropna().unique():
        in_segment = segment.eq(segment_id)
        prior_left = left.where(in_segment).shift(1)
        prior_right = right.where(in_segment).shift(1)
        correlated = prior_left.rolling(window, min_periods=window).corr(prior_right)
        result.loc[in_segment] = correlated.loc[in_segment]
    return result


def _nullable_flag(
    index: pd.Index,
    available: pd.Series,
    predicate: Callable[[pd.Index], pd.Series | np.ndarray],
) -> pd.Series:
    result = pd.Series(pd.NA, index=index, dtype="boolean")
    selected = available[available].index
    if len(selected):
        result.loc[selected] = np.asarray(predicate(selected), dtype=bool)
    return result


def _analysis_split(
    signal_dates: pd.Series,
    label_available_dates: pd.Series,
    cfg: Config,
) -> pd.Series:
    dates = pd.to_datetime(signal_dates).dt.normalize()
    availability = pd.to_datetime(label_available_dates).dt.normalize()
    discovery_end = pd.Timestamp(cfg.discovery_end_date)
    validation_end = pd.Timestamp(cfg.validation_end_date)
    holdout_cutoff = pd.Timestamp(cfg.holdout_cutoff_date)
    discovery_eligible = (
        (dates <= discovery_end)
        & availability.notna()
        & (availability <= discovery_end)
    )
    validation_eligible = (
        (dates > discovery_end)
        & (dates <= validation_end)
        & availability.notna()
        & (availability <= validation_end)
    )
    diagnostic_eligible = (
        (dates > validation_end)
        & (dates <= holdout_cutoff)
        & availability.notna()
        & (availability <= holdout_cutoff)
    )
    values = np.where(
        discovery_eligible,
        "discovery",
        np.where(
            validation_eligible,
            "validation",
            np.where(
                diagnostic_eligible,
                "diagnostic",
                np.where(dates > holdout_cutoff, "holdout", "purged"),
            ),
        ),
    )
    return pd.Series(values, index=signal_dates.index, dtype="string")


def _nullable_bool(value: bool | None) -> Any:
    return pd.NA if value is None else bool(value)


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _date_at(trading_dates: pd.DatetimeIndex, position: int) -> pd.Timestamp:
    if 0 <= position < len(trading_dates):
        return pd.Timestamp(trading_dates[position])
    return pd.NaT


def _observed_same_segment(
    indexed: pd.DataFrame,
    segment: pd.Series,
    start: int,
    end: int,
    segment_id: float,
) -> bool:
    """Return whether every inclusive global-session position is one chain."""

    if start < 0 or end >= len(indexed) or end < start or pd.isna(segment_id):
        return False
    rows = indexed.iloc[start : end + 1]
    if not rows["symbol"].notna().all():
        return False
    return bool(segment.iloc[start : end + 1].eq(segment_id).all())


def _complete_price_path(
    indexed: pd.DataFrame,
    segment: pd.Series,
    start: int,
    end: int,
    segment_id: float,
) -> bool:
    if not _observed_same_segment(indexed, segment, start, end, segment_id):
        return False
    values = indexed.iloc[start : end + 1].loc[
        :, ["adjusted_close", "adjusted_high", "adjusted_low"]
    ]
    numeric = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(numeric.size and np.isfinite(numeric).all() and (numeric > 0).all())


def _first_hit_day(mask: np.ndarray, first_day: int) -> int | None:
    positions = np.flatnonzero(mask)
    return None if not len(positions) else int(first_day + positions[0])


def _gain_loss_order(
    first_strong_day: int | None,
    first_loss_day: int | None,
) -> str:
    if first_strong_day is None and first_loss_day is None:
        return "neither"
    if first_loss_day is None:
        return "gain_only"
    if first_strong_day is None:
        return "loss_only"
    if first_strong_day == first_loss_day:
        return "same_day_ambiguous"
    return "gain_first" if first_strong_day < first_loss_day else "loss_first"


def _continuation_order(
    first_strong_day: int | None,
    first_loss_day: int | None,
) -> str:
    if first_strong_day is None and first_loss_day is None:
        return "neither"
    if first_loss_day is None:
        return "strong_first"
    if first_strong_day is None:
        return "loss_first"
    if first_strong_day == first_loss_day:
        return "same_session_ambiguous"
    return "strong_first" if first_strong_day < first_loss_day else "loss_first"


def _entry_path_outcome(
    indexed: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segment: pd.Series,
    signal_position: int,
    cfg: Config,
) -> dict[str, Any]:
    """Calculate complete D1..D90 paths without relying on future SQL labels."""

    horizons = (5, 10, 20, 30, 40, 60, 90)
    result: dict[str, Any] = {}
    for horizon in horizons:
        result.update(
            {
                f"forward_{horizon}d_max_gain_pct": np.nan,
                f"forward_{horizon}d_max_loss_pct": np.nan,
                f"forward_{horizon}d_label_end_date": _date_at(
                    trading_dates, signal_position + horizon
                ),
                f"terminal_close_return_{horizon}d_pct": np.nan,
            }
        )
    result.update({
        "first_gain_2pct_day": pd.NA,
        "first_gain_1pct_day": pd.NA,
        "first_gain_3pct_day": pd.NA,
        "first_gain_5pct_day": pd.NA,
        "first_loss_5pct_day": pd.NA,
        "first_loss_10pct_day": pd.NA,
        "gain_loss_order_5d": pd.NA,
        "weak_5d": pd.NA,
        "strong_5d": pd.NA,
        "deep_loss_5d": pd.NA,
        "bad_5d": pd.NA,
        "loss_first_5d": pd.NA,
        "strong_first_5d": pd.NA,
        "terminal_stagnant_5d": pd.NA,
        "terminal_winner_5d": pd.NA,
        "stagnant_5d": pd.NA,
        "hard_stop_10pct_5d": pd.NA,
        "terminal_nonpositive_20d": pd.NA,
        "terminal_winner_20d": pd.NA,
        "terminal_nonpositive_30d": pd.NA,
        "terminal_winner_30d": pd.NA,
        "runner_60d": pd.NA,
        "runner_90d": pd.NA,
        "mfe_to_abs_mae_5d_ratio": np.nan,
        "terminal_return_to_mfe_5d_ratio": np.nan,
        "_full_path_available": False,
    })
    segment_id = segment.iloc[signal_position]
    signal_close = float(indexed["adjusted_close"].iloc[signal_position])
    if not _finite_positive(signal_close):
        return result
    complete: dict[int, bool] = {}
    for horizon in horizons:
        complete[horizon] = _complete_price_path(
            indexed, segment, signal_position, signal_position + horizon, segment_id
        )
        if not complete[horizon]:
            continue
        path = indexed.iloc[signal_position + 1 : signal_position + horizon + 1]
        highs = _numeric(path["adjusted_high"]).to_numpy(dtype=float)
        lows = _numeric(path["adjusted_low"]).to_numpy(dtype=float)
        terminal_close = float(
            indexed["adjusted_close"].iloc[signal_position + horizon]
        )
        result[f"forward_{horizon}d_max_gain_pct"] = max(
            0.0, float(((highs / signal_close - 1.0) * 100.0).max())
        )
        result[f"forward_{horizon}d_max_loss_pct"] = min(
            0.0, float(((lows / signal_close - 1.0) * 100.0).min())
        )
        result[f"terminal_close_return_{horizon}d_pct"] = (
            terminal_close / signal_close - 1.0
        ) * 100.0

    if not complete[5]:
        return result

    highs = _numeric(
        indexed["adjusted_high"].iloc[signal_position + 1 : signal_position + 6]
    ).to_numpy(dtype=float)
    lows = _numeric(
        indexed["adjusted_low"].iloc[signal_position + 1 : signal_position + 6]
    ).to_numpy(dtype=float)
    terminal_close = float(indexed["adjusted_close"].iloc[signal_position + 5])
    high_returns = (highs / signal_close - 1.0) * 100.0
    low_returns = (lows / signal_close - 1.0) * 100.0
    max_gain = max(0.0, float(high_returns.max()))
    max_loss = min(0.0, float(low_returns.min()))
    first_gain_2 = _first_hit_day(high_returns >= cfg.weak_5d_max_gain_pct, 1)
    first_gain_1 = _first_hit_day(high_returns >= 1.0, 1)
    first_gain_3 = _first_hit_day(high_returns >= 3.0, 1)
    first_gain_5 = _first_hit_day(high_returns >= cfg.strong_5d_min_gain_pct, 1)
    first_loss_5 = _first_hit_day(low_returns <= cfg.deep_loss_5d_max_loss_pct, 1)
    first_loss_10 = _first_hit_day(
        low_returns <= cfg.hard_stop_5d_max_loss_pct, 1
    )
    order = _gain_loss_order(first_gain_5, first_loss_5)
    weak = max_gain < cfg.weak_5d_max_gain_pct
    strong = max_gain >= cfg.strong_5d_min_gain_pct
    deep_loss = max_loss <= cfg.deep_loss_5d_max_loss_pct
    terminal_return = (terminal_close / signal_close - 1.0) * 100.0

    result.update(
        {
            "forward_5d_max_gain_pct": max_gain,
            "forward_5d_max_loss_pct": max_loss,
            "terminal_close_return_5d_pct": terminal_return,
            "first_gain_1pct_day": (pd.NA if first_gain_1 is None else first_gain_1),
            "first_gain_2pct_day": (pd.NA if first_gain_2 is None else first_gain_2),
            "first_gain_3pct_day": (pd.NA if first_gain_3 is None else first_gain_3),
            "first_gain_5pct_day": (pd.NA if first_gain_5 is None else first_gain_5),
            "first_loss_5pct_day": (pd.NA if first_loss_5 is None else first_loss_5),
            "first_loss_10pct_day": (
                pd.NA if first_loss_10 is None else first_loss_10
            ),
            "gain_loss_order_5d": order,
            "weak_5d": weak,
            "strong_5d": strong,
            "deep_loss_5d": deep_loss,
            "bad_5d": weak or deep_loss,
            "loss_first_5d": _nullable_bool(
                None
                if order == "same_day_ambiguous"
                else order in {"loss_first", "loss_only"}
            ),
            "strong_first_5d": _nullable_bool(
                None
                if order == "same_day_ambiguous"
                else order in {"gain_first", "gain_only"}
            ),
            "terminal_stagnant_5d": (
                terminal_return < cfg.terminal_stagnant_5d_max_return_pct
            ),
            "terminal_winner_5d": (
                terminal_return >= cfg.terminal_winner_5d_min_return_pct
            ),
            "stagnant_5d": (
                max_gain < cfg.weak_5d_max_gain_pct
                and terminal_return <= cfg.stagnant_5d_max_return_pct
            ),
            "hard_stop_10pct_5d": first_loss_10 is not None,
            "mfe_to_abs_mae_5d_ratio": (
                max_gain / abs(max_loss) if max_loss < 0 else np.nan
            ),
            "terminal_return_to_mfe_5d_ratio": (
                terminal_return / max_gain if max_gain > 0 else np.nan
            ),
            "_full_path_available": True,
        }
    )
    for horizon in (20, 30):
        if complete[horizon]:
            terminal = float(result[f"terminal_close_return_{horizon}d_pct"])
            result[f"terminal_nonpositive_{horizon}d"] = terminal <= 0.0
            result[f"terminal_winner_{horizon}d"] = (
                terminal >= cfg.continuation_winner_min_return_pct
            )
    for horizon in (60, 90):
        if complete[horizon]:
            result[f"runner_{horizon}d"] = bool(
                float(result[f"terminal_close_return_{horizon}d_pct"])
                >= cfg.continuation_winner_min_return_pct
                and float(result[f"forward_{horizon}d_max_loss_pct"])
                > cfg.hard_stop_5d_max_loss_pct
            )
    return result


def _event_max_drawdown(
    close: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for position in signal_positions:
        if position < window:
            continue
        values = close.iloc[position - window : position].to_numpy(dtype=float)
        segments = segment.iloc[position - window : position]
        current_segment = segment.iloc[position]
        if (
            len(values) != window
            or not np.isfinite(values).all()
            or current_segment != current_segment
            or not segments.eq(current_segment).all()
            or np.any(values <= 0)
        ):
            continue
        running_peak = np.maximum.accumulate(values)
        result.iloc[position] = float(np.min(values / running_peak - 1.0) * 100.0)
    return result


def _event_prior_chart_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
) -> dict[str, pd.Series]:
    """Calculate time-ordered OHLC features using sessions t-40 through t-1."""

    names = (
        "prior_base_width_20_pct",
        "prior_trend_slope_20_pct_per_session",
        "prior_trend_r2_20",
        "prior_trend_efficiency_20",
        "prior_positive_return_share_20",
        "prior_peak_age_40_sessions",
        "prior_pullback_from_40d_high_pct",
        "prior_trough_age_40_sessions",
        "prior_drawdown_to_trough_40_pct",
        "prior_recovery_from_trough_40_pct",
        "prior_v_recovery_fraction_40",
    )
    result = {
        name: pd.Series(np.nan, index=close.index, dtype=float) for name in names
    }
    x = np.arange(20, dtype=float)
    centered_x = x - x.mean()
    x_sum_squares = float(np.square(centered_x).sum())

    for position in signal_positions:
        if position < 40:
            continue
        current_segment = segment.iloc[position]
        history_segment = segment.iloc[position - 40 : position]
        close40 = close.iloc[position - 40 : position].to_numpy(dtype=float)
        high40 = high.iloc[position - 40 : position].to_numpy(dtype=float)
        low40 = low.iloc[position - 40 : position].to_numpy(dtype=float)
        if (
            pd.isna(current_segment)
            or not history_segment.eq(current_segment).all()
            or not np.isfinite(close40).all()
            or not np.isfinite(high40).all()
            or not np.isfinite(low40).all()
            or np.any(close40 <= 0)
            or np.any(high40 <= 0)
            or np.any(low40 <= 0)
            or np.any(high40 < low40)
            or np.any(close40 > high40)
            or np.any(close40 < low40)
        ):
            continue

        close20 = close40[-20:]
        high20 = high40[-20:]
        low20 = low40[-20:]
        result["prior_base_width_20_pct"].iloc[position] = float(
            (np.max(high20) / np.min(low20) - 1.0) * 100.0
        )

        log_close20 = np.log(close20)
        centered_y = log_close20 - log_close20.mean()
        beta = float(np.dot(centered_x, centered_y) / x_sum_squares)
        fitted = log_close20.mean() + beta * centered_x
        residual_sum_squares = float(np.square(log_close20 - fitted).sum())
        total_sum_squares = float(np.square(centered_y).sum())
        r_squared = (
            max(0.0, min(1.0, 1.0 - residual_sum_squares / total_sum_squares))
            if total_sum_squares > 0
            else 0.0
        )
        close_changes = np.diff(close20)
        absolute_path = float(np.abs(close_changes).sum())
        efficiency = (
            min(1.0, abs(float(close20[-1] - close20[0])) / absolute_path)
            if absolute_path > 0
            else 0.0
        )
        result["prior_trend_slope_20_pct_per_session"].iloc[position] = float(
            np.expm1(beta) * 100.0
        )
        result["prior_trend_r2_20"].iloc[position] = r_squared
        result["prior_trend_efficiency_20"].iloc[position] = efficiency
        result["prior_positive_return_share_20"].iloc[position] = float(
            np.mean(close_changes > 0)
        )

        peak_value = float(np.max(high40))
        peak_position = int(np.flatnonzero(high40 == peak_value)[-1])
        result["prior_peak_age_40_sessions"].iloc[position] = 39 - peak_position
        result["prior_pullback_from_40d_high_pct"].iloc[position] = float(
            (close40[-1] / peak_value - 1.0) * 100.0
        )

        running_peak = np.maximum.accumulate(high40)
        drawdowns = low40 / running_peak - 1.0
        trough_position = int(np.argmin(drawdowns))
        trough_value = float(low40[trough_position])
        peak_at_trough = float(running_peak[trough_position])
        recovery_denominator = peak_at_trough - trough_value
        result["prior_trough_age_40_sessions"].iloc[position] = 39 - trough_position
        result["prior_drawdown_to_trough_40_pct"].iloc[position] = float(
            drawdowns[trough_position] * 100.0
        )
        result["prior_recovery_from_trough_40_pct"].iloc[position] = float(
            (close40[-1] / trough_value - 1.0) * 100.0
        )
        if recovery_denominator > 0:
            result["prior_v_recovery_fraction_40"].iloc[position] = float(
                (close40[-1] - trough_value) / recovery_denominator
            )
    return result


def _event_v3_chart_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
) -> dict[str, pd.Series]:
    """Calculate base, supply and contraction geometry without future bars."""

    names = (
        "prior_base_width_10_pct",
        "prior_base_width_40_pct",
        "prior_base_width_63_pct",
        "prior_tight_close_range_5_pct",
        "prior_tight_close_range_10_pct",
        "prior_tight_close_range_15_pct",
        "prior_overhead_supply_share63",
        "prior_high_test_count_20",
        "prior_high_slope_20_pct_per_session",
        "prior_low_slope_20_pct_per_session",
        "prior_contraction_count_40",
        "prior_return_efficiency_63",
        "signal_undercut_reclaim_10",
    )
    result = {
        name: pd.Series(np.nan, index=close.index, dtype=float) for name in names
    }

    for position in signal_positions:
        current_segment = segment.iloc[position]
        if pd.isna(current_segment):
            continue
        for window, target in (
            (10, "prior_base_width_10_pct"),
            (40, "prior_base_width_40_pct"),
            (63, "prior_base_width_63_pct"),
        ):
            if position < window:
                continue
            history_segment = segment.iloc[position - window : position]
            highs = high.iloc[position - window : position].to_numpy(dtype=float)
            lows = low.iloc[position - window : position].to_numpy(dtype=float)
            if (
                history_segment.eq(current_segment).all()
                and np.isfinite(highs).all()
                and np.isfinite(lows).all()
                and np.all(highs >= lows)
                and np.all(lows > 0)
            ):
                result[target].iloc[position] = (
                    float(np.max(highs) / np.min(lows) - 1.0) * 100.0
                )

        if position >= 10:
            closes10 = close.iloc[position - 10 : position].to_numpy(dtype=float)
            if (
                segment.iloc[position - 10 : position].eq(current_segment).all()
                and np.isfinite(closes10).all()
                and np.all(closes10 > 0)
            ):
                result["prior_tight_close_range_5_pct"].iloc[position] = float(
                    (np.max(closes10[-5:]) / np.min(closes10[-5:]) - 1.0) * 100.0
                )
                result["prior_tight_close_range_10_pct"].iloc[position] = float(
                    (np.max(closes10) / np.min(closes10) - 1.0) * 100.0
                )
                prior_low10 = float(
                    np.min(low.iloc[position - 10 : position].to_numpy(dtype=float))
                )
                current_low = _finite_scalar(low.iloc[position])
                current_close = _finite_scalar(close.iloc[position])
                if (
                    np.isfinite(prior_low10)
                    and prior_low10 > 0
                    and np.isfinite(current_low)
                    and np.isfinite(current_close)
                ):
                    result["signal_undercut_reclaim_10"].iloc[position] = float(
                        current_low < prior_low10 and current_close > prior_low10
                    )

        if position >= 15:
            closes15 = close.iloc[position - 15 : position].to_numpy(dtype=float)
            if (
                segment.iloc[position - 15 : position].eq(current_segment).all()
                and np.isfinite(closes15).all()
                and np.all(closes15 > 0)
            ):
                result["prior_tight_close_range_15_pct"].iloc[position] = float(
                    (np.max(closes15) / np.min(closes15) - 1.0) * 100.0
                )

        if position >= 20:
            highs20 = high.iloc[position - 20 : position].to_numpy(dtype=float)
            lows20 = low.iloc[position - 20 : position].to_numpy(dtype=float)
            if (
                segment.iloc[position - 20 : position].eq(current_segment).all()
                and np.isfinite(highs20).all()
                and np.isfinite(lows20).all()
                and np.all(highs20 > 0)
                and np.all(lows20 > 0)
            ):
                x = np.arange(20, dtype=float)
                high_beta = float(np.polyfit(x, np.log(highs20), 1)[0])
                low_beta = float(np.polyfit(x, np.log(lows20), 1)[0])
                result["prior_high_slope_20_pct_per_session"].iloc[position] = (
                    np.expm1(high_beta) * 100.0
                )
                result["prior_low_slope_20_pct_per_session"].iloc[position] = (
                    np.expm1(low_beta) * 100.0
                )
                recent_high = float(np.max(highs20))
                tolerance = recent_high * 0.01
                result["prior_high_test_count_20"].iloc[position] = float(
                    np.sum(highs20 >= recent_high - tolerance)
                )

        if position >= 40:
            highs40 = high.iloc[position - 40 : position].to_numpy(dtype=float)
            lows40 = low.iloc[position - 40 : position].to_numpy(dtype=float)
            if (
                segment.iloc[position - 40 : position].eq(current_segment).all()
                and np.isfinite(highs40).all()
                and np.isfinite(lows40).all()
                and np.all(lows40 > 0)
            ):
                widths = np.asarray(
                    [
                        np.max(highs40[start : start + 10])
                        / np.min(lows40[start : start + 10])
                        - 1.0
                        for start in (0, 10, 20, 30)
                    ],
                    dtype=float,
                )
                result["prior_contraction_count_40"].iloc[position] = float(
                    np.sum(np.diff(widths) < 0)
                )

        if position >= 63:
            closes63 = close.iloc[position - 63 : position].to_numpy(dtype=float)
            if (
                segment.iloc[position - 63 : position].eq(current_segment).all()
                and np.isfinite(closes63).all()
                and np.all(closes63 > 0)
            ):
                current_prior_close = closes63[-1]
                result["prior_overhead_supply_share63"].iloc[position] = float(
                    np.mean(closes63 > current_prior_close)
                )
                path = float(np.abs(np.diff(closes63)).sum())
                if path > 0:
                    result["prior_return_efficiency_63"].iloc[position] = float(
                        abs(closes63[-1] - closes63[0]) / path
                    )
    return result


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else np.nan


def _linear01(value: float, low: float, high: float) -> float:
    if not np.isfinite(value) or not np.isfinite(low) or not np.isfinite(high):
        return np.nan
    if high <= low:
        return np.nan
    return _clip01((value - low) / (high - low))


def _inverse_linear01(value: float, good: float, bad: float) -> float:
    score = _linear01(value, good, bad)
    return np.nan if not np.isfinite(score) else 1.0 - score


def _triangle01(value: float, left: float, peak: float, right: float) -> float:
    if (
        not np.isfinite(value)
        or not np.isfinite(left)
        or not np.isfinite(peak)
        or not np.isfinite(right)
        or not left < peak < right
    ):
        return np.nan
    if value <= left or value >= right:
        return 0.0
    if value <= peak:
        return _linear01(value, left, peak)
    return _inverse_linear01(value, peak, right)


def _mean01(*values: float) -> float:
    array = np.asarray(values, dtype=float)
    return float(array.mean()) if len(array) and np.isfinite(array).all() else np.nan


def _weighted01(items: tuple[tuple[float, float], ...]) -> float:
    values = np.asarray([value for value, _weight in items], dtype=float)
    weights = np.asarray([weight for _value, weight in items], dtype=float)
    if (
        not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(weights).all()
        or np.any(weights < 0)
        or weights.sum() <= 0
    ):
        return np.nan
    return float(np.average(values, weights=weights))


def _soft_pattern_window_scores(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    activities: np.ndarray,
    signal_close: float,
    signal_high: float,
    signal_low: float,
    signal_activity: float,
) -> dict[str, tuple[float, float, float]]:
    """Return scores using either share volume or USD notional activity."""

    window = len(closes)
    if window < 10:
        return {}
    prior_close = float(closes[-1])
    prior_high = float(np.max(highs))
    width = float(np.max(highs) / np.min(lows) - 1.0)
    close_width = float(np.max(closes) / np.min(closes) - 1.0)
    net_return = float(closes[-1] / closes[0] - 1.0)
    close_changes = np.diff(closes)
    daily_returns = closes[1:] / closes[:-1] - 1.0
    path = float(np.abs(close_changes).sum())
    efficiency = (
        abs(float(closes[-1] - closes[0])) / path if path > 0 else 0.0
    )
    positive_share = float(np.mean(close_changes > 0))
    x = np.arange(window, dtype=float)
    log_close = np.log(closes)
    slope, intercept = np.polyfit(x, log_close, 1)
    fitted = intercept + slope * x
    total = float(np.square(log_close - log_close.mean()).sum())
    residual = float(np.square(log_close - fitted).sum())
    r_squared = max(0.0, min(1.0, 1.0 - residual / total)) if total > 0 else 0.0
    normalized_ranges = (highs - lows) / closes
    recent_count = min(5, max(2, window // 4))
    older_ranges = normalized_ranges[:-recent_count]
    recent_ranges = normalized_ranges[-recent_count:]
    range_compression = (
        float(np.mean(recent_ranges) / np.mean(older_ranges))
        if len(older_ranges) and np.mean(older_ranges) > 0
        else np.nan
    )
    older_activity = activities[:-recent_count]
    recent_activity = activities[-recent_count:]
    activity_contraction = (
        float(np.mean(recent_activity) / np.mean(older_activity))
        if len(older_activity) and np.mean(older_activity) > 0
        else np.nan
    )
    running_peak = np.maximum.accumulate(highs)
    drawdowns = lows / running_peak - 1.0
    max_drawdown = abs(float(np.min(drawdowns)))

    signal_return = float(signal_close / prior_close - 1.0)
    signal_close_location = (
        float((signal_close - signal_low) / (signal_high - signal_low))
        if signal_high > signal_low
        else np.nan
    )
    signal_activity_ratio = (
        float(signal_activity / np.mean(activities))
        if np.mean(activities) > 0
        else np.nan
    )
    signal_breakout = float(signal_close / prior_high - 1.0)
    generic_trigger = _mean01(
        _linear01(signal_return, -0.005, 0.05),
        _linear01(signal_close_location, 0.45, 0.95),
        _linear01(signal_breakout, -0.04, 0.02),
    )
    activity_breakout_trigger = _mean01(
        _linear01(signal_return, 0.0, 0.05),
        _linear01(signal_close_location, 0.50, 0.95),
        _linear01(signal_activity_ratio, 1.0, 2.5),
        _linear01(signal_breakout, -0.03, 0.02),
    )

    results: dict[str, tuple[float, float, float]] = {}

    flat_setup = _mean01(
        _inverse_linear01(width, 0.03, 0.25),
        _inverse_linear01(close_width, 0.02, 0.18),
        _inverse_linear01(
            abs(net_return) / max(width, 1e-12), 0.10, 0.85
        ),
        _inverse_linear01(range_compression, 0.55, 1.35),
    )
    results["flat_base"] = (
        flat_setup,
        generic_trigger,
        _weighted01(((flat_setup, 0.75), (generic_trigger, 0.25))),
    )

    ordered_setup = _mean01(
        _linear01(net_return, 0.0, 0.25 * np.sqrt(window / 20.0)),
        r_squared,
        _linear01(positive_share, 0.45, 0.72),
        _linear01(efficiency, 0.15, 0.70),
        _inverse_linear01(max_drawdown, 0.04, 0.30),
    )
    results["ordered_uptrend"] = (
        ordered_setup,
        generic_trigger,
        _weighted01(((ordered_setup, 0.80), (generic_trigger, 0.20))),
    )

    peak_position = int(np.argmax(highs))
    peak_value = float(highs[peak_position])
    peak_age = window - 1 - peak_position
    pullback_depth = max(0.0, float(1.0 - prior_close / peak_value))
    runup = max(0.0, float(peak_value / closes[0] - 1.0))
    reclaim_fraction = (
        float((signal_close - prior_close) / (peak_value - prior_close))
        if peak_value > prior_close
        else 1.0
    )
    pullback_setup = _mean01(
        _triangle01(pullback_depth, 0.01, 0.08, 0.25),
        _triangle01(
            float(peak_age),
            0.5,
            max(2.0, window * 0.15),
            max(4.0, window * 0.65),
        ),
        _linear01(runup, 0.03, 0.30),
        _inverse_linear01(activity_contraction, 0.55, 1.25),
    )
    pullback_trigger = _mean01(
        generic_trigger,
        _linear01(reclaim_fraction, 0.0, 0.80),
    )
    results["pullback"] = (
        pullback_setup,
        pullback_trigger,
        _weighted01(((pullback_setup, 0.65), (pullback_trigger, 0.35))),
    )

    trough_position = int(np.argmin(drawdowns))
    trough_value = float(lows[trough_position])
    peak_at_trough = float(running_peak[trough_position])
    recovery_denominator = peak_at_trough - trough_value
    recovery_fraction = (
        float((prior_close - trough_value) / recovery_denominator)
        if recovery_denominator > 0
        else np.nan
    )
    trough_age = window - 1 - trough_position
    down_duration = max(1, trough_position)
    recovery_duration = max(1, trough_age)
    symmetry_error = abs(down_duration - recovery_duration) / float(window)
    v_setup = _mean01(
        _triangle01(max_drawdown, 0.04, 0.20, 0.50),
        _triangle01(recovery_fraction, 0.35, 0.90, 1.30),
        _triangle01(
            float(trough_age),
            0.5,
            max(2.0, window * 0.25),
            max(4.0, window * 0.75),
        ),
        _inverse_linear01(symmetry_error, 0.0, 0.55),
    )
    v_trigger = _mean01(
        _linear01(signal_return, -0.01, 0.05),
        _linear01(signal_close_location, 0.40, 0.95),
        _linear01(signal_close / peak_at_trough, 0.80, 1.02),
    )
    results["v_recovery"] = (
        v_setup,
        v_trigger,
        _weighted01(((v_setup, 0.75), (v_trigger, 0.25))),
    )

    dryup_setup = _mean01(
        _inverse_linear01(activity_contraction, 0.45, 1.20),
        _inverse_linear01(range_compression, 0.50, 1.25),
        _inverse_linear01(close_width, 0.02, 0.18),
        _linear01(prior_close / prior_high, 0.88, 1.0),
    )
    results["volume_dryup_breakout"] = (
        dryup_setup,
        activity_breakout_trigger,
        _weighted01(((dryup_setup, 0.60), (activity_breakout_trigger, 0.40))),
    )

    activity_reference = float(np.mean(activities))
    relative_activity = (
        activities[1:] / activity_reference if activity_reference > 0 else np.nan
    )
    locations = np.divide(
        closes - lows,
        highs - lows,
        out=np.full(window, np.nan),
        where=highs > lows,
    )
    distribution_count = int(
        np.count_nonzero((daily_returns <= -0.002) & (relative_activity >= 1.20))
    )
    churning_count = int(
        np.count_nonzero(
            (np.abs(daily_returns) <= 0.005)
            & (relative_activity >= 1.20)
            & (locations[1:] <= 0.50)
        )
    )
    failed_breakouts = 0
    for local_position in range(5, window):
        previous_high = float(np.max(highs[:local_position]))
        if (
            highs[local_position] >= previous_high * 1.001
            and closes[local_position] < previous_high
            and daily_returns[local_position - 1] < 0
        ):
            failed_breakouts += 1
    signed_activity_balance = _safe_scalar_ratio(
        np.sum(np.sign(daily_returns) * activities[1:]),
        np.sum(activities[1:]),
    )
    scale = 20.0 / window
    distribution_setup = _mean01(
        _linear01(distribution_count * scale, 0.5, 3.5),
        _linear01(churning_count * scale, 0.5, 3.5),
        _linear01(failed_breakouts * scale, 0.0, 1.5),
        _linear01(prior_close / prior_high, 0.85, 1.0),
        _linear01(-signed_activity_balance, -0.10, 0.35),
    )
    distribution_trigger = _mean01(
        _linear01(signal_activity_ratio, 1.0, 2.5),
        _inverse_linear01(signal_close_location, 0.20, 0.80),
        _inverse_linear01(signal_return, -0.03, 0.02),
    )
    results["distribution_top"] = (
        distribution_setup,
        distribution_trigger,
        _weighted01(((distribution_setup, 0.85), (distribution_trigger, 0.15))),
    )

    block_edges = np.linspace(0, window, 5, dtype=int)
    block_widths = np.asarray(
        [
            np.max(highs[left:right]) / np.min(lows[left:right]) - 1.0
            for left, right in zip(block_edges[:-1], block_edges[1:])
            if right > left
        ],
        dtype=float,
    )
    contraction_fraction = (
        float(np.mean(np.diff(block_widths) < 0))
        if len(block_widths) >= 2
        else np.nan
    )
    width_contraction = _safe_scalar_ratio(
        block_widths[-1] if len(block_widths) else np.nan,
        block_widths[0] if len(block_widths) else np.nan,
    )
    vcp_setup = _mean01(
        contraction_fraction,
        _inverse_linear01(width_contraction, 0.25, 1.10),
        _inverse_linear01(range_compression, 0.45, 1.25),
        _inverse_linear01(activity_contraction, 0.45, 1.20),
        _linear01(prior_close / prior_high, 0.88, 1.0),
    )
    results["vcp"] = (
        vcp_setup,
        activity_breakout_trigger,
        _weighted01(((vcp_setup, 0.70), (activity_breakout_trigger, 0.30))),
    )

    split = max(5, int(round(window * 0.67)))
    pole_high = float(np.max(highs[:split]))
    flag_high = float(np.max(highs[split:])) if split < window else pole_high
    flag_low = float(np.min(lows[split:])) if split < window else float(np.min(lows))
    flag_width = flag_high / flag_low - 1.0 if flag_low > 0 else np.nan
    pole_return = pole_high / closes[0] - 1.0
    flag_pullback = max(0.0, 1.0 - prior_close / pole_high)
    flag_activity_ratio = _safe_scalar_ratio(
        np.mean(activities[split:]) if split < window else np.nan,
        np.mean(activities[:split]) if split < window else np.nan,
    )
    high_tight_setup = _mean01(
        _linear01(pole_return, 0.15, 0.80),
        _inverse_linear01(flag_width, 0.02, 0.18),
        _inverse_linear01(flag_pullback, 0.0, 0.18),
        _inverse_linear01(flag_activity_ratio, 0.45, 1.10),
        _linear01(prior_close / prior_high, 0.88, 1.0),
    )
    results["high_tight_flag"] = (
        high_tight_setup,
        activity_breakout_trigger,
        _weighted01(((high_tight_setup, 0.70), (activity_breakout_trigger, 0.30))),
    )

    if window >= 63:
        left_end = max(10, window // 4)
        handle_length = max(5, window // 10)
        handle_start = window - handle_length
        middle_end = max(left_end + 1, handle_start)
        left_rim = float(np.max(highs[:left_end]))
        middle_lows = lows[left_end:middle_end]
        trough_local = int(np.argmin(middle_lows))
        trough_global = left_end + trough_local
        trough = float(middle_lows[trough_local])
        right_start = max(left_end, window // 2)
        right_rim = float(np.max(highs[right_start:handle_start]))
        cup_depth = max(0.0, 1.0 - trough / left_rim)
        rim_recovery = right_rim / left_rim
        trough_location = trough_global / float(window - 1)
        handle_high = float(np.max(highs[handle_start:]))
        handle_low = float(np.min(lows[handle_start:]))
        handle_depth = max(0.0, 1.0 - handle_low / handle_high)
        handle_activity_ratio = _safe_scalar_ratio(
            np.mean(activities[handle_start:]),
            np.mean(activities[:handle_start]),
        )
        cup_setup = _mean01(
            _triangle01(cup_depth, 0.08, 0.25, 0.55),
            _triangle01(trough_location, 0.20, 0.50, 0.78),
            _triangle01(rim_recovery, 0.75, 0.98, 1.12),
            _triangle01(handle_depth, 0.005, 0.06, 0.18),
            _inverse_linear01(handle_activity_ratio, 0.45, 1.15),
        )
        cup_trigger = _mean01(
            activity_breakout_trigger,
            _linear01(signal_close / max(left_rim, right_rim), 0.95, 1.03),
        )
        results["cup_with_handle"] = (
            cup_setup,
            cup_trigger,
            _weighted01(((cup_setup, 0.75), (cup_trigger, 0.25))),
        )

    return results


def _event_soft_pattern_features(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    notional: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
) -> dict[str, pd.Series]:
    """Build share-volume and USD-notional chart scores without look-ahead."""

    result = {
        name: pd.Series(np.nan, index=close.index, dtype=float)
        for name in SOFT_PATTERN_FEATURE_COLUMNS
    }
    specs = {
        name: (tuple(windows), canonical)
        for name, windows, canonical in SOFT_PATTERN_SPECS
    }
    notional_specs = {
        base_name: (notional_name, tuple(windows), canonical)
        for notional_name, windows, canonical in NOTIONAL_SOFT_PATTERN_SPECS
        for base_name in (notional_name.removesuffix("_notional"),)
    }
    if set(notional_specs) != set(NOTIONAL_SOFT_PATTERN_BASE_NAMES):
        raise AssertionError("notional soft-pattern specification is incomplete")
    windows = sorted(
        {
            window
            for _name, values, _canonical in SOFT_PATTERN_SPECS
            for window in values
        }
    )
    for position in signal_positions:
        current_segment = segment.iloc[position]
        if pd.isna(current_segment):
            continue
        signal_price_values = np.asarray(
            [
                _finite_scalar(close.iloc[position]),
                _finite_scalar(high.iloc[position]),
                _finite_scalar(low.iloc[position]),
            ],
            dtype=float,
        )
        if (
            not np.isfinite(signal_price_values).all()
            or np.any(signal_price_values <= 0)
            or signal_price_values[1] < signal_price_values[2]
        ):
            continue
        signal_volume = _finite_scalar(volume.iloc[position])
        signal_notional = _finite_scalar(notional.iloc[position])
        for window in windows:
            if position < window:
                continue
            history_segment = segment.iloc[position - window : position]
            closes = close.iloc[position - window : position].to_numpy(dtype=float)
            highs = high.iloc[position - window : position].to_numpy(dtype=float)
            lows = low.iloc[position - window : position].to_numpy(dtype=float)
            volumes = volume.iloc[position - window : position].to_numpy(dtype=float)
            notionals = notional.iloc[position - window : position].to_numpy(
                dtype=float
            )
            if (
                not history_segment.eq(current_segment).all()
                or not np.isfinite(closes).all()
                or not np.isfinite(highs).all()
                or not np.isfinite(lows).all()
                or np.any(closes <= 0)
                or np.any(lows <= 0)
                or np.any(highs < lows)
            ):
                continue

            volume_valid = (
                np.isfinite(signal_volume)
                and signal_volume >= 0
                and np.isfinite(volumes).all()
                and np.all(volumes >= 0)
                and np.mean(volumes) > 0
            )
            if volume_valid:
                scores = _soft_pattern_window_scores(
                    closes,
                    highs,
                    lows,
                    volumes,
                    *signal_price_values,
                    signal_volume,
                )
                for pattern_name, (pattern_windows, canonical) in specs.items():
                    if window not in pattern_windows or pattern_name not in scores:
                        continue
                    setup, trigger, combined = scores[pattern_name]
                    result[
                        f"pattern_{pattern_name}_score_{window}d"
                    ].iloc[position] = (
                        combined * 100.0 if np.isfinite(combined) else np.nan
                    )
                    if window == canonical:
                        result[
                            f"pattern_{pattern_name}_setup_score"
                        ].iloc[position] = (
                            setup * 100.0 if np.isfinite(setup) else np.nan
                        )
                        result[
                            f"pattern_{pattern_name}_trigger_score"
                        ].iloc[position] = (
                            trigger * 100.0 if np.isfinite(trigger) else np.nan
                        )

            notional_valid = (
                np.isfinite(signal_notional)
                and signal_notional >= 0
                and np.isfinite(notionals).all()
                and np.all(notionals >= 0)
                and np.mean(notionals) > 0
            )
            if notional_valid:
                scores = _soft_pattern_window_scores(
                    closes,
                    highs,
                    lows,
                    notionals,
                    *signal_price_values,
                    signal_notional,
                )
                for pattern_name, (
                    output_name,
                    pattern_windows,
                    canonical,
                ) in notional_specs.items():
                    if window not in pattern_windows or pattern_name not in scores:
                        continue
                    setup, trigger, combined = scores[pattern_name]
                    result[
                        f"pattern_{output_name}_score_{window}d"
                    ].iloc[position] = (
                        combined * 100.0 if np.isfinite(combined) else np.nan
                    )
                    if window == canonical:
                        result[
                            f"pattern_{output_name}_setup_score"
                        ].iloc[position] = (
                            setup * 100.0 if np.isfinite(setup) else np.nan
                        )
                        result[
                            f"pattern_{output_name}_trigger_score"
                        ].iloc[position] = (
                            trigger * 100.0 if np.isfinite(trigger) else np.nan
                        )
    return result


def _event_text_and_history(
    criteria: dict[str, pd.Series],
    trend_pass: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    trigger_text = pd.Series(pd.NA, index=trend_pass.index, dtype="string")
    trigger_count = pd.Series(np.nan, index=trend_pass.index, dtype=float)
    previous_count = pd.Series(np.nan, index=trend_pass.index, dtype=float)
    sessions_since = pd.Series(np.nan, index=trend_pass.index, dtype=float)

    signal_set = set(int(value) for value in signal_positions)
    last_pass_by_segment: dict[int, int] = {}
    last_observed_segment: int | None = None
    for position in range(len(trend_pass)):
        segment_value = segment.iloc[position]
        if pd.isna(segment_value):
            # A missing global session terminates every history chain for this
            # identity, even if an upstream segment identifier were reused.
            last_pass_by_segment.clear()
            last_observed_segment = None
            continue
        segment_id = int(segment_value)
        if last_observed_segment is not None and segment_id != last_observed_segment:
            last_pass_by_segment.clear()
        last_observed_segment = segment_id
        pass_value = trend_pass.iloc[position]
        is_pass = bool(pass_value) if pd.notna(pass_value) else False
        if position in signal_set:
            triggered: list[str] = []
            for name, values in criteria.items():
                current_value = values.iloc[position]
                previous_value = values.iloc[position - 1]
                if (
                    pd.notna(current_value)
                    and pd.notna(previous_value)
                    and bool(current_value)
                    and not bool(previous_value)
                ):
                    triggered.append(name)
            trigger_text.iloc[position] = ",".join(triggered)
            trigger_count.iloc[position] = len(triggered)
            previous_count.iloc[position] = sum(
                bool(values.iloc[position - 1]) for values in criteria.values()
            )
            previous_position = last_pass_by_segment.get(segment_id)
            if previous_position is not None:
                sessions_since.iloc[position] = position - previous_position
        if is_pass:
            last_pass_by_segment[segment_id] = position
    return trigger_text, trigger_count, previous_count, sessions_since


def calculate_identity_signals(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    if rows.empty:
        return empty_signal_frame()
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")
    if rows.duplicated(["period_end_date"]).any():
        raise ValueError("one identity contains duplicate dates")
    identities = rows.loc[:, list(IDENTITY_COLUMNS)].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("calculate_identity_signals requires exactly one identity")

    rows = rows.sort_values("period_end_date", kind="stable")
    indexed = rows.set_index("period_end_date").reindex(trading_dates)
    observed = indexed["symbol"].notna()
    segment = _numeric(indexed["price_continuity_segment"])
    same_previous_segment = (
        observed
        & observed.shift(1, fill_value=False)
        & segment.notna()
        & segment.eq(segment.shift(1))
    )

    criteria = {
        column: indexed[column].astype("boolean") for column in CRITERION_COLUMNS
    }
    trend_pass = indexed["trend_template_pass"].astype("boolean")
    previous_pass = trend_pass.shift(1)
    signal_mask = (
        trend_pass.eq(True).fillna(False)
        & previous_pass.eq(False).fillna(False)
        & same_previous_segment
    )
    signal_positions = np.flatnonzero(signal_mask.to_numpy(dtype=bool))
    if not len(signal_positions):
        return empty_signal_frame()

    close = _numeric(indexed["adjusted_close"])
    high = _numeric(indexed["adjusted_high"])
    low = _numeric(indexed["adjusted_low"])
    volume = _numeric(indexed["adjusted_volume"])
    notional = _numeric(indexed["daily_traded_notional_usd"])
    rs_rating = _numeric(indexed["rs_rating"])

    previous_close = _lag_in_segment(close, segment, 1)
    daily_return = (close / previous_close - 1.0).where(
        close.gt(0) & previous_close.gt(0)
    )
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    normalized_true_range = (true_range / previous_close).where(previous_close.gt(0))

    prior_close = _lag_in_segment(close, segment, 1)
    prior_high20 = _rolling_in_segment(high, segment, 20, "max", offset=1)
    prior_high63 = _rolling_in_segment(high, segment, 63, "max", offset=1)
    prior_high126 = _rolling_in_segment(high, segment, 126, "max", offset=1)
    prior_high252 = _rolling_in_segment(high, segment, 252, "max", offset=1)
    recent_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=1
    )
    older_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=11
    )

    prior_volume5 = _rolling_in_segment(volume, segment, 5, "mean", offset=1)
    prior_volume7 = _rolling_in_segment(volume, segment, 7, "mean", offset=1)
    prior_volume10 = _rolling_in_segment(volume, segment, 10, "mean", offset=1)
    prior_volume14 = _rolling_in_segment(volume, segment, 14, "mean", offset=1)
    prior_volume21 = _rolling_in_segment(volume, segment, 21, "mean", offset=1)
    prior_volume50 = _rolling_in_segment(volume, segment, 50, "mean", offset=1)
    prior_volume100 = _rolling_in_segment(volume, segment, 100, "mean", offset=1)
    prior_notional5 = _rolling_in_segment(notional, segment, 5, "mean", offset=1)
    prior_notional7 = _rolling_in_segment(notional, segment, 7, "mean", offset=1)
    prior_notional10 = _rolling_in_segment(notional, segment, 10, "mean", offset=1)
    prior_notional14 = _rolling_in_segment(notional, segment, 14, "mean", offset=1)
    prior_notional21 = _rolling_in_segment(notional, segment, 21, "mean", offset=1)
    prior_notional50 = _rolling_in_segment(notional, segment, 50, "mean", offset=1)
    prior_notional100 = _rolling_in_segment(notional, segment, 100, "mean", offset=1)

    up_volume = volume.where(daily_return.gt(0), 0.0).where(
        volume.notna() & daily_return.notna()
    )
    nonflat_volume = volume.where(daily_return.ne(0), 0.0).where(
        volume.notna() & daily_return.notna()
    )
    up_notional = notional.where(daily_return.gt(0), 0.0).where(
        notional.notna() & daily_return.notna()
    )
    nonflat_notional = notional.where(daily_return.ne(0), 0.0).where(
        notional.notna() & daily_return.notna()
    )
    prior_up_volume = _rolling_in_segment(up_volume, segment, 21, "sum", offset=1)
    prior_nonflat_volume = _rolling_in_segment(
        nonflat_volume, segment, 21, "sum", offset=1
    )
    prior_up_notional = _rolling_in_segment(up_notional, segment, 21, "sum", offset=1)
    prior_nonflat_notional = _rolling_in_segment(
        nonflat_notional, segment, 21, "sum", offset=1
    )

    criteria_count = sum(
        values.astype("Float64") for values in criteria.values()
    ).astype(float)
    exactly_seven = criteria_count.eq(7).astype(float).where(observed)
    prior_seven_count = _rolling_in_segment(exactly_seven, segment, 10, "sum", offset=1)
    (
        trigger_text,
        trigger_count,
        previous_criteria_count,
        sessions_since_previous_pass,
    ) = _event_text_and_history(criteria, trend_pass, segment, signal_positions)

    ma50 = _numeric(indexed["ma50"])
    ma150 = _numeric(indexed["ma150"])
    ma200 = _numeric(indexed["ma200"])
    ma200_prior = _numeric(indexed["ma200_21_sessions_ago"])
    low52 = _numeric(indexed["low_52w"])
    high52 = _numeric(indexed["high_52w"])

    prior_return5 = _prior_return(close, segment, 5)
    prior_return10 = _prior_return(close, segment, 10)
    prior_return21 = _prior_return(close, segment, 21)
    prior_return42 = _prior_return(close, segment, 42)
    prior_return63 = _prior_return(close, segment, 63)
    prior_return126 = _prior_return(close, segment, 126)
    prior_return252 = _prior_return(close, segment, 252)
    prior_return_previous5 = _pct_ratio(
        _lag_in_segment(close, segment, 6),
        _lag_in_segment(close, segment, 11),
    ).where(_complete_prior_window(close, segment, 11))
    prior_momentum_acceleration = prior_return5 - prior_return_previous5
    prior_return_previous21 = _pct_ratio(
        _lag_in_segment(close, segment, 22),
        _lag_in_segment(close, segment, 43),
    ).where(_complete_prior_window(close, segment, 43))
    prior_return_acceleration21 = prior_return21 - prior_return_previous21

    prior_atr5 = _rolling_in_segment(true_range, segment, 5, "mean", offset=1)
    prior_atr10 = _rolling_in_segment(true_range, segment, 10, "mean", offset=1)
    prior_atr14 = _rolling_in_segment(true_range, segment, 14, "mean", offset=1)
    prior_atr21 = _rolling_in_segment(true_range, segment, 21, "mean", offset=1)
    prior_atr5_pct = (prior_atr5 / prior_close).mul(100.0).where(prior_close.gt(0))
    prior_atr14_pct = (prior_atr14 / prior_close).mul(100.0).where(prior_close.gt(0))
    prior_atr21_pct = (prior_atr21 / prior_close).mul(100.0).where(prior_close.gt(0))
    prior_daily_std21 = _rolling_in_segment(
        daily_return, segment, 21, "std", offset=1
    ).mul(100.0)
    prior_daily_std63 = _rolling_in_segment(
        daily_return, segment, 63, "std", offset=1
    ).mul(100.0)
    downside_return = daily_return.where(daily_return.lt(0), 0.0).where(
        daily_return.notna()
    )
    prior_downside_std21 = _rolling_in_segment(
        downside_return, segment, 21, "std", offset=1
    ).mul(100.0)
    max_drawdown21 = _event_max_drawdown(close, segment, signal_positions, 21)
    max_drawdown63 = _event_max_drawdown(close, segment, signal_positions, 63)
    max_drawdown126 = _event_max_drawdown(close, segment, signal_positions, 126)
    close_location = ((close - low) / (high - low)).where(high.gt(low))
    prior_normalized_range14 = _rolling_in_segment(
        normalized_true_range, segment, 14, "mean", offset=1
    )
    distribution_available = (
        daily_return.notna() & volume.notna() & prior_volume21.gt(0)
    )
    distribution_event = (
        daily_return.le(-0.002) & volume.ge(prior_volume21.mul(1.20))
    ).astype(float).where(distribution_available)
    churning_available = (
        daily_return.notna()
        & volume.notna()
        & prior_volume21.gt(0)
        & normalized_true_range.notna()
        & prior_normalized_range14.gt(0)
        & close_location.notna()
    )
    churning_event = (
        daily_return.abs().le(0.005)
        & volume.ge(prior_volume21.mul(1.20))
        & normalized_true_range.ge(prior_normalized_range14)
        & close_location.le(0.50)
    ).astype(float).where(churning_available)
    failed_breakout_available = (
        high.notna()
        & close.notna()
        & daily_return.notna()
        & prior_high20.gt(0)
    )
    failed_breakout_event = (
        high.ge(prior_high20.mul(1.001))
        & close.lt(prior_high20)
        & daily_return.lt(0)
    ).astype(float).where(failed_breakout_available)
    prior_distribution_count20 = _rolling_in_segment(
        distribution_event, segment, 20, "sum", offset=1
    )
    prior_churning_count20 = _rolling_in_segment(
        churning_event, segment, 20, "sum", offset=1
    )
    prior_failed_breakout_count20 = _rolling_in_segment(
        failed_breakout_event, segment, 20, "sum", offset=1
    )
    prior_chart_features = _event_prior_chart_features(
        close, high, low, segment, signal_positions
    )
    v3_chart_features = _event_v3_chart_features(
        close, high, low, segment, signal_positions
    )
    soft_pattern_features = _event_soft_pattern_features(
        close, high, low, volume, notional, segment, signal_positions
    )
    prior_rs_change5 = (
        _lag_in_segment(rs_rating, segment, 1) - _lag_in_segment(rs_rating, segment, 6)
    ).where(_complete_prior_window(rs_rating, segment, 6))
    prior_rs_change21 = (
        _lag_in_segment(rs_rating, segment, 1) - _lag_in_segment(rs_rating, segment, 22)
    ).where(_complete_prior_window(rs_rating, segment, 22))

    volume_vs_sma50 = _numeric(indexed["adjusted_volume_vs_sma50_prior_ratio"])
    volume_dryup = volume_vs_sma50.lt(0.75).astype(float).where(
        volume_vs_sma50.notna()
    )
    prior_volume_dryup_share10 = _rolling_in_segment(
        volume_dryup, segment, 10, "mean", offset=1
    )
    prior_volume_dryup_share20 = _rolling_in_segment(
        volume_dryup, segment, 20, "mean", offset=1
    )

    signed_volume = np.sign(daily_return).mul(volume).where(
        daily_return.notna() & volume.notna()
    )
    prior_signed_volume20 = _rolling_in_segment(
        signed_volume, segment, 20, "sum", offset=1
    )
    prior_total_volume20 = _rolling_in_segment(volume, segment, 20, "sum", offset=1)
    prior_obv_slope20 = (prior_signed_volume20 / prior_total_volume20).where(
        prior_total_volume20.gt(0)
    )
    accumulation_event = (
        daily_return.ge(0.005)
        & volume.ge(prior_volume21.mul(1.20))
        & close_location.ge(0.60)
    ).astype(float).where(distribution_available & close_location.notna())
    high_volume_down_event = (
        daily_return.lt(0)
        & volume.ge(prior_volume21.mul(1.50))
        & close_location.le(0.40)
    ).astype(float).where(distribution_available & close_location.notna())
    prior_accumulation_count20 = _rolling_in_segment(
        accumulation_event, segment, 20, "sum", offset=1
    )
    prior_high_volume_down_count20 = _rolling_in_segment(
        high_volume_down_event, segment, 20, "sum", offset=1
    )
    down_volume = volume.where(daily_return.lt(0), 0.0).where(
        daily_return.notna() & volume.notna()
    )
    prior_max_down_volume10 = _rolling_in_segment(
        down_volume, segment, 10, "max", offset=1
    )

    recent_normalized_range5 = _rolling_in_segment(
        normalized_true_range, segment, 5, "mean", offset=1
    )
    prior_normalized_range20 = _rolling_in_segment(
        normalized_true_range, segment, 20, "mean", offset=1
    )
    ma50_slope21 = _pct_ratio(ma50, _lag_in_segment(ma50, segment, 21))
    ma150_slope21 = _pct_ratio(ma150, _lag_in_segment(ma150, segment, 21))
    ma200_slope63 = _pct_ratio(ma200, _lag_in_segment(ma200, segment, 63))
    observed_run = (~same_previous_segment).cumsum()
    history_sessions = observed.groupby(observed_run, sort=False).cumsum().sub(1).where(
        observed
    )

    log_volume = np.log1p(volume.where(volume.ge(0)))
    log_notional = np.log1p(notional.where(notional.ge(0)))

    features: dict[str, pd.Series] = {
        "adjusted_volume_vs_sma7_prior_ratio": (volume / prior_volume7).where(
            prior_volume7.gt(0)
        ),
        "adjusted_volume_vs_sma14_prior_ratio": (volume / prior_volume14).where(
            prior_volume14.gt(0)
        ),
        "adjusted_volume_vs_sma100_prior_ratio": (volume / prior_volume100).where(
            prior_volume100.gt(0)
        ),
        "daily_traded_notional_vs_sma7_prior_ratio": (
            notional / prior_notional7
        ).where(prior_notional7.gt(0)),
        "daily_traded_notional_vs_sma14_prior_ratio": (
            notional / prior_notional14
        ).where(prior_notional14.gt(0)),
        "daily_traded_notional_vs_sma100_prior_ratio": (
            notional / prior_notional100
        ).where(prior_notional100.gt(0)),
        "trigger_criteria": trigger_text,
        "trigger_count": trigger_count,
        "previous_criteria_pass_count": previous_criteria_count,
        "prior_7_of_8_count_10d": prior_seven_count,
        "sessions_since_previous_pass": sessions_since_previous_pass,
        "distance_to_ma50_pct": _pct_ratio(close, ma50),
        "distance_to_ma150_pct": _pct_ratio(close, ma150),
        "distance_to_ma200_pct": _pct_ratio(close, ma200),
        "ma50_vs_ma150_pct": _pct_ratio(ma50, ma150),
        "ma50_vs_ma200_pct": _pct_ratio(ma50, ma200),
        "ma150_vs_ma200_pct": _pct_ratio(ma150, ma200),
        "ma200_slope_21d_pct": _pct_ratio(ma200, ma200_prior),
        "price_vs_52w_low_pct": _pct_ratio(close, low52),
        "price_vs_52w_high_pct": _pct_ratio(close, high52),
        "prior_return_5d_pct": prior_return5,
        "prior_return_10d_pct": prior_return10,
        "prior_return_21d_pct": prior_return21,
        "prior_adjusted_close": prior_close,
        "prior_return_42d_pct": prior_return42,
        "prior_return_63d_pct": prior_return63,
        "prior_return_126d_pct": prior_return126,
        "prior_return_252d_pct": prior_return252,
        "prior_return_acceleration_21d_pct_points": prior_return_acceleration21,
        "prior_daily_return_std_21d_pct": prior_daily_std21,
        "prior_daily_return_std_63d_pct": prior_daily_std63,
        "prior_downside_return_std_21d_pct": prior_downside_std21,
        "prior_max_drawdown_63d_pct": max_drawdown63,
        "prior_max_drawdown_126d_pct": max_drawdown126,
        "prior_momentum_acceleration_5d_pct_points": (prior_momentum_acceleration),
        "prior_atr_14d_pct": prior_atr14_pct,
        "prior_atr_5d_pct": prior_atr5_pct,
        "prior_atr_21d_pct": prior_atr21_pct,
        "prior_atr_5_vs21_ratio": (prior_atr5 / prior_atr21).where(
            prior_atr21.gt(0)
        ),
        "prior_atr_10_vs21_ratio": (prior_atr10 / prior_atr21).where(
            prior_atr21.gt(0)
        ),
        "prior_max_drawdown_21d_pct": max_drawdown21,
        "prior_close_vs_20d_high_pct": _pct_ratio(prior_close, prior_high20),
        "prior_close_vs_63d_high_pct": _pct_ratio(prior_close, prior_high63),
        "prior_close_vs_126d_high_pct": _pct_ratio(prior_close, prior_high126),
        "prior_close_vs_252d_high_pct": _pct_ratio(prior_close, prior_high252),
        "signal_close_vs_prior_20d_high_pct": _pct_ratio(close, prior_high20),
        "prior_range_compression_10_vs_10_ratio": (
            recent_normalized_range10 / older_normalized_range10
        ).where(older_normalized_range10.gt(0)),
        "signal_close_location_value": close_location,
        "prior_rs_rating_change_5d": prior_rs_change5,
        "prior_volume_sma5_vs21_ratio": (prior_volume5 / prior_volume21).where(
            prior_volume21.gt(0)
        ),
        "prior_volume_sma10_vs21_ratio": (prior_volume10 / prior_volume21).where(
            prior_volume21.gt(0)
        ),
        "prior_volume_sma5_vs50_ratio": (prior_volume5 / prior_volume50).where(
            prior_volume50.gt(0)
        ),
        "prior_volume_sma10_vs50_ratio": (prior_volume10 / prior_volume50).where(
            prior_volume50.gt(0)
        ),
        "prior_volume_sma21_vs50_ratio": (prior_volume21 / prior_volume50).where(
            prior_volume50.gt(0)
        ),
        "prior_notional_sma5_vs21_ratio": (prior_notional5 / prior_notional21).where(
            prior_notional21.gt(0)
        ),
        "prior_notional_sma10_vs21_ratio": (prior_notional10 / prior_notional21).where(
            prior_notional21.gt(0)
        ),
        "prior_notional_sma5_vs50_ratio": (prior_notional5 / prior_notional50).where(
            prior_notional50.gt(0)
        ),
        "prior_notional_sma10_vs50_ratio": (
            prior_notional10 / prior_notional50
        ).where(prior_notional50.gt(0)),
        "prior_notional_sma21_vs50_ratio": (
            prior_notional21 / prior_notional50
        ).where(prior_notional50.gt(0)),
        "prior_up_volume_share21": (prior_up_volume / prior_nonflat_volume).where(
            prior_nonflat_volume.gt(0)
        ),
        "prior_up_notional_share21": (prior_up_notional / prior_nonflat_notional).where(
            prior_nonflat_notional.gt(0)
        ),
        "prior_up_down_volume_ratio21": (
            prior_up_volume / (prior_nonflat_volume - prior_up_volume)
        ).where((prior_nonflat_volume - prior_up_volume).gt(0)),
        "prior_up_down_notional_ratio21": (
            prior_up_notional / (prior_nonflat_notional - prior_up_notional)
        ).where((prior_nonflat_notional - prior_up_notional).gt(0)),
        "prior_volume_dryup_share10": prior_volume_dryup_share10,
        "prior_volume_dryup_share20": prior_volume_dryup_share20,
        "prior_obv_slope_20": prior_obv_slope20,
        "prior_accumulation_day_count_20": prior_accumulation_count20,
        "prior_high_volume_down_day_count_20": prior_high_volume_down_count20,
        "prior_price_volume_corr21": _prior_correlation(
            daily_return, log_volume, segment, 21
        ),
        "prior_price_notional_corr21": _prior_correlation(
            daily_return, log_notional, segment, 21
        ),
        **prior_chart_features,
        **v3_chart_features,
        **soft_pattern_features,
        "prior_range_compression_5_vs20_ratio": (
            recent_normalized_range5 / prior_normalized_range20
        ).where(prior_normalized_range20.gt(0)),
        "ma50_slope_21d_pct": ma50_slope21,
        "ma150_slope_21d_pct": ma150_slope21,
        "ma200_slope_63d_pct": ma200_slope63,
        "prior_rs_rating_change_21d": prior_rs_change21,
        "prior_history_sessions": history_sessions,
        "signal_volume_vs_prior_10d_max_down_volume_ratio": (
            volume / prior_max_down_volume10
        ).where(prior_max_down_volume10.gt(0)),
        "prior_distribution_day_count_20": prior_distribution_count20,
        "prior_churning_day_count_20": prior_churning_count20,
        "prior_failed_breakout_count_20": prior_failed_breakout_count20,
    }

    signal_index = trading_dates[signal_positions]
    output = indexed.loc[
        signal_index,
        [
            *IDENTITY_COLUMNS,
            "price_continuity_segment",
            "currency",
            "raw_close",
            "adjusted_close",
            "adjusted_high",
            "adjusted_low",
            "adjusted_volume",
            "daily_price_change_pct",
            "adjusted_volume_sma21_prior",
            "adjusted_volume_vs_sma21_prior_ratio",
            "adjusted_volume_sma50_prior",
            "adjusted_volume_vs_sma50_prior_ratio",
            "daily_traded_notional_usd",
            "daily_traded_notional_sma21_prior_usd",
            "daily_traded_notional_vs_sma21_prior_ratio",
            "daily_traded_notional_sma50_prior_usd",
            "daily_traded_notional_vs_sma50_prior_ratio",
            "dollar_volume_63d",
            "ma50",
            "ma150",
            "ma200",
            "ma200_21_sessions_ago",
            "low_52w",
            "high_52w",
            "rs_raw",
            "rs_rating",
            "forward_5d_max_gain_pct",
            "forward_5d_max_loss_pct",
            "forward_10d_max_gain_pct",
            "forward_10d_max_loss_pct",
            "forward_20d_max_gain_pct",
            "forward_20d_max_loss_pct",
        ],
    ].copy()
    output.insert(0, "signal_date", signal_index)
    output.insert(
        1,
        "previous_session_date",
        trading_dates[signal_positions - 1],
    )
    feature_frame = pd.DataFrame(
        {
            name: values.loc[signal_index].to_numpy()
            for name, values in features.items()
        },
        index=output.index,
    )
    output = pd.concat([output, feature_frame], axis=1)

    output = output.reset_index(drop=True)
    output["signal_date"] = pd.to_datetime(output["signal_date"])
    output["previous_session_date"] = pd.to_datetime(output["previous_session_date"])
    in_date_range = output["signal_date"].dt.date >= cfg.signal_start_date
    if cfg.signal_end_date is not None:
        in_date_range &= output["signal_date"].dt.date <= cfg.signal_end_date
    kept_signal_positions = signal_positions[in_date_range.to_numpy(dtype=bool)]
    output = output.loc[in_date_range].reset_index(drop=True)
    if output.empty:
        return empty_signal_frame()

    path_outcomes = [
        _entry_path_outcome(indexed, trading_dates, segment, int(position), cfg)
        for position in kept_signal_positions
    ]
    for column in (
        *(f"forward_{day}d_max_gain_pct" for day in (5, 10, 20, 30, 40, 60, 90)),
        *(f"forward_{day}d_max_loss_pct" for day in (5, 10, 20, 30, 40, 60, 90)),
        *(f"forward_{day}d_label_end_date" for day in (5, 10, 20, 30, 40, 60, 90)),
        *(f"terminal_close_return_{day}d_pct" for day in (5, 10, 20, 30, 40, 60, 90)),
        "first_gain_2pct_day",
        "first_gain_1pct_day",
        "first_gain_3pct_day",
        "first_gain_5pct_day",
        "first_loss_5pct_day",
        "first_loss_10pct_day",
        "gain_loss_order_5d",
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
        "mfe_to_abs_mae_5d_ratio",
        "terminal_return_to_mfe_5d_ratio",
    ):
        output[column] = [item[column] for item in path_outcomes]

    gain5 = _numeric(output["forward_5d_max_gain_pct"])
    gain10 = _numeric(output["forward_10d_max_gain_pct"])
    gain20 = _numeric(output["forward_20d_max_gain_pct"])
    complete5 = pd.Series(
        [bool(item["_full_path_available"]) for item in path_outcomes],
        index=output.index,
    )
    complete10 = complete5 & gain10.notna()
    complete20 = complete5 & gain20.notna()
    output["late_strong_10d"] = _nullable_flag(
        output.index,
        complete10,
        lambda selected: (gain5.loc[selected] < cfg.strong_5d_min_gain_pct)
        & (gain10.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    output["late_strong_20d"] = _nullable_flag(
        output.index,
        complete20,
        lambda selected: (gain5.loc[selected] < cfg.strong_5d_min_gain_pct)
        & (gain20.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    output["analysis_split"] = _analysis_split(
        output["signal_date"], output["forward_5d_label_end_date"], cfg
    )
    output["include_weak_filter"] = True
    output["include_loss_first_filter"] = True
    output["include_final"] = True
    output["strong_confirmation"] = False
    output["weak_matched_rule_ids"] = pd.NA
    output["loss_first_matched_rule_ids"] = pd.NA
    output["matched_rule_ids"] = pd.NA
    output["confirmation_matched_rule_ids"] = pd.NA
    output["confirmation_reason"] = pd.NA
    output["filter_decision"] = "include"
    output["exclusion_reason"] = pd.NA

    # Fundamental data is joined once per bounded identity batch so that the
    # same point-in-time enrichment code is used for every worker. Direct
    # identity calculations retain the complete result contract with nulls.
    nullable_enrichment_columns = tuple(
        dict.fromkeys(
            (
                *FUNDAMENTAL_FEATURE_COLUMNS,
                *EARNINGS_EVENT_FEATURE_COLUMNS,
                *MARKET_CAP_FEATURE_COLUMNS,
                *SUPPLY_DEMAND_FEATURE_COLUMNS,
                *GLOBAL_MARKET_FEATURE_COLUMNS,
                *CURRENT_TAXONOMY_BACKCAST_LABEL_COLUMNS,
                *CURRENT_TAXONOMY_BACKCAST_FEATURE_COLUMNS,
            )
        )
    )
    missing_enrichment_columns = [
        column for column in nullable_enrichment_columns if column not in output
    ]
    if missing_enrichment_columns:
        output = output.reindex(
            columns=[*output.columns, *missing_enrichment_columns],
            fill_value=np.nan,
        )

    for column in SIGNAL_COLUMNS:
        if column not in output:
            raise AssertionError(f"calculation did not produce {column}")
    return output.loc[:, SIGNAL_COLUMNS]


def _empty_early_cut_row(signal: pd.Series, landmark_day: int) -> dict[str, Any]:
    row = {column: pd.NA for column in EARLY_CUT_COLUMNS}
    row.update(
        {
            "signal_date": pd.Timestamp(signal["signal_date"]),
            "landmark_day": int(landmark_day),
            "symbol": signal["symbol"],
            "exchange": signal["exchange"],
            "cik": signal["cik"],
            "price_continuity_segment": signal["price_continuity_segment"],
            "currency": signal["currency"],
            "decision_stage": (
                "early_cut"
                if landmark_day in EARLY_CUT_LANDMARK_DAYS
                else "stagnation_review"
                if landmark_day == 5
                else "profit_review"
            ),
            "landmark_observed": False,
            "same_continuity_segment": False,
            "eligible_at_landmark": False,
            "early_confirmation_eligible_at_landmark": False,
            "active_at_landmark": False,
            "prior_policy_cut_day": pd.NA,
            "full_outcome_available": False,
            "early_confirmation_outcome_available": False,
            "analysis_split": signal["analysis_split"],
            "include_stagnation_filter": False,
            "include_loss_filter": False,
            "include_final": False,
            "stagnation_matched_rule_ids": pd.NA,
            "loss_matched_rule_ids": pd.NA,
            "matched_rule_ids": pd.NA,
            "cut_decision": "not_evaluable",
            "cut_reason": pd.NA,
            "early_confirmation_include_final": False,
            "early_confirmation_matched_rule_ids": pd.NA,
            "early_confirmation_decision": "not_evaluable",
            "early_confirmation_reason": pd.NA,
            "management_include_final": False,
            "management_matched_rule_ids": pd.NA,
            "management_decision": "not_evaluable",
            "management_reason": pd.NA,
        }
    )
    return row


def _safe_scalar_ratio(
    numerator: Any,
    denominator: Any,
    *,
    percentage: bool = False,
) -> float:
    try:
        left = float(numerator)
        right = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(left) or not np.isfinite(right) or right <= 0:
        return np.nan
    value = left / right - (1.0 if percentage else 0.0)
    return float(value * 100.0 if percentage else value)


def _landmark_activity_ratio(
    indexed: pd.DataFrame,
    segment: pd.Series,
    position: int,
    segment_id: float,
    column: str,
    window: int,
) -> float:
    start = position - window
    if start < 0 or not _observed_same_segment(
        indexed, segment, start, position, segment_id
    ):
        return np.nan
    current = pd.to_numeric(
        pd.Series([indexed[column].iloc[position]]), errors="coerce"
    ).iloc[0]
    prior = pd.to_numeric(
        indexed[column].iloc[start:position], errors="coerce"
    ).astype(float)
    if (
        pd.isna(current)
        or not np.isfinite(float(current))
        or float(current) < 0
        or len(prior) != window
        or not np.isfinite(prior.to_numpy()).all()
    ):
        return np.nan
    mean = float(prior.mean())
    return float(current) / mean if mean > 0 else np.nan


def _landmark_technical_snapshot(
    indexed: pd.DataFrame,
    segment: pd.Series,
    position: int,
    segment_id: float,
) -> dict[str, float]:
    result = {
        "landmark_return_5d_pct": np.nan,
        "landmark_return_10d_pct": np.nan,
        "landmark_return_20d_pct": np.nan,
        "landmark_max_drawdown_20d_pct": np.nan,
        "landmark_trend_slope_20_pct_per_session": np.nan,
        "landmark_trend_r2_20": np.nan,
        "landmark_trend_efficiency_20": np.nan,
        "landmark_range_compression_10_vs_10_ratio": np.nan,
        "landmark_distribution_day_count_20": np.nan,
        "landmark_churning_day_count_20": np.nan,
    }
    close = _numeric(indexed["adjusted_close"])
    for window in (5, 10, 20):
        prior_position = position - window
        if prior_position >= 0 and _observed_same_segment(
            indexed, segment, prior_position, position, segment_id
        ):
            current = close.iloc[position]
            prior = close.iloc[prior_position]
            if _finite_positive(current) and _finite_positive(prior):
                result[f"landmark_return_{window}d_pct"] = (
                    float(current) / float(prior) - 1.0
                ) * 100.0
    start = position - 19
    if start < 0 or not _observed_same_segment(
        indexed, segment, start, position, segment_id
    ):
        return result
    closes = close.iloc[start : position + 1].to_numpy(dtype=float)
    highs = _numeric(indexed["adjusted_high"].iloc[start : position + 1]).to_numpy(
        dtype=float
    )
    lows = _numeric(indexed["adjusted_low"].iloc[start : position + 1]).to_numpy(
        dtype=float
    )
    volumes = _numeric(indexed["adjusted_volume"].iloc[start : position + 1]).to_numpy(
        dtype=float
    )
    volume_ratios = _numeric(
        indexed["adjusted_volume_vs_sma21_prior_ratio"].iloc[start : position + 1]
    ).to_numpy(dtype=float)
    daily_returns = _numeric(
        indexed["daily_price_change_pct"].iloc[start : position + 1]
    ).to_numpy(dtype=float)
    if not (
        np.isfinite(closes).all()
        and np.isfinite(highs).all()
        and np.isfinite(lows).all()
        and np.all(closes > 0)
        and np.all(highs > 0)
        and np.all(lows > 0)
    ):
        return result
    running_high = np.maximum.accumulate(closes)
    result["landmark_max_drawdown_20d_pct"] = float(
        np.min(closes / running_high - 1.0) * 100.0
    )
    x = np.arange(20, dtype=float)
    log_close = np.log(closes)
    slope, intercept = np.polyfit(x, log_close, 1)
    fitted = intercept + slope * x
    residual = float(np.square(log_close - fitted).sum())
    total = float(np.square(log_close - log_close.mean()).sum())
    result["landmark_trend_slope_20_pct_per_session"] = float(
        np.expm1(slope) * 100.0
    )
    result["landmark_trend_r2_20"] = 1.0 - residual / total if total > 0 else 0.0
    path = float(np.abs(np.diff(closes)).sum())
    result["landmark_trend_efficiency_20"] = (
        abs(float(closes[-1] - closes[0])) / path if path > 0 else 0.0
    )
    ranges = (highs - lows) / closes
    older = float(np.mean(ranges[:10]))
    result["landmark_range_compression_10_vs_10_ratio"] = (
        float(np.mean(ranges[10:])) / older if older > 0 else np.nan
    )
    close_location = np.divide(
        closes - lows,
        highs - lows,
        out=np.full(20, np.nan),
        where=highs > lows,
    )
    valid_activity = np.isfinite(volumes) & np.isfinite(volume_ratios)
    result["landmark_distribution_day_count_20"] = float(
        np.count_nonzero(valid_activity & (daily_returns <= -0.2) & (volume_ratios >= 1.2))
    )
    result["landmark_churning_day_count_20"] = float(
        np.count_nonzero(
            valid_activity
            & (np.abs(daily_returns) <= 0.5)
            & (volume_ratios >= 1.2)
            & (close_location <= 0.5)
        )
    )
    return result


def _early_confirmation_path_features(
    indexed: pd.DataFrame,
    segment: pd.Series,
    signal_position: int,
    landmark_position: int,
    segment_id: float,
    signal_prior_atr_pct: float,
) -> dict[str, float]:
    """Return path features known at the landmark close, without future bars."""

    columns = (
        "signal_adjusted_open",
        "signal_adjusted_high",
        "signal_adjusted_low",
        "signal_prior_adjusted_close",
        "signal_gap_pct",
        "close_vs_signal_high_pct",
        "signal_day_move_retention_ratio",
        "closes_above_signal_high_share_since_signal",
        "closes_above_signal_close_share_since_signal",
        "higher_high_share_since_signal",
        "higher_low_share_since_signal",
        "mean_close_location_since_signal",
        "path_efficiency_since_signal",
        "mfe_retention_ratio",
        "close_return_from_signal_atr_units",
        "max_gain_to_landmark_atr_units",
        "max_loss_to_landmark_atr_units",
        "drawdown_from_post_signal_high_atr_units",
        "up_volume_share_since_signal",
        "up_notional_share_since_signal",
        "pullback_volume_vs_advance_volume_ratio_since_signal",
        "pullback_notional_vs_advance_notional_ratio_since_signal",
        "breakout_acceptance_score",
        "early_path_quality_score",
    )
    result = {column: np.nan for column in columns}
    if signal_position <= 0 or landmark_position <= signal_position:
        return result
    if not _complete_price_path(
        indexed, segment, signal_position, landmark_position, segment_id
    ):
        return result

    signal = indexed.iloc[signal_position]
    prior_close = _finite_scalar(indexed["adjusted_close"].iloc[signal_position - 1])
    signal_open = _finite_scalar(signal.get("adjusted_open"))
    signal_close = float(signal["adjusted_close"])
    signal_high = float(signal["adjusted_high"])
    signal_low = float(signal["adjusted_low"])
    landmark_close = float(indexed["adjusted_close"].iloc[landmark_position])
    path = indexed.iloc[signal_position + 1 : landmark_position + 1]
    path_closes = _numeric(path["adjusted_close"]).to_numpy(dtype=float)
    path_highs = _numeric(path["adjusted_high"]).to_numpy(dtype=float)
    path_lows = _numeric(path["adjusted_low"]).to_numpy(dtype=float)
    path_volumes = _numeric(path["adjusted_volume"]).to_numpy(dtype=float)
    path_notionals = _numeric(
        path["daily_traded_notional_usd"]
    ).to_numpy(dtype=float)
    path_returns = _numeric(path["daily_price_change_pct"]).to_numpy(dtype=float)
    close_locations = np.divide(
        path_closes - path_lows,
        path_highs - path_lows,
        out=np.full(len(path), np.nan),
        where=path_highs > path_lows,
    )
    close_sequence = np.concatenate(([signal_close], path_closes))
    high_sequence = np.concatenate(([signal_high], path_highs))
    low_sequence = np.concatenate(([signal_low], path_lows))
    high_returns = (path_highs / signal_close - 1.0) * 100.0
    low_returns = (path_lows / signal_close - 1.0) * 100.0
    max_gain = max(0.0, float(high_returns.max()))
    max_loss = min(0.0, float(low_returns.min()))
    close_return = (landmark_close / signal_close - 1.0) * 100.0
    drawdown = (landmark_close / float(path_highs.max()) - 1.0) * 100.0
    result.update(
        {
            "signal_adjusted_open": signal_open,
            "signal_adjusted_high": signal_high,
            "signal_adjusted_low": signal_low,
            "signal_prior_adjusted_close": prior_close,
            "close_vs_signal_high_pct": (
                (landmark_close / signal_high - 1.0) * 100.0
                if signal_high > 0
                else np.nan
            ),
            "closes_above_signal_high_share_since_signal": float(
                np.mean(path_closes >= signal_high)
            ),
            "closes_above_signal_close_share_since_signal": float(
                np.mean(path_closes >= signal_close)
            ),
            "higher_high_share_since_signal": float(
                np.mean(np.diff(high_sequence) > 0)
            ),
            "higher_low_share_since_signal": float(
                np.mean(np.diff(low_sequence) > 0)
            ),
            "mean_close_location_since_signal": (
                float(np.mean(close_locations))
                if np.isfinite(close_locations).all()
                else np.nan
            ),
        }
    )
    if np.isfinite(prior_close) and prior_close > 0:
        if np.isfinite(signal_open) and signal_open > 0:
            result["signal_gap_pct"] = (signal_open / prior_close - 1.0) * 100.0
        signal_move = signal_close / prior_close - 1.0
        if signal_move > 0:
            result["signal_day_move_retention_ratio"] = (
                landmark_close / prior_close - 1.0
            ) / signal_move
    path_distance = float(np.abs(np.diff(close_sequence)).sum())
    if path_distance > 0:
        result["path_efficiency_since_signal"] = (
            landmark_close - signal_close
        ) / path_distance
    if max_gain > 0:
        result["mfe_retention_ratio"] = close_return / max_gain
    if np.isfinite(signal_prior_atr_pct) and signal_prior_atr_pct > 0:
        result["close_return_from_signal_atr_units"] = (
            close_return / signal_prior_atr_pct
        )
        result["max_gain_to_landmark_atr_units"] = (
            max_gain / signal_prior_atr_pct
        )
        result["max_loss_to_landmark_atr_units"] = (
            max_loss / signal_prior_atr_pct
        )
        result["drawdown_from_post_signal_high_atr_units"] = (
            drawdown / signal_prior_atr_pct
        )

    up = path_returns > 0
    down = path_returns < 0
    for activity, share_column, ratio_column in (
        (
            path_volumes,
            "up_volume_share_since_signal",
            "pullback_volume_vs_advance_volume_ratio_since_signal",
        ),
        (
            path_notionals,
            "up_notional_share_since_signal",
            "pullback_notional_vs_advance_notional_ratio_since_signal",
        ),
    ):
        if not np.isfinite(activity).all():
            continue
        up_activity = float(activity[up].sum())
        down_activity = float(activity[down].sum())
        nonflat_activity = up_activity + down_activity
        if nonflat_activity > 0:
            result[share_column] = up_activity / nonflat_activity
        if up_activity > 0:
            result[ratio_column] = down_activity / up_activity

    result["breakout_acceptance_score"] = 100.0 * _mean01(
        _linear01(result["close_vs_signal_high_pct"], -2.0, 1.0),
        _linear01(result["signal_day_move_retention_ratio"], 0.5, 1.1),
        _linear01(
            result["closes_above_signal_high_share_since_signal"], 0.0, 1.0
        ),
        _linear01(result["mean_close_location_since_signal"], 0.4, 0.8),
    )
    result["early_path_quality_score"] = 100.0 * _mean01(
        _linear01(result["close_return_from_signal_atr_units"], 0.0, 2.0),
        _linear01(result["max_loss_to_landmark_atr_units"], -2.0, 0.0),
        _linear01(result["path_efficiency_since_signal"], 0.0, 0.75),
        _linear01(result["mfe_retention_ratio"], 0.0, 1.0),
    )
    return result


def _common_landmark_values(
    indexed: pd.DataFrame,
    segment: pd.Series,
    true_range: pd.Series,
    landmark_atr14: pd.Series,
    signal: pd.Series,
    signal_position: int,
    landmark_position: int,
    cfg: Config,
) -> dict[str, Any]:
    segment_id = segment.iloc[signal_position]
    signal_close = float(indexed["adjusted_close"].iloc[signal_position])
    landmark = indexed.iloc[landmark_position]
    landmark_close = float(landmark["adjusted_close"])
    landmark_high = float(landmark["adjusted_high"])
    landmark_low = float(landmark["adjusted_low"])
    path = indexed.iloc[signal_position + 1 : landmark_position + 1]
    path_highs = _numeric(path["adjusted_high"]).to_numpy(dtype=float)
    path_lows = _numeric(path["adjusted_low"]).to_numpy(dtype=float)
    high_returns = (path_highs / signal_close - 1.0) * 100.0
    low_returns = (path_lows / signal_close - 1.0) * 100.0
    first_gain_2 = _first_hit_day(high_returns >= cfg.weak_5d_max_gain_pct, 1)
    first_gain_5 = _first_hit_day(high_returns >= cfg.strong_5d_min_gain_pct, 1)
    first_loss_5 = _first_hit_day(low_returns <= cfg.deep_loss_5d_max_loss_pct, 1)
    first_loss_10 = _first_hit_day(
        low_returns <= cfg.hard_stop_5d_max_loss_pct, 1
    )
    prior_close = float(indexed["adjusted_close"].iloc[landmark_position - 1])
    criteria_values = pd.Series(
        [landmark[column] for column in CRITERION_COLUMNS], dtype="boolean"
    )
    signal_volume = indexed["adjusted_volume"].iloc[signal_position]
    signal_notional = indexed["daily_traded_notional_usd"].iloc[signal_position]
    landmark_rs = landmark["rs_rating"]
    signal_rs = indexed["rs_rating"].iloc[signal_position]
    values: dict[str, Any] = {
        "signal_adjusted_close": signal_close,
        "landmark_adjusted_close": landmark_close,
        "landmark_adjusted_high": landmark_high,
        "landmark_adjusted_low": landmark_low,
        "landmark_adjusted_volume": landmark["adjusted_volume"],
        "landmark_daily_traded_notional_usd": landmark[
            "daily_traded_notional_usd"
        ],
        "landmark_daily_price_change_pct": landmark["daily_price_change_pct"],
        "landmark_volume_vs_sma21_prior_ratio": landmark[
            "adjusted_volume_vs_sma21_prior_ratio"
        ],
        "landmark_volume_vs_sma50_prior_ratio": landmark[
            "adjusted_volume_vs_sma50_prior_ratio"
        ],
        "landmark_notional_vs_sma21_prior_ratio": landmark[
            "daily_traded_notional_vs_sma21_prior_ratio"
        ],
        "landmark_notional_vs_sma50_prior_ratio": landmark[
            "daily_traded_notional_vs_sma50_prior_ratio"
        ],
        "landmark_rs_rating": landmark_rs,
        "landmark_criteria_pass_count": (
            int(criteria_values.sum()) if criteria_values.notna().all() else pd.NA
        ),
        "landmark_trend_template_pass": (
            pd.NA
            if pd.isna(landmark["trend_template_pass"])
            else bool(landmark["trend_template_pass"])
        ),
        "close_return_from_signal_pct": (landmark_close / signal_close - 1.0)
        * 100.0,
        "max_gain_to_landmark_pct": max(0.0, float(high_returns.max())),
        "max_loss_to_landmark_pct": min(0.0, float(low_returns.min())),
        "drawdown_from_post_signal_high_pct": (
            landmark_close / float(path_highs.max()) - 1.0
        )
        * 100.0,
        "rebound_from_post_signal_low_pct": (
            landmark_close / float(path_lows.min()) - 1.0
        )
        * 100.0,
        "landmark_true_range_pct": (
            float(true_range.iloc[landmark_position]) / prior_close * 100.0
            if pd.notna(true_range.iloc[landmark_position]) and prior_close > 0
            else np.nan
        ),
        "landmark_close_location_value": (
            (landmark_close - landmark_low) / (landmark_high - landmark_low)
            if landmark_high > landmark_low
            else np.nan
        ),
        "volume_vs_signal_ratio": _safe_scalar_ratio(
            landmark["adjusted_volume"], signal_volume
        ),
        "notional_vs_signal_ratio": _safe_scalar_ratio(
            landmark["daily_traded_notional_usd"], signal_notional
        ),
        "rs_rating_change_from_signal": (
            float(landmark_rs) - float(signal_rs)
            if pd.notna(landmark_rs) and pd.notna(signal_rs)
            else np.nan
        ),
        "landmark_distance_to_ma50_pct": _safe_scalar_ratio(
            landmark_close, landmark["ma50"], percentage=True
        ),
        "landmark_distance_to_ma150_pct": _safe_scalar_ratio(
            landmark_close, landmark["ma150"], percentage=True
        ),
        "landmark_distance_to_ma200_pct": _safe_scalar_ratio(
            landmark_close, landmark["ma200"], percentage=True
        ),
        "landmark_price_vs_52w_high_pct": _safe_scalar_ratio(
            landmark_close, landmark["high_52w"], percentage=True
        ),
        "landmark_atr_14d_pct": (
            float(landmark_atr14.iloc[landmark_position]) / landmark_close * 100.0
            if pd.notna(landmark_atr14.iloc[landmark_position])
            else np.nan
        ),
        "mean_volume_since_signal_vs_prior21_ratio": _safe_scalar_ratio(
            _numeric(path["adjusted_volume"]).mean(),
            indexed["adjusted_volume_sma21_prior"].iloc[signal_position],
        ),
        "mean_notional_since_signal_vs_prior21_ratio": _safe_scalar_ratio(
            _numeric(path["daily_traded_notional_usd"]).mean(),
            indexed["daily_traded_notional_sma21_prior_usd"].iloc[signal_position],
        ),
        "hit_gain_2pct_so_far": first_gain_2 is not None,
        "hit_gain_5pct_so_far": first_gain_5 is not None,
        "hit_loss_5pct_so_far": first_loss_5 is not None,
        "hit_loss_10pct_so_far": first_loss_10 is not None,
        "first_gain_2pct_day_so_far": pd.NA if first_gain_2 is None else first_gain_2,
        "first_gain_5pct_day_so_far": pd.NA if first_gain_5 is None else first_gain_5,
        "first_loss_5pct_day_so_far": pd.NA if first_loss_5 is None else first_loss_5,
        "first_loss_10pct_day_so_far": (
            pd.NA if first_loss_10 is None else first_loss_10
        ),
    }
    for window in (7, 14, 100):
        values[f"landmark_volume_vs_sma{window}_prior_ratio"] = (
            _landmark_activity_ratio(
                indexed,
                segment,
                landmark_position,
                segment_id,
                "adjusted_volume",
                window,
            )
        )
        values[f"landmark_notional_vs_sma{window}_prior_ratio"] = (
            _landmark_activity_ratio(
                indexed,
                segment,
                landmark_position,
                segment_id,
                "daily_traded_notional_usd",
                window,
            )
        )
    values.update(
        _landmark_technical_snapshot(
            indexed, segment, landmark_position, segment_id
        )
    )
    values.update(
        _early_confirmation_path_features(
            indexed,
            segment,
            signal_position,
            landmark_position,
            segment_id,
            _finite_scalar(signal.get("prior_atr_14d_pct")),
        )
    )
    signal_feature_mapping = {
        "signal_daily_price_change_pct": "daily_price_change_pct",
        "signal_volume_vs_sma21_prior_ratio": (
            "adjusted_volume_vs_sma21_prior_ratio"
        ),
        "signal_notional_vs_sma21_prior_ratio": (
            "daily_traded_notional_vs_sma21_prior_ratio"
        ),
        "signal_close_location_value": "signal_close_location_value",
        "signal_close_vs_prior_20d_high_pct": (
            "signal_close_vs_prior_20d_high_pct"
        ),
        "signal_prior_range_compression_10_vs_10_ratio": (
            "prior_range_compression_10_vs_10_ratio"
        ),
        "signal_prior_atr_14d_pct": "prior_atr_14d_pct",
        "signal_prior_base_width_20_pct": "prior_base_width_20_pct",
        "signal_prior_volume_sma5_vs21_ratio": "prior_volume_sma5_vs21_ratio",
        "signal_prior_notional_sma5_vs21_ratio": (
            "prior_notional_sma5_vs21_ratio"
        ),
        "signal_volume_dryup_breakout_score_20d": (
            "pattern_volume_dryup_breakout_score_20d"
        ),
        "signal_volume_dryup_breakout_notional_score_20d": (
            "pattern_volume_dryup_breakout_notional_score_20d"
        ),
    }
    for target, source in signal_feature_mapping.items():
        values[target] = _finite_scalar(signal.get(source))
    return values


def _populate_early_confirmation_outcome(
    row: dict[str, Any],
    indexed: pd.DataFrame,
    segment: pd.Series,
    signal_position: int,
    landmark_day: int,
    segment_id: float,
    cfg: Config,
) -> None:
    """Attach executable-next-open D20 outcomes to a D1-D3 row."""

    effective_position = signal_position + landmark_day + 1
    horizon_position = signal_position + 20
    if not (0 <= effective_position < len(indexed)):
        return
    effective_open = _finite_scalar(indexed["adjusted_open"].iloc[effective_position])
    if not np.isfinite(effective_open) or effective_open <= 0:
        return
    if not _complete_price_path(
        indexed, segment, effective_position, horizon_position, segment_id
    ):
        return

    future = indexed.iloc[effective_position : horizon_position + 1]
    highs = _numeric(future["adjusted_high"]).to_numpy(dtype=float)
    lows = _numeric(future["adjusted_low"]).to_numpy(dtype=float)
    high_returns = (highs / effective_open - 1.0) * 100.0
    low_returns = (lows / effective_open - 1.0) * 100.0
    terminal_close = float(indexed["adjusted_close"].iloc[horizon_position])
    terminal_return = (terminal_close / effective_open - 1.0) * 100.0
    max_gain = max(0.0, float(high_returns.max()))
    max_loss = min(0.0, float(low_returns.min()))
    first_gain = _first_hit_day(
        high_returns >= cfg.strong_5d_min_gain_pct,
        landmark_day + 1,
    )
    first_loss = _first_hit_day(
        low_returns <= cfg.deep_loss_5d_max_loss_pct,
        landmark_day + 1,
    )
    order = _gain_loss_order(first_gain, first_loss)

    row.update(
        {
            "early_confirmation_outcome_available": True,
            "effective_adjusted_open": effective_open,
            "effective_open_return_from_signal_pct": (
                effective_open / float(row["signal_adjusted_close"]) - 1.0
            )
            * 100.0,
            "max_gain_from_effective_open_to_day20_pct": max_gain,
            "max_loss_from_effective_open_to_day20_pct": max_loss,
            "terminal_return_from_effective_open_to_day20_pct": terminal_return,
            "take_profit_better_to_day20": terminal_return <= 0.0,
            "continue_winner_to_day20": bool(
                terminal_return >= cfg.continuation_winner_min_return_pct
                and max_loss > cfg.hard_stop_5d_max_loss_pct
            ),
            "early_first_gain_5pct_day_to_day20": (
                pd.NA if first_gain is None else first_gain
            ),
            "early_first_loss_5pct_day_to_day20": (
                pd.NA if first_loss is None else first_loss
            ),
            "early_gain_loss_order_to_day20": order,
            "early_winner_to_day20": bool(
                terminal_return >= cfg.continuation_winner_min_return_pct
                and max_loss > cfg.hard_stop_5d_max_loss_pct
            ),
            "early_bad_to_day20": bool(
                terminal_return <= 0.0
                or max_loss <= cfg.hard_stop_5d_max_loss_pct
            ),
        }
    )
    if order == "same_day_ambiguous":
        row["early_strong_first_to_day20"] = pd.NA
        row["early_loss_first_to_day20"] = pd.NA
    else:
        row["early_strong_first_to_day20"] = order in {
            "gain_first",
            "gain_only",
        }
        row["early_loss_first_to_day20"] = order in {
            "loss_first",
            "loss_only",
        }


def _early_cut_landmark_row(
    indexed: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segment: pd.Series,
    true_range: pd.Series,
    landmark_atr14: pd.Series,
    signal: pd.Series,
    signal_position: int,
    landmark_day: int,
    cfg: Config,
) -> dict[str, Any]:
    """Build one causal D1-D3 decision row and its censored outcome."""

    row = _empty_early_cut_row(signal, landmark_day)
    landmark_position = signal_position + landmark_day
    effective_position = landmark_position + 1
    horizon_position = signal_position + 5
    day20_position = signal_position + 20
    landmark_date = _date_at(trading_dates, landmark_position)
    effective_date = _date_at(trading_dates, effective_position)
    horizon_date = _date_at(trading_dates, horizon_position)
    row.update(
        {
            "landmark_date": landmark_date,
            "effective_session_date": effective_date,
            "horizon_end_date": horizon_date,
            "day20_end_date": _date_at(trading_dates, day20_position),
            # Early-cut folds are anchored at the close where their features
            # become known, not at the original entry signal.  The unchanged
            # D+5 horizon still determines whether a pre-holdout label belongs
            # wholly inside its split.
            "analysis_split": _analysis_split(
                pd.Series([landmark_date]),
                pd.Series([horizon_date]),
                cfg,
            ).iloc[0],
        }
    )

    if signal_position < 0 or signal_position >= len(indexed):
        return row
    segment_id = segment.iloc[signal_position]
    signal_close = indexed["adjusted_close"].iloc[signal_position]
    if _finite_positive(signal_close):
        row["signal_adjusted_close"] = float(signal_close)

    landmark_observed = False
    if 0 <= landmark_position < len(indexed):
        landmark_symbol = indexed["symbol"].iloc[landmark_position]
        signal_symbol = signal["symbol"]
        landmark_observed = bool(
            pd.notna(landmark_symbol)
            and pd.notna(signal_symbol)
            and landmark_symbol == signal_symbol
        )
    row["landmark_observed"] = landmark_observed
    # Eligibility is decided at the landmark close.  The following session is
    # only when a possible cut becomes effective; requiring that observation
    # here would leak future survivorship into the decision population.
    same_through_landmark = _observed_same_segment(
        indexed,
        segment,
        signal_position,
        landmark_position,
        segment_id,
    )
    row["same_continuity_segment"] = same_through_landmark
    complete_through_landmark = _complete_price_path(
        indexed,
        segment,
        signal_position,
        landmark_position,
        segment_id,
    )
    full_outcome_available = _complete_price_path(
        indexed,
        segment,
        signal_position,
        horizon_position,
        segment_id,
    )
    row["full_outcome_available"] = full_outcome_available

    if complete_through_landmark:
        signal_close = float(signal_close)
        landmark = indexed.iloc[landmark_position]
        landmark_close = float(landmark["adjusted_close"])
        landmark_high = float(landmark["adjusted_high"])
        landmark_low = float(landmark["adjusted_low"])
        path = indexed.iloc[signal_position + 1 : landmark_position + 1]
        path_highs = _numeric(path["adjusted_high"]).to_numpy(dtype=float)
        path_lows = _numeric(path["adjusted_low"]).to_numpy(dtype=float)
        high_returns = (path_highs / signal_close - 1.0) * 100.0
        low_returns = (path_lows / signal_close - 1.0) * 100.0
        max_gain = max(0.0, float(high_returns.max()))
        max_loss = min(0.0, float(low_returns.min()))
        first_gain_2 = _first_hit_day(high_returns >= cfg.weak_5d_max_gain_pct, 1)
        first_gain_5 = _first_hit_day(high_returns >= cfg.strong_5d_min_gain_pct, 1)
        first_loss_5 = _first_hit_day(low_returns <= cfg.deep_loss_5d_max_loss_pct, 1)
        hit_gain_2 = first_gain_2 is not None
        hit_gain_5 = first_gain_5 is not None
        hit_loss_5 = first_loss_5 is not None

        prior_close = float(indexed["adjusted_close"].iloc[landmark_position - 1])
        current_true_range = true_range.iloc[landmark_position]
        current_atr14 = landmark_atr14.iloc[landmark_position]
        landmark_volume = landmark["adjusted_volume"]
        landmark_notional = landmark["daily_traded_notional_usd"]
        signal_volume = indexed["adjusted_volume"].iloc[signal_position]
        signal_notional = indexed["daily_traded_notional_usd"].iloc[signal_position]
        signal_rs = indexed["rs_rating"].iloc[signal_position]
        landmark_rs = landmark["rs_rating"]
        criteria_values = pd.Series(
            [landmark[column] for column in CRITERION_COLUMNS],
            dtype="boolean",
        )
        criteria_count: Any = (
            int(criteria_values.sum()) if criteria_values.notna().all() else pd.NA
        )
        trend_value = landmark["trend_template_pass"]

        row.update(
            {
                "landmark_adjusted_close": landmark_close,
                "landmark_adjusted_high": landmark_high,
                "landmark_adjusted_low": landmark_low,
                "landmark_adjusted_volume": landmark_volume,
                "landmark_daily_traded_notional_usd": landmark_notional,
                "landmark_daily_price_change_pct": landmark["daily_price_change_pct"],
                "landmark_volume_vs_sma21_prior_ratio": landmark[
                    "adjusted_volume_vs_sma21_prior_ratio"
                ],
                "landmark_volume_vs_sma50_prior_ratio": landmark[
                    "adjusted_volume_vs_sma50_prior_ratio"
                ],
                "landmark_notional_vs_sma21_prior_ratio": landmark[
                    "daily_traded_notional_vs_sma21_prior_ratio"
                ],
                "landmark_notional_vs_sma50_prior_ratio": landmark[
                    "daily_traded_notional_vs_sma50_prior_ratio"
                ],
                "landmark_rs_rating": landmark_rs,
                "landmark_criteria_pass_count": criteria_count,
                "landmark_trend_template_pass": (
                    pd.NA if pd.isna(trend_value) else bool(trend_value)
                ),
                "close_return_from_signal_pct": (
                    (landmark_close / signal_close - 1.0) * 100.0
                ),
                "max_gain_to_landmark_pct": max_gain,
                "max_loss_to_landmark_pct": max_loss,
                "drawdown_from_post_signal_high_pct": (
                    (landmark_close / float(path_highs.max()) - 1.0) * 100.0
                ),
                "rebound_from_post_signal_low_pct": (
                    (landmark_close / float(path_lows.min()) - 1.0) * 100.0
                ),
                "landmark_true_range_pct": (
                    float(current_true_range) / prior_close * 100.0
                    if pd.notna(current_true_range) and prior_close > 0
                    else np.nan
                ),
                "landmark_close_location_value": (
                    (landmark_close - landmark_low) / (landmark_high - landmark_low)
                    if landmark_high > landmark_low
                    else np.nan
                ),
                "volume_vs_signal_ratio": _safe_scalar_ratio(
                    landmark_volume, signal_volume
                ),
                "notional_vs_signal_ratio": _safe_scalar_ratio(
                    landmark_notional, signal_notional
                ),
                "rs_rating_change_from_signal": (
                    float(landmark_rs) - float(signal_rs)
                    if pd.notna(landmark_rs)
                    and pd.notna(signal_rs)
                    and np.isfinite(float(landmark_rs))
                    and np.isfinite(float(signal_rs))
                    else np.nan
                ),
                "landmark_distance_to_ma50_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma50"], percentage=True
                ),
                "landmark_distance_to_ma150_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma150"], percentage=True
                ),
                "landmark_distance_to_ma200_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma200"], percentage=True
                ),
                "landmark_price_vs_52w_high_pct": _safe_scalar_ratio(
                    landmark_close, landmark["high_52w"], percentage=True
                ),
                "landmark_atr_14d_pct": (
                    float(current_atr14) / landmark_close * 100.0
                    if pd.notna(current_atr14) and np.isfinite(float(current_atr14))
                    else np.nan
                ),
                "mean_volume_since_signal_vs_prior21_ratio": (
                    _safe_scalar_ratio(
                        _numeric(path["adjusted_volume"]).mean(),
                        indexed["adjusted_volume_sma21_prior"].iloc[signal_position],
                    )
                ),
                "mean_notional_since_signal_vs_prior21_ratio": (
                    _safe_scalar_ratio(
                        _numeric(path["daily_traded_notional_usd"]).mean(),
                        indexed["daily_traded_notional_sma21_prior_usd"].iloc[
                            signal_position
                        ],
                    )
                ),
                "hit_gain_2pct_so_far": hit_gain_2,
                "hit_gain_5pct_so_far": hit_gain_5,
                "hit_loss_5pct_so_far": hit_loss_5,
                "first_gain_2pct_day_so_far": (
                    pd.NA if first_gain_2 is None else first_gain_2
                ),
                "first_gain_5pct_day_so_far": (
                    pd.NA if first_gain_5 is None else first_gain_5
                ),
                "first_loss_5pct_day_so_far": (
                    pd.NA if first_loss_5 is None else first_loss_5
                ),
            }
        )
        row.update(
            _common_landmark_values(
                indexed,
                segment,
                true_range,
                landmark_atr14,
                signal,
                signal_position,
                landmark_position,
                cfg,
            )
        )
        row["eligible_at_landmark"] = bool(
            same_through_landmark and not hit_gain_5 and not hit_loss_5
        )
        row["early_confirmation_eligible_at_landmark"] = bool(
            same_through_landmark and not bool(row["hit_loss_10pct_so_far"])
        )
        row["active_at_landmark"] = row["eligible_at_landmark"]
        row["cut_decision"] = (
            "hold"
            if row["eligible_at_landmark"]
            else "not_eligible" if same_through_landmark else "not_evaluable"
        )
        if row["eligible_at_landmark"]:
            row["include_stagnation_filter"] = True
            row["include_loss_filter"] = True
            row["include_final"] = True

    if full_outcome_available:
        signal_close = float(indexed["adjusted_close"].iloc[signal_position])
        landmark_close = float(indexed["adjusted_close"].iloc[landmark_position])
        future = indexed.iloc[effective_position : horizon_position + 1]
        future_highs = _numeric(future["adjusted_high"]).to_numpy(dtype=float)
        future_lows = _numeric(future["adjusted_low"]).to_numpy(dtype=float)
        signal_high_returns = (future_highs / signal_close - 1.0) * 100.0
        signal_low_returns = (future_lows / signal_close - 1.0) * 100.0
        landmark_high_returns = (future_highs / landmark_close - 1.0) * 100.0
        landmark_low_returns = (future_lows / landmark_close - 1.0) * 100.0
        first_future_gain_2 = _first_hit_day(
            landmark_high_returns >= cfg.weak_5d_max_gain_pct,
            landmark_day + 1,
        )
        first_future_gain_5 = _first_hit_day(
            landmark_high_returns >= cfg.strong_5d_min_gain_pct,
            landmark_day + 1,
        )
        first_future_loss_5 = _first_hit_day(
            landmark_low_returns <= cfg.deep_loss_5d_max_loss_pct,
            landmark_day + 1,
        )
        terminal_close = float(indexed["adjusted_close"].iloc[horizon_position])
        row.update(
            {
                "remaining_max_gain_from_signal_pct": max(
                    0.0, float(signal_high_returns.max())
                ),
                "remaining_max_loss_from_signal_pct": min(
                    0.0, float(signal_low_returns.min())
                ),
                "remaining_max_gain_from_landmark_pct": max(
                    0.0, float(landmark_high_returns.max())
                ),
                "remaining_max_loss_from_landmark_pct": min(
                    0.0, float(landmark_low_returns.min())
                ),
                "terminal_close_return_from_signal_pct": (
                    (terminal_close / signal_close - 1.0) * 100.0
                ),
                "terminal_close_return_from_landmark_pct": (
                    (terminal_close / landmark_close - 1.0) * 100.0
                ),
                "future_first_gain_2pct_day": (
                    pd.NA if first_future_gain_2 is None else first_future_gain_2
                ),
                "future_first_gain_5pct_day": (
                    pd.NA if first_future_gain_5 is None else first_future_gain_5
                ),
                "future_first_loss_5pct_day": (
                    pd.NA if first_future_loss_5 is None else first_future_loss_5
                ),
            }
        )

        if bool(row["eligible_at_landmark"]):
            continuation = _continuation_order(first_future_gain_5, first_future_loss_5)
            if continuation == "neither":
                continuation = (
                    "stagnant"
                    if float(landmark_high_returns.max())
                    < cfg.weak_5d_max_gain_pct
                    else "neutral"
                )
            row["continuation_outcome"] = continuation
            if continuation == "same_session_ambiguous":
                row["stagnant_to_day5"] = pd.NA
                row["loss_first_to_day5"] = pd.NA
                row["strong_first_to_day5"] = pd.NA
                row["bad_to_day5"] = pd.NA
            else:
                row["stagnant_to_day5"] = continuation == "stagnant"
                row["loss_first_to_day5"] = continuation == "loss_first"
                row["strong_first_to_day5"] = continuation == "strong_first"
                row["bad_to_day5"] = continuation in ("stagnant", "loss_first")

    if complete_through_landmark:
        _populate_early_confirmation_outcome(
            row,
            indexed,
            segment,
            signal_position,
            landmark_day,
            segment_id,
            cfg,
        )

    return row


def _management_landmark_row(
    indexed: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segment: pd.Series,
    true_range: pd.Series,
    landmark_atr14: pd.Series,
    signal: pd.Series,
    signal_position: int,
    landmark_day: int,
    cfg: Config,
) -> dict[str, Any]:
    """Build a causal D5/D20/D30 decision row with next-open outcomes."""

    row = _empty_early_cut_row(signal, landmark_day)
    landmark_position = signal_position + landmark_day
    effective_position = landmark_position + 1
    end_dates = {
        horizon: _date_at(trading_dates, signal_position + horizon)
        for horizon in (20, 40, 60, 90)
    }
    row.update(
        {
            "landmark_date": _date_at(trading_dates, landmark_position),
            "effective_session_date": _date_at(trading_dates, effective_position),
            "horizon_end_date": end_dates[90],
            **{f"day{day}_end_date": value for day, value in end_dates.items()},
        }
    )
    row["analysis_split"] = _analysis_split(
        pd.Series([row["landmark_date"]]),
        pd.Series([row["horizon_end_date"]]),
        cfg,
    ).iloc[0]
    if signal_position < 0 or signal_position >= len(indexed):
        return row
    segment_id = segment.iloc[signal_position]
    row["landmark_observed"] = bool(
        0 <= landmark_position < len(indexed)
        and pd.notna(indexed["symbol"].iloc[landmark_position])
        and indexed["symbol"].iloc[landmark_position] == signal["symbol"]
    )
    same_through_landmark = _observed_same_segment(
        indexed, segment, signal_position, landmark_position, segment_id
    )
    row["same_continuity_segment"] = same_through_landmark
    complete_through_landmark = _complete_price_path(
        indexed, segment, signal_position, landmark_position, segment_id
    )
    if not complete_through_landmark:
        return row

    row.update(
        _common_landmark_values(
            indexed,
            segment,
            true_range,
            landmark_atr14,
            signal,
            signal_position,
            landmark_position,
            cfg,
        )
    )
    hard_stop_hit = bool(row["hit_loss_10pct_so_far"])
    if landmark_day == 5:
        stagnant = bool(
            float(row["max_gain_to_landmark_pct"])
            < cfg.weak_5d_max_gain_pct
            and float(row["close_return_from_signal_pct"])
            <= cfg.stagnant_5d_max_return_pct
        )
        row["stagnant_at_day5"] = stagnant
        eligible = stagnant and not hard_stop_hit
    else:
        eligible = (
            float(row["close_return_from_signal_pct"]) > 0.0
            and not hard_stop_hit
        )
    row["eligible_at_landmark"] = eligible
    row["management_include_final"] = eligible
    row["management_decision"] = (
        "hard_stop"
        if hard_stop_hit
        else "hold" if eligible else "not_eligible"
    )

    effective_open = (
        indexed["adjusted_open"].iloc[effective_position]
        if 0 <= effective_position < len(indexed)
        and "adjusted_open" in indexed
        else np.nan
    )
    if _finite_positive(effective_open) and _observed_same_segment(
        indexed, segment, effective_position, effective_position, segment_id
    ):
        effective_open = float(effective_open)
        row["effective_adjusted_open"] = effective_open
        row["effective_open_return_from_signal_pct"] = (
            effective_open / float(row["signal_adjusted_close"]) - 1.0
        ) * 100.0
        applicable_horizons = (
            (20,) if landmark_day == 5 else (40, 60, 90)
        )
        available_horizons: list[int] = []
        for horizon in applicable_horizons:
            horizon_position = signal_position + horizon
            if not _complete_price_path(
                indexed,
                segment,
                effective_position,
                horizon_position,
                segment_id,
            ):
                continue
            future = indexed.iloc[effective_position : horizon_position + 1]
            highs = _numeric(future["adjusted_high"]).to_numpy(dtype=float)
            lows = _numeric(future["adjusted_low"]).to_numpy(dtype=float)
            terminal = float(indexed["adjusted_close"].iloc[horizon_position])
            max_gain = max(
                0.0, float(((highs / effective_open - 1.0) * 100.0).max())
            )
            max_loss = min(
                0.0, float(((lows / effective_open - 1.0) * 100.0).min())
            )
            terminal_return = (terminal / effective_open - 1.0) * 100.0
            row[f"max_gain_from_effective_open_to_day{horizon}_pct"] = max_gain
            row[f"max_loss_from_effective_open_to_day{horizon}_pct"] = max_loss
            row[
                f"terminal_return_from_effective_open_to_day{horizon}_pct"
            ] = terminal_return
            row[f"take_profit_better_to_day{horizon}"] = terminal_return <= 0.0
            row[f"continue_winner_to_day{horizon}"] = bool(
                terminal_return >= cfg.continuation_winner_min_return_pct
                and max_loss > cfg.hard_stop_5d_max_loss_pct
            )
            available_horizons.append(horizon)
        row["full_outcome_available"] = set(applicable_horizons).issubset(
            available_horizons
        )
    return row


def calculate_early_cut_landmarks(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return causal early-cut and position-management decision landmarks."""

    if signals is None:
        signals = calculate_identity_signals(rows, trading_dates, cfg)
    if signals.empty:
        return empty_early_cut_frame()
    if rows.empty:
        raise ValueError("non-empty signals require identity source rows")
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")

    ordered = rows.sort_values("period_end_date", kind="stable").copy()
    ordered["period_end_date"] = pd.to_datetime(
        ordered["period_end_date"], errors="raise"
    ).dt.normalize()
    indexed = ordered.set_index("period_end_date").reindex(trading_dates)
    segment = _numeric(indexed["price_continuity_segment"])
    close = _numeric(indexed["adjusted_close"])
    high = _numeric(indexed["adjusted_high"])
    low = _numeric(indexed["adjusted_low"])
    previous_close = _lag_in_segment(close, segment, 1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    landmark_atr14 = _rolling_in_segment(true_range, segment, 14, "mean")
    positions_by_date = {
        pd.Timestamp(value): position for position, value in enumerate(trading_dates)
    }

    output_rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        if signal_date not in positions_by_date:
            raise ValueError("a signal date is absent from trading_dates")
        signal_position = positions_by_date[signal_date]
        for landmark_day in EARLY_CUT_LANDMARK_DAYS:
            output_rows.append(
                _early_cut_landmark_row(
                    indexed,
                    trading_dates,
                    segment,
                    true_range,
                    landmark_atr14,
                    signal,
                    signal_position,
                    landmark_day,
                    cfg,
                )
            )
        for landmark_day in MANAGEMENT_LANDMARK_DAYS:
            output_rows.append(
                _management_landmark_row(
                    indexed,
                    trading_dates,
                    segment,
                    true_range,
                    landmark_atr14,
                    signal,
                    signal_position,
                    landmark_day,
                    cfg,
                )
            )

    output = pd.DataFrame(output_rows, columns=EARLY_CUT_COLUMNS)
    expected = len(signals) * len(POSITION_LANDMARK_DAYS)
    if len(output) != expected:
        raise AssertionError(
            f"expected {expected} landmark rows, calculated {len(output)}"
        )
    return output.loc[:, EARLY_CUT_COLUMNS]


def calculate_identity_results(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> CalculationBatchResult:
    """Calculate entry signals and their paired landmark observations."""

    signals = calculate_identity_signals(rows, trading_dates, cfg)
    early_cut = calculate_early_cut_landmarks(rows, trading_dates, cfg, signals=signals)
    return CalculationBatchResult(signals=signals, early_cut=early_cut)


def calculate_signal_batch(
    source: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
    *,
    fundamental_snapshots: pd.DataFrame | None = None,
    quarterly_fundamental_events: pd.DataFrame | None = None,
    earnings_events: pd.DataFrame | None = None,
) -> CalculationBatchResult:
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError("source data is missing columns: " + ", ".join(missing))
    if source.empty:
        return empty_calculation_batch()

    data = source.copy()
    if "adjusted_open" not in data:
        data["adjusted_open"] = np.nan
    data["period_end_date"] = pd.to_datetime(
        data["period_end_date"], errors="raise"
    ).dt.normalize()
    if data.duplicated([*IDENTITY_COLUMNS, "period_end_date"]).any():
        raise ValueError("source contains duplicate identity/date rows")
    if data.loc[:, list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError("source identity columns must not be null")
    currency = data["currency"].astype("string").str.upper()
    if currency.isna().any() or currency.ne("USD").any():
        raise ValueError("source currency must be USD")
    data["currency"] = currency
    for column in SOURCE_BOOLEAN_COLUMNS:
        data[column] = data[column].astype(bool)

    results = [
        calculate_identity_results(group, trading_dates, cfg)
        for _, group in data.groupby(list(IDENTITY_COLUMNS), sort=True, observed=True)
    ]
    signal_frames = [result.signals for result in results if not result.signals.empty]
    early_cut_frames = [
        result.early_cut for result in results if not result.early_cut.empty
    ]
    if not signal_frames:
        return empty_calculation_batch()
    signals = pd.concat(signal_frames, ignore_index=True).loc[:, SIGNAL_COLUMNS]
    snapshots = (
        fundamental_snapshots
        if fundamental_snapshots is not None
        else pd.DataFrame(columns=FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS)
    )
    events = (
        quarterly_fundamental_events
        if quarterly_fundamental_events is not None
        else pd.DataFrame(columns=QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS)
    )
    earnings = (
        earnings_events
        if earnings_events is not None
        else pd.DataFrame(columns=EARNINGS_EVENT_SOURCE_COLUMNS)
    )
    signals = enrich_signal_fundamentals(signals, snapshots, events, earnings)
    early_cut = pd.concat(early_cut_frames, ignore_index=True).loc[:, EARLY_CUT_COLUMNS]
    signals = _normalize_result_dtypes(
        signals,
        integer_columns=SIGNAL_INTEGER_COLUMNS,
        boolean_columns=SIGNAL_BOOLEAN_COLUMNS,
        text_columns=_SIGNAL_TEXT_COLUMNS,
        date_columns=_SIGNAL_DATE_COLUMNS,
    )
    early_cut = _normalize_result_dtypes(
        early_cut,
        integer_columns=EARLY_CUT_INTEGER_COLUMNS,
        boolean_columns=EARLY_CUT_BOOLEAN_COLUMNS,
        text_columns=_EARLY_CUT_TEXT_COLUMNS,
        date_columns=_EARLY_CUT_DATE_COLUMNS,
    )
    expected = len(signals) * len(POSITION_LANDMARK_DAYS)
    if len(early_cut) != expected:
        raise AssertionError(
            f"expected {expected} landmark rows, calculated {len(early_cut)}"
        )
    return CalculationBatchResult(signals=signals, early_cut=early_cut)
