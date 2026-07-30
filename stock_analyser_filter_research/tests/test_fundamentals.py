from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research.computation import (
    _signal_decision_timestamp,
    enrich_signal_fundamentals,
)
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_FEATURE_GROUPS,
    EARNINGS_EVENT_FEATURE_COLUMNS,
    EARNINGS_EVENT_SOURCE_COLUMNS,
    ENTRY_FEATURE_GROUPS,
    FUNDAMENTAL_FEATURE_COLUMNS,
    FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
    QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
    SIGNAL_COLUMNS,
)


IDENTITY = {"symbol": "ABC", "exchange": "NYSE", "cik": 123}


def test_fundamentals_compete_only_in_entry_group_f() -> None:
    assert ENTRY_FEATURE_GROUPS["F"] == FUNDAMENTAL_FEATURE_COLUMNS
    early_features = {
        feature
        for features in EARLY_CUT_FEATURE_GROUPS.values()
        for feature in features
    }
    assert not early_features.intersection(FUNDAMENTAL_FEATURE_COLUMNS)


def _signals(signal_date: str = "2026-07-15") -> pd.DataFrame:
    row = {column: pd.NA for column in SIGNAL_COLUMNS}
    row.update(IDENTITY)
    row["signal_date"] = pd.Timestamp(signal_date)
    return pd.DataFrame([row], columns=SIGNAL_COLUMNS)


def _snapshot(**overrides: object) -> dict[str, object]:
    row = {column: pd.NA for column in FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS}
    row.update(
        {
            **IDENTITY,
            "period_end_date": "2026-03-31",
            "sec_fundamental_currency": "USD",
            "sec_latest_period_end_date": "2026-03-31",
            "sec_data_available_at": "2026-07-15T19:59:00Z",
            "sec_revenue_ttm": 1_000,
            "sec_share_based_compensation_ttm": 50,
            "sec_gross_margin_ttm": 0.40,
            "sec_operating_margin_ttm": 0.11,
            "sec_net_margin_ttm": 0.08,
            "sec_fcf_margin_ttm": 0.09,
            "sec_fcf_sbc_adjusted_margin_ttm": 0.04,
            "sec_debt_to_capital": 0.30,
            "sec_cash_to_assets": 0.15,
            "sec_current_ratio": 1.5,
            "sec_accruals_ratio": -0.03,
        }
    )
    row.update(overrides)
    return row


def _event(**overrides: object) -> dict[str, object]:
    row = {column: pd.NA for column in QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS}
    row.update(
        {
            **IDENTITY,
            "accession_number": "0001",
            "accepted_at": "2026-07-15T19:59:00Z",
            "effective_date": "2026-07-15",
            "fiscal_period_end_date": "2026-03-31",
            "diluted_eps": 2.0,
            "prior_year_diluted_eps": 1.0,
            "currency": "USD",
            "quarterly_revenue": 120.0,
            "prior_year_quarterly_revenue": 100.0,
            "quarterly_operating_margin": 0.12,
            "prior_year_quarterly_operating_margin": 0.10,
            "quarterly_net_margin": 0.08,
            "prior_year_quarterly_net_margin": 0.06,
        }
    )
    row.update(overrides)
    return row


def _snapshot_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS)


def _event_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS)


def _earnings(**overrides: object) -> dict[str, object]:
    row = {column: pd.NA for column in EARNINGS_EVENT_SOURCE_COLUMNS}
    row.update(
        {
            **IDENTITY,
            "earnings_date": "2026-07-15",
            "announcement_ts": "2026-07-15T19:00:00Z",
            "source": "sec_8k_item_2_02",
            "source_event_id": "sec-event-1",
            "known_as_of_ts": "2026-07-15T19:00:00Z",
            "is_confirmed": True,
        }
    )
    row.update(overrides)
    return row


def test_decision_boundary_is_1600_new_york_with_dst() -> None:
    assert _signal_decision_timestamp("2026-01-15") == pd.Timestamp(
        "2026-01-15T21:00:00Z"
    )
    assert _signal_decision_timestamp("2026-07-15") == pd.Timestamp(
        "2026-07-15T20:00:00Z"
    )


def test_same_day_after_close_data_and_future_amendments_are_excluded() -> None:
    signals = _signals()
    public_snapshot = _snapshot()
    after_close_snapshot = _snapshot(
        period_end_date="2026-06-30",
        sec_latest_period_end_date="2026-06-30",
        sec_data_available_at="2026-07-15T20:01:00Z",
        sec_operating_margin_ttm=0.99,
    )
    public_event = _event()
    after_close_event = _event(
        accession_number="0002",
        fiscal_period_end_date="2026-06-30",
        accepted_at="2026-07-15T20:01:00Z",
        quarterly_revenue=999.0,
    )

    baseline = enrich_signal_fundamentals(
        signals,
        _snapshot_frame(public_snapshot),
        _event_frame(public_event),
    )
    with_future_mutations = enrich_signal_fundamentals(
        signals,
        _snapshot_frame(public_snapshot, after_close_snapshot),
        _event_frame(public_event, after_close_event),
    )

    pd.testing.assert_frame_equal(
        baseline.loc[:, FUNDAMENTAL_FEATURE_COLUMNS],
        with_future_mutations.loc[:, FUNDAMENTAL_FEATURE_COLUMNS],
    )
    row = with_future_mutations.iloc[0]
    assert row["fundamental_operating_margin_ttm_ratio"] == pytest.approx(0.11)
    assert row["fundamental_sbc_to_revenue_ttm_ratio"] == pytest.approx(0.05)
    assert row["fundamental_quarterly_revenue_yoy_growth_ratio"] == pytest.approx(
        0.20
    )
    assert row["fundamental_quarterly_eps_yoy_change_ratio"] == pytest.approx(1 / 3)
    assert row["fundamental_quarterly_operating_margin_yoy_change"] == (
        pytest.approx(0.02)
    )


def test_effective_date_and_fiscal_period_must_not_be_in_the_future() -> None:
    output = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(
            _snapshot(
                period_end_date="2026-07-16",
                sec_latest_period_end_date="2026-07-16",
                sec_data_available_at="2026-07-15T19:00:00Z",
            )
        ),
        _event_frame(
            _event(
                effective_date="2026-07-16",
                accepted_at="2026-07-15T19:00:00Z",
            )
        ),
    )

    assert output.loc[0, list(FUNDAMENTAL_FEATURE_COLUMNS)].isna().all()


def test_non_usd_ratios_are_currency_neutral_but_currency_must_be_known() -> None:
    verified = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(_snapshot(sec_fundamental_currency="EUR")),
        _event_frame(_event(currency="EUR")),
    ).iloc[0]
    assert verified["fundamental_operating_margin_ttm_ratio"] == pytest.approx(0.11)
    assert verified["fundamental_quarterly_revenue_yoy_growth_ratio"] == (
        pytest.approx(0.20)
    )

    unknown = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(_snapshot(sec_fundamental_currency=pd.NA)),
        _event_frame(_event(currency=pd.NA)),
    ).iloc[0]
    assert np.isfinite(float(unknown["fundamental_snapshot_age_days"]))
    assert np.isfinite(float(unknown["fundamental_quarter_filing_age_days"]))
    assert pd.isna(unknown["fundamental_operating_margin_ttm_ratio"])
    assert pd.isna(unknown["fundamental_quarterly_revenue_yoy_growth_ratio"])


def test_stale_public_fundamentals_remain_available_with_explicit_age() -> None:
    output = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(
            _snapshot(
                period_end_date="2023-12-31",
                sec_latest_period_end_date="2023-12-31",
                sec_data_available_at="2024-02-01T12:00:00Z",
            )
        ),
        _event_frame(
            _event(
                accepted_at="2024-02-01T12:00:00Z",
                effective_date="2024-02-01",
                fiscal_period_end_date="2023-12-31",
            )
        ),
    ).iloc[0]

    assert output["fundamental_snapshot_age_days"] > 800
    assert output["fundamental_report_age_days"] > 900
    assert output["fundamental_operating_margin_ttm_ratio"] == pytest.approx(0.11)


def test_duplicate_source_keys_are_rejected() -> None:
    duplicate = _snapshot()
    with pytest.raises(ValueError, match="duplicate identity/date"):
        enrich_signal_fundamentals(
            _signals(),
            _snapshot_frame(duplicate, duplicate),
            _event_frame(_event()),
        )


def test_extended_quality_and_growth_features_use_only_public_sec_rows() -> None:
    current = _event(
        accession_number="0002",
        fiscal_period_end_date="2026-03-31",
        quarterly_revenue=150.0,
        prior_year_quarterly_revenue=100.0,
        diluted_eps=3.0,
        prior_year_diluted_eps=1.0,
    )
    previous = _event(
        accession_number="0001",
        accepted_at="2026-04-15T19:00:00Z",
        effective_date="2026-04-15",
        fiscal_period_end_date="2025-12-31",
        quarterly_revenue=120.0,
        prior_year_quarterly_revenue=100.0,
        diluted_eps=1.5,
        prior_year_diluted_eps=1.0,
    )
    output = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(
            _snapshot(
                sec_operating_income_ttm=300,
                sec_net_income_ttm=100,
                sec_operating_cashflow_ttm=120,
                sec_free_cashflow_ttm=90,
                sec_free_cashflow_sbc_adjusted_ttm=50,
                sec_research_and_development_ttm=100,
                sec_selling_general_and_admin_ttm=150,
                sec_interest_expense_ttm=50,
                sec_assets=2_000,
                sec_stockholders_equity=1_000,
                sec_cash_and_equivalents=200,
                sec_total_debt=400,
                sec_current_assets=600,
                sec_current_liabilities=300,
                sec_inventory=100,
                sec_weighted_avg_shares_diluted=110,
                sec_shares_outstanding=100,
            )
        ),
        _event_frame(previous, current),
    ).iloc[0]

    assert output["fundamental_roe_ttm_ratio"] == pytest.approx(0.10)
    assert output["fundamental_cash_conversion_ttm_ratio"] == pytest.approx(1.20)
    assert output["fundamental_interest_coverage_ttm_ratio"] == pytest.approx(6.0)
    assert output["fundamental_debt_to_assets_ratio"] == pytest.approx(0.20)
    assert output["fundamental_quick_ratio"] == pytest.approx(5 / 3)
    assert output["fundamental_diluted_share_pressure_ratio"] == pytest.approx(0.10)
    assert output["fundamental_quarterly_revenue_growth_acceleration"] == (
        pytest.approx(0.30)
    )
    assert output["fundamental_quarterly_eps_growth_acceleration"] == (
        pytest.approx(1.50)
    )


def test_cashflow_trends_compare_matching_point_in_time_reports() -> None:
    previous_prior_year = _snapshot(
        period_end_date="2025-02-10",
        sec_latest_period_end_date="2024-12-31",
        sec_data_available_at="2025-02-10T19:00:00Z",
        sec_revenue_ttm=950,
        sec_operating_cashflow_ttm=100,
        sec_free_cashflow_ttm=50,
        sec_free_cashflow_sbc_adjusted_ttm=40,
    )
    prior_year = _snapshot(
        period_end_date="2025-05-01",
        sec_latest_period_end_date="2025-03-25",
        sec_data_available_at="2025-05-01T19:00:00Z",
        sec_revenue_ttm=1_000,
        sec_operating_cashflow_ttm=100,
        sec_free_cashflow_ttm=40,
        sec_free_cashflow_sbc_adjusted_ttm=20,
    )
    previous = _snapshot(
        period_end_date="2026-02-10",
        sec_latest_period_end_date="2025-12-31",
        sec_data_available_at="2026-02-10T19:00:00Z",
        sec_revenue_ttm=1_100,
        sec_operating_cashflow_ttm=140,
        sec_free_cashflow_ttm=80,
        sec_free_cashflow_sbc_adjusted_ttm=60,
    )
    current = _snapshot(
        period_end_date="2026-05-01",
        sec_latest_period_end_date="2026-03-31",
        sec_data_available_at="2026-05-01T19:00:00Z",
        sec_revenue_ttm=1_200,
        sec_operating_cashflow_ttm=180,
        sec_free_cashflow_ttm=120,
        sec_free_cashflow_sbc_adjusted_ttm=90,
    )

    output = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(previous_prior_year, prior_year, previous, current),
        _event_frame(_event()),
    ).iloc[0]

    assert output["fundamental_operating_cashflow_ttm_yoy_change_ratio"] == (
        pytest.approx(80 / 280)
    )
    assert output["fundamental_fcf_ttm_yoy_change_ratio"] == pytest.approx(0.5)
    assert output[
        "fundamental_fcf_sbc_adjusted_ttm_yoy_change_ratio"
    ] == pytest.approx(70 / 110)
    assert output[
        "fundamental_operating_cashflow_ttm_sequential_change_ratio"
    ] == pytest.approx(40 / 320)
    assert output["fundamental_fcf_ttm_sequential_change_ratio"] == pytest.approx(
        0.2
    )
    assert output[
        "fundamental_operating_cashflow_margin_ttm_yoy_change"
    ] == pytest.approx(0.05)
    assert output["fundamental_fcf_margin_ttm_yoy_change"] == pytest.approx(0.06)
    assert output[
        "fundamental_fcf_sbc_adjusted_margin_ttm_yoy_change"
    ] == pytest.approx(0.055)
    assert output[
        "fundamental_operating_cashflow_ttm_growth_acceleration"
    ] == pytest.approx(80 / 280 - 40 / 240)
    assert output["fundamental_fcf_ttm_growth_acceleration"] == pytest.approx(
        0.5 - 30 / 130
    )
    assert output[
        "fundamental_fcf_sbc_adjusted_ttm_growth_acceleration"
    ] == pytest.approx(70 / 110 - 20 / 100)


def test_cashflow_turnarounds_and_currency_boundaries_are_explicit() -> None:
    prior_year = _snapshot(
        period_end_date="2025-05-01",
        sec_latest_period_end_date="2025-03-31",
        sec_data_available_at="2025-05-01T19:00:00Z",
        sec_operating_cashflow_ttm=-100,
        sec_free_cashflow_ttm=-50,
        sec_free_cashflow_sbc_adjusted_ttm=-20,
    )
    current = _snapshot(
        period_end_date="2026-05-01",
        sec_latest_period_end_date="2026-03-31",
        sec_data_available_at="2026-05-01T19:00:00Z",
        sec_operating_cashflow_ttm=100,
        sec_free_cashflow_ttm=50,
        sec_free_cashflow_sbc_adjusted_ttm=20,
    )
    turnaround = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(prior_year, current),
        _event_frame(_event()),
    ).iloc[0]

    for feature in (
        "fundamental_operating_cashflow_ttm_yoy_negative_to_positive",
        "fundamental_fcf_ttm_yoy_negative_to_positive",
        "fundamental_fcf_sbc_adjusted_ttm_yoy_negative_to_positive",
    ):
        assert turnaround[feature] == 1.0

    mismatched_currency = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(
            {**prior_year, "sec_fundamental_currency": "EUR"}, current
        ),
        _event_frame(_event()),
    ).iloc[0]
    cashflow_trend_features = tuple(
        feature
        for feature in FUNDAMENTAL_FEATURE_COLUMNS
        if (
            feature.startswith("fundamental_operating_cashflow")
            or feature.startswith("fundamental_fcf")
        )
        and (
            "_yoy_" in feature
            or "_sequential_" in feature
            or feature.endswith("_growth_acceleration")
        )
    )
    assert mismatched_currency.loc[list(cashflow_trend_features)].isna().all()


def test_only_confirmed_sec_earnings_known_by_close_are_features() -> None:
    safe = _earnings()
    unsafe_current_snapshot = _earnings(
        source="yfinance_calendar",
        source_event_id="yf-current",
        announcement_ts="2026-07-15T18:00:00Z",
        known_as_of_ts="2026-07-15T18:00:00Z",
    )
    after_close = _earnings(
        source_event_id="sec-after-close",
        announcement_ts="2026-07-15T20:01:00Z",
        known_as_of_ts="2026-07-15T20:01:00Z",
    )
    earnings = pd.DataFrame(
        [safe, unsafe_current_snapshot, after_close],
        columns=EARNINGS_EVENT_SOURCE_COLUMNS,
    )
    output = enrich_signal_fundamentals(
        _signals(),
        _snapshot_frame(_snapshot()),
        _event_frame(_event()),
        earnings,
    ).iloc[0]

    assert output["earnings_event_age_days"] == pytest.approx(1 / 24)
    assert output["earnings_event_on_signal_day"] == 1.0
    assert output["earnings_event_within_5d"] == 1.0
    assert output["earnings_event_within_21d"] == 1.0
    assert set(EARNINGS_EVENT_FEATURE_COLUMNS) <= set(output.index)
