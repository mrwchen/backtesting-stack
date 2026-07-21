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
