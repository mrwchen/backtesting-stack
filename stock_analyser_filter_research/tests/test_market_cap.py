from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research.computation import enrich_signal_market_metrics
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_FEATURE_GROUPS,
    ENTRY_FEATURE_GROUPS,
    MARKET_CAP_FEATURE_COLUMNS,
    MARKET_METRIC_SOURCE_COLUMNS,
    SIGNAL_COLUMNS,
    SUPPLY_DEMAND_FEATURE_COLUMNS,
)


IDENTITY = {"symbol": "ABC", "exchange": "NYSE", "cik": 123}


def _signals(signal_date: str = "2026-07-15") -> pd.DataFrame:
    row = {column: pd.NA for column in SIGNAL_COLUMNS}
    row.update(IDENTITY)
    row["signal_date"] = pd.Timestamp(signal_date)
    row["prior_adjusted_close"] = 90.0
    row["adjusted_close"] = 100.0
    row["adjusted_volume"] = 1_000_000
    return pd.DataFrame([row], columns=SIGNAL_COLUMNS)


def _metric(**overrides: object) -> dict[str, object]:
    row = {column: pd.NA for column in MARKET_METRIC_SOURCE_COLUMNS}
    row.update(
        {
            **IDENTITY,
            "period_end_date": "2026-07-15",
            "market_cap": 12_345_678_901,
            "market_cap_currency": "USD",
            "shares_outstanding_staleness_days": 17,
            "adjusted_open": 95.0,
            "raw_volume": 1_000_000,
            "shares_outstanding": 100_000_000,
            "shares_outstanding_source": "sec_xbrl_instant",
        }
    )
    row.update(overrides)
    return row


def _metric_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=MARKET_METRIC_SOURCE_COLUMNS)


def test_market_cap_features_compete_only_in_entry_group_m() -> None:
    assert ENTRY_FEATURE_GROUPS["M"] == (
        "log_market_cap_usd",
        "market_cap_shares_staleness_days",
    )
    early_features = {
        feature
        for features in EARLY_CUT_FEATURE_GROUPS.values()
        for feature in features
    }
    assert not early_features.intersection(MARKET_CAP_FEATURE_COLUMNS)
    assert ENTRY_FEATURE_GROUPS["S"] == SUPPLY_DEMAND_FEATURE_COLUMNS
    assert not early_features.intersection(SUPPLY_DEMAND_FEATURE_COLUMNS)


def test_exact_signal_day_usd_market_cap_is_attached_with_log_and_staleness() -> None:
    output = enrich_signal_market_metrics(
        _signals(),
        _metric_frame(_metric()),
    ).iloc[0]

    assert output["market_cap_usd"] == 12_345_678_901
    assert output["log_market_cap_usd"] == pytest.approx(
        np.log(12_345_678_901)
    )
    assert output["market_cap_shares_staleness_days"] == 17


def test_future_or_other_identity_market_cap_is_never_forward_filled() -> None:
    future_only = enrich_signal_market_metrics(
        _signals(),
        _metric_frame(
            _metric(period_end_date="2026-07-16", market_cap=99_000_000_000),
            _metric(symbol="XYZ", cik=456, market_cap=88_000_000_000),
        ),
    ).iloc[0]

    assert future_only.loc[list(MARKET_CAP_FEATURE_COLUMNS)].isna().all()


@pytest.mark.parametrize("currency", ["EUR", None, pd.NA])
def test_non_usd_or_unknown_market_cap_is_not_comparable(currency) -> None:
    output = enrich_signal_market_metrics(
        _signals(),
        _metric_frame(_metric(market_cap_currency=currency)),
    ).iloc[0]

    assert output.loc[list(MARKET_CAP_FEATURE_COLUMNS)].isna().all()
    assert output["signal_adjusted_open"] == pytest.approx(95.0)
    assert output["shares_outstanding"] == 100_000_000
    assert output["signal_turnover_ratio"] == pytest.approx(0.01)


def test_invalid_market_metric_source_rows_are_rejected() -> None:
    duplicate = _metric()
    with pytest.raises(ValueError, match="duplicate identity/date"):
        enrich_signal_market_metrics(
            _signals(),
            _metric_frame(duplicate, duplicate),
        )
    with pytest.raises(ValueError, match="staleness must not be negative"):
        enrich_signal_market_metrics(
            _signals(),
            _metric_frame(_metric(shares_outstanding_staleness_days=-1)),
        )
    with pytest.raises(ValueError, match="must be integral"):
        enrich_signal_market_metrics(
            _signals(),
            _metric_frame(_metric(shares_outstanding_staleness_days=1.5)),
        )


def test_missing_or_nonpositive_market_cap_leaves_all_features_null() -> None:
    for market_cap in (None, 0, -1):
        output = enrich_signal_market_metrics(
            _signals(),
            _metric_frame(_metric(market_cap=market_cap)),
        ).iloc[0]
        assert output.loc[list(MARKET_CAP_FEATURE_COLUMNS)].isna().all()

    one_dollar = enrich_signal_market_metrics(
        _signals(),
        _metric_frame(_metric(market_cap=1)),
    ).iloc[0]
    assert one_dollar["market_cap_usd"] == 1
    assert one_dollar["log_market_cap_usd"] == 0.0


def test_supply_demand_uses_adjusted_open_raw_volume_and_point_in_time_sec_shares() -> None:
    safe = enrich_signal_market_metrics(_signals(), _metric_frame(_metric())).iloc[0]
    assert safe["signal_adjusted_open"] == pytest.approx(95.0)
    assert safe["signal_gap_pct"] == pytest.approx((95 / 90 - 1) * 100)
    assert safe["signal_intraday_return_pct"] == pytest.approx((100 / 95 - 1) * 100)
    assert safe["shares_outstanding"] == 100_000_000
    assert safe["signal_turnover_ratio"] == pytest.approx(0.01)

    unsafe = enrich_signal_market_metrics(
        _signals(),
        _metric_frame(_metric(shares_outstanding_source="yfinance_shares_full_asof")),
    ).iloc[0]
    assert pd.isna(unsafe["shares_outstanding"])
    assert pd.isna(unsafe["log_shares_outstanding"])
    assert pd.isna(unsafe["signal_turnover_ratio"])
