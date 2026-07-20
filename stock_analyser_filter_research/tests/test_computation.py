from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research.computation import (
    calculate_identity_signals,
)
from stock_analyser_filter_research.contracts import CRITERION_COLUMNS


def _source_frame(dates: pd.DatetimeIndex) -> pd.DataFrame:
    size = len(dates)
    sequence = np.arange(size, dtype=float)
    close = 100.0 + sequence
    volume = 1_000.0 + sequence * 10.0
    frame = pd.DataFrame(
        {
            "period_end_date": dates,
            "symbol": "TEST",
            "exchange": "NYSE",
            "cik": 123456,
            "price_continuity_segment": 1,
            "currency": "USD",
            "raw_close": close,
            "adjusted_close": close,
            "adjusted_high": close + 1.0,
            "adjusted_low": close - 1.0,
            "adjusted_volume": volume,
            "daily_price_change_pct": 1.0,
            "adjusted_volume_sma21_prior": 900.0,
            "adjusted_volume_vs_sma21_prior_ratio": volume / 900.0,
            "adjusted_volume_sma50_prior": 850.0,
            "adjusted_volume_vs_sma50_prior_ratio": volume / 850.0,
            "daily_traded_notional_usd": close * volume,
            "daily_traded_notional_sma21_prior_usd": 100_000.0,
            "daily_traded_notional_vs_sma21_prior_ratio": (
                close * volume / 100_000.0
            ),
            "daily_traded_notional_sma50_prior_usd": 95_000.0,
            "daily_traded_notional_vs_sma50_prior_ratio": (
                close * volume / 95_000.0
            ),
            "dollar_volume_63d": 1_000_000.0,
            "ma50": close - 2.0,
            "ma150": close - 4.0,
            "ma200": close - 6.0,
            "ma200_21_sessions_ago": close - 7.0,
            "low_52w": close / 1.5,
            "high_52w": close / 0.8,
            "rs_raw": 1.5,
            "rs_rating": 80.0,
            "forward_5d_max_gain_pct": 3.0,
            "forward_5d_max_loss_pct": -2.0,
            "forward_10d_max_gain_pct": 4.0,
            "forward_10d_max_loss_pct": -3.0,
            "forward_20d_max_gain_pct": 6.0,
            "forward_20d_max_loss_pct": -4.0,
            "trend_template_pass": False,
        }
    )
    for criterion in CRITERION_COLUMNS:
        frame[criterion] = False
    return frame


def _set_pass(frame: pd.DataFrame, position: int, value: bool = True) -> None:
    frame.loc[frame.index[position], "trend_template_pass"] = value
    for criterion in CRITERION_COLUMNS:
        frame.loc[frame.index[position], criterion] = value


def _row_for_date(result: pd.DataFrame, value: pd.Timestamp) -> pd.Series:
    matching = result.loc[result["signal_date"].eq(value)]
    assert len(matching) == 1
    return matching.iloc[0]


def test_signal_requires_immediate_false_to_true_same_segment_and_observed_row(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-01-01", periods=11, freq="D")
    source = _source_frame(dates)

    # First-row True has no prior session and therefore is not an event.
    _set_pass(source, 0)
    # True after a missing global session is not an event.
    source = source.loc[source["period_end_date"].ne(dates[3])].copy()
    gap_true_index = source.index[source["period_end_date"].eq(dates[4])][0]
    _set_pass(source, source.index.get_loc(gap_true_index))
    # False -> True over a segment boundary is not an event.
    source.loc[source["period_end_date"].eq(dates[6]), "price_continuity_segment"] = 2
    source.loc[source["period_end_date"].eq(dates[6]), "trend_template_pass"] = True
    for criterion in CRITERION_COLUMNS:
        source.loc[source["period_end_date"].eq(dates[6]), criterion] = True
    source.loc[source["period_end_date"].ge(dates[6]), "price_continuity_segment"] = 2
    # This is the only valid immediate False -> True transition.
    valid_index = source.index[source["period_end_date"].eq(dates[9])][0]
    _set_pass(source, source.index.get_loc(valid_index))

    result = calculate_identity_signals(source, dates, cfg_factory())

    assert result["signal_date"].tolist() == [dates[9]]
    assert result["previous_session_date"].tolist() == [dates[8]]


def test_multiple_pass_episodes_emit_one_signal_each_and_preserve_history(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-02-01", periods=18, freq="D")
    source = _source_frame(dates)
    _set_pass(source, 2)
    _set_pass(source, 3)  # Continuation, not a second event.

    # Seven criteria passed one session before the second episode.
    for criterion in CRITERION_COLUMNS[:-1]:
        source.loc[source.index[11], criterion] = True
    _set_pass(source, 12)

    result = calculate_identity_signals(source, dates, cfg_factory())

    assert result["signal_date"].tolist() == [dates[2], dates[12]]
    second = _row_for_date(result, dates[12])
    assert second["previous_criteria_pass_count"] == 7
    assert second["trigger_count"] == 1
    assert second["trigger_criteria"] == CRITERION_COLUMNS[-1]
    assert second["prior_7_of_8_count_10d"] == 1
    assert second["sessions_since_previous_pass"] == 9


def test_missing_global_session_resets_previous_pass_history(cfg_factory) -> None:
    dates = pd.date_range("2022-02-20", periods=12, freq="D")
    source = _source_frame(dates)
    _set_pass(source, 2)
    source = source.loc[source["period_end_date"].ne(dates[5])].copy()
    signal_index = source.index[source["period_end_date"].eq(dates[8])][0]
    _set_pass(source, source.index.get_loc(signal_index))

    result = calculate_identity_signals(source, dates, cfg_factory())

    after_gap = _row_for_date(result, dates[8])
    assert pd.isna(after_gap["sessions_since_previous_pass"])


def test_prior_windows_exclude_signal_session_values(cfg_factory) -> None:
    dates = pd.date_range("2022-03-01", periods=45, freq="D")
    signal_position = 30
    source = _source_frame(dates)
    _set_pass(source, signal_position)

    baseline = calculate_identity_signals(source, dates, cfg_factory())
    baseline_row = _row_for_date(baseline, dates[signal_position])

    prior_volume = source["adjusted_volume"].iloc[
        signal_position - 21 : signal_position
    ]
    prior_notional = source["daily_traded_notional_usd"].iloc[
        signal_position - 21 : signal_position
    ]
    expected_volume_ratio = prior_volume.iloc[-5:].mean() / prior_volume.mean()
    expected_notional_ratio = (
        prior_notional.iloc[-5:].mean() / prior_notional.mean()
    )
    expected_return_5d = (
        source["adjusted_close"].iloc[signal_position - 1]
        / source["adjusted_close"].iloc[signal_position - 6]
        - 1.0
    ) * 100.0

    assert baseline_row["prior_volume_sma5_vs21_ratio"] == pytest.approx(
        expected_volume_ratio
    )
    assert baseline_row["prior_notional_sma5_vs21_ratio"] == pytest.approx(
        expected_notional_ratio
    )
    assert baseline_row["prior_return_5d_pct"] == pytest.approx(
        expected_return_5d
    )

    mutated = source.copy()
    mutated.loc[mutated.index[signal_position], "adjusted_volume"] = 10**12
    mutated.loc[
        mutated.index[signal_position], "daily_traded_notional_usd"
    ] = 10**15
    mutated.loc[mutated.index[signal_position], "adjusted_close"] = 10**6
    mutated_result = calculate_identity_signals(mutated, dates, cfg_factory())
    mutated_row = _row_for_date(mutated_result, dates[signal_position])

    prior_features = [
        "prior_volume_sma5_vs21_ratio",
        "prior_volume_sma10_vs21_ratio",
        "prior_notional_sma5_vs21_ratio",
        "prior_notional_sma10_vs21_ratio",
        "prior_return_5d_pct",
        "prior_return_10d_pct",
        "prior_return_21d_pct",
        "prior_atr_14d_pct",
        "prior_max_drawdown_21d_pct",
        "prior_price_volume_corr21",
        "prior_price_notional_corr21",
    ]
    pd.testing.assert_series_equal(
        baseline_row[prior_features],
        mutated_row[prior_features],
        check_names=False,
    )


def test_future_input_mutation_cannot_change_existing_signal_features(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-04-01", periods=50, freq="D")
    signal_position = 30
    source = _source_frame(dates)
    _set_pass(source, signal_position)
    baseline = calculate_identity_signals(source, dates, cfg_factory())
    baseline_row = _row_for_date(baseline, dates[signal_position])

    mutated = source.copy()
    future = mutated.index > mutated.index[signal_position]
    for column in (
        "adjusted_close",
        "adjusted_high",
        "adjusted_low",
        "adjusted_volume",
        "daily_traded_notional_usd",
        "ma50",
        "ma150",
        "ma200",
        "rs_rating",
        "forward_5d_max_gain_pct",
        "forward_20d_max_loss_pct",
    ):
        mutated.loc[future, column] = np.arange(future.sum()) + 10**8

    changed = calculate_identity_signals(mutated, dates, cfg_factory())
    changed_row = _row_for_date(changed, dates[signal_position])

    pd.testing.assert_series_equal(
        baseline_row,
        changed_row,
        check_names=False,
        check_dtype=False,
    )


def test_label_boundaries_nullability_and_split_boundaries(cfg_factory) -> None:
    dates = pd.date_range("2022-05-01", periods=36, freq="D")
    discovery_end = dates[10].date()
    validation_end = dates[20].date()
    cfg = cfg_factory(
        signal_start_date=dates[0].date(),
        discovery_end_date=discovery_end,
        validation_end_date=validation_end,
    )
    source = _source_frame(dates)
    for position in (5, 10, 15, 20, 25, 30):
        _set_pass(source, position)

    source.loc[source.index[10], [
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_10d_max_gain_pct",
        "forward_20d_max_gain_pct",
    ]] = [2.0, -5.0, 5.0, 5.0]
    source.loc[source.index[20], [
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_10d_max_gain_pct",
        "forward_20d_max_gain_pct",
    ]] = [5.0, -4.999, 8.0, 9.0]
    source.loc[source.index[25], [
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_10d_max_gain_pct",
        "forward_20d_max_gain_pct",
    ]] = [np.nan, -10.0, 10.0, 10.0]
    source.loc[source.index[30], [
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_10d_max_gain_pct",
        "forward_20d_max_gain_pct",
    ]] = [1.999, -4.0, 4.999, 5.0]

    result = calculate_identity_signals(source, dates, cfg)
    boundary = _row_for_date(result, dates[10])
    strong = _row_for_date(result, dates[20])
    incomplete = _row_for_date(result, dates[25])
    weak = _row_for_date(result, dates[30])

    assert boundary["weak_5d"] == False  # noqa: E712
    assert boundary["strong_5d"] == False  # noqa: E712
    assert boundary["deep_loss_5d"] == True  # noqa: E712
    assert boundary["bad_5d"] == True  # noqa: E712
    assert boundary["late_strong_10d"] == True  # noqa: E712
    assert boundary["late_strong_20d"] == True  # noqa: E712

    assert strong["weak_5d"] == False  # noqa: E712
    assert strong["strong_5d"] == True  # noqa: E712
    assert strong["deep_loss_5d"] == False  # noqa: E712
    assert strong["bad_5d"] == False  # noqa: E712
    assert strong["late_strong_10d"] == False  # noqa: E712

    for column in (
        "weak_5d",
        "strong_5d",
        "deep_loss_5d",
        "bad_5d",
        "late_strong_10d",
        "late_strong_20d",
    ):
        assert pd.isna(incomplete[column])

    assert weak["weak_5d"] == True  # noqa: E712
    assert weak["strong_5d"] == False  # noqa: E712
    assert weak["deep_loss_5d"] == False  # noqa: E712
    assert weak["bad_5d"] == True  # noqa: E712
    assert weak["late_strong_10d"] == False  # noqa: E712
    assert weak["late_strong_20d"] == True  # noqa: E712

    assert result.set_index("signal_date")["analysis_split"].to_dict() == {
        dates[5]: "discovery",
        dates[10]: "purged",
        dates[15]: "validation",
        dates[20]: "purged",
        dates[25]: "test",
        dates[30]: "test",
    }
