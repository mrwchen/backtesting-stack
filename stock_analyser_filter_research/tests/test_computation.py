from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research.computation import (
    CalculationBatchResult,
    calculate_early_cut_landmarks,
    calculate_identity_results,
    calculate_identity_signals,
    calculate_signal_batch,
)
from stock_analyser_filter_research.contracts import (
    CRITERION_COLUMNS,
    EARLY_CUT_COLUMNS,
    EARLY_CUT_FEATURE_GROUPS,
    SIGNAL_COLUMNS,
)


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
            "daily_traded_notional_vs_sma21_prior_ratio": (close * volume / 100_000.0),
            "daily_traded_notional_sma50_prior_usd": 95_000.0,
            "daily_traded_notional_vs_sma50_prior_ratio": (close * volume / 95_000.0),
            "dollar_volume_63d": 1_000_000.0,
            "ma50": close - 2.0,
            "ma150": close - 4.0,
            "ma200": close - 6.0,
            "ma200_21_sessions_ago": close - 7.0,
            "low_52w": close / 1.5,
            "high_52w": close / 0.8,
            "rs_raw": 1.5,
            "rs_rating": 80.0,
            # These upstream forward columns must not determine D1-D5 labels.
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


def _set_path(
    frame: pd.DataFrame,
    signal_position: int,
    *,
    closes: list[float],
    highs: list[float],
    lows: list[float],
) -> None:
    assert len(closes) == len(highs) == len(lows) == 6
    positions = frame.index[signal_position : signal_position + 6]
    frame.loc[positions, "adjusted_close"] = closes
    frame.loc[positions, "raw_close"] = closes
    frame.loc[positions, "adjusted_high"] = highs
    frame.loc[positions, "adjusted_low"] = lows


def _row_for_date(result: pd.DataFrame, value: pd.Timestamp) -> pd.Series:
    matching = result.loc[result["signal_date"].eq(value)]
    assert len(matching) == 1
    return matching.iloc[0]


def _landmark_row(result: pd.DataFrame, landmark_day: int) -> pd.Series:
    matching = result.loc[result["landmark_day"].eq(landmark_day)]
    assert len(matching) == 1
    return matching.iloc[0]


def test_signal_requires_immediate_false_to_true_same_segment_and_observed_row(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-01-01", periods=11, freq="D")
    source = _source_frame(dates)

    _set_pass(source, 0)
    source = source.loc[source["period_end_date"].ne(dates[3])].copy()
    gap_true_index = source.index[source["period_end_date"].eq(dates[4])][0]
    _set_pass(source, source.index.get_loc(gap_true_index))
    source.loc[source["period_end_date"].ge(dates[6]), "price_continuity_segment"] = 2
    source.loc[source["period_end_date"].eq(dates[6]), "trend_template_pass"] = True
    for criterion in CRITERION_COLUMNS:
        source.loc[source["period_end_date"].eq(dates[6]), criterion] = True
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
    _set_pass(source, 3)
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

    assert pd.isna(_row_for_date(result, dates[8])["sessions_since_previous_pass"])


def test_prior_windows_exclude_signal_session_values(cfg_factory) -> None:
    dates = pd.date_range("2022-03-01", periods=45, freq="D")
    signal_position = 30
    source = _source_frame(dates)
    _set_pass(source, signal_position)

    baseline_row = _row_for_date(
        calculate_identity_signals(source, dates, cfg_factory()),
        dates[signal_position],
    )
    prior_volume = source["adjusted_volume"].iloc[
        signal_position - 21 : signal_position
    ]
    prior_notional = source["daily_traded_notional_usd"].iloc[
        signal_position - 21 : signal_position
    ]
    assert baseline_row["prior_volume_sma5_vs21_ratio"] == pytest.approx(
        prior_volume.iloc[-5:].mean() / prior_volume.mean()
    )
    assert baseline_row["prior_notional_sma5_vs21_ratio"] == pytest.approx(
        prior_notional.iloc[-5:].mean() / prior_notional.mean()
    )
    assert baseline_row["prior_return_5d_pct"] == pytest.approx(
        (
            source["adjusted_close"].iloc[signal_position - 1]
            / source["adjusted_close"].iloc[signal_position - 6]
            - 1.0
        )
        * 100.0
    )

    mutated = source.copy()
    mutated.loc[mutated.index[signal_position], "adjusted_volume"] = 10**12
    mutated.loc[mutated.index[signal_position], "daily_traded_notional_usd"] = 10**15
    mutated.loc[mutated.index[signal_position], "adjusted_close"] = 10**6
    mutated_row = _row_for_date(
        calculate_identity_signals(mutated, dates, cfg_factory()),
        dates[signal_position],
    )
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


@pytest.mark.parametrize(
    ("highs", "lows", "expected_order", "strong_first", "loss_first"),
    [
        (
            [101, 101, 105.1, 101, 101],
            [94.9, 99, 99, 99, 99],
            "loss_first",
            False,
            True,
        ),
        (
            [105.1, 101, 101, 101, 101],
            [99, 99, 94.9, 99, 99],
            "gain_first",
            True,
            False,
        ),
        (
            [101, 105.1, 101, 101, 101],
            [99, 94.9, 99, 99, 99],
            "same_day_ambiguous",
            None,
            None,
        ),
        ([105.1, 101, 101, 101, 101], [99, 99, 99, 99, 99], "gain_only", True, False),
        ([101, 101, 101, 101, 101], [94.9, 99, 99, 99, 99], "loss_only", False, True),
        ([101, 101, 101, 101, 101], [99, 99, 99, 99, 99], "neither", False, False),
    ],
)
def test_entry_outcomes_use_actual_high_low_path_and_barrier_order(
    cfg_factory,
    highs,
    lows,
    expected_order,
    strong_first,
    loss_first,
) -> None:
    dates = pd.date_range("2022-04-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 100, 100, 100, 102],
        highs=[101, *highs],
        lows=[99, *lows],
    )
    source.loc[source.index[position], "forward_5d_max_gain_pct"] = -999.0
    source.loc[source.index[position], "forward_5d_max_loss_pct"] = 999.0

    row = _row_for_date(
        calculate_identity_signals(source, dates, cfg_factory()),
        dates[position],
    )

    assert row["forward_5d_max_gain_pct"] == pytest.approx(
        (max(highs) / 100.0 - 1.0) * 100.0
    )
    assert row["forward_5d_max_loss_pct"] == pytest.approx(
        (min(lows) / 100.0 - 1.0) * 100.0
    )
    assert row["forward_5d_label_end_date"] == dates[position + 5]
    assert row["terminal_close_return_5d_pct"] == pytest.approx(2.0)
    assert row["gain_loss_order_5d"] == expected_order
    if strong_first is None:
        assert pd.isna(row["strong_first_5d"])
        assert pd.isna(row["loss_first_5d"])
    else:
        assert bool(row["strong_first_5d"]) is strong_first
        assert bool(row["loss_first_5d"]) is loss_first


def test_entry_first_hit_days_and_upstream_forward5_columns_are_ignored(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-05-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 101, 102, 103, 104],
        highs=[101, 101, 102.1, 104, 105.1, 104.5],
        lows=[99, 99, 98, 97, 96, 94.9],
    )
    baseline = calculate_identity_signals(source, dates, cfg_factory())
    mutated = source.copy()
    mutated.loc[mutated.index[position], "forward_5d_max_gain_pct"] = 12345
    mutated.loc[mutated.index[position], "forward_5d_max_loss_pct"] = -12345
    changed = calculate_identity_signals(mutated, dates, cfg_factory())
    pd.testing.assert_frame_equal(baseline, changed)

    row = baseline.iloc[0]
    assert row["first_gain_2pct_day"] == 2
    assert row["first_gain_5pct_day"] == 4
    assert row["first_loss_5pct_day"] == 5
    assert row["gain_loss_order_5d"] == "gain_first"


def test_future_prices_change_labels_but_not_entry_features(cfg_factory) -> None:
    dates = pd.date_range("2022-06-01", periods=50, freq="D")
    position = 30
    source = _source_frame(dates)
    _set_pass(source, position)
    baseline = calculate_identity_signals(source, dates, cfg_factory()).iloc[0]

    mutated = source.copy()
    future_rows = mutated.index[position + 1 : position + 6]
    mutated.loc[future_rows, "adjusted_close"] = 100.0
    mutated.loc[future_rows, "adjusted_high"] = [101, 106, 101, 101, 101]
    mutated.loc[future_rows, "adjusted_low"] = [99, 99, 99, 94, 99]
    changed = calculate_identity_signals(mutated, dates, cfg_factory()).iloc[0]

    causal_end = SIGNAL_COLUMNS.index("forward_5d_max_gain_pct")
    causal_columns = list(SIGNAL_COLUMNS[:causal_end])
    pd.testing.assert_series_equal(
        baseline[causal_columns],
        changed[causal_columns],
        check_names=False,
        check_dtype=False,
    )
    assert baseline["gain_loss_order_5d"] != changed["gain_loss_order_5d"]


def test_analysis_splits_use_label_end_and_holdout_starts_after_cutoff(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-01-01", periods=42, freq="D")
    cfg = cfg_factory(
        discovery_end_date=dates[10].date(),
        validation_end_date=dates[20].date(),
        holdout_cutoff_date=dates[30].date(),
        min_walk_forward_folds=3,
    )
    source = _source_frame(dates)
    for position in (5, 8, 12, 18, 22, 28, 31):
        _set_pass(source, position)

    result = calculate_identity_results(source, dates, cfg)

    assert result.signals.set_index("signal_date")["analysis_split"].to_dict() == {
        dates[5]: "discovery",
        dates[8]: "purged",
        dates[12]: "validation",
        dates[18]: "purged",
        dates[22]: "diagnostic",
        dates[28]: "purged",
        dates[31]: "holdout",
    }
    early_splits = (
        result.early_cut.groupby("signal_date", sort=False)["analysis_split"]
        .apply(list)
        .to_dict()
    )
    assert early_splits == {
        dates[5]: ["discovery", "discovery", "discovery"],
        dates[8]: ["purged", "purged", "validation"],
        dates[12]: ["validation", "validation", "validation"],
        dates[18]: ["purged", "purged", "diagnostic"],
        dates[22]: ["diagnostic", "diagnostic", "diagnostic"],
        dates[28]: ["purged", "purged", "holdout"],
        dates[31]: ["holdout", "holdout", "holdout"],
    }


def test_early_cut_emits_exactly_three_rows_with_causal_feature_formulas(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-07-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 102, 103, 103, 103, 103],
        highs=[101, 103, 104, 104, 104, 104],
        lows=[99, 99, 100, 100, 100, 100],
    )
    signal_index = source.index[position]
    landmark_index = source.index[position + 1]
    source.loc[signal_index, "adjusted_volume"] = 1_000
    source.loc[signal_index, "daily_traded_notional_usd"] = 100_000
    source.loc[signal_index, "adjusted_volume_sma21_prior"] = 800
    source.loc[signal_index, "daily_traded_notional_sma21_prior_usd"] = 80_000
    source.loc[signal_index, "rs_rating"] = 80
    source.loc[landmark_index, "adjusted_volume"] = 1_500
    source.loc[landmark_index, "daily_traded_notional_usd"] = 150_000
    source.loc[landmark_index, "rs_rating"] = 83
    source.loc[landmark_index, "adjusted_volume_vs_sma21_prior_ratio"] = 1.5
    source.loc[landmark_index, "adjusted_volume_vs_sma50_prior_ratio"] = 1.6
    source.loc[landmark_index, "daily_traded_notional_vs_sma21_prior_ratio"] = 1.7
    source.loc[landmark_index, "daily_traded_notional_vs_sma50_prior_ratio"] = 1.8

    result = calculate_identity_results(source, dates, cfg_factory())

    assert list(result.early_cut.columns) == list(EARLY_CUT_COLUMNS)
    assert result.early_cut["landmark_day"].tolist() == [1, 2, 3]
    day1 = _landmark_row(result.early_cut, 1)
    assert day1["landmark_date"] == dates[position + 1]
    assert day1["effective_session_date"] == dates[position + 2]
    assert day1["horizon_end_date"] == dates[position + 5]
    assert bool(day1["landmark_observed"])
    assert bool(day1["same_continuity_segment"])
    assert bool(day1["eligible_at_landmark"])
    assert bool(day1["full_outcome_available"])
    assert day1["cut_decision"] == "hold"
    assert bool(day1["include_stagnation_filter"])
    assert bool(day1["include_loss_filter"])
    assert bool(day1["include_final"])
    assert day1["close_return_from_signal_pct"] == pytest.approx(2.0)
    assert day1["max_gain_to_landmark_pct"] == pytest.approx(3.0)
    assert day1["max_loss_to_landmark_pct"] == pytest.approx(-1.0)
    assert day1["drawdown_from_post_signal_high_pct"] == pytest.approx(
        (102 / 103 - 1) * 100
    )
    assert day1["rebound_from_post_signal_low_pct"] == pytest.approx(
        (102 / 99 - 1) * 100
    )
    assert day1["landmark_true_range_pct"] == pytest.approx(4.0)
    assert day1["landmark_close_location_value"] == pytest.approx(0.75)
    assert day1["volume_vs_signal_ratio"] == pytest.approx(1.5)
    assert day1["notional_vs_signal_ratio"] == pytest.approx(1.5)
    assert day1["rs_rating_change_from_signal"] == pytest.approx(3.0)
    assert day1["mean_volume_since_signal_vs_prior21_ratio"] == pytest.approx(1.875)
    assert day1["mean_notional_since_signal_vs_prior21_ratio"] == pytest.approx(1.875)
    assert day1["landmark_criteria_pass_count"] == 0


def test_early_landmark_features_do_not_use_sessions_after_landmark(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-08-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 100, 100, 100, 100],
        highs=[101, 101, 101, 101, 101, 101],
        lows=[99, 99, 99, 99, 99, 99],
    )
    baseline = calculate_identity_results(source, dates, cfg_factory())

    mutated = source.copy()
    future = mutated.index[position + 2 : position + 6]
    mutated.loc[future, "adjusted_close"] = [100, 101, 99, 100]
    mutated.loc[future, "adjusted_high"] = [106, 102, 101, 101]
    mutated.loc[future, "adjusted_low"] = [94, 99, 98, 99]
    mutated.loc[future, "adjusted_volume"] = 10**8
    mutated.loc[future, "daily_traded_notional_usd"] = 10**10
    mutated.loc[future, "rs_rating"] = 1
    changed = calculate_identity_results(mutated, dates, cfg_factory())

    baseline_day1 = _landmark_row(baseline.early_cut, 1)
    changed_day1 = _landmark_row(changed.early_cut, 1)
    causal = list(EARLY_CUT_FEATURE_GROUPS["E"])
    pd.testing.assert_series_equal(
        baseline_day1[causal],
        changed_day1[causal],
        check_names=False,
        check_dtype=False,
    )
    assert baseline_day1["continuation_outcome"] == "stagnant"
    assert changed_day1["continuation_outcome"] == "same_session_ambiguous"


def test_landmark_paths_are_inclusive_and_future_paths_start_next_session(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-09-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 100, 100, 100, 100],
        highs=[101, 102.1, 105.1, 101, 101, 101],
        lows=[99, 99, 99, 99, 99, 99],
    )

    early = calculate_identity_results(source, dates, cfg_factory()).early_cut
    day1 = _landmark_row(early, 1)
    day2 = _landmark_row(early, 2)

    assert day1["first_gain_2pct_day_so_far"] == 1
    assert pd.isna(day1["first_gain_5pct_day_so_far"])
    assert day1["future_first_gain_2pct_day"] == 2
    assert day1["future_first_gain_5pct_day"] == 2
    assert bool(day1["eligible_at_landmark"])
    assert day2["first_gain_5pct_day_so_far"] == 2
    assert pd.isna(day2["future_first_gain_5pct_day"])
    assert not bool(day2["eligible_at_landmark"])
    assert pd.isna(day2["continuation_outcome"])


@pytest.mark.parametrize(
    ("future_highs", "future_lows", "outcome", "flags"),
    [
        (
            [105.1, 101, 101, 101],
            [99, 94.9, 99, 99],
            "strong_first",
            (False, False, True),
        ),
        (
            [101, 105.1, 101, 101],
            [94.9, 99, 99, 99],
            "loss_first",
            (False, True, False),
        ),
        ([105.1, 101, 101, 101], [94.9, 99, 99, 99], "same_session_ambiguous", None),
        ([101, 101, 101, 101], [99, 99, 99, 99], "stagnant", (True, False, False)),
        ([103, 101, 101, 101], [99, 99, 99, 99], "neutral", (False, False, False)),
    ],
)
def test_early_continuation_outcome_classes_and_nullable_objectives(
    cfg_factory,
    future_highs,
    future_lows,
    outcome,
    flags,
) -> None:
    dates = pd.date_range("2022-10-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 100, 100, 100, 100],
        highs=[101, 101, *future_highs],
        lows=[99, 99, *future_lows],
    )

    day1 = _landmark_row(
        calculate_identity_results(source, dates, cfg_factory()).early_cut,
        1,
    )

    assert bool(day1["eligible_at_landmark"])
    assert day1["continuation_outcome"] == outcome
    label_columns = [
        "stagnant_to_day5",
        "loss_first_to_day5",
        "strong_first_to_day5",
    ]
    if flags is None:
        assert day1[label_columns].isna().all()
    else:
        assert tuple(bool(day1[column]) for column in label_columns) == flags


def test_prior_barrier_hit_removes_later_landmarks_from_risk_set(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-11-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 100, 100, 100, 100, 100],
        highs=[101, 105.1, 101, 101, 101, 101],
        lows=[99, 99, 99, 99, 99, 99],
    )

    early = calculate_identity_results(source, dates, cfg_factory()).early_cut

    assert early["full_outcome_available"].astype(bool).all()
    assert not early["eligible_at_landmark"].astype(bool).any()
    assert early["cut_decision"].eq("not_eligible").all()
    assert not early["include_stagnation_filter"].astype(bool).any()
    assert not early["include_loss_filter"].astype(bool).any()
    assert not early["include_final"].astype(bool).any()
    assert (
        early[["stagnant_to_day5", "loss_first_to_day5", "strong_first_to_day5"]]
        .isna()
        .all()
        .all()
    )


@pytest.mark.parametrize("censoring", ["gap", "segment"])
def test_gap_or_segment_change_censors_landmarks_without_dropping_them(
    cfg_factory,
    censoring,
) -> None:
    dates = pd.date_range("2022-12-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    if censoring == "gap":
        source = source.loc[source["period_end_date"].ne(dates[position + 2])]
    else:
        source.loc[
            source["period_end_date"].ge(dates[position + 2]),
            "price_continuity_segment",
        ] = 2

    result = calculate_identity_results(source, dates, cfg_factory())

    assert len(result.early_cut) == 3
    assert result.early_cut["landmark_day"].tolist() == [1, 2, 3]
    day1 = _landmark_row(result.early_cut, 1)
    assert bool(day1["same_continuity_segment"])
    assert bool(day1["eligible_at_landmark"])
    assert (
        not result.early_cut.loc[
            result.early_cut["landmark_day"].ge(2), "same_continuity_segment"
        ]
        .astype(bool)
        .any()
    )
    assert not result.early_cut["full_outcome_available"].astype(bool).any()
    assert day1["cut_decision"] == "hold"
    assert (
        result.early_cut.loc[result.early_cut["landmark_day"].ge(2), "cut_decision"]
        .eq("not_evaluable")
        .all()
    )
    assert (
        result.early_cut[
            ["continuation_outcome", "stagnant_to_day5", "loss_first_to_day5"]
        ]
        .isna()
        .all()
        .all()
    )
    assert bool(day1["landmark_observed"])
    assert pd.notna(day1["close_return_from_signal_pct"])


def test_next_session_missing_does_not_change_landmark_eligibility(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-12-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    source = source.loc[source["period_end_date"].ne(dates[position + 2])]

    day1 = _landmark_row(
        calculate_identity_results(source, dates, cfg_factory()).early_cut,
        1,
    )

    assert bool(day1["same_continuity_segment"])
    assert bool(day1["eligible_at_landmark"])
    assert day1["cut_decision"] == "hold"
    assert not bool(day1["full_outcome_available"])
    assert pd.isna(day1["continuation_outcome"])


def test_end_of_calendar_is_censored_but_still_emits_three_landmarks(
    cfg_factory,
) -> None:
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    position = 27
    source = _source_frame(dates)
    _set_pass(source, position)

    result = calculate_identity_results(source, dates, cfg_factory())

    signal = result.signals.iloc[0]
    assert pd.isna(signal["forward_5d_label_end_date"])
    assert pd.isna(signal["weak_5d"])
    assert signal["analysis_split"] == "purged"
    assert len(result.early_cut) == 3
    day1 = _landmark_row(result.early_cut, 1)
    day2 = _landmark_row(result.early_cut, 2)
    day3 = _landmark_row(result.early_cut, 3)
    assert day1["effective_session_date"] == dates[29]
    assert pd.isna(day2["effective_session_date"])
    assert pd.isna(day3["landmark_date"])
    assert not result.early_cut["full_outcome_available"].astype(bool).any()
    assert result.early_cut["continuation_outcome"].isna().all()


def test_early_split_is_anchored_at_landmark_date_across_holdout_cutoff(
    cfg_factory,
) -> None:
    dates = pd.bdate_range("2026-06-01", "2026-08-03")
    signal_date = pd.Timestamp("2026-07-20")
    position = dates.get_loc(signal_date)
    source = _source_frame(dates)
    _set_pass(source, position)

    result = calculate_identity_results(source, dates, cfg_factory())

    assert result.signals.iloc[0]["analysis_split"] == "purged"
    assert result.early_cut["landmark_date"].gt(signal_date).all()
    assert result.early_cut["analysis_split"].eq("holdout").all()


def test_public_batch_result_is_partition_stable_and_handles_empty_source(
    cfg_factory,
) -> None:
    dates = pd.date_range("2023-02-01", periods=40, freq="D")
    first = _source_frame(dates)
    second = _source_frame(dates)
    second["symbol"] = "OTHER"
    second["cik"] = 654321
    _set_pass(first, 25)
    _set_pass(second, 26)
    combined = pd.concat([second, first], ignore_index=True)
    cfg = cfg_factory()

    batch = calculate_signal_batch(combined, dates, cfg)
    first_batch = calculate_signal_batch(first, dates, cfg)
    second_batch = calculate_signal_batch(second, dates, cfg)

    assert isinstance(batch, CalculationBatchResult)
    assert list(batch.signals.columns) == list(SIGNAL_COLUMNS)
    assert list(batch.early_cut.columns) == list(EARLY_CUT_COLUMNS)
    assert len(batch.signals) == 2
    assert len(batch.early_cut) == 6
    expected_signals = pd.concat(
        [first_batch.signals, second_batch.signals], ignore_index=True
    ).sort_values(["symbol", "signal_date"], ignore_index=True)
    expected_early = pd.concat(
        [first_batch.early_cut, second_batch.early_cut], ignore_index=True
    ).sort_values(["symbol", "signal_date", "landmark_day"], ignore_index=True)
    pd.testing.assert_frame_equal(
        batch.signals.sort_values(["symbol", "signal_date"], ignore_index=True),
        expected_signals,
    )
    pd.testing.assert_frame_equal(
        batch.early_cut.sort_values(
            ["symbol", "signal_date", "landmark_day"], ignore_index=True
        ),
        expected_early,
    )

    empty = calculate_signal_batch(first.iloc[0:0], dates, cfg)
    assert empty.signals.empty
    assert empty.early_cut.empty
    assert list(empty.signals.columns) == list(SIGNAL_COLUMNS)
    assert list(empty.early_cut.columns) == list(EARLY_CUT_COLUMNS)


def test_calculate_early_cut_landmarks_accepts_precomputed_signals(
    cfg_factory,
) -> None:
    dates = pd.date_range("2023-03-01", periods=35, freq="D")
    source = _source_frame(dates)
    _set_pass(source, 25)
    cfg = cfg_factory()
    signals = calculate_identity_signals(source, dates, cfg)

    direct = calculate_early_cut_landmarks(source, dates, cfg, signals=signals)
    wrapped = calculate_identity_results(source, dates, cfg).early_cut

    pd.testing.assert_frame_equal(direct, wrapped)
