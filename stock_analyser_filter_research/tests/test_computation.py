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
    build_global_market_context,
    enrich_global_features,
)
from stock_analyser_filter_research.contracts import (
    CRITERION_COLUMNS,
    EARLY_CUT_COLUMNS,
    EARLY_CUT_FEATURE_GROUPS,
    SIGNAL_COLUMNS,
    SOFT_PATTERN_FEATURE_COLUMNS,
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
            "adjusted_open": close,
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
    dates = pd.date_range("2022-03-01", periods=60, freq="D")
    signal_position = 45
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
    mutated.loc[mutated.index[signal_position], "adjusted_high"] = 10**7
    mutated.loc[mutated.index[signal_position], "adjusted_low"] = 0.01
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
        "prior_max_drawdown_63d_pct",
        "prior_max_drawdown_126d_pct",
        "prior_price_volume_corr21",
        "prior_price_notional_corr21",
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
        "prior_distribution_day_count_20",
        "prior_churning_day_count_20",
        "prior_failed_breakout_count_20",
        "prior_return_42d_pct",
        "prior_daily_return_std_21d_pct",
        "prior_downside_return_std_21d_pct",
        "prior_atr_5d_pct",
        "prior_atr_21d_pct",
        "prior_atr_5_vs21_ratio",
        "prior_volume_sma5_vs50_ratio",
        "prior_volume_sma21_vs50_ratio",
        "prior_notional_sma5_vs50_ratio",
        "prior_notional_sma21_vs50_ratio",
        "prior_up_down_volume_ratio21",
        "prior_volume_dryup_share10",
        "prior_obv_slope_20",
        "prior_accumulation_day_count_20",
        "prior_high_volume_down_day_count_20",
        "prior_base_width_10_pct",
        "prior_base_width_40_pct",
        "prior_tight_close_range_5_pct",
        "prior_tight_close_range_15_pct",
        "prior_range_compression_5_vs20_ratio",
        "prior_overhead_supply_share63",
        "prior_high_test_count_20",
        "prior_high_slope_20_pct_per_session",
        "prior_low_slope_20_pct_per_session",
        "prior_contraction_count_40",
        "prior_return_efficiency_63",
        "prior_rs_rating_change_21d",
        *[
            column
            for column in SOFT_PATTERN_FEATURE_COLUMNS
            if column.endswith("_setup_score")
        ],
    ]
    pd.testing.assert_series_equal(
        baseline_row[prior_features],
        mutated_row[prior_features],
        check_names=False,
    )


def test_prior_chart_geometry_distinguishes_ordered_trend_and_v_recovery(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-08-01", periods=75, freq="D")
    signal_position = 60

    ordered = _source_frame(dates)
    ordered_close = np.linspace(80.0, 119.0, 40)
    prior_index = ordered.index[signal_position - 40 : signal_position]
    ordered.loc[prior_index, "adjusted_close"] = ordered_close
    ordered.loc[prior_index, "adjusted_high"] = ordered_close * 1.005
    ordered.loc[prior_index, "adjusted_low"] = ordered_close * 0.995
    _set_pass(ordered, signal_position)
    ordered_row = _row_for_date(
        calculate_identity_signals(ordered, dates, cfg_factory()),
        dates[signal_position],
    )

    assert ordered_row["prior_trend_slope_20_pct_per_session"] > 0
    assert ordered_row["prior_trend_r2_20"] > 0.99
    assert ordered_row["prior_trend_efficiency_20"] == pytest.approx(1.0)
    assert ordered_row["prior_positive_return_share_20"] == pytest.approx(1.0)
    assert ordered_row["prior_peak_age_40_sessions"] == 0
    assert ordered_row["pattern_ordered_uptrend_setup_score"] > 80.0
    assert ordered_row["pattern_ordered_uptrend_score_20d"] > 70.0

    v_shape = _source_frame(dates)
    v_close = np.concatenate(
        [
            np.linspace(100.0, 120.0, 12),
            np.linspace(116.0, 75.0, 10),
            np.linspace(78.0, 116.0, 18),
        ]
    )
    v_shape.loc[prior_index, "adjusted_close"] = v_close
    v_shape.loc[prior_index, "adjusted_high"] = v_close + 1.0
    v_shape.loc[prior_index, "adjusted_low"] = v_close - 1.0
    _set_pass(v_shape, signal_position)
    v_row = _row_for_date(
        calculate_identity_signals(v_shape, dates, cfg_factory()),
        dates[signal_position],
    )

    assert v_row["prior_drawdown_to_trough_40_pct"] < -35.0
    assert v_row["prior_recovery_from_trough_40_pct"] > 50.0
    assert v_row["prior_v_recovery_fraction_40"] > 0.80
    assert v_row["prior_trough_age_40_sessions"] > 0
    assert v_row["pattern_v_recovery_setup_score"] > 60.0
    assert (
        v_row["pattern_v_recovery_setup_score"]
        > ordered_row["pattern_v_recovery_setup_score"]
    )
    for column in SOFT_PATTERN_FEATURE_COLUMNS:
        for row in (ordered_row, v_row):
            value = row[column]
            assert pd.isna(value) or 0.0 <= float(value) <= 100.0


def test_global_market_context_is_same_close_causal_and_prior_returns_exclude_signal_day() -> None:
    dates = pd.bdate_range("2026-06-01", periods=30)
    rows: list[dict[str, object]] = []
    for position, market_date in enumerate(dates):
        close_at = pd.Timestamp(market_date, tz="America/New_York") + pd.Timedelta(
            hours=16
        )
        rows.append(
            {
                "source": "twelve_data",
                "series_id": "SPY",
                "observation_time": close_at,
                "value": 100.0 + position,
                "available_at": close_at + pd.Timedelta(minutes=30),
                "asof_known_at": close_at + pd.Timedelta(minutes=30),
                "is_revision_prone": False,
                "is_final": True,
                "source_local_date": market_date.date(),
            }
        )
    last_close = pd.Timestamp(dates[-1], tz="America/New_York") + pd.Timedelta(
        hours=16
    )
    rows.append(
        {
            **rows[-1],
            "value": 999.0,
            "available_at": last_close + pd.Timedelta(hours=2),
            "asof_known_at": last_close + pd.Timedelta(hours=2),
        }
    )

    context = build_global_market_context(
        dates,
        pd.DataFrame(),
        pd.DataFrame(rows),
    )
    last = context.iloc[-1]
    assert last["_market_spy_level"] == pytest.approx(129.0)
    assert last["market_spy_prior_return_5d_pct"] == pytest.approx(
        (128.0 / 123.0 - 1.0) * 100.0
    )

    signal_rows = []
    for symbol, return_21d in (("LOW", 5.0), ("HIGH", 15.0)):
        row = {column: pd.NA for column in SIGNAL_COLUMNS}
        row.update(
            {
                "signal_date": dates[-1],
                "symbol": symbol,
                "prior_return_21d_pct": return_21d,
                "prior_return_63d_pct": return_21d,
                "prior_return_126d_pct": return_21d,
                "prior_return_252d_pct": return_21d,
            }
        )
        signal_rows.append(row)
    enriched = enrich_global_features(
        pd.DataFrame(signal_rows, columns=SIGNAL_COLUMNS),
        pd.DataFrame(columns=EARLY_CUT_COLUMNS),
        context,
    ).signals
    assert enriched["cross_sectional_rs_21d_pct_rank"].tolist() == [0.5, 1.0]
    expected_market_21d = (128.0 / 107.0 - 1.0) * 100.0
    assert enriched.loc[1, "relative_return_vs_spy_21d_pct_points"] == (
        pytest.approx(15.0 - expected_market_21d)
    )


def test_prior_distribution_churning_and_failed_breakout_counts_are_causal(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-11-01", periods=80, freq="D")
    signal_position = 65
    source = _source_frame(dates)
    source["adjusted_close"] = 100.0
    source["adjusted_high"] = 101.0
    source["adjusted_low"] = 99.0
    source["adjusted_volume"] = 1_000.0

    churning_position = signal_position - 10
    source.loc[source.index[churning_position], "adjusted_high"] = 102.0
    source.loc[source.index[churning_position], "adjusted_low"] = 98.0
    source.loc[source.index[churning_position], "adjusted_volume"] = 2_000.0

    distribution_position = signal_position - 5
    source.loc[source.index[distribution_position], "adjusted_close"] = 99.0
    source.loc[source.index[distribution_position], "adjusted_high"] = 103.0
    source.loc[source.index[distribution_position], "adjusted_low"] = 98.5
    source.loc[source.index[distribution_position], "adjusted_volume"] = 2_000.0
    _set_pass(source, signal_position)

    row = _row_for_date(
        calculate_identity_signals(source, dates, cfg_factory()),
        dates[signal_position],
    )
    assert row["prior_distribution_day_count_20"] == 1
    assert row["prior_churning_day_count_20"] == 1
    assert row["prior_failed_breakout_count_20"] == 1


def test_gap_inside_pattern_history_invalidates_ordered_features(cfg_factory) -> None:
    dates = pd.date_range("2023-02-01", periods=80, freq="D")
    signal_position = 65
    source = _source_frame(dates)
    source = source.loc[source["period_end_date"].ne(dates[50])].copy()
    signal_index = source.index[source["period_end_date"].eq(dates[signal_position])][0]
    _set_pass(source, source.index.get_loc(signal_index))

    row = _row_for_date(
        calculate_identity_signals(source, dates, cfg_factory()),
        dates[signal_position],
    )

    assert pd.isna(row["prior_base_width_20_pct"])
    assert pd.isna(row["prior_trend_r2_20"])
    assert pd.isna(row["prior_v_recovery_fraction_40"])
    assert pd.isna(row["prior_distribution_day_count_20"])
    assert pd.isna(row["pattern_ordered_uptrend_score_20d"])
    assert pd.isna(row["pattern_v_recovery_score_40d"])


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
    early_only = result.early_cut.loc[result.early_cut["landmark_day"].le(3)]
    early_splits = (
        early_only.groupby("signal_date", sort=False)["analysis_split"]
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
    assert result.early_cut["landmark_day"].tolist() == [1, 2, 3, 5, 20, 30]
    early_only = result.early_cut.loc[result.early_cut["landmark_day"].le(3)]
    assert early_only["active_at_landmark"].astype(bool).equals(
        early_only["eligible_at_landmark"].astype(bool)
    )
    assert early_only["prior_policy_cut_day"].isna().all()
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
    early = early.loc[early["landmark_day"].le(3)]
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


def test_entry_horizons_and_intraday_hard_stop_are_derived_from_ohlc(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-10-01", periods=130, freq="D")
    signal_position = 25
    source = _source_frame(dates)
    _set_pass(source, signal_position)

    positions = source.index[signal_position : signal_position + 91]
    offsets = np.arange(91)
    closes = np.interp(
        offsets,
        [0, 5, 20, 30, 60, 90],
        [100.0, 100.0, 104.0, 106.0, 110.0, 120.0],
    )
    source.loc[positions, "adjusted_close"] = closes
    source.loc[positions, "raw_close"] = closes
    source.loc[positions, "adjusted_open"] = closes
    source.loc[positions, "adjusted_high"] = closes + 0.5
    source.loc[positions, "adjusted_low"] = closes - 0.5
    source.loc[positions[2], "adjusted_low"] = 89.9

    result = calculate_identity_results(source, dates, cfg_factory())
    signal = _row_for_date(result.signals, dates[signal_position])

    assert signal["first_loss_10pct_day"] == 2
    assert bool(signal["hard_stop_10pct_5d"])
    assert bool(signal["stagnant_5d"])
    assert signal["terminal_close_return_20d_pct"] == pytest.approx(4.0)
    assert signal["terminal_close_return_30d_pct"] == pytest.approx(6.0)
    assert bool(signal["terminal_winner_30d"])
    assert not bool(signal["runner_60d"])
    assert not bool(signal["runner_90d"])
    assert signal["forward_90d_label_end_date"] == dates[signal_position + 90]

    day5 = _landmark_row(result.early_cut, 5)
    assert bool(day5["hit_loss_10pct_so_far"])
    assert day5["management_decision"] == "hard_stop"
    assert not bool(day5["management_include_final"])


def test_management_landmarks_compare_next_open_with_d40_d60_and_d90(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-11-01", periods=135, freq="D")
    signal_position = 25
    source = _source_frame(dates)
    _set_pass(source, signal_position)

    positions = source.index[signal_position : signal_position + 91]
    offsets = np.arange(91)
    closes = np.interp(
        offsets,
        [0, 5, 20, 30, 40, 60, 90],
        [100.0, 103.0, 110.0, 115.0, 105.0, 120.0, 125.0],
    )
    opens = closes.copy()
    opens[21] = 111.0
    opens[31] = 114.0
    highs = np.maximum(closes + 0.5, opens)
    lows = np.minimum(closes - 0.5, opens)
    source.loc[positions, "adjusted_close"] = closes
    source.loc[positions, "raw_close"] = closes
    source.loc[positions, "adjusted_open"] = opens
    source.loc[positions, "adjusted_high"] = highs
    source.loc[positions, "adjusted_low"] = lows

    landmarks = calculate_identity_results(
        source, dates, cfg_factory()
    ).early_cut
    day20 = _landmark_row(landmarks, 20)
    day30 = _landmark_row(landmarks, 30)

    assert bool(day20["eligible_at_landmark"])
    assert day20["effective_session_date"] == dates[signal_position + 21]
    assert day20["effective_adjusted_open"] == pytest.approx(111.0)
    assert day20["terminal_return_from_effective_open_to_day40_pct"] == (
        pytest.approx((105.0 / 111.0 - 1.0) * 100.0)
    )
    assert bool(day20["take_profit_better_to_day40"])
    assert bool(day20["continue_winner_to_day60"])
    assert bool(day20["continue_winner_to_day90"])
    assert bool(day20["full_outcome_available"])

    assert bool(day30["eligible_at_landmark"])
    assert day30["effective_session_date"] == dates[signal_position + 31]
    assert day30["effective_adjusted_open"] == pytest.approx(114.0)
    assert bool(day30["take_profit_better_to_day40"])
    assert bool(day30["continue_winner_to_day60"])
    assert bool(day30["continue_winner_to_day90"])
    assert bool(day30["full_outcome_available"])


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
        assert pd.isna(day1["bad_to_day5"])
    else:
        assert tuple(bool(day1[column]) for column in label_columns) == flags
        assert bool(day1["bad_to_day5"]) is bool(flags[0] or flags[1])


def test_prior_decline_without_further_landmark_loss_is_not_loss_first(
    cfg_factory,
) -> None:
    dates = pd.date_range("2022-10-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    _set_path(
        source,
        position,
        closes=[100, 96, 95, 95, 95, 95],
        highs=[101, 97, 96, 96, 96, 96],
        lows=[99, 95.1, 94.9, 94.9, 94.9, 94.9],
    )

    day1 = _landmark_row(
        calculate_identity_results(source, dates, cfg_factory()).early_cut,
        1,
    )

    assert bool(day1["eligible_at_landmark"])
    assert day1["remaining_max_loss_from_signal_pct"] == pytest.approx(-5.1)
    assert day1["remaining_max_loss_from_landmark_pct"] == pytest.approx(
        (94.9 / 96 - 1) * 100
    )
    assert pd.isna(day1["future_first_loss_5pct_day"])
    assert day1["continuation_outcome"] == "stagnant"
    assert not bool(day1["loss_first_to_day5"])
    assert bool(day1["stagnant_to_day5"])


@pytest.mark.parametrize(
    ("landmark_close", "future_high", "future_low", "outcome", "hit_column"),
    [
        (96.0, 100.9, 95.0, "strong_first", "future_first_gain_5pct_day"),
        (104.0, 105.0, 98.7, "loss_first", "future_first_loss_5pct_day"),
    ],
)
def test_future_gain_and_loss_barriers_are_relative_to_landmark_close(
    cfg_factory,
    landmark_close,
    future_high,
    future_low,
    outcome,
    hit_column,
) -> None:
    dates = pd.date_range("2022-10-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    _set_pass(source, position)
    landmark_high = min(104.9, landmark_close + 1.0)
    landmark_low = max(95.1, landmark_close - 1.0)
    _set_path(
        source,
        position,
        closes=[
            100,
            landmark_close,
            landmark_close,
            landmark_close,
            landmark_close,
            landmark_close,
        ],
        highs=[
            101,
            landmark_high,
            future_high,
            landmark_close + 1,
            landmark_close + 1,
            landmark_close + 1,
        ],
        lows=[
            99,
            landmark_low,
            future_low,
            landmark_close - 1,
            landmark_close - 1,
            landmark_close - 1,
        ],
    )

    day1 = _landmark_row(
        calculate_identity_results(source, dates, cfg_factory()).early_cut,
        1,
    )

    assert bool(day1["eligible_at_landmark"])
    assert day1[hit_column] == 2
    assert day1["continuation_outcome"] == outcome
    assert bool(day1[f"{outcome}_to_day5"])
    if outcome == "strong_first":
        assert day1["remaining_max_gain_from_signal_pct"] < 5.0
        assert day1["remaining_max_gain_from_landmark_pct"] > 5.0
    else:
        assert day1["remaining_max_loss_from_signal_pct"] > -5.0
        assert day1["remaining_max_loss_from_landmark_pct"] < -5.0


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
    early = early.loc[early["landmark_day"].le(3)]

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

    assert len(result.early_cut) == 6
    early = result.early_cut.loc[result.early_cut["landmark_day"].le(3)]
    assert early["landmark_day"].tolist() == [1, 2, 3]
    day1 = _landmark_row(early, 1)
    assert bool(day1["same_continuity_segment"])
    assert bool(day1["eligible_at_landmark"])
    assert (
        not early.loc[
            early["landmark_day"].ge(2), "same_continuity_segment"
        ]
        .astype(bool)
        .any()
    )
    assert not early["full_outcome_available"].astype(bool).any()
    assert day1["cut_decision"] == "hold"
    assert (
        early.loc[early["landmark_day"].ge(2), "cut_decision"]
        .eq("not_evaluable")
        .all()
    )
    assert (
        early[
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


def test_nullable_string_gap_does_not_require_boolean_value_of_na(
    cfg_factory,
) -> None:
    dates = pd.date_range("2023-01-01", periods=40, freq="D")
    position = 25
    source = _source_frame(dates)
    source["symbol"] = source["symbol"].astype("string")
    source["exchange"] = source["exchange"].astype("string")
    _set_pass(source, position)
    source = source.loc[source["period_end_date"].ne(dates[position + 2])]

    result = calculate_identity_results(source, dates, cfg_factory())

    assert len(result.early_cut) == 6
    day2 = _landmark_row(result.early_cut, 2)
    assert not bool(day2["landmark_observed"])
    assert day2["cut_decision"] == "not_evaluable"


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
    assert len(result.early_cut) == 6
    day1 = _landmark_row(result.early_cut, 1)
    day2 = _landmark_row(result.early_cut, 2)
    day3 = _landmark_row(result.early_cut, 3)
    assert day1["effective_session_date"] == dates[29]
    assert pd.isna(day2["effective_session_date"])
    assert pd.isna(day3["landmark_date"])
    early = result.early_cut.loc[result.early_cut["landmark_day"].le(3)]
    assert not early["full_outcome_available"].astype(bool).any()
    assert early["continuation_outcome"].isna().all()


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
    early = result.early_cut.loc[result.early_cut["landmark_day"].le(3)]
    assert early["landmark_date"].gt(signal_date).all()
    assert early["analysis_split"].eq("holdout").all()


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
    assert len(batch.early_cut) == 12
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
