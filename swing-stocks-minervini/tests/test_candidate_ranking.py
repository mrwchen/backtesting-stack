from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.candidate_ranking import (
    CandidateRanker,
    FillCalibrationLabel,
    QUALITY_VALIDATION_MIN_DAYS_PER_GROUP,
    QUALITY_VALIDATION_MIN_LABELS,
    QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER,
    QUALITY_VALIDATION_QUARTERS_PER_REVIEW,
    QualityCalibrationLabel,
)


def _matrices(periods: int = 80, *, start: str = "2024-01-02"):
    dates = pd.bdate_range(start, periods=periods)
    symbols = pd.Index(["AAA", "BBB"])
    close = pd.DataFrame(
        {
            "AAA": np.linspace(90.0, 99.0, periods),
            "BBB": np.linspace(88.0, 99.0, periods),
        },
        index=dates,
    )
    volume = pd.DataFrame(1_000_000.0, index=dates, columns=symbols)
    volume.iloc[-5:, :] = 500_000.0
    rs = pd.DataFrame({"AAA": 92.0, "BBB": 92.0}, index=dates)
    fundamentals = pd.DataFrame({"AAA": 4.0, "BBB": 4.0}, index=dates)
    return dates, symbols, close, volume, rs, fundamentals


def _setup(dates, symbol="AAA", setup_type="vcp", setup_id=1, **overrides):
    values = {
        "setup_id": setup_id,
        "symbol": symbol,
        "setup_type": setup_type,
        "detect_date": dates[45].date(),
        "pivot": 100.0,
        "dryup_ratio": 0.7,
        "rs_rating": 80.0,
        "fundamental_score": 3.0,
        "eps_yoy": 0.25,
        "revenue_yoy": 0.20,
        "structure_quality_score": 20.0,
        "tightness_score": 15.0,
        "prior_advance_score": 18.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _segments(dates, symbols):
    return pd.DataFrame(1, index=dates, columns=symbols, dtype=int)


def _ranker(dates, symbols, close, volume=None, rs=None, fundamentals=None, **kwargs):
    context = {"fundamental_score": fundamentals} if fundamentals is not None else None
    return CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        volume=volume,
        rs_rating=rs,
        context=context,
        **kwargs,
    )


def _validated_quality_labels(
    setup_type,
    raw_score,
    group_outcomes=(-1.5, -1.0, -0.5, -0.1),
    *,
    start_quarter="2020Q1",
    quarters=QUALITY_VALIDATION_QUARTERS_PER_REVIEW,
):
    """Plausible OOF outcomes spread over fixed quarters and completion days."""
    labels = []
    raw_scores = np.linspace(
        max(0.0, float(raw_score) - 30.0),
        min(100.0, float(raw_score) + 30.0),
        4,
    )
    days_per_quarter = QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER // 4
    assert days_per_quarter >= QUALITY_VALIDATION_MIN_DAYS_PER_GROUP
    for quarter in pd.period_range(start_quarter, periods=quarters, freq="Q"):
        available_days = pd.bdate_range(
            quarter.start_time + pd.Timedelta(days=14),
            quarter.end_time - pd.Timedelta(days=7),
        ).normalize()
        completion_days = available_days[
            np.linspace(
                0, len(available_days) - 1, num=days_per_quarter, dtype=int
            )
        ]
        for completion_date in completion_days:
            for group, (oof_score, outcome) in enumerate(
                zip((-1.5, -0.5, 0.5, 1.5), group_outcomes)
            ):
                labels.append(
                    QualityCalibrationLabel(
                        setup_type=setup_type,
                        information_date=completion_date
                        - pd.offsets.BDay(3 + group),
                        available_date=completion_date,
                        # Distinct raw signals can genuinely produce distinct
                        # OOF posteriors from one deterministic prior model.
                        raw_quality_score=raw_scores[group],
                        realized_r_multiple=outcome,
                        walk_forward_quality_score=oof_score,
                    )
                )
    return labels


def test_snapshot_is_strictly_pre_session_causal():
    dates, symbols, close, volume, rs, fundamentals = _matrices()
    session_idx = 60
    baseline = _ranker(
        dates, symbols, close, volume, rs, fundamentals
    ).snapshot(_setup(dates), session_idx)

    changed_close = close.copy()
    changed_volume = volume.copy()
    changed_rs = rs.copy()
    changed_fundamentals = fundamentals.copy()
    changed_close.iloc[session_idx:, 0] = 10_000.0
    changed_volume.iloc[session_idx:, 0] = 100.0
    changed_rs.iloc[session_idx:, 0] = 1.0
    changed_fundamentals.iloc[session_idx:, 0] = 0.0
    changed = _ranker(
        dates,
        symbols,
        changed_close,
        changed_volume,
        changed_rs,
        changed_fundamentals,
    ).snapshot(_setup(dates), session_idx)

    assert changed == baseline
    assert baseline.information_date == dates[session_idx - 1]
    assert baseline.session_date == dates[session_idx]
    assert baseline.setup_age_sessions == session_idx - 46


def test_readiness_changes_fill_but_never_quality_signal_or_score():
    dates, symbols, close, _, _, _ = _matrices()
    near_close = close.copy()
    far_close = close.copy()
    near_close.iloc[59, 0] = 99.0
    far_close.iloc[59, 0] = 90.0
    labels = [
        FillCalibrationLabel("vcp", dates[20], dates[21], 90.0, True),
        FillCalibrationLabel("vcp", dates[22], dates[23], 0.0, False),
    ]
    near = CandidateRanker(
        dates,
        symbols,
        near_close,
        continuity_segment=_segments(dates, symbols),
        fill_labels=labels,
        fill_prior_strength=1.0,
    ).snapshot(_setup(dates), 60)
    far = CandidateRanker(
        dates,
        symbols,
        far_close,
        continuity_segment=_segments(dates, symbols),
        fill_labels=labels,
        fill_prior_strength=1.0,
    ).snapshot(_setup(dates), 60)

    assert near.readiness_score > far.readiness_score
    assert near.raw_quality_score == pytest.approx(far.raw_quality_score)
    assert near.quality_score == pytest.approx(far.quality_score)
    assert near.fill_probability > far.fill_probability
    assert near.distance_to_pivot_pct == pytest.approx(1.0)


def test_future_labels_are_ignored_until_their_available_date():
    dates, symbols, close, _, _, _ = _matrices()
    setup = _setup(dates)
    probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(setup, 60)
    future_quality = QualityCalibrationLabel(
        "vcp",
        dates[50],
        dates[65],
        probe.raw_quality_score,
        4.0,
        0.0,
    )
    future_fill = FillCalibrationLabel(
        "vcp", dates[50], dates[65], probe.readiness_score, True
    )
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=[future_quality],
        fill_labels=[future_fill],
        quality_prior_strength=1.0,
        fill_prior_strength=1.0,
    )

    before = ranker.snapshot(setup, 60)
    after = ranker.snapshot(setup, 67)

    assert before.quality_calibration_count == 0
    assert before.fill_calibration_count == 0
    assert before.quality_score == 0.0
    assert before.fill_probability == 0.5
    assert after.quality_calibration_count == 1
    assert after.fill_calibration_count == 1
    assert after.walk_forward_quality_score > before.walk_forward_quality_score
    assert after.quality_score == before.quality_score == 0.0
    assert not after.quality_model_validated
    assert after.fill_probability > before.fill_probability


def test_calibration_is_strictly_isolated_by_setup_class():
    dates, symbols, close, _, _, _ = _matrices()
    vcp_probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(
        _setup(dates, setup_type="vcp"), 60
    )
    labels = [
        QualityCalibrationLabel(
            "vcp",
            dates[20],
            dates[21],
            vcp_probe.raw_quality_score,
            3.0,
            0.0,
        )
    ]
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
        quality_prior_strength=1.0,
    )
    vcp = ranker.snapshot(_setup(dates, setup_type="vcp"), 60)
    power = ranker.snapshot(_setup(dates, setup_type="power_play"), 60)

    assert vcp.walk_forward_quality_score > 0.0
    assert vcp.quality_score == 0.0
    assert not vcp.quality_model_validated
    assert vcp.quality_calibration_count == 1
    assert power.quality_score == 0.0
    assert power.quality_calibration_count == 0


def test_small_samples_are_shrunk_toward_explicit_neutral_priors():
    dates, symbols, close, _, _, _ = _matrices()
    probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(_setup(dates), 60)
    one_winner = QualityCalibrationLabel(
        "vcp",
        dates[20],
        dates[21],
        probe.raw_quality_score,
        6.0,
        walk_forward_quality_score=0.0,
    )
    one_fill = FillCalibrationLabel(
        "vcp", dates[20], dates[21], probe.readiness_score, True
    )
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=[one_winner],
        fill_labels=[one_fill],
        quality_prior_strength=12.0,
        fill_prior_strength=12.0,
    ).snapshot(_setup(dates), 60)

    assert 0.0 < snapshot.walk_forward_quality_score < 1.0
    assert snapshot.quality_score == 0.0
    assert snapshot.slate_priority == 0.0
    assert not snapshot.quality_model_validated
    assert snapshot.quality_calibration_count == 1
    assert snapshot.quality_validation_count == 0
    assert 0.5 < snapshot.fill_probability < 0.55


def test_quality_activates_only_after_robust_monotone_purged_oof_lift():
    dates, symbols, close, _, _, _ = _matrices(periods=800, start="2020-01-02")
    setup = _setup(dates)
    session_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(setup, session_idx)
    labels = _validated_quality_labels(
        "vcp", probe.raw_quality_score
    )
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    ).snapshot(setup, session_idx)

    assert snapshot.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert snapshot.quality_model_validated
    assert snapshot.quality_score == pytest.approx(
        snapshot.walk_forward_quality_score
    )
    # Validation proves ranking lift, not profitability.  All four outcome
    # groups in this fixture lose money and are still correctly ordered.
    assert snapshot.quality_score < 0.0
    assert snapshot.slate_priority == pytest.approx(
        snapshot.quality_score * snapshot.fill_probability
    )


def test_online_snapshot_and_add_label_path_can_validate_next_epoch():
    dates, symbols, close, _, _, _ = _matrices(
        periods=1100, start="2018-01-02"
    )
    raw_scores = (12.5, 37.5, 62.5, 87.5)
    outcomes = (-1.5, -1.0, -0.5, -0.1)
    seed_labels = []
    seed_start = pd.Timestamp("2019-01-03")
    for repeat in range(12):
        available_date = seed_start + pd.offsets.BDay(repeat + 1)
        for raw_score, outcome in zip(raw_scores, outcomes):
            seed_labels.append(
                QualityCalibrationLabel(
                    setup_type="vcp",
                    information_date=available_date - pd.offsets.BDay(1),
                    available_date=available_date,
                    raw_quality_score=raw_score,
                    realized_r_multiple=outcome,
                    # The seed is outside the epoch under review; it exists
                    # only to define a causal prior mapping for the first
                    # production-generated OOF forecasts.
                    walk_forward_quality_score=0.0,
                )
            )
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_type_weights={
            "default": {"structure": 1.0},
            "vcp": {"structure": 1.0},
        },
        quality_labels=seed_labels,
    )

    generated = []
    days_per_quarter = QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER // 4
    for quarter in pd.period_range(
        "2020Q1", periods=QUALITY_VALIDATION_QUARTERS_PER_REVIEW, freq="Q"
    ):
        available_days = pd.bdate_range(
            quarter.start_time + pd.Timedelta(days=14),
            quarter.end_time - pd.Timedelta(days=7),
        ).normalize()
        completion_days = available_days[
            np.linspace(
                0,
                len(available_days) - 1,
                num=days_per_quarter,
                dtype=int,
            )
        ]
        for completion_date in completion_days:
            session_idx = int(dates.get_loc(completion_date))
            snapshots = [
                ranker.snapshot(
                    _setup(
                        dates,
                        setup_id=group + 1,
                        structure_quality_score=raw_score / 4.0,
                    ),
                    session_idx,
                    setup_key=group,
                )
                for group, raw_score in enumerate(raw_scores)
            ]
            assert all(
                later.walk_forward_quality_score
                > earlier.walk_forward_quality_score
                for earlier, later in zip(snapshots, snapshots[1:])
            )
            for snapshot, outcome in zip(snapshots, outcomes):
                label = ranker.add_quality_label(
                    snapshot,
                    available_date=completion_date,
                    realized_r_multiple=outcome,
                )
                assert label.walk_forward_quality_score == pytest.approx(
                    snapshot.walk_forward_quality_score
                )
                generated.append(label)

    review_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    reviewed = ranker.snapshot(
        _setup(dates, structure_quality_score=20.0), review_idx
    )

    assert len(generated) == QUALITY_VALIDATION_MIN_LABELS
    assert reviewed.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert reviewed.quality_model_validated


def test_inverted_oof_relationship_never_activates_quality():
    dates, symbols, close, _, _, _ = _matrices(periods=800, start="2020-01-02")
    setup = _setup(dates)
    session_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(setup, session_idx)
    labels = _validated_quality_labels(
        "vcp",
        probe.raw_quality_score,
        group_outcomes=(1.5, 0.5, -0.5, -1.5),
    )
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    ).snapshot(setup, session_idx)

    assert snapshot.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert not snapshot.quality_model_validated
    assert snapshot.quality_score == 0.0
    assert snapshot.slate_priority == 0.0


def test_fewer_than_eight_complete_quarter_blocks_fails_closed():
    dates, symbols, close, _, _, _ = _matrices(periods=800, start="2020-01-02")
    setup = _setup(dates)
    session_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    probe = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(setup, session_idx)
    labels = _validated_quality_labels(
        "vcp",
        probe.raw_quality_score,
        quarters=QUALITY_VALIDATION_QUARTERS_PER_REVIEW - 1,
    )
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    ).snapshot(setup, session_idx)

    assert snapshot.quality_validation_count == (
        QUALITY_VALIDATION_MIN_LABELS
        - QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER
    )
    assert not snapshot.quality_model_validated
    assert snapshot.quality_score == snapshot.slate_priority == 0.0


def test_regime_drift_across_quarter_blocks_stays_neutral():
    dates, symbols, close, _, _, _ = _matrices(periods=800, start="2020-01-02")
    labels = [
        *_validated_quality_labels(
            "vcp", 50.0, start_quarter="2020Q1", quarters=4
        ),
        *_validated_quality_labels(
            "vcp",
            50.0,
            group_outcomes=(1.5, 0.5, -0.5, -1.5),
            start_quarter="2021Q1",
            quarters=4,
        )
    ]
    session_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    ).snapshot(_setup(dates), session_idx)

    assert snapshot.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert not snapshot.quality_model_validated
    assert snapshot.quality_score == snapshot.slate_priority == 0.0


def test_same_day_exit_cluster_cannot_manufacture_or_destroy_lift():
    dates, symbols, close, _, _, _ = _matrices(periods=800, start="2020-01-02")
    labels = _validated_quality_labels("vcp", 50.0)
    cluster_date = pd.Timestamp("2020-02-03")
    for _ in range(200):
        for group, (score, outcome) in enumerate(
            zip((-1.5, -0.5, 0.5, 1.5), (1.5, 0.5, -0.5, -1.5))
        ):
            labels.append(
                QualityCalibrationLabel(
                    setup_type="vcp",
                    information_date=cluster_date - pd.offsets.BDay(3 + group),
                    available_date=cluster_date,
                    raw_quality_score=50.0,
                    realized_r_multiple=outcome,
                    walk_forward_quality_score=score,
                )
            )
    session_idx = int(dates.searchsorted(pd.Timestamp("2022-01-04")))
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    ).snapshot(_setup(dates), session_idx)

    assert snapshot.quality_validation_count == len(labels)
    assert snapshot.quality_model_validated

    inverted_labels = _validated_quality_labels(
        "vcp", 50.0, group_outcomes=(1.5, 0.5, -0.5, -1.5)
    )
    for quarter in pd.period_range("2020Q1", periods=8, freq="Q"):
        good_cluster_date = (
            quarter.start_time + pd.offsets.BDay(25)
        ).normalize()
        for _ in range(200):
            for group, (score, outcome) in enumerate(
                zip((-1.5, -0.5, 0.5, 1.5), (-1.5, -1.0, -0.5, -0.1))
            ):
                inverted_labels.append(
                    QualityCalibrationLabel(
                        setup_type="vcp",
                        information_date=good_cluster_date
                        - pd.offsets.BDay(3 + group),
                        available_date=good_cluster_date,
                        raw_quality_score=50.0,
                        realized_r_multiple=outcome,
                        walk_forward_quality_score=score,
                    )
                )
    inverted_snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=inverted_labels,
    ).snapshot(_setup(dates), session_idx)

    assert not inverted_snapshot.quality_model_validated
    assert inverted_snapshot.quality_score == inverted_snapshot.slate_priority == 0.0


def test_review_is_frozen_until_next_non_overlapping_epoch_checkpoint():
    dates, symbols, close, _, _, _ = _matrices(periods=1150, start="2020-01-02")
    setup = _setup(dates)
    labels = [
        *_validated_quality_labels("vcp", 50.0, start_quarter="2020Q1"),
        *_validated_quality_labels(
            "vcp",
            50.0,
            group_outcomes=(1.5, 0.5, -0.5, -1.5),
            start_quarter="2022Q1",
        ),
    ]
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=labels,
    )
    early_idx = int(dates.searchsorted(pd.Timestamp("2022-02-01")))
    late_idx = int(dates.searchsorted(pd.Timestamp("2023-12-15")))
    next_checkpoint_idx = int(dates.searchsorted(pd.Timestamp("2024-01-03")))

    early = ranker.snapshot(setup, early_idx)
    late = ranker.snapshot(setup, late_idx)
    next_checkpoint = ranker.snapshot(setup, next_checkpoint_idx)

    assert early.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert late.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert early.quality_model_validated
    assert late.quality_model_validated
    assert next_checkpoint.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert not next_checkpoint.quality_model_validated
    assert next_checkpoint.quality_score == next_checkpoint.slate_priority == 0.0


def test_review_cache_changes_epoch_even_without_a_new_label():
    dates, symbols, close, _, _, _ = _matrices(periods=1150, start="2020-01-02")
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_labels=_validated_quality_labels("vcp", 50.0),
    )
    before_idx = int(dates.searchsorted(pd.Timestamp("2023-12-15")))
    after_idx = int(dates.searchsorted(pd.Timestamp("2024-01-03")))

    before = ranker.snapshot(_setup(dates), before_idx)
    after = ranker.snapshot(_setup(dates), after_idx)

    assert before.quality_model_validated
    assert before.quality_validation_count == QUALITY_VALIDATION_MIN_LABELS
    assert not after.quality_model_validated
    assert after.quality_validation_count == 0


def test_unvalidated_quality_and_fill_do_not_hide_in_neutral_tie_breaks():
    dates, symbols, close, _, _, _ = _matrices()
    close.loc[dates[59], "AAA"] = 90.0
    close.loc[dates[59], "BBB"] = 99.0
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        fill_labels=[
            FillCalibrationLabel("vcp", dates[20], dates[21], 0.0, False),
            FillCalibrationLabel("vcp", dates[22], dates[23], 90.0, True),
        ],
        fill_prior_strength=0.1,
        fill_kernel_bandwidth=10.0,
    )
    low_far_aaa = _setup(
        dates,
        symbol="AAA",
        setup_id=2,
        rs_rating=1.0,
        dryup_ratio=1.25,
        fundamental_score=0.0,
        structure_quality_score=0.0,
        tightness_score=0.0,
        prior_advance_score=0.0,
    )
    high_ready_bbb = _setup(dates, symbol="BBB", setup_id=1)

    first_symbols = set()
    for session_idx in range(60, 72):
        forward = ranker.rank([high_ready_bbb, low_far_aaa], session_idx)
        reverse = ranker.rank([low_far_aaa, high_ready_bbb], session_idx)
        assert forward == reverse
        by_symbol = {item.symbol: item for item in forward}
        assert by_symbol["BBB"].raw_quality_score > by_symbol["AAA"].raw_quality_score
        assert by_symbol["BBB"].fill_probability != by_symbol["AAA"].fill_probability
        assert all(not item.quality_model_validated for item in forward)
        assert all(
            item.quality_score == item.slate_priority == 0.0 for item in forward
        )
        first_symbols.add(forward[0].symbol)

    # The stable session lottery is reproducible but creates no permanent
    # lexical winner inside the unvalidated class.
    assert first_symbols == {"AAA", "BBB"}


def test_online_labels_cannot_affect_the_session_that_generated_them():
    dates, symbols, close, _, _, _ = _matrices()
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        quality_prior_strength=1.0,
        fill_prior_strength=1.0,
    )
    session_idx = 60
    before = ranker.snapshot(_setup(dates), session_idx)
    ranker.add_fill_label(before, available_date=dates[session_idx], filled=True)
    ranker.add_quality_label(
        before, available_date=dates[session_idx], realized_r_multiple=2.0
    )

    same_session = ranker.snapshot(_setup(dates), session_idx)
    next_session = ranker.snapshot(_setup(dates), session_idx + 1)

    assert same_session.quality_score == before.quality_score
    assert same_session.fill_probability == before.fill_probability
    assert next_session.walk_forward_quality_score > before.walk_forward_quality_score
    assert next_session.quality_score == before.quality_score == 0.0
    assert not next_session.quality_model_validated
    assert next_session.fill_probability > before.fill_probability
    assert ranker.quality_labels[0].walk_forward_quality_score == pytest.approx(
        before.walk_forward_quality_score
    )


def test_missing_features_are_neutral_and_reduce_quality_coverage():
    dates, symbols, close, _, _, _ = _matrices()
    missing_setup = _setup(
        dates,
        rs_rating=np.nan,
        dryup_ratio=np.nan,
        fundamental_score=np.nan,
        structure_quality_score=np.nan,
        tightness_score=np.nan,
        prior_advance_score=np.nan,
    )
    missing = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(missing_setup, 60)
    neutral_setup = _setup(
        dates,
        rs_rating=50.0,
        dryup_ratio=0.625,
        fundamental_score=3.0,
        structure_quality_score=12.5,
        tightness_score=10.0,
        prior_advance_score=12.5,
    )
    neutral = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    ).snapshot(neutral_setup, 60)

    assert missing.raw_quality_score == pytest.approx(neutral.raw_quality_score)
    assert missing.quality_feature_coverage < neutral.quality_feature_coverage
    assert missing.quality_score == neutral.quality_score == 0.0


def test_incomplete_dynamic_volume_does_not_use_stale_setup_dryup():
    dates, symbols, close, volume, _, _ = _matrices()
    volume.iloc[56:60, 0] = np.nan
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        volume=volume,
    ).snapshot(_setup(dates, dryup_ratio=0.2), 60)

    assert snapshot.dynamic_dryup_ratio is None
    assert snapshot.dryup_source == "missing"
    assert snapshot.dryup_coverage == 0.0


def test_dynamic_dryup_is_unknown_after_a_short_new_continuity_segment():
    dates, symbols, close, volume, _, _ = _matrices()
    segments = _segments(dates, symbols)
    segments.loc[dates[50]:, "AAA"] = 2
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=segments,
        volume=volume,
    ).snapshot(_setup(dates, detect_date=dates[51].date()), 60)

    assert snapshot.dynamic_dryup_ratio is None
    assert snapshot.dryup_source == "missing"
    assert snapshot.dryup_coverage == 0.0


def test_dynamic_dryup_uses_only_the_current_continuity_segment():
    dates, symbols, close, volume, _, _ = _matrices(periods=100)
    segments = _segments(dates, symbols)
    segments.loc[dates[60]:, "AAA"] = 2
    volume.loc[: dates[59], "AAA"] = 100_000.0
    volume.loc[dates[60] : dates[84], "AAA"] = 2_000_000.0
    volume.loc[dates[85] : dates[89], "AAA"] = 500_000.0
    snapshot = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=segments,
        volume=volume,
    ).snapshot(_setup(dates, detect_date=dates[65].date()), 90)

    assert snapshot.dynamic_dryup_ratio == pytest.approx(0.25)
    assert snapshot.dryup_source == "dynamic_partial"
    assert snapshot.dryup_coverage == pytest.approx(0.5)


def test_rank_is_deterministic_and_assigns_positive_quality_ranks():
    dates, symbols, close, _, _, _ = _matrices()
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    )
    aaa = _setup(dates, symbol="AAA", setup_id=2)
    bbb = _setup(dates, symbol="BBB", setup_id=1)

    forward = ranker.rank([bbb, aaa], 60)
    reverse = ranker.rank([aaa, bbb], 60)

    assert forward == reverse
    assert [item.quality_rank for item in forward] == [1, 1]
    assert all(np.isfinite(item.quality_score) for item in forward)
    assert all(np.isfinite(item.slate_priority) for item in forward)


def test_cold_start_round_robin_rotates_classes_without_cross_class_raw_comparison():
    dates, symbols, close, _, _, _ = _matrices()
    symbols = pd.Index(["A1", "A2", "B1", "B2", "C1", "C2"])
    close = pd.DataFrame(99.0, index=dates, columns=symbols)
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    )
    setups = [
        _setup(dates, symbol="A1", setup_type="flat_base", setup_id=1),
        _setup(dates, symbol="A2", setup_type="flat_base", setup_id=2),
        _setup(dates, symbol="B1", setup_type="power_play", setup_id=3),
        _setup(dates, symbol="B2", setup_type="power_play", setup_id=4),
        _setup(dates, symbol="C1", setup_type="vcp", setup_id=5),
        _setup(dates, symbol="C2", setup_type="vcp", setup_id=6),
    ]

    day_60 = ranker.rank(setups, 60)
    assert {item.setup_type for item in day_60[:3]} == {
        "flat_base",
        "power_play",
        "vcp",
    }
    assert {
        ranker.rank(setups, session_idx)[0].setup_type
        for session_idx in range(60, 80)
    } == {"flat_base", "power_play", "vcp"}
    assert all(item.quality_rank == 1 for item in day_60)


def test_cross_class_order_ignores_incomparable_quality_and_slate_values():
    dates, symbols, close, _, _, _ = _matrices()
    close.loc[dates[59], "AAA"] = 90.0
    close.loc[dates[59], "BBB"] = 99.0
    continuity = _segments(dates, symbols)
    vcp_setup = _setup(dates, symbol="AAA", setup_type="vcp", setup_id=1)
    power_setup = _setup(
        dates, symbol="BBB", setup_type="power_play", setup_id=2
    )
    probe_ranker = CandidateRanker(
        dates, symbols, close, continuity_segment=continuity
    )
    vcp_probe = probe_ranker.snapshot(vcp_setup, 60)
    power_probe = probe_ranker.snapshot(power_setup, 60)
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_prior_strength=0.1,
        fill_prior_strength=0.1,
        quality_labels=[
            *_validated_quality_labels(
                "vcp",
                vcp_probe.raw_quality_score,
                group_outcomes=(0.5, 1.5, 2.5, 3.5),
                start_quarter="2022Q1",
            ),
            *_validated_quality_labels(
                "power_play",
                power_probe.raw_quality_score,
                group_outcomes=(-0.5, 0.0, 0.5, 1.0),
                start_quarter="2022Q1",
            ),
        ],
        fill_labels=[
            FillCalibrationLabel(
                "vcp", dates[20], dates[21], vcp_probe.readiness_score, False
            ),
            FillCalibrationLabel(
                "power_play",
                dates[20],
                dates[21],
                power_probe.readiness_score,
                True,
            ),
        ],
    )

    slate = ranker.rank([vcp_setup, power_setup], 60)

    by_type = {item.setup_type: item for item in slate}
    assert by_type["vcp"].quality_score > by_type["power_play"].quality_score
    assert by_type["power_play"].slate_priority > by_type["vcp"].slate_priority
    # A quality rank is meaningful only relative to the same setup class.
    assert [item.quality_rank for item in slate] == [1, 1]

    # Reversing both classes' outcome scales must not change their neutral
    # cross-class order for the same session and salt.
    reversed_ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_prior_strength=0.1,
        fill_prior_strength=0.1,
        quality_labels=[
            *_validated_quality_labels(
                "vcp",
                vcp_probe.raw_quality_score,
                group_outcomes=(-0.5, 0.0, 0.5, 1.0),
                start_quarter="2022Q1",
            ),
            *_validated_quality_labels(
                "power_play",
                power_probe.raw_quality_score,
                group_outcomes=(0.5, 1.5, 2.5, 3.5),
                start_quarter="2022Q1",
            ),
        ],
        fill_labels=ranker.fill_labels,
    )
    reversed_slate = reversed_ranker.rank([power_setup, vcp_setup], 60)

    assert [item.setup_type for item in reversed_slate] == [
        item.setup_type for item in slate
    ]


def test_validated_quality_orders_and_ranks_only_inside_its_setup_class():
    dates, symbols, close, _, _, _ = _matrices()
    continuity = _segments(dates, symbols)
    high = _setup(dates, symbol="AAA", setup_type="vcp", setup_id=1)
    low = _setup(
        dates,
        symbol="BBB",
        setup_type="vcp",
        setup_id=2,
        rs_rating=1.0,
        dryup_ratio=1.25,
        fundamental_score=0.0,
        structure_quality_score=0.0,
        tightness_score=0.0,
        prior_advance_score=0.0,
    )
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_prior_strength=0.1,
        quality_labels=_validated_quality_labels(
            "vcp",
            50.0,
            group_outcomes=(-1.5, -0.5, 0.5, 1.5),
            start_quarter="2022Q1",
        ),
    )

    slate = ranker.rank([low, high], 60)

    assert all(item.quality_model_validated for item in slate)
    assert [item.symbol for item in slate] == ["AAA", "BBB"]
    assert [item.quality_rank for item in slate] == [1, 2]
    assert slate[0].quality_score > slate[1].quality_score


def test_unvalidated_neutral_does_not_globally_beat_validated_negative_quality():
    dates, symbols, close, _, _, _ = _matrices()
    continuity = _segments(dates, symbols)
    vcp = _setup(dates, symbol="AAA", setup_type="vcp", setup_id=1)
    unvalidated = _setup(
        dates, symbol="BBB", setup_type="power_play", setup_id=2
    )
    probe = CandidateRanker(
        dates, symbols, close, continuity_segment=continuity
    ).snapshot(vcp, 60)
    labels = _validated_quality_labels(
        "vcp",
        probe.raw_quality_score,
        group_outcomes=(-1.5, -1.0, -0.5, -0.1),
        start_quarter="2022Q1",
    )
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_labels=labels,
    )

    first_types = set()
    for session_idx in range(60, 75):
        slate = ranker.rank([unvalidated, vcp], session_idx)
        by_type = {item.setup_type: item for item in slate}
        assert by_type["vcp"].quality_model_validated
        assert by_type["vcp"].quality_score < 0.0
        assert not by_type["power_play"].quality_model_validated
        assert by_type["power_play"].quality_score == 0.0
        first_types.add(slate[0].setup_type)

    # The neutral class may lead on a rotated day, but it has no permanent
    # priority merely because zero is numerically above the negative estimate.
    assert first_types == {"power_play", "vcp"}


def test_neutral_rank_salt_changes_candidate_hash_and_class_rotation():
    dates, _, _, _, _, _ = _matrices()
    symbols = pd.Index(["A1", "A2", "B1", "C1"])
    close = pd.DataFrame(99.0, index=dates, columns=symbols)
    continuity = _segments(dates, symbols)
    setups = [
        _setup(dates, symbol="A1", setup_type="flat_base", setup_id=1),
        _setup(dates, symbol="A2", setup_type="flat_base", setup_id=2),
        _setup(dates, symbol="B1", setup_type="power_play", setup_id=3),
        _setup(dates, symbol="C1", setup_type="vcp", setup_id=4),
    ]

    orders = []
    for salt_number in range(16):
        ranker = CandidateRanker(
            dates,
            symbols,
            close,
            continuity_segment=continuity,
            neutral_rank_salt=f"robustness-{salt_number}",
        )
        orders.append(
            tuple((item.setup_type, item.symbol) for item in ranker.rank(setups, 60))
        )

    assert len(set(orders)) > 1
    assert len({order[0][0] for order in orders}) > 1
    assert len(
        {
            tuple(symbol for setup_type, symbol in order if setup_type == "flat_base")
            for order in orders
        }
    ) > 1


def test_cross_class_hash_order_is_input_and_deletion_stable():
    dates, _, _, _, _, _ = _matrices()
    symbols = pd.Index(["A1", "B1", "C1", "D1"])
    close = pd.DataFrame(99.0, index=dates, columns=symbols)
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        neutral_rank_salt="deletion-stability",
    )
    setups = [
        _setup(dates, symbol="A1", setup_type="flat_base", setup_id=1),
        _setup(dates, symbol="B1", setup_type="power_play", setup_id=2),
        _setup(dates, symbol="C1", setup_type="vcp", setup_id=3),
        _setup(dates, symbol="D1", setup_type="tight_shelf", setup_id=4),
    ]

    complete = ranker.rank(setups, 60)
    reversed_input = ranker.rank(list(reversed(setups)), 60)
    without_research_only = ranker.rank(
        [setup for setup in setups if setup.setup_type != "tight_shelf"], 60
    )

    assert complete == reversed_input
    assert [
        item.setup_type
        for item in complete
        if item.setup_type != "tight_shelf"
    ] == [item.setup_type for item in without_research_only]


def test_neutral_rank_salt_must_be_a_nonempty_string():
    dates, symbols, close, _, _, _ = _matrices()
    kwargs = {
        "continuity_segment": _segments(dates, symbols),
    }
    with pytest.raises(ValueError, match="must not be empty"):
        CandidateRanker(dates, symbols, close, neutral_rank_salt="", **kwargs)
    with pytest.raises(TypeError, match="must be a string"):
        CandidateRanker(dates, symbols, close, neutral_rank_salt=7, **kwargs)


def test_trend_is_a_soft_quality_feature_and_never_a_hard_gate():
    dates, symbols, close, _, _, _ = _matrices()
    trend_pass = pd.DataFrame(1.0, index=dates, columns=symbols)
    trend_fail = pd.DataFrame(0.0, index=dates, columns=symbols)
    passed = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        context={"trend_template_pass": trend_pass},
    ).snapshot(_setup(dates), 60)
    failed = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        context={"trend_template_pass": trend_fail},
    ).snapshot(_setup(dates), 60)

    assert passed.raw_quality_score > failed.raw_quality_score
    assert passed.quality_score == failed.quality_score == 0.0


def test_readiness_is_rejected_as_a_quality_feature():
    dates, symbols, close, _, _, _ = _matrices()
    with pytest.raises(ValueError, match="non-whitelisted quality features"):
        CandidateRanker(
            dates,
            symbols,
            close,
            continuity_segment=_segments(dates, symbols),
            quality_type_weights={
                "default": {"rs": 1.0, "readiness": 1.0},
            },
        )


def test_labels_must_become_available_after_their_snapshot():
    dates, *_ = _matrices()
    with pytest.raises(ValueError, match="after its snapshot"):
        QualityCalibrationLabel("vcp", dates[20], dates[20], 50.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="after its snapshot"):
        FillCalibrationLabel("vcp", dates[20], dates[20], 50.0, True)


def test_quality_labels_require_a_finite_walk_forward_prediction():
    dates, *_ = _matrices()
    with pytest.raises(TypeError, match="walk_forward_quality_score"):
        QualityCalibrationLabel("vcp", dates[20], dates[21], 50.0, 1.0)
    with pytest.raises(ValueError, match="walk_forward_quality_score must be finite"):
        QualityCalibrationLabel(
            "vcp", dates[20], dates[21], 50.0, 1.0, float("nan")
        )
