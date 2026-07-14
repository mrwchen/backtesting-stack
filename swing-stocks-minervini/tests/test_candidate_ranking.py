from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.candidate_ranking import (
    CandidateRanker,
    FillCalibrationLabel,
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


def _quality_labels(
    setup_type,
    raw_score,
    group_outcomes=(-1.5, -0.5, 0.5, 1.5),
    *,
    start_quarter="2020Q1",
    quarters=8,
):
    """Return dense causal historical outcomes for calibration tests."""
    labels = []
    raw_scores = np.linspace(
        max(0.0, float(raw_score) - 30.0),
        min(100.0, float(raw_score) + 30.0),
        4,
    )
    scores_and_outcomes = tuple(
        (score, outcome, raw_scores[tier])
        for tier, (score, outcome) in enumerate(
            zip((-1.5, -0.5, 0.5, 1.5), group_outcomes)
        )
        for _ in range(2)
    )
    observations_per_quarter = 5
    for quarter in pd.period_range(start_quarter, periods=quarters, freq="Q"):
        available_days = pd.bdate_range(
            quarter.start_time + pd.Timedelta(days=14),
            quarter.end_time - pd.Timedelta(days=7),
        ).normalize()
        completion_days = available_days[
            np.linspace(
                0,
                len(available_days) - 1,
                num=observations_per_quarter,
                dtype=int,
            )
        ]
        for completion_date in completion_days:
            information_date = completion_date - pd.offsets.BDay(10)
            for oof_score, outcome, label_raw_score in scores_and_outcomes:
                labels.append(
                    QualityCalibrationLabel(
                        setup_type=setup_type,
                        information_date=information_date,
                        available_date=completion_date,
                        raw_quality_score=label_raw_score,
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
    assert after.quality_score == after.walk_forward_quality_score
    assert before.quality_score == 0.0
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
    assert vcp.quality_score == vcp.walk_forward_quality_score
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
    assert snapshot.quality_score == snapshot.walk_forward_quality_score
    assert snapshot.slate_priority == snapshot.quality_score
    assert snapshot.quality_calibration_count == 1
    assert 0.5 < snapshot.fill_probability < 0.55




def test_uncalibrated_quality_and_fill_do_not_hide_in_neutral_tie_breaks():
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
        assert all(
            item.quality_score == item.slate_priority == 0.0 for item in forward
        )
        first_symbols.add(forward[0].symbol)

    # The stable session lottery is reproducible but creates no permanent
    # lexical winner inside the uncalibrated class.
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
    assert next_session.quality_score == next_session.walk_forward_quality_score
    assert next_session.quality_score > before.quality_score == 0.0
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
    assert [item.quality_rank for item in forward] == [1, 2]
    assert all(np.isfinite(item.quality_score) for item in forward)
    assert all(np.isfinite(item.slate_priority) for item in forward)


def test_neutral_mode_is_one_global_stable_salted_order():
    dates, symbols, close, _, _, _ = _matrices()
    symbols = pd.Index(["A1", "A2", "B1", "B2", "C1", "C2"])
    close = pd.DataFrame(99.0, index=dates, columns=symbols)
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
        ranking_mode="neutral",
    )
    setups = [
        _setup(dates, symbol="A1", setup_type="flat_base", setup_id=1),
        _setup(dates, symbol="A2", setup_type="flat_base", setup_id=2),
        _setup(dates, symbol="B1", setup_type="power_play", setup_id=3),
        _setup(dates, symbol="B2", setup_type="power_play", setup_id=4),
        _setup(dates, symbol="C1", setup_type="vcp", setup_id=5),
        _setup(dates, symbol="C2", setup_type="vcp", setup_id=6),
    ]

    forward = ranker.rank(setups, 60)
    reverse = ranker.rank(list(reversed(setups)), 60)

    assert forward == reverse
    assert len(forward) == len(setups)
    assert all(item.quality_score == item.slate_priority == 0.0 for item in forward)
    assert [item.quality_rank for item in forward] == list(
        range(1, len(setups) + 1)
    )


def test_global_expected_r_order_ignores_fill_probability_across_classes():
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
            *_quality_labels(
                "vcp",
                vcp_probe.raw_quality_score,
                group_outcomes=(0.5, 1.5, 2.5, 3.5),
                start_quarter="2022Q1",
            ),
            *_quality_labels(
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
    assert by_type["vcp"].slate_priority == by_type["vcp"].quality_score
    assert by_type["power_play"].slate_priority == by_type["power_play"].quality_score
    assert [item.setup_type for item in slate] == ["vcp", "power_play"]
    assert [item.quality_rank for item in slate] == [1, 2]

    # Reversing the common net-R scale reverses the global quality order.
    reversed_ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_prior_strength=0.1,
        fill_prior_strength=0.1,
        quality_labels=[
            *_quality_labels(
                "vcp",
                vcp_probe.raw_quality_score,
                group_outcomes=(-0.5, 0.0, 0.5, 1.0),
                start_quarter="2022Q1",
            ),
            *_quality_labels(
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
        "power_play",
        "vcp",
    ]


def test_relative_quality_orders_and_ranks_globally():
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
        quality_labels=_quality_labels(
            "vcp",
            50.0,
            group_outcomes=(-1.5, -0.5, 0.5, 1.5),
            start_quarter="2022Q1",
        ),
    )

    slate = ranker.rank([low, high], 60)

    assert [item.symbol for item in slate] == ["AAA", "BBB"]
    assert [item.quality_rank for item in slate] == [1, 2]
    assert slate[0].quality_score > slate[1].quality_score


def test_relative_quality_orders_negative_expected_r_below_neutral_prior():
    dates, symbols, close, _, _, _ = _matrices()
    continuity = _segments(dates, symbols)
    vcp = _setup(
        dates,
        symbol="AAA",
        setup_type="vcp",
        setup_id=1,
        rs_rating=1.0,
        dryup_ratio=1.25,
        fundamental_score=0.0,
        structure_quality_score=0.0,
        tightness_score=0.0,
        prior_advance_score=0.0,
    )
    uncalibrated = _setup(
        dates, symbol="BBB", setup_type="power_play", setup_id=2
    )
    labels = _quality_labels(
        "vcp",
        50.0,
        group_outcomes=(-1.5, -0.5, 0.5, 1.5),
        start_quarter="2022Q1",
    )
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=continuity,
        quality_labels=labels,
    )

    for session_idx in range(60, 64):
        slate = ranker.rank([uncalibrated, vcp], session_idx)
        by_type = {item.setup_type: item for item in slate}
        assert by_type["vcp"].quality_score < 0.0
        assert by_type["power_play"].quality_score == 0.0
        assert [item.setup_type for item in slate] == ["power_play", "vcp"]
        assert [item.quality_rank for item in slate] == [1, 2]


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


def test_ranker_uses_the_v9_default_salt():
    dates, symbols, close, _, _, _ = _matrices()
    ranker = CandidateRanker(
        dates,
        symbols,
        close,
        continuity_segment=_segments(dates, symbols),
    )

    assert ranker.neutral_rank_salt == "v9-relative-quality-shadow"


def test_ranking_mode_controls_only_expected_r_exposure():
    dates, symbols, close, _, _, _ = _matrices()
    labels = _quality_labels(
        "vcp",
        50.0,
        group_outcomes=(-1.5, -1.0, -0.5, -0.1),
        start_quarter="2022Q1",
    )
    kwargs = {
        "continuity_segment": _segments(dates, symbols),
        "quality_labels": labels,
        "quality_prior_strength": 0.1,
    }

    neutral = CandidateRanker(
        dates, symbols, close, ranking_mode="neutral", **kwargs
    ).snapshot(_setup(dates), 60)
    relative_quality = CandidateRanker(
        dates, symbols, close, ranking_mode="relative_quality", **kwargs
    ).snapshot(_setup(dates), 60)

    assert neutral.walk_forward_quality_score == pytest.approx(
        relative_quality.walk_forward_quality_score
    )
    assert neutral.quality_score == neutral.slate_priority == 0.0
    assert relative_quality.quality_score == pytest.approx(
        relative_quality.walk_forward_quality_score
    )
    assert relative_quality.slate_priority == pytest.approx(
        relative_quality.quality_score
    )
    assert relative_quality.quality_score < 0.0


@pytest.mark.parametrize(
    "mode", ["", "quality", "quality_only", "validated", "VALIDATED", 7]
)
def test_ranking_mode_must_be_neutral_or_relative_quality(mode):
    dates, symbols, close, _, _, _ = _matrices()
    kwargs = {"continuity_segment": _segments(dates, symbols)}

    expected = TypeError if not isinstance(mode, str) else ValueError
    with pytest.raises(expected, match="ranking_mode"):
        CandidateRanker(dates, symbols, close, ranking_mode=mode, **kwargs)


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
