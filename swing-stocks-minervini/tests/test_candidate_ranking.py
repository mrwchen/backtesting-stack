from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.candidate_ranking import (
    CandidateRanker,
    FillCalibrationLabel,
    QualityCalibrationLabel,
)


def _matrices(periods: int = 80):
    dates = pd.bdate_range("2024-01-02", periods=periods)
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
        "vcp", dates[50], dates[65], probe.raw_quality_score, 4.0
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
    assert after.quality_score > before.quality_score
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
            "vcp", dates[20], dates[21], vcp_probe.raw_quality_score, 3.0
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

    assert vcp.quality_score > 0.0
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
        "vcp", dates[20], dates[21], probe.raw_quality_score, 6.0
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

    assert 0.0 < snapshot.quality_score < 1.0
    assert 0.5 < snapshot.fill_probability < 0.55
    assert snapshot.slate_priority == pytest.approx(
        snapshot.quality_score * snapshot.fill_probability
    )


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
    assert next_session.quality_score > before.quality_score
    assert next_session.fill_probability > before.fill_probability


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
    assert [item.symbol for item in forward] == ["AAA", "BBB"]
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
    day_61 = ranker.rank(setups, 61)

    assert {item.setup_type for item in day_60[:3]} == {
        "flat_base",
        "power_play",
        "vcp",
    }
    assert day_60[0].setup_type != day_61[0].setup_type
    assert all(item.quality_rank == 1 for item in day_60)


def test_slate_order_and_pure_quality_rank_are_deliberately_distinct():
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
            QualityCalibrationLabel(
                "vcp", dates[20], dates[21], vcp_probe.raw_quality_score, 2.0
            ),
            QualityCalibrationLabel(
                "power_play",
                dates[20],
                dates[21],
                power_probe.raw_quality_score,
                1.0,
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

    assert [item.setup_type for item in slate] == ["power_play", "vcp"]
    assert [item.quality_rank for item in slate] == [2, 1]
    assert slate[1].quality_score > slate[0].quality_score
    assert slate[0].slate_priority > slate[1].slate_priority


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
        QualityCalibrationLabel("vcp", dates[20], dates[20], 50.0, 1.0)
    with pytest.raises(ValueError, match="after its snapshot"):
        FillCalibrationLabel("vcp", dates[20], dates[20], 50.0, True)
