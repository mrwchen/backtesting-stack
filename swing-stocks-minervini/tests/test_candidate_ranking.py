from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.candidate_ranking import CandidateRanker


def _matrices(periods: int = 70):
    dates = pd.bdate_range("2024-01-02", periods=periods)
    symbols = pd.Index(["AAA", "BBB"])
    close = pd.DataFrame(
        {"AAA": np.linspace(90.0, 99.0, periods), "BBB": np.linspace(88.0, 99.0, periods)},
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


def _ranker(dates, symbols, close, volume, rs, fundamentals):
    return CandidateRanker(
        dates,
        symbols,
        close,
        volume=volume,
        rs_rating=rs,
        context={"fundamental_score": fundamentals},
    )


def test_snapshot_is_strictly_pre_session_causal():
    dates, symbols, close, volume, rs, fundamentals = _matrices()
    session_idx = 60
    baseline = _ranker(dates, symbols, close, volume, rs, fundamentals).snapshot(
        _setup(dates), session_idx
    )

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


def test_previous_close_change_does_change_snapshot():
    dates, symbols, close, volume, rs, fundamentals = _matrices()
    session_idx = 60
    baseline = _ranker(dates, symbols, close, volume, rs, fundamentals).snapshot(
        _setup(dates), session_idx
    )
    changed_close = close.copy()
    changed_close.iloc[session_idx - 1, 0] = 90.0
    changed = _ranker(
        dates, symbols, changed_close, volume, rs, fundamentals
    ).snapshot(_setup(dates), session_idx)

    assert changed.distance_to_pivot_pct != baseline.distance_to_pivot_pct
    assert changed.ranking_score < baseline.ranking_score


def test_missing_features_are_neutral_and_report_lower_coverage():
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
    missing = CandidateRanker(dates, symbols, close).snapshot(missing_setup, 60)

    neutral_setup = _setup(
        dates,
        rs_rating=50.0,
        dryup_ratio=0.625,
        fundamental_score=3.0,
        structure_quality_score=12.5,
        tightness_score=10.0,
        prior_advance_score=12.5,
    )
    neutral = CandidateRanker(dates, symbols, close).snapshot(neutral_setup, 60)

    assert np.isfinite(missing.ranking_score)
    assert np.isfinite(missing.context_score)
    assert missing.feature_coverage < neutral.feature_coverage
    assert missing.ranking_score == pytest.approx(neutral.ranking_score)
    assert missing.dryup_source == "missing"


def test_incomplete_daily_volume_does_not_fall_back_to_stale_setup_dryup():
    dates, symbols, close, volume, _, _ = _matrices()
    volume.iloc[56:60, 0] = np.nan
    setup = _setup(dates, dryup_ratio=0.2)

    snapshot = CandidateRanker(
        dates, symbols, close, volume=volume
    ).snapshot(setup, 60)

    assert snapshot.dynamic_dryup_ratio is None
    assert snapshot.dryup_source == "missing"
    assert snapshot.dryup_coverage == 0.0


def test_partial_dynamic_dryup_window_has_partial_feature_coverage():
    dates, symbols, close, volume, _, _ = _matrices()
    full = CandidateRanker(dates, symbols, close, volume=volume).snapshot(
        _setup(dates), 60
    )
    partial_volume = volume.copy()
    # The configured baseline has 50 sessions. Keep only 20 valid observations
    # while retaining a complete five-session recent window.
    partial_volume.iloc[5:35, 0] = np.nan
    partial = CandidateRanker(
        dates, symbols, close, volume=partial_volume
    ).snapshot(_setup(dates), 60)

    assert full.dryup_coverage == 1.0
    assert full.dryup_source == "dynamic"
    assert partial.dynamic_dryup_ratio == pytest.approx(full.dynamic_dryup_ratio)
    assert partial.dryup_coverage == pytest.approx(0.4)
    assert partial.dryup_source == "dynamic_partial"
    assert partial.feature_coverage < full.feature_coverage


def test_partial_fundamental_coverage_blends_unknown_criteria_neutrally():
    dates, symbols, close, _, _, _ = _matrices()
    full_score = pd.DataFrame(6.0, index=dates, columns=symbols)
    partial_score = pd.DataFrame(3.0, index=dates, columns=symbols)
    full_coverage = pd.DataFrame(1.0, index=dates, columns=symbols)
    half_coverage = pd.DataFrame(0.5, index=dates, columns=symbols)
    setup = _setup(dates, fundamental_score=np.nan)

    full = CandidateRanker(
        dates,
        symbols,
        close,
        context={
            "fundamental_score": full_score,
            "fundamental_coverage": full_coverage,
        },
    ).snapshot(setup, 60)
    partial = CandidateRanker(
        dates,
        symbols,
        close,
        context={
            "fundamental_score": partial_score,
            "fundamental_coverage": half_coverage,
        },
    ).snapshot(setup, 60)
    missing = CandidateRanker(dates, symbols, close).snapshot(setup, 60)

    assert partial.fundamental_score == 3.0
    assert partial.fundamental_coverage == 0.5
    assert full.ranking_score > partial.ranking_score > missing.ranking_score
    assert full.context_score > partial.context_score > missing.context_score
    assert full.feature_coverage > partial.feature_coverage > missing.feature_coverage


def test_setup_type_profiles_make_comparable_continuous_scores():
    dates, symbols, close, _, _, _ = _matrices()
    # Strong prior advance and weak dry-up is intentionally more relevant to a
    # power play than a VCP; there is no hard setup-type priority or bucket.
    common = {
        "rs_rating": 80.0,
        "dryup_ratio": 1.20,
        "structure_quality_score": 12.5,
        "tightness_score": 10.0,
        "prior_advance_score": 25.0,
        "fundamental_score": 3.0,
    }
    ranker = CandidateRanker(dates, symbols, close)
    power = ranker.snapshot(_setup(dates, setup_type="power_play", **common), 60)
    vcp = ranker.snapshot(_setup(dates, setup_type="vcp", **common), 60)

    assert 0.0 <= power.ranking_score <= 100.0
    assert 0.0 <= vcp.ranking_score <= 100.0
    assert power.ranking_score > vcp.ranking_score


def test_rank_is_deterministic_across_input_order():
    dates, symbols, close, volume, rs, fundamentals = _matrices()
    ranker = _ranker(dates, symbols, close, volume, rs, fundamentals)
    aaa = _setup(dates, symbol="AAA", setup_id=2)
    bbb = _setup(dates, symbol="BBB", setup_id=1)

    forward = ranker.rank([bbb, aaa], 60)
    reverse = ranker.rank([aaa, bbb], 60)

    assert forward == reverse
    assert [item.symbol for item in forward] == ["AAA", "BBB"]


def test_non_whitelisted_group_or_raw_count_feature_cannot_enter_score():
    dates, symbols, close, _, _, _ = _matrices()
    with pytest.raises(ValueError, match="non-whitelisted"):
        CandidateRanker(
            dates,
            symbols,
            close,
            type_weights={
                "default": {
                    "readiness": 1.0,
                    "institutional_manager_count": 1.0,
                }
            },
        )
