import numpy as np
import pandas as pd

import backtest_models.minervini as minervini
from backtest_models.minervini import GOLD_CASES, find_setups, find_swings
from tests.util import make_cfg


def _series(parts: list[np.ndarray], *, quiet_from: int, quiet_volume: float = 500_000.0):
    close = np.concatenate(parts).astype(float)
    high = close * 1.004
    low = close * 0.996
    volume = np.full(len(close), 1_000_000.0)
    volume[quiet_from:] = quiet_volume
    dates = pd.bdate_range("2020-01-02", periods=len(close))
    return dates, high, low, close, volume


def _vcp_series(*, second_low: float = 90.0, outside_bar: bool = False, quiet_volume: float = 500_000.0):
    values = [
        np.linspace(50, 100, 60),
        np.linspace(98.5, 84, 8),
        np.linspace(86, 98, 8),
        np.linspace(96.5, second_low, 6),
        np.linspace(second_low + 1.5, 97, 6),
        np.linspace(96, 92.5, 5),
        np.linspace(93.0, 95.8, 10),
    ]
    dates, high, low, close, volume = _series(values, quiet_from=88, quiet_volume=quiet_volume)
    if outside_bar:
        # Ambiguous bar between the first and second contractions. It must not
        # erase the already confirmed first contraction.
        high[72] = 100.5
        low[72] = 82.0
    return dates, high, low, close, volume


def _detect(data, *, pass_start: int = 25, **overrides):
    dates, high, low, close, volume = data
    defaults = {
        "dryup_score_zero_ratio": 1.25,
        # The synthetic VCP fixtures deliberately exercise all three legs.
        # Production configuration may still choose two as its minimum.
        "contractions_min": 3,
    }
    defaults.update(overrides)
    cfg = make_cfg(**defaults)
    continuity_segment = np.ones(len(dates), dtype=int)
    continuity_break = np.zeros(len(dates), dtype=bool)
    continuity_break[0] = True
    return find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        continuity_segment,
        continuity_break,
        np.arange(pass_start, len(dates)),
        cfg,
        trading_dates=dates,
    )


def _synthetic_candidate(
    setup_type: str,
    *,
    t: int,
    pivot_idx: int,
    pivot: float,
    score: float,
    base_start: int = 5,
):
    return {
        "setup_type": setup_type,
        "identity": (pivot_idx, round(pivot, 8)),
        "pivot_idx": pivot_idx,
        "structure_end": t,
        "base_start": base_start,
        "base_days": t - base_start,
        "pivot": pivot,
        "last_low": pivot * 0.94,
        "depths": (0.06, 0.03) if setup_type == "vcp" else (0.06,),
        "dryup": 0.5,
        "prior": 0.8,
        "tightness": 0.03,
        "score": (score, 20.0, 8.0, 12.0, 10.0),
    }


def _patched_detection(
    monkeypatch,
    range_factory,
    vcp_factory,
    pass_idx,
    **config_overrides,
):
    calls: list[str] = []

    def range_candidate(setup_type, t, *args):
        calls.append(setup_type)
        return range_factory(setup_type, t)

    def vcp_candidate(t, *args):
        calls.append("vcp")
        return vcp_factory(t)

    monkeypatch.setattr(minervini, "_range_candidate", range_candidate)
    monkeypatch.setattr(minervini, "_vcp_candidate", vcp_candidate)
    dates = pd.bdate_range("2020-01-02", periods=40)
    close = np.full(len(dates), 95.0)
    continuity_segment = np.ones(len(dates), dtype=int)
    continuity_break = np.zeros(len(dates), dtype=bool)
    continuity_break[0] = True
    found = find_setups(
        "TEST",
        dates,
        close * 1.01,
        close * 0.99,
        close,
        np.full(len(dates), 1_000_000.0),
        continuity_segment,
        continuity_break,
        np.asarray(pass_idx),
        make_cfg(**config_overrides),
        trading_dates=dates,
    )
    return found, calls


def test_vcp_requires_prior_advance():
    data = _vcp_series()
    found = _detect(data, pass_start=95)
    assert any(item.setup_type == "vcp" for item in found)

    dates, high, low, close, volume = data
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[:60] = np.linspace(92, 100, 60)
    low[:60] = high[:60] * 0.996
    close[:60] = high[:60] / 1.004
    assert not any(
        item.setup_type == "vcp"
        for item in _detect((dates, high, low, close, volume), pass_start=95)
    )


def test_vcp_tolerates_one_modestly_wider_contraction():
    # 16%, then roughly 9%, then 4.5%: normal VCP.
    normal = [item for item in _detect(_vcp_series(second_low=90.0), pass_start=95) if item.setup_type == "vcp"]
    # A roughly 10.5% middle contraction is slightly noisier but still
    # substantially tighter than the first contraction.
    noisy = [item for item in _detect(_vcp_series(second_low=88.0), pass_start=95) if item.setup_type == "vcp"]
    assert normal and noisy


def test_outside_bar_does_not_clear_confirmed_swing_history():
    dates, high, low, close, volume = _vcp_series(outside_bar=True)
    swings = find_swings(high, low, 3)
    assert len(swings) >= 4
    # The ambiguous observation can prevent a nearby extremum from being
    # confirmed, but it must not erase the contraction confirmed before it.
    assert swings[0][0] == 59 and swings[1][0] == 67
    assert swings[-2][0] == 87 and swings[-1][0] == 92


def test_long_vcp_base_beyond_old_75_day_ceiling_is_detected():
    dates, high, low, close, volume = _vcp_series()
    # Extend the middle recovery/contraction with 35 calm bars while retaining
    # the causal extrema; total base age is now over 75 sessions.
    insert_at = 76
    filler_close = np.linspace(92, 95, 50)
    close = np.insert(close, insert_at, filler_close)
    high = np.insert(high, insert_at, filler_close * 1.004)
    low = np.insert(low, insert_at, filler_close * 0.996)
    volume = np.insert(volume, insert_at, np.full(50, 850_000.0))
    dates = pd.bdate_range(dates[0], periods=len(close))
    found = _detect(
        (dates, high, low, close, volume),
        pass_start=145,
        base_max_days=130,
    )
    assert any(item.setup_type == "vcp" and item.base_days > 75 for item in found)


def test_stronger_dryup_is_monotonically_better():
    weak = [
        item
        for item in _detect(_vcp_series(quiet_volume=700_000.0), pass_start=95)
        if item.setup_type == "vcp"
    ][0]
    strong = [
        item
        for item in _detect(_vcp_series(quiet_volume=300_000.0), pass_start=95)
        if item.setup_type == "vcp"
    ][0]
    assert strong.dryup_ratio < weak.dryup_ratio
    assert strong.volume_dryup_score > weak.volume_dryup_score
    assert strong.setup_score > weak.setup_score


def test_final_area_must_be_tight_and_below_pivot():
    dates, high, low, close, volume = _vcp_series()
    high = high.copy()
    low = low.copy()
    # This bar is already known when the final swing-low becomes confirmed.
    high[93] *= 1.08
    low[93] *= 0.92
    assert not any(item.setup_type == "vcp" for item in _detect((dates, high, low, close, volume)))


def test_flat_base_setup_class():
    data = _series(
        [
            np.linspace(50, 72, 55),
            70 + np.sin(np.linspace(0, 6 * np.pi, 42)) * 2.0,
            np.linspace(70.5, 71.5, 8),
        ],
        quiet_from=90,
    )
    # Before this point the same chart is causally a shorter tight shelf.  Start
    # on the first day where only the mature flat-base classification applies.
    found = _detect(data, pass_start=73)
    assert any(item.setup_type == "flat_base" for item in found)


def test_power_play_setup_class():
    data = _series(
        [
            np.full(20, 30.0),
            np.linspace(30, 66, 20),
            62 + np.sin(np.linspace(0, 3 * np.pi, 18)) * 2.0,
            np.linspace(62.5, 64.5, 7),
        ],
        quiet_from=55,
    )
    dates, high, low, close, volume = data
    volume[24:40] = 2_000_000.0
    data = dates, high, low, close, volume
    found = _detect(data)
    assert any(item.setup_type == "power_play" for item in found)


def test_tight_shelf_setup_class():
    data = _series(
        [
            np.linspace(45, 70, 60),
            68.5 + np.sin(np.linspace(0, 2 * np.pi, 11)) * 1.0,
            np.linspace(68.8, 69.0, 5),
        ],
        quiet_from=66,
    )
    found = _detect(data)
    assert any(item.setup_type == "tight_shelf" for item in found)


def test_all_setup_classes_are_evaluated_and_best_quality_label_wins(monkeypatch):
    scores = {
        "power_play": 61.0,
        "vcp": 84.0,
        "tight_shelf": 73.0,
        "flat_base": 55.0,
    }

    def candidate(setup_type, t):
        return _synthetic_candidate(
            setup_type,
            t=t,
            pivot_idx=10,
            pivot=100.0,
            score=scores[setup_type],
        )

    found, calls = _patched_detection(
        monkeypatch,
        candidate,
        lambda t: candidate("vcp", t),
        [30],
    )

    assert calls == ["power_play", "vcp", "tight_shelf", "flat_base"]
    assert [setup.setup_type for setup in found] == ["vcp"]


def test_suppressed_first_candidate_does_not_hide_distinct_valid_class(monkeypatch):
    def range_candidate(setup_type, t):
        if setup_type != "power_play":
            return None
        return _synthetic_candidate(
            "power_play", t=t, pivot_idx=10, pivot=100.0, score=95.0
        )

    def vcp_candidate(t):
        if t != 31:
            return None
        return _synthetic_candidate(
            "vcp", t=t, pivot_idx=24, pivot=120.0, score=75.0, base_start=20
        )

    found, _ = _patched_detection(
        monkeypatch,
        range_candidate,
        vcp_candidate,
        [30, 31],
    )

    assert [(setup.detect_date, setup.setup_type) for setup in found] == [
        (pd.Timestamp("2020-02-13").date(), "power_play"),
        (pd.Timestamp("2020-02-14").date(), "vcp"),
    ]


def test_recent_overlapping_structure_is_not_reemitted_under_new_label(monkeypatch):
    def range_candidate(setup_type, t):
        if setup_type == "power_play" and t == 30:
            return _synthetic_candidate(
                "power_play", t=t, pivot_idx=10, pivot=100.0, score=80.0
            )
        return None

    def vcp_candidate(t):
        if t == 31:
            return _synthetic_candidate(
                "vcp", t=t, pivot_idx=11, pivot=100.5, score=95.0, base_start=8
            )
        return None

    found, _ = _patched_detection(
        monkeypatch,
        range_candidate,
        vcp_candidate,
        [30, 31],
    )

    assert [setup.setup_type for setup in found] == ["power_play"]


def test_nested_nearby_pivots_collapse_to_one_deterministic_classification():
    candidates = [
        _synthetic_candidate(
            "power_play", t=30, pivot_idx=10, pivot=100.0, score=70.0
        ),
        _synthetic_candidate(
            "vcp", t=30, pivot_idx=11, pivot=101.0, score=82.0, base_start=8
        ),
        _synthetic_candidate(
            "tight_shelf", t=30, pivot_idx=12, pivot=99.5, score=74.0, base_start=15
        ),
    ]

    collapsed = minervini._collapse_overlapping_classifications(candidates)

    assert len(collapsed) == 1
    assert collapsed[0]["setup_type"] == "vcp"


def test_distinct_same_day_structures_are_both_emitted(monkeypatch):
    def range_candidate(setup_type, t):
        if setup_type != "power_play":
            return None
        return _synthetic_candidate(
            "power_play", t=t, pivot_idx=10, pivot=100.0, score=90.0
        )

    found, _ = _patched_detection(
        monkeypatch,
        range_candidate,
        lambda t: _synthetic_candidate(
            "vcp", t=t, pivot_idx=24, pivot=120.0, score=80.0, base_start=20
        ),
        [30],
    )

    assert [(setup.setup_type, setup.pivot) for setup in found] == [
        ("power_play", 100.0),
        ("vcp", 120.0),
    ]


def test_bridge_candidate_does_not_merge_nonoverlapping_endpoint_pivots():
    candidates = [
        _synthetic_candidate(
            "vcp", t=30, pivot_idx=10, pivot=100.0, score=90.0
        ),
        _synthetic_candidate(
            "power_play", t=30, pivot_idx=11, pivot=102.0, score=80.0
        ),
        _synthetic_candidate(
            "tight_shelf", t=30, pivot_idx=12, pivot=104.0, score=70.0
        ),
    ]

    collapsed = minervini._collapse_overlapping_classifications(candidates)

    assert len(collapsed) == 2
    assert {candidate["pivot"] for candidate in collapsed} == {100.0, 104.0}


def test_structure_expiry_alone_does_not_reemit_it(monkeypatch):
    def range_candidate(setup_type, t):
        if setup_type != "power_play":
            return None
        return _synthetic_candidate(
            "power_play", t=t, pivot_idx=10, pivot=100.0, score=80.0
        )

    found, _ = _patched_detection(
        monkeypatch,
        range_candidate,
        lambda _t: None,
        [30, 32],
        setup_valid_days=2,
    )

    assert [setup.detect_date for setup in found] == [
        pd.Timestamp("2020-02-13").date(),
    ]


def test_equal_quality_classification_uses_specificity_not_input_order():
    power_play = _synthetic_candidate(
        "power_play", t=30, pivot_idx=10, pivot=100.0, score=80.0
    )
    vcp = _synthetic_candidate(
        "vcp", t=30, pivot_idx=10, pivot=100.0, score=80.0
    )

    forward = minervini._collapse_overlapping_classifications([power_play, vcp])
    reverse = minervini._collapse_overlapping_classifications([vcp, power_play])

    assert forward[0]["setup_type"] == "vcp"
    assert reverse[0]["setup_type"] == "vcp"


def test_structure_overlap_is_symmetric_at_pivot_tolerance_boundary():
    low = _synthetic_candidate(
        "vcp", t=30, pivot_idx=10, pivot=100.0, score=80.0
    )
    high = _synthetic_candidate(
        "tight_shelf", t=30, pivot_idx=11, pivot=103.0, score=79.0
    )

    assert minervini._same_structure(low, high)
    assert minervini._same_structure(high, low)


def test_range_setup_cannot_move_pivot_to_detection_day_breakout():
    dates, high, low, close, volume = _series(
        [
            np.linspace(45, 70, 60),
            68.5 + np.sin(np.linspace(0, 2 * np.pi, 11)),
            np.linspace(68.8, 69.0, 5),
        ],
        quiet_from=66,
    )
    established_pivot = float(np.max(high[:-5]))
    high[-1] = established_pivot * 1.03
    close[-1] = established_pivot * 0.99

    found = _detect(
        (dates, high, low, close, volume), pass_start=len(dates) - 1
    )

    assert not any(
        item.setup_type in {"flat_base", "power_play", "tight_shelf"}
        for item in found
    )


def test_missing_volume_does_not_erase_valid_price_structure():
    dates, high, low, close, volume = _vcp_series()
    volume = volume.copy()
    volume[72] = np.nan

    found = _detect(
        (dates, high, low, close, volume), pass_start=len(dates) - 1
    )

    assert any(item.setup_type == "vcp" for item in found)


def test_zero_recent_volume_is_unknown_not_a_perfect_dryup_score():
    dates, high, low, close, volume = _vcp_series()
    volume = volume.copy()
    volume[-5:] = 0.0

    found = _detect(
        (dates, high, low, close, volume), pass_start=len(dates) - 1
    )
    setup = next(item for item in found if item.setup_type == "vcp")

    assert np.isnan(setup.dryup_ratio)
    assert setup.volume_dryup_score == 0.0


def test_setup_detection_is_causal_under_future_mutation():
    data = _vcp_series()
    original = _detect(data, pass_start=95)
    first = next(item for item in original if item.setup_type == "vcp")
    cutoff = pd.Timestamp(first.detect_date)
    dates, high, low, close, volume = data
    future = dates > cutoff
    high2, low2, close2, volume2 = (x.copy() for x in (high, low, close, volume))
    high2[future] *= 1.7
    low2[future] *= 0.6
    close2[future] *= 1.2
    volume2[future] *= 4
    mutated = _detect((dates, high2, low2, close2, volume2), pass_start=95)
    first_mutated = next(item for item in mutated if item.setup_type == "vcp")
    assert first_mutated == first


def test_gold_catalog_is_balanced_metadata_not_asserted_signals():
    positives = [case for case in GOLD_CASES if case.role == "positive"]
    negatives = [case for case in GOLD_CASES if case.role == "negative"]
    assert {case.symbol for case in positives} >= {"NVDA", "SMCI", "CELH", "ELF", "CROX", "MOD", "AXON"}
    assert len(negatives) >= 5
    assert all(case.reference_date and case.rationale and case.source for case in GOLD_CASES)
    assert len({(case.symbol, case.role, case.reference_date) for case in GOLD_CASES}) == len(GOLD_CASES)


def test_setup_expiry_uses_global_sessions():
    dates, high, low, close, volume = _vcp_series()
    global_dates = pd.bdate_range(dates[0], periods=len(dates) + 40)
    continuity_segment = np.ones(len(dates), dtype=int)
    continuity_break = np.zeros(len(dates), dtype=bool)
    continuity_break[0] = True
    found = find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        continuity_segment,
        continuity_break,
        np.arange(95, len(dates)),
        make_cfg(setup_valid_days=10, dryup_score_zero_ratio=1.25),
        trading_dates=global_dates,
    )
    setup = next(item for item in found if item.setup_type == "vcp")
    detect_idx = global_dates.get_loc(pd.Timestamp(setup.detect_date))
    assert setup.valid_until == global_dates[detect_idx + 10].date()


def test_setup_dataclass_has_explicit_type_and_no_legacy_vcp_score():
    setup = _detect(_vcp_series())[0]
    assert setup.setup_type in {"vcp", "flat_base", "power_play", "tight_shelf"}
    assert setup.setup_score > 0
    assert setup.price_continuity_segment == 1
    assert not hasattr(setup, "vcp_score")


def test_setup_geometry_cannot_cross_price_continuity_segment():
    dates, high, low, close, volume = _vcp_series()
    continuity_segment = np.ones(len(dates), dtype=int)
    continuity_segment[88:] = 2
    continuity_break = np.zeros(len(dates), dtype=bool)
    continuity_break[[0, 88]] = True

    found = find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        continuity_segment,
        continuity_break,
        np.arange(95, len(dates)),
        make_cfg(contractions_min=3, dryup_score_zero_ratio=1.25),
        trading_dates=dates,
    )

    assert found == []
