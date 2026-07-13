import numpy as np
import pandas as pd

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
    return find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        np.arange(pass_start, len(dates)),
        cfg,
        trading_dates=dates,
    )


def test_vcp_requires_prior_advance():
    data = _vcp_series()
    found = _detect(data)
    assert any(item.setup_type == "vcp" for item in found)

    dates, high, low, close, volume = data
    high = high.copy()
    low = low.copy()
    close = close.copy()
    high[:60] = np.linspace(92, 100, 60)
    low[:60] = high[:60] * 0.996
    close[:60] = high[:60] / 1.004
    assert not any(item.setup_type == "vcp" for item in _detect((dates, high, low, close, volume)))


def test_vcp_tolerates_one_modestly_wider_contraction():
    # 16%, then roughly 9%, then 4.5%: normal VCP.
    normal = [item for item in _detect(_vcp_series(second_low=90.0)) if item.setup_type == "vcp"]
    # A roughly 10.5% middle contraction is slightly noisier but still
    # substantially tighter than the first contraction.
    noisy = [item for item in _detect(_vcp_series(second_low=88.0)) if item.setup_type == "vcp"]
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
        pass_start=140,
        base_max_days=130,
    )
    assert any(item.setup_type == "vcp" and item.base_days > 75 for item in found)


def test_stronger_dryup_is_monotonically_better():
    weak = [item for item in _detect(_vcp_series(quiet_volume=700_000.0)) if item.setup_type == "vcp"][0]
    strong = [item for item in _detect(_vcp_series(quiet_volume=300_000.0)) if item.setup_type == "vcp"][0]
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
    found = _detect(data)
    assert any(item.setup_type == "flat_base" for item in found)


def test_power_play_setup_class():
    data = _series(
        [
            np.linspace(30, 35, 35),
            np.linspace(35, 66, 20),
            62 + np.sin(np.linspace(0, 3 * np.pi, 18)) * 2.0,
            np.linspace(62.5, 64.5, 7),
        ],
        quiet_from=66,
    )
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
    original = _detect(data)
    first = next(item for item in original if item.setup_type == "vcp")
    cutoff = pd.Timestamp(first.detect_date)
    dates, high, low, close, volume = data
    future = dates > cutoff
    high2, low2, close2, volume2 = (x.copy() for x in (high, low, close, volume))
    high2[future] *= 1.7
    low2[future] *= 0.6
    close2[future] *= 1.2
    volume2[future] *= 4
    mutated = _detect((dates, high2, low2, close2, volume2))
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
    found = find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        np.arange(25, len(dates)),
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
    assert not hasattr(setup, "vcp_score")
