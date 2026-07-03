import numpy as np
import pandas as pd

from src.vcp import find_setups, find_swings
from tests.util import make_cfg


def _base_series():
    """Uptrend into a 3-contraction VCP base: 100 -> 85 (15%), 98 -> 90 (8%),
    97 -> 93 (4%), then a quiet drift just below the pivot."""
    close = np.concatenate(
        [
            np.linspace(50, 100, 60),   # 0..59  uptrend, swing high 100 @ 59
            np.linspace(98.5, 85, 8),   # 60..67 contraction 1 low @ 67
            np.linspace(87, 98, 8),     # 68..75 recovery, swing high 98 @ 75
            np.linspace(96.5, 90, 6),   # 76..81 contraction 2 low @ 81
            np.linspace(91.5, 97, 6),   # 82..87 recovery, swing high 97 @ 87
            np.linspace(96, 93, 5),     # 88..92 contraction 3 low @ 92
            np.linspace(93.5, 95.2, 8), # 93..100 tight drift below pivot
        ]
    )
    high = close * 1.005
    low = close * 0.995
    volume = np.where(np.arange(len(close)) >= 88, 250_000.0, 1_000_000.0)
    dates = pd.bdate_range("2023-01-02", periods=len(close))
    return dates, high, low, close, volume


def test_swings_alternate():
    dates, high, low, close, volume = _base_series()
    swings = find_swings(high, low, k=3)
    kinds = [s[2] for s in swings]
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))


def test_detects_three_contraction_base():
    dates, high, low, close, volume = _base_series()
    cfg = make_cfg()
    pass_idx = np.arange(93, 101)

    setups = find_setups("TEST", dates, high, low, close, volume, pass_idx, cfg)

    assert len(setups) == 1  # same base emitted once, not once per day
    setup = setups[0]
    assert setup.n_contractions == 3
    assert abs(setup.pivot - 97 * 1.005) < 0.2
    depths = setup.contraction_depths
    assert depths[0] > depths[1] > depths[2]
    assert setup.dryup_ratio < cfg.dryup_ratio_max
    assert setup.last_low < setup.close < setup.pivot
    # final low @ 92 confirms at bar 95 (k=3): no detection before that
    assert dates.searchsorted(pd.Timestamp(setup.detect_date)) >= 95


def test_no_detection_without_volume_dryup():
    dates, high, low, close, volume = _base_series()
    volume[:] = 1_000_000.0  # no dry-up
    setups = find_setups(
        "TEST", dates, high, low, close, volume, np.arange(93, 101), make_cfg()
    )
    assert setups == []


def test_no_detection_when_contractions_widen():
    dates, high, low, close, volume = _base_series()
    # invert the base: widening contractions (4% -> 8% -> 15%)
    inverted = np.concatenate(
        [
            np.linspace(50, 100, 60),
            np.linspace(99.5, 96, 8),    # 4% first
            np.linspace(96.5, 98, 8),
            np.linspace(97, 90, 6),      # 8%
            np.linspace(90.5, 97, 6),
            np.linspace(96, 85, 5),      # 15% last -> widening
            np.linspace(85.5, 95.2, 8),
        ]
    )
    setups = find_setups(
        "TEST", dates, inverted * 1.005, inverted * 0.995, inverted, volume,
        np.arange(93, 101), make_cfg(),
    )
    assert setups == []
