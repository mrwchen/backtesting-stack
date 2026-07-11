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
    volume = np.where(np.arange(len(close)) >= 88, 600_000.0, 1_000_000.0)
    dates = pd.bdate_range("2023-01-02", periods=len(close))
    return dates, high, low, close, volume


def test_swings_alternate():
    dates, high, low, close, volume = _base_series()
    swings = find_swings(high, low, k=3)
    kinds = [s[2] for s in swings]
    assert all(kinds[i] != kinds[i + 1] for i in range(len(kinds) - 1))


def test_swing_plateau_uses_first_bar_deterministically():
    high = np.array([10.0, 12.0, 12.0, 11.0, 10.0])
    low = np.array([9.0, 10.0, 10.0, 9.5, 9.0])

    swings = find_swings(high, low, k=1)

    assert (1, 12.0, "H") in swings
    assert not any(index == 2 and kind == "H" for index, _, kind in swings)


def test_outside_bar_is_a_barrier_not_an_intrabar_swing_pair():
    # Bar 3 is simultaneously the local highest high and lowest low.  Its
    # intraday ordering is unknowable and therefore clears the earlier chain.
    high = np.array([10.0, 12.0, 11.0, 14.0, 11.0, 10.0])
    low = np.array([9.0, 10.0, 8.0, 7.0, 8.0, 9.0])

    assert find_swings(high, low, k=1) == []


def test_detects_three_contraction_base():
    dates, high, low, close, volume = _base_series()
    cfg = make_cfg()
    pass_idx = np.arange(93, 101)

    setups = find_setups(
        "TEST", dates, high, low, close, volume, pass_idx, cfg,
        trading_dates=dates,
    )

    assert len(setups) == 1  # same base emitted once, not once per day
    setup = setups[0]
    assert setup.n_contractions == 3
    assert abs(setup.pivot - 97 * 1.005) < 0.2
    depths = setup.contraction_depths
    assert depths[0] > depths[1] > depths[2]
    assert cfg.dryup_ratio_min <= setup.dryup_ratio <= cfg.dryup_ratio_max
    assert setup.last_low < setup.close < setup.pivot
    # final low @ 92 confirms at bar 95 (k=3): no detection before that
    assert dates.searchsorted(pd.Timestamp(setup.detect_date)) >= 95


def test_setup_expiry_counts_global_sessions_during_symbol_halt():
    _, high, low, close, volume = _base_series()
    global_dates = pd.bdate_range("2023-01-02", periods=140)
    # The setup is detected on local bar 95. The symbol then has no bars for
    # thirty global sessions and resumes with its remaining five observations.
    symbol_dates = global_dates[:96].append(global_dates[126:131])
    cfg = make_cfg(setup_valid_days=10)

    setups = find_setups(
        "TEST",
        symbol_dates,
        high,
        low,
        close,
        volume,
        np.array([95]),
        cfg,
        trading_dates=global_dates,
    )

    assert len(setups) == 1
    assert setups[0].detect_date == global_dates[95].date()
    assert setups[0].valid_until == global_dates[105].date()
    assert setups[0].valid_until < symbol_dates[96].date()


def test_vcp_structure_cannot_span_a_symbol_halt():
    _, high, low, close, volume = _base_series()
    global_dates = pd.bdate_range("2023-01-02", periods=150)
    # The first contraction high is before a 40-session halt. Without a gap
    # barrier, local-bar aging incorrectly combines it with post-resumption
    # contractions into a fresh-looking three-contraction VCP.
    symbol_dates = global_dates[:60].append(global_dates[100:141])
    cfg = make_cfg(contractions_min=3)

    setups = find_setups(
        "TEST",
        symbol_dates,
        high,
        low,
        close,
        volume,
        np.arange(93, 101),
        cfg,
        trading_dates=global_dates,
    )

    assert setups == []


def test_vcp_structure_cannot_span_an_incomplete_ohlcv_row():
    dates, high, low, close, volume = _base_series()
    high = high.copy()
    low = low.copy()
    high[71] = np.nan
    low[71] = np.nan

    setups = find_setups(
        "TEST",
        dates,
        high,
        low,
        close,
        volume,
        np.arange(93, 101),
        make_cfg(contractions_min=3),
        trading_dates=dates,
    )

    assert setups == []


def test_future_same_kind_extreme_cannot_rewrite_historical_setup():
    dates, high, low, close, volume = _base_series()
    prefix_len = len(dates)
    tail_len = 16
    dates = pd.bdate_range(dates[0], periods=prefix_len + tail_len)
    tail_high = np.linspace(high[-1] + 0.05, high[-1] + 0.8, tail_len)
    tail_close = np.linspace(close[-1] + 0.02, close[-1] + 0.5, tail_len)
    benign_low = np.linspace(low[-1] + 0.01, low[-1] + 0.4, tail_len)
    high_extended = np.concatenate([high, tail_high])
    close_extended = np.concatenate([close, tail_close])
    volume_extended = np.concatenate([volume, np.full(tail_len, 600_000.0)])
    pass_idx = np.arange(93, prefix_len)

    expected = find_setups(
        "TEST",
        dates,
        high_extended,
        np.concatenate([low, benign_low]),
        close_extended,
        volume_extended,
        pass_idx,
        make_cfg(),
        trading_dates=dates,
    )
    assert len(expected) == 1

    # Each suffix contains a newly confirmed, deeper low without an intervening
    # swing high.  The old global merge used to replace the historical final
    # low with this future event and make the already detected setup disappear.
    for valley_offset in (4, 7, 10):
        adversarial_low = benign_low.copy()
        adversarial_low[valley_offset] = 80.0 - valley_offset
        actual = find_setups(
            "TEST",
            dates,
            high_extended,
            np.concatenate([low, adversarial_low]),
            close_extended,
            volume_extended,
            pass_idx,
            make_cfg(),
            trading_dates=dates,
        )
        assert actual == expected


def test_unchanged_structure_is_not_reemitted_after_validity_window():
    dates, high, low, close, volume = _base_series()
    # Extend far enough that the oldest contraction falls outside BASE_MAX_DAYS.
    # The detector may then select a shorter suffix of the same swing chain,
    # which still must not resurrect its unchanged terminal pivot/low pair.
    extension_len = 50
    dates = pd.bdate_range(dates[0], periods=len(dates) + extension_len)
    extension_close = np.linspace(close[-1] + 0.01, 96.4, extension_len)
    close = np.concatenate([close, extension_close])
    high = np.concatenate([high, extension_close * 1.005])
    low = np.concatenate([low, extension_close * 0.995])
    volume = np.concatenate([volume, np.full(extension_len, 600_000.0)])
    cfg = make_cfg(
        setup_valid_days=5,
        dryup_ratio_min=0.0,
        dryup_ratio_max=2.0,
    )

    setups = find_setups(
        "TEST", dates, high, low, close, volume, np.arange(93, len(dates)), cfg,
        trading_dates=dates,
    )

    assert len(setups) == 1


def test_unconfirmed_stop_breach_prevents_setup_emission():
    dates, high, low, close, volume = _base_series()
    low = low.copy()
    # The final confirmed swing low is at bar 92. A fresh breakdown at bar 96
    # is not yet a confirmed swing, but it is already known on the evaluation
    # day and must invalidate the base immediately.
    low[96] = 80.0

    setups = find_setups(
        "TEST", dates, high, low, close, volume, np.array([96]), make_cfg(),
        trading_dates=dates,
    )

    assert setups == []


def test_no_detection_without_volume_dryup():
    dates, high, low, close, volume = _base_series()
    volume[:] = 1_000_000.0  # no dry-up
    setups = find_setups(
        "TEST", dates, high, low, close, volume, np.arange(93, 101), make_cfg(),
        trading_dates=dates,
    )
    assert setups == []


def test_no_detection_when_volume_is_too_dead():
    dates, high, low, close, volume = _base_series()
    volume[np.arange(len(close)) >= 88] = 250_000.0
    setups = find_setups(
        "TEST", dates, high, low, close, volume, np.arange(93, 101), make_cfg(),
        trading_dates=dates,
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
        trading_dates=dates,
    )
    assert setups == []
