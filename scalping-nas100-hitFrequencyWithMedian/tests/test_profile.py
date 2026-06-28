import unittest
from dataclasses import replace

import numpy as np

from hfmed_core import config
from hfmed_core.data import BarData
from hfmed_core.profile import rolling_profile_arrays


def _bars(times_seconds: list[int], prices: list[float]) -> BarData:
    values = np.array(prices, dtype=np.float64)
    return BarData(
        bar_start_ns=np.array(times_seconds, dtype=np.int64) * 1_000_000_000,
        open=values,
        high=values,
        low=values,
        close=values,
        tick_count=np.ones(len(values), dtype=np.int32),
    )


def _ohlc_bars(
    times_seconds: list[int],
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
) -> BarData:
    return BarData(
        bar_start_ns=np.array(times_seconds, dtype=np.int64) * 1_000_000_000,
        open=np.array(opens, dtype=np.float64),
        high=np.array(highs, dtype=np.float64),
        low=np.array(lows, dtype=np.float64),
        close=np.array(closes, dtype=np.float64),
        tick_count=np.ones(len(times_seconds), dtype=np.int32),
    )


def _cfg(**overrides):
    base = config.active_run_config()
    fields = {
        "bar_seconds": 10,
        "lookback_bars": 3,
        "min_lookback_bars": 3,
        "profile_max_lookback_seconds": None,
        "price_step": 1.0,
        "band_lower_quantile": 0.45,
        "median_quantile": 0.5,
        "band_upper_quantile": 0.55,
        "long_cross_quantile": 0.5,
        "short_cross_quantile": 0.5,
        "stop_profile_lower_quantile": 0.0,
        "stop_profile_upper_quantile": 1.0,
    }
    fields.update(overrides)
    return replace(base, **fields)


class RollingProfileTests(unittest.TestCase):
    def test_wall_clock_gap_clears_stale_lookback_bars(self):
        bars = _bars(
            [0, 10, 20, 1000, 1010, 1020, 1030],
            [1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0],
        )

        profile = rolling_profile_arrays(bars, _cfg())

        self.assertTrue(np.isnan(profile.median_level[3:6]).all())
        self.assertEqual(profile.median_level[6], 101.0)

    def test_atr_includes_sub_step_bars_without_price_levels(self):
        bars = _ohlc_bars(
            [0, 10, 20, 30],
            [100.10, 100.20, 101.00, 102.00],
            [100.40, 100.50, 101.00, 102.00],
            [100.10, 100.20, 101.00, 102.00],
            [100.20, 100.30, 101.00, 102.00],
        )

        profile = rolling_profile_arrays(bars, _cfg(min_lookback_bars=1))

        expected_atr = (0.30 + 0.30 + 0.70) / 3.0
        self.assertAlmostEqual(profile.atr_points[3], expected_atr)


if __name__ == "__main__":
    unittest.main()
