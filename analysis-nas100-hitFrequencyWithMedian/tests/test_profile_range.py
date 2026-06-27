import unittest

import numpy as np

from hfmed_range_analysis.data import BarData
from hfmed_range_analysis.profile import RangeProfileConfig, rolling_range_profile_arrays


def _bars(times_seconds: list[int], lows: list[float], highs: list[float]) -> BarData:
    low_values = np.array(lows, dtype=np.float64)
    high_values = np.array(highs, dtype=np.float64)
    return BarData(
        bar_start_ns=np.array(times_seconds, dtype=np.int64) * 1_000_000_000,
        open=low_values,
        high=high_values,
        low=low_values,
        close=high_values,
        tick_count=np.ones(len(low_values), dtype=np.int32),
    )


def _cfg(**overrides) -> RangeProfileConfig:
    values = {
        "price_step": 1.0,
        "lookback_bars": 2,
        "min_lookback_bars": 2,
        "bar_seconds": 5,
        "profile_max_lookback_seconds": None,
    }
    values.update(overrides)
    return RangeProfileConfig(**values)


class RollingRangeProfileTests(unittest.TestCase):
    def test_range_uses_prior_completed_bars_only(self):
        bars = _bars(
            [0, 5, 10, 15],
            [100.0, 101.0, 200.0, 300.0],
            [100.0, 103.0, 201.0, 301.0],
        )

        arrays = rolling_range_profile_arrays(bars, _cfg())

        self.assertTrue(np.isnan(arrays.profile_range_points[:2]).all())
        self.assertEqual(arrays.profile_low[2], 100.0)
        self.assertEqual(arrays.median_level[2], 101.0)
        self.assertEqual(arrays.profile_high[2], 103.0)
        self.assertEqual(arrays.profile_range_points[2], 3.0)
        self.assertEqual(arrays.profile_low[3], 101.0)
        self.assertEqual(arrays.median_level[3], 103.0)
        self.assertEqual(arrays.profile_high[3], 201.0)
        self.assertEqual(arrays.profile_range_points[3], 100.0)

    def test_wall_clock_gap_clears_stale_lookback_bars(self):
        bars = _bars(
            [0, 5, 10, 1000, 1005, 1010, 1015],
            [1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0],
            [1.0, 2.0, 3.0, 100.0, 101.0, 102.0, 103.0],
        )

        arrays = rolling_range_profile_arrays(bars, _cfg(lookback_bars=3, min_lookback_bars=3))

        self.assertTrue(np.isnan(arrays.profile_range_points[3:6]).all())
        self.assertEqual(arrays.profile_low[6], 100.0)
        self.assertEqual(arrays.median_level[6], 101.0)
        self.assertEqual(arrays.profile_high[6], 102.0)
        self.assertEqual(arrays.profile_range_points[6], 2.0)


if __name__ == "__main__":
    unittest.main()
