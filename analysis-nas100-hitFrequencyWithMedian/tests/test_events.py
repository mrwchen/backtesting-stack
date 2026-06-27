import unittest

import numpy as np

from hfmed_range_analysis.data import BarData, TickData
from hfmed_range_analysis.events import detect_q50_crossing_events
from hfmed_range_analysis.profile import RangeProfileArrays


class CrossingEventTests(unittest.TestCase):
    def test_detects_up_and_down_q50_crossings(self):
        ticks = TickData(
            tick_time_ns=np.arange(5, dtype=np.int64) * 1_000_000_000,
            bid=np.array([98.5, 100.5, 100.5, 97.5, 99.5], dtype=np.float64),
            ask=np.array([99.5, 101.5, 101.5, 98.5, 100.5], dtype=np.float64),
            mid=np.array([99.0, 101.0, 101.0, 98.0, 100.0], dtype=np.float64),
            bar_index=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        )
        bars = BarData(
            bar_start_ns=np.array([0, 2_000_000_000], dtype=np.int64),
            open=np.array([99.0, 101.0], dtype=np.float64),
            high=np.array([101.0, 101.0], dtype=np.float64),
            low=np.array([99.0, 98.0], dtype=np.float64),
            close=np.array([101.0, 100.0], dtype=np.float64),
            tick_count=np.array([2, 3], dtype=np.int32),
        )
        profile = RangeProfileArrays(
            profile_low=np.array([98.0, 97.0], dtype=np.float64),
            median_level=np.array([100.0, 100.0], dtype=np.float64),
            profile_high=np.array([102.0, 103.0], dtype=np.float64),
            profile_range_points=np.array([4.0, 6.0], dtype=np.float64),
        )

        events = detect_q50_crossing_events(ticks, bars, profile)

        self.assertEqual(len(events), 3)
        np.testing.assert_array_equal(events.tick_index, np.array([1, 3, 4], dtype=np.int64))
        np.testing.assert_array_equal(events.direction_code, np.array([1, -1, 1], dtype=np.int8))
        np.testing.assert_allclose(events.q50_level, np.array([100.0, 100.0, 100.0]))
        np.testing.assert_allclose(events.profile_range_points, np.array([4.0, 6.0, 6.0]))
        np.testing.assert_allclose(events.range_to_price_pct, np.array([4.0 / 101.0 * 100.0, 6.0 / 98.0 * 100.0, 6.0]))
        np.testing.assert_allclose(events.range_to_price_bps, np.array([4.0 / 101.0 * 10000.0, 6.0 / 98.0 * 10000.0, 600.0]))


if __name__ == "__main__":
    unittest.main()
