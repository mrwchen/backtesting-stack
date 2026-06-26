import unittest
from dataclasses import replace

import numpy as np

from hfmed_core import config
from hfmed_core.data import BarData, TickData
from hfmed_core.profile import ProfileArrays
from hfmed_core.sessions import SESSION_CODE_BY_KEY
from hfmed_core.simulation import run_simulation


def _ticks() -> TickData:
    mid = np.array([9.0, 10.5, 12.0], dtype=np.float64)
    return TickData(
        tick_time_ns=np.array([0, 1_000_000_000, 2_000_000_000], dtype=np.int64),
        bid=mid.copy(),
        ask=mid.copy(),
        mid=mid,
        bar_index=np.zeros(3, dtype=np.int32),
        entry_session_code=np.full(3, SESSION_CODE_BY_KEY["ny_midday"], dtype=np.uint8),
        parameter_reference_price=np.full(3, 10_000.0, dtype=np.float64),
    )


def _bars() -> BarData:
    values = np.array([10.0], dtype=np.float64)
    return BarData(
        bar_start_ns=np.array([0], dtype=np.int64),
        open=values,
        high=values,
        low=values,
        close=values,
        tick_count=np.array([3], dtype=np.int32),
    )


def _profile() -> ProfileArrays:
    one = np.array([1.0], dtype=np.float64)
    return ProfileArrays(
        profile_low=np.array([0.0], dtype=np.float64),
        band_lower=np.array([10.0], dtype=np.float64),
        median_level=np.array([50.0], dtype=np.float64),
        band_upper=np.array([90.0], dtype=np.float64),
        long_cross_level=np.array([10.0], dtype=np.float64),
        short_cross_level=np.array([np.nan], dtype=np.float64),
        profile_high=np.array([100.0], dtype=np.float64),
        stop_profile_lower=np.array([0.0], dtype=np.float64),
        stop_profile_upper=np.array([100.0], dtype=np.float64),
        band_width_points=one,
        profile_range_points=np.array([100.0], dtype=np.float64),
    )


def _cfg(deviation: float):
    return replace(
        config.active_run_config(),
        bar_seconds=1,
        lookback_bars=1,
        min_lookback_bars=1,
        long_cross_quantile=0.50,
        short_cross_quantile=0.50,
        entry_price_range_position_max_deviation_pct=deviation,
        stop_mode="band",
        take_profit_bps=1.0,
        min_profile_range_bps=0.0,
        stop_profile_lower_quantile=0.0,
        stop_profile_upper_quantile=1.0,
        stop_profile_buffer_points=0.0,
        min_stop_distance_bps=1.0,
        max_stop_distance_bps=100.0,
        initial_equity=1000.0,
        risk_per_trade_pct=1.0,
        max_margin_pct=100.0,
        margin_requirement_pct=1.0,
        lot_size=0.1,
    )


class EntryRangePositionFilterTests(unittest.TestCase):
    def test_rejects_cross_when_price_position_deviates_too_far_from_quantile(self):
        result = run_simulation(
            _ticks(),
            _bars(),
            np.zeros(3, dtype=np.int32),
            _cfg(15.0),
            profile=_profile(),
            log_result=False,
        )

        self.assertEqual(result.signals_total, 1)
        self.assertEqual(result.rejected_signals_price_range_position, 1)
        self.assertEqual(len(result.trades), 0)

    def test_allows_cross_inside_configured_price_position_deviation(self):
        result = run_simulation(
            _ticks(),
            _bars(),
            np.zeros(3, dtype=np.int32),
            _cfg(50.0),
            profile=_profile(),
            log_result=False,
        )

        self.assertEqual(result.rejected_signals_price_range_position, 0)
        self.assertEqual(len(result.trades), 1)
        self.assertAlmostEqual(result.trades[0].cross_price_range_position_pct, 10.0)
        self.assertAlmostEqual(result.trades[0].range_position_deviation_pct, 40.0)


if __name__ == "__main__":
    unittest.main()
