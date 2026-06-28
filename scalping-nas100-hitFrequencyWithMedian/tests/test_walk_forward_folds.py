import unittest
from dataclasses import replace
from datetime import datetime

import numpy as np

from hfmed_core import config
from hfmed_core.data import TickData, datetime_to_ns
from hfmed_core.optimizer import build_folds


def _ns(value: str) -> int:
    return datetime_to_ns(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _ticks(*values: str) -> TickData:
    tick_time_ns = np.array([_ns(value) for value in values], dtype=np.int64)
    zeros_float = np.zeros(len(tick_time_ns), dtype=np.float64)
    return TickData(
        tick_time_ns=tick_time_ns,
        bid=zeros_float,
        ask=zeros_float,
        mid=zeros_float,
        bar_index=np.arange(len(tick_time_ns), dtype=np.int32),
        entry_session_code=np.zeros(len(tick_time_ns), dtype=np.uint8),
        parameter_reference_price=zeros_float,
    )


class WalkForwardFoldTests(unittest.TestCase):
    def test_folds_count_observed_new_york_trading_days(self):
        ticks = _ticks(
            "2026-01-06T02:00:00Z",  # Monday 21:00 New York.
            "2026-01-06T15:00:00Z",  # Tuesday 10:00 New York.
            "2026-01-06T20:00:00Z",  # Tuesday 15:00 New York.
        )
        opt_cfg = replace(
            config.active_optimizer_config(),
            train_trading_days=1,
            test_trading_days=1,
            step_trading_days=1,
            trading_day_timezone="America/New_York",
        )

        folds = build_folds(ticks, opt_cfg)

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].train_start_ns, _ns("2026-01-06T02:00:00Z"))
        self.assertEqual(folds[0].train_end_ns, _ns("2026-01-06T15:00:00Z"))
        self.assertEqual(folds[0].test_start_ns, _ns("2026-01-06T15:00:00Z"))
        self.assertEqual(folds[0].test_end_ns, _ns("2026-01-06T20:00:00Z") + 1)

    def test_folds_skip_weekend_dates_without_ticks(self):
        ticks = _ticks(
            "2026-01-02T15:00:00Z",  # Friday 10:00 New York.
            "2026-01-05T15:00:00Z",  # Monday 10:00 New York.
            "2026-01-06T15:00:00Z",  # Tuesday 10:00 New York.
        )
        opt_cfg = replace(
            config.active_optimizer_config(),
            train_trading_days=2,
            test_trading_days=1,
            step_trading_days=1,
            trading_day_timezone="America/New_York",
        )

        folds = build_folds(ticks, opt_cfg)

        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].train_start_ns, _ns("2026-01-02T15:00:00Z"))
        self.assertEqual(folds[0].train_end_ns, _ns("2026-01-06T15:00:00Z"))
        self.assertEqual(folds[0].test_start_ns, _ns("2026-01-06T15:00:00Z"))
        self.assertEqual(folds[0].test_end_ns, _ns("2026-01-06T15:00:00Z") + 1)


if __name__ == "__main__":
    unittest.main()
