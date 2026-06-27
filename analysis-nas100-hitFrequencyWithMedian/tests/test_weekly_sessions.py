import unittest
from datetime import datetime, timezone

import numpy as np

from hfmed_range_analysis.data import datetime_to_ns
from hfmed_range_analysis.events import CrossingEvents
from hfmed_range_analysis.weekly import aggregate_weekly_session_stats


def _events(times: list[datetime], ranges: list[float]) -> CrossingEvents:
    n = len(times)
    cross_ts_ns = np.array([datetime_to_ns(value) for value in times], dtype=np.int64)
    return CrossingEvents(
        tick_index=np.arange(n, dtype=np.int64),
        cross_ts_ns=cross_ts_ns,
        bar_start_ns=cross_ts_ns,
        direction_code=np.ones(n, dtype=np.int8),
        previous_mid=np.full(n, 99.0, dtype=np.float64),
        signal_mid=np.full(n, 101.0, dtype=np.float64),
        q50_level=np.full(n, 100.0, dtype=np.float64),
        profile_low=np.full(n, 99.0, dtype=np.float64),
        profile_high=np.full(n, 101.0, dtype=np.float64),
        profile_range_points=np.array(ranges, dtype=np.float64),
    )


class WeeklySessionTests(unittest.TestCase):
    def test_groups_by_new_york_calendar_week_and_session(self):
        event_times = [
            datetime(2026, 1, 5, 14, 35, tzinfo=timezone.utc),  # Monday 09:35 NY
            datetime(2026, 1, 5, 15, 5, tzinfo=timezone.utc),   # Monday 10:05 NY
            datetime(2026, 1, 6, 2, 10, tzinfo=timezone.utc),   # Monday 21:10 NY
        ]
        stats = aggregate_weekly_session_stats(_events(event_times, [2.0, 4.0, 8.0]))
        by_session = {item.session_label: item for item in stats}

        self.assertEqual(by_session["NY Open Impulse 09:30-10:00"].week_start_ts.isoformat(), "2026-01-05T05:00:00+00:00")
        self.assertEqual(by_session["NY Open Impulse 09:30-10:00"].crossings_total, 1)
        self.assertEqual(by_session["NY Open Impulse 09:30-10:00"].median_range_points, 2.0)
        self.assertEqual(by_session["NY Morning 10:00-11:30"].median_range_points, 4.0)
        self.assertEqual(by_session["Asia Early 20:00-00:00"].median_range_points, 8.0)


if __name__ == "__main__":
    unittest.main()
