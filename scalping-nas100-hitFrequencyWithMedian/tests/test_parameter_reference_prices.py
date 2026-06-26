import unittest
from datetime import datetime, timezone

import numpy as np

from hfmed_core.data import build_parameter_reference_prices, datetime_to_ns


class ParameterReferencePriceTests(unittest.TestCase):
    def test_uses_first_mid_per_local_midnight_day(self):
        tick_time_ns = np.array(
            [
                datetime_to_ns(datetime(2026, 1, 1, 4, 59, 59, tzinfo=timezone.utc)),
                datetime_to_ns(datetime(2026, 1, 1, 5, 0, 0, tzinfo=timezone.utc)),
                datetime_to_ns(datetime(2026, 1, 1, 6, 0, 0, tzinfo=timezone.utc)),
                datetime_to_ns(datetime(2026, 1, 2, 5, 0, 0, tzinfo=timezone.utc)),
            ],
            dtype=np.int64,
        )
        mid = np.array([100.0, 200.0, 210.0, 300.0], dtype=np.float64)

        refs = build_parameter_reference_prices(tick_time_ns, mid, "America/New_York")

        np.testing.assert_array_equal(refs, np.array([100.0, 200.0, 200.0, 300.0]))


if __name__ == "__main__":
    unittest.main()
