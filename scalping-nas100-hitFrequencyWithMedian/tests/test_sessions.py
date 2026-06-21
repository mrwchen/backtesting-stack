import unittest

import numpy as np

from hfmed_core.sessions import classify_minutes


class SessionBoundaryTests(unittest.TestCase):
    def test_refined_session_boundaries(self):
        minutes = np.array([
            3 * 60 + 59,
            4 * 60,
            7 * 60,
            8 * 60 + 30,
            9 * 60 + 30,
            10 * 60,
            11 * 60 + 30,
            14 * 60,
            15 * 60,
            16 * 60,
            17 * 60,
            20 * 60,
        ])

        self.assertEqual(classify_minutes(minutes).tolist(), [
            "overnight",
            "pre_market_early",
            "pre_market_active",
            "pre_market_macro",
            "ny_open_impulse",
            "ny_morning",
            "ny_midday",
            "ny_late",
            "ny_power_hour",
            "after_close_shock",
            "after_hours_late",
            "overnight",
        ])


if __name__ == "__main__":
    unittest.main()
