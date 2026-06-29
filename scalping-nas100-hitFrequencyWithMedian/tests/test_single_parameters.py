import tempfile
import unittest
from pathlib import Path

from hfmed_core import parameters


class SingleParameterFileTests(unittest.TestCase):
    def test_loads_live_default_file(self):
        path = Path(__file__).resolve().parents[1] / "single_parameter.ini"
        values = parameters.load_single_session_parameters(str(path))

        # Phase 2 validation: one static consensus global set applied to every session.
        self.assertEqual(13, len(values))
        self.assertEqual(180, values["asia_early"]["LOOKBACK_BARS"])
        self.assertEqual(0.55, values["asia_early"]["LONG_CROSS_QUANTILE"])
        self.assertEqual(180, values["ny_morning"]["LOOKBACK_BARS"])
        self.assertEqual(0.475, values["ny_morning"]["SHORT_CROSS_QUANTILE"])
        self.assertEqual(24.0, values["after_hours_late"]["ALL_STOP_MODES_TAKE_PROFIT_ATR_MULT"])

    def test_rejects_multiple_values_for_single_parameter(self):
        path = Path(__file__).resolve().parents[1] / "single_parameter.ini"
        text = path.read_text(encoding="utf-8").replace(
            "LOOKBACK_BARS = 180",
            "LOOKBACK_BARS = 180, 90",
            1,
        )

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp) / "single_parameter.ini"
            tmp_path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exactly one value"):
                parameters.load_single_session_parameters(str(tmp_path))


if __name__ == "__main__":
    unittest.main()
