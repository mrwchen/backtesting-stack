import numpy as np
import pandas as pd

from src.runner import _attach_regime_attribution, _regime_entry_allowed

from .util import make_cfg


def test_regime_gate_uses_score_only_from_next_calendar_day() -> None:
    dates = pd.DatetimeIndex(["2024-01-05", "2024-01-08", "2024-01-09"])
    regime = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-07", "2024-01-08"]),
            "regime_label": ["RISK-ON", "RISK-OFF", "RISK-ON", "RISK-OFF"],
        }
    )
    cfg = make_cfg(regime_allowed_labels=("RISK-ON",))

    allowed = _regime_entry_allowed(dates, regime, cfg)

    # Friday sees Thursday, Monday sees Sunday's row, Tuesday sees Monday.
    assert allowed.tolist() == [True, True, False]


def test_regime_attribution_uses_same_causal_availability_mapping() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
            "symbol": ["AAA", "BBB"],
        }
    )
    regime = pd.DataFrame(
        {
            "day": pd.to_datetime(["2024-01-04", "2024-01-05", "2024-01-07"]),
            "regime_composite": [25.0, 85.0, 35.0],
            "regime_label": ["RISK-ON", "RISK-OFF", "CONSTRUCTIVE"],
        }
    )

    attributed = _attach_regime_attribution(trades, regime)

    assert attributed["regime_composite"].tolist() == [25.0, 35.0]
    assert attributed["regime_label"].tolist() == ["RISK-ON", "CONSTRUCTIVE"]


def test_regime_attribution_normalizes_mixed_datetime_resolutions() -> None:
    trades = pd.DataFrame(
        {
            "entry_date": np.array(["2024-01-05", "2024-01-08"], dtype="datetime64[s]"),
            "symbol": ["AAA", "BBB"],
        }
    )
    regime = pd.DataFrame(
        {
            "day": np.array(
                ["2024-01-04", "2024-01-07"], dtype="datetime64[us]"
            ),
            "regime_composite": [25.0, 35.0],
            "regime_label": ["RISK-ON", "CONSTRUCTIVE"],
        }
    )

    attributed = _attach_regime_attribution(trades, regime)

    assert attributed["regime_composite"].tolist() == [25.0, 35.0]
    assert attributed["regime_label"].tolist() == ["RISK-ON", "CONSTRUCTIVE"]
