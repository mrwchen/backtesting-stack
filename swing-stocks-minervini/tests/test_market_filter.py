import numpy as np
import pandas as pd

from src.market_filter import _hysteresis, compute_breadth
from tests.util import make_cfg


def test_hysteresis_on_off_thresholds():
    values = np.array([0.60, 0.48, 0.44, 0.47, 0.49, 0.52, np.nan, 0.60])
    out = _hysteresis(values, on_threshold=0.50, off_threshold=0.45)
    # on at 0.60; stays on at 0.48 (>= off); off at 0.44; stays off at
    # 0.47/0.49 (< on); on again at 0.52; NaN forces off; back on at 0.60
    assert out.tolist() == [True, True, False, False, False, True, False, True]


def test_breadth_counts_stocks_above_ma200():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(
        {
            "UP1": np.linspace(50, 150, n),
            "UP2": np.linspace(30, 90, n),
            "UP3": np.linspace(20, 60, n),
            "DOWN": np.linspace(150, 50, n),
        },
        index=dates,
    )
    cfg = make_cfg(min_price=1.0)
    market = compute_breadth(close, cfg)

    assert market["market_breadth"].iloc[:199].isna().all()  # MA200 warmup
    assert abs(market["market_breadth"].iloc[-1] - 0.75) < 1e-9  # 3 of 4 above
    assert bool(market["market_on"].iloc[-1]) is True
    assert not market["market_on"].iloc[:199].any()  # NaN warmup -> gate off
