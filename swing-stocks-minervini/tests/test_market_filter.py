import numpy as np
import pandas as pd

from src.market_filter import (
    CONFIRMED_UPTREND,
    CORRECTION,
    UPTREND_UNDER_PRESSURE,
    _hysteresis,
    compute_breadth,
    compute_market_model,
)
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
    assert bool(market["breadth_confirmed"].iloc[-1]) is True
    assert not market["breadth_confirmed"].iloc[:199].any()


def _index_bars(dates, qqq_close, qqq_volume):
    rows = []
    for symbol, scale in (("QQQ", 1.0), ("VOO", 0.8)):
        for day, close, volume in zip(dates, qqq_close, qqq_volume):
            rows.append(
                {
                    "symbol": symbol,
                    "date": day,
                    "open": close,
                    "high": close * 1.002,
                    "low": close * 0.998,
                    "close": close * scale,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_follow_through_and_distribution_state_machine():
    dates = pd.bdate_range("2024-01-02", periods=11)
    qqq_close = [100.0, 101.0, 101.2, 101.4, 103.0]
    for _ in range(6):
        qqq_close.append(qqq_close[-1] * 0.997)
    volumes = [100, 110, 100, 100, 150, 160, 170, 180, 190, 200, 210]
    stock_close = pd.DataFrame({"AAA": np.linspace(50, 55, len(dates))}, index=dates)
    cfg = make_cfg(
        min_price=1.0,
        distribution_pressure_count=4,
        distribution_correction_count=6,
    )

    market = compute_market_model(
        stock_close, _index_bars(dates, qqq_close, volumes), cfg
    )

    assert bool(market["follow_through_day"].iloc[4])
    assert market["market_status"].iloc[4] == CONFIRMED_UPTREND
    assert market["market_status"].iloc[8] == UPTREND_UNDER_PRESSURE
    assert market["distribution_days"].iloc[8] == 4
    assert market["market_status"].iloc[10] == CORRECTION
    assert market["distribution_days"].iloc[10] == 6
    assert market["entry_exposure_cap"].iloc[10] == 0.0


def test_distribution_days_expire_and_restore_confirmed_uptrend():
    dates = pd.bdate_range("2024-02-01", periods=10)
    closes = [100.0, 101.0, 101.2, 101.4, 103.0, 102.5, 102.0, 102.2, 102.4, 102.6]
    volumes = [100, 110, 100, 100, 150, 160, 170, 100, 100, 100]
    stock_close = pd.DataFrame({"AAA": np.linspace(50, 55, len(dates))}, index=dates)
    cfg = make_cfg(
        min_price=1.0,
        distribution_lookback_sessions=3,
        distribution_pressure_count=2,
        distribution_correction_count=4,
    )

    market = compute_market_model(stock_close, _index_bars(dates, closes, volumes), cfg)

    assert market["market_status"].iloc[6] == UPTREND_UNDER_PRESSURE
    assert market["market_status"].iloc[9] == CONFIRMED_UPTREND
    assert market["distribution_days"].iloc[9] == 0
