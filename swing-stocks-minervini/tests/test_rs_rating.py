import numpy as np
import pandas as pd

from src.rs_rating import compute_rs
from tests.util import make_cfg


def _frame(values: dict, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(values, index=dates)


def test_ranking_orders_momentum():
    n = 300
    up = 100 * (1.005 ** np.arange(n))
    flat = np.full(n, 100.0)
    down = 100 * (0.997 ** np.arange(n))
    close = _frame({"UP": up, "FLAT": flat, "DOWN": down}, n)
    volume = _frame({s: np.full(n, 1e6) for s in close.columns}, n)

    cfg = make_cfg(min_price=1.0, min_dollar_volume=1.0)
    rs = compute_rs(close, volume, cfg)

    last = rs["rs_rating"].iloc[-1]
    assert last["UP"] > last["FLAT"] > last["DOWN"]
    assert 1 <= last.min() and last.max() <= 99
    assert rs["universe_size"].iloc[-1] == 3


def test_not_rankable_before_full_lookback():
    n = 300
    close = _frame({"A": np.linspace(100, 200, n), "B": np.linspace(100, 120, n)}, n)
    volume = _frame({"A": np.full(n, 1e6), "B": np.full(n, 1e6)}, n)

    cfg = make_cfg(min_price=1.0, min_dollar_volume=1.0)
    rs = compute_rs(close, volume, cfg)

    assert rs["rs_rating"].iloc[250].isna().all()  # < 252 bars of history
    assert rs["rs_rating"].iloc[-1].notna().all()


def test_universe_filters_exclude_cheap_and_illiquid():
    n = 300
    close = _frame({"OK": np.full(n, 50.0), "CHEAP": np.full(n, 2.0)}, n)
    volume = _frame({"OK": np.full(n, 1e6), "CHEAP": np.full(n, 1e6)}, n)

    cfg = make_cfg(min_price=5.0, min_dollar_volume=1.0)
    rs = compute_rs(close, volume, cfg)

    assert np.isnan(rs["rs_rating"].iloc[-1]["CHEAP"])
    assert not np.isnan(rs["rs_rating"].iloc[-1]["OK"])
