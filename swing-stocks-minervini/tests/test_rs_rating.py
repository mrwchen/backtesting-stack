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
    rs = compute_rs(close, volume, cfg, raw_close=close)

    last = rs["rs_rating"].iloc[-1]
    assert last["UP"] > last["FLAT"] > last["DOWN"]
    assert 1 <= last.min() and last.max() <= 99
    assert rs["universe_size"].iloc[-1] == 3


def test_not_rankable_before_full_lookback():
    n = 300
    close = _frame({"A": np.linspace(100, 200, n), "B": np.linspace(100, 120, n)}, n)
    volume = _frame({"A": np.full(n, 1e6), "B": np.full(n, 1e6)}, n)

    cfg = make_cfg(min_price=1.0, min_dollar_volume=1.0)
    rs = compute_rs(close, volume, cfg, raw_close=close)

    assert rs["rs_rating"].iloc[250].isna().all()  # < 252 bars of history
    assert rs["rs_rating"].iloc[-1].notna().all()


def test_universe_min_price_uses_raw_as_traded_close():
    n = 300
    # SPLIT looks cheap only in the adjusted technical series.  At the time it
    # traded at $20, so a future split must not retroactively exclude it.
    close = _frame({"SPLIT": np.full(n, 2.0), "CHEAP": np.full(n, 20.0)}, n)
    raw_close = _frame({"SPLIT": np.full(n, 20.0), "CHEAP": np.full(n, 2.0)}, n)
    volume = _frame({s: np.full(n, 1e6) for s in close.columns}, n)

    cfg = make_cfg(min_price=5.0, min_dollar_volume=1.0)
    rs = compute_rs(close, volume, cfg, raw_close=raw_close)

    assert np.isnan(rs["rs_rating"].iloc[-1]["CHEAP"])
    assert not np.isnan(rs["rs_rating"].iloc[-1]["SPLIT"])


def test_rs_uses_disjoint_quarterly_returns_with_40_20_20_20_weights():
    n = 260
    close = _frame({"A": np.full(n, 100.0)}, n)
    # At t: 200; t-63: 100; older quarter boundaries remain 100.  Only the
    # latest-quarter 2x term should contribute, giving 2 * (2 - 1) = 2.
    close.iloc[-1, 0] = 200.0
    volume = _frame({"A": np.full(n, 1e6)}, n)

    rs = compute_rs(
        close,
        volume,
        make_cfg(min_price=1.0, min_dollar_volume=1.0),
        raw_close=close,
    )

    assert rs["rs_raw"].iloc[-1, 0] == 2.0


def test_raw_close_shape_must_match_adjusted_close():
    close = _frame({"A": np.full(260, 100.0)}, 260)
    volume = close.copy()
    raw_close = close.rename(columns={"A": "B"})

    with np.testing.assert_raises_regex(ValueError, "same index and columns"):
        compute_rs(close, volume, make_cfg(), raw_close=raw_close)
