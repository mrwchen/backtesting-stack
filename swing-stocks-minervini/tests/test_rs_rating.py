import numpy as np
import pandas as pd

from src.rs_rating import compute_rs
from tests.util import make_cfg


def _frame(values: dict, periods: int) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(values, index=dates)


def _continuity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment = pd.DataFrame(1, index=frame.index, columns=frame.columns)
    boundary = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    boundary.iloc[0] = True
    return segment, boundary


def test_ranking_orders_momentum():
    n = 300
    up = 100 * (1.005 ** np.arange(n))
    flat = np.full(n, 100.0)
    down = 100 * (0.997 ** np.arange(n))
    close = _frame({"UP": up, "FLAT": flat, "DOWN": down}, n)
    volume = _frame({s: np.full(n, 1e6) for s in close.columns}, n)
    segment, boundary = _continuity(close)

    cfg = make_cfg(min_price=1.0, min_dollar_volume=1.0)
    rs = compute_rs(
        close,
        volume,
        cfg,
        raw_close=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    last = rs["rs_rating"].iloc[-1]
    assert last["UP"] > last["FLAT"] > last["DOWN"]
    assert 1 <= last.min() and last.max() <= 99
    assert rs["universe_size"].iloc[-1] == 3


def test_not_rankable_before_full_lookback():
    n = 300
    close = _frame({"A": np.linspace(100, 200, n), "B": np.linspace(100, 120, n)}, n)
    volume = _frame({"A": np.full(n, 1e6), "B": np.full(n, 1e6)}, n)
    segment, boundary = _continuity(close)

    cfg = make_cfg(min_price=1.0, min_dollar_volume=1.0)
    rs = compute_rs(
        close,
        volume,
        cfg,
        raw_close=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert rs["rs_rating"].iloc[250].isna().all()  # < 252 bars of history
    assert rs["rs_rating"].iloc[-1].notna().all()


def test_universe_min_price_uses_raw_as_traded_close():
    n = 300
    # SPLIT looks cheap only in the adjusted technical series.  At the time it
    # traded at $20, so a future split must not retroactively exclude it.
    close = _frame({"SPLIT": np.full(n, 2.0), "CHEAP": np.full(n, 20.0)}, n)
    raw_close = _frame({"SPLIT": np.full(n, 20.0), "CHEAP": np.full(n, 2.0)}, n)
    volume = _frame({s: np.full(n, 1e6) for s in close.columns}, n)
    segment, boundary = _continuity(close)

    cfg = make_cfg(min_price=5.0, min_dollar_volume=1.0)
    rs = compute_rs(
        close,
        volume,
        cfg,
        raw_close=raw_close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert np.isnan(rs["rs_rating"].iloc[-1]["CHEAP"])
    assert not np.isnan(rs["rs_rating"].iloc[-1]["SPLIT"])


def test_rs_uses_disjoint_quarterly_returns_with_40_20_20_20_weights():
    n = 260
    close = _frame({"A": np.full(n, 100.0)}, n)
    # At t: 200; t-63: 100; older quarter boundaries remain 100.  Only the
    # latest-quarter 2x term should contribute, giving 2 * (2 - 1) = 2.
    close.iloc[-1, 0] = 200.0
    volume = _frame({"A": np.full(n, 1e6)}, n)
    segment, boundary = _continuity(close)

    rs = compute_rs(
        close,
        volume,
        make_cfg(min_price=1.0, min_dollar_volume=1.0),
        raw_close=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert rs["rs_raw"].iloc[-1, 0] == 2.0


def test_raw_close_shape_must_match_adjusted_close():
    close = _frame({"A": np.full(260, 100.0)}, 260)
    volume = close.copy()
    raw_close = close.rename(columns={"A": "B"})
    segment, boundary = _continuity(close)

    with np.testing.assert_raises_regex(ValueError, "same index and columns"):
        compute_rs(
            close,
            volume,
            make_cfg(),
            raw_close=raw_close,
            continuity_segment=segment,
            continuity_break=boundary,
        )


def test_reorganisation_jump_cannot_enter_rs_returns():
    n = 570
    break_at = 300
    close = _frame(
        {
            "REORG": np.r_[
                np.linspace(8.0, 12.0, break_at),
                np.linspace(900.0, 1200.0, n - break_at),
            ],
            "CONTROL": np.linspace(50.0, 100.0, n),
        },
        n,
    )
    volume = _frame({symbol: np.full(n, 1e6) for symbol in close.columns}, n)
    segment, boundary = _continuity(close)
    segment.loc[segment.index[break_at] :, "REORG"] = 2
    boundary.loc[boundary.index[break_at], "REORG"] = True

    rs = compute_rs(
        close,
        volume,
        make_cfg(min_price=1.0, min_dollar_volume=1.0),
        raw_close=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert np.isnan(rs["rs_raw"].iloc[break_at, close.columns.get_loc("REORG")])
    assert np.isnan(rs["rs_raw"].iloc[break_at + 251, close.columns.get_loc("REORG")])
    assert not np.isnan(
        rs["rs_raw"].iloc[break_at + 252, close.columns.get_loc("REORG")]
    )


def test_dollar_volume_window_restarts_at_reorganisation_boundary():
    n = 100
    break_at = 70
    close = _frame(
        {"REORG": np.r_[np.full(break_at, 100.0), np.full(n - break_at, 10.0)]},
        n,
    )
    volume = _frame(
        {"REORG": np.r_[np.full(break_at, 1_000_000.0), np.full(n - break_at, 100.0)]},
        n,
    )
    segment, boundary = _continuity(close)
    segment.iloc[break_at:, 0] = 2
    boundary.iloc[break_at, 0] = True

    rs = compute_rs(
        close,
        volume,
        make_cfg(min_price=1.0, min_dollar_volume=1.0),
        raw_close=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert np.isnan(rs["dollar_volume"].iloc[break_at + 18, 0])
    assert rs["dollar_volume"].iloc[break_at + 19, 0] == 1_000.0
