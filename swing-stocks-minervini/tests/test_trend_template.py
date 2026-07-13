import numpy as np
import pandas as pd

from src.trend_template import compute_template
from tests.util import make_cfg


def _continuity(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    segment = pd.DataFrame(1, index=frame.index, columns=frame.columns)
    boundary = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    boundary.iloc[0] = True
    return segment, boundary


def test_uptrend_passes_downtrend_fails():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(
        {"UP": np.linspace(50, 150, n), "DOWN": np.linspace(150, 50, n)}, index=dates
    )
    rs = pd.DataFrame({"UP": np.full(n, 90.0), "DOWN": np.full(n, 90.0)}, index=dates)
    segment, boundary = _continuity(close)

    cfg = make_cfg()
    template = compute_template(
        close,
        rs,
        cfg,
        high=close,
        low=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )
    last = template["template_pass"].iloc[-1]

    assert bool(last["UP"]) is True
    assert bool(last["DOWN"]) is False


def test_rs_criterion_gates_template():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n)}, index=dates)
    rs_low = pd.DataFrame({"UP": np.full(n, 50.0)}, index=dates)
    segment, boundary = _continuity(close)

    cfg = make_cfg(rs_min=70)
    template = compute_template(
        close,
        rs_low,
        cfg,
        high=close,
        low=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert bool(template["template_pass"].iloc[-1]["UP"]) is False
    assert bool(template["crit_price_above_ma50"].iloc[-1]["UP"]) is True


def test_no_pass_without_ma_history():
    n = 100  # fewer bars than the 200d MA needs
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n)}, index=dates)
    rs = pd.DataFrame({"UP": np.full(n, 90.0)}, index=dates)
    segment, boundary = _continuity(close)

    template = compute_template(
        close,
        rs,
        make_cfg(),
        high=close,
        low=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )
    assert not template["template_pass"].to_numpy().any()


def test_52_week_distance_uses_session_highs_and_lows() -> None:
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n)}, index=dates)
    high = close.copy()
    low = close.copy()
    high.iloc[-20, 0] = 220.0
    low.iloc[-30, 0] = 30.0
    rs = pd.DataFrame({"UP": np.full(n, 95.0)}, index=dates)
    segment, boundary = _continuity(close)

    template = compute_template(
        close,
        rs,
        make_cfg(),
        high=high,
        low=low,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    assert not bool(template["crit_near_52w_high"].iloc[-1, 0])
    assert bool(template["crit_above_52w_low"].iloc[-1, 0])


def test_reorganisation_jump_restarts_every_trend_window():
    n = 360
    break_at = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    post_break = np.linspace(1000.0, 1100.0, n - break_at)
    close = pd.DataFrame(
        {"REORG": np.r_[np.linspace(8.0, 12.0, break_at), post_break]},
        index=dates,
    )
    rs = pd.DataFrame({"REORG": np.full(n, 95.0)}, index=dates)
    segment, boundary = _continuity(close)
    segment.iloc[break_at:, 0] = 2
    boundary.iloc[break_at, 0] = True

    template = compute_template(
        close,
        rs,
        make_cfg(),
        high=close,
        low=close,
        continuity_segment=segment,
        continuity_break=boundary,
    )

    # A standard rolling mean would import 49 pre-reorganisation observations
    # immediately. The segment-aware mean needs 50 observations from segment 2.
    assert np.isnan(template["ma50"].iloc[break_at + 48, 0])
    assert np.isclose(
        template["ma50"].iloc[break_at + 49, 0], np.mean(post_break[:50])
    )
    assert not template["template_pass"].iloc[break_at:].to_numpy().any()


def test_continuity_break_must_match_segment_change():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"A": np.linspace(50.0, 150.0, n)}, index=dates)
    rs = pd.DataFrame({"A": np.full(n, 95.0)}, index=dates)
    segment, boundary = _continuity(close)
    segment.iloc[200:, 0] = 2

    with np.testing.assert_raises_regex(ValueError, "are inconsistent"):
        compute_template(
            close,
            rs,
            make_cfg(),
            high=close,
            low=close,
            continuity_segment=segment,
            continuity_break=boundary,
        )
