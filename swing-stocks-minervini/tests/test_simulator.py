import numpy as np
import pandas as pd
import pytest

from src.simulator import simulate as _simulate
from tests.util import make_cfg


def simulate(
    dates,
    symbols,
    open_m,
    high_m,
    low_m,
    close_m,
    setups,
    cfg,
    *args,
    **kwargs,
):
    """Supply explicit single-segment metadata for synthetic unit fixtures."""
    continuity = kwargs.pop(
        "continuity_segment_m",
        pd.DataFrame(1, index=dates, columns=symbols, dtype="int64"),
    )
    setups = setups.copy()
    if "price_continuity_segment" not in setups.columns:
        setups["price_continuity_segment"] = 1
    return _simulate(
        dates,
        symbols,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        cfg,
        *args,
        continuity_segment_m=continuity,
        **kwargs,
    )


def _matrices(bars: list[tuple[float, float, float, float]]):
    dates = pd.bdate_range("2024-01-01", periods=len(bars))
    arr = np.array(bars)
    frames = [
        pd.DataFrame({"TEST": arr[:, i]}, index=dates) for i in range(4)
    ]
    return dates, frames[0], frames[1], frames[2], frames[3]


def _multi_matrices(
    bars_by_symbol: dict[str, list[tuple[float, float, float, float]]],
):
    lengths = {len(bars) for bars in bars_by_symbol.values()}
    assert len(lengths) == 1
    dates = pd.bdate_range("2024-01-01", periods=lengths.pop())
    arrays = {symbol: np.asarray(bars) for symbol, bars in bars_by_symbol.items()}
    frames = [
        pd.DataFrame({symbol: values[:, field] for symbol, values in arrays.items()}, index=dates)
        for field in range(4)
    ]
    return dates, frames[0], frames[1], frames[2], frames[3]


def _setup_row(dates, detect_idx, pivot, stop, valid_idx):
    return pd.DataFrame(
        [
            {
                "setup_id": 1,
                "symbol": "TEST",
                "detect_date": dates[detect_idx].date(),
                "pivot": pivot,
                "last_low": stop,
                "stop_level": stop,
                "valid_until": dates[valid_idx].date(),
            }
        ]
    )


def _clean_cfg():
    return make_cfg(
        slippage_pct=0.0, commission_pct=0.0, risk_pct=0.01,
        partial_at_r=2.0, partial_fraction=0.5, breakeven_after_partial=True,
        pivot_buffer_pct=0.0, max_buy_zone_pct=0.05,
        time_stop_sessions=999,
        exposure_levels=(1.0,),
    )


def test_new_default_rule_rejects_entry_above_two_percent_buy_zone():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(103.0, 104.0, 102.0, 103.0)]
    bars += [(103.0, 104.0, 102.0, 103.0)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8)
    cfg = make_cfg(
        slippage_pct=0.0, commission_pct=0.0, partial_fraction=0.0,
        pivot_buffer_pct=0.001, max_buy_zone_pct=0.02,
    )
    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )
    assert result.trades.empty


def test_time_stop_exits_next_open_after_no_one_r_progress():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.2)]
    bars += [(100.2, 102, 99, 100.5)]
    bars += [(100.2, 102, 99, 99.8)]  # minimum age and back below pivot
    bars += [(99.5, 101, 99, 100.0)]
    bars += [(100.0, 101, 99, 100.0)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 9)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__, "partial_fraction": 0.0,
            "time_stop_sessions": 2, "time_stop_min_r": 1.0,
        }
    )
    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )
    assert result.trades.iloc[0]["exit_reason"] == "time_stop"
    assert result.trades.iloc[0]["exit_date"] == dates[8].date()
    assert result.trades.iloc[0]["exit_price"] == 99.5


def test_time_stop_does_not_exit_slow_trade_while_close_holds_pivot():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.2)]
    bars += [(100.2, 102, 99, 100.2)] * 4
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 9)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__, "partial_fraction": 0.0,
            "time_stop_sessions": 2, "time_stop_min_r": 1.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades.iloc[0]["exit_reason"] == "eod"


def test_progressive_exposure_steps_up_after_two_confirmed_winners():
    dates = pd.bdate_range("2024-01-01", periods=14)
    symbols = ["AAA", "BBB", "CCC"]
    fields = {
        symbol: np.array([(98.0, 99.0, 97.0, 98.0)] * len(dates))
        for symbol in symbols
    }
    for symbol, entry in (("AAA", 5), ("BBB", 8), ("CCC", 11)):
        fields[symbol][entry] = (99.0, 101.0, 98.0, 100.2)
        fields[symbol][entry + 1] = (101.0, 111.0, 100.0, 108.0)
        fields[symbol][entry + 2 :] = (106.0, 107.0, 105.0, 106.0)
    frames = [
        pd.DataFrame({symbol: fields[symbol][:, field] for symbol in symbols}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, detect, 100.0, 95.0, 13).assign(symbol=symbol, setup_id=index)
            for index, (symbol, detect) in enumerate((("AAA", 4), ("BBB", 7), ("CCC", 10)), 1)
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__, "simulation_mode": "portfolio",
            "partial_fraction": 0.0, "pivot_buffer_pct": 0.0,
            "max_buy_zone_pct": 0.02,
            "exposure_levels": (0.5, 0.75, 1.0),
            "portfolio_max_open_positions": 8,
        }
    )
    result = simulate(dates, pd.Index(symbols), *frames, setups, cfg)
    final_legs = result.trades[result.trades["leg"] == "final"].set_index("symbol")
    assert final_legs.loc["AAA", "pnl"] > 0
    assert final_legs.loc["BBB", "pnl"] > 0
    # Feedback changes only aggregate gross capacity. It must not scale the
    # otherwise identical per-trade risk budget a second time.
    assert final_legs.loc["CCC", "shares"] > final_legs.loc["BBB", "shares"]
    assert result.equity.loc[
        result.equity["period_end_date"] == dates[11].date(),
        "feedback_exposure_level",
    ].iloc[0] == 0.75


def test_previous_close_open_winner_can_raise_next_session_exposure():
    dates = pd.bdate_range("2024-01-01", periods=9)
    aaa = [(98, 99, 97, 98)] * 5
    aaa += [(99, 106, 98, 105.5)]  # entry; close confirms more than +1R
    aaa += [(106, 107, 105, 106)] * 3
    bbb = [(98, 99, 97, 98)] * 6
    bbb += [(99, 101, 98, 100.5)]
    bbb += [(101, 102, 100, 101)] * 2
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": aaa, "BBB": bbb}
    )
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 8).assign(symbol="AAA", setup_id=1),
            _setup_row(dates, 5, 100.0, 95.0, 8).assign(symbol="BBB", setup_id=2),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "exposure_levels": (0.2, 0.4),
            "exposure_winners_to_step_up": 1,
            "portfolio_max_open_positions": 2,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    bbb_trade = result.trades.loc[result.trades["symbol"] == "BBB"].iloc[0]
    assert bbb_trade["entry_date"] == dates[6].date()
    assert result.equity.loc[
        result.equity["period_end_date"] == dates[6].date(),
        "feedback_exposure_level",
    ].iloc[0] == 0.4


def test_fractional_winner_does_not_count_as_a_full_feedback_unit():
    dates = pd.bdate_range("2024-01-01", periods=9)
    aaa = [(98, 99, 97, 98)] * 5
    aaa += [(99, 106, 98, 105.5)]
    aaa += [(106, 107, 105, 106)] * 3
    bbb = [(98, 99, 97, 98)] * 6
    bbb += [(99, 101, 98, 100.5)]
    bbb += [(101, 102, 100, 101)] * 2
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": aaa, "BBB": bbb}
    )
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 8).assign(symbol="AAA", setup_id=1),
            _setup_row(dates, 5, 100.0, 95.0, 8).assign(symbol="BBB", setup_id=2),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "exposure_levels": (0.055, 0.2),
            "exposure_winners_to_step_up": 1,
            "portfolio_max_open_positions": 2,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades.loc[result.trades["symbol"] == "AAA", "shares"].iloc[0] == 52
    assert "BBB" not in result.trades["symbol"].values
    assert result.breakout_events.set_index("symbol").loc["BBB", "decision"] == "portfolio_capacity"
    assert result.equity.loc[
        result.equity["period_end_date"] == dates[6].date(),
        "feedback_exposure_level",
    ].iloc[0] == 0.055


def test_fractional_loss_does_not_reduce_exposure_by_a_full_step():
    dates = pd.bdate_range("2024-01-01", periods=10)
    aaa = [(98, 99, 97, 98)] * 5
    aaa += [(99, 101, 98, 100.2), (101, 111, 100, 108)]
    aaa += [(106, 107, 105, 106)] * 3
    bbb = [(98, 99, 97, 98)] * 7
    bbb += [(99, 101, 94, 95)]
    bbb += [(95, 96, 94, 95)] * 2
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": aaa, "BBB": bbb}
    )
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 9).assign(symbol="AAA", setup_id=1),
            _setup_row(dates, 6, 100.0, 95.0, 9).assign(symbol="BBB", setup_id=2),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "exposure_levels": (0.15, 0.18),
            "exposure_winners_to_step_up": 1,
            "portfolio_max_open_positions": 2,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    bbb_trade = result.trades.loc[result.trades["symbol"] == "BBB"].iloc[0]
    assert bbb_trade["shares"] == 69
    assert bbb_trade["exit_reason"] == "stop_entry_day"
    assert result.equity.loc[
        result.equity["period_end_date"] == dates[8].date(),
        "feedback_exposure_level",
    ].iloc[0] == 0.18


def test_breakout_partial_and_breakeven_stop():
    # (open, high, low, close)
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]   # day 5: breakout over pivot 100 -> fill 100
    bars += [(106, 111, 104, 108)]   # day 6: hits 2R target 110 -> partial, stop -> BE
    bars += [(100.5, 101, 99, 100)]  # day 7: low 99 <= BE stop 100 -> stopped out
    bars += [(100, 100.5, 99.5, 100)] * 12
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(
        dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15
    ).assign(setup_type="vcp")

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )
    trades = result.trades

    assert len(trades) == 2
    partial, final = trades.iloc[0], trades.iloc[1]

    # Quantity is fixed before the session at the worst allowed 5% gap:
    # 1% of 100k / (105 worst fill - 95 stop) = 100 shares.
    assert partial["shares"] == 50 and final["shares"] == 50
    assert partial["exit_reason"] == "partial_target"
    assert partial["exit_price"] == 110.0 and partial["pnl"] == 500.0
    # The partial-specific break-even stop is hit intraday on the next bar.
    assert final["exit_reason"] == "stop" and final["exit_price"] == 100.0
    assert final["pnl"] == 0.0
    assert result.metrics["final_equity"] == 100500.0
    assert result.metrics["win_rate"] == 1.0


def test_no_entry_on_excessive_gap():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(107, 108, 106, 107)]  # gaps 7% over pivot -> skip
    bars += [(99, 101, 98, 100)]    # a later pullback cannot revive the setup
    bars += [(107, 108, 106, 107)] * 9
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(
        dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15
    ).assign(setup_type="vcp")

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups,
        _clean_cfg(),
    )
    assert result.trades.empty
    assert result.metrics["final_equity"] == 100000.0
    assert len(result.breakout_events) == 1
    assert result.breakout_events.iloc[0]["decision"] == "excessive_gap"


def test_stop_hit_on_entry_day_exits_same_day():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 94, 95)]  # breakout over 100 AND low breaches stop 95
    bars += [(95, 96, 94, 95)] * 10
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(
        dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15
    ).assign(setup_type="vcp")

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop_entry_day"
    assert trades.iloc[0]["holding_days"] == 0
    assert trades.iloc[0]["pnl"] == 100 * (95.0 - 100.0)
    assert result.metrics["final_equity"] == 99500.0
    event = result.breakout_events.iloc[0]
    assert event["setup_type"] == "vcp"
    assert event["decision"] == "filled"
    assert event["entry_date"] == event["breakout_date"] == dates[5].date()


def test_setup_cannot_fill_on_its_detection_session() -> None:
    bars = [(98, 99, 97, 98)] * 4
    bars += [(99, 101, 98, 100.5)]  # pivot touched on detection day
    bars += [(99, 99.5, 98, 99)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.empty
    assert result.breakout_events.empty


def test_stop_buy_fills_when_high_exactly_touches_trigger() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 100, 98, 99.5)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.iloc[0]["entry_date"] == dates[5].date()
    assert result.trades.iloc[0]["entry_price"] == 100.0
    event = result.breakout_events.iloc[0]
    assert event["breakout_date"] == dates[5].date()
    assert event["entry_date"] == dates[5].date()
    assert event["decision"] == "filled"


def test_breakout_event_contains_previous_session_candidate_snapshot() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 7).assign(
        setup_type="vcp",
        structure_quality_score=20.0,
        tightness_score=15.0,
        prior_advance_score=18.0,
        dryup_ratio=0.6,
        rs_rating=90.0,
        fundamental_score=4.0,
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    event = result.breakout_events.iloc[0]
    assert event["snapshot_date"] == dates[4].date()
    assert event["quality_rank"] == 1
    assert event["setup_age_sessions"] == 0
    assert event["distance_to_pivot_pct"] == pytest.approx(2.0)
    for column in (
        "quality_score",
        "fill_probability",
        "slate_priority",
    ):
        assert np.isfinite(event[column])


def test_breakout_without_setup_id_fails_instead_of_hiding_event() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(
        dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=6
    ).drop(columns="setup_id")

    with pytest.raises(RuntimeError, match="no setup_id"):
        simulate(
            dates,
            close_m.columns,
            open_m,
            high_m,
            low_m,
            close_m,
            setups,
            _clean_cfg(),
        )


def test_same_day_fill_does_not_depend_on_breakout_close() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, np.nan)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.iloc[0]["entry_date"] == dates[5].date()
    assert result.breakout_events.iloc[0]["entry_filled"]


def test_entry_day_two_r_high_arms_break_even_ratchet_for_next_session() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 111, 98, 108)]  # entry 100, reaches 2R on entry day
    bars += [(101, 102, 99, 101)]  # break-even stop 100 becomes active and is hit
    bars += [(106, 107, 104, 106)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "partial_fraction": 0.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades.iloc[0]["exit_reason"] == "stop"
    assert result.trades.iloc[0]["exit_price"] == 100.0


@pytest.mark.parametrize(
    ("qualified_high", "expected_stop"),
    [(111.0, 100.0), (116.0, 105.0), (121.0, 110.0)],
)
def test_progressive_stop_ladder_is_armed_from_completed_high_for_next_day(
    qualified_high, expected_stop
):
    next_open = expected_stop + 1.0
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, qualified_high, 98, qualified_high - 1.0)]
    bars += [(next_open, next_open + 1.0, expected_stop - 1.0, next_open)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)
    cfg = make_cfg(
        **{**_clean_cfg().__dict__, "partial_fraction": 0.0}
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades.iloc[0]["exit_reason"] == "stop"
    assert result.trades.iloc[0]["exit_price"] == expected_stop


def test_market_blocked_breakout_is_recorded_once_and_consumed() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(99, 102, 98, 101)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)
    market_cap = np.array([1.0] * 4 + [0.0, 1.0, 1.0, 1.0])

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        market_exposure_cap=market_cap,
    )

    assert result.trades.empty
    assert len(result.breakout_events) == 1
    assert result.breakout_events.iloc[0]["decision"] == "market_gate_blocked"


def test_state_preroll_does_not_resurrect_preperiod_breakout() -> None:
    bars = [(98, 99, 97, 98)] * 3
    bars += [(99, 101, 98, 100)]  # breakout before measured period
    bars += [(99, 99, 98, 99)]
    bars += [(99, 102, 98, 101)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=1, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        sim_start_idx=5,
        state_start_idx=0,
    )

    assert result.trades.empty
    assert result.breakout_events.empty


def test_state_preroll_survivor_fills_first_oos_day_using_previous_cap() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=1, pivot=100.0, stop=95.0, valid_idx=7)
    market_cap = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        sim_start_idx=5,
        state_start_idx=0,
        market_exposure_cap=market_cap,
    )

    assert result.trades.iloc[0]["entry_date"] == dates[5].date()
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_state_preroll_invalidation_consumes_setup() -> None:
    bars = [(98, 99, 97, 98)] * 3
    bars += [(96, 99, 94, 96)]  # preperiod low invalidates stop 95
    bars += [(98, 99, 97, 98)]
    bars += [(99, 101, 98, 100)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=1, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        sim_start_idx=5,
        state_start_idx=0,
    )

    assert result.trades.empty
    assert result.breakout_events.empty


def test_buy_zone_upper_boundary_is_fillable() -> None:
    bars = [(98, 99, 97, 98)] * 5
    bars += [(105, 106, 104, 105)]
    bars += [(105, 106, 104, 105)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.iloc[0]["entry_price"] == 105.0
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_full_portfolio_records_held_symbol_and_capacity_breakouts() -> None:
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {
            "AAA": [
                (98, 99, 97, 98),
                (99, 101, 98, 100),
                (100, 101, 99, 100),
                (104, 106, 103, 105),
                (105, 106, 104, 105),
            ],
            "BBB": [
                (98, 99, 97, 98),
                (98, 99, 97, 98),
                (98, 99, 97, 98),
                (99, 101, 98, 100),
                (100, 101, 99, 100),
            ],
        }
    )
    setups = pd.DataFrame(
        [
            {
                "setup_id": 1, "symbol": "AAA", "detect_date": dates[0].date(),
                "pivot": 100.0, "last_low": 95.0, "stop_level": 95.0,
                "valid_until": dates[1].date(), "setup_score": 90.0,
            },
            {
                "setup_id": 2, "symbol": "AAA", "detect_date": dates[2].date(),
                "pivot": 105.0, "last_low": 100.0, "stop_level": 100.0,
                "valid_until": dates[4].date(), "setup_score": 80.0,
            },
            {
                "setup_id": 3, "symbol": "BBB", "detect_date": dates[2].date(),
                "pivot": 100.0, "last_low": 95.0, "stop_level": 95.0,
                "valid_until": dates[4].date(), "setup_score": 95.0,
            },
        ]
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
            "partial_fraction": 0.0,
        }
    )

    result = simulate(dates, pd.Index(["AAA", "BBB"]), open_m, high_m, low_m, close_m, setups, cfg)

    decisions = result.breakout_events.set_index("symbol")["decision"].to_dict()
    assert decisions["AAA"] == "existing_position"
    assert decisions["BBB"] == "portfolio_capacity"


def test_gap_below_stop_exits_at_open():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]  # entry at 100
    bars += [(90, 91, 89, 90)]      # gaps below stop 95 -> exit at open 90
    bars += [(90, 91, 89, 90)] * 9
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop_gap"
    assert trades.iloc[0]["exit_price"] == 90.0
    assert trades.iloc[0]["pnl"] == 100 * (90.0 - 100.0)


def test_all_simultaneous_signals_are_taken():
    """No portfolio limit: many symbols triggering on the same day all trade."""
    n_symbols = 12
    dates = pd.bdate_range("2024-01-01", periods=12)
    bars = np.array([(98, 99, 97, 98)] * 5 + [(99, 101, 98, 100.5)] + [(101, 102, 100.5, 101.5)] * 6)
    frames = [
        pd.DataFrame({f"S{i}": bars[:, f] for i in range(n_symbols)}, index=dates)
        for f in range(4)
    ]
    setups = pd.concat(
        [_setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(symbol=f"S{i}", setup_id=i)
         for i in range(n_symbols)],
        ignore_index=True,
    )

    result = simulate(dates, frames[0].columns, *frames, setups, _clean_cfg())

    assert result.metrics["num_positions"] == n_symbols
    assert (result.trades["entry_date"] == dates[5].date()).all()
    assert (result.trades["shares"] == 100).all()  # fixed pre-session sizing


def test_portfolio_mode_limits_simultaneous_positions():
    n_symbols = 12
    dates = pd.bdate_range("2024-01-01", periods=12)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100.0)]
        + [(100, 101, 99, 100.0)] * 6
    )
    frames = [
        pd.DataFrame({f"S{i:02d}": bars[:, f] for i in range(n_symbols)}, index=dates)
        for f in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
                symbol=f"S{i:02d}", setup_id=i, n_contractions=2, dryup_ratio=0.6
            )
            for i in range(n_symbols)
        ],
        ignore_index=True,
    )
    cfg = _clean_cfg()
    cfg = make_cfg(
        **{**cfg.__dict__, "simulation_mode": "portfolio", "portfolio_max_open_positions": 3}
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    assert result.metrics["num_positions"] == 3
    assert result.trades["symbol"].nunique() == 3
    assert result.equity["open_positions"].max() == 3


def test_portfolio_mode_limits_gross_exposure():
    n_symbols = 12
    dates = pd.bdate_range("2024-01-01", periods=12)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100.0)]
        + [(100, 101, 99, 100.0)] * 6
    )
    frames = [
        pd.DataFrame({f"S{i:02d}": bars[:, f] for i in range(n_symbols)}, index=dates)
        for f in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
                symbol=f"S{i:02d}", setup_id=i, n_contractions=2, dryup_ratio=0.6
            )
            for i in range(n_symbols)
        ],
        ignore_index=True,
    )
    cfg = _clean_cfg()
    cfg = make_cfg(
        **{
            **cfg.__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 10,
            "portfolio_max_gross_exposure_pct": 0.4,
            "min_slate_risk_utilization": 0.50,
        }
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    # The whole pre-session slate shares the 40k gross budget proportionally.
    # Seven orders remain above the 50% target floor; an eighth would not.
    assert result.metrics["num_positions"] == 7
    assert sorted(result.trades["shares"].tolist()) == [54, 54, 54, 54, 54, 55, 55]
    decisions = result.breakout_events["decision"].value_counts().to_dict()
    assert decisions == {"filled": 7, "portfolio_capacity": 5}
    assert result.equity["exposure_pct"].max() <= 0.4

    reverse = list(reversed(frames[0].columns))
    reversed_result = simulate(
        dates,
        pd.Index(reverse),
        frames[0][reverse],
        frames[1][reverse],
        frames[2][reverse],
        frames[3][reverse],
        setups.iloc[::-1].reset_index(drop=True),
        cfg,
    )
    expected = result.trades[["symbol", "shares"]].sort_values("symbol").reset_index(drop=True)
    actual = reversed_result.trades[["symbol", "shares"]].sort_values("symbol").reset_index(drop=True)
    pd.testing.assert_frame_equal(actual, expected)


def test_portfolio_mode_prioritizes_quality_metrics():
    dates = pd.bdate_range("2024-01-01", periods=12)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100.0)]
        + [(100, 101, 99, 100.0)] * 6
    )
    frames = [
        pd.DataFrame({"AAA": bars[:, f], "ZZZ": bars[:, f]}, index=dates)
        for f in range(4)
    ]
    weak = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
        symbol="AAA", setup_id=1, n_contractions=2, dryup_ratio=0.6,
        rs_rating=70, stock_industry_rs_rating=70, stock_category_rs_rating=70,
        ibkr_industry_rs_rating=70, ibkr_category_rs_rating=70,
        eps_yoy=0.2, revenue_yoy=0.1,
    )
    strong = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
        symbol="ZZZ", setup_id=2, n_contractions=4, dryup_ratio=0.65,
        rs_rating=99, stock_industry_rs_rating=98, stock_category_rs_rating=97,
        ibkr_industry_rs_rating=96, ibkr_category_rs_rating=95,
        eps_yoy=1.5, revenue_yoy=0.5,
    )
    cfg = _clean_cfg()
    cfg = make_cfg(
        **{**cfg.__dict__, "simulation_mode": "portfolio", "portfolio_max_open_positions": 1}
    )

    result = simulate(
        dates, frames[0].columns, frames[0], frames[1], frames[2], frames[3],
        pd.concat([weak, strong], ignore_index=True), cfg,
    )

    assert result.metrics["num_positions"] == 1
    assert result.trades.iloc[0]["symbol"] == "ZZZ"


def test_slate_minimum_rounds_an_odd_standalone_target_up():
    dates = pd.bdate_range("2024-01-01", periods=8)
    aaa = np.array([(350, 390, 340, 350)] * 5 + [(390, 401, 350, 400)] + [(400, 401, 399, 400)] * 2)
    bbb = np.array([(95, 99, 94, 95)] * 5 + [(99, 101, 95, 100)] + [(100, 101, 99, 100)] * 2)
    frames = [
        pd.DataFrame({"AAA": aaa[:, field], "BBB": bbb[:, field]}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 400.0, 100.0, 7).assign(symbol="AAA", setup_id=1),
            _setup_row(dates, 4, 100.0, 90.0, 7).assign(symbol="BBB", setup_id=2),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "max_buy_zone_pct": 0.0,
            "portfolio_max_open_positions": 2,
            "portfolio_max_gross_exposure_pct": 0.067,
        }
    )

    result = simulate(dates, pd.Index(["AAA", "BBB"]), *frames, setups, cfg)

    shares = result.trades.set_index("symbol")["shares"].to_dict()
    # AAA's standalone risk target is exactly three shares. A 50% floor is
    # therefore two shares, never floor(1.5) = one share.
    assert shares == {"AAA": 2, "BBB": 59}


def test_slate_cash_budget_includes_entry_commission():
    n_symbols = 6
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99.5, 101, 99.5, 100)]
        + [(100, 101, 99.5, 100)] * 2
    )
    frames = [
        pd.DataFrame({f"S{i}": bars[:, field] for i in range(n_symbols)}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 99.0, 7).assign(
                symbol=f"S{i}", setup_id=i + 1
            )
            for i in range(n_symbols)
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "max_buy_zone_pct": 0.0,
            "commission_pct": 0.01,
            "portfolio_max_open_positions": n_symbols,
            "portfolio_max_gross_exposure_pct": 2.0,
        }
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    total_shares = int(result.trades["shares"].sum())
    assert total_shares == 990
    assert total_shares * 100.0 * (1 + cfg.commission_pct) <= cfg.initial_equity


def test_gross_cap_reserves_the_entry_commission_equity_reduction():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99.5, 101, 99.5, 100)]
    bars += [(100, 101, 99.5, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 99.0, 7)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "max_buy_zone_pct": 0.0,
            "commission_pct": 0.01,
            "portfolio_max_open_positions": 1,
            "portfolio_max_gross_exposure_pct": 1.0,
            "exposure_levels": (0.20,),
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades.iloc[0]["shares"] == 199
    entry_row = result.equity.loc[
        result.equity["period_end_date"] == dates[5].date()
    ].iloc[0]
    assert entry_row["exposure_pct"] <= 0.20


def test_capacity_feasible_slate_can_skip_a_large_candidate():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = {
        "TOP": np.array([(95, 99, 94, 95)] * 5 + [(99, 101, 95, 100)] + [(100, 101, 99, 100)] * 2),
        "LARGE": np.array([(98, 99, 97, 98)] * 5 + [(99.5, 101, 99.5, 100)] + [(100, 101, 99.5, 100)] * 2),
        "SMALL": np.array([(45, 49, 44, 45)] * 5 + [(49, 51, 45, 50)] + [(50, 51, 49, 50)] * 2),
    }
    frames = [
        pd.DataFrame({symbol: values[:, field] for symbol, values in bars.items()}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 90.0, 7).assign(
                symbol="TOP", setup_id=1, setup_type="vcp",
                structure_quality_score=25.0, tightness_score=20.0,
                prior_advance_score=25.0, dryup_ratio=0.4, rs_rating=99,
            ),
            _setup_row(dates, 4, 100.0, 99.0, 7).assign(
                symbol="LARGE", setup_id=2, setup_type="vcp",
                structure_quality_score=10.0, tightness_score=10.0,
                prior_advance_score=10.0, dryup_ratio=0.6, rs_rating=70,
            ),
            _setup_row(dates, 4, 50.0, 40.0, 7).assign(
                symbol="SMALL", setup_id=3, setup_type="vcp",
                structure_quality_score=0.0, tightness_score=0.0,
                prior_advance_score=0.0, dryup_ratio=0.9, rs_rating=50,
            ),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "max_buy_zone_pct": 0.0,
            "portfolio_max_open_positions": 3,
            "portfolio_max_gross_exposure_pct": 0.15,
            "min_slate_risk_utilization": 0.50,
        }
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    assert set(result.trades["symbol"]) == {"TOP", "SMALL"}
    decisions = result.breakout_events.set_index("symbol")["decision"].to_dict()
    assert decisions["LARGE"] == "portfolio_capacity"
    assert decisions["SMALL"] == "filled"


def test_capacity_reject_allows_a_smaller_setup_of_the_same_symbol():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(95, 99, 94, 95)] * 5
        + [(99.5, 101, 99.5, 100)]
        + [(100, 101, 99.5, 100)] * 2
    )
    frames = [
        pd.DataFrame({"TOP": bars[:, field], "SAME": bars[:, field]}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 90.0, 7).assign(
                symbol="TOP", setup_id=1, setup_type="vcp",
                structure_quality_score=25.0, tightness_score=20.0,
                prior_advance_score=25.0, dryup_ratio=0.4, rs_rating=99,
            ),
            _setup_row(dates, 4, 100.0, 99.0, 7).assign(
                symbol="SAME", setup_id=2, setup_type="vcp",
                structure_quality_score=12.5, tightness_score=10.0,
                prior_advance_score=12.5, dryup_ratio=0.6, rs_rating=80,
            ),
            _setup_row(dates, 4, 100.0, 90.0, 7).assign(
                symbol="SAME", setup_id=3, setup_type="vcp",
                structure_quality_score=0.0, tightness_score=0.0,
                prior_advance_score=0.0, dryup_ratio=1.0, rs_rating=40,
            ),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "max_buy_zone_pct": 0.0,
            "portfolio_max_open_positions": 3,
            "portfolio_max_gross_exposure_pct": 0.15,
            "min_slate_risk_utilization": 0.50,
        }
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    same_trade = result.trades.loc[result.trades["symbol"] == "SAME"].iloc[0]
    assert same_trade["setup_id"] == 3
    decisions = result.breakout_events.set_index("setup_id")["decision"].to_dict()
    assert decisions[2] == "portfolio_capacity"
    assert decisions[3] == "filled"


def test_portfolio_ranking_uses_continuous_dynamic_context():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100.0)]
        + [(100, 101, 99, 100.0)] * 2
    )
    frames = [
        pd.DataFrame({"TECH": bars[:, field], "LEADER": bars[:, field]}, index=dates)
        for field in range(4)
    ]
    technical_only = _setup_row(dates, 4, 100.0, 95.0, 7).assign(
        symbol="TECH", setup_id=1, setup_score=74.0,
        fundamental_score=0, rs_rating=70,
    )
    contextual_leader = _setup_row(dates, 4, 100.0, 95.0, 7).assign(
        symbol="LEADER", setup_id=2, setup_score=71.0,
        fundamental_score=6, rs_rating=99,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
        }
    )

    result = simulate(
        dates,
        frames[0].columns,
        frames[0],
        frames[1],
        frames[2],
        frames[3],
        pd.concat([technical_only, contextual_leader], ignore_index=True),
        cfg,
    )

    assert result.trades["symbol"].unique().tolist() == ["LEADER"]


def test_current_session_context_cannot_change_same_session_selection():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)] * 2
    )
    frames = [
        pd.DataFrame({"AAA": bars[:, field], "BBB": bars[:, field]}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 7).assign(
                symbol=symbol, setup_id=setup_id, setup_type="vcp",
                structure_quality_score=15.0, tightness_score=12.0,
                prior_advance_score=15.0, dryup_ratio=0.6,
            )
            for setup_id, symbol in enumerate(("AAA", "BBB"), start=1)
        ],
        ignore_index=True,
    )
    rs = pd.DataFrame(50.0, index=dates, columns=frames[0].columns)
    rs.loc[dates[4], ["AAA", "BBB"]] = [99.0, 1.0]
    rs.loc[dates[5], ["AAA", "BBB"]] = [1.0, 99.0]
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
        }
    )

    result = simulate(
        dates,
        frames[0].columns,
        *frames,
        setups,
        cfg,
        candidate_context={"rs_rating": rs},
    )

    assert result.trades["symbol"].unique().tolist() == ["AAA"]
    events = result.breakout_events.set_index("symbol")
    assert events.loc["AAA", "snapshot_date"] == dates[4].date()
    assert events.loc["AAA", "quality_rank"] == 1
    assert events.loc["BBB", "quality_rank"] == 1


def test_prior_day_dynamic_context_can_change_portfolio_rank():
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)] * 2
    )
    frames = [
        pd.DataFrame({"AAA": bars[:, field], "BBB": bars[:, field]}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 7).assign(
                symbol=symbol, setup_id=setup_id, setup_type="vcp",
                structure_quality_score=15.0, tightness_score=12.0,
                prior_advance_score=15.0, dryup_ratio=0.6,
            )
            for setup_id, symbol in enumerate(("AAA", "BBB"), start=1)
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
        }
    )

    def selected_with_previous_rs(aaa: float, bbb: float) -> str:
        rs = pd.DataFrame(50.0, index=dates, columns=frames[0].columns)
        rs.loc[dates[4], ["AAA", "BBB"]] = [aaa, bbb]
        result = simulate(
            dates,
            frames[0].columns,
            *frames,
            setups,
            cfg,
            candidate_context={"rs_rating": rs},
        )
        return str(result.trades.iloc[0]["symbol"])

    assert selected_with_previous_rs(99.0, 1.0) == "AAA"
    assert selected_with_previous_rs(1.0, 99.0) == "BBB"


def test_temporary_pre_period_eligibility_failure_does_not_destroy_structure():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setup = _setup_row(dates, 2, 100.0, 95.0, 7).assign(
        setup_id=1,
        setup_type="vcp",
    )
    trend = pd.DataFrame(True, index=dates, columns=close_m.columns)
    # This completed-session failure governs session 4, before the measured
    # period. It blocks that day but does not damage the chart structure.
    trend.iloc[3, 0] = False

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        make_cfg(**{**_clean_cfg().__dict__, "simulation_mode": "portfolio"}),
        sim_start_idx=5,
        state_start_idx=0,
        candidate_context={"trend_template_pass": trend},
    )

    assert result.metrics["num_positions"] == 1
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_portfolio_uses_trend_as_soft_feature_not_hard_gate():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setup = _setup_row(dates, 4, 100.0, 95.0, 7).assign(setup_id=1)
    trend = pd.DataFrame(True, index=dates, columns=close_m.columns)
    trend.iloc[4, 0] = False

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        make_cfg(**{**_clean_cfg().__dict__, "simulation_mode": "portfolio"}),
        candidate_context={"trend_template_pass": trend},
    )

    assert not result.trades.empty
    assert len(result.breakout_events) == 1
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_independent_first_touch_is_not_filtered_by_daily_trend_context():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setup = _setup_row(dates, 4, 100.0, 95.0, 7).assign(setup_id=1)
    trend = pd.DataFrame(False, index=dates, columns=close_m.columns)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        _clean_cfg(),
        candidate_context={"trend_template_pass": trend},
    )

    assert result.metrics["num_positions"] == 1
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_pre_session_slate_does_not_nominate_an_ex_post_trigger():
    bars_a = (
        [(98, 99, 97, 98)] * 5
        + [(98, 99, 97, 98)]
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)] * 3
    )
    bars_b = (
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)] * 3
    )
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": bars_a, "BBB": bars_b}
    )
    top = _setup_row(dates, 4, 100.0, 95.0, 9).assign(
        symbol="AAA", setup_id=1, dryup_ratio=0.65, rs_rating=99
    )
    lower = _setup_row(dates, 4, 100.0, 95.0, 9).assign(
        symbol="BBB", setup_id=2, dryup_ratio=0.60, rs_rating=70
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
        }
    )

    setups = pd.concat([lower, top], ignore_index=True)
    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    # AAA reserved the only slot before day 5. It did not trigger, so BBB may
    # not be nominated after observing BBB's high. BBB's breakout is missed
    # and consumed; AAA remains valid and fills on day 6.
    assert result.trades["symbol"].unique().tolist() == ["AAA"]
    assert result.trades.iloc[0]["entry_date"] == dates[6].date()

    reverse = ["BBB", "AAA"]
    reversed_result = simulate(
        dates,
        pd.Index(reverse),
        open_m[reverse],
        high_m[reverse],
        low_m[reverse],
        close_m[reverse],
        setups.iloc[::-1].reset_index(drop=True),
        cfg,
    )
    cols = ["symbol", "entry_date", "shares", "exit_reason"]
    pd.testing.assert_frame_equal(
        result.trades[cols].reset_index(drop=True),
        reversed_result.trades[cols].reset_index(drop=True),
    )


def test_portfolio_sizing_uses_previous_close_equity():
    bars_a = (
        [(98, 99, 97, 98)] * 3
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)]
        + [(199, 201, 198, 200)]
        + [(200, 201, 199, 200)] * 2
    )
    bars_b = (
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)] * 2
    )
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": bars_a, "BBB": bars_b}
    )
    setups = pd.concat(
        [
            _setup_row(dates, 2, 100.0, 95.0, 7).assign(
                symbol="AAA", setup_id=1
            ),
            _setup_row(dates, 4, 100.0, 95.0, 7).assign(
                symbol="BBB", setup_id=2
            ),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 2,
            "portfolio_max_gross_exposure_pct": 2.0,
            "partial_at_r": 100.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    b_trade = result.trades.loc[result.trades["symbol"] == "BBB"].iloc[0]
    # AAA doubles at day 5's close, but that close was unavailable when BBB's
    # stop order triggered during day 5. Previous-close equity gives 100 shares;
    # using the unavailable current close would have produced 110.
    assert b_trade["entry_date"] == dates[5].date()
    assert b_trade["shares"] == 100


def test_pre_session_quantity_is_unchanged_by_an_allowed_gap():
    def run(entry_open: float):
        bars = [(98, 99, 97, 98)] * 5
        bars += [(entry_open, max(106.0, entry_open), 98, 101)]
        bars += [(101, 102, 100, 101)] * 2
        dates, open_m, high_m, low_m, close_m = _matrices(bars)
        setups = _setup_row(dates, 4, 100.0, 95.0, 7)
        return simulate(
            dates,
            close_m.columns,
            open_m,
            high_m,
            low_m,
            close_m,
            setups,
            _clean_cfg(),
        )

    pivot_fill = run(99.0).trades.iloc[0]
    gap_fill = run(104.0).trades.iloc[0]

    assert pivot_fill["entry_price"] == 100.0
    assert gap_fill["entry_price"] == 104.0
    assert pivot_fill["shares"] == gap_fill["shares"] == 100


def test_same_day_exit_does_not_release_a_slot_or_revive_missed_setup():
    bars_a = (
        [(98, 99, 97, 98)] * 3
        + [(99, 101, 98, 100)]
        + [(100, 101, 98, 100)]
        + [(96, 97, 94, 95)]
        + [(95, 96, 94, 95)] * 2
    )
    bars_b = (
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(99, 101, 98, 100)]
        + [(100, 101, 99, 100)]
    )
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"AAA": bars_a, "BBB": bars_b}
    )
    setups = pd.concat(
        [
            _setup_row(dates, 2, 100.0, 95.0, 7).assign(
                symbol="AAA", setup_id=1
            ),
            _setup_row(dates, 4, 100.0, 95.0, 7).assign(
                symbol="BBB", setup_id=2
            ),
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "portfolio_max_open_positions": 1,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades["symbol"].unique().tolist() == ["AAA"]
    assert result.trades.iloc[0]["exit_date"] == dates[5].date()
    assert result.trades.iloc[0]["exit_reason"] == "stop"


def test_independent_first_touch_keeps_coexisting_setups_for_same_symbol():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 97, 100)]
    bars += [(109, 111, 100, 110)]
    bars += [(110, 111, 109, 110)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    old = _setup_row(dates, 3, 100.0, 95.0, 8).assign(setup_id=1)
    new = _setup_row(dates, 4, 110.0, 95.0, 8).assign(setup_id=2)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        pd.concat([new, old], ignore_index=True),
        _clean_cfg(),
    )

    assert result.metrics["num_positions"] == 2
    events = result.breakout_events.sort_values("setup_id").reset_index(drop=True)
    assert events["setup_id"].tolist() == [1, 2]
    assert events["decision"].tolist() == ["filled", "filled"]
    assert events["entry_date"].tolist() == [dates[5].date(), dates[6].date()]
    assert set(result.trades["setup_id"]) == {1, 2}


def test_new_setup_can_reenter_symbol_after_prior_position_is_closed():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]       # first setup fills
    bars += [(96, 99, 94, 96)]         # first position stops
    bars += [(104, 106, 103, 105)]     # new setup fills next session
    bars += [(106, 107, 105, 106)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    first = _setup_row(dates, 4, 100.0, 95.0, 6).assign(setup_id=1)
    second = _setup_row(dates, 6, 105.0, 100.0, 9).assign(setup_id=2)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        pd.concat([first, second], ignore_index=True),
        _clean_cfg(),
    )

    final_trades = result.trades.loc[result.trades["leg"] == "final"]
    assert final_trades["setup_id"].tolist() == [1, 2]
    assert final_trades["position_id"].tolist() == [1, 2]
    assert final_trades.iloc[1]["entry_date"] == dates[7].date()


def test_known_next_open_exit_allows_later_same_session_reentry():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.2)]      # first setup fills
    bars += [(100.2, 102, 99, 100.2)]
    bars += [(100.2, 102, 99, 99.8)]    # schedules time stop next open
    bars += [(104, 106, 103, 105)]       # old exits open; new setup fills later
    bars += [(106, 107, 105, 106)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    first = _setup_row(dates, 4, 100.0, 95.0, 7).assign(setup_id=1)
    second = _setup_row(dates, 7, 105.0, 100.0, 9).assign(setup_id=2)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "partial_fraction": 0.0,
            "time_stop_sessions": 2,
            "time_stop_min_r": 1.0,
        }
    )

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        pd.concat([first, second], ignore_index=True),
        cfg,
    )

    final_trades = result.trades.loc[result.trades["leg"] == "final"]
    assert final_trades["setup_id"].tolist() == [1, 2]
    assert final_trades.iloc[0]["exit_reason"] == "time_stop"
    assert final_trades.iloc[0]["exit_date"] == dates[8].date()
    assert final_trades.iloc[1]["entry_date"] == dates[8].date()


def test_setup_is_consumed_when_its_base_invalidation_level_breaks():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(97, 99, 94, 96)]
    bars += [(99, 101, 98, 100)]
    bars += [(100, 101, 99, 100)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8).assign(stop_level=90.0)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    # The structural last low invalidates the setup even if the configured
    # protective stop happens to be lower.
    assert result.trades.empty


def test_open_below_setup_invalidation_cancels_later_intraday_breakout():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(94, 101, 93, 100)]  # open damage is known before the later high
    bars += [(99, 101, 98, 100)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.empty


def test_partial_target_and_new_breakeven_stop_in_one_bar_take_adverse_path():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(105, 111, 99, 105)]
    bars += [(105, 106, 104, 105)] * 2
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades["exit_reason"].tolist() == [
        "partial_target",
        "stop_after_partial",
    ]
    assert result.trades["shares"].tolist() == [50, 50]
    assert result.trades["exit_price"].tolist() == [110.0, 100.0]


def test_open_above_partial_target_fills_before_later_stop():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(112, 113, 94, 100)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    # The open is already above the 110 target, so chronology is known: half
    # exits at 112 before the later low stops the remainder at break-even.
    assert result.trades["exit_reason"].tolist() == [
        "partial_target",
        "stop_after_partial",
    ]
    assert result.trades["shares"].tolist() == [50, 50]
    assert result.trades["exit_price"].tolist() == [112.0, 100.0]


def test_zero_share_partial_does_not_raise_stop_for_existing_position():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(105, 111, 99, 105)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)
    cfg = make_cfg(**{**_clean_cfg().__dict__, "risk_pct": 0.0001})

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    # Worst-gap sizing creates one share. floor(1 * 0.5) is zero, therefore no
    # partial fill exists and the low at 99 must not activate a phantom BE stop.
    assert result.trades["exit_reason"].tolist() == ["eod"]
    assert result.trades.iloc[0]["shares"] == 1
    assert result.trades.iloc[0]["exit_price"] == 105.0


def test_zero_share_partial_does_not_raise_stop_on_entry_bar():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 111, 98, 105)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 5)
    cfg = make_cfg(**{**_clean_cfg().__dict__, "risk_pct": 0.0001})

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades["exit_reason"].tolist() == ["eod"]
    assert result.trades.iloc[0]["shares"] == 1
    assert result.trades.iloc[0]["exit_price"] == 105.0


def test_strong_volume_dryup_has_no_artificial_lower_rejection():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]
    bars += [(101, 102, 100.5, 101.5)] * 6
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
        dryup_ratio=0.4
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.metrics["num_positions"] == 1
    assert result.metrics["final_equity"] > result.metrics["initial_equity"]


def test_portfolio_uses_known_bad_growth_as_feature_not_hard_gate():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]
    bars += [(101, 102, 100.5, 101.5)] * 6
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
        dryup_ratio=0.65, eps_yoy=0.1, revenue_yoy=0.05
    )

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        make_cfg(
            **{
                **_clean_cfg().__dict__,
                "simulation_mode": "portfolio",
            }
        ),
    )

    assert not result.trades.empty
    assert result.breakout_events.iloc[0]["decision"] == "filled"


def test_v5_simulation_rejects_missing_continuity_contract():
    bars = [(98, 99, 97, 98)] * 6
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 5).assign(
        price_continuity_segment=1
    )

    with pytest.raises(ValueError, match="requires continuity_segment_m"):
        _simulate(
            dates,
            close_m.columns,
            open_m,
            high_m,
            low_m,
            close_m,
            setups,
            _clean_cfg(),
        )


def test_old_segment_setup_is_cancelled_before_break_session_ordering():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 105, 97, 103), (103, 106, 101, 104)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    continuity = pd.DataFrame(1, index=dates, columns=close_m.columns)
    continuity.iloc[5:, 0] = 2
    setup = _setup_row(dates, 4, 100.0, 95.0, 6).assign(
        setup_type="vcp", price_continuity_segment=1
    )

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        _clean_cfg(),
        continuity_segment_m=continuity,
    )

    assert result.trades.empty
    assert result.breakout_events.empty


def test_open_position_exits_at_break_session_open_without_backdating():
    bars = [(98, 99, 97, 98)] * 5
    bars += [
        (99, 101, 98, 100),
        (101, 104, 99, 103),
        (70, 72, 68, 71),
        (71, 73, 70, 72),
    ]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    continuity = pd.DataFrame(1, index=dates, columns=close_m.columns)
    continuity.iloc[7:, 0] = 2
    setup = _setup_row(dates, 4, 100.0, 95.0, 8).assign(
        setup_type="vcp", price_continuity_segment=1
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "partial_fraction": 0.0,
            "failed_breakout_exit_enable": False,
            "time_stop_sessions": 999,
        }
    )

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        cfg,
        continuity_segment_m=continuity,
    )

    final = result.trades.iloc[-1]
    assert final["exit_reason"] == "continuity_break"
    assert final["exit_date"] == dates[7].date()
    assert final["exit_price"] == 70.0


def test_trailing_average_restarts_inside_new_continuity_segment():
    bars = [(200, 202, 198, 200)] * 5
    bars += [
        (98, 99, 97, 98),
        (99, 101, 96, 96),
        (110, 111, 100, 110),
        (111, 112, 109, 111),
    ]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    continuity = pd.DataFrame(1, index=dates, columns=close_m.columns)
    continuity.iloc[5:, 0] = 2
    setup = _setup_row(dates, 5, 100.0, 95.0, 8).assign(
        setup_type="vcp", price_continuity_segment=2
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "trail_ma_days": 3,
            "partial_fraction": 0.0,
            "failed_breakout_exit_enable": False,
            "time_stop_sessions": 999,
        }
    )

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        cfg,
        continuity_segment_m=continuity,
    )

    assert result.trades.iloc[-1]["exit_reason"] == "eod"
    assert result.trades.iloc[-1]["exit_date"] == dates[-1].date()


def test_tight_shelf_is_first_touch_research_only_not_portfolio_order():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100), (100, 102, 99, 101)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setup = _setup_row(dates, 4, 100.0, 95.0, 6).assign(
        setup_type="tight_shelf"
    )

    independent = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        _clean_cfg(),
    )
    portfolio = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        make_cfg(**{**_clean_cfg().__dict__, "simulation_mode": "portfolio"}),
    )

    assert not independent.trades.empty
    assert portfolio.trades.empty
    assert portfolio.breakout_events.iloc[0]["decision"] == (
        "setup_class_research_only"
    )


def test_power_play_one_r_ratchet_arms_only_for_following_session():
    bars = [(98, 99, 97, 98)] * 5
    bars += [
        (99, 105, 98, 104),
        (101, 102, 99, 101),
        (101, 102, 100, 101),
    ]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    base_setup = _setup_row(dates, 4, 100.0, 95.0, 7)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "partial_fraction": 0.0,
            "failed_breakout_exit_enable": False,
            "time_stop_sessions": 999,
        }
    )

    power = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        base_setup.assign(setup_type="power_play"),
        cfg,
    )
    vcp = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        base_setup.assign(setup_type="vcp"),
        cfg,
    )

    assert power.trades.iloc[-1]["exit_reason"] == "stop"
    assert power.trades.iloc[-1]["exit_date"] == dates[6].date()
    assert power.trades.iloc[-1]["exit_price"] == 100.0
    assert vcp.trades.iloc[-1]["exit_reason"] == "eod"


def test_first_touch_exports_labels_and_portfolio_does_not_duplicate_them():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100), (101, 103, 100, 102)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setup = _setup_row(dates, 4, 100.0, 95.0, 6).assign(setup_type="vcp")

    first_touch = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        _clean_cfg(),
    )
    portfolio = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setup,
        make_cfg(**{**_clean_cfg().__dict__, "simulation_mode": "portfolio"}),
        quality_labels=first_touch.quality_labels,
        fill_labels=first_touch.fill_labels,
        online_calibration=False,
    )

    assert len(first_touch.quality_labels) == 1
    assert len(first_touch.fill_labels) == 1
    assert first_touch.fill_labels[0].filled
    assert portfolio.quality_labels == first_touch.quality_labels
    assert portfolio.fill_labels == first_touch.fill_labels


def test_failed_breakout_exits_next_open():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.0)]       # entry at 100
    bars += [(100.2, 101, 98, 100.2)]    # day 1: still alive
    bars += [(99.8, 100.5, 98, 99.8)]    # day 2
    bars += [(99.7, 100.4, 98, 99.7)]    # day 3
    bars += [(99.9, 100.4, 98, 99.9)]    # day 4
    bars += [(99.8, 100.4, 98, 99.8)]    # day 5: failed, exit next open
    bars += [(99.5, 100.0, 98, 99.6)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=11)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "failed_breakout_exit_enable": True,
            "failed_breakout_days": 5,
            "failed_breakout_min_r": 0.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "failed_breakout"
    assert trades.iloc[0]["exit_date"] == dates[11].date()
    assert trades.iloc[0]["exit_price"] == 99.5
    assert trades.iloc[0]["pnl"] == -50.0


def test_failed_breakout_waits_for_pivot_loss():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.2)]       # entry at 100
    bars += [(100.2, 101, 98, 100.2)] * 9
    bars += [(100.2, 101, 98, 99.8)]     # stale and back below pivot
    bars += [(99.5, 100.0, 98, 99.6)]    # exit next open
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=16)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "failed_breakout_exit_enable": True,
            "failed_breakout_days": 10,
            "failed_breakout_min_r": -0.5,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "failed_breakout"
    assert trades.iloc[0]["exit_date"] == dates[16].date()


def test_failed_breakout_does_not_exit_while_above_pivot():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.2)]       # entry at 100
    bars += [(100.2, 101, 98, 100.2)] * 10
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "failed_breakout_exit_enable": True,
            "failed_breakout_days": 5,
            "failed_breakout_min_r": -0.5,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "eod"


def test_market_gate_blocks_entries_but_not_exits():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]   # breakout day 5 (gate on) -> entry at 100
    bars += [(101, 102, 100, 101)]   # day 6: gate goes off, position keeps running
    bars += [(96, 97, 94, 95)]       # day 7: stop 95 hit while gate off -> exit works
    bars += [(99, 101, 98, 100.5)]   # day 8: second breakout, but gate off -> no entry
    bars += [(99, 101, 98, 100.5)] * 8
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = pd.concat(
        [
            _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=6),
            _setup_row(dates, detect_idx=6, pivot=100.0, stop=95.0, valid_idx=15).assign(setup_id=2),
        ],
        ignore_index=True,
    )
    market_on = np.array([True] * 6 + [False] * (len(dates) - 6))

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups,
        _clean_cfg(), market_exposure_cap=market_on.astype(float),
    )
    trades = result.trades

    assert len(trades) == 1  # only the first breakout traded
    assert trades.iloc[0]["exit_reason"] == "stop"  # exit executed despite gate off
    assert trades.iloc[0]["entry_date"] == dates[5].date()


def test_market_gate_uses_previous_close_not_breakout_day_close():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(99, 101, 98, 100)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8)

    turns_on_at_breakout_close = np.array(
        [False] * 5 + [True] * (len(dates) - 5)
    )
    blocked = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        market_exposure_cap=turns_on_at_breakout_close.astype(float),
    )
    assert blocked.trades.empty

    turns_off_at_breakout_close = np.array(
        [True] * 5 + [False] * (len(dates) - 5)
    )
    allowed = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        _clean_cfg(),
        market_exposure_cap=turns_off_at_breakout_close.astype(float),
    )
    assert allowed.trades.iloc[0]["entry_date"] == dates[5].date()


def test_market_exposure_cap_limits_portfolio_order_size():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(101, 102, 100, 101)] * 3
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 8)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "exposure_levels": (1.0,),
        }
    )
    market_cap = np.full(len(dates), 0.25)

    result = simulate(
        dates,
        close_m.columns,
        open_m,
        high_m,
        low_m,
        close_m,
        setups,
        cfg,
        market_exposure_cap=market_cap,
    )

    # The 25% cap is aggregate gross capacity. It must not shrink the normal
    # 1%-risk order itself while that order fits below the portfolio cap.
    assert result.trades.iloc[0]["shares"] == 100
    entry_equity = result.equity.loc[
        result.equity["period_end_date"] == dates[5].date()
    ].iloc[0]
    assert entry_equity["market_exposure_cap"] == 0.25
    assert entry_equity["entry_exposure_limit"] == 0.25


def test_exposure_cap_does_not_reduce_configured_portfolio_slots():
    n_symbols = 3
    dates = pd.bdate_range("2024-01-01", periods=8)
    bars = np.array(
        [(98, 99, 97, 98)] * 5
        + [(99, 101, 98, 100)]
        + [(101, 102, 100, 101)] * 2
    )
    frames = [
        pd.DataFrame({f"S{i}": bars[:, field] for i in range(n_symbols)}, index=dates)
        for field in range(4)
    ]
    setups = pd.concat(
        [
            _setup_row(dates, 4, 100.0, 95.0, 7).assign(
                symbol=f"S{i}", setup_id=i + 1
            )
            for i in range(n_symbols)
        ],
        ignore_index=True,
    )
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "partial_fraction": 0.0,
            "portfolio_max_open_positions": 3,
            "exposure_levels": (1.0,),
        }
    )

    result = simulate(
        dates,
        frames[0].columns,
        *frames,
        setups,
        cfg,
        market_exposure_cap=np.full(len(dates), 0.25),
    )

    # All three configured slots share the single aggregate 25% reservation
    # budget proportionally; no candidate becomes a residual dust order.
    assert result.trades["symbol"].nunique() == 3
    assert sorted(result.trades["shares"].tolist()) == [79, 79, 80]
    assert result.equity["exposure_pct"].max() <= 0.25


def test_regime_gate_blocks_entries_but_not_exits():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]   # breakout day 5 (gate on) -> entry at 100
    bars += [(101, 102, 100, 101)]   # day 6: gate goes off, position keeps running
    bars += [(96, 97, 94, 95)]       # day 7: stop 95 hit while gate off -> exit works
    bars += [(99, 101, 98, 100.5)]   # day 8: second breakout, but gate off -> no entry
    bars += [(99, 101, 98, 100.5)] * 8
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = pd.concat(
        [
            _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=6),
            _setup_row(dates, detect_idx=6, pivot=100.0, stop=95.0, valid_idx=15).assign(setup_id=2),
        ],
        ignore_index=True,
    )
    regime_entry_allowed = np.array([True] * 6 + [False] * (len(dates) - 6))

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups,
        _clean_cfg(), regime_entry_allowed=regime_entry_allowed,
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop"
    assert trades.iloc[0]["entry_date"] == dates[5].date()


def test_open_position_closed_at_end():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]
    bars += [(101, 102, 100.5, 101.5)] * 6
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )
    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_reason"] == "eod"
    assert result.trades.iloc[0]["exit_price"] == 101.5


def test_forced_eod_close_updates_final_equity_with_costs():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 102, 98, 100)]
    bars += [(104, 106, 103, 105)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "simulation_mode": "portfolio",
            "commission_pct": 0.01,
            "slippage_pct": 0.01,
            "partial_at_r": 100.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    trade = result.trades.iloc[0]
    entry_fill = 100.0 * (1 + cfg.slippage_pct)
    exit_fill = 105.0 * (1 - cfg.slippage_pct)
    expected_pnl = int(trade["shares"]) * (
        exit_fill * (1 - cfg.commission_pct)
        - entry_fill * (1 + cfg.commission_pct)
    )
    expected_equity = cfg.initial_equity + expected_pnl

    assert trade["exit_reason"] == "eod"
    assert trade["entry_price"] == 101.0
    assert trade["exit_price"] == 103.95
    assert abs(trade["pnl"] - expected_pnl) <= 0.01
    assert abs(result.metrics["final_equity"] - expected_equity) <= 0.01
    assert abs(result.equity.iloc[-1]["equity"] - expected_equity) <= 0.01
    assert result.equity.iloc[-1]["open_positions"] == 0
    assert result.equity.iloc[-1]["exposure_pct"] == 0.0


def test_missing_final_symbol_bar_does_not_invent_an_eod_fill():
    test_bars = [(98, 99, 97, 98)] * 5
    test_bars += [(99, 102, 98, 100)]
    test_bars += [(np.nan, np.nan, np.nan, np.nan)]
    other_bars = [(50, 51, 49, 50)] * len(test_bars)
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"TEST": test_bars, "OTHER": other_bars}
    )
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.empty
    assert result.equity.iloc[-1]["open_positions"] == 1
    assert result.equity.iloc[-1]["exposure_pct"] > 0
    assert result.metrics["num_positions"] == 1
    assert len(result.breakout_events) == 1
    event = result.breakout_events.iloc[0]
    assert event["entry_filled"]
    assert event["entry_date"] == event["breakout_date"] == dates[5].date()


def test_open_end_position_with_partial_is_excluded_from_closed_trade_metrics():
    test_bars = [(98, 99, 97, 98)] * 5
    test_bars += [(99, 101, 98, 100)]
    test_bars += [(105, 111, 104, 108)]
    test_bars += [(np.nan, np.nan, np.nan, np.nan)]
    other_bars = [(50, 51, 49, 50)] * len(test_bars)
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"TEST": test_bars, "OTHER": other_bars}
    )
    setups = _setup_row(dates, 4, 100.0, 95.0, 7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades["leg"].tolist() == ["partial"]
    assert result.equity.iloc[-1]["open_positions"] == 1
    assert result.metrics["num_positions"] == 1
    assert result.metrics["win_rate"] is None
    assert result.metrics["profit_factor"] is None
    assert result.metrics["avg_r_multiple"] is None


def test_partial_metrics_are_aggregated_at_position_granularity():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100)]
    bars += [(105, 111, 96, 108)]
    bars += [(100, 101, 94, 95)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 7)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "partial_fraction": 0.25,
            "breakeven_after_partial": False,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    assert result.trades["shares"].tolist() == [25, 75]
    # The +2R high arms a break-even-or-better ratchet for the next day even
    # when partial-specific break-even behavior is disabled.
    assert result.trades["r_multiple"].tolist() == [2.0, 0.0]
    assert result.metrics["win_rate"] == 1.0
    assert result.metrics["profit_factor"] is None
    assert result.metrics["avg_r_multiple"] == 0.5


def test_active_setup_is_consumed_by_a_missing_symbol_session():
    test_bars = [(98, 99, 97, 98)] * 5
    test_bars += [(np.nan, np.nan, np.nan, np.nan)]
    test_bars += [(99, 101, 98, 100)]
    test_bars += [(100, 101, 99, 100)]
    other_bars = [(50, 51, 49, 50)] * len(test_bars)
    dates, open_m, high_m, low_m, close_m = _multi_matrices(
        {"TEST": test_bars, "OTHER": other_bars}
    )
    setups = _setup_row(dates, 4, 100.0, 95.0, 7)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.empty
    assert result.metrics["num_positions"] == 0


def test_independent_equity_accrues_entry_commission_while_position_is_open():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 102, 98, 100)]
    bars += [(100, 101, 99, 100)]
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, 4, 100.0, 95.0, 6)
    cfg = make_cfg(
        **{
            **_clean_cfg().__dict__,
            "commission_pct": 0.01,
            "partial_at_r": 100.0,
        }
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, cfg
    )

    entry_day = result.equity.iloc[-2]
    shares = int(result.trades.iloc[0]["shares"])
    expected = cfg.initial_equity - shares * 100.0 * cfg.commission_pct
    assert entry_day["open_positions"] == 1
    assert entry_day["equity"] == expected
