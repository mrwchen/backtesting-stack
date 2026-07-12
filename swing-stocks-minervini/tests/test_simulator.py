import numpy as np
import pandas as pd

from src.simulator import simulate
from tests.util import make_cfg


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
        time_stop_sessions=999, profit_protection_trigger_r=999.0,
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
    bars += [(100.2, 102, 99, 100.5)] * 3
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


def test_progressive_exposure_steps_up_only_after_two_closed_winners():
    dates = pd.bdate_range("2024-01-01", periods=14)
    symbols = ["AAA", "BBB", "CCC"]
    fields = {
        symbol: np.array([(98.0, 99.0, 97.0, 98.0)] * len(dates))
        for symbol in symbols
    }
    for symbol, entry in (("AAA", 5), ("BBB", 8), ("CCC", 11)):
        fields[symbol][entry] = (99.0, 101.0, 98.0, 100.2)
        fields[symbol][entry + 1] = (101.0, 111.0, 100.0, 108.0)
        fields[symbol][entry + 2] = (102.0, 103.0, 101.0, 102.0)
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
            "max_buy_zone_pct": 0.02, "profit_protection_trigger_r": 2.0,
            "profit_protection_lock_r": 0.5,
            "exposure_levels": (0.25, 0.5, 0.75, 1.0),
            "portfolio_max_open_positions": 8,
        }
    )
    result = simulate(dates, pd.Index(symbols), *frames, setups, cfg)
    final_legs = result.trades[result.trades["leg"] == "final"].set_index("symbol")
    assert final_legs.loc["AAA", "pnl"] > 0
    assert final_legs.loc["BBB", "pnl"] > 0
    assert final_legs.loc["CCC", "shares"] > final_legs.loc["BBB", "shares"]
    assert result.equity.loc[result.equity["period_end_date"] == dates[11].date(), "exposure_level"].iloc[0] == 0.5


def test_breakout_partial_and_breakeven_stop():
    # (open, high, low, close)
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]   # day 5: breakout over pivot 100 -> fill 100
    bars += [(106, 111, 104, 108)]   # day 6: hits 2R target 110 -> partial, stop -> BE
    bars += [(100.5, 101, 99, 100)]  # day 7: low 99 <= BE stop 100 -> stopped out
    bars += [(100, 100.5, 99.5, 100)] * 12
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15)

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
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups,
        _clean_cfg(),
    )
    assert result.trades.empty
    assert result.metrics["final_equity"] == 100000.0


def test_stop_hit_on_entry_day_exits_same_day():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 94, 95)]  # breakout over 100 AND low breaches stop 95
    bars += [(95, 96, 94, 95)] * 10
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=15)

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )
    trades = result.trades

    assert len(trades) == 1
    assert trades.iloc[0]["exit_reason"] == "stop_entry_day"
    assert trades.iloc[0]["holding_days"] == 0
    assert trades.iloc[0]["pnl"] == 100 * (95.0 - 100.0)
    assert result.metrics["final_equity"] == 99500.0


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
        }
    )

    result = simulate(dates, frames[0].columns, *frames, setups, cfg)

    # Three full 10.5k reservations and one reduced 8.4k reservation fit in
    # the 40k gross budget. No fifth order may be nominated ex post.
    assert result.metrics["num_positions"] == 4
    assert sorted(result.trades["shares"].tolist()) == [80, 100, 100, 100]
    assert result.equity["exposure_pct"].max() <= 0.4


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


def test_new_setup_supersedes_an_older_setup_for_the_same_symbol():
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

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["setup_id"] == 2
    assert result.trades.iloc[0]["pivot"] == 110.0
    assert result.trades.iloc[0]["entry_date"] == dates[6].date()


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


def test_dead_dryup_setup_is_skipped():
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

    assert result.trades.empty
    assert result.metrics["final_equity"] == 100000.0


def test_known_bad_growth_setup_is_skipped():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(99, 101, 98, 100.5)]
    bars += [(101, 102, 100.5, 101.5)] * 6
    dates, open_m, high_m, low_m, close_m = _matrices(bars)
    setups = _setup_row(dates, detect_idx=4, pivot=100.0, stop=95.0, valid_idx=10).assign(
        dryup_ratio=0.65, eps_yoy=0.1, revenue_yoy=0.05
    )

    result = simulate(
        dates, close_m.columns, open_m, high_m, low_m, close_m, setups, _clean_cfg()
    )

    assert result.trades.empty
    assert result.metrics["final_equity"] == 100000.0


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
        _clean_cfg(), market_on=market_on,
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
        market_on=turns_on_at_breakout_close,
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
        market_on=turns_off_at_breakout_close,
    )
    assert allowed.trades.iloc[0]["entry_date"] == dates[5].date()


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
    assert result.trades["r_multiple"].tolist() == [2.0, -1.0]
    assert result.metrics["win_rate"] == 0.0
    assert result.metrics["profit_factor"] == 0.0
    assert result.metrics["avg_r_multiple"] == -0.25


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
