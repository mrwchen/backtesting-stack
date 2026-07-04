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
    )


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

    # sizing: 1% of 100k = 1000 risk / 5 per-share risk = 200 shares
    assert partial["shares"] == 100 and final["shares"] == 100
    assert partial["exit_reason"] == "partial_target"
    assert partial["exit_price"] == 110.0 and partial["pnl"] == 1000.0
    assert final["exit_reason"] == "stop" and final["exit_price"] == 100.0
    assert final["pnl"] == 0.0
    assert result.metrics["final_equity"] == 101000.0
    assert result.metrics["win_rate"] == 1.0


def test_no_entry_on_excessive_gap():
    bars = [(98, 99, 97, 98)] * 5
    bars += [(107, 108, 106, 107)]  # gaps 7% over pivot -> skip
    bars += [(107, 108, 106, 107)] * 10
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
    assert trades.iloc[0]["pnl"] == 200 * (95.0 - 100.0)
    assert result.metrics["final_equity"] == 99000.0


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
    assert trades.iloc[0]["pnl"] == 200 * (90.0 - 100.0)


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
    assert (result.trades["shares"] == 200).all()  # fixed sizing base, no competition


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

    assert result.metrics["num_positions"] == 2
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
    assert trades.iloc[0]["pnl"] == -100.0


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
