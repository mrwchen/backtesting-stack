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
