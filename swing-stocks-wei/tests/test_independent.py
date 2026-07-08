from datetime import date, timedelta

import numpy as np
import pytest

from src.independent import run_independent_trades

WEIGHTS = {"deep": 5.0, "mild": 3.0, "pos": 2.0}


def _days(n):
    return [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]


def _run(closes, positions, *, stress=None, cats=None, cat_mom=None,
         entry_confirm_days=0, sl_pct=0.0, time_stop_days=0,
         time_stop_min_ret_pct=0.0):
    n_days, n_sym = closes.shape
    symbols = [f"S{i}" for i in range(n_sym)]
    cats = cats or {s: "CatA" for s in symbols}
    mom_days = {c: (cat_mom[c] if cat_mom else np.full(n_days, np.nan))
                for c in set(cats.values())}
    return run_independent_trades(
        days=_days(n_days), symbols=symbols, categories=cats,
        closes=closes, fresh=np.ones_like(closes, dtype=bool),
        positions=positions,
        stress_on=stress if stress is not None else np.zeros(n_days, dtype=bool),
        cat_momentum=mom_days, weight_pct_by_tier=WEIGHTS,
        deep_threshold=-0.10, entry_confirm_days=entry_confirm_days,
        sl_pct=sl_pct, time_stop_days=time_stop_days,
        time_stop_min_ret_pct=time_stop_min_ret_pct,
    )


def test_independent_mode_takes_all_simultaneous_signals():
    closes = np.array([[100.0, 100.0, 100.0], [110.0, 90.0, 120.0]])
    positions = np.ones((2, 3), dtype=np.int8)
    res = _run(closes, positions)
    assert len(res.trades) == 3
    assert [t.symbol for t in res.trades] == ["S0", "S1", "S2"]
    assert [t.gross_return_pct for t in res.trades] == pytest.approx([10.0, -10.0, 20.0])


def test_independent_mode_does_not_use_portfolio_caps():
    closes = np.full((2, 4), 100.0)
    positions = np.ones((2, 4), dtype=np.int8)
    res = _run(closes, positions)
    assert len(res.trades) == 4


def test_independent_mode_still_blocks_entries_during_stress():
    closes = np.full((4, 1), 100.0)
    positions = np.ones((4, 1), dtype=np.int8)
    stress = np.array([True, True, False, False])
    res = _run(closes, positions, stress=stress)
    assert len(res.trades) == 0


def test_independent_mode_enters_on_flip_after_stress_clears():
    closes = np.full((4, 1), 100.0)
    positions = np.array([[1], [0], [1], [1]], dtype=np.int8)
    stress = np.array([True, False, False, False])
    res = _run(closes, positions, stress=stress)
    assert len(res.trades) == 1
    assert res.trades[0].entry_date == _days(4)[2]


def test_independent_mode_applies_entry_confirmation():
    closes = np.full((5, 1), 100.0)
    positions = np.array([[0], [1], [1], [1], [1]], dtype=np.int8)
    res = _run(closes, positions, entry_confirm_days=2)
    assert len(res.trades) == 1
    assert res.trades[0].entry_date == _days(5)[3]


def test_independent_mode_applies_stop_loss_with_lock():
    closes = np.array([[100.0], [70.0], [70.0], [70.0]])
    positions = np.ones((4, 1), dtype=np.int8)  # signal never resets
    res = _run(closes, positions, sl_pct=20.0)
    assert len(res.trades) == 1  # stopped once, locked afterwards
    assert res.trades[0].gross_return_pct == pytest.approx(-30.0)
    assert not res.trades[0].is_open


def test_independent_mode_applies_time_stop():
    closes = np.array([[100.0], [99.0], [99.0], [99.0], [99.0]])
    positions = np.ones((5, 1), dtype=np.int8)
    res = _run(closes, positions, time_stop_days=3)
    assert res.trades[0].exit_date == _days(5)[3]
    assert not res.trades[0].is_open


def test_independent_mode_keeps_sizing_tiers():
    closes = np.array([[100.0], [120.0]])
    positions = np.ones((2, 1), dtype=np.int8)
    res = _run(closes, positions,
               cat_mom={"CatA": np.array([-0.20, -0.20])})
    assert res.trades[0].tier == "deep"
    assert res.trades[0].target_weight_pct == 5.0
    assert res.trades[0].effective_weight_pct == 5.0
    assert res.avg_trade_return_pct == pytest.approx(20.0)
