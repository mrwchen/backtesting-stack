from datetime import date, timedelta

import numpy as np
import pytest

from src.portfolio import run_portfolio

WEIGHTS = {"deep": 5.0, "mild": 3.0, "pos": 2.0}


def _days(n):
    return [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]


def _run(closes, positions, *, stress=None, cats=None, cat_mom=None,
         max_positions=25, max_per_category=2, cost=0.0):
    n_days, n_sym = closes.shape
    symbols = [f"S{i}" for i in range(n_sym)]
    cats = cats or {s: "CatA" for s in symbols}
    mom_days = {c: (cat_mom[c] if cat_mom else np.full(n_days, np.nan))
                for c in set(cats.values())}
    return run_portfolio(
        days=_days(n_days), symbols=symbols, categories=cats,
        closes=closes, fresh=np.ones_like(closes, dtype=bool),
        positions=positions,
        stress_on=stress if stress is not None else np.zeros(n_days, dtype=bool),
        cat_momentum=mom_days, weight_pct_by_tier=WEIGHTS, deep_threshold=-0.10,
        max_positions=max_positions, max_per_category=max_per_category,
        cost_bps_per_side=cost,
    )


def test_single_trade_pnl_matches_weight_times_move():
    closes = np.array([[100.0], [110.0], [121.0], [121.0]])
    positions = np.array([[1], [1], [1], [0]], dtype=np.int8)
    res = _run(closes, positions)
    # mild tier (unknown momentum) = 3% of equity; stock gains 21% -> +0.63%
    assert res.trades[0].tier == "mild"
    assert res.trades[0].gross_return_pct == pytest.approx(21.0)
    assert res.total_return_pct == pytest.approx(0.63, abs=1e-6)
    assert not res.trades[0].is_open


def test_deep_tier_gets_full_weight():
    closes = np.array([[100.0], [120.0]])
    positions = np.array([[1], [1]], dtype=np.int8)
    res = _run(closes, positions,
               cat_mom={"CatA": np.array([-0.20, -0.20])})
    assert res.trades[0].tier == "deep"
    assert res.trades[0].target_weight_pct == 5.0
    assert res.trades[0].effective_weight_pct == 5.0
    assert res.total_return_pct == pytest.approx(1.0, abs=1e-6)  # 5% * +20%


def test_no_new_entries_while_stress_is_on():
    closes = np.full((4, 1), 100.0)
    positions = np.ones((4, 1), dtype=np.int8)
    stress = np.array([True, True, False, False])
    res = _run(closes, positions, stress=stress)
    # entry only happens once the light turns green (day 2)...
    assert len(res.trades) == 0 or res.trades[0].entry_date == _days(4)[2]
    # ...and because the signal was already long, the flat->long flip never
    # happened after day 0, so no entry at all is the expected behaviour
    assert len(res.trades) == 0


def test_entry_on_flip_after_stress_clears():
    closes = np.full((4, 1), 100.0)
    positions = np.array([[1], [0], [1], [1]], dtype=np.int8)
    stress = np.array([True, False, False, False])
    res = _run(closes, positions, stress=stress)
    assert len(res.trades) == 1
    assert res.trades[0].entry_date == _days(4)[2]


def test_max_per_category_is_enforced():
    closes = np.full((2, 3), 100.0)
    positions = np.ones((2, 3), dtype=np.int8)
    res = _run(closes, positions, max_per_category=2)
    assert len(res.trades) == 2


def test_deepest_category_momentum_is_admitted_first():
    closes = np.full((2, 2), 100.0)
    positions = np.ones((2, 2), dtype=np.int8)
    cats = {"S0": "Hot", "S1": "Cold"}
    cat_mom = {"Hot": np.full(2, 0.15), "Cold": np.full(2, -0.15)}
    res = _run(closes, positions, cats=cats, cat_mom=cat_mom, max_positions=1)
    assert len(res.trades) == 1
    assert res.trades[0].symbol == "S1"
    assert res.trades[0].tier == "deep"


def test_costs_reduce_equity_on_round_trip():
    closes = np.array([[100.0], [100.0], [100.0]])
    positions = np.array([[1], [1], [0]], dtype=np.int8)
    res = _run(closes, positions, cost=100.0)  # 1% per side, absurdly high
    # 3% position, ~2% round-trip costs on it -> ~ -0.06% on equity
    assert res.total_return_pct == pytest.approx(-0.0597, abs=1e-3)


def test_open_trade_is_marked_open_and_valued():
    closes = np.array([[100.0], [150.0]])
    positions = np.ones((2, 1), dtype=np.int8)
    res = _run(closes, positions)
    assert res.trades[0].is_open
    assert res.trades[0].gross_return_pct == pytest.approx(50.0)
    assert res.total_return_pct == pytest.approx(1.5, abs=1e-6)  # 3% * +50%


def test_benchmark_is_equal_weight_of_universe():
    closes = np.array([[100.0, 200.0], [110.0, 180.0]])  # +10% and -10%
    positions = np.zeros((2, 2), dtype=np.int8)
    res = _run(closes, positions)
    assert res.bh_equity[-1] == pytest.approx(1.0)
    assert res.total_return_pct == pytest.approx(0.0)
