from datetime import date, timedelta

import numpy as np
import pytest

from src.portfolio import run_portfolio

WEIGHTS = {"deep": 5.0, "mild": 3.0, "pos": 2.0}


def _days(n):
    return [date(2024, 1, 2) + timedelta(days=i) for i in range(n)]


def _run(closes, positions, *, stress=None, cats=None, cat_mom=None,
         max_positions=25, max_per_category=2, cost=0.0, stock_mom=None,
         entry_confirm_days=0, trim_above_pct=0.0, trim_target_pct=0.0,
         sl_pct=0.0, time_stop_days=0, time_stop_min_ret_pct=0.0):
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
        cost_bps_per_side=cost, stock_mom=stock_mom,
        entry_confirm_days=entry_confirm_days,
        trim_above_pct=trim_above_pct, trim_target_pct=trim_target_pct,
        sl_pct=sl_pct, time_stop_days=time_stop_days,
        time_stop_min_ret_pct=time_stop_min_ret_pct,
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


def test_entry_confirmation_delays_and_filters_whipsaws():
    closes = np.full((6, 2), 100.0)
    # S0 flips long on day 1 and stays long; S1 whipsaws back to flat on day 2
    positions = np.array([[0, 0], [1, 1], [1, 0], [1, 1], [1, 1], [1, 1]],
                         dtype=np.int8)
    res = _run(closes, positions, entry_confirm_days=2)
    entries = {t.symbol: t.entry_date for t in res.trades}
    # S0 enters 2 trading days after its flip; S1's day-1 flip died, its day-3
    # flip confirms on day 5
    assert entries == {"S0": _days(6)[3], "S1": _days(6)[5]}


def test_stock_momentum_breaks_category_momentum_ties():
    closes = np.full((2, 2), 100.0)
    positions = np.ones((2, 2), dtype=np.int8)
    # both stocks share the category (same cat momentum); S1 is more beaten down
    stock_mom = np.array([[0.10, -0.30], [0.10, -0.30]])
    res = _run(closes, positions, max_positions=1, stock_mom=stock_mom)
    assert len(res.trades) == 1
    assert res.trades[0].symbol == "S1"


def test_trimming_sells_position_down_to_target():
    closes = np.array([[100.0], [1000.0], [1000.0]])
    positions = np.ones((3, 1), dtype=np.int8)
    res = _run(closes, positions, trim_above_pct=10.0, trim_target_pct=7.5)
    # 3% entry grows 10x -> ~23.6% of equity on day 1, trimmed back to 7.5%
    value_share = res.gross_exposure_pct[1]
    assert value_share == pytest.approx(7.5, abs=1e-6)
    # equity keeps the full gain (trim only converts stock into cash)
    assert res.equity[1] == pytest.approx(1.27, abs=1e-6)


def test_stop_loss_exits_and_locks_until_signal_reset():
    closes = np.array([[100.0], [75.0], [75.0], [75.0], [75.0], [75.0]])
    # signal stays long through the crash, resets to flat on day 3, flips back
    positions = np.array([[1], [1], [1], [0], [1], [1]], dtype=np.int8)
    res = _run(closes, positions, sl_pct=20.0)
    assert len(res.trades) == 2
    assert res.trades[0].exit_date == _days(6)[1]  # stopped at -25%
    assert res.trades[0].gross_return_pct == pytest.approx(-25.0)
    # locked on days 1-2 despite long signal; re-entry only after the reset
    assert res.trades[1].entry_date == _days(6)[4]


def test_time_stop_exits_losers_but_not_winners():
    closes = np.column_stack([np.full(5, 100.0), 100.0 * 1.01 ** np.arange(5)])
    closes[1:, 0] = 99.0  # S0 sits below entry, S1 grinds up
    positions = np.ones((5, 2), dtype=np.int8)
    res = _run(closes, positions, time_stop_days=3)
    by_symbol = {t.symbol: t for t in res.trades}
    assert by_symbol["S0"].exit_date == _days(5)[3]
    assert not by_symbol["S0"].is_open
    assert by_symbol["S1"].is_open  # winner is never time-stopped


def test_trade_fields_are_plain_python_floats():
    # psycopg2 cannot adapt np.float64; equity flows through numpy closes, so
    # every persisted trade field must be converted to a plain float
    # S1 enters on day 1 while S0 is already held, so equity at entry is an
    # np.float64 (cash + shares * closes) — the case that broke persistence
    closes = np.array([[100.0, 50.0], [110.0, 55.0], [121.0, 60.0]])
    positions = np.array([[1, 0], [1, 1], [0, 0]], dtype=np.int8)
    res = _run(closes, positions)
    for t in res.trades:
        for value in (t.entry_price, t.exit_price, t.gross_return_pct,
                      t.target_weight_pct, t.effective_weight_pct):
            assert type(value) is float


def test_benchmark_is_equal_weight_of_universe():
    closes = np.array([[100.0, 200.0], [110.0, 180.0]])  # +10% and -10%
    positions = np.zeros((2, 2), dtype=np.int8)
    res = _run(closes, positions)
    assert res.bh_equity[-1] == pytest.approx(1.0)
    assert res.total_return_pct == pytest.approx(0.0)
