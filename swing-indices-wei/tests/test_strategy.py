import numpy as np

from src.strategy import ema, positions, stress_gate


def test_ema_constant_series_stays_constant():
    x = np.full(50, 100.0)
    assert np.allclose(ema(x, 9), 100.0)


def test_ema_converges_towards_new_level():
    x = np.concatenate([np.full(50, 100.0), np.full(200, 200.0)])
    e = ema(x, 9)
    assert e[49] == 100.0
    assert abs(e[-1] - 200.0) < 0.01


def test_ema_fast_reacts_faster_than_slow():
    x = np.concatenate([np.full(30, 100.0), np.linspace(100, 150, 30)])
    assert ema(x, 9)[-1] > ema(x, 21)[-1]


def test_stress_gate_hysteresis():
    scores = np.array([50.0, 58.0, 55.0, 53.0, 52.0, 55.0, 60.0])
    on = stress_gate(scores, enter=57.0, exit_=52.0)
    #             50   58    55    53    52     55    60
    expected = [False, True, True, True, False, False, True]
    assert on.tolist() == expected


def test_stress_gate_dead_zone_keeps_state():
    # Values between exit and enter must never toggle the gate.
    scores = np.array([55.0, 56.0, 54.0, 55.5])
    assert not stress_gate(scores, 57.0, 52.0).any()


def test_stress_gate_nan_keeps_state():
    scores = np.array([58.0, np.nan, np.nan, 52.0])
    on = stress_gate(scores, 57.0, 52.0)
    assert on.tolist() == [True, True, True, False]


def test_position_flat_only_when_both_bearish():
    n = 60
    # Falling prices -> EMA fast below slow after a while.
    closes = np.linspace(100.0, 60.0, n)
    calm = np.full(n, 45.0)
    stressed = np.full(n, 70.0)

    pos_calm = positions(closes, calm, 9, 21, 57.0, 52.0)["position"]
    pos_stressed = positions(closes, stressed, 9, 21, 57.0, 52.0)["position"]

    # EMA is bearish either way, but without stress we stay long ...
    assert (pos_calm == 1).all()
    # ... and with stress we are flat once the EMA cross is bearish.
    assert (pos_stressed[-20:] == 0).all()


def test_position_long_in_uptrend_despite_stress():
    n = 60
    closes = np.linspace(100.0, 160.0, n)
    stressed = np.full(n, 70.0)
    pos = positions(closes, stressed, 9, 21, 57.0, 52.0)["position"]
    assert (pos == 1).all()
