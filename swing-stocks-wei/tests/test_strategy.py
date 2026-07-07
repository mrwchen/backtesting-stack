import numpy as np

from src.strategy import ema, sizing_tier, stock_positions, stress_gate


def test_ema_converges_to_constant():
    values = np.full(300, 42.0)
    assert abs(ema(values, 9)[-1] - 42.0) < 1e-9


def test_ema_carries_through_nan_gaps():
    values = np.array([10.0, np.nan, np.nan, 10.0, 10.0])
    out = ema(values, 3)
    assert not np.isnan(out[1:]).any()
    assert abs(out[-1] - 10.0) < 1e-9


def test_ema_leading_nans_stay_nan_until_first_bar():
    values = np.array([np.nan, np.nan, 5.0, 5.0])
    out = ema(values, 3)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 5.0


def test_stress_gate_hysteresis():
    scores = np.array([50.0, 58.0, 55.0, 53.0, 52.0, 54.0, 60.0])
    gate = stress_gate(scores, enter=57, exit_=52)
    # switches ON at 58, stays ON through the 55/53 dead zone, OFF at 52,
    # stays OFF at 54, ON again at 60
    assert gate.tolist() == [False, True, True, True, False, False, True]


def test_stress_gate_nan_keeps_state():
    scores = np.array([58.0, np.nan, np.nan, 52.0, np.nan])
    gate = stress_gate(scores, enter=57, exit_=52)
    assert gate.tolist() == [True, True, True, False, False]


def test_stock_positions_flat_only_when_bearish_and_stressed():
    # falling closes -> EMA bearish from some point on
    closes = np.linspace(100, 60, 40)
    stress = np.zeros(40, dtype=bool)
    stress[20:] = True
    pos = stock_positions(closes, stress, ema_fast=3, ema_slow=10)
    assert pos[:20].min() == 1          # bearish alone is not enough
    assert pos[25:].max() == 0          # bearish AND stressed -> flat


def test_stock_positions_zero_before_first_bar():
    closes = np.concatenate([np.full(5, np.nan), np.linspace(10, 12, 20)])
    pos = stock_positions(closes, np.zeros(25, dtype=bool), 3, 10)
    assert pos[:5].max() == 0
    assert pos[10:].min() == 1


def test_sizing_tiers():
    assert sizing_tier(-0.15, deep_threshold=-0.10) == "deep"
    assert sizing_tier(-0.10, deep_threshold=-0.10) == "deep"
    assert sizing_tier(-0.03, deep_threshold=-0.10) == "mild"
    assert sizing_tier(0.0, deep_threshold=-0.10) == "mild"
    assert sizing_tier(0.08, deep_threshold=-0.10) == "pos"
    assert sizing_tier(float("nan"), deep_threshold=-0.10) == "mild"
