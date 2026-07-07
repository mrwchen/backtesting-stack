"""Signal logic: EMA trend filter x composite-score stress gate.

Rule (decided at each trading day's close, position held into the next close):

    flat  if EMA_fast < EMA_slow  AND  stress gate is ON
    long  otherwise

The stress gate is a hysteresis state driven by the previous day's composite
score: it switches ON at >= stress_enter and OFF again at <= stress_exit, so a
score oscillating between the two thresholds does not flip the position.
"""
from __future__ import annotations

import numpy as np


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA (alpha = 2/(span+1)), seeded with the first value."""
    alpha = 2.0 / (span + 1.0)
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = out[i - 1] + alpha * (values[i] - out[i - 1])
    return out


def stress_gate(lagged_scores: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Hysteresis state per trading day. NaN scores keep the previous state."""
    on = np.zeros(len(lagged_scores), dtype=bool)
    cur = False
    for i, score in enumerate(lagged_scores):
        if not np.isnan(score):
            if not cur and score >= enter:
                cur = True
            elif cur and score <= exit_:
                cur = False
        on[i] = cur
    return on


def positions(closes: np.ndarray, lagged_scores: np.ndarray,
              ema_fast: int, ema_slow: int,
              stress_enter: float, stress_exit: float) -> dict[str, np.ndarray]:
    """Per-day signal state. position[t] is decided at close t."""
    ema_f = ema(closes, ema_fast)
    ema_s = ema(closes, ema_slow)
    stress = stress_gate(lagged_scores, stress_enter, stress_exit)
    ema_bearish = ema_f < ema_s
    pos = np.where(ema_bearish & stress, 0, 1).astype(np.int8)
    return {"ema_fast": ema_f, "ema_slow": ema_s, "stress_on": stress, "position": pos}
