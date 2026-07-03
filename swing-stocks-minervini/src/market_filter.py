"""Market regime filter: breadth of eligible stocks above their 200-day MA.

market_on uses hysteresis so the gate does not flap around the threshold:
it switches on when breadth >= BREADTH_ON and only switches off again when
breadth drops below BREADTH_OFF. The gate blocks new entries only — open
positions keep running into their regular exits.

Computed from the same adjusted-close matrix the screen uses, so it is
point-in-time by construction. Note the survivorship caveat: with today's
universe backfilled, historical breadth is biased slightly upward — keep the
thresholds round and do not fine-tune them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _hysteresis(values: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    state = False
    out = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        if np.isnan(value):
            state = False
        elif state:
            state = value >= off_threshold
        else:
            state = value >= on_threshold
        out[i] = state
    return out


def compute_breadth(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Returns a per-day frame with market_breadth (0..1) and market_on (bool)."""
    ma200 = close.rolling(200, min_periods=200).mean()
    eligible = (close >= cfg.min_price) & ma200.notna()
    above = (close > ma200) & eligible
    breadth = above.sum(axis=1) / eligible.sum(axis=1).replace(0, np.nan)
    market_on = _hysteresis(
        breadth.to_numpy(dtype=float), cfg.breadth_on_threshold, cfg.breadth_off_threshold
    )
    return pd.DataFrame(
        {"market_breadth": breadth, "market_on": market_on}, index=close.index
    )
