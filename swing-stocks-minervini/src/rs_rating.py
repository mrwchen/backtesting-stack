"""IBD-style relative-strength rating: cross-sectional 1-99 percentile rank.

rs_raw = 2*(C/C63) + (C/C126) + (C/C189) + (C/C252)   (weights/lookbacks configurable)

Ranked per day against all eligible symbols (price and dollar-volume filters
applied as-of-date). Symbols without a full lookback history are not rankable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def compute_rs(close: pd.DataFrame, volume: pd.DataFrame, cfg: Config) -> dict:
    dollar_vol = (close * volume).rolling(63, min_periods=20).mean()
    eligible = (close >= cfg.min_price) & (dollar_vol >= cfg.min_dollar_volume)

    rs_raw = None
    for weight, lookback in zip(cfg.rs_weights, cfg.rs_lookbacks):
        term = weight * (close / close.shift(lookback))
        rs_raw = term if rs_raw is None else rs_raw + term

    rs_masked = rs_raw.where(eligible)
    pct = rs_masked.rank(axis=1, pct=True)
    rs_rating = np.ceil(pct * 99).clip(1, 99)
    universe_size = rs_masked.notna().sum(axis=1)

    return {
        "rs_raw": rs_masked,
        "rs_rating": rs_rating,
        "eligible": eligible & rs_masked.notna(),
        "universe_size": universe_size,
    }
