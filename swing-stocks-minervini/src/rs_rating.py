"""IBD-style relative-strength rating: cross-sectional 1-99 percentile rank.

The four configured lookbacks delimit disjoint return windows.  With the
default 63/126/189/252 sessions and 2/1/1/1 weights this is a 40/20/20/20
blend of the latest four quarters, rather than four overlapping returns.

Ranked per day against all eligible symbols (price and dollar-volume filters
applied as-of-date). Symbols without a full lookback history are not rankable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def compute_rs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: Config,
    *,
    raw_close: pd.DataFrame,
) -> dict:
    """Compute RS from adjusted series, with nominal eligibility from raw close."""
    if tuple(sorted(cfg.rs_lookbacks)) != tuple(cfg.rs_lookbacks):
        raise ValueError("RS_LOOKBACKS must be strictly increasing")
    if len(set(cfg.rs_lookbacks)) != len(cfg.rs_lookbacks):
        raise ValueError("RS_LOOKBACKS must be strictly increasing")
    if len(cfg.rs_weights) != len(cfg.rs_lookbacks):
        raise ValueError("RS_WEIGHTS and RS_LOOKBACKS must have equal length")
    if not close.index.equals(raw_close.index) or not close.columns.equals(raw_close.columns):
        raise ValueError("raw_close must have the same index and columns as adjusted close")

    dollar_vol = (close * volume).rolling(63, min_periods=20).mean()
    eligible = (raw_close >= cfg.min_price) & (dollar_vol >= cfg.min_dollar_volume)

    rs_raw = None
    previous_lookback = 0
    for weight, lookback in zip(cfg.rs_weights, cfg.rs_lookbacks):
        window_end = close if previous_lookback == 0 else close.shift(previous_lookback)
        term = weight * (window_end / close.shift(lookback) - 1.0)
        rs_raw = term if rs_raw is None else rs_raw + term
        previous_lookback = lookback

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
