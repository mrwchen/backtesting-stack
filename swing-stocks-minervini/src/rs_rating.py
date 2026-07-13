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


def require_aligned_matrix(
    reference: pd.DataFrame, matrix: pd.DataFrame, name: str
) -> None:
    if not reference.index.equals(matrix.index) or not reference.columns.equals(
        matrix.columns
    ):
        raise ValueError(f"{name} must have the same index and columns as adjusted close")


def validate_continuity_matrices(
    close: pd.DataFrame,
    continuity_segment: pd.DataFrame,
    continuity_break: pd.DataFrame,
) -> None:
    """Validate the mandatory daily-bar continuity matrix contract.

    Segment IDs are positive and monotonic per symbol. On every observed row
    after the first loaded row, the break flag must be exactly equivalent to a
    segment change. The first loaded row may legitimately be in the middle of
    a segment and therefore need not carry a break flag.
    """

    require_aligned_matrix(close, continuity_segment, "continuity_segment")
    require_aligned_matrix(close, continuity_break, "continuity_break")
    observed = close.notna()
    if (observed & continuity_segment.isna()).to_numpy().any():
        raise ValueError("continuity_segment is required for every adjusted close")
    if (observed & continuity_break.isna()).to_numpy().any():
        raise ValueError("continuity_break is required for every adjusted close")

    try:
        segment_values = continuity_segment.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("continuity_segment must contain positive integers") from exc
    observed_values = observed.to_numpy()
    valid_segments = (
        np.isfinite(segment_values)
        & (segment_values > 0)
        & (segment_values == np.floor(segment_values))
    )
    if not valid_segments[observed_values].all():
        raise ValueError("continuity_segment must contain positive integers")

    segment = continuity_segment.where(observed)
    previous = segment.ffill().shift()
    has_previous = observed & previous.notna()
    changed = continuity_segment.ne(previous)
    actual_break = continuity_break.fillna(False).astype(bool)
    if (
        has_previous & actual_break.ne(changed)
    ).to_numpy().any() or (
        has_previous & continuity_segment.lt(previous)
    ).to_numpy().any():
        raise ValueError("continuity_segment and continuity_break are inconsistent")


def rolling_within_continuity(
    values: pd.DataFrame,
    continuity_segment: pd.DataFrame,
    window: int,
    *,
    min_periods: int,
    operation: str,
) -> pd.DataFrame:
    """Roll a matrix without allowing any window to span a segment boundary.

    The common case (one segment per symbol in the loaded window) remains
    vectorized. Only symbols that actually contain a boundary are recomputed
    segment by segment.
    """

    roller = values.rolling(window, min_periods=min_periods)
    result = getattr(roller, operation)().where(continuity_segment.notna())
    segment_counts = continuity_segment.where(values.notna()).nunique(dropna=True)
    for symbol in segment_counts.index[segment_counts > 1]:
        result[symbol] = np.nan
        symbol_segments = continuity_segment[symbol]
        for segment_id in symbol_segments.dropna().unique():
            in_segment = symbol_segments.eq(segment_id)
            isolated = values[symbol].where(in_segment)
            rolled = getattr(
                isolated.rolling(window, min_periods=min_periods), operation
            )()
            result.loc[in_segment, symbol] = rolled.loc[in_segment]
    return result


def same_continuity_segment(
    continuity_segment: pd.DataFrame, left_shift: int, right_shift: int
) -> pd.DataFrame:
    left = continuity_segment.shift(left_shift)
    right = continuity_segment.shift(right_shift)
    return left.notna() & right.notna() & left.eq(right)


def compute_rs(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    cfg: Config,
    *,
    raw_close: pd.DataFrame,
    continuity_segment: pd.DataFrame,
    continuity_break: pd.DataFrame,
) -> dict:
    """Compute RS from adjusted series, with nominal eligibility from raw close."""
    if tuple(sorted(cfg.rs_lookbacks)) != tuple(cfg.rs_lookbacks):
        raise ValueError("RS_LOOKBACKS must be strictly increasing")
    if len(set(cfg.rs_lookbacks)) != len(cfg.rs_lookbacks):
        raise ValueError("RS_LOOKBACKS must be strictly increasing")
    if len(cfg.rs_weights) != len(cfg.rs_lookbacks):
        raise ValueError("RS_WEIGHTS and RS_LOOKBACKS must have equal length")
    require_aligned_matrix(close, raw_close, "raw_close")
    require_aligned_matrix(close, volume, "volume")
    validate_continuity_matrices(close, continuity_segment, continuity_break)

    dollar_vol = rolling_within_continuity(
        close * volume,
        continuity_segment,
        63,
        min_periods=20,
        operation="mean",
    )
    eligible = (raw_close >= cfg.min_price) & (dollar_vol >= cfg.min_dollar_volume)

    rs_raw = None
    previous_lookback = 0
    for weight, lookback in zip(cfg.rs_weights, cfg.rs_lookbacks):
        window_end = close if previous_lookback == 0 else close.shift(previous_lookback)
        term = weight * (window_end / close.shift(lookback) - 1.0)
        term = term.where(
            same_continuity_segment(
                continuity_segment, previous_lookback, lookback
            )
        )
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
        "dollar_volume": dollar_vol,
    }
