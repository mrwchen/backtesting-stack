"""Hit-frequency profile, exact 50% median level and 45%-55% data band."""

from collections import deque
import math

import numpy as np
import pandas as pd

from . import config


def level_indices_between(low: float, high: float, step: float) -> list[int]:
    """Return all price-step indices whose levels lie inside [low, high]."""
    if not np.isfinite(low) or not np.isfinite(high) or high < low:
        return []
    tolerance = step * 1e-9
    first_idx = math.ceil((low - tolerance) / step)
    last_idx = math.floor((high + tolerance) / step)
    if last_idx < first_idx:
        return []
    return list(range(first_idx, last_idx + 1))


def rolling_profile_levels(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute q45/q50/q55 profile levels from prior completed bars only."""
    step = config.PRICE_STEP
    lookback = config.LOOKBACK_BARS
    min_lookback = config.MIN_LOOKBACK_BARS
    counts: dict[int, int] = {}
    window: deque[list[int]] = deque()
    total_hits = 0
    q45 = np.full(len(bars), np.nan, dtype=np.float64)
    q50 = np.full(len(bars), np.nan, dtype=np.float64)
    q55 = np.full(len(bars), np.nan, dtype=np.float64)

    def add_levels(levels: list[int]) -> None:
        nonlocal total_hits
        for level in levels:
            counts[level] = counts.get(level, 0) + 1
        total_hits += len(levels)

    def remove_levels(levels: list[int]) -> None:
        nonlocal total_hits
        for level in levels:
            current = counts.get(level, 0)
            if current <= 1:
                counts.pop(level, None)
            else:
                counts[level] = current - 1
        total_hits -= len(levels)

    def current_quantile(quantile: float) -> float:
        target = total_hits * quantile
        cumulative = 0
        for level in sorted(counts):
            cumulative += counts[level]
            if cumulative >= target:
                return float(level) * step
        return float("nan")

    for pos, row in enumerate(bars.itertuples(index=False)):
        if len(window) >= min_lookback and total_hits > 0:
            q45[pos] = current_quantile(config.BAND_LOWER_QUANTILE)
            q50[pos] = current_quantile(config.MEDIAN_QUANTILE)
            q55[pos] = current_quantile(config.BAND_UPPER_QUANTILE)

        levels = level_indices_between(float(row.low), float(row.high), step)
        window.append(levels)
        add_levels(levels)
        while len(window) > lookback:
            remove_levels(window.popleft())

    return pd.DataFrame(
        {
            "band_lower": q45,
            "median_level": q50,
            "band_upper": q55,
            "band_width_points": q55 - q45,
        },
        index=bars.index,
    )


def rolling_median_levels(bars: pd.DataFrame) -> pd.Series:
    """Compute median profile levels for each bar from prior completed bars only."""
    return rolling_profile_levels(bars)["median_level"]
