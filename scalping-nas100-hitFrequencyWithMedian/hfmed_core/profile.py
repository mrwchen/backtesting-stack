"""Hit-frequency profile and exact 50% data-median level."""

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


def rolling_median_levels(bars: pd.DataFrame) -> pd.Series:
    """Compute median profile levels for each bar from prior completed bars only."""
    step = config.PRICE_STEP
    lookback = config.LOOKBACK_BARS
    min_lookback = config.MIN_LOOKBACK_BARS
    counts: dict[int, int] = {}
    window: deque[list[int]] = deque()
    total_hits = 0
    medians = np.full(len(bars), np.nan, dtype=np.float64)

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

    def current_median() -> float:
        target = total_hits * config.MEDIAN_QUANTILE
        cumulative = 0
        for level in sorted(counts):
            cumulative += counts[level]
            if cumulative >= target:
                return float(level) * step
        return float("nan")

    for pos, row in enumerate(bars.itertuples(index=False)):
        if len(window) >= min_lookback and total_hits > 0:
            medians[pos] = current_median()

        levels = level_indices_between(float(row.low), float(row.high), step)
        window.append(levels)
        add_levels(levels)
        while len(window) > lookback:
            remove_levels(window.popleft())

    return pd.Series(medians, index=bars.index, name="median_level")

