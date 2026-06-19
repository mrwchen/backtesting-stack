"""Hit-frequency profile arrays from prior completed bars only."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .config import RunConfig
from .data import BarData


@dataclass(frozen=True)
class ProfileArrays:
    profile_low: np.ndarray
    band_lower: np.ndarray
    median_level: np.ndarray
    band_upper: np.ndarray
    long_cross_level: np.ndarray
    short_cross_level: np.ndarray
    profile_high: np.ndarray
    stop_profile_lower: np.ndarray
    stop_profile_upper: np.ndarray
    band_width_points: np.ndarray
    profile_range_points: np.ndarray


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


def rolling_profile_arrays(bars: BarData, cfg: RunConfig) -> ProfileArrays:
    step = cfg.price_step
    lookback = cfg.lookback_bars
    min_lookback = cfg.min_lookback_bars
    counts: dict[int, int] = {}
    window: deque[list[int]] = deque()
    total_hits = 0
    n = len(bars)
    q0 = np.full(n, np.nan, dtype=np.float64)
    q45 = np.full(n, np.nan, dtype=np.float64)
    q50 = np.full(n, np.nan, dtype=np.float64)
    q55 = np.full(n, np.nan, dtype=np.float64)
    long_cross = np.full(n, np.nan, dtype=np.float64)
    short_cross = np.full(n, np.nan, dtype=np.float64)
    q100 = np.full(n, np.nan, dtype=np.float64)
    stop_lower = np.full(n, np.nan, dtype=np.float64)
    stop_upper = np.full(n, np.nan, dtype=np.float64)

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

    def current_min_level() -> float:
        return float(min(counts)) * step if counts else float("nan")

    def current_max_level() -> float:
        return float(max(counts)) * step if counts else float("nan")

    for pos in range(n):
        if len(window) >= min_lookback and total_hits > 0:
            q0[pos] = current_min_level()
            q45[pos] = current_quantile(cfg.band_lower_quantile)
            q50[pos] = current_quantile(cfg.median_quantile)
            q55[pos] = current_quantile(cfg.band_upper_quantile)
            long_cross[pos] = current_quantile(cfg.long_cross_quantile)
            short_cross[pos] = current_quantile(cfg.short_cross_quantile)
            q100[pos] = current_max_level()
            stop_lower[pos] = current_quantile(cfg.stop_profile_lower_quantile)
            stop_upper[pos] = current_quantile(cfg.stop_profile_upper_quantile)

        levels = level_indices_between(float(bars.low[pos]), float(bars.high[pos]), step)
        window.append(levels)
        add_levels(levels)
        while len(window) > lookback:
            remove_levels(window.popleft())

    return ProfileArrays(
        profile_low=q0,
        band_lower=q45,
        median_level=q50,
        band_upper=q55,
        long_cross_level=long_cross,
        short_cross_level=short_cross,
        profile_high=q100,
        stop_profile_lower=stop_lower,
        stop_profile_upper=stop_upper,
        band_width_points=q55 - q45,
        profile_range_points=q100 - q0,
    )
