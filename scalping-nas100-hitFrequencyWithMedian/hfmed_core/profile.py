"""Hit-frequency profile arrays from prior completed bars only.

The rolling profile is a bit-identical, Numba-accelerated reimplementation of the
former pure-Python loop. The rolling window is always a *contiguous* range of bar
positions (both the age cap and the lookback-count cap only ever evict the oldest
bar), so the window is maintained with two pointers over an integer price-level
histogram instead of a per-bar deque of Python lists. If Numba is unavailable the
``njit`` decorator degrades to a no-op and the same code runs as plain Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import RunConfig
from .data import BarData

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback keeps the code runnable
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(func):
            return func

        return _wrap


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
    atr_points: np.ndarray


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


@njit(cache=True)
def _rolling_profile_core(
    bar_start_ns,
    lo_idx,
    hi_idx,
    true_range,
    offset,
    counts,
    lookback,
    min_lookback,
    max_age_ns,
    step,
    q_band_lower,
    q_median,
    q_band_upper,
    q_long,
    q_short,
    q_stop_lower,
    q_stop_upper,
    out_q0,
    out_bl,
    out_med,
    out_bu,
    out_long,
    out_short,
    out_q100,
    out_sl,
    out_su,
    out_atr,
):
    n = bar_start_ns.shape[0]
    total_hits = 0
    tr_sum = 0.0
    wl = 0  # window left bar index (inclusive)
    wr = 0  # window right bar index (exclusive) == bars added so far
    win_lo = 0  # loosely-tracked min active level (<= true min)
    win_hi = 0  # loosely-tracked max active level (>= true max)
    have_win = False

    for pos in range(n):
        cur_ns = bar_start_ns[pos]

        # 1. evict by wall-clock age from the front (oldest first).
        while wl < wr and cur_ns - bar_start_ns[wl] > max_age_ns:
            lo = lo_idx[wl]
            hi = hi_idx[wl]
            for lv in range(lo, hi + 1):
                counts[lv - offset] -= 1
                total_hits -= 1
            tr_sum -= true_range[wl]
            wl += 1

        # 2. compute quantiles from prior completed bars only.
        if (wr - wl) >= min_lookback and total_hits > 0:
            out_atr[pos] = tr_sum / (wr - wl)
            # scan the active level range once; capture min/max and 7 quantiles.
            lo_scan = win_lo
            hi_scan = win_hi
            cumulative = 0
            first_nonzero = 0
            last_nonzero = 0
            seen = False
            t_bl = total_hits * q_band_lower
            t_med = total_hits * q_median
            t_bu = total_hits * q_band_upper
            t_long = total_hits * q_long
            t_short = total_hits * q_short
            t_sl = total_hits * q_stop_lower
            t_su = total_hits * q_stop_upper
            d_bl = False
            d_med = False
            d_bu = False
            d_long = False
            d_short = False
            d_sl = False
            d_su = False
            for lv in range(lo_scan, hi_scan + 1):
                c = counts[lv - offset]
                if c <= 0:
                    continue
                if not seen:
                    first_nonzero = lv
                    seen = True
                last_nonzero = lv
                cumulative += c
                level_price = lv * step
                if not d_bl and cumulative >= t_bl:
                    out_bl[pos] = level_price
                    d_bl = True
                if not d_med and cumulative >= t_med:
                    out_med[pos] = level_price
                    d_med = True
                if not d_bu and cumulative >= t_bu:
                    out_bu[pos] = level_price
                    d_bu = True
                if not d_long and cumulative >= t_long:
                    out_long[pos] = level_price
                    d_long = True
                if not d_short and cumulative >= t_short:
                    out_short[pos] = level_price
                    d_short = True
                if not d_sl and cumulative >= t_sl:
                    out_sl[pos] = level_price
                    d_sl = True
                if not d_su and cumulative >= t_su:
                    out_su[pos] = level_price
                    d_su = True
            out_q0[pos] = first_nonzero * step
            out_q100[pos] = last_nonzero * step
            # tighten the scan bounds to the true active range for next time.
            win_lo = first_nonzero
            win_hi = last_nonzero

        # 3. append current bar.
        lo = lo_idx[pos]
        hi = hi_idx[pos]
        if hi >= lo:
            if not have_win:
                win_lo = lo
                win_hi = hi
                have_win = True
            else:
                if lo < win_lo:
                    win_lo = lo
                if hi > win_hi:
                    win_hi = hi
            for lv in range(lo, hi + 1):
                counts[lv - offset] += 1
                total_hits += 1
        tr_sum += true_range[pos]
        wr = pos + 1

        # 4. evict by lookback-count cap from the front (oldest first).
        while (wr - wl) > lookback:
            lo = lo_idx[wl]
            hi = hi_idx[wl]
            for lv in range(lo, hi + 1):
                counts[lv - offset] -= 1
                total_hits -= 1
            tr_sum -= true_range[wl]
            wl += 1


def rolling_profile_arrays(bars: BarData, cfg: RunConfig) -> ProfileArrays:
    step = float(cfg.price_step)
    lookback = int(cfg.lookback_bars)
    min_lookback = int(cfg.min_lookback_bars)
    max_age_seconds = cfg.profile_max_lookback_seconds or (cfg.lookback_bars * cfg.bar_seconds)
    max_age_ns = int(max_age_seconds) * 1_000_000_000
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
    atr = np.full(n, np.nan, dtype=np.float64)

    if n > 0:
        low = np.ascontiguousarray(bars.low, dtype=np.float64)
        high = np.ascontiguousarray(bars.high, dtype=np.float64)
        close = np.ascontiguousarray(bars.close, dtype=np.float64)
        bar_start_ns = np.ascontiguousarray(bars.bar_start_ns, dtype=np.int64)
        prev_close = np.empty(n, dtype=np.float64)
        prev_close[0] = close[0]
        if n > 1:
            prev_close[1:] = close[:-1]
        true_range = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
        true_range = np.ascontiguousarray(true_range, dtype=np.float64)
        # integer level bounds per bar, matching level_indices_between exactly.
        tol = step * 1e-9
        lo_idx = np.ceil((low - tol) / step).astype(np.int64)
        hi_idx = np.floor((high + tol) / step).astype(np.int64)
        offset = int(np.floor(low.min() / step)) - 2
        max_level = int(np.floor(high.max() / step)) + 2
        size = max_level - offset + 1
        counts = np.zeros(size, dtype=np.int64)
        _rolling_profile_core(
            bar_start_ns,
            lo_idx,
            hi_idx,
            true_range,
            np.int64(offset),
            counts,
            np.int64(lookback),
            np.int64(min_lookback),
            np.int64(max_age_ns),
            step,
            float(cfg.band_lower_quantile),
            float(cfg.median_quantile),
            float(cfg.band_upper_quantile),
            float(cfg.long_cross_quantile),
            float(cfg.short_cross_quantile),
            float(cfg.stop_profile_lower_quantile),
            float(cfg.stop_profile_upper_quantile),
            q0, q45, q50, q55, long_cross, short_cross, q100, stop_lower, stop_upper, atr,
        )

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
        atr_points=atr,
    )


def warmup() -> None:
    """Force JIT compilation once so worker processes inherit the cache."""
    bars = BarData(
        bar_start_ns=np.arange(4, dtype=np.int64) * 1_000_000_000,
        open=np.array([100.0, 101.0, 102.0, 103.0]),
        high=np.array([100.0, 101.0, 102.0, 103.0]),
        low=np.array([100.0, 101.0, 102.0, 103.0]),
        close=np.array([100.0, 101.0, 102.0, 103.0]),
        tick_count=np.ones(4, dtype=np.int32),
    )

    class _C:
        price_step = 1.0
        lookback_bars = 2
        min_lookback_bars = 2
        bar_seconds = 10
        profile_max_lookback_seconds = None
        band_lower_quantile = 0.45
        median_quantile = 0.5
        band_upper_quantile = 0.55
        long_cross_quantile = 0.5
        short_cross_quantile = 0.5
        stop_profile_lower_quantile = 0.0
        stop_profile_upper_quantile = 1.0

    rolling_profile_arrays(bars, _C())
