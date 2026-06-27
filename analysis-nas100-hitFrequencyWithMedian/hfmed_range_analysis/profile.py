"""Hit-frequency profile range arrays from prior completed bars only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import BarData

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(func):
            return func

        return _wrap


@dataclass(frozen=True)
class RangeProfileConfig:
    price_step: float
    lookback_bars: int
    min_lookback_bars: int
    bar_seconds: int
    profile_max_lookback_seconds: int | None


@dataclass(frozen=True)
class RangeProfileArrays:
    profile_low: np.ndarray
    profile_high: np.ndarray
    profile_range_points: np.ndarray


@njit(cache=True)
def _rolling_range_core(
    bar_start_ns,
    lo_idx,
    hi_idx,
    offset,
    counts,
    lookback,
    min_lookback,
    max_age_ns,
    step,
    out_q0,
    out_q100,
):
    n = bar_start_ns.shape[0]
    total_hits = 0
    wl = 0
    wr = 0
    win_lo = 0
    win_hi = 0
    have_win = False

    for pos in range(n):
        cur_ns = bar_start_ns[pos]

        while wl < wr and cur_ns - bar_start_ns[wl] > max_age_ns:
            lo = lo_idx[wl]
            hi = hi_idx[wl]
            for lv in range(lo, hi + 1):
                counts[lv - offset] -= 1
                total_hits -= 1
            wl += 1

        if (wr - wl) >= min_lookback and total_hits > 0:
            first_nonzero = 0
            last_nonzero = 0
            seen = False
            for lv in range(win_lo, win_hi + 1):
                if counts[lv - offset] <= 0:
                    continue
                if not seen:
                    first_nonzero = lv
                    seen = True
                last_nonzero = lv
            if seen:
                out_q0[pos] = first_nonzero * step
                out_q100[pos] = last_nonzero * step
                win_lo = first_nonzero
                win_hi = last_nonzero

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
        wr = pos + 1

        while (wr - wl) > lookback:
            lo = lo_idx[wl]
            hi = hi_idx[wl]
            for lv in range(lo, hi + 1):
                counts[lv - offset] -= 1
                total_hits -= 1
            wl += 1


def rolling_range_profile_arrays(bars: BarData, cfg: RangeProfileConfig) -> RangeProfileArrays:
    step = float(cfg.price_step)
    lookback = int(cfg.lookback_bars)
    min_lookback = int(cfg.min_lookback_bars)
    max_age_seconds = cfg.profile_max_lookback_seconds or (cfg.lookback_bars * cfg.bar_seconds)
    max_age_ns = int(max_age_seconds) * 1_000_000_000
    n = len(bars)

    q0 = np.full(n, np.nan, dtype=np.float64)
    q100 = np.full(n, np.nan, dtype=np.float64)

    if n > 0:
        low = np.ascontiguousarray(bars.low, dtype=np.float64)
        high = np.ascontiguousarray(bars.high, dtype=np.float64)
        bar_start_ns = np.ascontiguousarray(bars.bar_start_ns, dtype=np.int64)
        tol = step * 1e-9
        lo_idx = np.ceil((low - tol) / step).astype(np.int64)
        hi_idx = np.floor((high + tol) / step).astype(np.int64)
        offset = int(np.floor(low.min() / step)) - 2
        max_level = int(np.floor(high.max() / step)) + 2
        size = max_level - offset + 1
        counts = np.zeros(size, dtype=np.int64)
        _rolling_range_core(
            bar_start_ns,
            lo_idx,
            hi_idx,
            np.int64(offset),
            counts,
            np.int64(lookback),
            np.int64(min_lookback),
            np.int64(max_age_ns),
            step,
            q0,
            q100,
        )

    return RangeProfileArrays(
        profile_low=q0,
        profile_high=q100,
        profile_range_points=q100 - q0,
    )


def warmup() -> None:
    bars = BarData(
        bar_start_ns=np.arange(4, dtype=np.int64) * 1_000_000_000,
        open=np.array([100.0, 101.0, 102.0, 103.0]),
        high=np.array([100.0, 101.0, 102.0, 103.0]),
        low=np.array([100.0, 101.0, 102.0, 103.0]),
        close=np.array([100.0, 101.0, 102.0, 103.0]),
        tick_count=np.ones(4, dtype=np.int32),
    )
    rolling_range_profile_arrays(
        bars,
        RangeProfileConfig(
            price_step=1.0,
            lookback_bars=2,
            min_lookback_bars=2,
            bar_seconds=5,
            profile_max_lookback_seconds=None,
        ),
    )

