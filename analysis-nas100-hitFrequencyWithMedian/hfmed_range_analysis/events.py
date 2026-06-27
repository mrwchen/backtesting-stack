"""Tick-level q50 crossing event detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import BarData, TickData
from .profile import RangeProfileArrays


@dataclass(frozen=True)
class CrossingEvents:
    tick_index: np.ndarray
    cross_ts_ns: np.ndarray
    bar_start_ns: np.ndarray
    direction_code: np.ndarray
    previous_mid: np.ndarray
    signal_mid: np.ndarray
    q50_level: np.ndarray
    profile_low: np.ndarray
    profile_high: np.ndarray
    profile_range_points: np.ndarray

    def __len__(self) -> int:
        return int(self.tick_index.shape[0])


def detect_q50_crossing_events(
    ticks: TickData,
    bars: BarData,
    profile: RangeProfileArrays,
) -> CrossingEvents:
    if len(ticks) <= 1:
        return _empty_events()

    bar_index = np.asarray(ticks.bar_index, dtype=np.int64)
    in_range = (bar_index >= 0) & (bar_index < len(bars))
    q50 = np.full(len(ticks), np.nan, dtype=np.float64)
    q50[in_range] = profile.median_level[bar_index[in_range]]
    valid = np.isfinite(q50)

    prev_mid = np.empty(len(ticks), dtype=np.float64)
    prev_mid[0] = np.nan
    prev_mid[1:] = ticks.mid[:-1]
    mid = np.asarray(ticks.mid, dtype=np.float64)

    up_cross = valid & (prev_mid < q50) & (q50 <= mid)
    down_cross = valid & (prev_mid > q50) & (q50 >= mid)
    mask = up_cross | down_cross
    idx = np.flatnonzero(mask).astype(np.int64, copy=False)
    if idx.size == 0:
        return _empty_events()

    event_bar_index = bar_index[idx]
    direction = np.where(up_cross[idx], 1, -1).astype(np.int8, copy=False)
    return CrossingEvents(
        tick_index=idx,
        cross_ts_ns=ticks.tick_time_ns[idx].astype(np.int64, copy=False),
        bar_start_ns=bars.bar_start_ns[event_bar_index].astype(np.int64, copy=False),
        direction_code=direction,
        previous_mid=prev_mid[idx].astype(np.float64, copy=False),
        signal_mid=mid[idx].astype(np.float64, copy=False),
        q50_level=q50[idx].astype(np.float64, copy=False),
        profile_low=profile.profile_low[event_bar_index].astype(np.float64, copy=False),
        profile_high=profile.profile_high[event_bar_index].astype(np.float64, copy=False),
        profile_range_points=profile.profile_range_points[event_bar_index].astype(np.float64, copy=False),
    )


def _empty_events() -> CrossingEvents:
    return CrossingEvents(
        tick_index=np.empty(0, dtype=np.int64),
        cross_ts_ns=np.empty(0, dtype=np.int64),
        bar_start_ns=np.empty(0, dtype=np.int64),
        direction_code=np.empty(0, dtype=np.int8),
        previous_mid=np.empty(0, dtype=np.float64),
        signal_mid=np.empty(0, dtype=np.float64),
        q50_level=np.empty(0, dtype=np.float64),
        profile_low=np.empty(0, dtype=np.float64),
        profile_high=np.empty(0, dtype=np.float64),
        profile_range_points=np.empty(0, dtype=np.float64),
    )
