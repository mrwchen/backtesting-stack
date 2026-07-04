"""VCP (Volatility Contraction Pattern) detection on daily bars.

A base is a sequence of 2-4 (swing high -> swing low) contractions with
non-increasing depth, the last one tight (<= FINAL_DEPTH_MAX), volume drying
up, and price sitting just below the pivot (= high of the final contraction).

Look-ahead discipline: a swing at bar i is confirmed only k bars later, so on
evaluation day t only swings with index <= t-k are used, and price must not
have exceeded the pivot since the final swing low.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import Config


@dataclass
class Setup:
    symbol: str
    detect_date: date
    pivot: float
    last_low: float
    stop_level: float
    base_start_date: date
    base_days: int
    n_contractions: int
    contraction_depths: list[float]
    dryup_ratio: float
    close: float
    valid_until: date


def find_swings(high: np.ndarray, low: np.ndarray, k: int) -> list[tuple[int, float, str]]:
    """Alternating swing points as (index, price, kind) with kind 'H'/'L'."""
    window = 2 * k + 1
    roll_max = pd.Series(high).rolling(window, center=True).max().to_numpy()
    roll_min = pd.Series(low).rolling(window, center=True).min().to_numpy()

    events: list[tuple[int, float, str]] = []
    for i in range(len(high)):
        if not np.isnan(roll_max[i]) and high[i] >= roll_max[i]:
            events.append((i, float(high[i]), "H"))
        if not np.isnan(roll_min[i]) and low[i] <= roll_min[i]:
            events.append((i, float(low[i]), "L"))

    merged: list[tuple[int, float, str]] = []
    for event in events:
        if merged and merged[-1][2] == event[2]:
            is_h = event[2] == "H"
            if (is_h and event[1] >= merged[-1][1]) or (not is_h and event[1] <= merged[-1][1]):
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def _evaluate_day(
    t: int,
    pairs: list[tuple[tuple, tuple]],
    high: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vol_ma: np.ndarray,
    cfg: Config,
) -> dict | None:
    close_t = close[t]
    if np.isnan(close_t) or np.isnan(vol_ma[t]) or vol_ma[t] <= 0:
        return None

    for m in range(min(cfg.contractions_max, len(pairs)), cfg.contractions_min - 1, -1):
        sel = pairs[-m:]
        first_high_idx = sel[0][0][0]
        base_days = t - first_high_idx
        if not (cfg.base_min_days <= base_days <= cfg.base_max_days):
            continue

        depths = [(hi[1] - lo[1]) / hi[1] for hi, lo in sel]
        if any(d <= 0 for d in depths):
            continue
        if any(depths[i + 1] > depths[i] + 1e-9 for i in range(len(depths) - 1)):
            continue
        if depths[0] > cfg.base_depth_max or depths[-1] > cfg.final_depth_max:
            continue

        pivot = sel[-1][0][1]
        base_high = max(hi[1] for hi, _ in sel)
        if pivot < base_high * (1 - cfg.pivot_below_base_high_max):
            continue

        last_low_swing = sel[-1][1]
        last_low, last_low_idx = last_low_swing[1], last_low_swing[0]
        if not (last_low < close_t < pivot):
            continue
        if (pivot - close_t) / pivot > cfg.final_depth_max:
            continue
        # already broke out since the final low -> stale base
        if np.nanmax(high[last_low_idx : t + 1]) > pivot:
            continue

        recent_vol = np.nanmean(volume[max(0, t - 4) : t + 1])
        dryup = recent_vol / vol_ma[t]
        if (
            not np.isfinite(dryup)
            or dryup < cfg.dryup_ratio_min
            or dryup > cfg.dryup_ratio_max
        ):
            continue

        return {
            "pivot": pivot,
            "last_low": last_low,
            "last_low_idx": last_low_idx,
            "first_high_idx": first_high_idx,
            "base_days": int(base_days),
            "depths": [round(d, 4) for d in depths],
            "dryup": float(dryup),
        }
    return None


def find_setups(
    symbol: str,
    dates: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    pass_idx: np.ndarray,
    cfg: Config,
) -> list[Setup]:
    """pass_idx: positional indices (into dates) of days where the screen passes."""
    k = cfg.swing_window
    swings = find_swings(high, low, k)
    if len(swings) < 2:
        return []
    vol_ma = pd.Series(volume).rolling(50, min_periods=20).mean().to_numpy()

    setups: list[Setup] = []
    last_key = None
    last_emit_t = -(10**9)
    ptr = 0

    for t in np.sort(pass_idx):
        confirm_limit = t - k
        while ptr < len(swings) and swings[ptr][0] <= confirm_limit:
            ptr += 1
        confirmed = swings[:ptr]
        if len(confirmed) < 2 or confirmed[-1][2] != "L":
            continue

        # collect trailing (H, L) pairs, newest last
        pairs: list[tuple[tuple, tuple]] = []
        j = len(confirmed) - 1
        while j >= 1 and len(pairs) < cfg.contractions_max:
            lo, hi = confirmed[j], confirmed[j - 1]
            if lo[2] != "L" or hi[2] != "H":
                break
            pairs.append((hi, lo))
            j -= 2
        if not pairs:
            continue
        pairs.reverse()

        found = _evaluate_day(int(t), pairs, high, close, volume, vol_ma, cfg)
        if found is None:
            continue

        key = (round(found["pivot"], 4), found["last_low_idx"])
        if key == last_key and t <= last_emit_t + cfg.setup_valid_days:
            continue  # same base already emitted and still valid
        last_key, last_emit_t = key, int(t)

        valid_idx = min(int(t) + cfg.setup_valid_days, len(dates) - 1)
        stop_level = max(found["last_low"], found["pivot"] * (1 - cfg.stop_max_pct))
        setups.append(
            Setup(
                symbol=symbol,
                detect_date=dates[int(t)].date(),
                pivot=found["pivot"],
                last_low=found["last_low"],
                stop_level=stop_level,
                base_start_date=dates[found["first_high_idx"]].date(),
                base_days=found["base_days"],
                n_contractions=len(found["depths"]),
                contraction_depths=found["depths"],
                dryup_ratio=round(found["dryup"], 4),
                close=float(close[t]),
                valid_until=dates[valid_idx].date(),
            )
        )
    return setups
