"""VCP (Volatility Contraction Pattern) detection on daily bars.

A base is a sequence of 2-4 (swing high -> swing low) contractions with
non-increasing depth, the last one tight (<= FINAL_DEPTH_MAX), volume drying
up, and price sitting just below the pivot (= high of the final contraction).

Look-ahead discipline: a swing at bar i is confirmed only k bars later.  Swing
candidates are therefore applied in confirmation order while the evaluation
clock advances; a later same-kind extreme can change future state, but it can
never rewrite the swing state used by an already emitted setup.  Price must
also not have exceeded the pivot since the final swing low.

Symbol gaps are structural barriers. Swing state and volume windows restart
after a missing global market session, so a halt cannot make an old base look
young by pausing the symbol-local bar count.
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


Swing = tuple[int, float, str]


def _candidate_at(
    high: np.ndarray,
    low: np.ndarray,
    i: int,
    k: int,
) -> tuple[Swing | None, bool]:
    """Return the swing at ``i`` and whether the bar is an outside-bar barrier.

    The first bar of an equal-price plateau wins: the centre must be strictly
    more extreme than the bars to its left and at least as extreme as the bars
    to its right.  A bar that is both a local high and local low has unknowable
    intraday ordering in daily OHLC data.  Treating it as a barrier avoids
    inventing either an H->L or L->H sequence.
    """
    if k < 1:
        raise ValueError("swing window must be >= 1")
    if i < k or i + k >= len(high):
        return None, False

    high_window = high[i - k : i + k + 1]
    low_window = low[i - k : i + k + 1]
    if not np.isfinite(high_window).all() or not np.isfinite(low_window).all():
        return None, False

    high_i = float(high[i])
    low_i = float(low[i])
    is_high = high_i > float(np.max(high[i - k : i])) and high_i >= float(
        np.max(high[i + 1 : i + k + 1])
    )
    is_low = low_i < float(np.min(low[i - k : i])) and low_i <= float(
        np.min(low[i + 1 : i + k + 1])
    )
    if is_high and is_low:
        return None, True
    if is_high:
        return (i, high_i, "H"), False
    if is_low:
        return (i, low_i, "L"), False
    return None, False


def _apply_candidate(swings: list[Swing], event: Swing) -> None:
    """Apply one newly confirmed event to the alternating as-of swing state."""
    if not swings or swings[-1][2] != event[2]:
        swings.append(event)
        return

    previous = swings[-1]
    more_extreme = event[1] > previous[1] if event[2] == "H" else event[1] < previous[1]
    if more_extreme:
        swings[-1] = event


def find_swings(high: np.ndarray, low: np.ndarray, k: int) -> list[Swing]:
    """Return alternating swings after causal, confirmation-order processing."""
    if len(high) != len(low):
        raise ValueError("high and low must have equal length")
    if k < 1:
        raise ValueError("swing window must be >= 1")

    swings: list[Swing] = []
    # Candidate i is knowable at confirmation bar i+k.  Iterating centres is
    # equivalent to iterating confirmation bars and never consults a later
    # candidate before it becomes available.
    for i in range(k, len(high) - k):
        event, barrier = _candidate_at(high, low, i, k)
        if barrier:
            swings.clear()
        elif event is not None:
            _apply_candidate(swings, event)
    return swings


def _evaluate_day(
    t: int,
    pairs: list[tuple[tuple, tuple]],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    vol_ma: np.ndarray,
    session_positions: np.ndarray,
    segment_ids: np.ndarray,
    cfg: Config,
) -> dict | None:
    close_t = close[t]
    if np.isnan(close_t) or np.isnan(vol_ma[t]) or vol_ma[t] <= 0:
        return None

    for m in range(min(cfg.contractions_max, len(pairs)), cfg.contractions_min - 1, -1):
        sel = pairs[-m:]
        first_high_idx = sel[0][0][0]
        base_days = int(session_positions[t] - session_positions[first_high_idx])
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
        invalidation_level = max(last_low, pivot * (1 - cfg.stop_max_pct))
        later_lows = low[last_low_idx + 1 : t + 1]
        if len(later_lows) and (
            not np.isfinite(later_lows).all()
            or np.min(later_lows) <= invalidation_level
        ):
            continue
        if (pivot - close_t) / pivot > cfg.final_depth_max:
            continue
        # already broke out since the final low -> stale base
        if np.nanmax(high[last_low_idx : t + 1]) > pivot:
            continue

        recent_start = max(0, t - 4)
        if segment_ids[recent_start] != segment_ids[t]:
            continue
        recent_vol = np.nanmean(volume[recent_start : t + 1])
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
    *,
    trading_dates: pd.DatetimeIndex,
) -> list[Setup]:
    """Detect setups on symbol bars and expire them on the global session grid.

    ``pass_idx`` contains positional indices into the symbol-local ``dates``.
    ``trading_dates`` is the global ordered bar calendar shared by the
    simulation.  Setup validity must use that calendar so a symbol suspension
    cannot pause the expiry clock.
    """
    k = cfg.swing_window
    n = len(dates)
    if not (len(high) == len(low) == len(close) == len(volume) == n):
        raise ValueError("dates and OHLCV arrays must have equal length")
    if k < 1:
        raise ValueError("swing window must be >= 1")
    trading_dates = pd.DatetimeIndex(trading_dates)
    if trading_dates.empty:
        raise ValueError("trading_dates must not be empty")
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")
    if len(dates.difference(trading_dates)):
        raise ValueError("every symbol date must exist in trading_dates")
    session_positions = trading_dates.get_indexer(dates)
    complete_bars = (
        np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(volume)
    )
    segment_starts = np.r_[
        True,
        (np.diff(session_positions) != 1)
        | ~complete_bars[1:]
        | ~complete_bars[:-1],
    ]
    segment_ids = np.cumsum(segment_starts)
    vol_ma = np.full(n, np.nan, dtype=float)
    for segment_id in np.unique(segment_ids):
        idx = np.flatnonzero(segment_ids == segment_id)
        vol_ma[idx] = (
            pd.Series(volume[idx]).rolling(50, min_periods=20).mean().to_numpy()
        )

    setups: list[Setup] = []
    # The terminal H/L pair is the stable identity of a tradable structure.
    # As the base ages, _evaluate_day may drop an older leading contraction;
    # that must not resurrect the unchanged final pivot/low as a new setup.
    emitted_terminal_pairs: set[tuple[int, int]] = set()
    confirmed: list[Swing] = []
    pass_days = {int(t) for t in pass_idx if 0 <= int(t) < n}
    if not pass_days:
        return []

    # Advance every market day, not just screen-pass days: swings confirmed
    # between two pass days still belong to the later day's as-of state.
    for t in range(max(pass_days) + 1):
        if not complete_bars[t]:
            confirmed.clear()
            continue
        if t > 0 and segment_ids[t] != segment_ids[t - 1]:
            # A missing market session makes swing ordering and structure age
            # discontinuous. Never carry a pre-halt chain across the gap.
            confirmed.clear()
        candidate_idx = t - k
        if candidate_idx >= k:
            candidate_window = session_positions[
                candidate_idx - k : candidate_idx + k + 1
            ]
            complete_window = complete_bars[
                candidate_idx - k : candidate_idx + k + 1
            ]
            if (
                len(candidate_window) != 2 * k + 1
                or np.any(np.diff(candidate_window) != 1)
                or not complete_window.all()
            ):
                event, barrier = None, True
            else:
                event, barrier = _candidate_at(high, low, candidate_idx, k)
            if barrier:
                confirmed.clear()
            elif event is not None:
                _apply_candidate(confirmed, event)

        if t not in pass_days or len(confirmed) < 2 or confirmed[-1][2] != "L":
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

        found = _evaluate_day(
            int(t),
            pairs,
            high,
            low,
            close,
            volume,
            vol_ma,
            session_positions,
            segment_ids,
            cfg,
        )
        if found is None:
            continue

        selected_pairs = pairs[-len(found["depths"]) :]
        terminal_pair = (int(selected_pairs[-1][0][0]), int(selected_pairs[-1][1][0]))
        if terminal_pair in emitted_terminal_pairs:
            continue
        emitted_terminal_pairs.add(terminal_pair)

        detect_ts = dates[int(t)]
        detect_session_idx = int(trading_dates.searchsorted(detect_ts))
        valid_idx = min(
            detect_session_idx + cfg.setup_valid_days,
            len(trading_dates) - 1,
        )
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
                valid_until=trading_dates[valid_idx].date(),
            )
        )
    return setups
