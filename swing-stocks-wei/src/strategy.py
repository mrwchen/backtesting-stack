"""Signal logic: per-stock EMA trend filter x market-wide stress gate,
plus the category-momentum sizing tiers.

Per-stock rule (decided at each trading day's close):

    flat  if EMA_fast < EMA_slow  AND  stress gate is ON
    long  otherwise

The stress gate is one market-wide hysteresis state driven by the previous
day's composite score: ON at >= stress_enter, OFF again at <= stress_exit.

Sizing tiers by the stock's category momentum at entry (research result: entries
into beaten-down categories are the robust sweet spot in-sample AND
out-of-sample; positive-momentum entries are regime-dependent, so they are
down-weighted rather than banned):

    deep  if momentum <= deep_threshold (e.g. -10%)   -> full weight
    mild  if deep_threshold < momentum <= 0            -> normal weight
    pos   if momentum > 0                              -> reduced weight

Unknown momentum (NaN, e.g. not enough history yet) falls back to 'mild'.
"""
from __future__ import annotations

import numpy as np


def ema(values: np.ndarray, span: int) -> np.ndarray:
    """Standard EMA (alpha = 2/(span+1)), seeded with the first value.

    NaN-tolerant: leading NaNs stay NaN, later NaNs carry the previous EMA.
    """
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(values), np.nan, dtype=float)
    prev = np.nan
    for i, v in enumerate(values):
        if np.isnan(prev):
            prev = v
        elif not np.isnan(v):
            prev = prev + alpha * (v - prev)
        out[i] = prev
    return out


def stress_gate(lagged_scores: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """Hysteresis state per trading day. NaN scores keep the previous state."""
    on = np.zeros(len(lagged_scores), dtype=bool)
    cur = False
    for i, score in enumerate(lagged_scores):
        if not np.isnan(score):
            if not cur and score >= enter:
                cur = True
            elif cur and score <= exit_:
                cur = False
        on[i] = cur
    return on


def stock_positions(closes: np.ndarray, stress_on: np.ndarray,
                    ema_fast: int, ema_slow: int) -> np.ndarray:
    """Per-day signal state for one stock. position[t] is decided at close t.

    Days before the stock has any data (leading NaN closes) are position 0.
    """
    ema_f = ema(closes, ema_fast)
    ema_s = ema(closes, ema_slow)
    ema_bearish = np.where(np.isnan(ema_f) | np.isnan(ema_s), True, ema_f < ema_s)
    no_data = np.isnan(closes) & np.isnan(ema_f)
    pos = np.where(no_data | (ema_bearish & stress_on), 0, 1).astype(np.int8)
    return pos


def stock_momentum(closes: np.ndarray, window: int) -> np.ndarray:
    """Trailing per-stock momentum: (n_days, n_symbols) change over `window` rows.

    Rows before the window has data are NaN. Used as the deterministic
    tie-breaker for entry ranking (ties in category momentum are common on
    mass-entry days when the stress gate opens).
    """
    out = np.full_like(closes, np.nan, dtype=float)
    if closes.shape[0] > window:
        with np.errstate(invalid="ignore", divide="ignore"):
            out[window:] = closes[window:] / closes[:-window] - 1.0
    return out


def entry_candidates(positions: np.ndarray, confirm_days: int) -> np.ndarray:
    """(n_days, n_symbols) bool: stock is an entry candidate on day t.

    confirm_days=0: candidate on the flat->long flip day (original behaviour,
    day 0 counts as a flip for stocks that start long).
    confirm_days=N: candidate N trading days after the flip, and only if the
    signal stayed long the entire time. This skips the whipsaw cohort that
    flips right back after the stress gate opens.
    """
    n_days, n_sym = positions.shape
    flip = np.zeros((n_days, n_sym), dtype=bool)
    flip[0] = positions[0] == 1
    flip[1:] = (positions[1:] == 1) & (positions[:-1] == 0)
    if confirm_days <= 0:
        return flip
    cand = np.zeros_like(flip)
    if n_days > confirm_days:
        stayed = np.ones((n_days - confirm_days, n_sym), dtype=bool)
        for k in range(confirm_days + 1):
            stayed &= positions[k:n_days - confirm_days + k] == 1
        cand[confirm_days:] = flip[:n_days - confirm_days] & stayed
    return cand


def sizing_tier(cat_momentum: float, deep_threshold: float) -> str:
    """Map category momentum at entry to a sizing tier: deep | mild | pos."""
    if np.isnan(cat_momentum):
        return "mild"
    if cat_momentum <= deep_threshold:
        return "deep"
    if cat_momentum <= 0.0:
        return "mild"
    return "pos"
