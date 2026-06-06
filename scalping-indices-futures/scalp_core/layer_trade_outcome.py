"""Trade-outcome labels for the scalping decision layer.

The decision model predicts whether a full trade setup wins after costs, not
whether the next minute closes up. Labels are generated with the same entry,
stop, take-profit, trailing, max-hold and session-flat rules used by simulation.
"""

from dataclasses import dataclass
from datetime import time
from typing import Optional

import numpy as np

from . import broker, config


@dataclass(frozen=True)
class TradePlan:
    stop_pct: float
    tp_pct: float
    trail_activation_pct: float
    trail_distance_pct: float
    sigma_pts: float
    basis_pts: float
    atr_pts: float


@dataclass(frozen=True)
class TradeOutcome:
    net_r: float
    exit_idx: int
    exit_price: float
    outcome_status: str
    bars_held: int


@dataclass(frozen=True)
class OutcomeCache:
    long_valid: np.ndarray
    long_win: np.ndarray
    long_net_r: np.ndarray
    long_exit_idx: np.ndarray
    short_valid: np.ndarray
    short_win: np.ndarray
    short_net_r: np.ndarray
    short_exit_idx: np.ndarray


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def make_trade_plan(cur_price: float, sigma_ret: float, atr_pts: float) -> Optional[TradePlan]:
    """Build stop/TP distances available at the signal close."""
    if cur_price <= 0:
        return None

    sigma_pts = sigma_ret * cur_price if (np.isfinite(sigma_ret) and sigma_ret > 0) else 0.0
    if config.STOP_MODE == "vol":
        if sigma_pts <= 0:
            return None
        basis = sigma_pts
        stop_dist = config.STOP_VOL_MULT * basis
        tp_dist = config.TP_VOL_MULT * basis
    else:
        if not (np.isfinite(atr_pts) and atr_pts > 0):
            return None
        basis = atr_pts
        stop_dist = config.STOP_ATR_MULT * basis
        tp_dist = config.TP_ATR_MULT * basis
        if sigma_pts <= 0:
            sigma_pts = basis

    stop_pct = _clamp(stop_dist / cur_price * 100.0, config.MIN_STOP_PCT, config.MAX_STOP_PCT)
    tp_pct = max(tp_dist / cur_price * 100.0, stop_pct)
    return TradePlan(
        stop_pct=stop_pct,
        tp_pct=tp_pct,
        trail_activation_pct=config.TRAILING_ACTIVATION_MULT * basis / cur_price * 100.0,
        trail_distance_pct=max(config.TRAILING_DISTANCE_MULT * basis / cur_price * 100.0, 1e-6),
        sigma_pts=sigma_pts,
        basis_pts=basis,
        atr_pts=atr_pts,
    )


def _net_r(entry_price: float, exit_price: float, stop_price: float, direction: str) -> float:
    gross = broker.gross_pnl(1.0, entry_price, exit_price, direction)
    costs = broker.round_trip_costs(1.0, entry_price, exit_price)
    risk = abs(broker.gross_pnl(1.0, entry_price, stop_price, direction))
    if risk <= 0:
        return 0.0
    return (gross - costs) / risk


def simulate_trade_outcome(
    signal_idx: int,
    direction: str,
    plan: TradePlan,
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    session_date: np.ndarray,
    local_time: np.ndarray,
    cutoff: time,
) -> Optional[TradeOutcome]:
    """Simulate one candidate from signal close to eventual exit."""
    entry_idx = signal_idx + 1
    if entry_idx >= len(c):
        return None

    entry_price = float(o[entry_idx])
    if entry_price <= 0:
        return None

    if direction == "LONG":
        stop_price = entry_price * (1.0 - plan.stop_pct / 100.0)
        tp_price = entry_price * (1.0 + plan.tp_pct / 100.0)
        trail_activation_price = entry_price * (1.0 + plan.trail_activation_pct / 100.0)
    else:
        stop_price = entry_price * (1.0 + plan.stop_pct / 100.0)
        tp_price = entry_price * (1.0 - plan.tp_pct / 100.0)
        trail_activation_price = entry_price * (1.0 - plan.trail_activation_pct / 100.0)

    trail_active = False
    trailing_stop = None
    extreme = entry_price

    for idx in range(entry_idx, len(c)):
        bars_held = idx - entry_idx + 1

        if config.TP_MODE == "fixed":
            stop_first = config.INTRABAR_FILL_PRIORITY == "stop"
            if direction == "LONG":
                hit_stop, hit_tp = low[idx] <= stop_price, h[idx] >= tp_price
            else:
                hit_stop, hit_tp = h[idx] >= stop_price, low[idx] <= tp_price
            order = [("stop", hit_stop, stop_price, "HIT_SL"), ("tp", hit_tp, tp_price, "HIT_TP")]
            if not stop_first:
                order.reverse()
            for _kind, hit, px, status in order:
                if hit:
                    return TradeOutcome(_net_r(entry_price, px, stop_price, direction), idx, px, status, bars_held)
        else:
            eff_stop = trailing_stop if (trail_active and trailing_stop is not None) else stop_price
            if direction == "LONG":
                if low[idx] <= eff_stop:
                    status = "HIT_TRAILING_STOP" if trail_active else "HIT_SL"
                    return TradeOutcome(_net_r(entry_price, eff_stop, stop_price, direction), idx, eff_stop, status, bars_held)
                extreme = max(extreme, h[idx])
                if not trail_active and h[idx] >= trail_activation_price:
                    trail_active = True
                if trail_active:
                    new_stop = extreme * (1.0 - plan.trail_distance_pct / 100.0)
                    trailing_stop = new_stop if trailing_stop is None else max(trailing_stop, new_stop)
            else:
                if h[idx] >= eff_stop:
                    status = "HIT_TRAILING_STOP" if trail_active else "HIT_SL"
                    return TradeOutcome(_net_r(entry_price, eff_stop, stop_price, direction), idx, eff_stop, status, bars_held)
                extreme = min(extreme, low[idx])
                if not trail_active and low[idx] <= trail_activation_price:
                    trail_active = True
                if trail_active:
                    new_stop = extreme * (1.0 + plan.trail_distance_pct / 100.0)
                    trailing_stop = new_stop if trailing_stop is None else min(trailing_stop, new_stop)

        new_session = session_date[idx] != session_date[entry_idx]
        if new_session or local_time[idx] >= cutoff:
            px = float(c[idx])
            return TradeOutcome(_net_r(entry_price, px, stop_price, direction), idx, px, "SESSION_FLAT", bars_held)
        if bars_held >= config.MAX_HOLD_BARS:
            px = float(c[idx])
            return TradeOutcome(_net_r(entry_price, px, stop_price, direction), idx, px, "MAX_HOLD", bars_held)

    px = float(c[-1])
    return TradeOutcome(_net_r(entry_price, px, stop_price, direction), len(c) - 1, px, "SESSION_FLAT", len(c) - entry_idx)


def build_atr_outcome_cache(
    o: np.ndarray,
    h: np.ndarray,
    low: np.ndarray,
    c: np.ndarray,
    atr: np.ndarray,
    session_date: np.ndarray,
    local_time: np.ndarray,
    entry_start: time,
    entry_end: time,
    cutoff: time,
) -> OutcomeCache:
    """Precompute ATR-based long/short labels once per run."""
    n = len(c)
    long_valid = np.zeros(n, dtype=bool)
    long_win = np.zeros(n, dtype=np.float64)
    long_net_r = np.full(n, np.nan, dtype=np.float64)
    long_exit_idx = np.full(n, -1, dtype=np.int64)
    short_valid = np.zeros(n, dtype=bool)
    short_win = np.zeros(n, dtype=np.float64)
    short_net_r = np.full(n, np.nan, dtype=np.float64)
    short_exit_idx = np.full(n, -1, dtype=np.int64)

    for idx in range(0, max(0, n - 1)):
        if local_time[idx] < entry_start or local_time[idx] >= entry_end:
            continue
        plan = make_trade_plan(float(c[idx]), np.nan, float(atr[idx]))
        if plan is None:
            continue

        long_out = simulate_trade_outcome(idx, "LONG", plan, o, h, low, c, session_date, local_time, cutoff)
        if long_out is not None:
            long_valid[idx] = True
            long_win[idx] = 1.0 if long_out.net_r > 0.0 else 0.0
            long_net_r[idx] = long_out.net_r
            long_exit_idx[idx] = long_out.exit_idx

        if config.ALLOW_SHORT:
            short_out = simulate_trade_outcome(idx, "SHORT", plan, o, h, low, c, session_date, local_time, cutoff)
            if short_out is not None:
                short_valid[idx] = True
                short_win[idx] = 1.0 if short_out.net_r > 0.0 else 0.0
                short_net_r[idx] = short_out.net_r
                short_exit_idx[idx] = short_out.exit_idx

    return OutcomeCache(
        long_valid=long_valid,
        long_win=long_win,
        long_net_r=long_net_r,
        long_exit_idx=long_exit_idx,
        short_valid=short_valid,
        short_win=short_win,
        short_net_r=short_net_r,
        short_exit_idx=short_exit_idx,
    )
