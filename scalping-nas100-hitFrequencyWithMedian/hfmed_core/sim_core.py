"""Numba-accelerated tick-level core for the hit-frequency median rule.

The hot loop (cross detection, exit scan, sizing, equity, ruin) is compiled with
Numba and works on plain numpy arrays. It is a 1:1 port of the previous pure
Python loop in ``simulation.run_simulation`` and produces bit-identical trades.
Per-trade bookkeeping (building ``ClosedTrade`` objects) stays in Python because
it is only O(trades), not O(ticks).

If Numba is unavailable the ``njit`` decorator degrades to a no-op and the exact
same function runs as plain (slow) Python, so the backtester stays functional.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - fallback keeps the code runnable
    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _wrap(func):
            return func

        return _wrap


# Number of per-trade output columns written by the core (kept here so the
# Python wrapper and the core agree on the array set).
@njit(cache=True)
def simulate_core(
    mid,
    bid,
    ask,
    tick_bar_index,
    median_level,
    long_cross,
    short_cross,
    profile_low,
    profile_high,
    stop_lower,
    stop_upper,
    profile_range,
    entry_allowed,
    stop_mode_fixed,
    stop_points,
    take_profit_points,
    min_profile_range_points,
    stop_buffer,
    min_stop_distance,
    max_stop_distance,
    initial_equity,
    mult,
    eurusd,
    risk_pct,
    margin_pct,
    max_margin_pct,
    lot_size,
    spread_points,
    slippage_points,
    commission_per_unit,
    out_entry_idx,
    out_exit_idx,
    out_direction,
    out_cross_level,
    out_median,
    out_prev_mid,
    out_signal_mid,
    out_entry_price,
    out_exit_price,
    out_stop_price,
    out_tp_price,
    out_units,
    out_notional,
    out_margin,
    out_gross,
    out_extra,
    out_pnl,
    out_equity_before,
    out_equity_after,
    out_status,
    out_ticks_held,
    cap,
):
    n = mid.shape[0]
    eff_rate = eurusd if eurusd > 0.0 else 1.0

    equity = initial_equity
    has_pos = False
    p_dir = 0
    p_entry_idx = 0
    p_entry_price = 0.0
    p_stop = 0.0
    p_tp = 0.0
    p_units = 0.0
    p_notional = 0.0
    p_margin = 0.0
    p_equity_before = 0.0
    p_cross_level = 0.0
    p_median = 0.0
    p_prev_mid = 0.0
    p_signal_mid = 0.0
    p_ticks_held = 0

    signals_total = 0
    long_signals = 0
    short_signals = 0
    rej_missing = 0
    rej_narrow = 0
    rej_small = 0
    rej_large = 0
    skipped_no_size = 0
    ruined = 0
    ticks_simulated = 0
    n_trades = 0

    for i in range(n):
        ticks_simulated = i + 1

        if has_pos:
            p_ticks_held += 1
            if p_dir == 1:
                exit_quote = bid[i]
                hit_stop = exit_quote <= p_stop
                hit_tp = exit_quote >= p_tp
            else:
                exit_quote = ask[i]
                hit_stop = exit_quote >= p_stop
                hit_tp = exit_quote <= p_tp

            if hit_stop or hit_tp:
                status = 0 if hit_stop else 1
                sign = 1.0 if p_dir == 1 else -1.0
                fill_price = p_tp if status == 1 else exit_quote
                gross = (p_units * mult * (fill_price - p_entry_price) * sign) / eff_rate
                extra = (p_units * (spread_points + slippage_points) * mult) / eff_rate
                extra += commission_per_unit * p_units * 2.0
                pnl = gross - extra
                equity_after = equity + pnl
                if n_trades < cap:
                    out_entry_idx[n_trades] = p_entry_idx
                    out_exit_idx[n_trades] = i
                    out_direction[n_trades] = p_dir
                    out_cross_level[n_trades] = p_cross_level
                    out_median[n_trades] = p_median
                    out_prev_mid[n_trades] = p_prev_mid
                    out_signal_mid[n_trades] = p_signal_mid
                    out_entry_price[n_trades] = p_entry_price
                    out_exit_price[n_trades] = fill_price
                    out_stop_price[n_trades] = p_stop
                    out_tp_price[n_trades] = p_tp
                    out_units[n_trades] = p_units
                    out_notional[n_trades] = p_notional
                    out_margin[n_trades] = p_margin
                    out_gross[n_trades] = gross
                    out_extra[n_trades] = extra
                    out_pnl[n_trades] = pnl
                    out_equity_before[n_trades] = p_equity_before
                    out_equity_after[n_trades] = equity_after
                    out_status[n_trades] = status
                    out_ticks_held[n_trades] = p_ticks_held
                n_trades += 1
                equity = equity_after
                has_pos = False
                if equity <= 0.0:
                    ruined = 1
                    break
            continue

        if i == 0:
            continue
        if entry_allowed[i] == 0:
            continue

        prev_mid = mid[i - 1]
        m = mid[i]
        bar_i = tick_bar_index[i]
        lc = np.nan
        sc = np.nan
        if 0 <= bar_i < long_cross.shape[0]:
            lc = long_cross[bar_i]
            sc = short_cross[bar_i]
        direction = 0
        cross_level = np.nan
        if (lc == lc) and (prev_mid < lc) and (lc <= m):
            direction = 1
            cross_level = lc
        elif (sc == sc) and (prev_mid > sc) and (sc >= m):
            direction = -1
            cross_level = sc
        if direction == 0:
            continue

        signals_total += 1
        if direction == 1:
            long_signals += 1
        else:
            short_signals += 1

        entry_price = ask[i] if direction == 1 else bid[i]

        stop_price = 0.0
        tp_price = 0.0
        stop_distance = 0.0
        valid = True
        reject = 0
        if stop_mode_fixed == 1:
            stop_distance = stop_points
            if direction == 1:
                stop_price = entry_price - stop_points
                tp_price = entry_price + take_profit_points
            else:
                stop_price = entry_price + stop_points
                tp_price = entry_price - take_profit_points
        else:
            pl = np.nan
            ph = np.nan
            sl = np.nan
            su = np.nan
            pr = np.nan
            if 0 <= bar_i < profile_low.shape[0]:
                pl = profile_low[bar_i]
                ph = profile_high[bar_i]
                sl = stop_lower[bar_i]
                su = stop_upper[bar_i]
                pr = profile_range[bar_i]
            if not ((pl == pl) and (ph == ph) and (sl == sl) and (su == su) and (pr == pr)):
                valid = False
                reject = 1
            elif pr < min_profile_range_points:
                valid = False
                reject = 2
            else:
                if direction == 1:
                    stop_price = sl - stop_buffer
                    tp_price = entry_price + take_profit_points
                    stop_distance = entry_price - stop_price
                else:
                    stop_price = su + stop_buffer
                    tp_price = entry_price - take_profit_points
                    stop_distance = stop_price - entry_price
                if stop_distance < min_stop_distance:
                    valid = False
                    reject = 3
                elif stop_distance > max_stop_distance:
                    valid = False
                    reject = 4

        if not valid:
            if reject == 1:
                rej_missing += 1
            elif reject == 2:
                rej_narrow += 1
            elif reject == 3:
                rej_small += 1
            elif reject == 4:
                rej_large += 1
            continue

        units = 0.0
        notional = 0.0
        margin = 0.0
        if stop_distance > 0.0 and entry_price > 0.0 and equity > 0.0:
            risk_budget = equity * (risk_pct / 100.0)
            risk_per_unit = (stop_distance * mult) / eff_rate
            if risk_per_unit > 0.0:
                units = risk_budget / risk_per_unit
                notional = (units * entry_price * mult) / eff_rate
                margin = notional * (margin_pct / 100.0)
                max_margin = equity * (max_margin_pct / 100.0)
                if margin > max_margin and margin > 0.0:
                    units *= max_margin / margin
                if units < lot_size:
                    units = 0.0
                else:
                    units = math.floor((units + 1e-12) / lot_size) * lot_size
                if units > 0.0:
                    notional = (units * entry_price * mult) / eff_rate
                    margin = notional * (margin_pct / 100.0)
        if units <= 0.0:
            skipped_no_size += 1
            continue

        has_pos = True
        p_dir = direction
        p_entry_idx = i
        p_entry_price = entry_price
        p_stop = stop_price
        p_tp = tp_price
        p_units = units
        p_notional = notional
        p_margin = margin
        p_equity_before = equity
        p_cross_level = cross_level
        p_median = median_level[bar_i] if 0 <= bar_i < median_level.shape[0] else np.nan
        p_prev_mid = prev_mid
        p_signal_mid = m
        p_ticks_held = 0

    if has_pos and n > 0:
        i = n - 1
        exit_quote = bid[i] if p_dir == 1 else ask[i]
        sign = 1.0 if p_dir == 1 else -1.0
        gross = (p_units * mult * (exit_quote - p_entry_price) * sign) / eff_rate
        extra = (p_units * (spread_points + slippage_points) * mult) / eff_rate
        extra += commission_per_unit * p_units * 2.0
        pnl = gross - extra
        equity_after = equity + pnl
        if n_trades < cap:
            out_entry_idx[n_trades] = p_entry_idx
            out_exit_idx[n_trades] = i
            out_direction[n_trades] = p_dir
            out_cross_level[n_trades] = p_cross_level
            out_median[n_trades] = p_median
            out_prev_mid[n_trades] = p_prev_mid
            out_signal_mid[n_trades] = p_signal_mid
            out_entry_price[n_trades] = p_entry_price
            out_exit_price[n_trades] = exit_quote
            out_stop_price[n_trades] = p_stop
            out_tp_price[n_trades] = p_tp
            out_units[n_trades] = p_units
            out_notional[n_trades] = p_notional
            out_margin[n_trades] = p_margin
            out_gross[n_trades] = gross
            out_extra[n_trades] = extra
            out_pnl[n_trades] = pnl
            out_equity_before[n_trades] = p_equity_before
            out_equity_after[n_trades] = equity_after
            out_status[n_trades] = 2
            out_ticks_held[n_trades] = p_ticks_held
        n_trades += 1
        equity = equity_after

    return (
        n_trades,
        signals_total,
        long_signals,
        short_signals,
        rej_missing,
        rej_narrow,
        rej_small,
        rej_large,
        skipped_no_size,
        ruined,
        ticks_simulated,
        equity,
    )


def warmup() -> None:
    """Force JIT compilation once (cheap) so worker processes inherit the cache."""
    n = 3
    zeros = np.zeros(n, dtype=np.float64)
    bar_index = np.arange(n, dtype=np.int32)
    allowed = np.zeros(n, dtype=np.uint8)
    out_i = np.zeros(1, dtype=np.int64)
    out_d = np.zeros(1, dtype=np.int8)
    out_f = np.zeros(1, dtype=np.float64)
    simulate_core(
        zeros, zeros, zeros, bar_index, zeros, zeros, zeros, zeros, zeros, zeros, zeros, zeros,
        allowed,
        0, 10.0, 10.0, 0.0, 1.0, 1.0, 1e18,
        5000.0, 1.0, 1.0, 1.5, 5.0, 45.0, 0.1, 0.0, 0.0, 0.0,
        out_i.copy(), out_i.copy(), out_d.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_d.copy(),
        out_i.copy(),
        0,
    )
