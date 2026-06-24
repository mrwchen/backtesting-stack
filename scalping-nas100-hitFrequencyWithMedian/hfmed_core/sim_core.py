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
    long_cross_quantile,
    short_cross_quantile,
    entry_price_range_position_max_deviation_pct,
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
    out_margin_capped,
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
    p_margin_capped = 0

    signals_total = 0
    long_signals = 0
    short_signals = 0
    rej_missing = 0
    rej_narrow = 0
    rej_price_range_position = 0
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
                    out_margin_capped[n_trades] = p_margin_capped
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

        pl = np.nan
        ph = np.nan
        pr = np.nan
        if 0 <= bar_i < profile_low.shape[0]:
            pl = profile_low[bar_i]
            ph = profile_high[bar_i]
            pr = profile_range[bar_i]
        if not ((pl == pl) and (ph == ph) and (pr == pr) and pr > 0.0):
            rej_missing += 1
            continue
        if stop_mode_fixed != 1 and pr < min_profile_range_points:
            rej_narrow += 1
            continue
        expected_position_pct = long_cross_quantile * 100.0 if direction == 1 else short_cross_quantile * 100.0
        cross_position_pct = ((cross_level - pl) / pr) * 100.0
        if not (cross_position_pct == cross_position_pct) or abs(cross_position_pct - expected_position_pct) > entry_price_range_position_max_deviation_pct:
            rej_price_range_position += 1
            continue

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
            sl = np.nan
            su = np.nan
            if 0 <= bar_i < profile_low.shape[0]:
                sl = stop_lower[bar_i]
                su = stop_upper[bar_i]
            if not ((sl == sl) and (su == su)):
                valid = False
                reject = 1
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
        margin_capped = 0
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
                    margin_capped = 1
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
        p_margin_capped = margin_capped

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
            out_margin_capped[n_trades] = p_margin_capped
        n_trades += 1
        equity = equity_after

    return (
        n_trades,
        signals_total,
        long_signals,
        short_signals,
        rej_missing,
        rej_narrow,
        rej_price_range_position,
        rej_small,
        rej_large,
        skipped_no_size,
        ruined,
        ticks_simulated,
        equity,
    )


@njit(cache=True)
def simulate_session_portfolio_core(
    mid,
    bid,
    ask,
    tick_bar_index,
    entry_session_code,
    median_level_by_slot,
    long_cross_by_slot,
    short_cross_by_slot,
    profile_low_by_slot,
    profile_high_by_slot,
    stop_lower_by_slot,
    stop_upper_by_slot,
    profile_range_by_slot,
    session_to_slot,
    entry_allowed,
    stop_mode_fixed_by_slot,
    stop_points_by_slot,
    take_profit_points_by_slot,
    min_profile_range_points_by_slot,
    long_cross_quantile_by_slot,
    short_cross_quantile_by_slot,
    entry_price_range_position_max_deviation_pct_by_slot,
    stop_buffer_by_slot,
    min_stop_distance_by_slot,
    max_stop_distance_by_slot,
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
    out_slot,
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
    out_margin_capped,
    cap,
):
    n = mid.shape[0]
    eff_rate = eurusd if eurusd > 0.0 else 1.0

    equity = initial_equity
    has_pos = False
    p_dir = 0
    p_slot = 0
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
    p_margin_capped = 0

    signals_total = 0
    long_signals = 0
    short_signals = 0
    rej_missing = 0
    rej_narrow = 0
    rej_price_range_position = 0
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
                    out_slot[n_trades] = p_slot
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
                    out_margin_capped[n_trades] = p_margin_capped
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

        code = int(entry_session_code[i])
        if code < 0 or code >= session_to_slot.shape[0]:
            continue
        slot = int(session_to_slot[code])
        if slot < 0:
            continue

        prev_mid = mid[i - 1]
        m = mid[i]
        bar_i = tick_bar_index[i]
        lc = np.nan
        sc = np.nan
        if 0 <= bar_i < long_cross_by_slot.shape[1]:
            lc = long_cross_by_slot[slot, bar_i]
            sc = short_cross_by_slot[slot, bar_i]
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

        pl = np.nan
        ph = np.nan
        pr = np.nan
        if 0 <= bar_i < profile_low_by_slot.shape[1]:
            pl = profile_low_by_slot[slot, bar_i]
            ph = profile_high_by_slot[slot, bar_i]
            pr = profile_range_by_slot[slot, bar_i]
        if not ((pl == pl) and (ph == ph) and (pr == pr) and pr > 0.0):
            rej_missing += 1
            continue
        if stop_mode_fixed_by_slot[slot] != 1 and pr < min_profile_range_points_by_slot[slot]:
            rej_narrow += 1
            continue
        expected_position_pct = (
            long_cross_quantile_by_slot[slot] * 100.0
            if direction == 1
            else short_cross_quantile_by_slot[slot] * 100.0
        )
        cross_position_pct = ((cross_level - pl) / pr) * 100.0
        if not (cross_position_pct == cross_position_pct) or abs(cross_position_pct - expected_position_pct) > entry_price_range_position_max_deviation_pct_by_slot[slot]:
            rej_price_range_position += 1
            continue

        entry_price = ask[i] if direction == 1 else bid[i]

        stop_price = 0.0
        tp_price = 0.0
        stop_distance = 0.0
        valid = True
        reject = 0
        if stop_mode_fixed_by_slot[slot] == 1:
            stop_distance = stop_points_by_slot[slot]
            if direction == 1:
                stop_price = entry_price - stop_distance
                tp_price = entry_price + take_profit_points_by_slot[slot]
            else:
                stop_price = entry_price + stop_distance
                tp_price = entry_price - take_profit_points_by_slot[slot]
        else:
            sl = np.nan
            su = np.nan
            if 0 <= bar_i < profile_low_by_slot.shape[1]:
                sl = stop_lower_by_slot[slot, bar_i]
                su = stop_upper_by_slot[slot, bar_i]
            if not ((sl == sl) and (su == su)):
                valid = False
                reject = 1
            else:
                if direction == 1:
                    stop_price = sl - stop_buffer_by_slot[slot]
                    tp_price = entry_price + take_profit_points_by_slot[slot]
                    stop_distance = entry_price - stop_price
                else:
                    stop_price = su + stop_buffer_by_slot[slot]
                    tp_price = entry_price - take_profit_points_by_slot[slot]
                    stop_distance = stop_price - entry_price
                if stop_distance < min_stop_distance_by_slot[slot]:
                    valid = False
                    reject = 3
                elif stop_distance > max_stop_distance_by_slot[slot]:
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
        margin_capped = 0
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
                    margin_capped = 1
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
        p_slot = slot
        p_entry_idx = i
        p_entry_price = entry_price
        p_stop = stop_price
        p_tp = tp_price
        p_units = units
        p_notional = notional
        p_margin = margin
        p_equity_before = equity
        p_cross_level = cross_level
        p_median = median_level_by_slot[slot, bar_i] if 0 <= bar_i < median_level_by_slot.shape[1] else np.nan
        p_prev_mid = prev_mid
        p_signal_mid = m
        p_ticks_held = 0
        p_margin_capped = margin_capped

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
            out_slot[n_trades] = p_slot
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
            out_margin_capped[n_trades] = p_margin_capped
        n_trades += 1
        equity = equity_after

    return (
        n_trades,
        signals_total,
        long_signals,
        short_signals,
        rej_missing,
        rej_narrow,
        rej_price_range_position,
        rej_small,
        rej_large,
        skipped_no_size,
        ruined,
        ticks_simulated,
        equity,
    )


@njit(cache=True)
def precompute_crossing_events(
    mid,
    tick_bar_index,
    long_cross,
    short_cross,
    entry_allowed,
    out_tick,
    out_dir,
    out_cross,
    cap,
):
    """Detect all profile crossings once. Candidate-independent within a profile
    group (long/short cross levels and entry_allowed are shared), so the per-tick
    cross scan is done once per group instead of once per candidate.

    The detection logic is a 1:1 copy of the flat-state branch in ``simulate_core``;
    because crossing detection depends only on mid[i-1], mid[i] and the (shared)
    cross levels, acting on these events only while flat is bit-identical to
    detecting crossings only while flat.
    """
    n = mid.shape[0]
    k = 0
    for i in range(n):
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
        if k < cap:
            out_tick[k] = i
            out_dir[k] = direction
            out_cross[k] = cross_level
        k += 1
    return k


@njit(cache=True)
def simulate_core_from_events(
    ev_tick,
    ev_dir,
    ev_cross,
    n_events,
    mid,
    bid,
    ask,
    tick_bar_index,
    median_level,
    profile_low,
    profile_high,
    stop_lower,
    stop_upper,
    profile_range,
    n_ticks,
    stop_mode_fixed,
    stop_points,
    take_profit_points,
    min_profile_range_points,
    long_cross_quantile,
    short_cross_quantile,
    entry_price_range_position_max_deviation_pct,
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
    out_margin_capped,
    cap,
):
    """Event-driven equivalent of ``simulate_core``. Walks precomputed crossing
    events (always evaluated while flat) and scans ticks only over holding periods,
    producing bit-identical trades/counters/equity."""
    n = n_ticks
    eff_rate = eurusd if eurusd > 0.0 else 1.0

    equity = initial_equity
    signals_total = 0
    long_signals = 0
    short_signals = 0
    rej_missing = 0
    rej_narrow = 0
    rej_price_range_position = 0
    rej_small = 0
    rej_large = 0
    skipped_no_size = 0
    ruined = 0
    ticks_simulated = n
    n_trades = 0

    ev = 0
    while ev < n_events:
        i = ev_tick[ev]
        direction = ev_dir[ev]
        cross_level = ev_cross[ev]

        signals_total += 1
        if direction == 1:
            long_signals += 1
        else:
            short_signals += 1

        bar_i = tick_bar_index[i]
        pl = np.nan
        ph = np.nan
        pr = np.nan
        if 0 <= bar_i < profile_low.shape[0]:
            pl = profile_low[bar_i]
            ph = profile_high[bar_i]
            pr = profile_range[bar_i]
        if not ((pl == pl) and (ph == ph) and (pr == pr) and pr > 0.0):
            rej_missing += 1
            ev += 1
            continue
        if stop_mode_fixed != 1 and pr < min_profile_range_points:
            rej_narrow += 1
            ev += 1
            continue
        expected_position_pct = long_cross_quantile * 100.0 if direction == 1 else short_cross_quantile * 100.0
        cross_position_pct = ((cross_level - pl) / pr) * 100.0
        if not (cross_position_pct == cross_position_pct) or abs(cross_position_pct - expected_position_pct) > entry_price_range_position_max_deviation_pct:
            rej_price_range_position += 1
            ev += 1
            continue

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
            sl = np.nan
            su = np.nan
            if 0 <= bar_i < profile_low.shape[0]:
                sl = stop_lower[bar_i]
                su = stop_upper[bar_i]
            if not ((sl == sl) and (su == su)):
                valid = False
                reject = 1
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
            ev += 1
            continue

        units = 0.0
        notional = 0.0
        margin = 0.0
        margin_capped = 0
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
                    margin_capped = 1
                if units < lot_size:
                    units = 0.0
                else:
                    units = math.floor((units + 1e-12) / lot_size) * lot_size
                if units > 0.0:
                    notional = (units * entry_price * mult) / eff_rate
                    margin = notional * (margin_pct / 100.0)
        if units <= 0.0:
            skipped_no_size += 1
            ev += 1
            continue

        # Open position at tick i. Record entry-side fields now.
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
        p_prev_mid = mid[i - 1]
        p_signal_mid = mid[i]
        p_margin_capped = margin_capped

        # Scan forward for the exit (stop priority over take-profit on the same tick).
        exit_idx = -1
        status = 0
        fill_price = 0.0
        j = i + 1
        while j < n:
            if p_dir == 1:
                exit_quote = bid[j]
                hit_stop = exit_quote <= p_stop
                hit_tp = exit_quote >= p_tp
            else:
                exit_quote = ask[j]
                hit_stop = exit_quote >= p_stop
                hit_tp = exit_quote <= p_tp
            if hit_stop or hit_tp:
                status = 0 if hit_stop else 1
                fill_price = p_tp if status == 1 else exit_quote
                exit_idx = j
                break
            j += 1

        if exit_idx == -1:
            # End of data with an open position: close at the last tick (status 2).
            i_last = n - 1
            exit_quote = bid[i_last] if p_dir == 1 else ask[i_last]
            sign = 1.0 if p_dir == 1 else -1.0
            gross = (p_units * mult * (exit_quote - p_entry_price) * sign) / eff_rate
            extra = (p_units * (spread_points + slippage_points) * mult) / eff_rate
            extra += commission_per_unit * p_units * 2.0
            pnl = gross - extra
            equity_after = equity + pnl
            if n_trades < cap:
                out_entry_idx[n_trades] = p_entry_idx
                out_exit_idx[n_trades] = i_last
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
                out_ticks_held[n_trades] = i_last - p_entry_idx
                out_margin_capped[n_trades] = p_margin_capped
            n_trades += 1
            equity = equity_after
            break

        sign = 1.0 if p_dir == 1 else -1.0
        gross = (p_units * mult * (fill_price - p_entry_price) * sign) / eff_rate
        extra = (p_units * (spread_points + slippage_points) * mult) / eff_rate
        extra += commission_per_unit * p_units * 2.0
        pnl = gross - extra
        equity_after = equity + pnl
        if n_trades < cap:
            out_entry_idx[n_trades] = p_entry_idx
            out_exit_idx[n_trades] = exit_idx
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
            out_ticks_held[n_trades] = exit_idx - p_entry_idx
            out_margin_capped[n_trades] = p_margin_capped
        n_trades += 1
        equity = equity_after
        if equity <= 0.0:
            ruined = 1
            ticks_simulated = exit_idx + 1
            break

        ev += 1
        while ev < n_events and ev_tick[ev] <= exit_idx:
            ev += 1

    return (
        n_trades,
        signals_total,
        long_signals,
        short_signals,
        rej_missing,
        rej_narrow,
        rej_price_range_position,
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
        0, 10.0, 10.0, 0.0, 0.5, 0.5, 100.0, 1.0, 1.0, 1e18,
        5000.0, 1.0, 1.0, 1.5, 5.0, 45.0, 0.1, 0.0, 0.0, 0.0,
        out_i.copy(), out_i.copy(), out_d.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_d.copy(),
        out_i.copy(), out_d.copy(),
        0,
    )

    ev_tick = np.zeros(1, dtype=np.int64)
    ev_dir = np.ones(1, dtype=np.int8)
    ev_cross = np.zeros(1, dtype=np.float64)
    precompute_crossing_events(zeros, bar_index, zeros, zeros, allowed, ev_tick, ev_dir, ev_cross, 0)
    simulate_core_from_events(
        ev_tick, ev_dir, ev_cross, 0,
        zeros, zeros, zeros, bar_index, zeros, zeros, zeros, zeros, zeros, zeros,
        n,
        0, 10.0, 10.0, 0.0, 0.5, 0.5, 100.0, 1.0, 1.0, 1e18,
        5000.0, 1.0, 1.0, 1.5, 5.0, 45.0, 0.1, 0.0, 0.0, 0.0,
        out_i.copy(), out_i.copy(), out_d.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(),
        out_f.copy(), out_f.copy(), out_f.copy(), out_f.copy(), out_d.copy(),
        out_i.copy(), out_d.copy(),
        0,
    )
