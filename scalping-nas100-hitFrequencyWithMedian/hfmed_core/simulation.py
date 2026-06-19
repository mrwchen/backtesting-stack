"""Tick-level simulation for the hit-frequency median crossing rule."""

from __future__ import annotations

import logging

import numpy as np

from .config import RunConfig
from .data import BarData, TickData, ns_to_datetime
from .entities import ClosedTrade, SimulationResult
from .profile import ProfileArrays, rolling_profile_arrays
from .sessions import SESSION_CODE_BY_KEY, session_key_for_code
from .sim_core import simulate_core

log = logging.getLogger(__name__)


def _all_sessions_enabled(cfg: RunConfig) -> bool:
    return (
        cfg.session_pre_market_enabled
        and cfg.session_ny_open_power_enabled
        and cfg.session_ny_midday_enabled
        and cfg.session_ny_late_enabled
        and cfg.session_ny_power_hour_enabled
        and cfg.session_after_hours_enabled
        and cfg.session_overnight_enabled
    )


def _entry_session_allowed_mask(entry_session_code: np.ndarray, cfg: RunConfig) -> np.ndarray | None:
    if _all_sessions_enabled(cfg):
        return None
    enabled = []
    if cfg.session_pre_market_enabled:
        enabled.append(SESSION_CODE_BY_KEY["pre_market"])
    if cfg.session_ny_open_power_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_open_power"])
    if cfg.session_ny_midday_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_midday"])
    if cfg.session_ny_late_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_late"])
    if cfg.session_ny_power_hour_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_power_hour"])
    if cfg.session_after_hours_enabled:
        enabled.append(SESSION_CODE_BY_KEY["after_hours"])
    if cfg.session_overnight_enabled:
        enabled.append(SESSION_CODE_BY_KEY["overnight"])
    return np.isin(entry_session_code, np.array(enabled, dtype=np.uint8))


def _build_entry_allowed(
    ticks: TickData,
    cfg: RunConfig,
    trade_start_ns: int | None,
    trade_end_ns: int | None,
) -> np.ndarray:
    n = len(ticks)
    allow = np.ones(n, dtype=bool)
    session_mask = _entry_session_allowed_mask(ticks.entry_session_code, cfg)
    if session_mask is not None:
        allow &= session_mask
    if trade_start_ns is not None:
        allow &= ticks.tick_time_ns >= int(trade_start_ns)
    if trade_end_ns is not None:
        allow &= ticks.tick_time_ns < int(trade_end_ns)
    return allow.astype(np.uint8)


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_int32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int32)


def run_simulation(
    ticks: TickData,
    bars: BarData,
    tick_bar_index: np.ndarray,
    cfg: RunConfig,
    trade_start_ns: int | None = None,
    trade_end_ns: int | None = None,
    log_result: bool = True,
    profile: ProfileArrays | None = None,
) -> SimulationResult:
    if profile is None:
        profile = rolling_profile_arrays(bars, cfg)

    result = SimulationResult(
        initial_equity=cfg.initial_equity,
        final_equity=cfg.initial_equity,
        ticks_total=len(ticks),
        bars_total=len(bars),
    )

    n = len(ticks)
    if n == 0:
        if log_result:
            _log_result(result)
        return result

    mid = _as_float(ticks.mid)
    bid = _as_float(ticks.bid)
    ask = _as_float(ticks.ask)
    local_bar_index = _as_int32(tick_bar_index)
    median_level = _as_float(profile.median_level)
    long_cross = _as_float(profile.long_cross_level)
    short_cross = _as_float(profile.short_cross_level)
    profile_low = _as_float(profile.profile_low)
    profile_high = _as_float(profile.profile_high)
    stop_lower = _as_float(profile.stop_profile_lower)
    stop_upper = _as_float(profile.stop_profile_upper)
    profile_range = _as_float(profile.profile_range_points)
    entry_allowed = _build_entry_allowed(ticks, cfg, trade_start_ns, trade_end_ns)

    stop_mode_fixed = 1 if cfg.stop_mode == "fixed" else 0
    scalar_args = (
        stop_mode_fixed,
        float(cfg.stop_points),
        float(cfg.take_profit_points),
        float(cfg.min_profile_range_points),
        float(cfg.stop_profile_buffer_points),
        float(cfg.min_stop_distance_points),
        float(cfg.max_stop_distance_points),
        float(cfg.initial_equity),
        float(cfg.contract_multiplier),
        float(cfg.eurusd_rate),
        float(cfg.risk_per_trade_pct),
        float(cfg.margin_requirement_pct),
        float(cfg.max_margin_pct),
        float(cfg.lot_size),
        float(cfg.spread_points),
        float(cfg.slippage_points),
        float(cfg.commission_per_unit),
    )
    price_args = (
        mid, bid, ask, local_bar_index, median_level, long_cross, short_cross,
        profile_low, profile_high, stop_lower, stop_upper, profile_range,
        entry_allowed,
    )

    def _empty_outputs(cap: int) -> tuple:
        return (
            np.empty(cap, dtype=np.int64),    # entry_idx
            np.empty(cap, dtype=np.int64),    # exit_idx
            np.empty(cap, dtype=np.int8),     # direction
            np.empty(cap, dtype=np.float64),  # cross_level
            np.empty(cap, dtype=np.float64),  # median
            np.empty(cap, dtype=np.float64),  # prev_mid
            np.empty(cap, dtype=np.float64),  # signal_mid
            np.empty(cap, dtype=np.float64),  # entry_price
            np.empty(cap, dtype=np.float64),  # exit_price
            np.empty(cap, dtype=np.float64),  # stop_price
            np.empty(cap, dtype=np.float64),  # tp_price
            np.empty(cap, dtype=np.float64),  # units
            np.empty(cap, dtype=np.float64),  # notional
            np.empty(cap, dtype=np.float64),  # margin
            np.empty(cap, dtype=np.float64),  # gross
            np.empty(cap, dtype=np.float64),  # extra
            np.empty(cap, dtype=np.float64),  # pnl
            np.empty(cap, dtype=np.float64),  # equity_before
            np.empty(cap, dtype=np.float64),  # equity_after
            np.empty(cap, dtype=np.int8),     # status
            np.empty(cap, dtype=np.int64),    # ticks_held
        )

    count_out = _empty_outputs(0)
    counted = simulate_core(*price_args, *scalar_args, *count_out, 0)
    n_trades = counted[0]

    cap = max(int(n_trades), 1)
    out = _empty_outputs(cap)
    stats = simulate_core(*price_args, *scalar_args, *out, cap)
    n_trades = stats[0]

    (
        out_entry_idx, out_exit_idx, out_direction, out_cross_level, out_median,
        out_prev_mid, out_signal_mid, out_entry_price, out_exit_price, out_stop_price,
        out_tp_price, out_units, out_notional, out_margin, out_gross, out_extra,
        out_pnl, out_equity_before, out_equity_after, out_status, out_ticks_held,
    ) = out

    status_labels = ("HIT_SL", "HIT_TP", "END_OF_DATA")
    trades: list[ClosedTrade] = []
    for k in range(n_trades):
        d = int(out_direction[k])
        e = int(out_entry_idx[k])
        x = int(out_exit_idx[k])
        entry_ts = ns_to_datetime(int(ticks.tick_time_ns[e]))
        exit_ts = ns_to_datetime(int(ticks.tick_time_ns[x]))
        entry_price = float(out_entry_price[k])
        exit_price = float(out_exit_price[k])
        equity_before = float(out_equity_before[k])
        pnl = float(out_pnl[k])
        sign = 1.0 if d == 1 else -1.0
        trades.append(ClosedTrade(
            signal_ts=entry_ts,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction="LONG" if d == 1 else "SHORT",
            entry_session=session_key_for_code(int(ticks.entry_session_code[e])),
            cross_quantile=cfg.long_cross_quantile if d == 1 else cfg.short_cross_quantile,
            cross_level=float(out_cross_level[k]),
            median_level=float(out_median[k]),
            signal_mid=float(out_signal_mid[k]),
            previous_mid=float(out_prev_mid[k]),
            entry_bid=float(bid[e]),
            entry_ask=float(ask[e]),
            entry_price=entry_price,
            exit_bid=float(bid[x]),
            exit_ask=float(ask[x]),
            exit_price=exit_price,
            stop_price=float(out_stop_price[k]),
            take_profit_price=float(out_tp_price[k]),
            units=float(out_units[k]),
            notional_eur=float(out_notional[k]),
            margin_used_eur=float(out_margin[k]),
            gross_pnl_eur=float(out_gross[k]),
            extra_costs_eur=float(out_extra[k]),
            pnl_eur=pnl,
            equity_before=equity_before,
            equity_after=float(out_equity_after[k]),
            return_pct=(pnl / equity_before * 100.0) if equity_before > 0 else 0.0,
            price_pnl_points=(exit_price - entry_price) * sign,
            outcome_status=status_labels[int(out_status[k])],
            ticks_held=int(out_ticks_held[k]),
            seconds_held=float((exit_ts - entry_ts).total_seconds()),
        ))

    result.trades = trades
    result.signals_total = int(stats[1])
    result.long_signals = int(stats[2])
    result.short_signals = int(stats[3])
    result.rejected_signals_missing_band = int(stats[4])
    result.rejected_signals_band_too_narrow = int(stats[5])
    result.rejected_signals_stop_too_small = int(stats[6])
    result.rejected_signals_stop_too_large = int(stats[7])
    result.skipped_signals_no_size = int(stats[8])
    result.ruined = bool(stats[9])
    result.ticks_simulated = int(stats[10])
    result.final_equity = float(stats[11])

    if log_result:
        _log_result(result)
    return result


def _log_result(result: SimulationResult) -> None:
    log.info(
        "Simulation done ticks %d bars %d signals %d trades %d rejected_missing_band %d rejected_profile_range_too_narrow %d rejected_stop_too_small %d rejected_stop_too_large %d skipped_no_size %d final_equity %.2f ruined %s",
        result.ticks_simulated, result.bars_total, result.signals_total,
        len(result.trades), result.rejected_signals_missing_band,
        result.rejected_signals_band_too_narrow, result.rejected_signals_stop_too_small,
        result.rejected_signals_stop_too_large, result.skipped_signals_no_size,
        result.final_equity, result.ruined,
    )
