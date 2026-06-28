"""Tick-level simulation for the hit-frequency median crossing rule."""

from __future__ import annotations

import logging

import numpy as np

try:
    from numba.typed import List as NumbaList
except ImportError:  # pragma: no cover - sim_core falls back to plain Python too
    NumbaList = None

from .config import RunConfig
from .data import BarData, TickData, ns_to_datetime
from .entities import ClosedTrade, SimulationResult
from .profile import ProfileArrays, rolling_profile_arrays
from .sessions import SESSION_CODE_BY_KEY, SESSION_TYPES, session_key_for_code
from .sim_core import (
    precompute_crossing_events,
    simulate_core_from_events,
    simulate_session_portfolio_core,
)

log = logging.getLogger(__name__)


def _all_sessions_enabled(cfg: RunConfig) -> bool:
    return (
        cfg.session_asia_early_enabled
        and cfg.session_asia_late_enabled
        and cfg.session_london_open_enabled
        and cfg.session_pre_market_early_enabled
        and cfg.session_pre_market_active_enabled
        and cfg.session_pre_market_macro_enabled
        and cfg.session_ny_open_impulse_enabled
        and cfg.session_ny_morning_enabled
        and cfg.session_ny_midday_enabled
        and cfg.session_ny_late_enabled
        and cfg.session_ny_power_hour_enabled
        and cfg.session_after_close_shock_enabled
        and cfg.session_after_hours_late_enabled
    )


def _entry_session_allowed_mask(entry_session_code: np.ndarray, cfg: RunConfig) -> np.ndarray | None:
    if _all_sessions_enabled(cfg):
        return None
    enabled = []
    if cfg.session_asia_early_enabled:
        enabled.append(SESSION_CODE_BY_KEY["asia_early"])
    if cfg.session_asia_late_enabled:
        enabled.append(SESSION_CODE_BY_KEY["asia_late"])
    if cfg.session_london_open_enabled:
        enabled.append(SESSION_CODE_BY_KEY["london_open"])
    if cfg.session_pre_market_early_enabled:
        enabled.append(SESSION_CODE_BY_KEY["pre_market_early"])
    if cfg.session_pre_market_active_enabled:
        enabled.append(SESSION_CODE_BY_KEY["pre_market_active"])
    if cfg.session_pre_market_macro_enabled:
        enabled.append(SESSION_CODE_BY_KEY["pre_market_macro"])
    if cfg.session_ny_open_impulse_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_open_impulse"])
    if cfg.session_ny_morning_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_morning"])
    if cfg.session_ny_midday_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_midday"])
    if cfg.session_ny_late_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_late"])
    if cfg.session_ny_power_hour_enabled:
        enabled.append(SESSION_CODE_BY_KEY["ny_power_hour"])
    if cfg.session_after_close_shock_enabled:
        enabled.append(SESSION_CODE_BY_KEY["after_close_shock"])
    if cfg.session_after_hours_late_enabled:
        enabled.append(SESSION_CODE_BY_KEY["after_hours_late"])
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


def _build_time_allowed(
    ticks: TickData,
    trade_start_ns: int | None,
    trade_end_ns: int | None,
) -> np.ndarray:
    n = len(ticks)
    allow = np.ones(n, dtype=bool)
    if trade_start_ns is not None:
        allow &= ticks.tick_time_ns >= int(trade_start_ns)
    if trade_end_ns is not None:
        allow &= ticks.tick_time_ns < int(trade_end_ns)
    return allow.astype(np.uint8)


def _as_float(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _as_int32(values: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int32)


def _as_float_array_list(values: list[np.ndarray]):
    arrays = [np.ascontiguousarray(value, dtype=np.float64) for value in values]
    if NumbaList is None:
        return arrays
    typed = NumbaList()
    for array in arrays:
        typed.append(array)
    return typed


def _profile_position_values(
    profile: ProfileArrays,
    bar_i: int,
    cross_level: float,
    entry_price: float,
    cross_quantile: float,
) -> tuple[float, float, float, float, float, float]:
    if not 0 <= bar_i < len(profile.profile_low):
        return (float("nan"),) * 6

    profile_low = float(profile.profile_low[bar_i])
    profile_high = float(profile.profile_high[bar_i])
    profile_range = float(profile.profile_range_points[bar_i])
    if not (np.isfinite(profile_low) and np.isfinite(profile_high) and np.isfinite(profile_range) and profile_range > 0.0):
        return profile_low, profile_high, profile_range, float("nan"), float("nan"), float("nan")

    cross_position_pct = ((cross_level - profile_low) / profile_range) * 100.0
    entry_position_pct = ((entry_price - profile_low) / profile_range) * 100.0
    deviation_pct = abs(cross_position_pct - (cross_quantile * 100.0))
    return profile_low, profile_high, profile_range, cross_position_pct, entry_position_pct, deviation_pct


def _empty_trade_outputs(cap: int) -> tuple:
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
        np.empty(cap, dtype=np.int8),     # margin_capped
    )


CrossingEvents = tuple  # (ev_tick int64[k], ev_dir int8[k], ev_cross float64[k])


def precompute_events(
    ticks: TickData,
    tick_bar_index: np.ndarray,
    cfg: RunConfig,
    profile: ProfileArrays,
    trade_start_ns: int | None,
    trade_end_ns: int | None,
) -> CrossingEvents:
    """Detect all profile crossings once for a (profile, session, trade-window) combo.

    The result is candidate-independent within an optimizer profile group, so it is
    computed once per group and reused across every candidate in that group instead
    of being re-derived by a full per-tick scan in every candidate simulation.
    """
    n = len(ticks)
    if n == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int8), np.empty(0, np.float64))
    mid = _as_float(ticks.mid)
    local_bar_index = _as_int32(tick_bar_index)
    long_cross = _as_float(profile.long_cross_level)
    short_cross = _as_float(profile.short_cross_level)
    entry_allowed = _build_entry_allowed(ticks, cfg, trade_start_ns, trade_end_ns)
    empty_i = np.empty(0, np.int64)
    empty_d = np.empty(0, np.int8)
    empty_f = np.empty(0, np.float64)
    k = int(precompute_crossing_events(
        mid, local_bar_index, long_cross, short_cross, entry_allowed,
        empty_i, empty_d, empty_f, 0,
    ))
    ev_tick = np.empty(k, np.int64)
    ev_dir = np.empty(k, np.int8)
    ev_cross = np.empty(k, np.float64)
    if k > 0:
        precompute_crossing_events(
            mid, local_bar_index, long_cross, short_cross, entry_allowed,
            ev_tick, ev_dir, ev_cross, k,
        )
    return (ev_tick, ev_dir, ev_cross)


def _simulate_events(
    ticks: TickData,
    tick_bar_index: np.ndarray,
    cfg: RunConfig,
    profile: ProfileArrays,
    trade_start_ns: int | None,
    trade_end_ns: int | None,
    events: CrossingEvents | None,
) -> tuple:
    """Run the event-driven core; returns (out_tuple, stats, n_trades, bid, ask)."""
    n = len(ticks)
    mid = _as_float(ticks.mid)
    bid = _as_float(ticks.bid)
    ask = _as_float(ticks.ask)
    local_bar_index = _as_int32(tick_bar_index)
    median_level = _as_float(profile.median_level)
    profile_low = _as_float(profile.profile_low)
    profile_high = _as_float(profile.profile_high)
    stop_lower = _as_float(profile.stop_profile_lower)
    stop_upper = _as_float(profile.stop_profile_upper)
    profile_range = _as_float(profile.profile_range_points)
    atr_points = _as_float(profile.atr_points)

    if events is None:
        events = precompute_events(ticks, tick_bar_index, cfg, profile, trade_start_ns, trade_end_ns)
    ev_tick, ev_dir, ev_cross = events
    n_events = int(ev_tick.shape[0])

    scalar_args = (
        1 if cfg.stop_mode == "fixed" else 0,
        float(cfg.stop_points),
        float(cfg.take_profit_atr_mult),
        float(cfg.min_profile_range_atr_mult),
        float(cfg.long_cross_quantile),
        float(cfg.short_cross_quantile),
        float(cfg.entry_price_range_position_max_deviation_pct),
        float(cfg.stop_profile_buffer_points),
        float(cfg.min_stop_distance_atr_mult),
        float(cfg.max_stop_distance_atr_mult),
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
    cap = max(n_events, 1)
    out = _empty_trade_outputs(cap)
    stats = simulate_core_from_events(
        ev_tick, ev_dir, ev_cross, n_events,
        mid, bid, ask, local_bar_index, median_level,
        profile_low, profile_high, stop_lower, stop_upper, profile_range,
        atr_points, n, *scalar_args, *out, cap,
    )
    return out, stats, int(stats[0]), bid, ask


def _empty_summary(final_equity: float) -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "gross_profit_eur": 0.0,
        "gross_loss_eur": 0.0,
        "net_profit_eur": 0.0,
        "avg_trade_pnl_eur": 0.0,
        "avg_realized_risk_pct": 0.0,
        "median_realized_risk_pct": 0.0,
        "max_realized_risk_pct": 0.0,
        "margin_capped_share_pct": 0.0,
        "final_equity": round(float(final_equity), 2),
    }


def _empty_session_stats() -> dict[str, dict]:
    return {
        session_type: {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "net_profit_eur": 0.0,
            "avg_trade_pnl_eur": 0.0,
            "std_trade_pnl_eur": 0.0,
            "median_trade_pnl_eur": 0.0,
            "p25_trade_pnl_eur": 0.0,
        }
        for session_type, _label, _sort_order in SESSION_TYPES
    }


def _apply_core_stats(result: SimulationResult, stats: tuple) -> None:
    result.signals_total = int(stats[1])
    result.long_signals = int(stats[2])
    result.short_signals = int(stats[3])
    result.rejected_signals_missing_band = int(stats[4])
    result.rejected_signals_band_too_narrow = int(stats[5])
    result.rejected_signals_price_range_position = int(stats[6])
    result.rejected_signals_stop_too_small = int(stats[7])
    result.rejected_signals_stop_too_large = int(stats[8])
    result.skipped_signals_no_size = int(stats[9])
    result.ruined = bool(stats[10])
    result.ticks_simulated = int(stats[11])
    result.final_equity = float(stats[12])


def _summary_from_arrays(
    initial_equity: float,
    final_equity: float,
    pnl: np.ndarray,
    equity_before: np.ndarray,
    equity_after: np.ndarray,
    realized_risk_pct: np.ndarray,
    margin_capped: np.ndarray,
) -> dict:
    n = int(pnl.shape[0])
    if n <= 0:
        return _empty_summary(final_equity)

    returns = np.divide(
        pnl,
        equity_before,
        out=np.zeros(n, dtype=np.float64),
        where=equity_before > 0.0,
    ) * 100.0
    wins = pnl > 0.0
    losses = pnl < 0.0
    gross_win = float(pnl[wins].sum())
    gross_loss = float(-pnl[losses].sum())
    if gross_loss > 0.0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0.0:
        profit_factor = None
    else:
        profit_factor = 0.0

    equity = np.concatenate([[initial_equity], equity_after])
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100.0

    return {
        "total_trades": n,
        "winning_trades": int(wins.sum()),
        "losing_trades": int(losses.sum()),
        "breakeven_trades": int(n - wins.sum() - losses.sum()),
        "win_rate_pct": round(float(wins.mean() * 100.0), 2),
        "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
        "total_return_pct": round((final_equity - initial_equity) / initial_equity * 100.0, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "avg_win_pct": round(float(returns[wins].mean()), 4) if wins.any() else 0.0,
        "avg_loss_pct": round(float(returns[losses].mean()), 4) if losses.any() else 0.0,
        "gross_profit_eur": round(gross_win, 2),
        "gross_loss_eur": round(gross_loss, 2),
        "net_profit_eur": round(float(pnl.sum()), 2),
        "avg_trade_pnl_eur": round(float(pnl.mean()), 4),
        "avg_realized_risk_pct": round(float(realized_risk_pct.mean()), 4),
        "median_realized_risk_pct": round(float(np.median(realized_risk_pct)), 4),
        "max_realized_risk_pct": round(float(realized_risk_pct.max()), 4),
        "margin_capped_share_pct": round(float(margin_capped.mean() * 100.0), 2),
        "final_equity": round(float(final_equity), 2),
    }


def _session_stats_from_arrays(entry_session_codes: np.ndarray, pnl: np.ndarray) -> dict[str, dict]:
    stats = _empty_session_stats()
    if pnl.shape[0] <= 0:
        return stats

    for session_type, _label, _sort_order in SESSION_TYPES:
        code = SESSION_CODE_BY_KEY[session_type]
        session_pnl = pnl[entry_session_codes == code]
        total = int(session_pnl.shape[0])
        if total <= 0:
            continue
        wins = session_pnl > 0.0
        losses = session_pnl < 0.0
        gross_profit = float(session_pnl[wins].sum())
        gross_loss = float(-session_pnl[losses].sum())
        net_profit = float(session_pnl.sum())
        row = stats[session_type]
        row["total_trades"] = total
        row["winning_trades"] = int(wins.sum())
        row["losing_trades"] = int(losses.sum())
        row["breakeven_trades"] = int(total - wins.sum() - losses.sum())
        row["win_rate_pct"] = round(float(wins.sum()) / total * 100.0, 2)
        row["gross_profit_eur"] = round(gross_profit, 2)
        row["gross_loss_eur"] = round(gross_loss, 2)
        row["net_profit_eur"] = round(net_profit, 2)
        row["avg_trade_pnl_eur"] = round(net_profit / total, 4)
        row["std_trade_pnl_eur"] = round(float(session_pnl.std(ddof=0)), 4)
        row["median_trade_pnl_eur"] = round(float(np.median(session_pnl)), 4)
        row["p25_trade_pnl_eur"] = round(float(np.percentile(session_pnl, 25)), 4)
    return stats


def run_simulation_summary(
    ticks: TickData,
    bars: BarData,
    tick_bar_index: np.ndarray,
    cfg: RunConfig,
    trade_start_ns: int | None = None,
    trade_end_ns: int | None = None,
    profile: ProfileArrays | None = None,
    events: CrossingEvents | None = None,
) -> tuple[SimulationResult, dict, dict[str, dict]]:
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
        return result, _empty_summary(result.final_equity), _empty_session_stats()

    out, stats, n_trades, bid, ask = _simulate_events(
        ticks, tick_bar_index, cfg, profile, trade_start_ns, trade_end_ns, events
    )
    _apply_core_stats(result, stats)

    (
        out_entry_idx, _out_exit_idx, _out_direction, _out_cross_level, _out_median,
        _out_prev_mid, _out_signal_mid, out_entry_price, _out_exit_price, out_stop_price,
        _out_tp_price, out_units, _out_notional, _out_margin, _out_gross, _out_extra,
        out_pnl, out_equity_before, out_equity_after, _out_status, _out_ticks_held,
        out_margin_capped,
    ) = out
    sl = slice(0, n_trades)
    eff_rate = cfg.eurusd_rate if cfg.eurusd_rate > 0 else 1.0
    realized_risk_eur = out_units[sl] * np.abs(out_entry_price[sl] - out_stop_price[sl]) * cfg.contract_multiplier / eff_rate
    realized_risk_pct = np.divide(
        realized_risk_eur,
        out_equity_before[sl],
        out=np.zeros(n_trades, dtype=np.float64),
        where=out_equity_before[sl] > 0.0,
    ) * 100.0
    summary = _summary_from_arrays(
        cfg.initial_equity,
        result.final_equity,
        out_pnl[sl],
        out_equity_before[sl],
        out_equity_after[sl],
        realized_risk_pct,
        out_margin_capped[sl].astype(np.float64, copy=False),
    )
    entry_session_codes = ticks.entry_session_code[out_entry_idx[sl]] if n_trades > 0 else np.empty(0, dtype=np.uint8)
    session_stats = _session_stats_from_arrays(entry_session_codes, out_pnl[sl])
    return result, summary, session_stats


def run_simulation(
    ticks: TickData,
    bars: BarData,
    tick_bar_index: np.ndarray,
    cfg: RunConfig,
    trade_start_ns: int | None = None,
    trade_end_ns: int | None = None,
    log_result: bool = True,
    profile: ProfileArrays | None = None,
    events: CrossingEvents | None = None,
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

    out, stats, n_trades, bid, ask = _simulate_events(
        ticks, tick_bar_index, cfg, profile, trade_start_ns, trade_end_ns, events
    )

    (
        out_entry_idx, out_exit_idx, out_direction, out_cross_level, out_median,
        out_prev_mid, out_signal_mid, out_entry_price, out_exit_price, out_stop_price,
        out_tp_price, out_units, out_notional, out_margin, out_gross, out_extra,
        out_pnl, out_equity_before, out_equity_after, out_status, out_ticks_held,
        out_margin_capped,
    ) = out

    status_labels = ("HIT_SL", "HIT_TP", "END_OF_DATA")
    eff_rate = cfg.eurusd_rate if cfg.eurusd_rate > 0 else 1.0
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
        stop_price = float(out_stop_price[k])
        realized_risk_eur = float(out_units[k]) * abs(entry_price - stop_price) * cfg.contract_multiplier / eff_rate
        realized_risk_pct = (realized_risk_eur / equity_before * 100.0) if equity_before > 0 else 0.0
        cross_quantile = cfg.long_cross_quantile if d == 1 else cfg.short_cross_quantile
        (
            profile_low,
            profile_high,
            profile_range,
            cross_position_pct,
            entry_position_pct,
            range_position_deviation_pct,
        ) = _profile_position_values(
            profile,
            int(tick_bar_index[e]),
            float(out_cross_level[k]),
            entry_price,
            cross_quantile,
        )
        trades.append(ClosedTrade(
            signal_ts=entry_ts,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction="LONG" if d == 1 else "SHORT",
            entry_session=session_key_for_code(int(ticks.entry_session_code[e])),
            cross_quantile=cross_quantile,
            cross_level=float(out_cross_level[k]),
            profile_low=profile_low,
            profile_high=profile_high,
            profile_range=profile_range,
            cross_price_range_position_pct=cross_position_pct,
            entry_price_range_position_pct=entry_position_pct,
            range_position_deviation_pct=range_position_deviation_pct,
            median_level=float(out_median[k]),
            signal_mid=float(out_signal_mid[k]),
            previous_mid=float(out_prev_mid[k]),
            entry_bid=float(bid[e]),
            entry_ask=float(ask[e]),
            entry_price=entry_price,
            exit_bid=float(bid[x]),
            exit_ask=float(ask[x]),
            exit_price=exit_price,
            stop_price=stop_price,
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
            realized_risk_eur=realized_risk_eur,
            realized_risk_pct=realized_risk_pct,
            margin_capped=bool(out_margin_capped[k]),
        ))

    result.trades = trades
    result.signals_total = int(stats[1])
    result.long_signals = int(stats[2])
    result.short_signals = int(stats[3])
    result.rejected_signals_missing_band = int(stats[4])
    result.rejected_signals_band_too_narrow = int(stats[5])
    result.rejected_signals_price_range_position = int(stats[6])
    result.rejected_signals_stop_too_small = int(stats[7])
    result.rejected_signals_stop_too_large = int(stats[8])
    result.skipped_signals_no_size = int(stats[9])
    result.ruined = bool(stats[10])
    result.ticks_simulated = int(stats[11])
    result.final_equity = float(stats[12])

    if log_result:
        _log_result(result)
    return result


def run_session_portfolio_simulation(
    ticks: TickData,
    bars: BarData,
    tick_bar_index: np.ndarray,
    base_cfg: RunConfig,
    session_cfgs: dict[str, RunConfig],
    trade_start_ns: int | None = None,
    trade_end_ns: int | None = None,
    log_result: bool = True,
    profiles: dict[str, ProfileArrays] | None = None,
) -> SimulationResult:
    ordered = [(key, session_cfgs[key]) for key, _label, _sort_order in SESSION_TYPES if key in session_cfgs]
    result = SimulationResult(
        initial_equity=base_cfg.initial_equity,
        final_equity=base_cfg.initial_equity,
        ticks_total=len(ticks),
        bars_total=len(bars),
    )
    n = len(ticks)
    if n == 0 or not ordered:
        if log_result:
            _log_result(result)
        return result

    if profiles is None:
        profiles = {key: rolling_profile_arrays(bars, cfg) for key, cfg in ordered}

    slot_keys = [key for key, _cfg in ordered]
    slot_cfgs = [cfg for _key, cfg in ordered]
    slot_profiles = [profiles[key] for key in slot_keys]

    def _slot_arrays(name: str):
        return _as_float_array_list([getattr(profile, name) for profile in slot_profiles])

    session_to_slot = np.full(max(SESSION_CODE_BY_KEY.values()) + 1, -1, dtype=np.int32)
    for slot, key in enumerate(slot_keys):
        session_to_slot[SESSION_CODE_BY_KEY[key]] = slot

    stop_mode_fixed = np.ascontiguousarray([1 if cfg.stop_mode == "fixed" else 0 for cfg in slot_cfgs], dtype=np.int8)
    stop_points = np.ascontiguousarray([float(cfg.stop_points) for cfg in slot_cfgs], dtype=np.float64)
    take_profit_atr_mult = np.ascontiguousarray([float(cfg.take_profit_atr_mult) for cfg in slot_cfgs], dtype=np.float64)
    min_profile_range_atr_mult = np.ascontiguousarray([float(cfg.min_profile_range_atr_mult) for cfg in slot_cfgs], dtype=np.float64)
    long_cross_quantiles = np.ascontiguousarray([float(cfg.long_cross_quantile) for cfg in slot_cfgs], dtype=np.float64)
    short_cross_quantiles = np.ascontiguousarray([float(cfg.short_cross_quantile) for cfg in slot_cfgs], dtype=np.float64)
    entry_price_range_position_max_deviation = np.ascontiguousarray(
        [float(cfg.entry_price_range_position_max_deviation_pct) for cfg in slot_cfgs],
        dtype=np.float64,
    )
    stop_buffer = np.ascontiguousarray([float(cfg.stop_profile_buffer_points) for cfg in slot_cfgs], dtype=np.float64)
    min_stop_distance_atr_mult = np.ascontiguousarray([float(cfg.min_stop_distance_atr_mult) for cfg in slot_cfgs], dtype=np.float64)
    max_stop_distance_atr_mult = np.ascontiguousarray([float(cfg.max_stop_distance_atr_mult) for cfg in slot_cfgs], dtype=np.float64)

    mid = _as_float(ticks.mid)
    bid = _as_float(ticks.bid)
    ask = _as_float(ticks.ask)
    local_bar_index = _as_int32(tick_bar_index)
    entry_session_code = np.ascontiguousarray(ticks.entry_session_code, dtype=np.uint8)
    entry_allowed = _build_time_allowed(ticks, trade_start_ns, trade_end_ns)

    price_args = (
        mid,
        bid,
        ask,
        local_bar_index,
        entry_session_code,
        _slot_arrays("median_level"),
        _slot_arrays("long_cross_level"),
        _slot_arrays("short_cross_level"),
        _slot_arrays("profile_low"),
        _slot_arrays("profile_high"),
        _slot_arrays("stop_profile_lower"),
        _slot_arrays("stop_profile_upper"),
        _slot_arrays("profile_range_points"),
        _slot_arrays("atr_points"),
        session_to_slot,
        entry_allowed,
    )
    scalar_args = (
        stop_mode_fixed,
        stop_points,
        take_profit_atr_mult,
        min_profile_range_atr_mult,
        long_cross_quantiles,
        short_cross_quantiles,
        entry_price_range_position_max_deviation,
        stop_buffer,
        min_stop_distance_atr_mult,
        max_stop_distance_atr_mult,
        float(base_cfg.initial_equity),
        float(base_cfg.contract_multiplier),
        float(base_cfg.eurusd_rate),
        float(base_cfg.risk_per_trade_pct),
        float(base_cfg.margin_requirement_pct),
        float(base_cfg.max_margin_pct),
        float(base_cfg.lot_size),
        float(base_cfg.spread_points),
        float(base_cfg.slippage_points),
        float(base_cfg.commission_per_unit),
    )

    def _empty_outputs(cap: int) -> tuple:
        return (
            np.empty(cap, dtype=np.int64),    # entry_idx
            np.empty(cap, dtype=np.int64),    # exit_idx
            np.empty(cap, dtype=np.int8),     # direction
            np.empty(cap, dtype=np.int16),    # slot
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
            np.empty(cap, dtype=np.int8),     # margin_capped
        )

    count_out = _empty_outputs(0)
    counted = simulate_session_portfolio_core(*price_args, *scalar_args, *count_out, 0)
    n_trades = counted[0]

    cap = max(int(n_trades), 1)
    out = _empty_outputs(cap)
    stats = simulate_session_portfolio_core(*price_args, *scalar_args, *out, cap)
    n_trades = stats[0]

    (
        out_entry_idx, out_exit_idx, out_direction, out_slot, out_cross_level,
        out_median, out_prev_mid, out_signal_mid, out_entry_price, out_exit_price,
        out_stop_price, out_tp_price, out_units, out_notional, out_margin,
        out_gross, out_extra, out_pnl, out_equity_before, out_equity_after,
        out_status, out_ticks_held, out_margin_capped,
    ) = out

    status_labels = ("HIT_SL", "HIT_TP", "END_OF_DATA")
    eff_rate = base_cfg.eurusd_rate if base_cfg.eurusd_rate > 0 else 1.0
    trades: list[ClosedTrade] = []
    for k in range(n_trades):
        d = int(out_direction[k])
        e = int(out_entry_idx[k])
        x = int(out_exit_idx[k])
        slot = int(out_slot[k])
        trade_cfg = slot_cfgs[slot]
        entry_ts = ns_to_datetime(int(ticks.tick_time_ns[e]))
        exit_ts = ns_to_datetime(int(ticks.tick_time_ns[x]))
        entry_price = float(out_entry_price[k])
        exit_price = float(out_exit_price[k])
        equity_before = float(out_equity_before[k])
        pnl = float(out_pnl[k])
        sign = 1.0 if d == 1 else -1.0
        stop_price = float(out_stop_price[k])
        realized_risk_eur = float(out_units[k]) * abs(entry_price - stop_price) * base_cfg.contract_multiplier / eff_rate
        realized_risk_pct = (realized_risk_eur / equity_before * 100.0) if equity_before > 0 else 0.0
        cross_quantile = trade_cfg.long_cross_quantile if d == 1 else trade_cfg.short_cross_quantile
        (
            profile_low,
            profile_high,
            profile_range,
            cross_position_pct,
            entry_position_pct,
            range_position_deviation_pct,
        ) = _profile_position_values(
            slot_profiles[slot],
            int(local_bar_index[e]),
            float(out_cross_level[k]),
            entry_price,
            cross_quantile,
        )
        trades.append(ClosedTrade(
            signal_ts=entry_ts,
            entry_ts=entry_ts,
            exit_ts=exit_ts,
            direction="LONG" if d == 1 else "SHORT",
            entry_session=session_key_for_code(int(ticks.entry_session_code[e])),
            cross_quantile=cross_quantile,
            cross_level=float(out_cross_level[k]),
            profile_low=profile_low,
            profile_high=profile_high,
            profile_range=profile_range,
            cross_price_range_position_pct=cross_position_pct,
            entry_price_range_position_pct=entry_position_pct,
            range_position_deviation_pct=range_position_deviation_pct,
            median_level=float(out_median[k]),
            signal_mid=float(out_signal_mid[k]),
            previous_mid=float(out_prev_mid[k]),
            entry_bid=float(bid[e]),
            entry_ask=float(ask[e]),
            entry_price=entry_price,
            exit_bid=float(bid[x]),
            exit_ask=float(ask[x]),
            exit_price=exit_price,
            stop_price=stop_price,
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
            realized_risk_eur=realized_risk_eur,
            realized_risk_pct=realized_risk_pct,
            margin_capped=bool(out_margin_capped[k]),
        ))

    result.trades = trades
    result.signals_total = int(stats[1])
    result.long_signals = int(stats[2])
    result.short_signals = int(stats[3])
    result.rejected_signals_missing_band = int(stats[4])
    result.rejected_signals_band_too_narrow = int(stats[5])
    result.rejected_signals_price_range_position = int(stats[6])
    result.rejected_signals_stop_too_small = int(stats[7])
    result.rejected_signals_stop_too_large = int(stats[8])
    result.skipped_signals_no_size = int(stats[9])
    result.ruined = bool(stats[10])
    result.ticks_simulated = int(stats[11])
    result.final_equity = float(stats[12])

    if log_result:
        _log_result(result)
    return result


def _log_result(result: SimulationResult) -> None:
    log.info(
        "Simulation done ticks %d bars %d signals %d trades %d rejected_missing_band %d rejected_profile_range_too_narrow %d rejected_price_range_position %d rejected_stop_too_small %d rejected_stop_too_large %d skipped_no_size %d final_equity %.2f ruined %s",
        result.ticks_simulated, result.bars_total, result.signals_total,
        len(result.trades), result.rejected_signals_missing_band,
        result.rejected_signals_band_too_narrow, result.rejected_signals_price_range_position,
        result.rejected_signals_stop_too_small, result.rejected_signals_stop_too_large, result.skipped_signals_no_size,
        result.final_equity, result.ruined,
    )
