"""Tick-level simulation for the hit-frequency median crossing rule."""

import logging

import numpy as np
import pandas as pd

from . import broker, config
from .entities import ClosedTrade, SimulationResult
from .profile import rolling_profile_levels

log = logging.getLogger(__name__)


def run_simulation(ticks: pd.DataFrame, bars: pd.DataFrame) -> SimulationResult:
    bars = bars.copy()
    profile = rolling_profile_levels(bars)
    bars = pd.concat([bars, profile], axis=1)
    profile_by_bar = bars.set_index("bar_start")[
        ["band_lower", "median_level", "band_upper", "band_width_points"]
    ]

    sim_ticks = ticks.copy()
    sim_ticks = sim_ticks.join(profile_by_bar, on="bar_start")

    result = SimulationResult(
        initial_equity=config.INITIAL_EQUITY,
        final_equity=config.INITIAL_EQUITY,
        ticks_total=len(sim_ticks),
        bars_total=len(bars),
    )
    equity = config.INITIAL_EQUITY
    prev_mid: float | None = None
    position: dict | None = None

    def trade_levels(row, direction: str, entry_price: float) -> tuple[float, float, float] | None:
        if config.STOP_MODE == "fixed":
            stop_distance = config.STOP_POINTS
            if direction == "LONG":
                return entry_price - stop_distance, entry_price + config.TAKE_PROFIT_POINTS, stop_distance
            return entry_price + stop_distance, entry_price - config.TAKE_PROFIT_POINTS, stop_distance

        band_lower = float(row.band_lower) if pd.notna(row.band_lower) else np.nan
        band_upper = float(row.band_upper) if pd.notna(row.band_upper) else np.nan
        band_width = float(row.band_width_points) if pd.notna(row.band_width_points) else np.nan
        if not (np.isfinite(band_lower) and np.isfinite(band_upper) and np.isfinite(band_width)):
            return None
        if band_width < config.MIN_BAND_POINTS:
            return None

        if direction == "LONG":
            stop_price = band_lower - config.BAND_STOP_BUFFER_POINTS
            take_profit_price = entry_price + config.TAKE_PROFIT_POINTS
            stop_distance = entry_price - stop_price
        else:
            stop_price = band_upper + config.BAND_STOP_BUFFER_POINTS
            take_profit_price = entry_price - config.TAKE_PROFIT_POINTS
            stop_distance = stop_price - entry_price

        if stop_distance < config.MIN_STOP_POINTS or stop_distance > config.MAX_STOP_POINTS:
            return None
        return stop_price, take_profit_price, stop_distance

    def open_position(row, direction: str, median_level: float, previous_mid: float) -> None:
        nonlocal position
        entry_price = float(row.ask) if direction == "LONG" else float(row.bid)
        levels = trade_levels(row, direction, entry_price)
        if levels is None:
            return
        stop_price, take_profit_price, stop_distance = levels

        sizing = broker.size_position(equity, entry_price, stop_distance)
        if sizing.units <= 0:
            result.skipped_signals_no_size += 1
            return

        position = {
            "signal_ts": row.tick_time,
            "entry_ts": row.tick_time,
            "direction": direction,
            "median_level": median_level,
            "signal_mid": float(row.mid),
            "previous_mid": previous_mid,
            "entry_bid": float(row.bid),
            "entry_ask": float(row.ask),
            "entry_price": entry_price,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "units": sizing.units,
            "notional_eur": sizing.notional_eur,
            "margin_used_eur": sizing.margin_used_eur,
            "equity_before": equity,
            "ticks_held": 0,
        }

    def maybe_close_position(row) -> bool:
        nonlocal equity, position
        if position is None:
            return False

        position["ticks_held"] += 1
        direction = position["direction"]
        exit_quote = float(row.bid) if direction == "LONG" else float(row.ask)
        if direction == "LONG":
            hit_stop = exit_quote <= position["stop_price"]
            hit_tp = exit_quote >= position["take_profit_price"]
        else:
            hit_stop = exit_quote >= position["stop_price"]
            hit_tp = exit_quote <= position["take_profit_price"]

        if not hit_stop and not hit_tp:
            return False

        status = "HIT_SL" if hit_stop else "HIT_TP"
        close_position(row, exit_quote, status)
        return True

    def close_position(row, exit_price: float, status: str) -> None:
        nonlocal equity, position
        if position is None:
            return

        direction = position["direction"]
        units = position["units"]
        gross = broker.gross_pnl(units, position["entry_price"], exit_price, direction)
        extra_costs = broker.extra_round_trip_costs(units)
        pnl = gross - extra_costs
        equity_before = position["equity_before"]
        equity_after = equity + pnl
        sign = 1.0 if direction == "LONG" else -1.0
        price_pnl_points = (exit_price - position["entry_price"]) * sign
        seconds_held = (row.tick_time - position["entry_ts"]).total_seconds()

        result.trades.append(ClosedTrade(
            signal_ts=position["signal_ts"],
            entry_ts=position["entry_ts"],
            exit_ts=row.tick_time,
            direction=direction,
            median_level=position["median_level"],
            signal_mid=position["signal_mid"],
            previous_mid=position["previous_mid"],
            entry_bid=position["entry_bid"],
            entry_ask=position["entry_ask"],
            entry_price=position["entry_price"],
            exit_bid=float(row.bid),
            exit_ask=float(row.ask),
            exit_price=exit_price,
            stop_price=position["stop_price"],
            take_profit_price=position["take_profit_price"],
            units=units,
            notional_eur=position["notional_eur"],
            margin_used_eur=position["margin_used_eur"],
            gross_pnl_eur=gross,
            extra_costs_eur=extra_costs,
            pnl_eur=pnl,
            equity_before=equity_before,
            equity_after=equity_after,
            return_pct=(pnl / equity_before * 100.0) if equity_before > 0 else 0.0,
            price_pnl_points=price_pnl_points,
            outcome_status=status,
            ticks_held=position["ticks_held"],
            seconds_held=float(seconds_held),
        ))
        equity = equity_after
        position = None

    for idx, row in enumerate(sim_ticks.itertuples(index=False), start=1):
        had_position = position is not None
        if had_position:
            maybe_close_position(row)
            prev_mid = float(row.mid)
            result.ticks_simulated = idx
            if equity <= 0:
                result.ruined = True
                break
            continue

        median = float(row.median_level) if pd.notna(row.median_level) else np.nan
        mid = float(row.mid)
        if prev_mid is not None and np.isfinite(median):
            direction = None
            if prev_mid < median <= mid:
                direction = "LONG"
            elif prev_mid > median >= mid:
                direction = "SHORT"

            if direction is not None:
                result.signals_total += 1
                if direction == "LONG":
                    result.long_signals += 1
                else:
                    result.short_signals += 1
                open_position(row, direction, median, prev_mid)

        prev_mid = mid
        result.ticks_simulated = idx

    if position is not None and not sim_ticks.empty:
        row = sim_ticks.iloc[-1]
        exit_price = float(row["bid"]) if position["direction"] == "LONG" else float(row["ask"])
        close_position(row, exit_price, "END_OF_DATA")

    result.final_equity = equity
    log.info(
        "Simulation done ticks %d bars %d signals %d trades %d final_equity %.2f ruined %s",
        result.ticks_simulated, result.bars_total, result.signals_total,
        len(result.trades), result.final_equity, result.ruined,
    )
    return result
