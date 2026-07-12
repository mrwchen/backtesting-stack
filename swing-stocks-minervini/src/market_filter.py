"""Causal IBD-inspired index-state and exposure model.

The primary ETF proxy starts rally attempts and confirms them with a
price-and-volume follow-through day. All configured ETF proxies contribute
distribution days. Cross-sectional stock breadth is a secondary exposure
confirmation, not the primary regime switch.

Every row is calculated with information known only after that session's
close. The simulator therefore applies row t to entries in session t+1.
Stock breadth is causal within the loaded columns, but those columns come from
the currently canonical universe and are not historical point-in-time
membership. Independent-mode ablations use only the binary index-state gate;
breadth changes positive cap magnitudes without changing that binary decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


CORRECTION = "CORRECTION"
RALLY_ATTEMPT = "RALLY_ATTEMPT"
CONFIRMED_UPTREND = "CONFIRMED_UPTREND"
UPTREND_UNDER_PRESSURE = "UPTREND_UNDER_PRESSURE"
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"


def _hysteresis(values: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    state = False
    out = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        if np.isnan(value):
            state = False
        elif state:
            state = value >= off_threshold
        else:
            state = value >= on_threshold
        out[i] = state
    return out


def compute_breadth(close: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return causal breadth within the supplied stock columns."""
    ma200 = close.rolling(200, min_periods=200).mean()
    eligible = (close >= cfg.min_price) & ma200.notna()
    above = (close > ma200) & eligible
    breadth = above.sum(axis=1) / eligible.sum(axis=1).replace(0, np.nan)
    breadth_confirmed = _hysteresis(
        breadth.to_numpy(dtype=float),
        cfg.breadth_on_threshold,
        cfg.breadth_off_threshold,
    )
    return pd.DataFrame(
        {"market_breadth": breadth, "breadth_confirmed": breadth_confirmed},
        index=close.index,
    )


def _index_matrix(
    index_bars: pd.DataFrame,
    field: str,
    dates: pd.DatetimeIndex,
    symbols: tuple[str, ...],
) -> pd.DataFrame:
    if index_bars.empty:
        return pd.DataFrame(index=dates, columns=symbols, dtype=float)
    matrix = index_bars.pivot(index="date", columns="symbol", values=field)
    return matrix.reindex(index=dates, columns=symbols).astype(float)


def _market_exposure_cap(status: str, breadth_confirmed: bool, cfg: Config) -> float:
    if status == CONFIRMED_UPTREND:
        return (
            cfg.market_confirmed_max_exposure_pct
            if breadth_confirmed
            else cfg.market_confirmed_weak_breadth_max_exposure_pct
        )
    if status == UPTREND_UNDER_PRESSURE:
        return (
            cfg.market_under_pressure_max_exposure_pct
            if breadth_confirmed
            else cfg.market_under_pressure_weak_breadth_max_exposure_pct
        )
    return 0.0


def compute_market_model(
    stock_close: pd.DataFrame,
    index_bars: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Build a deterministic market state for each stock trading session."""
    dates = stock_close.index
    symbols = cfg.market_index_symbols
    close = _index_matrix(index_bars, "close", dates, symbols)
    low = _index_matrix(index_bars, "low", dates, symbols)
    volume = _index_matrix(index_bars, "volume", dates, symbols)
    returns = close.pct_change(fill_method=None)
    primary = cfg.market_primary_index
    breadth = compute_breadth(stock_close, cfg)

    statuses: list[str] = []
    rally_days: list[int] = []
    distribution_counts: list[int] = []
    distribution_flags: list[bool] = []
    follow_through_flags: list[bool] = []
    exposure_caps: list[float] = []

    state = CORRECTION
    rally_day = 0
    rally_low = np.nan
    distribution_sessions: list[int] = []

    for t in range(len(dates)):
        clear_distribution_after_output = False
        primary_close = close.iat[t, close.columns.get_loc(primary)]
        primary_low = low.iat[t, low.columns.get_loc(primary)]
        primary_return = returns.iat[t, returns.columns.get_loc(primary)]
        primary_volume = volume.iat[t, volume.columns.get_loc(primary)]
        previous_primary_volume = volume.iat[t - 1, volume.columns.get_loc(primary)] if t else np.nan
        primary_available = all(
            np.isfinite(value)
            for value in (primary_close, primary_low, primary_return, primary_volume, previous_primary_volume)
        )

        distribution_day = False
        for symbol in symbols:
            symbol_return = returns.iat[t, returns.columns.get_loc(symbol)]
            symbol_volume = volume.iat[t, volume.columns.get_loc(symbol)]
            previous_volume = volume.iat[t - 1, volume.columns.get_loc(symbol)] if t else np.nan
            if (
                np.isfinite(symbol_return)
                and np.isfinite(symbol_volume)
                and np.isfinite(previous_volume)
                and symbol_return <= -cfg.distribution_min_loss
                and symbol_volume > previous_volume
            ):
                distribution_day = True
                break

        follow_through_day = False
        if primary_available:
            if state == CORRECTION:
                distribution_sessions.clear()
                if primary_return > 0:
                    state = RALLY_ATTEMPT
                    rally_day = 1
                    rally_low = primary_low
            elif state == RALLY_ATTEMPT:
                if primary_low < rally_low:
                    if primary_return > 0:
                        rally_day = 1
                        rally_low = primary_low
                    else:
                        state = CORRECTION
                        rally_day = 0
                        rally_low = np.nan
                else:
                    rally_day += 1
                    if (
                        rally_day >= cfg.ftd_min_rally_day
                        and primary_return >= cfg.ftd_min_gain
                        and primary_volume > previous_primary_volume
                    ):
                        state = CONFIRMED_UPTREND
                        follow_through_day = True
                        distribution_sessions.clear()
            else:
                distribution_sessions = [
                    session
                    for session in distribution_sessions
                    if t - session < cfg.distribution_lookback_sessions
                ]
                if distribution_day:
                    distribution_sessions.append(t)
                if primary_close < rally_low or len(distribution_sessions) >= cfg.distribution_correction_count:
                    state = CORRECTION
                    rally_day = 0
                    rally_low = np.nan
                    clear_distribution_after_output = True
                elif len(distribution_sessions) >= cfg.distribution_pressure_count:
                    state = UPTREND_UNDER_PRESSURE
                else:
                    state = CONFIRMED_UPTREND

        output_status = state if primary_available else DATA_UNAVAILABLE
        breadth_on = bool(breadth["breadth_confirmed"].iat[t])
        statuses.append(output_status)
        rally_days.append(rally_day if state == RALLY_ATTEMPT else 0)
        distribution_counts.append(len(distribution_sessions))
        distribution_flags.append(distribution_day)
        follow_through_flags.append(follow_through_day)
        exposure_caps.append(_market_exposure_cap(output_status, breadth_on, cfg))
        if clear_distribution_after_output:
            distribution_sessions.clear()

    result = breadth.copy()
    result["primary_index"] = primary
    result["primary_index_close"] = close[primary]
    result["primary_index_volume"] = volume[primary]
    result["primary_index_return_pct"] = returns[primary] * 100.0
    result["market_status"] = statuses
    result["rally_attempt_day"] = rally_days
    result["distribution_day"] = distribution_flags
    result["distribution_days"] = distribution_counts
    result["follow_through_day"] = follow_through_flags
    result["entry_exposure_cap"] = exposure_caps
    result["market_on"] = result["entry_exposure_cap"] > 0
    return result
