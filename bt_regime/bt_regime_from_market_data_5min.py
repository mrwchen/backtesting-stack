from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


TICKER_MAP = {
    "SPOTCRUDE": "SpotCrude",
    "COPPER": "Copper",
    "QQQ": "QQQ.US",
}


@dataclass(frozen=True)
class Params:
    profile: str = "Dynamisch"
    fast_len: int = 20
    slow_len: int = 50
    di_len: int = 14
    adx_smooth: int = 14
    atr_len: int = 14
    slope_bars: int = 6
    pivot_left: int = 2
    pivot_right: int = 2
    regime_lookback: int = 48
    persistence_lookback: int = 24
    confirm_bars: int = 2
    cooldown_bars: int = 2
    adx_rise_bars: int = 2
    location_source: str = "HMA"  # "HMA" oder "Rolling VWAP"
    rvwap_len: int = 60
    hma_len: int = 21
    use_location: bool = True


@dataclass(frozen=True)
class Thresholds:
    adx_min: float
    ema_dist_min_atr: float
    breakout_dist_atr: float
    strong_score_min: int
    weak_score_min: int
    structure_buffer_atr: float
    er_entry_min: float
    er_exit_min: float
    max_centerline_crosses_entry: float
    max_centerline_crosses_exit: float
    long_persistence_entry: float
    short_persistence_entry: float
    long_persistence_exit: float
    short_persistence_exit: float
    neutral_band_low: float
    neutral_band_high: float


def thresholds_for_profile(profile: str) -> Thresholds:
    dynamic = profile == "Dynamisch"
    return Thresholds(
        adx_min=20.0 if dynamic else 16.0,
        ema_dist_min_atr=0.16 if dynamic else 0.10,
        breakout_dist_atr=0.32 if dynamic else 0.24,
        strong_score_min=6 if dynamic else 5,
        weak_score_min=5 if dynamic else 4,
        structure_buffer_atr=0.10 if dynamic else 0.20,
        er_entry_min=0.22 if dynamic else 0.18,
        er_exit_min=0.16 if dynamic else 0.14,
        max_centerline_crosses_entry=8.0 if dynamic else 10.0,
        max_centerline_crosses_exit=(8.0 if dynamic else 10.0) + 2.0,
        long_persistence_entry=0.58 if dynamic else 0.56,
        short_persistence_entry=0.42 if dynamic else 0.44,
        long_persistence_exit=0.52 if dynamic else 0.51,
        short_persistence_exit=0.48 if dynamic else 0.49,
        neutral_band_low=0.46 if dynamic else 0.47,
        neutral_band_high=0.54 if dynamic else 0.53,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet den Pine-Indikator v2.2 (HMA/Rolling VWAP Location-Filter) auf market_data_5min und schreibt bt_regime. Exchange wird ignoriert."
    )
    parser.add_argument("--start-date", required=True, help="Startdatum oder Startzeit, z. B. 2025-01-01 oder 2025-01-01T00:00:00Z")
    parser.add_argument("--profile", choices=["Dynamisch", "Stabil"], default="Dynamisch")
    parser.add_argument("--timeframe", default="5min")
    parser.add_argument(
        "--location-source",
        choices=["HMA", "Rolling VWAP"],
        default="HMA",
        help="Location-Filter wie im Pine-Indikator (Default: HMA).",
    )
    parser.add_argument("--rvwap-len", type=int, default=60, help="Fenstergröße für Rolling VWAP in Bars")
    parser.add_argument("--hma-len", type=int, default=21, help="Länge des Hull Moving Average")
    parser.add_argument("--warmup-bars", type=int, default=1000, help="Zusätzliche Bars vor dem Startdatum für EMA/ATR/ADX/Pivots")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None, help="Optional mehrfach nutzbar: nur diese Symbole verarbeiten")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--user", default=os.getenv("PGUSER", "backtesting-account"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "backtesting-account-pw"))
    parser.add_argument("--source-db", default=os.getenv("PGSOURCE_DB", "market-data"))
    parser.add_argument("--target-db", default=os.getenv("PGTARGET_DB", "backtesting"))
    parser.add_argument("--source-table", default="public.market_data_5min")
    parser.add_argument("--target-table", default="public.bt_regime")
    parser.add_argument("--disable-location", action="store_true", help="Location-Filter komplett deaktivieren")
    return parser.parse_args()


def parse_start_ts(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def mapped_ticker(symbol: str) -> str:
    return TICKER_MAP.get(symbol, symbol)


def rma(series: pd.Series, length: int) -> pd.Series:
    values = series.astype(float).to_numpy()
    out = np.full(len(values), np.nan, dtype=float)
    seed = series.rolling(length, min_periods=length).mean().to_numpy(dtype=float)
    valid_seed_idx = np.flatnonzero(~np.isnan(seed))
    if len(valid_seed_idx) == 0:
        return pd.Series(out, index=series.index)
    first = int(valid_seed_idx[0])
    out[first] = seed[first]
    for i in range(first + 1, len(values)):
        x = values[i]
        prev = out[i - 1]
        if np.isnan(prev) or np.isnan(x):
            out[i] = np.nan
        else:
            out[i] = (prev * (length - 1) + x) / length
    return pd.Series(out, index=series.index)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def compute_dmi_adx(df: pd.DataFrame, di_len: int, adx_smooth: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=df.index,
        dtype=float,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=df.index,
        dtype=float,
    )
    tr = true_range(df["high"], df["low"], df["close"])
    tr_rma = rma(tr, di_len)
    plus_di = 100.0 * rma(plus_dm, di_len) / tr_rma.replace(0.0, np.nan)
    minus_di = 100.0 * rma(minus_dm, di_len) / tr_rma.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx, adx_smooth)


def compute_rolling_vwap(df: pd.DataFrame, length: int) -> pd.Series:
    """Rolling VWAP über `length` Bars, analog zu Pine:
    rvwapNum = math.sum(hlc3 * volume, length)
    rvwapDen = math.sum(volume, length)
    rvwapValue = rvwapDen > 0 ? rvwapNum / rvwapDen : hlc3
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    if "volume" in df.columns:
        weights = df["volume"].fillna(0).astype(float)
    else:
        # Fallback: tick_count, falls keine Volume-Spalte vorhanden.
        weights = df["tick_count"].fillna(0).astype(float)
    weighted_price = hlc3 * weights
    num = weighted_price.rolling(length, min_periods=1).sum()
    den = weights.rolling(length, min_periods=1).sum()
    return np.where(den > 0, num / den.replace(0.0, np.nan), hlc3)


def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average wie in Pine ta.wma:
    Gewichte = 1, 2, 3, ..., length, Normalisierung durch Summe der Gewichte.
    """
    if length <= 0:
        return pd.Series(np.nan, index=series.index)
    weights = np.arange(1, length + 1, dtype=float)
    weight_sum = weights.sum()

    def _wma(window: np.ndarray) -> float:
        return float(np.dot(window, weights) / weight_sum)

    return series.astype(float).rolling(length, min_periods=length).apply(_wma, raw=True)


def compute_hma(series: pd.Series, length: int) -> pd.Series:
    """Hull Moving Average wie in Pine ta.hma:
    HMA = WMA( 2 * WMA(src, length/2) - WMA(src, length), sqrt(length) )
    """
    if length < 2:
        return series.astype(float).copy()
    half_len = max(1, int(length // 2))
    sqrt_len = max(1, int(math.floor(math.sqrt(length))))
    wma_half = wma(series, half_len)
    wma_full = wma(series, length)
    raw = 2.0 * wma_half - wma_full
    return wma(raw, sqrt_len)


def confirmed_pivot(values: pd.Series, left: int, right: int, is_high: bool) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < left + right + 1:
        return out
    for t in range(left + right, len(arr)):
        pivot_idx = t - right
        window = arr[pivot_idx - left : pivot_idx + right + 1]
        if np.isnan(window).any():
            continue
        center = arr[pivot_idx]
        others = np.concatenate([window[:left], window[left + 1 :]])
        if is_high:
            if center == np.max(window) and center > np.max(others):
                out[t] = center
        else:
            if center == np.min(window) and center < np.min(others):
                out[t] = center
    return out


def rolling_sum_bool(series: pd.Series, window: int) -> pd.Series:
    return series.astype(float).rolling(window, min_periods=window).sum()


def prepare_base_indicators(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=params.fast_len, adjust=False, min_periods=1).mean()
    out["ema_slow"] = out["close"].ewm(span=params.slow_len, adjust=False, min_periods=1).mean()
    tr = true_range(out["high"], out["low"], out["close"])
    out["atr_value"] = rma(tr, params.atr_len)
    out["adx_value"] = compute_dmi_adx(out, params.di_len, params.adx_smooth)

    # Location-Filter: Rolling VWAP oder HMA (analog Pine v2.2).
    rvwap_values = compute_rolling_vwap(out, params.rvwap_len)
    hma_values = compute_hma(out["close"], params.hma_len)
    out["rvwap_value"] = rvwap_values
    out["hma_value"] = hma_values
    if params.location_source == "Rolling VWAP":
        out["location_value"] = rvwap_values
    else:
        out["location_value"] = hma_values

    out["ema_fast_up"] = out["ema_fast"] > out["ema_fast"].shift(params.slope_bars)
    out["ema_fast_down"] = out["ema_fast"] < out["ema_fast"].shift(params.slope_bars)
    out["ema_slow_up"] = out["ema_slow"] > out["ema_slow"].shift(params.slope_bars)
    out["ema_slow_down"] = out["ema_slow"] < out["ema_slow"].shift(params.slope_bars)
    out["adx_rising"] = out["adx_value"] > out["adx_value"].shift(params.adx_rise_bars)

    out["ema_dist_atr"] = np.where(
        out["atr_value"] > 0,
        (out["ema_fast"] - out["ema_slow"]).abs() / out["atr_value"],
        0.0,
    )

    out["pivot_high_conf"] = confirmed_pivot(out["high"], params.pivot_left, params.pivot_right, True)
    out["pivot_low_conf"] = confirmed_pivot(out["low"], params.pivot_left, params.pivot_right, False)

    close_diff_abs = (out["close"] - out["close"].shift(1)).abs()
    path_sum = close_diff_abs.rolling(params.regime_lookback, min_periods=params.regime_lookback).sum()
    enough_regime_bars = np.arange(len(out)) >= params.regime_lookback
    out["er_value"] = np.where(
        enough_regime_bars & path_sum.notna() & (path_sum > 0),
        (out["close"] - out["close"].shift(params.regime_lookback)).abs() / path_sum,
        0.0,
    )

    centerline_cross = (
        ((out["close"] > out["ema_fast"]) & (out["close"].shift(1) <= out["ema_fast"].shift(1)))
        | ((out["close"] < out["ema_fast"]) & (out["close"].shift(1) >= out["ema_fast"].shift(1)))
    ).fillna(False)
    out["centerline_cross_count"] = np.where(
        enough_regime_bars,
        centerline_cross.astype(float).rolling(params.regime_lookback, min_periods=params.regime_lookback).sum(),
        0.0,
    )

    enough_persistence_bars = np.arange(len(out)) >= params.persistence_lookback
    above_slow = (out["close"] > out["ema_slow"]).astype(float)
    out["above_slow_pct"] = np.where(
        enough_persistence_bars,
        above_slow.rolling(params.persistence_lookback, min_periods=params.persistence_lookback).sum() / params.persistence_lookback,
        0.5,
    )

    return out


def apply_structure_logic(df: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    out = df.copy()
    n = len(out)

    has_pivot_structure = np.zeros(n, dtype=bool)
    higher_high = np.zeros(n, dtype=bool)
    higher_low = np.zeros(n, dtype=bool)
    lower_high = np.zeros(n, dtype=bool)
    lower_low = np.zeros(n, dtype=bool)
    structure_long = np.zeros(n, dtype=bool)
    structure_short = np.zeros(n, dtype=bool)

    last_pivot_high = math.nan
    prev_pivot_high = math.nan
    last_pivot_low = math.nan
    prev_pivot_low = math.nan

    highs = out["pivot_high_conf"].to_numpy(dtype=float)
    lows = out["pivot_low_conf"].to_numpy(dtype=float)
    closes = out["close"].to_numpy(dtype=float)
    ema_slow = out["ema_slow"].to_numpy(dtype=float)
    atr = out["atr_value"].to_numpy(dtype=float)

    for i in range(n):
        ph = highs[i]
        pl = lows[i]
        if not np.isnan(ph):
            prev_pivot_high = last_pivot_high
            last_pivot_high = ph
        if not np.isnan(pl):
            prev_pivot_low = last_pivot_low
            last_pivot_low = pl

        has = not any(np.isnan(x) for x in (last_pivot_high, prev_pivot_high, last_pivot_low, prev_pivot_low))
        has_pivot_structure[i] = has

        hh = has and last_pivot_high > prev_pivot_high
        hl = has and last_pivot_low > prev_pivot_low
        lh = has and last_pivot_high < prev_pivot_high
        ll = has and last_pivot_low < prev_pivot_low

        higher_high[i] = hh
        higher_low[i] = hl
        lower_high[i] = lh
        lower_low[i] = ll

        structure_long_sequence = hh and hl
        structure_short_sequence = lh and ll

        if np.isnan(last_pivot_low):
            structure_long_hold = closes[i] > ema_slow[i]
        else:
            structure_long_hold = closes[i] > last_pivot_low - atr[i] * thresholds.structure_buffer_atr

        if np.isnan(last_pivot_high):
            structure_short_hold = closes[i] < ema_slow[i]
        else:
            structure_short_hold = closes[i] < last_pivot_high + atr[i] * thresholds.structure_buffer_atr

        structure_long[i] = (structure_long_sequence and structure_long_hold) if has else (closes[i] > ema_slow[i])
        structure_short[i] = (structure_short_sequence and structure_short_hold) if has else (closes[i] < ema_slow[i])

    out["has_pivot_structure"] = has_pivot_structure
    out["higher_high"] = higher_high
    out["higher_low"] = higher_low
    out["lower_high"] = lower_high
    out["lower_low"] = lower_low
    out["structure_long"] = structure_long
    out["structure_short"] = structure_short
    return out


def apply_regime_logic(df: pd.DataFrame, params: Params, thresholds: Thresholds) -> pd.DataFrame:
    out = df.copy()

    out["adx_ready"] = (out["adx_value"] >= thresholds.adx_min) & (
        out["adx_rising"] | (out["adx_value"] >= thresholds.adx_min + 4.0)
    )
    out["spacing_ok"] = out["ema_dist_atr"] >= thresholds.ema_dist_min_atr

    out["location_long"] = out["close"] > out["ema_fast"]
    out["location_short"] = out["close"] < out["ema_fast"]
    if params.use_location:
        out["location_long"] &= out["close"] > out["location_value"]
        out["location_short"] &= out["close"] < out["location_value"]

    out["in_neutral_band"] = (
        (out["above_slow_pct"] > thresholds.neutral_band_low)
        & (out["above_slow_pct"] < thresholds.neutral_band_high)
    )

    out["fast_trend_long"] = (
        (out["ema_fast"] > out["ema_slow"])
        & out["ema_fast_up"]
        & out["ema_slow_up"]
        & out["location_long"]
    )
    out["fast_trend_short"] = (
        (out["ema_fast"] < out["ema_slow"])
        & out["ema_fast_down"]
        & out["ema_slow_down"]
        & out["location_short"]
    )

    out["breakout_long"] = (
        out["fast_trend_long"]
        & out["spacing_ok"]
        & (out["adx_value"] >= thresholds.adx_min)
        & (out["ema_dist_atr"] >= thresholds.breakout_dist_atr)
    )
    out["breakout_short"] = (
        out["fast_trend_short"]
        & out["spacing_ok"]
        & (out["adx_value"] >= thresholds.adx_min)
        & (out["ema_dist_atr"] >= thresholds.breakout_dist_atr)
    )

    enough_regime_bars = np.arange(len(out)) >= params.regime_lookback
    out["is_chop_entry_base"] = (
        (~pd.Series(enough_regime_bars, index=out.index))
        | (out["er_value"] < thresholds.er_entry_min)
        | (out["centerline_cross_count"] > thresholds.max_centerline_crosses_entry)
        | out["in_neutral_band"]
    )
    out["is_chop_exit_base"] = (
        (~pd.Series(enough_regime_bars, index=out.index))
        | (out["er_value"] < thresholds.er_exit_min)
        | (out["centerline_cross_count"] > thresholds.max_centerline_crosses_exit)
        | (
            (out["above_slow_pct"] > thresholds.short_persistence_exit)
            & (out["above_slow_pct"] < thresholds.long_persistence_exit)
        )
    )
    out["is_chop_entry"] = out["is_chop_entry_base"] & ~out["breakout_long"] & ~out["breakout_short"]
    out["is_chop_exit"] = out["is_chop_exit_base"] & ~out["breakout_long"] & ~out["breakout_short"]

    out["long_score"] = (
        (out["ema_fast"] > out["ema_slow"]).astype(int)
        + out["ema_fast_up"].astype(int)
        + out["ema_slow_up"].astype(int)
        + out["adx_ready"].astype(int)
        + out["spacing_ok"].astype(int)
        + out["location_long"].astype(int)
        + out["structure_long"].astype(int)
    )
    out["short_score"] = (
        (out["ema_fast"] < out["ema_slow"]).astype(int)
        + out["ema_fast_down"].astype(int)
        + out["ema_slow_down"].astype(int)
        + out["adx_ready"].astype(int)
        + out["spacing_ok"].astype(int)
        + out["location_short"].astype(int)
        + out["structure_short"].astype(int)
    )

    out["long_entry_ready"] = (
        ~out["is_chop_entry"]
        & (out["long_score"] >= thresholds.weak_score_min)
        & (out["long_score"] > out["short_score"])
        & out["fast_trend_long"]
        & (out["above_slow_pct"] >= thresholds.long_persistence_entry)
        & out["spacing_ok"]
        & (out["adx_ready"] | out["breakout_long"])
    )
    out["short_entry_ready"] = (
        ~out["is_chop_entry"]
        & (out["short_score"] >= thresholds.weak_score_min)
        & (out["short_score"] > out["long_score"])
        & out["fast_trend_short"]
        & (out["above_slow_pct"] <= thresholds.short_persistence_entry)
        & out["spacing_ok"]
        & (out["adx_ready"] | out["breakout_short"])
    )

    out["long_confirm"] = rolling_sum_bool(out["long_entry_ready"], params.confirm_bars) >= params.confirm_bars
    out["short_confirm"] = rolling_sum_bool(out["short_entry_ready"], params.confirm_bars) >= params.confirm_bars

    out["long_exit"] = (
        (out["is_chop_exit"] & ~out["breakout_long"])
        | ((out["ema_fast"] < out["ema_slow"]) & out["ema_slow_down"] & (out["close"] < out["ema_fast"]))
        | (out["above_slow_pct"] < thresholds.long_persistence_exit)
    )
    out["short_exit"] = (
        (out["is_chop_exit"] & ~out["breakout_short"])
        | ((out["ema_fast"] > out["ema_slow"]) & out["ema_slow_up"] & (out["close"] > out["ema_fast"]))
        | (out["above_slow_pct"] > thresholds.short_persistence_exit)
    )

    regime_state = np.zeros(len(out), dtype=int)
    last_state_change_bar: int | None = None
    current_state = 0

    long_confirm = out["long_confirm"].fillna(False).to_numpy(dtype=bool)
    short_confirm = out["short_confirm"].fillna(False).to_numpy(dtype=bool)
    long_exit = out["long_exit"].fillna(False).to_numpy(dtype=bool)
    short_exit = out["short_exit"].fillna(False).to_numpy(dtype=bool)
    long_score = out["long_score"].to_numpy(dtype=int)
    short_score = out["short_score"].to_numpy(dtype=int)

    for i in range(len(out)):
        prev_state = current_state
        bars_since_state_change = 100000 if last_state_change_bar is None else i - last_state_change_bar
        cooldown_ready = bars_since_state_change >= params.cooldown_bars
        next_state = prev_state

        if prev_state == 1:
            if short_confirm[i] and cooldown_ready:
                next_state = -1
            elif long_exit[i]:
                next_state = 0
            else:
                next_state = 1
        elif prev_state == -1:
            if long_confirm[i] and cooldown_ready:
                next_state = 1
            elif short_exit[i]:
                next_state = 0
            else:
                next_state = -1
        else:
            if long_confirm[i] and (not short_confirm[i]) and cooldown_ready:
                next_state = 1
            elif short_confirm[i] and (not long_confirm[i]) and cooldown_ready:
                next_state = -1
            elif long_confirm[i] and short_confirm[i] and cooldown_ready:
                next_state = 1 if long_score[i] > short_score[i] else -1 if short_score[i] > long_score[i] else 0
            else:
                next_state = 0

        current_state = next_state
        if current_state != prev_state:
            last_state_change_bar = i
        regime_state[i] = current_state

    out["regime_state"] = regime_state
    out["is_strong_long"] = (out["regime_state"] == 1) & (out["long_score"] >= thresholds.strong_score_min)
    out["is_weak_long"] = (out["regime_state"] == 1) & ~out["is_strong_long"]
    out["is_strong_short"] = (out["regime_state"] == -1) & (out["short_score"] >= thresholds.strong_score_min)
    out["is_weak_short"] = (out["regime_state"] == -1) & ~out["is_strong_short"]
    out["is_neutral"] = ~(
        out["is_strong_long"]
        | out["is_weak_long"]
        | out["is_strong_short"]
        | out["is_weak_short"]
    )

    out["trend_text"] = np.select(
        [
            out["is_strong_long"],
            out["is_weak_long"],
            out["is_strong_short"],
            out["is_weak_short"],
        ],
        ["Strong Long", "Weak Long", "Strong Short", "Weak Short"],
        default="Neutral / Chop",
    )
    out["entry_text"] = np.select(
        [out["is_strong_long"] | out["is_weak_long"], out["is_strong_short"] | out["is_weak_short"]],
        ["Nur Long", "Nur Short"],
        default="Kein Trade",
    )
    return out


def compute_indicator_for_symbol(df_symbol: pd.DataFrame, params: Params) -> pd.DataFrame:
    thresholds = thresholds_for_profile(params.profile)
    base = prepare_base_indicators(df_symbol, params)
    structured = apply_structure_logic(base, thresholds)
    final = apply_regime_logic(structured, params, thresholds)
    return final


def fetch_market_data(engine: Engine, table_name: str, start_ts: datetime, warmup_bars: int, symbols: list[str] | None) -> pd.DataFrame:
    from sqlalchemy import text

    fetch_from = start_ts - timedelta(minutes=5 * warmup_bars)
    # Versuche volume mitzuladen; falls die Spalte nicht existiert, fallen wir
    # auf ein Query ohne volume zurück (Rolling VWAP nutzt dann tick_count als Gewicht).
    base_cols = "symbol, bar_time, tick_count, open, high, low, close"
    symbol_filter = "AND symbol = ANY(:symbols)" if symbols else ""
    sql_with_volume = text(
        f"""
        SELECT {base_cols}, volume
        FROM {table_name}
        WHERE bar_time >= :fetch_from
        {symbol_filter}
        ORDER BY symbol, bar_time
        """
    )
    sql_without_volume = text(
        f"""
        SELECT {base_cols}
        FROM {table_name}
        WHERE bar_time >= :fetch_from
        {symbol_filter}
        ORDER BY symbol, bar_time
        """
    )
    query_params: dict[str, object] = {"fetch_from": fetch_from}
    if symbols:
        query_params["symbols"] = symbols
    try:
        df = pd.read_sql_query(sql_with_volume, engine, params=query_params, parse_dates=["bar_time"])
    except Exception as exc:
        # Wenn die volume-Spalte nicht existiert, liefert Postgres "UndefinedColumn".
        # SQLAlchemy verpackt das in ProgrammingError; wir prüfen den Text robust.
        message = str(exc).lower()
        if "volume" in message and ("column" in message or "undefinedcolumn" in message):
            df = pd.read_sql_query(sql_without_volume, engine, params=query_params, parse_dates=["bar_time"])
        else:
            raise
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    numeric_cols = ["tick_count", "open", "high", "low", "close"]
    if "volume" in df.columns:
        numeric_cols.append("volume")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "bar_time", "high", "low", "close"]).copy()
    return df


def build_output_rows(df_all: pd.DataFrame, start_ts: datetime, timeframe: str) -> pd.DataFrame:
    out = df_all.loc[df_all["bar_time"] >= start_ts].copy()
    if out.empty:
        return out
    now_utc = datetime.now(timezone.utc)
    out["received_at"] = now_utc
    out["event_time"] = out["bar_time"] + pd.Timedelta(minutes=5)
    out["ticker"] = out["symbol"].map(mapped_ticker)
    out["exchange"] = ""
    out["timeframe"] = timeframe
    out["long_score"] = out["long_score"].astype(float)
    out["short_score"] = out["short_score"].astype(float)

    cols = [
        "received_at",
        "event_time",
        "ticker",
        "exchange",
        "timeframe",
        "long_score",
        "short_score",
        "is_strong_long",
        "is_weak_long",
        "is_strong_short",
        "is_weak_short",
        "is_neutral",
        "trend_text",
        "entry_text",
    ]
    return out[cols].sort_values(["ticker", "event_time"]).reset_index(drop=True)


def delete_existing_rows(conn, table_name: str, start_ts: datetime, timeframe: str, tickers: list[str]) -> None:
    if not tickers:
        return
    sql = f"""
        DELETE FROM {table_name}
        WHERE event_time >= %s
          AND timeframe = %s
          AND ticker = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_ts, timeframe, list(tickers)))


def insert_rows(conn, table_name: str, df_out: pd.DataFrame) -> int:
    if df_out.empty:
        return 0
    sql = f"""
        INSERT INTO {table_name} (
            received_at,
            event_time,
            ticker,
            exchange,
            timeframe,
            long_score,
            short_score,
            is_strong_long,
            is_weak_long,
            is_strong_short,
            is_weak_short,
            is_neutral,
            trend_text,
            entry_text
        ) VALUES (
            %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s
        )
    """
    records = [
        (
            row.received_at.to_pydatetime() if hasattr(row.received_at, "to_pydatetime") else row.received_at,
            row.event_time.to_pydatetime() if hasattr(row.event_time, "to_pydatetime") else row.event_time,
            row.ticker,
            row.exchange,
            row.timeframe,
            float(row.long_score),
            float(row.short_score),
            bool(row.is_strong_long),
            bool(row.is_weak_long),
            bool(row.is_strong_short),
            bool(row.is_weak_short),
            bool(row.is_neutral),
            row.trend_text,
            row.entry_text,
        )
        for row in df_out.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, records)
    return len(records)


def _make_engine(host: str, port: int, user: str, password: str, dbname: str) -> Engine:
    from urllib.parse import quote_plus

    url = (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(dbname)}"
    )
    return create_engine(url, future=True)


def main() -> None:
    args = parse_args()
    start_ts = parse_start_ts(args.start_date)
    params = Params(
        profile=args.profile,
        location_source=args.location_source,
        rvwap_len=args.rvwap_len,
        hma_len=args.hma_len,
        use_location=not args.disable_location,
    )

    source_engine = _make_engine(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.source_db,
    )
    try:
        market_df = fetch_market_data(
            engine=source_engine,
            table_name=args.source_table,
            start_ts=start_ts,
            warmup_bars=args.warmup_bars,
            symbols=args.symbols,
        )
    finally:
        source_engine.dispose()

    if market_df.empty:
        print("Keine Marktdaten gefunden.")
        return

    results = []
    for symbol, df_symbol in market_df.groupby("symbol", sort=True):
        df_symbol = df_symbol.sort_values("bar_time").reset_index(drop=True)
        calc = compute_indicator_for_symbol(df_symbol, params)
        calc["symbol"] = symbol
        results.append(calc)

    full_df = pd.concat(results, ignore_index=True)
    output_df = build_output_rows(full_df, start_ts, args.timeframe)

    if output_df.empty:
        print("Es gibt ab dem Startdatum keine berechneten Zielzeilen.")
        return

    target_engine = _make_engine(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.target_db,
    )
    try:
        # Raw DBAPI-Connection aus der Engine holen, damit delete_existing_rows und
        # insert_rows unverändert mit psycopg-Cursor-API arbeiten können.
        raw_conn = target_engine.raw_connection()
        try:
            tickers_to_refresh = sorted(output_df["ticker"].dropna().unique().tolist())
            delete_existing_rows(
                conn=raw_conn,
                table_name=args.target_table,
                start_ts=start_ts,
                timeframe=args.timeframe,
                tickers=tickers_to_refresh,
            )
            inserted = insert_rows(raw_conn, args.target_table, output_df)
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()
    finally:
        target_engine.dispose()

    print(
        f"Fertig. {inserted} Zeilen nach {args.target_table} geschrieben "
        f"ab {start_ts.isoformat()} für {output_df['ticker'].nunique()} Ticker "
        f"({', '.join(tickers_to_refresh)})."
    )


if __name__ == "__main__":
    main()
