from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


TICKER_MAP = {
    "SPOTCRUDE": "SpotCrude",
    "COPPER": "Copper",
    "QQQ": "QQQ.US",
}

INDICATOR_NAME = "1m_buy_sell_indikator-wei"


# =========================================================================
# Parameter / Profile
# =========================================================================

@dataclass(frozen=True)
class Params:
    profile: str = "Stabil"                  # "Stabil" | "Dynamisch"
    entry_mode: str = "Aggressiv"            # "Aggressiv" | "Ausgeglichen" | "Konservativ"
    entry_fast_len: int = 10
    entry_slow_len: int = 30
    regime_fast_len: int = 100
    regime_slow_len: int = 250
    regime_slope_lookback: int = 15
    regime_pivot_left: int = 8
    regime_pivot_right: int = 8
    use_vwap: bool = True
    regime_di_len: int = 21
    regime_adx_smooth: int = 21
    atr_len: int = 14
    regime_atr_len: int = 50
    use_auto_rsi: bool = True
    manual_rsi_filter: bool = False
    confirmation_mode: str = "Break-and-Hold"  # "Break-and-Hold" | "Effizienz" | "Volumen" | "Keiner"
    vol_len: int = 20
    eff_len: int = 8
    rsi_len: int = 14
    stop_lookback: int = 5


@dataclass(frozen=True)
class Thresholds:
    adx_min: float
    ema_dist_min_atr: float
    max_entry_stretch_atr: float
    min_pullback_bars: int
    min_counter_bars: int
    max_pullback_depth_atr: float
    trigger_range_factor: float
    break_buffer_atr: float
    structure_buffer_atr: float
    pullback_ema50_tolerance_atr: float
    min_regime_score: int
    max_armed_bars: int
    long_rsi_threshold: float
    short_rsi_threshold: float
    break_hold_atr_min: float
    efficiency_min: float
    use_rsi_filter: bool


def thresholds_for(params: Params) -> Thresholds:
    is_aggressive = params.entry_mode == "Aggressiv"
    is_balanced = params.entry_mode == "Ausgeglichen"
    is_conservative = params.entry_mode == "Konservativ"
    is_volatile = params.profile == "Dynamisch"

    adx_min = 24.0 if is_volatile else 18.0
    ema_dist_min_atr = 0.22 if is_volatile else 0.12

    if is_aggressive:
        max_entry_stretch_atr = 0.95 if is_volatile else 1.25
        max_pullback_depth_atr = 1.00 if is_volatile else 1.35
        trigger_range_factor = 0.90 if is_volatile else 0.80
        break_buffer_atr = 0.00
        pullback_ema50_tolerance_atr = 0.15 if is_volatile else 0.25
        min_regime_score = 4 if is_volatile else 3
        max_armed_bars = 7 if is_volatile else 8
        long_rsi_threshold = 50.0
        short_rsi_threshold = 50.0
        break_hold_atr_min = 0.04 if is_volatile else 0.05
        efficiency_min = 0.28 if is_volatile else 0.24
    elif is_balanced:
        max_entry_stretch_atr = 0.85 if is_volatile else 1.10
        max_pullback_depth_atr = 0.88 if is_volatile else 1.18
        trigger_range_factor = 0.98 if is_volatile else 0.88
        break_buffer_atr = 0.01 if is_volatile else 0.005
        pullback_ema50_tolerance_atr = 0.11 if is_volatile else 0.20
        min_regime_score = 4 if is_volatile else 4
        max_armed_bars = 6 if is_volatile else 7
        long_rsi_threshold = 52.0 if is_volatile else 51.0
        short_rsi_threshold = 48.0 if is_volatile else 49.0
        break_hold_atr_min = 0.06 if is_volatile else 0.08
        efficiency_min = 0.34 if is_volatile else 0.30
    else:  # Konservativ
        max_entry_stretch_atr = 0.75 if is_volatile else 1.00
        max_pullback_depth_atr = 0.75 if is_volatile else 1.05
        trigger_range_factor = 1.05 if is_volatile else 0.95
        break_buffer_atr = 0.02 if is_volatile else 0.01
        pullback_ema50_tolerance_atr = 0.08 if is_volatile else 0.15
        min_regime_score = 5 if is_volatile else 4
        max_armed_bars = 5 if is_volatile else 6
        long_rsi_threshold = 54.0 if is_volatile else 52.0
        short_rsi_threshold = 46.0 if is_volatile else 48.0
        break_hold_atr_min = 0.10 if is_volatile else 0.12
        efficiency_min = 0.40 if is_volatile else 0.36

    structure_buffer_atr = 0.10 if is_volatile else 0.20
    use_rsi_filter = is_conservative if params.use_auto_rsi else params.manual_rsi_filter

    return Thresholds(
        adx_min=adx_min,
        ema_dist_min_atr=ema_dist_min_atr,
        max_entry_stretch_atr=max_entry_stretch_atr,
        min_pullback_bars=2 if is_aggressive else 1,
        min_counter_bars=1,
        max_pullback_depth_atr=max_pullback_depth_atr,
        trigger_range_factor=trigger_range_factor,
        break_buffer_atr=break_buffer_atr,
        structure_buffer_atr=structure_buffer_atr,
        pullback_ema50_tolerance_atr=pullback_ema50_tolerance_atr,
        min_regime_score=min_regime_score,
        max_armed_bars=max_armed_bars,
        long_rsi_threshold=long_rsi_threshold,
        short_rsi_threshold=short_rsi_threshold,
        break_hold_atr_min=break_hold_atr_min,
        efficiency_min=efficiency_min,
        use_rsi_filter=use_rsi_filter,
    )


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet den 1m Buy/Sell-Indikator (Pine v3) auf market_data_1min und schreibt bt_signal."
    )
    parser.add_argument("--start-date", required=True,
                        help="Startdatum oder -zeit, z. B. 2025-01-01 oder 2025-01-01T00:00:00Z")
    parser.add_argument("--profile", choices=["Stabil", "Dynamisch"], default="Stabil")
    parser.add_argument("--entry-mode", choices=["Aggressiv", "Ausgeglichen", "Konservativ"], default="Aggressiv")
    parser.add_argument("--confirmation-mode",
                        choices=["Break-and-Hold", "Effizienz", "Volumen", "Keiner"],
                        default="Break-and-Hold")
    parser.add_argument("--timeframe", default="1min")
    parser.add_argument("--session-timezone", default="UTC",
                        help="Tageswechsel für Session-VWAP. Pine resettet i.d.R. an der Börsensession; Default UTC kann abweichen.")
    parser.add_argument("--disable-vwap", action="store_true", help="VWAP-Regime-Filter deaktivieren")
    parser.add_argument("--disable-auto-rsi", action="store_true", help="RSI-Auto-Modus aus; stattdessen manual_rsi_filter nutzen")
    parser.add_argument("--manual-rsi-filter", action="store_true", help="Nur mit --disable-auto-rsi wirksam")
    parser.add_argument("--warmup-bars", type=int, default=500,
                        help="Zusätzliche Bars vor start-date zum Warmup von EMA/ADX/Pivots (250 EMA + Puffer)")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None,
                        help="Mehrfach nutzbar: nur diese Symbole verarbeiten")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--user", default=os.getenv("PGUSER", "backtesting-account"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "backtesting-account-pw"))
    parser.add_argument("--source-db", default=os.getenv("PGSOURCE_DB", "market-data"))
    parser.add_argument("--target-db", default=os.getenv("PGTARGET_DB", "backtesting"))
    parser.add_argument("--source-table", default="public.market_data_1min")
    parser.add_argument("--target-table", default="public.bt_signal")
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


# =========================================================================
# Indikator-Primitiven
# =========================================================================

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
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)


def compute_dmi_adx(df: pd.DataFrame, di_len: int, adx_smooth: int) -> pd.Series:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
                        index=df.index, dtype=float)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
                         index=df.index, dtype=float)
    tr = true_range(df["high"], df["low"], df["close"])
    tr_rma = rma(tr, di_len)
    plus_di = 100.0 * rma(plus_dm, di_len) / tr_rma.replace(0.0, np.nan)
    minus_di = 100.0 * rma(minus_dm, di_len) / tr_rma.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return rma(dx, adx_smooth)


def compute_rsi(close: pd.Series, length: int) -> pd.Series:
    """Wilder-RSI wie in Pine ta.rsi."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Wenn avg_loss == 0 -> rsi = 100; wenn avg_gain == 0 und avg_loss > 0 -> rsi = 0
    rsi = rsi.where(~(avg_loss == 0.0), 100.0)
    rsi = rsi.where(~((avg_gain == 0.0) & (avg_loss > 0.0)), 0.0)
    return rsi


def compute_session_vwap(df: pd.DataFrame, session_timezone: str) -> pd.Series:
    """Session-VWAP mit hlc3 als Preis. market_data_1min hat kein volume -> tick_count als Gewicht.
    Session-Reset zu Mitternacht der angegebenen Timezone.
    Hinweis: weicht von Pines ta.vwap ab, das echtes Volume und Börsensession nutzt.
    """
    hlc3 = (df["high"] + df["low"] + df["close"]) / 3.0
    weights = df["tick_count"].fillna(0).astype(float) if "tick_count" in df.columns else pd.Series(1.0, index=df.index)
    ts = pd.to_datetime(df["bar_time"], utc=True)
    session_key = ts.dt.tz_convert(session_timezone).dt.strftime("%Y-%m-%d").values
    # Gruppen-cumsum über session_key
    tmp = pd.DataFrame({"num": (hlc3 * weights).values, "den": weights.values, "k": session_key},
                       index=df.index)
    cum_num = tmp.groupby("k")["num"].cumsum()
    cum_den = tmp.groupby("k")["den"].cumsum().replace(0.0, np.nan)
    return cum_num / cum_den


def confirmed_pivot(values: pd.Series, left: int, right: int, is_high: bool) -> np.ndarray:
    """Bestätigte Pivot-Werte wie in Pine ta.pivothigh / ta.pivotlow:
    Der Wert taucht erst `right` Bars nach dem Pivot-Kerzen-Index an Index t=pivot_idx+right auf.
    """
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < left + right + 1:
        return out
    for t in range(left + right, len(arr)):
        pivot_idx = t - right
        window = arr[pivot_idx - left: pivot_idx + right + 1]
        if np.isnan(window).any():
            continue
        center = arr[pivot_idx]
        others = np.concatenate([window[:left], window[left + 1:]])
        if is_high:
            if center == np.max(window) and center > np.max(others):
                out[t] = center
        else:
            if center == np.min(window) and center < np.min(others):
                out[t] = center
    return out


def value_when(condition: np.ndarray, source: np.ndarray, occurrence: int) -> np.ndarray:
    """Emuliert Pine ta.valuewhen(condition, source, occurrence).
    occurrence=0 -> letzter passender Wert, 1 -> vorletzter, ...
    """
    n = len(condition)
    out = np.full(n, np.nan, dtype=float)
    # Ring-Buffer der letzten (occurrence+1) Werte, bei denen condition true war
    buf: list[float] = []
    for i in range(n):
        if condition[i] and not np.isnan(source[i]):
            buf.append(float(source[i]))
            if len(buf) > occurrence + 1:
                buf.pop(0)
        if len(buf) >= occurrence + 1:
            out[i] = buf[-1 - occurrence]
    return out


# =========================================================================
# Berechnungs-Pipeline
# =========================================================================

def prepare_base(df: pd.DataFrame, params: Params, session_timezone: str) -> pd.DataFrame:
    out = df.copy()

    # Entry-Ebene
    out["ema_fast"] = out["close"].ewm(span=params.entry_fast_len, adjust=False, min_periods=1).mean()
    out["ema_slow"] = out["close"].ewm(span=params.entry_slow_len, adjust=False, min_periods=1).mean()
    tr = true_range(out["high"], out["low"], out["close"])
    out["atr_value"] = rma(tr, params.atr_len)
    out["rsi_value"] = compute_rsi(out["close"], params.rsi_len)
    out["bar_range"] = out["high"] - out["low"]
    out["impulse_avg"] = out["bar_range"].rolling(10, min_periods=10).mean()

    # Efficiency (Kaufman-ähnlich)
    price_change_abs = (out["close"] - out["close"].shift(1)).abs()
    noise_sum = price_change_abs.rolling(params.eff_len, min_periods=1).sum().fillna(0.0)
    out["efficiency"] = np.where(
        noise_sum > 0,
        (out["close"] - out["close"].shift(params.eff_len)).abs() / noise_sum,
        0.0,
    )

    # Volumen-Proxy: market_data_1min hat kein volume; tick_count als Proxy.
    # Hinweis im Header dokumentiert.
    if "volume" in out.columns:
        vol_series = out["volume"].astype(float)
    else:
        vol_series = out["tick_count"].astype(float) if "tick_count" in out.columns else pd.Series(0.0, index=out.index)
    out["volume_proxy"] = vol_series
    out["vol_avg"] = vol_series.rolling(params.vol_len, min_periods=params.vol_len).mean()

    # Regime-Ebene
    out["regime_fast"] = out["close"].ewm(span=params.regime_fast_len, adjust=False, min_periods=1).mean()
    out["regime_slow"] = out["close"].ewm(span=params.regime_slow_len, adjust=False, min_periods=1).mean()
    out["regime_atr"] = rma(tr, params.regime_atr_len)
    out["regime_adx"] = compute_dmi_adx(out, params.regime_di_len, params.regime_adx_smooth)
    out["regime_vwap"] = compute_session_vwap(out, session_timezone)

    # Regime-Slopes
    out["regime_fast_up"] = out["regime_fast"] > out["regime_fast"].shift(params.regime_slope_lookback)
    out["regime_fast_down"] = out["regime_fast"] < out["regime_fast"].shift(params.regime_slope_lookback)
    out["regime_slow_up"] = out["regime_slow"] > out["regime_slow"].shift(params.regime_slope_lookback)
    out["regime_slow_down"] = out["regime_slow"] < out["regime_slow"].shift(params.regime_slope_lookback)

    out["regime_dist_atr"] = np.where(
        out["regime_atr"] > 0,
        (out["regime_fast"] - out["regime_slow"]).abs() / out["regime_atr"],
        0.0,
    )

    # Regime-Pivots mit valuewhen-Emulation
    ph_conf = confirmed_pivot(out["high"], params.regime_pivot_left, params.regime_pivot_right, True)
    pl_conf = confirmed_pivot(out["low"], params.regime_pivot_left, params.regime_pivot_right, False)
    out["regime_last_pivot_high"] = value_when(~np.isnan(ph_conf), ph_conf, 0)
    out["regime_prev_pivot_high"] = value_when(~np.isnan(ph_conf), ph_conf, 1)
    out["regime_last_pivot_low"] = value_when(~np.isnan(pl_conf), pl_conf, 0)
    out["regime_prev_pivot_low"] = value_when(~np.isnan(pl_conf), pl_conf, 1)

    return out


def apply_regime(df: pd.DataFrame, params: Params, thresholds: Thresholds) -> pd.DataFrame:
    out = df.copy()

    out["regime_spacing_ok"] = out["regime_dist_atr"] >= thresholds.ema_dist_min_atr

    has_struct = (
        out["regime_last_pivot_high"].notna()
        & out["regime_prev_pivot_high"].notna()
        & out["regime_last_pivot_low"].notna()
        & out["regime_prev_pivot_low"].notna()
    )
    out["regime_has_pivot_structure"] = has_struct

    long_seq = has_struct & (out["regime_last_pivot_high"] > out["regime_prev_pivot_high"]) & (out["regime_last_pivot_low"] > out["regime_prev_pivot_low"])
    short_seq = has_struct & (out["regime_last_pivot_high"] < out["regime_prev_pivot_high"]) & (out["regime_last_pivot_low"] < out["regime_prev_pivot_low"])

    # structure_long/short_hold: wenn Pivots da sind, mit ATR-Puffer gegen lastPivot; sonst Fallback auf regime_slow
    long_hold_with_piv = out["close"] > (out["regime_last_pivot_low"] - out["regime_atr"] * thresholds.structure_buffer_atr)
    long_hold_no_piv = out["close"] > out["regime_slow"]
    long_hold = out["regime_last_pivot_low"].notna() & long_hold_with_piv
    long_hold = long_hold | (out["regime_last_pivot_low"].isna() & long_hold_no_piv)

    short_hold_with_piv = out["close"] < (out["regime_last_pivot_high"] + out["regime_atr"] * thresholds.structure_buffer_atr)
    short_hold_no_piv = out["close"] < out["regime_slow"]
    short_hold = out["regime_last_pivot_high"].notna() & short_hold_with_piv
    short_hold = short_hold | (out["regime_last_pivot_high"].isna() & short_hold_no_piv)

    out["regime_structure_long"] = np.where(has_struct, long_seq & long_hold, out["close"] > out["regime_slow"])
    out["regime_structure_short"] = np.where(has_struct, short_seq & short_hold, out["close"] < out["regime_slow"])

    # Regime-Location
    loc_long = out["close"] > out["regime_fast"]
    loc_short = out["close"] < out["regime_fast"]
    if params.use_vwap:
        loc_long &= out["close"] > out["regime_vwap"]
        loc_short &= out["close"] < out["regime_vwap"]
    out["regime_location_long"] = loc_long
    out["regime_location_short"] = loc_short

    # Regime-Scores
    out["regime_long_score"] = (
        (out["regime_fast"] > out["regime_slow"]).astype(int)
        + out["regime_fast_up"].astype(int)
        + out["regime_slow_up"].astype(int)
        + (out["regime_adx"] >= thresholds.adx_min).astype(int)
        + out["regime_spacing_ok"].astype(int)
        + out["regime_location_long"].astype(int)
        + out["regime_structure_long"].astype(int)
    )
    out["regime_short_score"] = (
        (out["regime_fast"] < out["regime_slow"]).astype(int)
        + out["regime_fast_down"].astype(int)
        + out["regime_slow_down"].astype(int)
        + (out["regime_adx"] >= thresholds.adx_min).astype(int)
        + out["regime_spacing_ok"].astype(int)
        + out["regime_location_short"].astype(int)
        + out["regime_structure_short"].astype(int)
    )

    out["long_bias"] = (out["regime_long_score"] >= thresholds.min_regime_score) & (out["regime_long_score"] > out["regime_short_score"])
    out["short_bias"] = (out["regime_short_score"] >= thresholds.min_regime_score) & (out["regime_short_score"] > out["regime_long_score"])

    return out


def apply_signal_state_machine(df: pd.DataFrame, params: Params, thresholds: Thresholds) -> pd.DataFrame:
    """Portiert den var-basierten Zustandsautomaten aus Pine: Pullback-Tracking,
    Armed-Zustand und Entry-Auslösung.
    """
    out = df.copy()
    n = len(out)

    # Arrays vorbereiten
    close = out["close"].to_numpy(dtype=float)
    open_ = out["open"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    ema_fast = out["ema_fast"].to_numpy(dtype=float)
    ema_slow = out["ema_slow"].to_numpy(dtype=float)
    atr = out["atr_value"].to_numpy(dtype=float)
    rsi = out["rsi_value"].to_numpy(dtype=float)
    bar_range = out["bar_range"].to_numpy(dtype=float)
    impulse_avg = out["impulse_avg"].to_numpy(dtype=float)
    efficiency = out["efficiency"].to_numpy(dtype=float)
    vol_proxy = out["volume_proxy"].to_numpy(dtype=float)
    vol_avg = out["vol_avg"].to_numpy(dtype=float)
    long_bias = out["long_bias"].to_numpy(dtype=bool)
    short_bias = out["short_bias"].to_numpy(dtype=bool)

    # Overextension
    price_stretch_long = np.where(atr > 0, (close - ema_fast) / atr, 0.0)
    price_stretch_short = np.where(atr > 0, (ema_fast - close) / atr, 0.0)
    overext_long = price_stretch_long > thresholds.max_entry_stretch_atr
    overext_short = price_stretch_short > thresholds.max_entry_stretch_atr

    # State
    long_pb_bars = 0
    short_pb_bars = 0
    long_pb_bear_bars = 0
    short_pb_bull_bars = 0
    long_pb_high = math.nan
    long_pb_low = math.nan
    short_pb_high = math.nan
    short_pb_low = math.nan

    long_armed = False
    short_armed = False
    long_armed_bars = 0
    short_armed_bars = 0
    long_break_level = math.nan
    short_break_level = math.nan
    long_invalidation = math.nan
    short_invalidation = math.nan
    last_long_pullback_depth_atr = math.nan
    last_short_pullback_depth_atr = math.nan

    # Outputs
    long_entry_arr = np.zeros(n, dtype=bool)
    short_entry_arr = np.zeros(n, dtype=bool)
    long_armed_arr = np.zeros(n, dtype=bool)
    short_armed_arr = np.zeros(n, dtype=bool)

    # close[eff_len] für Effizienz-OK
    eff_len = params.eff_len
    close_eff = np.concatenate([np.full(eff_len, np.nan), close[:-eff_len]]) if eff_len < n else np.full(n, np.nan)

    for i in range(n):
        # Pullback-Kandidaten bestimmen (nur wenn Bias gesetzt und EMA-Konstellation stimmt)
        long_pb_cand = (
            long_bias[i] and ema_fast[i] > ema_slow[i] and low[i] <= ema_fast[i]
            and low[i] > ema_slow[i] - atr[i] * thresholds.pullback_ema50_tolerance_atr
        )
        short_pb_cand = (
            short_bias[i] and ema_fast[i] < ema_slow[i] and high[i] >= ema_fast[i]
            and high[i] < ema_slow[i] + atr[i] * thresholds.pullback_ema50_tolerance_atr
        )

        # Reset bei fehlendem Bias
        if not long_bias[i]:
            long_pb_bars = 0
            long_pb_bear_bars = 0
            long_pb_high = math.nan
            long_pb_low = math.nan
            long_armed = False
            long_armed_bars = 0
            long_break_level = math.nan
            long_invalidation = math.nan
            last_long_pullback_depth_atr = math.nan

        if not short_bias[i]:
            short_pb_bars = 0
            short_pb_bull_bars = 0
            short_pb_high = math.nan
            short_pb_low = math.nan
            short_armed = False
            short_armed_bars = 0
            short_break_level = math.nan
            short_invalidation = math.nan
            last_short_pullback_depth_atr = math.nan

        # Long-Pullback-Tracking
        if long_pb_cand:
            if long_pb_bars == 0:
                long_armed = False
                long_armed_bars = 0
                long_break_level = math.nan
                long_invalidation = math.nan
                long_pb_high = high[i]
                long_pb_low = low[i]
                long_pb_bear_bars = 1 if close[i] < open_[i] else 0
            else:
                long_pb_high = max(long_pb_high, high[i])
                long_pb_low = min(long_pb_low, low[i])
                if close[i] < open_[i]:
                    long_pb_bear_bars += 1
            long_pb_bars += 1
        elif long_pb_bars > 0:
            # Pullback abgeschlossen -> validieren
            if atr[i] > 0 and not math.isnan(long_pb_low):
                last_long_pullback_depth_atr = (ema_fast[i] - long_pb_low) / atr[i]
            else:
                last_long_pullback_depth_atr = math.nan

            valid = (
                long_pb_bars >= thresholds.min_pullback_bars
                and long_pb_bear_bars >= thresholds.min_counter_bars
                and not math.isnan(last_long_pullback_depth_atr)
                and last_long_pullback_depth_atr <= thresholds.max_pullback_depth_atr
                and long_pb_low > ema_slow[i] - atr[i] * thresholds.pullback_ema50_tolerance_atr
            )
            if valid:
                long_armed = True
                long_armed_bars = 0
                long_break_level = long_pb_high
                long_invalidation = long_pb_low
            long_pb_bars = 0
            long_pb_bear_bars = 0
            long_pb_high = math.nan
            long_pb_low = math.nan

        # Short-Pullback-Tracking
        if short_pb_cand:
            if short_pb_bars == 0:
                short_armed = False
                short_armed_bars = 0
                short_break_level = math.nan
                short_invalidation = math.nan
                short_pb_high = high[i]
                short_pb_low = low[i]
                short_pb_bull_bars = 1 if close[i] > open_[i] else 0
            else:
                short_pb_high = max(short_pb_high, high[i])
                short_pb_low = min(short_pb_low, low[i])
                if close[i] > open_[i]:
                    short_pb_bull_bars += 1
            short_pb_bars += 1
        elif short_pb_bars > 0:
            if atr[i] > 0 and not math.isnan(short_pb_high):
                last_short_pullback_depth_atr = (short_pb_high - ema_fast[i]) / atr[i]
            else:
                last_short_pullback_depth_atr = math.nan

            valid = (
                short_pb_bars >= thresholds.min_pullback_bars
                and short_pb_bull_bars >= thresholds.min_counter_bars
                and not math.isnan(last_short_pullback_depth_atr)
                and last_short_pullback_depth_atr <= thresholds.max_pullback_depth_atr
                and short_pb_high < ema_slow[i] + atr[i] * thresholds.pullback_ema50_tolerance_atr
            )
            if valid:
                short_armed = True
                short_armed_bars = 0
                short_break_level = short_pb_low
                short_invalidation = short_pb_high
            short_pb_bars = 0
            short_pb_bull_bars = 0
            short_pb_high = math.nan
            short_pb_low = math.nan

        # Armed-Zähler
        if long_armed:
            long_armed_bars += 1
        if short_armed:
            short_armed_bars += 1

        # Armed-Invalidierungen
        if long_armed and (
            (not long_bias[i]) or close[i] < ema_slow[i]
            or (not math.isnan(long_invalidation) and close[i] < long_invalidation)
            or long_armed_bars > thresholds.max_armed_bars
        ):
            long_armed = False
            long_armed_bars = 0
            long_break_level = math.nan
            long_invalidation = math.nan

        if short_armed and (
            (not short_bias[i]) or close[i] > ema_slow[i]
            or (not math.isnan(short_invalidation) and close[i] > short_invalidation)
            or short_armed_bars > thresholds.max_armed_bars
        ):
            short_armed = False
            short_armed_bars = 0
            short_break_level = math.nan
            short_invalidation = math.nan

        # Confirmation-Filter
        if params.confirmation_mode == "Keiner":
            conf_long_ok = True
            conf_short_ok = True
        elif params.confirmation_mode == "Volumen":
            # HINWEIS: market_data_1min hat kein volume -> tick_count als Proxy
            va = vol_avg[i]
            conf_long_ok = (not math.isnan(va)) and vol_proxy[i] > va
            conf_short_ok = conf_long_ok
        elif params.confirmation_mode == "Effizienz":
            ce = close_eff[i]
            conf_long_ok = (not math.isnan(ce)) and close[i] > ce and efficiency[i] >= thresholds.efficiency_min
            conf_short_ok = (not math.isnan(ce)) and close[i] < ce and efficiency[i] >= thresholds.efficiency_min
        else:  # Break-and-Hold
            if long_armed and not math.isnan(long_break_level) and atr[i] > 0:
                dist = (close[i] - long_break_level) / atr[i]
                conf_long_ok = close[i] > long_break_level and dist >= thresholds.break_hold_atr_min
            else:
                conf_long_ok = False
            if short_armed and not math.isnan(short_break_level) and atr[i] > 0:
                dist = (short_break_level - close[i]) / atr[i]
                conf_short_ok = close[i] < short_break_level and dist >= thresholds.break_hold_atr_min
            else:
                conf_short_ok = False

        # RSI-Filter
        if thresholds.use_rsi_filter:
            rsi_long_ok = (not math.isnan(rsi[i])) and rsi[i] > thresholds.long_rsi_threshold
            rsi_short_ok = (not math.isnan(rsi[i])) and rsi[i] < thresholds.short_rsi_threshold
        else:
            rsi_long_ok = True
            rsi_short_ok = True

        # Trigger-Bar (Ausbruch über longBreakLevel + Buffer, bzw. unter shortBreakLevel - Buffer)
        trig_long = (
            long_armed and not math.isnan(long_break_level)
            and high[i] > long_break_level + atr[i] * thresholds.break_buffer_atr
            and close[i] > ema_fast[i]
            and (not math.isnan(impulse_avg[i])) and bar_range[i] >= impulse_avg[i] * thresholds.trigger_range_factor
        )
        trig_short = (
            short_armed and not math.isnan(short_break_level)
            and low[i] < short_break_level - atr[i] * thresholds.break_buffer_atr
            and close[i] < ema_fast[i]
            and (not math.isnan(impulse_avg[i])) and bar_range[i] >= impulse_avg[i] * thresholds.trigger_range_factor
        )

        long_entry = trig_long and conf_long_ok and rsi_long_ok and not overext_long[i]
        short_entry = trig_short and conf_short_ok and rsi_short_ok and not overext_short[i]

        # Entry -> Armed zurücksetzen
        if long_entry:
            long_armed = False
            long_armed_bars = 0
            long_break_level = math.nan
            long_invalidation = math.nan

        if short_entry:
            short_armed = False
            short_armed_bars = 0
            short_break_level = math.nan
            short_invalidation = math.nan

        long_entry_arr[i] = long_entry
        short_entry_arr[i] = short_entry
        long_armed_arr[i] = long_armed
        short_armed_arr[i] = short_armed

    out["long_entry"] = long_entry_arr
    out["short_entry"] = short_entry_arr
    out["long_armed"] = long_armed_arr
    out["short_armed"] = short_armed_arr
    return out


def compute_indicator_for_symbol(df_symbol: pd.DataFrame, params: Params, session_timezone: str) -> pd.DataFrame:
    thresholds = thresholds_for(params)
    base = prepare_base(df_symbol, params, session_timezone)
    regime = apply_regime(base, params, thresholds)
    final = apply_signal_state_machine(regime, params, thresholds)
    return final


# =========================================================================
# DB
# =========================================================================

def _make_engine(host: str, port: int, user: str, password: str, dbname: str) -> Engine:
    from urllib.parse import quote_plus
    url = (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(dbname)}"
    )
    return create_engine(url, future=True)


def fetch_market_data(engine: Engine, table_name: str, start_ts: datetime,
                      warmup_bars: int, symbols: list[str] | None) -> pd.DataFrame:
    from sqlalchemy import text
    # 1-Minuten-Bars: warmup_bars in Minuten zurückgehen
    fetch_from = start_ts - timedelta(minutes=warmup_bars)
    symbol_filter = "AND symbol = ANY(:symbols)" if symbols else ""
    sql = text(
        f"""
        SELECT symbol, bar_time, tick_count, open, high, low, close
        FROM {table_name}
        WHERE bar_time >= :fetch_from
        {symbol_filter}
        ORDER BY symbol, bar_time
        """
    )
    query_params: dict[str, object] = {"fetch_from": fetch_from}
    if symbols:
        query_params["symbols"] = symbols
    df = pd.read_sql_query(sql, engine, params=query_params, parse_dates=["bar_time"])
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    for col in ["tick_count", "open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "bar_time", "high", "low", "close"]).copy()
    return df


def build_output_rows(df_all: pd.DataFrame, start_ts: datetime, timeframe: str) -> pd.DataFrame:
    """Nur Bars ab start_ts, und nur solche, auf denen long_entry oder short_entry true sind."""
    mask = (df_all["bar_time"] >= start_ts) & (df_all["long_entry"] | df_all["short_entry"])
    hits = df_all.loc[mask].copy()
    if hits.empty:
        return hits

    now_utc = datetime.now(timezone.utc)
    hits["received_at"] = now_utc
    # Bar-Schluss: bar_time ist i.d.R. Bar-Open. 1min -> +1 Minute = Close-Zeitpunkt
    hits["event_time"] = hits["bar_time"] + pd.Timedelta(minutes=1)
    hits["ticker"] = hits["symbol"].map(mapped_ticker)
    hits["exchange"] = ""
    hits["timeframe"] = timeframe
    hits["action"] = np.where(hits["long_entry"], "buy", "sell")
    # Preis mit numeric(20,2)-kompatibler Rundung
    hits["price"] = hits["close"].round(2)
    hits["indicator"] = INDICATOR_NAME
    hits["remoteip"] = ""

    cols = ["received_at", "event_time", "ticker", "exchange", "timeframe",
            "action", "price", "indicator", "remoteip"]
    return hits[cols].sort_values(["ticker", "event_time"]).reset_index(drop=True)


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
            received_at, event_time, ticker, exchange, timeframe,
            action, price, indicator, remoteip
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    records = [
        (
            row.received_at.to_pydatetime() if hasattr(row.received_at, "to_pydatetime") else row.received_at,
            row.event_time.to_pydatetime() if hasattr(row.event_time, "to_pydatetime") else row.event_time,
            row.ticker,
            row.exchange,
            row.timeframe,
            row.action,
            float(row.price),
            row.indicator,
            row.remoteip,
        )
        for row in df_out.itertuples(index=False)
    ]
    with conn.cursor() as cur:
        cur.executemany(sql, records)
    return len(records)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    start_ts = parse_start_ts(args.start_date)
    params = Params(
        profile=args.profile,
        entry_mode=args.entry_mode,
        use_vwap=not args.disable_vwap,
        use_auto_rsi=not args.disable_auto_rsi,
        manual_rsi_filter=args.manual_rsi_filter,
        confirmation_mode=args.confirmation_mode,
    )

    source_engine = _make_engine(args.host, args.port, args.user, args.password, args.source_db)
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
        calc = compute_indicator_for_symbol(df_symbol, params, args.session_timezone)
        calc["symbol"] = symbol
        results.append(calc)

    full_df = pd.concat(results, ignore_index=True)
    output_df = build_output_rows(full_df, start_ts, args.timeframe)

    if output_df.empty:
        print("Keine Entry-Signale im angegebenen Zeitraum gefunden.")
        # Trotzdem eventuelle bestehende Signale ab start_ts für die bearbeiteten Ticker löschen
        processed_tickers = sorted(full_df["symbol"].dropna().unique().tolist())
        processed_tickers_mapped = sorted({mapped_ticker(t) for t in processed_tickers})
        target_engine = _make_engine(args.host, args.port, args.user, args.password, args.target_db)
        try:
            raw_conn = target_engine.raw_connection()
            try:
                delete_existing_rows(
                    conn=raw_conn,
                    table_name=args.target_table,
                    start_ts=start_ts,
                    timeframe=args.timeframe,
                    tickers=processed_tickers_mapped,
                )
                raw_conn.commit()
            except Exception:
                raw_conn.rollback()
                raise
            finally:
                raw_conn.close()
        finally:
            target_engine.dispose()
        print(f"Alte Signale ab {start_ts.isoformat()} für Ticker {processed_tickers_mapped} entfernt.")
        return

    target_engine = _make_engine(args.host, args.port, args.user, args.password, args.target_db)
    try:
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

    buys = int((output_df["action"] == "buy").sum())
    sells = int((output_df["action"] == "sell").sum())
    print(
        f"Fertig. {inserted} Signale nach {args.target_table} geschrieben "
        f"({buys} buy, {sells} sell) ab {start_ts.isoformat()} "
        f"für {output_df['ticker'].nunique()} Ticker ({', '.join(tickers_to_refresh)})."
    )


if __name__ == "__main__":
    main()
