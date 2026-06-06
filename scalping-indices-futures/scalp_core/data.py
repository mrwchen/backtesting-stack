"""Load OHLCV bars from public.ibkr_market_data and compute causal base features.

Everything here is point-in-time safe: each feature at bar t uses only information
available up to (and including) the close of bar t. The prediction target is the
sign of the *next* bar's return, so labels are shifted by -1.
"""

import logging
from datetime import date, datetime, time, timezone
from typing import Optional

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from zoneinfo import ZoneInfo

from . import config

log = logging.getLogger(__name__)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range (Wilder smoothing), in price points."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def load_bars(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    """Load ordered OHLCV bars for the configured symbol/bar_size/date range."""
    where = [sql.SQL("symbol = %s"), sql.SQL("bar_size = %s"), sql.SQL("close IS NOT NULL")]
    params: list[object] = [config.SYMBOL, config.BAR_SIZE]
    if config.START_DATE is not None:
        where.append(sql.SQL("ts >= %s"))
        params.append(datetime.combine(config.START_DATE, time.min, tzinfo=timezone.utc))
    if config.END_DATE is not None:
        where.append(sql.SQL("ts < %s"))
        params.append(datetime.combine(config.END_DATE, time.min, tzinfo=timezone.utc))

    query = sql.SQL(
        "SELECT ts, open, high, low, close, volume FROM {tbl} WHERE {where} ORDER BY ts"
    ).format(
        tbl=sql.Identifier(*config.SOURCE_TABLE.split(".")) if "." in config.SOURCE_TABLE
        else sql.Identifier(config.SOURCE_TABLE),
        where=sql.SQL(" AND ").join(where),
    )

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No bars found in {config.SOURCE_TABLE} for symbol={config.SYMBOL!r} "
            f"bar_size={config.BAR_SIZE!r} (start={config.START_DATE} end={config.END_DATE})"
        )

    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df = df.sort_values("ts").reset_index(drop=True)
    log.info(
        "Loaded bars %d for %s %s from %s to %s",
        len(df), config.SYMBOL, config.BAR_SIZE, df["ts"].iloc[0], df["ts"].iloc[-1],
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal feature columns and the next-bar direction label.

    Columns added:
      log_ret        log return of close vs previous close
      abs_ret        |log_ret|
      roll_vol       rolling std of log_ret (short window)
      momentum       cumulative log return over MOMENTUM_BARS
      rsi            RSI_PERIOD RSI
      atr            ATR_BARS Average True Range (price points; for atr-based stops)
      session_date   local session day (for intraday flat-at-cutoff)
      local_time     local wall-clock time of the bar
      target_up      1 if next bar's log return > 0 else 0  (label, shifted -1)
    """
    out = df.copy()
    close = out["close"]

    out["log_ret"] = np.log(close).diff()
    out["abs_ret"] = out["log_ret"].abs()
    out["roll_vol"] = out["log_ret"].rolling(config.ROLL_VOL_BARS, min_periods=max(2, config.ROLL_VOL_BARS // 4)).std()
    out["momentum"] = np.log(close).diff(config.MOMENTUM_BARS)
    out["rsi"] = _rsi(close, config.RSI_PERIOD)
    out["atr"] = _atr(out["high"], out["low"], close, config.ATR_BARS)

    tz = ZoneInfo(config.SESSION_TZ)
    local = out["ts"].dt.tz_convert(tz)
    out["session_date"] = local.dt.date
    out["local_time"] = local.dt.time

    # Label: did the next bar close higher than this one? (causal target)
    out["target_up"] = (out["log_ret"].shift(-1) > 0.0).astype(float)

    out = out.replace([np.inf, -np.inf], np.nan)
    out["log_ret"] = out["log_ret"].fillna(0.0)
    out["abs_ret"] = out["abs_ret"].fillna(0.0)
    out["roll_vol"] = out["roll_vol"].fillna(out["roll_vol"].median())
    out["momentum"] = out["momentum"].fillna(0.0)
    out["atr"] = out["atr"].fillna(out["atr"].median())
    return out


def session_flat_cutoff() -> time:
    hh, mm = config.SESSION_FLAT_TIME.split(":")
    return time(int(hh), int(mm))


def session_entry_start() -> time:
    hh, mm = config.ENTRY_START_TIME.split(":")
    return time(int(hh), int(mm))


def session_entry_end() -> time:
    hh, mm = config.ENTRY_END_TIME.split(":")
    return time(int(hh), int(mm))
