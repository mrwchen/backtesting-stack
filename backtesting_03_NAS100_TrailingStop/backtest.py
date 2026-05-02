#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)
log = logging.getLogger(__name__)


@dataclass
class Config:
    initial_equity: float
    symbol_name: str
    timeframe: str
    required_margin: float
    spread_points: float
    position_fraction_of_max_margin: float
    risk_fraction_of_used_margin: float
    lot_step: float
    bt_db_name: str
    market_db_name: str
    output_dir: Path
    start_time_utc: Optional[pd.Timestamp]
    end_time_utc: Optional[pd.Timestamp]
    account_number: str
    account_type: str
    regime_validity_minutes: float
    atr_period: int
    atr_multiplier: float
    pivot_buffer_atr_multiplier: float
    trailing_buffer_atr_multiplier: float
    trailing_buffer_atr_multiplier_late: float
    max_bars_held: int
    trailing_activation_rr: float
    trailing_late_minutes: int
    breakeven_minutes: int
    breakeven_offset_points: float
    use_dynamic_spread: bool
    entry_start_weekday: int   # 0=Mon, 4=Fri
    entry_start_hour_utc: int
    entry_end_weekday: int
    entry_end_hour_utc: int
    weekly_close_weekday: int
    weekly_close_hour_utc: int


def env_str(name: str, default: Optional[str] = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw is not None else float(default)


def env_optional_timestamp(name: str) -> Optional[pd.Timestamp]:
    raw = os.getenv(name)
    if not raw:
        return None
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def load_config() -> Config:
    log.debug("Loading config from environment variables")
    cfg = Config(
        initial_equity=env_float("INITIAL_EQUITY", 1000.0),
        symbol_name=env_str("SYMBOL_NAME", "NAS100"),
        timeframe=env_str("TIMEFRAME", "1min"),
        required_margin=env_float("REQUIRED_MARGIN", 0.05),
        spread_points=env_float("SPREAD_POINTS", 1.5),
        position_fraction_of_max_margin=env_float("POSITION_FRACTION_OF_MAX_MARGIN", 0.45),
        risk_fraction_of_used_margin=env_float("RISK_FRACTION_OF_USED_MARGIN", 0.02),
        lot_step=env_float("LOT_STEP", 0.1),
        bt_db_name=env_str("BT_DB_NAME", "backtesting"),
        market_db_name=env_str("MARKET_DB_NAME", "market-data"),
        output_dir=Path(env_str("OUTPUT_DIR", "/app/output")),
        start_time_utc=env_optional_timestamp("START_TIME_UTC"),
        end_time_utc=env_optional_timestamp("END_TIME_UTC"),
        account_number=env_str("ACCOUNT_NUMBER", "00001"),
        account_type=env_str("ACCOUNT_TYPE", "backtester"),
        regime_validity_minutes=env_float("REGIME_VALIDITY_MINUTES", 5.0),
        atr_period=int(env_float("ATR_PERIOD", 14)),
        atr_multiplier=env_float("ATR_MULTIPLIER", 1.5),
        pivot_buffer_atr_multiplier=env_float("PIVOT_BUFFER_ATR_MULTIPLIER", 0.1),
        trailing_buffer_atr_multiplier=env_float("TRAILING_BUFFER_ATR_MULTIPLIER", 0.1),
        trailing_buffer_atr_multiplier_late=env_float("TRAILING_BUFFER_ATR_MULTIPLIER_LATE", 0.05),
        max_bars_held=int(env_float("MAX_BARS_HELD", 120)),
        trailing_activation_rr=env_float("TRAILING_ACTIVATION_RR", 2.0),
        trailing_late_minutes=int(env_float("TRAILING_LATE_MINUTES", 20)),
        breakeven_minutes=int(env_float("BREAKEVEN_MINUTES", 20)),
        breakeven_offset_points=env_float("BREAKEVEN_OFFSET_POINTS", 5.0),
        use_dynamic_spread=os.getenv("USE_DYNAMIC_SPREAD", "true").lower() in ("1", "true", "yes"),
        entry_start_weekday=int(env_float("ENTRY_START_WEEKDAY", 0)),
        entry_start_hour_utc=int(env_float("ENTRY_START_HOUR_UTC", 7)),
        entry_end_weekday=int(env_float("ENTRY_END_WEEKDAY", 4)),
        entry_end_hour_utc=int(env_float("ENTRY_END_HOUR_UTC", 19)),
        weekly_close_weekday=int(env_float("WEEKLY_CLOSE_WEEKDAY", 4)),
        weekly_close_hour_utc=int(env_float("WEEKLY_CLOSE_HOUR_UTC", 20)),
    )

    if cfg.initial_equity <= 0:
        raise ValueError("INITIAL_EQUITY must be > 0")
    if not (0 < cfg.required_margin <= 1):
        raise ValueError("REQUIRED_MARGIN must be in (0, 1]")
    if cfg.spread_points < 0:
        raise ValueError("SPREAD_POINTS must be >= 0")
    if not (0 < cfg.position_fraction_of_max_margin <= 1):
        raise ValueError("POSITION_FRACTION_OF_MAX_MARGIN must be in (0, 1]")
    if cfg.risk_fraction_of_used_margin <= 0:
        raise ValueError("RISK_FRACTION_OF_USED_MARGIN must be > 0")
    if cfg.lot_step <= 0:
        raise ValueError("LOT_STEP must be > 0")
    if cfg.start_time_utc and cfg.end_time_utc and cfg.start_time_utc > cfg.end_time_utc:
        raise ValueError("START_TIME_UTC must be <= END_TIME_UTC")

    log.info(
        "Config loaded: symbol=%s timeframe=%s equity=%.2f margin=%.4f spread=%.2f "
        "pos_frac=%.4f risk_frac=%.4f lot_step=%.2f regime_validity_min=%.1f "
        "atr_period=%d atr_multiplier=%.2f pivot_buffer=%.2f trailing_buffer=%.2f trailing_buffer_late=%.2f "
        "max_bars_held=%d trailing_activation_rr=%.2f trailing_late_min=%d breakeven_min=%d "
        "use_dynamic_spread=%s entry_window=weekday%d-%02d:00 to weekday%d-%02d:00 "
        "weekly_close=weekday%d-%02d:00 start=%s end=%s",
        cfg.symbol_name, cfg.timeframe, cfg.initial_equity, cfg.required_margin,
        cfg.spread_points, cfg.position_fraction_of_max_margin,
        cfg.risk_fraction_of_used_margin, cfg.lot_step,
        cfg.regime_validity_minutes, cfg.atr_period, cfg.atr_multiplier,
        cfg.pivot_buffer_atr_multiplier, cfg.trailing_buffer_atr_multiplier,
        cfg.trailing_buffer_atr_multiplier_late,
        cfg.max_bars_held, cfg.trailing_activation_rr,
        cfg.trailing_late_minutes, cfg.breakeven_minutes,
        cfg.use_dynamic_spread,
        cfg.entry_start_weekday, cfg.entry_start_hour_utc,
        cfg.entry_end_weekday, cfg.entry_end_hour_utc,
        cfg.weekly_close_weekday, cfg.weekly_close_hour_utc,
        cfg.start_time_utc, cfg.end_time_utc,
    )
    return cfg


def build_connection_kwargs(prefix: str, dbname: str) -> Dict[str, Any]:
    dsn = os.getenv(f"{prefix}_DB_DSN")
    if dsn:
        log.debug("Using DSN for %s DB connection", prefix)
        return {"dsn": dsn}

    host = env_str(f"{prefix}_DB_HOST", "host.docker.internal")
    port = int(env_str(f"{prefix}_DB_PORT", "5432"))
    log.debug("Using host/port for %s DB connection: %s:%s/%s", prefix, host, port, dbname)
    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": env_str(f"{prefix}_DB_USER"),
        "password": env_str(f"{prefix}_DB_PASSWORD"),
        "sslmode": os.getenv(f"{prefix}_DB_SSLMODE", "prefer"),
    }


def connect_postgres(prefix: str, dbname: str):
    kwargs = build_connection_kwargs(prefix, dbname)
    log.debug("Connecting to Postgres: prefix=%s dbname=%s", prefix, dbname)
    conn = psycopg2.connect(**kwargs)
    log.debug("Connected to Postgres: prefix=%s dbname=%s", prefix, dbname)
    return conn


def normalize_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True)


def floor_to_step(value: float, step: float) -> float:
    if value <= 0:
        return 0.0
    floored = math.floor((value / step) + 1e-12) * step
    return round(floored, 10)


# ---------------------------------------------------------------------------
# ATR & session helpers
# ---------------------------------------------------------------------------

def compute_atr_series(bars_df: pd.DataFrame, period: int = 14) -> pd.Series:
    """True Range ATR over a rolling window on the bars DataFrame."""
    high = bars_df["high"].astype(float)
    low = bars_df["low"].astype(float)
    prev_close = bars_df["close"].astype(float).shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period, min_periods=1).mean()


def get_dynamic_spread(bar_time: pd.Timestamp) -> float:
    """
    Return a session-aware spread estimate (in points) for NAS100 CFD.

    US session  (15:30–22:00 UTC): tight market, ~1.5 pts
    Europe      (07:00–15:30 UTC): moderate liquidity, ~4.0 pts
    Night       (22:00–07:00 UTC): thin market, ~8.0 pts
    """
    hour = bar_time.hour
    minute = bar_time.minute
    # US session: 15:30–22:00
    if (hour == 15 and minute >= 30) or (16 <= hour < 22):
        return 1.5
    # Europe: 07:00–15:30
    elif 7 <= hour < 16:
        return 4.0
    # Night: 22:00–07:00
    else:
        return 8.0


def session_label(bar_time: pd.Timestamp) -> str:
    hour = bar_time.hour
    minute = bar_time.minute
    if (hour == 15 and minute >= 30) or (16 <= hour < 22):
        return "us"
    elif 7 <= hour < 16:
        return "europe"
    else:
        return "night"


def entry_allowed(bar_time: pd.Timestamp, cfg) -> bool:
    """
    Returns True if a new entry is allowed at bar_time.
    Entry window: Mon 07:00 UTC (inclusive) to Fri 19:00 UTC (exclusive).
    Weekday: 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri
    """
    weekday = bar_time.weekday()
    hour = bar_time.hour

    # Before entry_start: e.g. Mon before 07:00
    if weekday == cfg.entry_start_weekday and hour < cfg.entry_start_hour_utc:
        return False
    # Weekend days before entry_start_weekday (Sun=6)
    if weekday < cfg.entry_start_weekday or weekday == 6:
        return False
    # After entry_end: e.g. Fri from 19:00 onwards
    if weekday == cfg.entry_end_weekday and hour >= cfg.entry_end_hour_utc:
        return False
    # Days after entry_end_weekday (Sat=5, Sun=6)
    if weekday > cfg.entry_end_weekday:
        return False
    return True


def weekly_close_due(bar_time: pd.Timestamp, cfg) -> bool:
    """Returns True if open positions must be closed at this bar (e.g. Fri 20:00 UTC)."""
    return (
        bar_time.weekday() == cfg.weekly_close_weekday
        and bar_time.hour >= cfg.weekly_close_hour_utc
    )


def regime_allows(direction: str, regime: Optional[Dict[str, Any]]) -> bool:
    if regime is None:
        return False
    if direction == "long":
        return bool(regime["is_strong_long"] or regime["is_weak_long"])
    if direction == "short":
        return bool(regime["is_strong_short"] or regime["is_weak_short"])
    raise ValueError(f"Unsupported direction: {direction}")


def regime_label(regime: Optional[Dict[str, Any]]) -> str:
    if regime is None:
        return "unknown"
    long_ok = bool(regime["is_strong_long"] or regime["is_weak_long"])
    short_ok = bool(regime["is_strong_short"] or regime["is_weak_short"])
    neutral_ok = bool(regime.get("is_neutral", False))

    if long_ok and not short_ok:
        return "long"
    if short_ok and not long_ok:
        return "short"
    if neutral_ok and not long_ok and not short_ok:
        return "neutral"
    if long_ok and short_ok:
        return "mixed"
    return "neutral"


def fetch_signals(conn, cfg: Config) -> pd.DataFrame:
    log.info("Fetching signals: symbol=%s timeframe=%s start=%s end=%s",
             cfg.symbol_name, cfg.timeframe, cfg.start_time_utc, cfg.end_time_utc)
    sql = """
        SELECT
            id,
            event_time,
            ticker,
            timeframe,
            LOWER(action) AS action,
            price,
            indicator
        FROM public.bt_signal
        WHERE ticker = %(symbol)s
          AND timeframe = %(timeframe)s
          AND event_time IS NOT NULL
          AND LOWER(action) IN ('buy', 'sell')
          AND (%(start_time)s IS NULL OR event_time >= %(start_time)s)
          AND (%(end_time)s IS NULL OR event_time <= %(end_time)s)
        ORDER BY event_time, id
    """
    df = pd.read_sql_query(
        sql,
        conn,
        params={
            "symbol": cfg.symbol_name,
            "timeframe": cfg.timeframe,
            "start_time": cfg.start_time_utc,
            "end_time": cfg.end_time_utc,
        },
    )
    if df.empty:
        log.warning("No signals found for symbol=%s timeframe=%s", cfg.symbol_name, cfg.timeframe)
        return df
    df["event_time"] = normalize_ts(df["event_time"])
    buy_count = int((df["action"] == "buy").sum())
    sell_count = int((df["action"] == "sell").sum())
    log.info(
        "Fetched %d signals (buy=%d sell=%d) from %s to %s",
        len(df), buy_count, sell_count,
        df["event_time"].min(), df["event_time"].max(),
    )
    return df


REGIME_TIMEFRAME = "5min"


def fetch_regimes(conn, cfg: Config) -> pd.DataFrame:
    log.info("Fetching regimes: symbol=%s timeframe=%s (fixed) start=%s end=%s",
             cfg.symbol_name, REGIME_TIMEFRAME, cfg.start_time_utc, cfg.end_time_utc)
    common_params = {
        "symbol": cfg.symbol_name,
        "timeframe": REGIME_TIMEFRAME,
        "start_time": cfg.start_time_utc,
        "end_time": cfg.end_time_utc,
    }

    if cfg.start_time_utc is None:
        sql = """
            SELECT
                id,
                event_time,
                ticker,
                timeframe,
                is_strong_long,
                is_weak_long,
                is_strong_short,
                is_weak_short,
                is_neutral,
                trend_text,
                entry_text
            FROM public.bt_regime
            WHERE ticker = %(symbol)s
              AND timeframe = %(timeframe)s
              AND (%(end_time)s IS NULL OR event_time <= %(end_time)s)
            ORDER BY event_time, id
        """
        df = pd.read_sql_query(sql, conn, params=common_params)
    else:
        log.debug("Fetching pre-start regime (lookback) for start=%s", cfg.start_time_utc)
        sql_pre = """
            SELECT
                id,
                event_time,
                ticker,
                timeframe,
                is_strong_long,
                is_weak_long,
                is_strong_short,
                is_weak_short,
                is_neutral,
                trend_text,
                entry_text
            FROM public.bt_regime
            WHERE ticker = %(symbol)s
              AND timeframe = %(timeframe)s
              AND event_time < %(start_time)s
            ORDER BY event_time DESC, id DESC
            LIMIT 1
        """
        sql_post = """
            SELECT
                id,
                event_time,
                ticker,
                timeframe,
                is_strong_long,
                is_weak_long,
                is_strong_short,
                is_weak_short,
                is_neutral,
                trend_text,
                entry_text
            FROM public.bt_regime
            WHERE ticker = %(symbol)s
              AND timeframe = %(timeframe)s
              AND event_time >= %(start_time)s
              AND (%(end_time)s IS NULL OR event_time <= %(end_time)s)
            ORDER BY event_time, id
        """
        df_pre = pd.read_sql_query(sql_pre, conn, params=common_params)
        df_post = pd.read_sql_query(sql_post, conn, params=common_params)
        log.debug("Regime lookback row count: %d", len(df_pre))
        log.debug("Regime in-range row count: %d", len(df_post))
        df = pd.concat([df_pre, df_post], ignore_index=True)

    if df.empty:
        log.warning("No regimes found for symbol=%s timeframe=%s", cfg.symbol_name, cfg.timeframe)
        return df
    df = df.drop_duplicates(subset=["id"]).sort_values(["event_time", "id"]).reset_index(drop=True)
    df["event_time"] = normalize_ts(df["event_time"])
    log.info(
        "Fetched %d regime rows from %s to %s",
        len(df), df["event_time"].min(), df["event_time"].max(),
    )
    return df


def fetch_bars(conn, cfg: Config, bars_start_time: pd.Timestamp) -> pd.DataFrame:
    log.info("Fetching bars: symbol=%s start=%s end=%s",
             cfg.symbol_name, bars_start_time, cfg.end_time_utc)
    sql = """
        SELECT
            bar_time,
            open,
            high,
            low,
            close,
            tick_count
        FROM public.market_data_1min
        WHERE symbol = %(symbol)s
          AND bar_time >= %(start_time)s
          AND (%(end_time)s IS NULL OR bar_time <= %(end_time)s)
        ORDER BY bar_time
    """
    df = pd.read_sql_query(
        sql,
        conn,
        params={
            "symbol": cfg.symbol_name,
            "start_time": bars_start_time,
            "end_time": cfg.end_time_utc,
        },
    )
    if df.empty:
        log.error("No bars found for symbol=%s start=%s end=%s",
                  cfg.symbol_name, bars_start_time, cfg.end_time_utc)
        return df
    df["bar_time"] = normalize_ts(df["bar_time"])
    log.info(
        "Fetched %d bars from %s to %s",
        len(df), df["bar_time"].min(), df["bar_time"].max(),
    )
    return df


def fetch_pivots(conn, cfg: Config, bars_start_time: pd.Timestamp) -> pd.DataFrame:
    log.info("Fetching pivots: symbol=%s start=%s end=%s",
             cfg.symbol_name, bars_start_time, cfg.end_time_utc)
    sql = """
        SELECT
            id,
            ticker,
            pivot_price,
            pivot_direction,
            confirmation_bar_time
        FROM public.bt_high_low
        WHERE ticker = %(symbol)s
          AND confirmation_bar_time >= %(start_time)s
          AND (%(end_time)s IS NULL OR confirmation_bar_time <= %(end_time)s)
        ORDER BY confirmation_bar_time, id
    """
    df = pd.read_sql_query(
        sql,
        conn,
        params={
            "symbol": cfg.symbol_name,
            "start_time": bars_start_time,
            "end_time": cfg.end_time_utc,
        },
    )
    if df.empty:
        log.warning("No pivots found for symbol=%s", cfg.symbol_name)
        return df
    df["confirmation_bar_time"] = normalize_ts(df["confirmation_bar_time"])
    log.info(
        "Fetched %d pivots (highs=%d lows=%d) from %s to %s",
        len(df),
        int((df["pivot_direction"] == 1).sum()),
        int((df["pivot_direction"] == -1).sum()),
        df["confirmation_bar_time"].min(),
        df["confirmation_bar_time"].max(),
    )
    return df


def get_last_pivot_price(
    pivots: pd.DataFrame,
    bar_time: pd.Timestamp,
    direction: int,  # +1 = high, -1 = low
) -> Optional[float]:
    """Return the pivot_price of the last confirmed pivot of the given direction up to bar_time."""
    mask = (pivots["pivot_direction"] == direction) & (pivots["confirmation_bar_time"] <= bar_time)
    filtered = pivots.loc[mask]
    if filtered.empty:
        return None
    return float(filtered.iloc[-1]["pivot_price"])


def build_event_map(df: pd.DataFrame, bar_times: pd.Series, time_col: str) -> Dict[int, pd.DataFrame]:
    if df.empty:
        return {}
    event_idx = bar_times.searchsorted(df[time_col], side="left")
    mapped = df.copy()
    mapped["exec_bar_idx"] = event_idx
    mapped = mapped[mapped["exec_bar_idx"] < len(bar_times)].copy()
    event_map: Dict[int, pd.DataFrame] = {}
    if mapped.empty:
        return event_map
    for exec_idx, grp in mapped.groupby("exec_bar_idx", sort=True):
        event_map[int(exec_idx)] = grp.sort_values([time_col, "id"]).reset_index(drop=True)
    log.debug("Built event_map with %d distinct bar indices (col=%s)", len(event_map), time_col)
    return event_map


def build_regime_state(row: pd.Series) -> Dict[str, Any]:
    return {
        "id": int(row["id"]),
        "event_time": row["event_time"],
        "ticker": row["ticker"],
        "timeframe": row["timeframe"],
        "is_strong_long": bool(row["is_strong_long"]),
        "is_weak_long": bool(row["is_weak_long"]),
        "is_strong_short": bool(row["is_strong_short"]),
        "is_weak_short": bool(row["is_weak_short"]),
        "is_neutral": bool(row["is_neutral"]),
        "trend_text": row.get("trend_text"),
        "entry_text": row.get("entry_text"),
    }


def close_position(
    *,
    position: Dict[str, Any],
    exit_reason: str,
    exit_bar_time: pd.Timestamp,
    exit_event_time: pd.Timestamp,
    exit_mid_price: float,
    exit_price: float,
    equity: float,
    spread_half: float,
) -> Dict[str, Any]:
    if position["direction"] == "long":
        pnl_usd = (exit_price - position["entry_price"]) * position["volume"]
        points = exit_price - position["entry_price"]
    else:
        pnl_usd = (position["entry_price"] - exit_price) * position["volume"]
        points = position["entry_price"] - exit_price

    equity_after = equity + pnl_usd
    holding_minutes = (exit_bar_time - position["entry_bar_time"]).total_seconds() / 60.0

    log.debug(
        "CLOSE | signal_id=%s dir=%s reason=%s entry=%.4f exit=%.4f vol=%.2f "
        "points=%.4f pnl=%.2f equity_after=%.2f held_min=%.1f",
        position["signal_id"], position["direction"], exit_reason,
        position["entry_price"], exit_price, position["volume"],
        points, pnl_usd, equity_after, holding_minutes,
    )

    return {
        "signal_id": position["signal_id"],
        "signal_time": position["signal_time"],
        "symbol": position["symbol"],
        "timeframe": position["timeframe"],
        "direction": position["direction"],
        "entry_bar_time": position["entry_bar_time"],
        "entry_mid_price": position["entry_mid_price"],
        "entry_price": position["entry_price"],
        "volume": position["volume"],
        "notional_usd": position["notional_usd"],
        "margin_used_usd": position["margin_used_usd"],
        "risk_budget_usd": position["risk_budget_usd"],
        "initial_stop_price": position.get("initial_stop_price", position["stop_price"]),
        "stop_price": position["stop_price"],
        "trailing_sl_active": position.get("trailing_sl_active", False),
        "entry_regime_time": position["entry_regime_time"],
        "entry_regime_label": position["entry_regime_label"],
        "entry_session": position.get("entry_session", "unknown"),
        "entry_atr": position.get("entry_atr", float("nan")),
        "stop_distance_points": position.get("stop_distance_points", float("nan")),
        "exit_reason": exit_reason,
        "exit_event_time": exit_event_time,
        "exit_bar_time": exit_bar_time,
        "exit_mid_price": exit_mid_price,
        "exit_price": exit_price,
        "spread_half_points": spread_half,
        "points_realized": points,
        "pnl_usd": pnl_usd,
        "return_on_margin": pnl_usd / position["margin_used_usd"] if position["margin_used_usd"] else 0.0,
        "equity_before_trade": position["equity_before_trade"],
        "equity_after_trade": equity_after,
        "holding_minutes": holding_minutes,
        "bars_held_inclusive": int(position["bars_held_counter"]),
    }


def build_summary(
    cfg: Config,
    signals: pd.DataFrame,
    regimes: pd.DataFrame,
    bars: pd.DataFrame,
    trades: pd.DataFrame,
    skipped_signals: Dict[str, int],
) -> Dict[str, Any]:
    total_trades = int(len(trades))
    final_equity = float(trades["equity_after_trade"].iloc[-1]) if total_trades else cfg.initial_equity

    if total_trades:
        wins = int((trades["pnl_usd"] > 0).sum())
        losses = int((trades["pnl_usd"] < 0).sum())
        breakeven = int((trades["pnl_usd"] == 0).sum())
        gross_profit = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].sum())
        gross_loss_abs = float(-trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].sum())
        net_profit = float(trades["pnl_usd"].sum())
        win_rate = wins / total_trades if total_trades else 0.0
        average_trade = net_profit / total_trades
        avg_win = float(trades.loc[trades["pnl_usd"] > 0, "pnl_usd"].mean()) if wins else 0.0
        avg_loss = float(trades.loc[trades["pnl_usd"] < 0, "pnl_usd"].mean()) if losses else 0.0
        profit_factor = (gross_profit / gross_loss_abs) if gross_loss_abs > 0 else (float("inf") if gross_profit > 0 else 0.0)
        expectancy = average_trade

        equity_curve = pd.Series([cfg.initial_equity] + trades["equity_after_trade"].astype(float).tolist(), dtype=float)
        running_max = equity_curve.cummax()
        drawdown_abs = running_max - equity_curve
        drawdown_pct = drawdown_abs / running_max.replace(0, pd.NA)
        max_drawdown_abs = float(drawdown_abs.max())
        max_drawdown_pct = float(drawdown_pct.fillna(0).max())

        exit_reason_counts = trades["exit_reason"].value_counts().to_dict()
        direction_counts = trades["direction"].value_counts().to_dict()
        if "entry_session" in trades.columns:
            session_counts = trades.groupby("entry_session").agg(
                trades=("pnl_usd", "count"),
                wins=("pnl_usd", lambda x: (x > 0).sum()),
                net_pnl=("pnl_usd", "sum"),
            ).to_dict(orient="index")
        else:
            session_counts = {}
    else:
        wins = losses = breakeven = 0
        gross_profit = gross_loss_abs = net_profit = 0.0
        win_rate = average_trade = avg_win = avg_loss = expectancy = 0.0
        profit_factor = 0.0
        max_drawdown_abs = max_drawdown_pct = 0.0
        exit_reason_counts = {}
        direction_counts = {}
        session_counts = {}

    return {
        "config": serializable_config(cfg),
        "data_window": {
            "signals_start": signals["event_time"].min().isoformat() if not signals.empty else None,
            "signals_end": signals["event_time"].max().isoformat() if not signals.empty else None,
            "regimes_start": regimes["event_time"].min().isoformat() if not regimes.empty else None,
            "regimes_end": regimes["event_time"].max().isoformat() if not regimes.empty else None,
            "bars_start": bars["bar_time"].min().isoformat() if not bars.empty else None,
            "bars_end": bars["bar_time"].max().isoformat() if not bars.empty else None,
        },
        "counts": {
            "total_signals": int(len(signals)),
            "total_regimes": int(len(regimes)),
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "skipped_signals": skipped_signals,
        },
        "performance": {
            "initial_equity": cfg.initial_equity,
            "final_equity": final_equity,
            "net_profit": net_profit,
            "return_pct": ((final_equity / cfg.initial_equity) - 1.0) * 100.0,
            "gross_profit": gross_profit,
            "gross_loss_abs": gross_loss_abs,
            "profit_factor": profit_factor,
            "win_rate_pct": win_rate * 100.0,
            "average_trade": average_trade,
            "average_win": avg_win,
            "average_loss": avg_loss,
            "expectancy": expectancy,
            "max_drawdown_abs": max_drawdown_abs,
            "max_drawdown_pct": max_drawdown_pct * 100.0,
        },
        "trade_breakdown": {
            "exit_reason_counts": exit_reason_counts,
            "direction_counts": direction_counts,
            "session_counts": session_counts,
        },
    }


def serializable_config(cfg: Config) -> Dict[str, Any]:
    data = asdict(cfg)
    data["output_dir"] = str(cfg.output_dir)
    data["start_time_utc"] = cfg.start_time_utc.isoformat() if cfg.start_time_utc is not None else None
    data["end_time_utc"] = cfg.end_time_utc.isoformat() if cfg.end_time_utc is not None else None
    return data


def write_outputs(cfg: Config, trades: pd.DataFrame, summary: Dict[str, Any]) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = cfg.output_dir / "summary.json"
    trades_path = cfg.output_dir / "trades.csv"

    if not trades.empty:
        trades_to_write = trades.copy()
        for col in [
            "signal_time",
            "entry_bar_time",
            "entry_regime_time",
            "exit_event_time",
            "exit_bar_time",
        ]:
            if col in trades_to_write.columns:
                trades_to_write[col] = pd.to_datetime(trades_to_write[col], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        trades_to_write.to_csv(trades_path, index=False)
        log.info("Wrote %d trades to %s", len(trades_to_write), trades_path)
    else:
        pd.DataFrame().to_csv(trades_path, index=False)
        log.info("No trades to write, empty CSV written to %s", trades_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info("Summary written to %s", summary_path)


def insert_bt_trade_history(conn, cfg: Config, trades: pd.DataFrame) -> int:
    if trades.empty:
        log.debug("insert_bt_trade_history: no trades to insert")
        return 0

    log.info("Inserting %d trades into bt_trade_history", len(trades))
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                COALESCE(MAX(closing_deal_id), 0) AS max_closing_deal_id,
                COALESCE(MAX(position_id), 0) AS max_position_id
            FROM public.bt_trade_history
            """
        )
        max_closing_deal_id, max_position_id = cur.fetchone()
        log.debug("Current max ids: closing_deal_id=%d position_id=%d", max_closing_deal_id, max_position_id)

        rows = []
        for offset, trade in enumerate(trades.itertuples(index=False), start=1):
            position_id = int(max_position_id) + offset
            closing_deal_id = int(max_closing_deal_id) + offset
            trade_type = "buy" if trade.direction == "long" else "sell"
            holding_duration = pd.to_datetime(trade.exit_bar_time, utc=True) - pd.to_datetime(trade.entry_bar_time, utc=True)
            label = f"backtest_{cfg.symbol_name}_{cfg.timeframe}"
            comment = (
                f"signal_id={trade.signal_id}; exit_reason={trade.exit_reason}; "
                f"trailing_sl={getattr(trade, 'trailing_sl_active', False)}; "
                f"sl_source={getattr(trade, 'sl_source', 'unknown')}"
            )

            rows.append(
                (
                    cfg.account_number,
                    closing_deal_id,
                    position_id,
                    cfg.symbol_name,
                    trade_type,
                    float(trade.volume),
                    float(trade.volume),
                    pd.to_datetime(trade.entry_bar_time, utc=True).to_pydatetime(),
                    float(trade.entry_price),
                    pd.to_datetime(trade.exit_bar_time, utc=True).to_pydatetime(),
                    float(trade.exit_price),
                    holding_duration.to_pytimedelta(),
                    float(trade.pnl_usd),
                    float(trade.pnl_usd),
                    0.0,
                    0.0,
                    float(trade.points_realized),
                    float(trade.equity_after_trade),
                    label,
                    comment,
                    "docker-compose",
                    None,
                    "USD",
                    cfg.account_type,
                )
            )

        execute_batch(
            cur,
            """
            INSERT INTO public.bt_trade_history (
                account_number,
                closing_deal_id,
                position_id,
                symbol_name,
                trade_type,
                volume_in_units,
                quantity_lots,
                entry_time_utc,
                entry_price,
                closing_time_utc,
                closing_price,
                holding_duration,
                gross_profit,
                net_profit,
                commissions,
                swap,
                pips,
                balance,
                label,
                comment,
                channel,
                broker_name,
                deposit_asset,
                account_type
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """,
            rows,
            page_size=500,
        )
    conn.commit()
    log.info("Inserted %d rows into bt_trade_history", len(rows))
    return len(rows)


def delete_bt_trade_history(conn, cfg: Config) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM public.bt_trade_history
            WHERE account_number = %s
              AND symbol_name = %s
            """,
            (cfg.account_number, cfg.symbol_name),
        )
        deleted = cur.rowcount
    conn.commit()
    log.info(
        "Deleted %d rows from bt_trade_history (account=%s symbol=%s)",
        deleted, cfg.account_number, cfg.symbol_name,
    )
    return deleted


def run_backtest(cfg: Config) -> Dict[str, Any]:
    log.info("=== Backtest started: %s %s ===", cfg.symbol_name, cfg.timeframe)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    with connect_postgres("BT", cfg.bt_db_name) as bt_conn:
        delete_bt_trade_history(bt_conn, cfg)
        signals = fetch_signals(bt_conn, cfg)
        regimes = fetch_regimes(bt_conn, cfg)

    if signals.empty:
        log.warning("No signals found — aborting backtest early")
        summary = {
            "config": serializable_config(cfg),
            "message": "No matching signals found for the selected symbol/timeframe/filter.",
            "total_signals": 0,
            "total_trades": 0,
            "trades_inserted_into_bt_trade_history": 0,
            "final_equity": cfg.initial_equity,
        }
        write_outputs(cfg, pd.DataFrame(), summary)
        return summary

    bars_start_time = signals["event_time"].min()
    log.debug("Bars start time set to earliest signal: %s", bars_start_time)
    with connect_postgres("MARKET", cfg.market_db_name) as market_conn:
        bars = fetch_bars(market_conn, cfg, bars_start_time)

    if bars.empty:
        raise RuntimeError("No market_data_1min bars found for the selected symbol and time window.")

    with connect_postgres("BT", cfg.bt_db_name) as bt_conn:
        pivots = fetch_pivots(bt_conn, cfg, bars_start_time)

    signals = signals.sort_values(["event_time", "id"]).reset_index(drop=True)
    regimes = regimes.sort_values(["event_time", "id"]).reset_index(drop=True) if not regimes.empty else regimes
    bars = bars.sort_values("bar_time").reset_index(drop=True)

    signal_map = build_event_map(signals, bars["bar_time"], "event_time")
    regime_map = build_event_map(regimes, bars["bar_time"], "event_time") if not regimes.empty else {}
    log.info("Event maps built: %d signal bars, %d regime bars", len(signal_map), len(regime_map))

    spread_half = cfg.spread_points / 2.0
    equity = cfg.initial_equity
    current_regime: Optional[Dict[str, Any]] = None
    position: Optional[Dict[str, Any]] = None
    trades: list[Dict[str, Any]] = []
    skipped_signals = {
        "no_regime": 0,
        "regime_filter": 0,
        "volume_too_small": 0,
        "entry_window": 0,
    }

    # Precompute ATR series for all bars
    atr_series = compute_atr_series(bars, period=cfg.atr_period)
    log.info(
        "ATR precomputed: period=%d  mean=%.2f  min=%.2f  max=%.2f",
        cfg.atr_period,
        float(atr_series.mean()),
        float(atr_series.min()),
        float(atr_series.max()),
    )

    log.info("Starting bar loop over %d bars", len(bars))
    for bar_idx, bar in bars.iterrows():
        bar_time = bar["bar_time"]
        open_mid = float(bar["open"])
        high_mid = float(bar["high"])
        low_mid = float(bar["low"])

        # Dynamic spread: override per bar if enabled
        if cfg.use_dynamic_spread:
            spread_half = get_dynamic_spread(bar_time) / 2.0
        else:
            spread_half = cfg.spread_points / 2.0

        # --- Regime update ---
        if bar_idx in regime_map:
            for _, rg in regime_map[bar_idx].iterrows():
                prev_label = regime_label(current_regime)
                current_regime = build_regime_state(rg)
                new_label = regime_label(current_regime)
                log.debug(
                    "BAR %d [%s] Regime update: %s -> %s (id=%d trend='%s')",
                    bar_idx, bar_time, prev_label, new_label,
                    current_regime["id"], current_regime.get("trend_text", ""),
                )

        # --- Weekly close ---
        if position is not None and weekly_close_due(bar_time, cfg):
            log.info(
                "BAR %d [%s] WEEKLY_CLOSE | signal_id=%s dir=%s",
                bar_idx, bar_time, position["signal_id"], position["direction"],
            )
            exit_price = open_mid - spread_half if position["direction"] == "long" else open_mid + spread_half
            trade = close_position(
                position=position,
                exit_reason="weekly_close",
                exit_bar_time=bar_time,
                exit_event_time=bar_time,
                exit_mid_price=open_mid,
                exit_price=exit_price,
                equity=equity,
                spread_half=spread_half,
            )
            equity = trade["equity_after_trade"]
            trades.append(trade)
            position = None

        # --- Regime-change exit ---
        if position is not None and not regime_allows(position["direction"], current_regime):
            log.info(
                "BAR %d [%s] Regime-change exit: signal_id=%s dir=%s regime=%s",
                bar_idx, bar_time, position["signal_id"], position["direction"],
                regime_label(current_regime),
            )
            exit_price = open_mid - spread_half if position["direction"] == "long" else open_mid + spread_half
            trade = close_position(
                position=position,
                exit_reason="regime_change",
                exit_bar_time=bar_time,
                exit_event_time=current_regime["event_time"] if current_regime is not None else bar_time,
                exit_mid_price=open_mid,
                exit_price=exit_price,
                equity=equity,
                spread_half=spread_half,
            )
            equity = trade["equity_after_trade"]
            trades.append(trade)
            position = None

        # --- Signal processing ---
        if bar_idx in signal_map:
            for _, sig in signal_map[bar_idx].iterrows():
                if position is not None:
                    log.debug(
                        "BAR %d [%s] Signal id=%s skipped: position already open (signal_id=%s)",
                        bar_idx, bar_time, sig["id"], position["signal_id"],
                    )
                    break

                direction = "long" if sig["action"] == "buy" else "short"

                if not entry_allowed(bar_time, cfg):
                    log.debug(
                        "BAR %d [%s] Signal id=%s skipped: outside entry window (weekday=%d hour=%d)",
                        bar_idx, bar_time, sig["id"], bar_time.weekday(), bar_time.hour,
                    )
                    skipped_signals["entry_window"] += 1
                    continue

                if current_regime is None:
                    log.debug(
                        "BAR %d [%s] Signal id=%s skipped: no regime active yet (dir=%s)",
                        bar_idx, bar_time, sig["id"], direction,
                    )
                    skipped_signals["no_regime"] += 1
                    continue
                regime_age_minutes = (bar_time - current_regime["event_time"]).total_seconds() / 60.0
                if regime_age_minutes > cfg.regime_validity_minutes:
                    log.debug(
                        "BAR %d [%s] Signal id=%s skipped: regime expired (age=%.1f min > %.1f min, regime_id=%d)",
                        bar_idx, bar_time, sig["id"], regime_age_minutes,
                        cfg.regime_validity_minutes, current_regime["id"],
                    )
                    skipped_signals["no_regime"] += 1
                    continue
                if not regime_allows(direction, current_regime):
                    log.debug(
                        "BAR %d [%s] Signal id=%s skipped: regime_filter (dir=%s regime=%s)",
                        bar_idx, bar_time, sig["id"], direction, regime_label(current_regime),
                    )
                    skipped_signals["regime_filter"] += 1
                    continue

                entry_mid = open_mid
                entry_price = open_mid + spread_half if direction == "long" else open_mid - spread_half

                # --- ATR-based stop distance ---
                atr_val = float(atr_series.iloc[bar_idx])
                stop_distance_points = atr_val * cfg.atr_multiplier
                if stop_distance_points <= 0:
                    log.warning(
                        "BAR %d [%s] Signal id=%s skipped: ATR stop_distance=%.6f <= 0",
                        bar_idx, bar_time, sig["id"], stop_distance_points,
                    )
                    skipped_signals["volume_too_small"] += 1
                    continue

                # --- Volume sizing: derive from risk budget and ATR stop ---
                max_volume = equity / (entry_price * cfg.required_margin)
                raw_target_volume = max_volume * cfg.position_fraction_of_max_margin

                notional_usd_target = entry_price * raw_target_volume
                margin_used_usd_target = notional_usd_target * cfg.required_margin
                risk_budget_usd = margin_used_usd_target * cfg.risk_fraction_of_used_margin

                # Volume derived from risk budget and market-based stop distance
                volume_from_risk = risk_budget_usd / stop_distance_points
                volume = floor_to_step(min(volume_from_risk, raw_target_volume), cfg.lot_step)

                notional_usd = entry_price * volume
                margin_used_usd = notional_usd * cfg.required_margin
                risk_budget_usd = margin_used_usd * cfg.risk_fraction_of_used_margin

                log.debug(
                    "BAR %d [%s] Signal id=%s sizing: equity=%.2f entry=%.4f atr=%.2f "
                    "stop_dist=%.2f risk_budget=%.2f vol_from_risk=%.4f floored=%.4f",
                    bar_idx, bar_time, sig["id"], equity, entry_price,
                    atr_val, stop_distance_points, risk_budget_usd, volume_from_risk, volume,
                )

                if volume < cfg.lot_step:
                    log.warning(
                        "BAR %d [%s] Signal id=%s skipped: volume_too_small (%.4f < %.4f)",
                        bar_idx, bar_time, sig["id"], volume, cfg.lot_step,
                    )
                    skipped_signals["volume_too_small"] += 1
                    continue

                # --- SL from last pivot + ATR buffer (no TP) ---
                pivot_buffer = atr_val * cfg.pivot_buffer_atr_multiplier

                if direction == "long":
                    # SL below last confirmed pivot low
                    pivot_price = get_last_pivot_price(pivots, bar_time, direction=-1)
                    if pivot_price is not None:
                        stop_price = pivot_price - pivot_buffer
                        stop_distance_points = entry_price - stop_price
                        sl_source = "pivot_low"
                    else:
                        # Fallback: ATR-based
                        stop_distance_points = atr_val * cfg.atr_multiplier
                        stop_price = entry_price - stop_distance_points
                        sl_source = "atr_fallback"
                else:
                    # SL above last confirmed pivot high
                    pivot_price = get_last_pivot_price(pivots, bar_time, direction=1)
                    if pivot_price is not None:
                        stop_price = pivot_price + pivot_buffer
                        stop_distance_points = stop_price - entry_price
                        sl_source = "pivot_high"
                    else:
                        stop_distance_points = atr_val * cfg.atr_multiplier
                        stop_price = entry_price + stop_distance_points
                        sl_source = "atr_fallback"

                if stop_distance_points <= 0:
                    log.warning(
                        "BAR %d [%s] Signal id=%s skipped: stop_distance=%.4f <= 0 (sl_source=%s)",
                        bar_idx, bar_time, sig["id"], stop_distance_points, sl_source,
                    )
                    skipped_signals["volume_too_small"] += 1
                    continue

                # Safeguard: SL must never be on wrong side of entry
                if direction == "long" and stop_price >= entry_price:
                    log.warning(
                        "BAR %d [%s] Signal id=%s: SL %.4f >= entry %.4f — clamping to entry - 1pt",
                        bar_idx, bar_time, sig["id"], stop_price, entry_price,
                    )
                    stop_price = entry_price - 1.0
                    stop_distance_points = 1.0
                elif direction == "short" and stop_price <= entry_price:
                    log.warning(
                        "BAR %d [%s] Signal id=%s: SL %.4f <= entry %.4f — clamping to entry + 1pt",
                        bar_idx, bar_time, sig["id"], stop_price, entry_price,
                    )
                    stop_price = entry_price + 1.0
                    stop_distance_points = 1.0

                log.info(
                    "BAR %d [%s] OPEN | signal_id=%s dir=%s entry=%.4f vol=%.2f "
                    "sl=%.4f sl_source=%s stop_dist=%.2f margin=%.2f risk=%.2f regime=%s",
                    bar_idx, bar_time, sig["id"], direction, entry_price, volume,
                    stop_price, sl_source, stop_distance_points,
                    margin_used_usd, risk_budget_usd,
                    regime_label(current_regime),
                )

                position = {
                    "signal_id": int(sig["id"]),
                    "signal_time": sig["event_time"],
                    "symbol": cfg.symbol_name,
                    "timeframe": cfg.timeframe,
                    "direction": direction,
                    "entry_bar_time": bar_time,
                    "entry_bar_index": int(bar_idx),
                    "entry_mid_price": entry_mid,
                    "entry_price": entry_price,
                    "volume": volume,
                    "notional_usd": notional_usd,
                    "margin_used_usd": margin_used_usd,
                    "risk_budget_usd": risk_budget_usd,
                    "initial_stop_price": stop_price,
                    "stop_price": stop_price,
                    "stop_distance_points": stop_distance_points,
                    "sl_source": sl_source,
                    "trailing_sl_active": False,
                    "entry_regime_time": current_regime["event_time"],
                    "entry_regime_label": regime_label(current_regime),
                    "equity_before_trade": equity,
                    "bars_held_counter": 0,
                    "entry_session": session_label(bar_time),
                    "entry_atr": atr_val,
                }
                break

        # --- SL check + trailing stop update ---
        if position is not None:
            position["bars_held_counter"] += 1

            # --- Timeout exit ---
            if position["bars_held_counter"] >= cfg.max_bars_held:
                log.info(
                    "BAR %d [%s] TIMEOUT | signal_id=%s dir=%s bars_held=%d",
                    bar_idx, bar_time, position["signal_id"],
                    position["direction"], position["bars_held_counter"],
                )
                exit_price = open_mid - spread_half if position["direction"] == "long" else open_mid + spread_half
                trade = close_position(
                    position=position,
                    exit_reason="timeout",
                    exit_bar_time=bar_time,
                    exit_event_time=bar_time,
                    exit_mid_price=open_mid,
                    exit_price=exit_price,
                    equity=equity,
                    spread_half=spread_half,
                )
                equity = trade["equity_after_trade"]
                trades.append(trade)
                position = None
                continue

            # --- SL check (against SL set in previous bar) ---
            # Gap handling: if the bar opens beyond the stop, the fill happens
            # at the open price (worse than the stop), not at the stop price.
            # Without this, a trailing-SL that is above the bar's high (long)
            # or below the bar's low (short) would trigger an exit AT the stop
            # price — a price the market never actually traded at.
            if position["direction"] == "long":
                if open_mid <= position["stop_price"]:
                    # Gap through stop at bar open — fill at open
                    exit_mid = open_mid
                    exit_price = open_mid - spread_half
                    exit_reason = "stop_loss_gap"
                elif (low_mid - spread_half) <= position["stop_price"]:
                    # Normal intrabar stop hit
                    exit_price = position["stop_price"]
                    exit_mid = exit_price + spread_half
                    exit_reason = "stop_loss"
                else:
                    exit_reason = None
            else:
                if open_mid >= position["stop_price"]:
                    # Gap through stop at bar open — fill at open
                    exit_mid = open_mid
                    exit_price = open_mid + spread_half
                    exit_reason = "stop_loss_gap"
                elif (high_mid + spread_half) >= position["stop_price"]:
                    # Normal intrabar stop hit
                    exit_price = position["stop_price"]
                    exit_mid = exit_price - spread_half
                    exit_reason = "stop_loss"
                else:
                    exit_reason = None

            if exit_reason in ("stop_loss", "stop_loss_gap"):
                log.info(
                    "BAR %d [%s] %s | signal_id=%s dir=%s exit=%.4f sl=%.4f open=%.4f",
                    bar_idx, bar_time, exit_reason.upper(),
                    position["signal_id"], position["direction"], exit_price,
                    position["stop_price"], open_mid,
                )
                trade = close_position(
                    position=position,
                    exit_reason=exit_reason,
                    exit_bar_time=bar_time,
                    exit_event_time=bar_time,
                    exit_mid_price=exit_mid,
                    exit_price=exit_price,
                    equity=equity,
                    spread_half=spread_half,
                )
                equity = trade["equity_after_trade"]
                trades.append(trade)
                position = None
                continue

            # --- SL update for next bar (trailing / breakeven) ---
            # Uses prev bar close as reference — no look-ahead into current bar
            if bar_idx > 0:
                prev_close = float(bars.at[bar_idx - 1, "close"])
            else:
                prev_close = position["entry_price"]

            if position["direction"] == "long":
                unrealised_pnl = (prev_close - position["entry_price"]) * position["volume"]
            else:
                unrealised_pnl = (position["entry_price"] - prev_close) * position["volume"]

            bars_held = position["bars_held_counter"]
            trailing_threshold = cfg.trailing_activation_rr * position["risk_budget_usd"]

            if unrealised_pnl >= trailing_threshold:
                # Phase 2: trailing — use late multiplier after TRAILING_LATE_MINUTES
                atr_val = float(atr_series.iloc[bar_idx])
                if bars_held >= cfg.trailing_late_minutes:
                    trail_buffer = atr_val * cfg.trailing_buffer_atr_multiplier_late
                else:
                    trail_buffer = atr_val * cfg.trailing_buffer_atr_multiplier

                # Safeguard: trailing buffer must be at least spread-wide.
                # A near-zero buffer causes the SL to sit right on prev_close,
                # which gets easily gapped through on the next bar.
                min_buffer = cfg.spread_points
                if trail_buffer < min_buffer:
                    trail_buffer = min_buffer

                if position["direction"] == "long":
                    candidate_sl = prev_close - trail_buffer
                    if candidate_sl > position["stop_price"]:
                        log.debug(
                            "BAR %d [%s] TRAIL long | signal_id=%s sl %.4f -> %.4f "
                            "(prev_close=%.4f late=%s)",
                            bar_idx, bar_time, position["signal_id"],
                            position["stop_price"], candidate_sl, prev_close,
                            bars_held >= cfg.trailing_late_minutes,
                        )
                        position["stop_price"] = candidate_sl
                        position["trailing_sl_active"] = True
                else:
                    candidate_sl = prev_close + trail_buffer
                    if candidate_sl < position["stop_price"]:
                        log.debug(
                            "BAR %d [%s] TRAIL short | signal_id=%s sl %.4f -> %.4f "
                            "(prev_close=%.4f late=%s)",
                            bar_idx, bar_time, position["signal_id"],
                            position["stop_price"], candidate_sl, prev_close,
                            bars_held >= cfg.trailing_late_minutes,
                        )
                        position["stop_price"] = candidate_sl
                        position["trailing_sl_active"] = True

            elif not position["trailing_sl_active"] and bars_held >= cfg.breakeven_minutes:
                # Phase 1 only: move SL to breakeven + offset after BREAKEVEN_MINUTES
                if position["direction"] == "long":
                    breakeven = position["entry_price"] + cfg.breakeven_offset_points
                    if breakeven > position["stop_price"]:
                        log.debug(
                            "BAR %d [%s] BREAKEVEN long | signal_id=%s sl %.4f -> %.4f",
                            bar_idx, bar_time, position["signal_id"],
                            position["stop_price"], breakeven,
                        )
                        position["stop_price"] = breakeven
                else:
                    breakeven = position["entry_price"] - cfg.breakeven_offset_points
                    if breakeven < position["stop_price"]:
                        log.debug(
                            "BAR %d [%s] BREAKEVEN short | signal_id=%s sl %.4f -> %.4f",
                            bar_idx, bar_time, position["signal_id"],
                            position["stop_price"], breakeven,
                        )
                        position["stop_price"] = breakeven

            log.debug(
                "BAR %d [%s] HOLD %s | low=%.4f high=%.4f sl=%.4f trail=%s bars_held=%d",
                bar_idx, bar_time, position["direction"], low_mid, high_mid,
                position["stop_price"], position["trailing_sl_active"],
                position["bars_held_counter"],
            )
            continue

    # --- End-of-data close ---
    if position is not None:
        log.info(
            "End-of-data: closing open position signal_id=%s dir=%s bars_held=%d",
            position["signal_id"], position["direction"], position["bars_held_counter"],
        )
        last_bar = bars.iloc[-1]
        last_time = last_bar["bar_time"]
        last_mid_close = float(last_bar["close"])
        exit_price = last_mid_close - spread_half if position["direction"] == "long" else last_mid_close + spread_half
        trade = close_position(
            position=position,
            exit_reason="end_of_data",
            exit_bar_time=last_time,
            exit_event_time=last_time,
            exit_mid_price=last_mid_close,
            exit_price=exit_price,
            equity=equity,
            spread_half=spread_half,
        )
        equity = trade["equity_after_trade"]
        trades.append(trade)

    trades_df = pd.DataFrame(trades)

    # --- Final summary log ---
    total = len(trades_df)
    if total:
        wins = int((trades_df["pnl_usd"] > 0).sum())
        losses = int((trades_df["pnl_usd"] < 0).sum())
        net = float(trades_df["pnl_usd"].sum())
        log.info(
            "Bar loop done | trades=%d wins=%d losses=%d net_pnl=%.2f final_equity=%.2f "
            "skipped(no_regime=%d regime_filter=%d vol_too_small=%d entry_window=%d)",
            total, wins, losses, net, equity,
            skipped_signals["no_regime"], skipped_signals["regime_filter"],
            skipped_signals["volume_too_small"], skipped_signals["entry_window"],
        )
    else:
        log.warning(
            "Bar loop done | NO trades executed. "
            "skipped(no_regime=%d regime_filter=%d vol_too_small=%d entry_window=%d)",
            skipped_signals["no_regime"], skipped_signals["regime_filter"],
            skipped_signals["volume_too_small"], skipped_signals["entry_window"],
        )

    inserted_count = 0
    if not trades_df.empty:
        with connect_postgres("BT", cfg.bt_db_name) as bt_conn:
            inserted_count = insert_bt_trade_history(bt_conn, cfg, trades_df)

    summary = build_summary(cfg, signals, regimes, bars, trades_df, skipped_signals)
    summary["trades_inserted_into_bt_trade_history"] = inserted_count
    write_outputs(cfg, trades_df, summary)
    log.info("=== Backtest finished ===")
    return summary


def main() -> None:
    cfg = load_config()
    summary = run_backtest(cfg)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()