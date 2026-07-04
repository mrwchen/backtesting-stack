"""Load adjusted daily OHLCV and IBKR-backed stock universe from TimescaleDB."""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

SOURCE_TABLES = (
    "stock_core_market_metrics_daily",
    "stock_core_security_master_current",
    "ibkr_symbols",
)

PRICES_SQL = """
SELECT symbol,
       period_end_date::date AS date,
       COALESCE(adjusted_open, raw_open)::float8 AS open,
       COALESCE(adjusted_high, raw_high)::float8 AS high,
       COALESCE(adjusted_low, raw_low)::float8 AS low,
       COALESCE(adjusted_close, raw_close)::float8 AS close,
       COALESCE(adjusted_volume, raw_volume)::float8 AS volume
FROM stock_core_market_metrics_daily
WHERE period_end_date BETWEEN %(start)s AND %(end)s
  AND COALESCE(adjusted_open, raw_open) IS NOT NULL
  AND COALESCE(adjusted_high, raw_high) IS NOT NULL
  AND COALESCE(adjusted_low, raw_low) IS NOT NULL
  AND COALESCE(adjusted_close, raw_close) IS NOT NULL
  AND COALESCE(adjusted_volume, raw_volume) IS NOT NULL
"""

UNIVERSE_SQL = """
WITH ibkr AS (
    SELECT UPPER(TRIM(source_symbol)) AS symbol,
           max(NULLIF(TRIM(ibkr_industry), '')) AS ibkr_industry,
           max(NULLIF(TRIM(ibkr_category), '')) AS ibkr_category,
           max(NULLIF(TRIM(ib_symbol), '')) AS ib_symbol,
           max(NULLIF(TRIM(currency), '')) AS ibkr_currency,
           max(NULLIF(TRIM(sec_type), '')) AS ibkr_sec_type
    FROM ibkr_symbols
    WHERE source_symbol IS NOT NULL
      AND TRIM(source_symbol) <> ''
      AND (sec_type IS NULL OR UPPER(TRIM(sec_type)) = 'STK')
      AND (currency IS NULL OR UPPER(TRIM(currency)) = 'USD')
    GROUP BY UPPER(TRIM(source_symbol))
)
SELECT UPPER(TRIM(sm.symbol)) AS symbol,
       max(NULLIF(TRIM(sm.currency), '')) AS currency,
       max(ibkr.ib_symbol) AS ib_symbol,
       max(ibkr.ibkr_currency) AS ibkr_currency,
       max(ibkr.ibkr_sec_type) AS ibkr_sec_type,
       max(ibkr.ibkr_industry) AS ibkr_industry,
       max(ibkr.ibkr_category) AS ibkr_category
FROM stock_core_security_master_current sm
JOIN ibkr ON ibkr.symbol = UPPER(TRIM(sm.symbol))
WHERE sm.symbol IS NOT NULL
  AND TRIM(sm.symbol) <> ''
  AND upper(sm.quote_type) = 'EQUITY'
  AND (sm.currency IS NULL OR upper(sm.currency) = 'USD')
GROUP BY UPPER(TRIM(sm.symbol))
"""


def warmup_start(cfg: Config) -> date:
    start = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
    return start - timedelta(days=cfg.warmup_calendar_days)


def effective_end(cfg: Config) -> date:
    if cfg.end_date:
        return datetime.strptime(cfg.end_date, "%Y-%m-%d").date()
    return date.today()


def _cached(cfg: Config, name: str, loader) -> pd.DataFrame:
    path = os.path.join(cfg.cache_dir, f"{name}.parquet")
    if not cfg.force_refresh and os.path.exists(path):
        log.info("cache hit: %s", path)
        return pd.read_parquet(path)
    df = loader()
    os.makedirs(cfg.cache_dir, exist_ok=True)
    df.to_parquet(path, index=False)
    log.info("cached %d rows -> %s", len(df), path)
    return df


def load_universe(conn, cfg: Config) -> pd.DataFrame:
    def _load() -> pd.DataFrame:
        df = db.read_df(conn, UNIVERSE_SQL)
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        return df.drop_duplicates("symbol").reset_index(drop=True)

    return _cached(cfg, f"wei_universe_ibkr_{effective_end(cfg)}", _load)


def load_prices(conn, cfg: Config) -> pd.DataFrame:
    start, end = warmup_start(cfg), effective_end(cfg)

    def _load() -> pd.DataFrame:
        df = db.read_df(conn, PRICES_SQL, {"start": start, "end": end})
        if df.empty:
            return df
        df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
        df["date"] = pd.to_datetime(df["date"])
        for column in ("open", "high", "low", "close", "volume"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["open", "high", "low", "close", "volume"])
        df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)]
        df = df[df["volume"] >= 0]
        df = df.sort_values(["symbol", "date", "volume"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        return df.reset_index(drop=True)

    return _cached(cfg, f"wei_prices_{start}_{end}", _load)


def pivot_prices(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    return prices.pivot(index="date", columns="symbol", values=field).sort_index()
