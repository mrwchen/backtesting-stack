"""Load source data from TimescaleDB with a local parquet cache.

Sources (all produced by market-data-stack/stock_core_and_fundamental_data_fetcher):
  - stock_core_market_metrics_daily        adjusted daily OHLCV
  - stock_core_security_master_current     universe / quote_type filter
  - ibkr_symbols                           IBKR industry/category taxonomy
  - stock_core_sec_fundamentals_asof_daily SEC TTM fundamentals; period_end_date is
    the point-in-time availability date (filing acceptance), see column comment.
    Quarterly EPS is derived from TTM net-income diffs — the earnings calendar
    carries no reported EPS in practice.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

PRICES_SQL = """
SELECT symbol,
       period_end_date::date                  AS date,
       adjusted_open::float8                  AS open,
       adjusted_high::float8                  AS high,
       adjusted_low::float8                   AS low,
       adjusted_close::float8                 AS close,
       COALESCE(adjusted_volume, raw_volume)::float8 AS volume
FROM stock_core_market_metrics_daily
WHERE period_end_date BETWEEN %(start)s AND %(end)s
  AND adjusted_close IS NOT NULL
"""

UNIVERSE_SQL = """
WITH ibkr AS (
    SELECT UPPER(TRIM(source_symbol)) AS symbol,
           max(NULLIF(TRIM(ibkr_industry), '')) AS ibkr_industry,
           max(NULLIF(TRIM(ibkr_category), '')) AS ibkr_category
    FROM ibkr_symbols
    WHERE source_symbol IS NOT NULL
      AND TRIM(source_symbol) <> ''
    GROUP BY UPPER(TRIM(source_symbol))
)
SELECT sm.symbol,
       max(ibkr.ibkr_industry) AS ibkr_industry,
       max(ibkr.ibkr_category) AS ibkr_category
FROM stock_core_security_master_current sm
LEFT JOIN ibkr ON ibkr.symbol = UPPER(TRIM(sm.symbol))
WHERE upper(sm.quote_type) = 'EQUITY'
GROUP BY sm.symbol
"""

REGIME_SQL = """
SELECT day::date              AS day,
       composite_score::float8 AS regime_composite,
       regime_label
FROM world_regime_daily_scores_mv
WHERE day <= %(end)s
"""

FUNDAMENTALS_SQL = """
SELECT symbol,
       period_end_date::date                        AS available_date,
       sec_revenue_ttm::float8                      AS revenue_ttm,
       sec_net_margin_ttm::float8                   AS net_margin_ttm,
       sec_net_income_ttm::float8                   AS net_income_ttm,
       sec_weighted_avg_shares_diluted::float8      AS shares_diluted
FROM stock_core_sec_fundamentals_asof_daily
WHERE period_end_date <= %(end)s
"""


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


def warmup_start(cfg: Config) -> date:
    start = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
    return start - timedelta(days=cfg.warmup_calendar_days)


def effective_end(cfg: Config) -> date:
    if cfg.end_date:
        return datetime.strptime(cfg.end_date, "%Y-%m-%d").date()
    return date.today()


def load_prices(conn, cfg: Config) -> pd.DataFrame:
    start, end = warmup_start(cfg), effective_end(cfg)

    def _load():
        df = db.read_df(conn, PRICES_SQL, {"start": start, "end": end})
        # a symbol may exist on several (exchange, cik) identities: keep the
        # row with the highest volume per (symbol, date)
        df = df.sort_values(["symbol", "date", "volume"]).drop_duplicates(
            ["symbol", "date"], keep="last"
        )
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)

    return _cached(cfg, f"prices_{start}_{end}", _load)


def load_universe(conn, cfg: Config) -> pd.DataFrame:
    """Equity universe with IBKR taxonomy (symbol, ibkr_industry, ibkr_category)."""

    def _load():
        return db.read_df(conn, UNIVERSE_SQL)

    return _cached(cfg, f"universe_ibkr_v1_{effective_end(cfg)}", _load)


def load_fundamentals(conn, cfg: Config) -> pd.DataFrame:
    end = effective_end(cfg)

    def _load():
        df = db.read_df(conn, FUNDAMENTALS_SQL, {"end": end})
        df["available_date"] = pd.to_datetime(df["available_date"])
        return df

    return _cached(cfg, f"fundamentals_v2_{end}", _load)


def load_regime_scores(conn, cfg: Config) -> pd.DataFrame:
    """World-regime composite per day, used only as trade attribution.
    Returns an empty frame if the materialized view is unavailable."""
    end = effective_end(cfg)

    def _load():
        try:
            return db.read_df(conn, REGIME_SQL, {"end": end})
        except Exception:
            conn.rollback()
            log.warning("world_regime_daily_scores_mv not readable - regime attribution skipped")
            return pd.DataFrame(columns=["day", "regime_composite", "regime_label"])

    return _cached(cfg, f"regime_{end}", _load)


def pivot_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Long price frame -> wide matrix (index=date, columns=symbol)."""
    return prices.pivot(index="date", columns="symbol", values=field).sort_index()
