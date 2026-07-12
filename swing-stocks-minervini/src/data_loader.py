"""Load source data from TimescaleDB with a local parquet cache.

Sources (all produced by market-data-stack/stock_core_and_fundamental_data_fetcher):
  - stock_core_market_metrics_current      canonical current security identity
  - stock_core_market_metrics_daily        adjusted daily OHLCV
  - stock_core_security_master_current     universe / quote_type filter
  - ibkr_symbols                           IBKR industry/category taxonomy
  - stock_core_sec_quarterly_fundamental_events SEC quarterly reported values,
    prior-year comparables and margins per filing accession.
  - stock_core_13f_sponsorship_events       SEC institutional holding changes,
    usable only from each filing's effective date.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta

import pandas as pd

from . import db
from .config import Config

log = logging.getLogger(__name__)

CANONICAL_IDENTITY_SQL = """
SELECT DISTINCT ON (symbol)
       symbol, exchange, cik
FROM stock_core_market_metrics_current
WHERE COALESCE(adjusted_close, raw_close, current_price) IS NOT NULL
ORDER BY symbol,
         COALESCE(source_asof_ts, source_last_success_ts, row_updated_ts) DESC NULLS LAST,
         COALESCE(adjusted_volume, raw_volume, average_daily_volume_3m, 0) DESC,
         exchange,
         cik
"""

PRICES_SQL = f"""
WITH canonical_identity AS (
    {CANONICAL_IDENTITY_SQL}
)
SELECT p.symbol,
       p.period_end_date::date                  AS date,
       p.adjusted_open::float8                  AS open,
       p.adjusted_high::float8                  AS high,
       p.adjusted_low::float8                   AS low,
       p.adjusted_close::float8                 AS close,
       COALESCE(p.adjusted_volume, p.raw_volume)::float8 AS volume
FROM stock_core_market_metrics_daily p
JOIN canonical_identity i
  ON i.symbol = p.symbol
 AND i.exchange = p.exchange
 AND i.cik = p.cik
WHERE p.period_end_date BETWEEN %(start)s AND %(end)s
  AND p.adjusted_open IS NOT NULL
  AND p.adjusted_high IS NOT NULL
  AND p.adjusted_low IS NOT NULL
  AND p.adjusted_close IS NOT NULL
  AND COALESCE(p.adjusted_volume, p.raw_volume) IS NOT NULL
"""

UNIVERSE_SQL = f"""
WITH canonical_identity AS (
    {CANONICAL_IDENTITY_SQL}
),
ibkr AS (
    SELECT UPPER(TRIM(source_symbol)) AS symbol,
           max(NULLIF(TRIM(ibkr_industry), '')) AS ibkr_industry,
           max(NULLIF(TRIM(ibkr_category), '')) AS ibkr_category
    FROM ibkr_symbols
    WHERE source_symbol IS NOT NULL
      AND TRIM(source_symbol) <> ''
    GROUP BY UPPER(TRIM(source_symbol))
)
SELECT sm.symbol,
       sm.exchange,
       sm.cik,
       ibkr.ibkr_industry,
       ibkr.ibkr_category
FROM canonical_identity i
JOIN stock_core_security_master_current sm
  ON sm.symbol = i.symbol
 AND sm.exchange = i.exchange
 AND sm.cik = i.cik
LEFT JOIN ibkr ON ibkr.symbol = UPPER(TRIM(sm.symbol))
WHERE upper(sm.quote_type) = 'EQUITY'
"""

REGIME_SQL = """
SELECT day::date              AS day,
       composite_score::float8 AS regime_composite,
       regime_label
FROM world_regime_daily_scores_mv
WHERE day <= %(end)s
"""

QUARTERLY_FUNDAMENTALS_SQL = f"""
WITH canonical_identity AS (
    {CANONICAL_IDENTITY_SQL}
)
SELECT e.symbol,
       e.effective_date::date                         AS available_date,
       e.accepted_at,
       e.accession_number,
       e.fiscal_period_end_date::date                 AS fiscal_period_end_date,
       e.diluted_eps::float8                          AS diluted_eps,
       e.prior_year_diluted_eps::float8               AS prior_year_diluted_eps,
       e.quarterly_revenue::float8                    AS quarterly_revenue,
       e.prior_year_quarterly_revenue::float8         AS prior_year_quarterly_revenue,
       e.quarterly_operating_margin::float8           AS quarterly_operating_margin,
       e.prior_year_quarterly_operating_margin::float8 AS prior_year_quarterly_operating_margin,
       e.quarterly_net_margin::float8                 AS quarterly_net_margin,
       e.prior_year_quarterly_net_margin::float8      AS prior_year_quarterly_net_margin
FROM stock_core_sec_quarterly_fundamental_events e
JOIN canonical_identity i
  ON i.symbol = e.symbol
 AND i.exchange = e.exchange
 AND i.cik = e.cik
WHERE e.effective_date <= %(end)s
"""

SPONSORSHIP_SQL = f"""
WITH canonical_identity AS (
    {CANONICAL_IDENTITY_SQL}
)
SELECT e.symbol,
       e.effective_date::date AS available_date,
       SUM(e.manager_count_delta)::float8 AS manager_count_delta,
       SUM(e.new_position_count + e.increased_count
           - e.decreased_count - e.exited_count)::float8 AS net_activity_delta
FROM stock_core_13f_sponsorship_events e
JOIN canonical_identity i
  ON i.symbol = e.symbol
 AND i.exchange = e.exchange
 AND i.cik = e.cik
WHERE e.effective_date <= %(end)s
GROUP BY e.symbol, e.effective_date
"""


def _cached(cfg: Config, name: str, loader, *, cache_empty: bool = True) -> pd.DataFrame:
    path = os.path.join(cfg.cache_dir, f"{name}.parquet")
    if not cfg.force_refresh and os.path.exists(path):
        cached = pd.read_parquet(path)
        if cache_empty or not cached.empty:
            log.info("cache hit: %s", path)
            return cached
    df = loader()
    if cache_empty or not df.empty:
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
        df["date"] = pd.to_datetime(df["date"])
        return df.reset_index(drop=True)

    return _cached(cfg, f"prices_identity_v2_{start}_{end}", _load)


def load_universe(conn, cfg: Config) -> pd.DataFrame:
    """Canonical equity identities with their symbol-level IBKR taxonomy."""

    def _load():
        return db.read_df(conn, UNIVERSE_SQL)

    return _cached(cfg, f"universe_identity_ibkr_v2_{effective_end(cfg)}", _load)


def _normalize_quarterly_fundamental_events(df: pd.DataFrame) -> pd.DataFrame:
    """Select the latest economic quarter after each accession becomes usable."""
    df = df.copy()
    df["available_date"] = pd.to_datetime(df["available_date"])
    df["accepted_at"] = pd.to_datetime(df["accepted_at"], utc=True)
    df["fiscal_period_end_date"] = pd.to_datetime(df["fiscal_period_end_date"])
    # Missing acceptance time has unknown intraday order. Sort it before every
    # known acceptance on that effective day so it cannot overwrite a filing
    # whose causal order is actually known. Multiple unknowns remain stable by
    # accession number.
    df["_accepted_known"] = df["accepted_at"].notna()
    accepted_fallback = (
        df["available_date"].dt.tz_localize("UTC") - pd.Timedelta(days=1)
    )
    df["_accepted_sort"] = df["accepted_at"].fillna(accepted_fallback)
    df = df.sort_values(
        [
            "symbol",
            "available_date",
            "_accepted_known",
            "_accepted_sort",
            "accession_number",
        ],
        kind="stable",
    )

    # A late amendment to an older fiscal period may revise history, but it
    # must not replace a more recent quarter as the company's current result.
    latest_period = df.groupby("symbol")["fiscal_period_end_date"].cummax()
    df = df.loc[df["fiscal_period_end_date"].eq(latest_period)]
    return (
        df.drop_duplicates(["symbol", "available_date"], keep="last")
        .drop(columns=["_accepted_known", "_accepted_sort"])
        .reset_index(drop=True)
    )


def _require_quarterly_fundamental_events(df: pd.DataFrame) -> pd.DataFrame:
    """Fail loudly when the upstream schema exists but was not backfilled."""
    if df.empty:
        raise RuntimeError(
            "stock_core_sec_quarterly_fundamental_events is empty; run the "
            "stock_core_and_fundamental_data_fetcher schema init and startup "
            "historical backfill before the Minervini backtest"
        )
    return df


def load_quarterly_fundamentals(conn, cfg: Config) -> pd.DataFrame:
    """Load accession-keyed SEC quarterly events in deterministic PIT order."""
    end = effective_end(cfg)

    def _load():
        df = db.read_df(conn, QUARTERLY_FUNDAMENTALS_SQL, {"end": end})
        return _normalize_quarterly_fundamental_events(_require_quarterly_fundamental_events(df))

    # Validate after the cache read as well so an accidentally cached empty
    # source cannot silently turn the EPS leg of the 2-of-3 screen off.
    return _require_quarterly_fundamental_events(
        _cached(cfg, f"quarterly_fundamentals_identity_v1_{end}", _load)
    )


def load_sponsorship_events(conn, cfg: Config) -> pd.DataFrame:
    end = effective_end(cfg)

    def _load():
        df = db.read_df(conn, SPONSORSHIP_SQL, {"end": end})
        if not df.empty:
            df["available_date"] = pd.to_datetime(df["available_date"])
        return df

    events = _cached(cfg, f"sponsorship_identity_v1_{end}", _load, cache_empty=False)
    if cfg.institutional_sponsorship_filter_enable and events.empty:
        raise RuntimeError(
            "stock_core_13f_sponsorship_events is empty while the institutional sponsorship filter is enabled"
        )
    return events


def load_regime_scores(conn, cfg: Config) -> pd.DataFrame:
    """Load world-regime attribution and optional entry-gate state.

    An unavailable source degrades to empty attribution only while the gate is
    disabled. With the gate enabled, missing state aborts instead of producing
    a silently all-blocked backtest.
    """
    end = effective_end(cfg)

    def _load():
        try:
            return db.read_df(conn, REGIME_SQL, {"end": end})
        except Exception as exc:
            conn.rollback()
            raise RuntimeError("world_regime_daily_scores_mv is not readable") from exc

    try:
        regime = _cached(cfg, f"regime_v2_{end}", _load, cache_empty=False)
    except RuntimeError:
        if cfg.regime_entry_filter_enable:
            raise
        log.warning("world_regime_daily_scores_mv not readable - regime attribution skipped")
        return pd.DataFrame(columns=["day", "regime_composite", "regime_label"])
    if cfg.regime_entry_filter_enable and regime.empty:
        raise RuntimeError(
            "world_regime_daily_scores_mv returned no rows while the regime entry filter is enabled"
        )
    return regime


def pivot_field(prices: pd.DataFrame, field: str) -> pd.DataFrame:
    """Long price frame -> wide matrix (index=date, columns=symbol)."""
    return prices.pivot(index="date", columns="symbol", values=field).sort_index()
