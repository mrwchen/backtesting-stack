"""Load universe, prices, category momentum and stress scores from TimescaleDB."""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from psycopg2 import sql

from .config import Config
from .db import read_df

log = logging.getLogger(__name__)


def load_universe(conn, cfg: Config) -> pd.DataFrame:
    """Top-N stocks per IBKR category by market cap, with enough history.

    Returns columns: symbol, ibkr_category, market_cap.

    UNIVERSE_MCAP_ASOF picks the market-cap anchor: 'start' selects by the
    market cap as of the window start (point-in-time, no peeking at future
    winners; source is the daily market cap in METRICS_TABLE), 'end' keeps the
    old biased behaviour (latest market cap up to the window end) for
    comparisons with legacy runs.

    Remaining survivorship caveats even with 'start': the IBKR category
    mapping is the current snapshot (ibkr_symbols has no history), the
    coverage filter looks at the whole run window, and delisted stocks are
    largely missing from the price data itself.
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    mcap_asof = cfg.start_date if cfg.universe_mcap_asof == "start" else cfg.end_date
    df = read_df(
        conn,
        sql.SQL("""
        WITH ib AS (
            SELECT DISTINCT ON (source_symbol)
                   source_symbol AS symbol, ibkr_category
            FROM {symbols}
            WHERE ibkr_category IS NOT NULL
            ORDER BY source_symbol, fetched_at DESC
        ),
        cov AS (
            SELECT symbol, count(*) AS n_rows, min(period_end_date) AS first_day
            FROM {metrics}
            WHERE period_end_date BETWEEN %(warmup_start)s AND %(end)s
              AND adjusted_close > 0
            GROUP BY symbol
        ),
        mc AS (
            SELECT DISTINCT ON (symbol) symbol, market_cap
            FROM {metrics}
            WHERE market_cap IS NOT NULL AND period_end_date <= %(mcap_asof)s
            ORDER BY symbol, period_end_date DESC
        )
        SELECT ib.symbol, ib.ibkr_category, mc.market_cap, cov.n_rows, cov.first_day
        FROM ib
        JOIN cov ON cov.symbol = ib.symbol
        JOIN mc  ON mc.symbol = ib.symbol
        WHERE mc.market_cap >= %(min_mcap)s
        """).format(metrics=sql.Identifier(cfg.metrics_table),
                    symbols=sql.Identifier(cfg.symbols_table)),
        {"warmup_start": warmup_start, "end": cfg.end_date,
         "mcap_asof": mcap_asof, "min_mcap": cfg.min_market_cap_usd},
    )
    if df.empty:
        raise RuntimeError("universe query returned no rows")
    if cfg.categories:
        df = df[df["ibkr_category"].isin(cfg.categories)]
    # enough history: relative to the best-covered stock in the window
    max_rows = int(df["n_rows"].max())
    df = df[(df["n_rows"] >= max_rows * cfg.min_coverage_pct / 100.0)
            & (df["first_day"] <= cfg.start_date)]
    df = df.sort_values(["ibkr_category", "market_cap"], ascending=[True, False])
    if cfg.top_n_per_category > 0:
        df = df.groupby("ibkr_category", sort=False).head(cfg.top_n_per_category)
    df = df.reset_index(drop=True)
    if df.empty:
        raise RuntimeError("universe empty after coverage/market-cap filters")
    log.info("universe: %d stocks across %d categories",
             len(df), df["ibkr_category"].nunique())
    return df[["symbol", "ibkr_category", "market_cap"]]


def load_prices(conn, cfg: Config, symbols: list[str]) -> pd.DataFrame:
    """Wide frame of split-adjusted closes: index=day, columns=symbol.

    Days with no bar for a symbol are NaN (handled downstream via ffill).
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    df = read_df(
        conn,
        sql.SQL("""
        SELECT symbol, period_end_date AS day, adjusted_close::float8 AS close,
               price_continuity_segment
        FROM {}
        WHERE symbol = ANY(%(symbols)s)
          AND period_end_date BETWEEN %(from)s AND %(to)s
          AND adjusted_close > 0
        ORDER BY period_end_date
        """).format(sql.Identifier(cfg.metrics_table)),
        {"symbols": symbols, "from": warmup_start, "to": cfg.end_date},
    )
    if df.empty:
        raise RuntimeError("no price rows for the selected universe")
    if df["price_continuity_segment"].isna().any():
        missing = sorted(df.loc[df["price_continuity_segment"].isna(), "symbol"].unique())
        raise RuntimeError(
            "market data lacks price continuity segments for: "
            + ", ".join(missing[:20])
        )
    segment_counts = df.groupby("symbol")["price_continuity_segment"].nunique()
    broken_symbols = sorted(segment_counts[segment_counts > 1].index.tolist())
    if broken_symbols:
        log.warning(
            "Excluded %d symbols with price continuity breaks in the loaded window: %s",
            len(broken_symbols), ", ".join(broken_symbols[:20]),
        )
        df = df[~df["symbol"].isin(broken_symbols)]
    if df.empty:
        raise RuntimeError("no continuous price series remain after continuity validation")
    wide = df.pivot(index="day", columns="symbol", values="close").sort_index()
    log.info("loaded %d trading days x %d symbols", len(wide), wide.shape[1])
    return wide


def load_category_momentum(conn, cfg: Config) -> pd.DataFrame:
    """Rolling category momentum: index=day, columns=ibkr_category.

    The category index is the equal-weight average of the clipped daily returns
    of ALL category members with data (not only the traded universe), matching
    the research setup. Momentum is the CAT_MOM_WINDOW-day change of that index.
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    cat_filter = sql.SQL("AND ib.ibkr_category = ANY(%(cats)s)") if cfg.categories else sql.SQL("")
    df = read_df(
        conn,
        sql.SQL("""
        WITH ib AS (
            SELECT DISTINCT ON (source_symbol)
                   source_symbol AS symbol, ibkr_category
            FROM {symbols}
            WHERE ibkr_category IS NOT NULL
            ORDER BY source_symbol, fetched_at DESC
        ),
        rets AS (
            SELECT ib.ibkr_category, m.period_end_date AS day,
                   m.adjusted_close::float8
                   / NULLIF(lag(m.adjusted_close::float8) OVER (
                         PARTITION BY m.symbol, m.price_continuity_segment
                         ORDER BY m.period_end_date), 0) - 1 AS r
            FROM {metrics} m
            JOIN ib ON ib.symbol = m.symbol
            WHERE m.period_end_date BETWEEN %(from)s AND %(to)s
              AND m.adjusted_close > 0
              AND m.price_continuity_segment IS NOT NULL
              {cat_filter}
        )
        SELECT ibkr_category, day,
               avg(LEAST(GREATEST(r, -0.5), 1.0)) AS avg_ret
        FROM rets
        WHERE r IS NOT NULL
        GROUP BY ibkr_category, day
        ORDER BY day
        """).format(metrics=sql.Identifier(cfg.metrics_table),
                    symbols=sql.Identifier(cfg.symbols_table),
                    cat_filter=cat_filter),
        {"from": warmup_start, "to": cfg.end_date, "cats": list(cfg.categories)},
    )
    if df.empty:
        raise RuntimeError("category return query returned no rows")
    wide = (df.pivot(index="day", columns="ibkr_category", values="avg_ret")
              .sort_index().astype(float))
    log_index = np.log1p(wide.fillna(0.0)).cumsum()
    momentum = np.expm1(log_index - log_index.shift(cfg.cat_mom_window))
    log.info("category momentum for %d categories, %d days",
             momentum.shape[1], len(momentum))
    return momentum


def load_composite_scores(conn, cfg: Config) -> pd.DataFrame:
    """Daily composite stress score (calendar days, 0-100).

    Returns columns: day (date), composite_score (float), sorted ascending.
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    df = read_df(
        conn,
        sql.SQL("""
        SELECT day, composite_score::float8 AS composite_score
        FROM {}
        WHERE day BETWEEN %(from)s AND %(to)s
        ORDER BY day
        """).format(sql.Identifier(cfg.scores_table)),
        {"from": warmup_start, "to": cfg.end_date},
    )
    if df.empty:
        raise RuntimeError(f"no rows in {cfg.scores_table} for the requested window")
    return df


def lagged_scores_for_trading_days(days: pd.Index, scores: pd.DataFrame) -> np.ndarray:
    """Most recent composite score strictly BEFORE each trading day.

    The score for calendar day d is computed with an as-of cutoff at 05:00 UTC
    on d+1, so at the close of trading day t only the score of t-1 (or older)
    is guaranteed to exist. merge_asof with allow_exact_matches=False gives
    exactly that without look-ahead.
    """
    left = pd.DataFrame({"day": pd.to_datetime(days)})
    right = pd.DataFrame({
        "day": pd.to_datetime(scores["day"]),
        "composite_score": scores["composite_score"].astype(float),
    })
    merged = pd.merge_asof(left, right, on="day", allow_exact_matches=False)
    return merged["composite_score"].to_numpy(dtype=float)
