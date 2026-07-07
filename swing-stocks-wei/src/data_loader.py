"""Load daily closes and world-regime composite scores from TimescaleDB."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

from .config import Config
from .db import read_df

log = logging.getLogger(__name__)


def load_prices(conn, cfg: Config) -> pd.DataFrame:
    """Split-adjusted daily closes, including EMA warmup before start_date.

    Returns columns: day (date), close (float), sorted ascending.
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    df = read_df(
        conn,
        """
        SELECT ts::date AS day, close::float8 AS close
        FROM alpaca_market_data_1day
        WHERE symbol = %(symbol)s AND ts::date BETWEEN %(from)s AND %(to)s
        ORDER BY ts
        """,
        {"symbol": cfg.symbol, "from": warmup_start, "to": cfg.end_date},
    )
    if df.empty:
        raise RuntimeError(f"no price rows for {cfg.symbol} in alpaca_market_data_1day")
    first_eval = df[df["day"] >= cfg.start_date]
    if first_eval.empty:
        raise RuntimeError(f"no price rows on/after START_DATE {cfg.start_date}")
    log.info("loaded %d closes for %s (%s .. %s, warmup from %s)",
             len(df), cfg.symbol, df["day"].iloc[0], df["day"].iloc[-1], warmup_start)
    return df


def load_composite_scores(conn, cfg: Config) -> pd.DataFrame:
    """Daily composite stress score (calendar days, 0-100).

    Returns columns: day (date), composite_score (float), sorted ascending.
    """
    warmup_start = cfg.start_date - timedelta(days=cfg.warmup_calendar_days)
    df = read_df(
        conn,
        """
        SELECT day, composite_score::float8 AS composite_score
        FROM world_regime_daily_scores_mv
        WHERE day BETWEEN %(from)s AND %(to)s
        ORDER BY day
        """,
        {"from": warmup_start, "to": cfg.end_date},
    )
    if df.empty:
        raise RuntimeError("no rows in world_regime_daily_scores_mv for the requested window")
    return df


def lagged_scores_for_trading_days(prices: pd.DataFrame, scores: pd.DataFrame) -> pd.Series:
    """Most recent composite score strictly BEFORE each trading day.

    The score for calendar day d is computed with an as-of cutoff at 05:00 UTC
    on d+1, so at the close of trading day t only the score of t-1 (or older)
    is guaranteed to exist. merge_asof with allow_exact_matches=False gives
    exactly that without look-ahead.
    """
    left = pd.DataFrame({"day": pd.to_datetime(prices["day"])})
    right = pd.DataFrame({
        "day": pd.to_datetime(scores["day"]),
        "composite_score": scores["composite_score"].astype(float),
    })
    merged = pd.merge_asof(left, right, on="day", allow_exact_matches=False)
    return merged["composite_score"]
