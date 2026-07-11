"""Point-in-time fundamental flags on the daily grid.

Diluted quarterly EPS comes from accession-keyed SEC filing events. Revenue
and margin come from the SEC as-of snapshot table. Effective dates are the
first UTC calendar dates on which the filing is usable, so no later filing is
visible to an earlier screen.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


def _event_matrix(
    events: pd.DataFrame,
    value_col: str,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    stale_limit: int,
) -> pd.DataFrame:
    """Scatter (symbol, available_date, value) events onto the daily grid and
    forward-fill each column up to stale_limit trading days."""
    col_index = {s: i for i, s in enumerate(symbols)}
    arr = np.full((len(dates), len(symbols)), np.nan)

    ev = events.dropna(subset=[value_col, "available_date"])
    ev = ev[ev["symbol"].isin(col_index)].sort_values("available_date")
    pos = dates.searchsorted(ev["available_date"].to_numpy())
    keep = pos < len(dates)
    for p, sym, val in zip(pos[keep], ev["symbol"].to_numpy()[keep], ev[value_col].to_numpy()[keep]):
        arr[p, col_index[sym]] = val  # sorted by date: later events win

    return pd.DataFrame(arr, index=dates, columns=symbols).ffill(limit=stale_limit)


def _event_matrix_with_null_resets(
    events: pd.DataFrame,
    value_col: str,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    stale_limit: int,
) -> pd.DataFrame:
    """Forward-fill events while treating an explicit NULL as a state reset."""
    col_index = {s: i for i, s in enumerate(symbols)}
    values = np.full((len(dates), len(symbols)), np.nan)
    event_pos = np.full((len(dates), len(symbols)), -1, dtype=np.int64)

    ev = events.dropna(subset=["available_date"])
    ev = ev[ev["symbol"].isin(col_index)].sort_values("available_date", kind="stable")
    pos = dates.searchsorted(ev["available_date"].to_numpy())
    keep = pos < len(dates)
    for p, sym, val in zip(
        pos[keep],
        ev["symbol"].to_numpy()[keep],
        ev[value_col].to_numpy()[keep],
    ):
        col = col_index[sym]
        event_pos[p, col] = p
        values[p, col] = val

    last_event = np.maximum.accumulate(event_pos, axis=0)
    out = np.full_like(values, np.nan)
    rows, cols = np.where(last_event >= 0)
    source_rows = last_event[rows, cols]
    fresh = rows - source_rows <= stale_limit
    out[rows[fresh], cols[fresh]] = values[source_rows[fresh], cols[fresh]]
    return pd.DataFrame(out, index=dates, columns=symbols)


def eps_flags(
    events: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return diluted-EPS growth pass and YoY matrices from SEC events."""
    f = events.copy()
    eps = pd.to_numeric(f["diluted_eps"], errors="coerce")
    prev = pd.to_numeric(f["prior_year_diluted_eps"], errors="coerce")
    comparable = eps.notna() & prev.notna()
    positive_growth_base = comparable & (prev > 0)
    turnaround = comparable & (eps > 0) & (prev <= 0)

    f["eps_yoy"] = np.where(positive_growth_base, eps / prev - 1.0, np.nan)
    f["eps_pass"] = (
        (eps > 0)
        & (
            turnaround
            | (positive_growth_base & (f["eps_yoy"] >= cfg.eps_yoy_min))
        )
    ).astype(float)

    pass_matrix = _event_matrix_with_null_resets(
        f, "eps_pass", dates, symbols, cfg.eps_stale_trading_days
    )
    yoy_matrix = _event_matrix_with_null_resets(
        f, "eps_yoy", dates, symbols, cfg.eps_stale_trading_days
    )
    return pass_matrix == 1.0, yoy_matrix


def revenue_margin_flags(
    fundamentals: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (revenue_pass, revenue_yoy, margin_pass) matrices."""
    revenue = _event_matrix(
        fundamentals, "revenue_ttm", dates, symbols, cfg.filing_stale_trading_days
    )
    margin = _event_matrix(
        fundamentals, "net_margin_ttm", dates, symbols, cfg.filing_stale_trading_days
    )

    prev_revenue = revenue.shift(252)
    revenue_yoy = revenue.div(prev_revenue.where(prev_revenue > 0)).sub(1.0)
    revenue_yoy = revenue_yoy.where(np.isfinite(revenue_yoy))
    revenue_pass = (revenue > 0) & (prev_revenue > 0) & (revenue_yoy >= cfg.revenue_yoy_min)
    margin_pass = margin > margin.shift(252)
    return revenue_pass, revenue_yoy, margin_pass


def combine(
    eps_pass: pd.DataFrame,
    revenue_pass: pd.DataFrame,
    margin_pass: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    count = eps_pass.astype(int) + revenue_pass.astype(int) + margin_pass.astype(int)
    return count >= cfg.fundamentals_min_pass
