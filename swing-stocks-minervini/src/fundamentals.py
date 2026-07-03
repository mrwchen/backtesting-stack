"""Point-in-time fundamental flags on the daily grid.

All inputs come from stock_core_sec_fundamentals_asof_daily, whose
period_end_date (loaded as available_date) is the date from which a filing is
usable — i.e. genuine point-in-time, no look-ahead.

EPS:      the earnings calendar carries no reported EPS in practice, so
          quarterly net income is reconstructed as the difference between
          consecutive quarterly TTM filings (gap 60-130 days), divided by
          diluted shares; YoY compares against 4 quarters earlier.
Revenue:  SEC TTM revenue YoY.
Margin:   SEC TTM net margin vs. one year earlier.

All values are forward-filled with a staleness limit so that a company that
stops reporting loses its flags after roughly two missed quarters.
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


def eps_flags(
    filings: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (eps_pass bool matrix, eps_yoy float matrix).

    Quarterly EPS = (TTM net income diff between consecutive quarterly filings)
    / diluted shares. Annual-only filers (gap > 130 days) never qualify."""
    f = filings.dropna(subset=["net_income_ttm", "available_date"]).copy()
    f = f.sort_values(["symbol", "available_date"]).drop_duplicates(
        ["symbol", "available_date"], keep="last"
    )

    grouped = f.groupby("symbol")
    filing_gap = grouped["available_date"].diff().dt.days
    quarterly_ni = grouped["net_income_ttm"].diff()
    quarterly_ni[~filing_gap.between(60, 130)] = np.nan
    f["eps_q"] = np.where(f["shares_diluted"] > 0, quarterly_ni / f["shares_diluted"], np.nan)

    f["prev_eps_q"] = grouped["eps_q"].shift(4)
    f["prev_available"] = grouped["available_date"].shift(4)
    yoy_gap = (f["available_date"] - f["prev_available"]).dt.days
    valid_prev = yoy_gap.between(300, 430) & f["prev_eps_q"].notna()

    eps, prev = f["eps_q"], f["prev_eps_q"]
    f["eps_yoy"] = np.where((prev > 0) & eps.notna(), eps / prev - 1.0, np.nan)
    f["eps_pass"] = np.where(
        (eps > 0) & valid_prev,
        np.where(prev > 0, f["eps_yoy"] >= cfg.eps_yoy_min, True),  # turnaround counts
        False,
    ).astype(float)

    pass_matrix = _event_matrix(f, "eps_pass", dates, symbols, cfg.eps_stale_trading_days)
    yoy_matrix = _event_matrix(f, "eps_yoy", dates, symbols, cfg.eps_stale_trading_days)
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
