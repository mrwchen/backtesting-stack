"""Point-in-time revenue growth filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _event_matrix(
    events: pd.DataFrame,
    value_col: str,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    stale_limit: int,
) -> pd.DataFrame:
    col_index = {symbol: i for i, symbol in enumerate(symbols)}
    arr = np.full((len(dates), len(symbols)), np.nan)

    ev = events.dropna(subset=[value_col, "available_date"])
    ev = ev[ev["symbol"].isin(col_index)].sort_values("available_date")
    pos = dates.searchsorted(ev["available_date"].to_numpy())
    keep = pos < len(dates)
    for p, sym, value in zip(
        pos[keep],
        ev["symbol"].to_numpy()[keep],
        ev[value_col].to_numpy()[keep],
    ):
        arr[p, col_index[sym]] = value

    return pd.DataFrame(arr, index=dates, columns=symbols).ffill(limit=stale_limit)


def _with_revenue_yoy(fundamentals: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    f = fundamentals.dropna(subset=["symbol", "available_date", "revenue_ttm"]).copy()
    if f.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "available_date",
                "revenue_ttm",
                "prev_revenue_ttm",
                "revenue_yoy",
                "revenue_pass_raw",
            ]
        )

    f["symbol"] = f["symbol"].astype(str).str.upper().str.strip()
    f["available_date"] = pd.to_datetime(f["available_date"])
    f["revenue_ttm"] = pd.to_numeric(f["revenue_ttm"], errors="coerce")
    f = f.sort_values(["symbol", "available_date"]).drop_duplicates(
        ["symbol", "available_date"], keep="last"
    )

    for _, sub in f.groupby("symbol", sort=True):
        sub = sub.sort_values("available_date").copy()
        prior_values: list[float] = []
        prior_dates: list[pd.Timestamp | pd.NaT] = []
        for row in sub.itertuples(index=False):
            low = row.available_date - pd.Timedelta(days=430)
            high = row.available_date - pd.Timedelta(days=300)
            candidates = sub[
                (sub["available_date"] >= low)
                & (sub["available_date"] <= high)
                & (sub["available_date"] < row.available_date)
            ]
            if candidates.empty:
                prior_values.append(np.nan)
                prior_dates.append(pd.NaT)
            else:
                prior = candidates.iloc[-1]
                prior_values.append(float(prior["revenue_ttm"]))
                prior_dates.append(prior["available_date"])
        sub["prev_revenue_ttm"] = prior_values
        sub["prev_revenue_available_date"] = prior_dates
        prev = sub["prev_revenue_ttm"]
        current = sub["revenue_ttm"]
        sub["revenue_yoy"] = current.div(prev.where(prev > 0)).sub(1.0)
        sub["revenue_yoy"] = sub["revenue_yoy"].where(np.isfinite(sub["revenue_yoy"]))
        rows.append(sub)

    return pd.concat(rows, ignore_index=True)


def compute_revenue_growth(
    fundamentals: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    cfg: Config,
) -> dict[str, pd.DataFrame]:
    """Return point-in-time revenue YoY diagnostics on the daily price grid."""
    f = _with_revenue_yoy(fundamentals)
    if f.empty:
        empty_bool = pd.DataFrame(False, index=dates, columns=symbols)
        empty_num = pd.DataFrame(np.nan, index=dates, columns=symbols)
        return {
            "revenue_pass": empty_bool,
            "revenue_yoy": empty_num,
            "revenue_ttm": empty_num,
            "prev_revenue_ttm": empty_num,
        }

    f["revenue_pass_raw"] = (
        (f["revenue_ttm"] > 0)
        & (f["prev_revenue_ttm"] > 0)
        & (f["revenue_yoy"] >= cfg.revenue_yoy_min)
    ).astype(float)

    revenue_pass = _event_matrix(
        f, "revenue_pass_raw", dates, symbols, cfg.revenue_stale_trading_days
    ) == 1.0
    revenue_yoy = _event_matrix(
        f, "revenue_yoy", dates, symbols, cfg.revenue_stale_trading_days
    )
    revenue_ttm = _event_matrix(
        f, "revenue_ttm", dates, symbols, cfg.revenue_stale_trading_days
    )
    prev_revenue_ttm = _event_matrix(
        f, "prev_revenue_ttm", dates, symbols, cfg.revenue_stale_trading_days
    )
    return {
        "revenue_pass": revenue_pass,
        "revenue_yoy": revenue_yoy,
        "revenue_ttm": revenue_ttm,
        "prev_revenue_ttm": prev_revenue_ttm,
    }
