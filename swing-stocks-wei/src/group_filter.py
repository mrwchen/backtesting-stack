"""IBKR industry breadth filter copied from the Minervini stack."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _clean_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype("string").str.strip()
    return cleaned.where(cleaned.ne(""))


def _hysteresis(values: np.ndarray, on_threshold: float, off_threshold: float) -> np.ndarray:
    state = False
    out = np.zeros(len(values), dtype=bool)
    for i, value in enumerate(values):
        if np.isnan(value):
            state = False
        elif state:
            state = value >= off_threshold
        else:
            state = value >= on_threshold
        out[i] = state
    return out


def compute_industry_breadth(
    close: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: Config,
) -> dict[str, pd.DataFrame]:
    """Return per-symbol daily IBKR industry breadth and pass flags."""
    symbols = close.columns
    meta = universe.drop_duplicates("symbol").set_index("symbol").reindex(symbols)
    industry = _clean_labels(meta["ibkr_industry"])

    ma200 = close.rolling(200, min_periods=200).mean()
    eligible = (close >= cfg.min_price) & ma200.notna()
    above = (close > ma200) & eligible

    breadth_by_symbol = pd.DataFrame(np.nan, index=close.index, columns=symbols)
    on_by_symbol = pd.DataFrame(False, index=close.index, columns=symbols)
    grouped = industry.dropna().groupby(industry.dropna(), sort=True).groups

    for _, symbol_index in grouped.items():
        columns = [symbol for symbol in symbol_index if symbol in close.columns]
        if len(columns) < cfg.ibkr_industry_breadth_min_symbols:
            continue
        eligible_count = eligible.loc[:, columns].sum(axis=1)
        breadth = (
            above.loc[:, columns].sum(axis=1)
            / eligible_count.replace(0, np.nan)
        ).where(eligible_count >= cfg.ibkr_industry_breadth_min_symbols)
        gate = _hysteresis(
            breadth.to_numpy(dtype=float),
            cfg.ibkr_industry_breadth_on_threshold,
            cfg.ibkr_industry_breadth_off_threshold,
        )
        for symbol in columns:
            breadth_by_symbol[symbol] = breadth
            on_by_symbol[symbol] = gate

    pass_by_symbol = on_by_symbol.copy()
    if not cfg.ibkr_industry_breadth_filter_enable:
        pass_by_symbol = pd.DataFrame(True, index=close.index, columns=symbols)

    return {
        "ibkr_industry_breadth": breadth_by_symbol,
        "ibkr_industry_breadth_on": on_by_symbol,
        "ibkr_industry_breadth_pass": pass_by_symbol,
    }
