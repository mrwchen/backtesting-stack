"""Minervini's 8-point trend template, vectorized over date x symbol matrices.

1. Close above the 150d and 200d MA
2. 150d MA above 200d MA
3. 200d MA rising for at least ~1 month
4. 50d MA above both the 150d and 200d MA
5. Close above the 50d MA
6. Close at least 30% above the 52-week low
7. Close within 25% of the 52-week high
8. RS rating >= threshold
"""
from __future__ import annotations

import pandas as pd

from .config import Config


def compute_template(
    close: pd.DataFrame,
    rs_rating: pd.DataFrame,
    cfg: Config,
    *,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> dict:
    ma50 = close.rolling(50, min_periods=50).mean()
    ma150 = close.rolling(150, min_periods=150).mean()
    ma200 = close.rolling(200, min_periods=200).mean()
    low52 = low.rolling(252, min_periods=200).min()
    high52 = high.rolling(252, min_periods=200).max()

    crit = {
        "crit_price_above_ma150_200": (close > ma150) & (close > ma200),
        "crit_ma150_above_ma200": ma150 > ma200,
        "crit_ma200_rising": ma200 > ma200.shift(cfg.ma200_trend_days),
        "crit_ma50_above_ma150_200": (ma50 > ma150) & (ma50 > ma200),
        "crit_price_above_ma50": close > ma50,
        "crit_above_52w_low": close >= low52 * cfg.min_above_52w_low,
        "crit_near_52w_high": close >= high52 * cfg.max_below_52w_high,
        "crit_rs_rating": rs_rating >= cfg.rs_min,
    }

    template_pass = None
    for matrix in crit.values():
        template_pass = matrix if template_pass is None else template_pass & matrix

    return {**crit, "template_pass": template_pass, "ma50": ma50}
