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
from .rs_rating import (
    require_aligned_matrix,
    rolling_within_continuity,
    same_continuity_segment,
    validate_continuity_matrices,
)


def compute_template(
    close: pd.DataFrame,
    rs_rating: pd.DataFrame,
    cfg: Config,
    *,
    high: pd.DataFrame,
    low: pd.DataFrame,
    continuity_segment: pd.DataFrame,
    continuity_break: pd.DataFrame,
) -> dict:
    require_aligned_matrix(close, rs_rating, "rs_rating")
    require_aligned_matrix(close, high, "high")
    require_aligned_matrix(close, low, "low")
    validate_continuity_matrices(close, continuity_segment, continuity_break)

    ma50 = rolling_within_continuity(
        close, continuity_segment, 50, min_periods=50, operation="mean"
    )
    ma150 = rolling_within_continuity(
        close, continuity_segment, 150, min_periods=150, operation="mean"
    )
    ma200 = rolling_within_continuity(
        close, continuity_segment, 200, min_periods=200, operation="mean"
    )
    low52 = rolling_within_continuity(
        low, continuity_segment, 252, min_periods=200, operation="min"
    )
    high52 = rolling_within_continuity(
        high, continuity_segment, 252, min_periods=200, operation="max"
    )

    crit = {
        "crit_price_above_ma150_200": (close > ma150) & (close > ma200),
        "crit_ma150_above_ma200": ma150 > ma200,
        "crit_ma200_rising": (
            ma200 > ma200.shift(cfg.ma200_trend_days)
        ) & same_continuity_segment(
            continuity_segment, 0, cfg.ma200_trend_days
        ),
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
