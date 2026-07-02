from __future__ import annotations

from datetime import date
from statistics import mean
from typing import Any


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def component_score(value: float | None, low: float, high: float, inverse: bool = False) -> float | None:
    if value is None or high <= low:
        return None
    score = (value - low) / (high - low) * 100.0
    score = clamp(score, 0.0, 100.0)
    return 100.0 - score if inverse else score


def moving_average(values: list[float | None], index: int, window: int, min_obs: int) -> float | None:
    start = max(0, index - window + 1)
    subset = [v for v in values[start:index + 1] if v is not None and v > 0]
    if len(subset) < min_obs:
        return None
    return mean(subset)


def rolling_max(values: list[float | None], index: int, window: int, min_obs: int) -> float | None:
    start = max(0, index - window + 1)
    subset = [v for v in values[start:index + 1] if v is not None and v > 0]
    if len(subset) < min_obs:
        return None
    return max(subset)


def lagged_return(values: list[float | None], index: int, lag: int) -> float | None:
    if index - lag < 0:
        return None
    current = values[index]
    prior = values[index - lag]
    if current is None or prior is None or current <= 0 or prior <= 0:
        return None
    return current / prior - 1.0


def true_range(row: dict[str, Any], previous_close: float | None) -> float | None:
    high = safe_float(row.get("high"))
    low = safe_float(row.get("low"))
    if high is None or low is None or high <= 0 or low <= 0 or high < low:
        return None
    if previous_close is None or previous_close <= 0:
        return high - low
    return max(high - low, abs(high - previous_close), abs(low - previous_close))


def average_true_range(rows: list[dict[str, Any]], index: int, window: int = 14) -> float | None:
    start = max(0, index - window + 1)
    values: list[float] = []
    for idx in range(start, index + 1):
        previous_close = safe_float(rows[idx - 1].get("close")) if idx > 0 else None
        tr = true_range(rows[idx], previous_close)
        if tr is not None and tr > 0:
            values.append(tr)
    if len(values) < max(5, window // 2):
        return None
    return mean(values)


def quality_score(row: dict[str, Any]) -> float | None:
    fcf_margin = safe_float(row.get("sec_fcf_sbc_adjusted_margin_ttm"))
    if fcf_margin is None:
        fcf_margin = safe_float(row.get("sec_fcf_margin_ttm"))

    raw_components = [
        component_score(safe_float(row.get("sec_gross_margin_ttm")), 0.20, 0.60),
        component_score(safe_float(row.get("sec_operating_margin_ttm")), 0.05, 0.30),
        component_score(fcf_margin, 0.03, 0.25),
        component_score(safe_float(row.get("sec_debt_to_capital")), 0.20, 0.80, inverse=True),
        component_score(safe_float(row.get("sec_cash_to_assets")), 0.03, 0.30),
        component_score(safe_float(row.get("sec_current_ratio")), 0.80, 2.00),
        component_score(abs(safe_float(row.get("sec_accruals_ratio")) or 0.0), 0.02, 0.20, inverse=True),
        component_score(safe_float(row.get("revenue_growth")), -0.05, 0.20),
        component_score(safe_float(row.get("earnings_growth")), -0.10, 0.30),
    ]
    components = [score for score in raw_components if score is not None]
    if len(components) < 4:
        return None
    return mean(components)


def momentum_score(row: dict[str, Any]) -> float | None:
    close = safe_float(row.get("close"))
    sma50 = safe_float(row.get("sma_50")) or safe_float(row.get("fifty_day_average"))
    sma200 = safe_float(row.get("sma_200")) or safe_float(row.get("two_hundred_day_average"))
    dist50 = close / sma50 - 1.0 if close and sma50 and sma50 > 0 else None
    dist200 = close / sma200 - 1.0 if close and sma200 and sma200 > 0 else None
    raw_components = [
        component_score(safe_float(row.get("ret_63")), -0.03, 0.15),
        component_score(safe_float(row.get("ret_126")), -0.05, 0.25),
        component_score(safe_float(row.get("ret_252")) or safe_float(row.get("week_52_change")), -0.10, 0.40),
        component_score(dist50, -0.02, 0.12),
        component_score(dist200, 0.00, 0.25),
    ]
    components = [score for score in raw_components if score is not None]
    if len(components) < 3:
        return None
    return mean(components)


def enrich_price_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    closes = [safe_float(row.get("close")) for row in rows]
    volumes = [safe_float(row.get("volume")) for row in rows]
    for index, row in enumerate(rows):
        row["sma_10"] = moving_average(closes, index, 10, 7)
        row["sma_20"] = moving_average(closes, index, 20, 14)
        row["sma_50"] = safe_float(row.get("fifty_day_average")) or moving_average(closes, index, 50, 40)
        row["sma_200"] = safe_float(row.get("two_hundred_day_average")) or moving_average(closes, index, 200, 160)
        row["avg_volume_20"] = moving_average(volumes, index, 20, 10)
        row["rolling_high_252"] = rolling_max(closes, index, 252, 180)
        row["ret_5"] = lagged_return(closes, index, 5)
        row["ret_10"] = lagged_return(closes, index, 10)
        row["ret_20"] = lagged_return(closes, index, 20)
        row["ret_63"] = lagged_return(closes, index, 63)
        row["ret_126"] = lagged_return(closes, index, 126)
        row["ret_252"] = lagged_return(closes, index, 252)
        row["atr_14"] = average_true_range(rows, index, 14)
        row["quality_score"] = quality_score(row)
        row["momentum_score"] = momentum_score(row)
        close = safe_float(row.get("close"))
        high_252 = safe_float(row.get("rolling_high_252"))
        row["distance_from_252d_high"] = close / high_252 - 1.0 if close and high_252 and high_252 > 0 else None
        row["day_index"] = index
    return rows


def first_index_on_or_after(days: list[date], target: date) -> int | None:
    for index, day in enumerate(days):
        if day >= target:
            return index
    return None


def first_index_after(days: list[date], target: date) -> int | None:
    for index, day in enumerate(days):
        if day > target:
            return index
    return None
