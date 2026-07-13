"""Causal quarterly growth, acceleration, margin and 13F sponsorship flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def _event_matrix_with_null_resets(
    events: pd.DataFrame,
    value_col: str,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    stale_limit: int,
) -> pd.DataFrame:
    col_index = {symbol: index for index, symbol in enumerate(symbols)}
    values = np.full((len(dates), len(symbols)), np.nan)
    event_pos = np.full((len(dates), len(symbols)), -1, dtype=np.int64)
    selected = events.dropna(subset=["available_date"])
    selected = selected[selected["symbol"].isin(col_index)].sort_values(
        "available_date", kind="stable"
    )
    positions = dates.searchsorted(selected["available_date"].to_numpy())
    keep = positions < len(dates)
    for position, symbol, value in zip(
        positions[keep], selected["symbol"].to_numpy()[keep],
        selected[value_col].to_numpy()[keep],
    ):
        column = col_index[symbol]
        event_pos[position, column] = position
        values[position, column] = value
    last_event = np.maximum.accumulate(event_pos, axis=0)
    output = np.full_like(values, np.nan)
    rows, columns = np.where(last_event >= 0)
    source_rows = last_event[rows, columns]
    fresh = rows - source_rows <= stale_limit
    output[rows[fresh], columns[fresh]] = values[source_rows[fresh], columns[fresh]]
    return pd.DataFrame(output, index=dates, columns=symbols)


def _safe_yoy(current: pd.Series, prior: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=current.index, dtype=float)
    positive = current.notna() & prior.notna() & (prior > 0)
    result.loc[positive] = current.loc[positive] / prior.loc[positive] - 1.0
    return result


def quarterly_flags(
    events: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, cfg: Config
) -> dict[str, pd.DataFrame]:
    frame = events.copy().sort_values(
        ["symbol", "available_date", "fiscal_period_end_date"], kind="stable"
    )
    eps = pd.to_numeric(frame["diluted_eps"], errors="coerce")
    prior_eps = pd.to_numeric(frame["prior_year_diluted_eps"], errors="coerce")
    revenue = pd.to_numeric(frame["quarterly_revenue"], errors="coerce")
    prior_revenue = pd.to_numeric(frame["prior_year_quarterly_revenue"], errors="coerce")
    frame["diluted_eps"] = eps
    frame["eps_yoy"] = _safe_yoy(eps, prior_eps)
    frame["revenue_yoy"] = _safe_yoy(revenue, prior_revenue)
    frame["eps_pass"] = (
        (eps > 0)
        & (((prior_eps <= 0) & prior_eps.notna()) | (frame["eps_yoy"] >= cfg.eps_yoy_min - 1e-12))
    ).astype(float)
    frame["revenue_pass"] = (
        (revenue > 0) & (frame["revenue_yoy"] >= cfg.revenue_yoy_min - 1e-12)
    ).astype(float)

    operating_margin = pd.to_numeric(frame["quarterly_operating_margin"], errors="coerce")
    prior_operating_margin = pd.to_numeric(
        frame["prior_year_quarterly_operating_margin"], errors="coerce"
    )
    net_margin = pd.to_numeric(frame["quarterly_net_margin"], errors="coerce")
    prior_net_margin = pd.to_numeric(frame["prior_year_quarterly_net_margin"], errors="coerce")
    frame["margin_delta"] = (operating_margin - prior_operating_margin).combine_first(
        net_margin - prior_net_margin
    )
    frame["margin_pass"] = (frame["margin_delta"] >= cfg.margin_expansion_min).astype(float)

    eps_acceleration = pd.Series(np.nan, index=frame.index, dtype=float)
    revenue_acceleration = pd.Series(np.nan, index=frame.index, dtype=float)
    streaks = pd.Series(0.0, index=frame.index, dtype=float)
    stability = pd.Series(np.nan, index=frame.index, dtype=float)
    for _symbol, indexes in frame.groupby("symbol", sort=False).groups.items():
        period_state: dict[pd.Timestamp, dict[str, float | bool]] = {}
        for index in indexes:
            period = pd.Timestamp(frame.at[index, "fiscal_period_end_date"])
            prior_periods = sorted(candidate for candidate in period_state if candidate < period)
            if prior_periods:
                prior = period_state[prior_periods[-1]]
                current_eps_yoy = frame.at[index, "eps_yoy"]
                current_revenue_yoy = frame.at[index, "revenue_yoy"]
                if pd.notna(current_eps_yoy) and pd.notna(prior["eps_yoy"]):
                    eps_acceleration.at[index] = float(current_eps_yoy) - float(prior["eps_yoy"])
                if pd.notna(current_revenue_yoy) and pd.notna(prior["revenue_yoy"]):
                    revenue_acceleration.at[index] = (
                        float(current_revenue_yoy) - float(prior["revenue_yoy"])
                    )
            period_state[period] = {
                "eps_yoy": frame.at[index, "eps_yoy"],
                "revenue_yoy": frame.at[index, "revenue_yoy"],
                "growth_pass": bool(
                    frame.at[index, "eps_pass"] == 1.0
                    and frame.at[index, "revenue_pass"] == 1.0
                ),
                "eps": frame.at[index, "diluted_eps"],
            }
            streak = 0
            for candidate in sorted(
                (candidate for candidate in period_state if candidate <= period), reverse=True
            ):
                if not period_state[candidate]["growth_pass"]:
                    break
                streak += 1
            streaks.at[index] = streak
            last_four = [
                period_state[candidate]["eps"]
                for candidate in sorted(
                    (candidate for candidate in period_state if candidate <= period), reverse=True
                )[:4]
            ]
            if len(last_four) == 4:
                stability.at[index] = float(
                    np.count_nonzero(np.asarray(last_four, dtype=float) > 0) >= 3
                )
    frame["eps_acceleration"] = eps_acceleration
    frame["revenue_acceleration"] = revenue_acceleration
    frame["acceleration_pass"] = (
        (eps_acceleration >= cfg.acceleration_min)
        | (revenue_acceleration >= cfg.acceleration_min)
    ).astype(float)
    frame["growth_streak"] = streaks
    frame["streak_pass"] = (streaks >= cfg.quarterly_growth_streak_min).astype(float)
    frame["stability_pass"] = stability

    matrix_columns = (
        "eps_pass", "revenue_pass", "margin_pass", "acceleration_pass",
        "streak_pass", "stability_pass", "eps_yoy", "revenue_yoy",
        "eps_acceleration", "revenue_acceleration", "margin_delta", "growth_streak",
    )
    result = {
        column: _event_matrix_with_null_resets(
            frame, column, dates, symbols, cfg.quarterly_fundamental_stale_trading_days
        )
        for column in matrix_columns
    }
    for column in (
        "eps_pass", "revenue_pass", "margin_pass", "acceleration_pass",
        "streak_pass", "stability_pass",
    ):
        result[column] = result[column] == 1.0
    score = sum(
        result[column].astype(int)
        for column in (
            "eps_pass", "revenue_pass", "margin_pass", "acceleration_pass",
            "streak_pass", "stability_pass",
        )
    )
    result["fundamental_score"] = score
    result["fundamentals_pass"] = score >= cfg.fundamentals_min_pass
    return result


def sponsorship_flags(
    events: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, cfg: Config
) -> dict[str, pd.DataFrame]:
    manager_delta = pd.DataFrame(0.0, index=dates, columns=symbols)
    activity_delta = pd.DataFrame(0.0, index=dates, columns=symbols)
    if not events.empty:
        symbol_pos = {symbol: index for index, symbol in enumerate(symbols)}
        positions = dates.searchsorted(events["available_date"].to_numpy())
        for position, row in zip(positions, events.itertuples(index=False)):
            if position >= len(dates) or row.symbol not in symbol_pos:
                continue
            manager_delta.iat[position, symbol_pos[row.symbol]] += float(row.manager_count_delta)
            activity_delta.iat[position, symbol_pos[row.symbol]] += float(row.net_activity_delta)
    manager_count = manager_delta.cumsum().clip(lower=0)
    net_activity = activity_delta.rolling(
        cfg.institutional_activity_lookback_sessions, min_periods=1
    ).sum()
    passed = (
        (manager_count >= cfg.institutional_min_managers)
        & (net_activity >= cfg.institutional_net_activity_min)
    )
    return {
        "institutional_manager_count": manager_count,
        "institutional_net_activity": net_activity,
        "institutional_sponsorship_pass": passed,
    }
