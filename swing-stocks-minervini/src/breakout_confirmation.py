"""Causal daily breakout-volume confirmation for Minervini entries.

A setup's first pivot break in session D consumes the setup. Complete volume
and close data from D can only confirm the signal after that session closes, so
the returned entry candidate is valid exclusively in the next global session.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from .config import Config


EVENT_COLUMNS = [
    "setup_id", "symbol", "setup_detect_date", "breakout_date",
    "planned_entry_date",
    "pivot", "trigger_price", "breakout_open", "breakout_high", "breakout_low",
    "breakout_close", "breakout_volume", "average_volume_prior",
    "volume_history_sessions", "breakout_volume_ratio", "close_above_pivot",
    "volume_confirmed", "confirmation_pass", "entry_filled", "entry_date",
    "entry_price", "decision",
]


def _number(value: object, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def confirm_daily_breakouts(
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    open_m: pd.DataFrame,
    high_m: pd.DataFrame,
    low_m: pd.DataFrame,
    close_m: pd.DataFrame,
    volume_m: pd.DataFrame,
    setups: pd.DataFrame,
    cfg: Config,
    *,
    start_idx: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return D+1-only confirmed setups and every evaluated breakout event."""
    col_index = {symbol: index for index, symbol in enumerate(symbols)}
    source = setups[setups["symbol"].isin(col_index)].copy()
    source["_start_idx"] = dates.searchsorted(pd.to_datetime(source["detect_date"])) + 1
    source["_end_idx"] = (
        dates.searchsorted(pd.to_datetime(source["valid_until"]), side="right") - 1
    )
    sort_columns = ["_start_idx", "symbol", "detect_date"]
    if "setup_id" in source.columns:
        sort_columns.append("setup_id")
    rows = [
        SimpleNamespace(**record)
        for record in source.sort_values(sort_columns, na_position="first").to_dict("records")
    ]

    o = open_m.to_numpy(dtype=float)
    h = high_m.to_numpy(dtype=float)
    lo = low_m.to_numpy(dtype=float)
    c = close_m.to_numpy(dtype=float)
    volume = volume_m.to_numpy(dtype=float)
    active: dict[str, object] = {}
    confirmed: list[dict] = []
    events: list[dict] = []
    next_setup = 0

    for t in range(start_idx, len(dates)):
        while next_setup < len(rows) and rows[next_setup]._start_idx <= t:
            setup = rows[next_setup]
            active[setup.symbol] = setup
            next_setup += 1
        active = {
            symbol: setup
            for symbol, setup in active.items()
            if setup._end_idx >= t
        }

        for symbol, setup in list(active.items()):
            col = col_index[symbol]
            day_prices = np.asarray([o[t, col], h[t, col], lo[t, col], c[t, col]])
            if not np.isfinite(day_prices).all():
                del active[symbol]
                continue

            pivot = _number(setup.pivot)
            trigger = pivot * (1 + cfg.pivot_buffer_pct)
            invalidation = max(
                _number(getattr(setup, "last_low", None), -np.inf),
                _number(getattr(setup, "stop_level", None), -np.inf),
            )
            breakout = np.isfinite(trigger) and h[t, col] > trigger
            damaged = np.isfinite(invalidation) and lo[t, col] <= invalidation
            if not breakout:
                if damaged:
                    del active[symbol]
                continue

            prior = volume[max(0, t - cfg.breakout_volume_lookback_sessions) : t, col]
            prior = prior[np.isfinite(prior) & (prior > 0)]
            average_prior = float(np.mean(prior)) if len(prior) else np.nan
            day_volume = volume[t, col]
            volume_ratio = (
                float(day_volume / average_prior)
                if np.isfinite(day_volume) and day_volume > 0 and average_prior > 0
                else np.nan
            )
            close_above_pivot = bool(c[t, col] > pivot)
            volume_confirmed = bool(
                np.isfinite(volume_ratio)
                and volume_ratio >= cfg.breakout_volume_min_ratio
            )
            enough_history = len(prior) >= cfg.breakout_volume_min_history_sessions
            planned_entry_date = dates[t + 1].date() if t + 1 < len(dates) else None

            if damaged:
                decision = "base_invalidated_on_breakout"
            elif not np.isfinite(day_volume) or day_volume <= 0:
                decision = "missing_breakout_volume"
            elif not enough_history:
                decision = "insufficient_volume_history"
            elif cfg.breakout_require_close_above_pivot and not close_above_pivot:
                decision = "close_below_pivot"
            elif not volume_confirmed:
                decision = "volume_below_threshold"
            elif planned_entry_date is None:
                decision = "no_next_session"
            else:
                decision = "confirmed"

            confirmation_pass = decision == "confirmed"
            setup_id = getattr(setup, "setup_id", None)
            event = {
                "setup_id": setup_id,
                "symbol": symbol,
                "setup_detect_date": pd.Timestamp(setup.detect_date).date(),
                "breakout_date": dates[t].date(),
                "planned_entry_date": planned_entry_date,
                "pivot": pivot,
                "trigger_price": trigger,
                "breakout_open": o[t, col],
                "breakout_high": h[t, col],
                "breakout_low": lo[t, col],
                "breakout_close": c[t, col],
                "breakout_volume": day_volume if np.isfinite(day_volume) else None,
                "average_volume_prior": average_prior if np.isfinite(average_prior) else None,
                "volume_history_sessions": int(len(prior)),
                "breakout_volume_ratio": volume_ratio if np.isfinite(volume_ratio) else None,
                "close_above_pivot": close_above_pivot,
                "volume_confirmed": volume_confirmed,
                "confirmation_pass": confirmation_pass,
                "entry_filled": False,
                "entry_date": None,
                "entry_price": None,
                "decision": decision,
            }
            events.append(event)

            if confirmation_pass:
                candidate = {
                    key: value
                    for key, value in vars(setup).items()
                    if key not in {"_start_idx", "_end_idx"}
                }
                candidate["detect_date"] = dates[t].date()
                candidate["valid_until"] = planned_entry_date
                candidate["breakout_date"] = dates[t].date()
                candidate["breakout_volume_ratio"] = volume_ratio
                confirmed.append(candidate)
            del active[symbol]

    confirmed_df = pd.DataFrame(confirmed)
    if confirmed_df.empty:
        confirmed_df = setups.iloc[0:0].copy()
    return confirmed_df, pd.DataFrame(events, columns=EVENT_COLUMNS)


def attach_fills(
    events: pd.DataFrame,
    trades: pd.DataFrame,
    entry_decisions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach actual fills after portfolio/gap constraints have been simulated."""
    if events.empty:
        return events
    result = events.copy()
    if entry_decisions is not None and not entry_decisions.empty:
        decisions = entry_decisions.set_index("setup_id")
        for index, setup_id in result.loc[result["confirmation_pass"], "setup_id"].items():
            if setup_id in decisions.index:
                entry = decisions.loc[setup_id]
                result.at[index, "decision"] = entry["entry_decision"]
                if entry["entry_decision"] == "filled":
                    result.at[index, "entry_filled"] = True
                    result.at[index, "entry_date"] = entry["entry_date"]
                    result.at[index, "entry_price"] = entry["entry_price"]
    if not trades.empty:
        fills = (
            trades.sort_values(["entry_date", "position_id"], kind="stable")
            .drop_duplicates("setup_id", keep="first")
            .set_index("setup_id")
        )
        for index, setup_id in result["setup_id"].items():
            if setup_id in fills.index:
                fill = fills.loc[setup_id]
                result.at[index, "entry_filled"] = True
                result.at[index, "entry_date"] = fill["entry_date"]
                result.at[index, "entry_price"] = fill["entry_price"]
                result.at[index, "decision"] = "filled"
    confirmed_without_decision = (
        result["confirmation_pass"] & result["decision"].eq("confirmed")
    )
    result.loc[confirmed_without_decision, "decision"] = "confirmed_not_filled"
    return result
