"""Signal generation for the 52-week-high pullback rule."""
from __future__ import annotations

import logging

import pandas as pd

from . import group_filter
from .config import Config

log = logging.getLogger(__name__)

SIGNAL_COLUMNS = [
    "period_end_date",
    "symbol",
    "ibkr_industry",
    "ibkr_category",
    "close",
    "high",
    "volume",
    "rolling_52w_high",
    "had_52w_high_last_10d",
    "ema_fast",
    "ema_slow",
    "prev_ema_fast",
    "prev_ema_slow",
    "volume_sma20",
    "volume_pass",
    "ibkr_industry_breadth_pct",
    "ibkr_industry_breadth_on",
    "ibkr_industry_breadth_pass",
    "entry_signal",
    "planned_entry_date",
    "planned_entry_open",
]


def compute_signals(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    cfg: Config,
    start,
    end,
) -> pd.DataFrame:
    """Return one row per actionable signal.

    Indicators are computed through the signal close. The trade entry is planned
    for the next available trading day's open to avoid same-day look-ahead.
    """
    if prices.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    close_m = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    industry_breadth = group_filter.compute_industry_breadth(close_m, universe, cfg)
    frames: list[pd.DataFrame] = []
    for symbol, sub in prices.groupby("symbol", sort=True):
        sub = sub.sort_values("date").copy()
        if len(sub) < max(cfg.high_lookback_days, cfg.ema_slow_days, cfg.volume_sma_days) + 1:
            continue

        high = sub["high"]
        close = sub["close"]
        volume = sub["volume"]
        rolling_high = high.rolling(
            cfg.high_lookback_days,
            min_periods=cfg.high_lookback_days,
        ).max()
        is_52w_high = (high >= rolling_high) & rolling_high.notna()
        had_high_recent = (
            is_52w_high.rolling(cfg.high_recent_days, min_periods=1)
            .max()
            .fillna(False)
            .astype(bool)
        )
        ema_fast = close.ewm(
            span=cfg.ema_fast_days,
            adjust=False,
            min_periods=cfg.ema_fast_days,
        ).mean()
        ema_slow = close.ewm(
            span=cfg.ema_slow_days,
            adjust=False,
            min_periods=cfg.ema_slow_days,
        ).mean()
        prev_ema_fast = ema_fast.shift(1)
        prev_ema_slow = ema_slow.shift(1)
        crossed_up = (prev_ema_fast <= prev_ema_slow) & (ema_fast > ema_slow)
        volume_sma = volume.rolling(
            cfg.volume_sma_days,
            min_periods=cfg.volume_sma_days,
        ).mean()
        volume_pass = volume > volume_sma
        volume_gate = volume_pass if cfg.volume_filter_enable else True
        dates = pd.DatetimeIndex(sub["date"])
        if symbol in industry_breadth["ibkr_industry_breadth_pass"].columns:
            industry_breadth_raw = industry_breadth["ibkr_industry_breadth"][symbol].reindex(dates)
            industry_breadth_on = industry_breadth["ibkr_industry_breadth_on"][symbol].reindex(dates).fillna(False)
            industry_breadth_pass = (
                industry_breadth["ibkr_industry_breadth_pass"][symbol]
                .reindex(dates)
                .fillna(False)
            )
        else:
            industry_breadth_raw = pd.Series(float("nan"), index=dates)
            industry_breadth_on = pd.Series(False, index=dates)
            industry_breadth_pass = pd.Series(not cfg.ibkr_industry_breadth_filter_enable, index=dates)

        sub["rolling_52w_high"] = rolling_high
        sub["had_52w_high_last_10d"] = had_high_recent
        sub["ema_fast"] = ema_fast
        sub["ema_slow"] = ema_slow
        sub["prev_ema_fast"] = prev_ema_fast
        sub["prev_ema_slow"] = prev_ema_slow
        sub["volume_sma20"] = volume_sma
        sub["volume_pass"] = volume_pass
        sub["ibkr_industry_breadth_pct"] = (industry_breadth_raw.to_numpy(dtype=float) * 100).round(4)
        sub["ibkr_industry_breadth_on"] = industry_breadth_on.to_numpy(dtype=bool)
        sub["ibkr_industry_breadth_pass"] = industry_breadth_pass.to_numpy(dtype=bool)
        sub["entry_signal"] = had_high_recent & crossed_up & volume_gate & sub["ibkr_industry_breadth_pass"]
        sub["planned_entry_date"] = sub["date"].shift(-1)
        sub["planned_entry_open"] = sub["open"].shift(-1)

        selected = sub[
            sub["entry_signal"]
            & (sub["date"].dt.date >= start)
            & (sub["date"].dt.date <= end)
            & (pd.to_datetime(sub["planned_entry_date"]).dt.date <= end)
            & (sub["close"] >= cfg.min_price)
            & sub["planned_entry_open"].notna()
            & (sub["planned_entry_open"] >= cfg.min_price)
        ].copy()
        if selected.empty:
            continue
        selected["period_end_date"] = selected["date"].dt.date
        selected["planned_entry_date"] = pd.to_datetime(selected["planned_entry_date"]).dt.date
        selected["symbol"] = symbol
        frames.append(selected)

    if not frames:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    signals = pd.concat(frames, ignore_index=True)
    signals = signals.merge(
        universe[["symbol", "ibkr_industry", "ibkr_category"]],
        on="symbol",
        how="left",
    )
    signals = signals[SIGNAL_COLUMNS].sort_values(["period_end_date", "symbol"])
    log.info("computed %d entry signals across %d symbols", len(signals), signals["symbol"].nunique())
    return signals.reset_index(drop=True)
