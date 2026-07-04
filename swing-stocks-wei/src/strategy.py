"""Signal generation for the 52-week-high pullback rule."""
from __future__ import annotations

import logging

import pandas as pd

from . import fundamentals as fundamental_filter
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
    "alpaca_price_feed",
    "entry_ref_market_cap_usd",
    "entry_ref_market_cap_currency",
    "market_cap_pass",
    "revenue_ttm",
    "prev_revenue_ttm",
    "revenue_yoy",
    "revenue_pass",
    "rolling_52w_high",
    "is_52w_high",
    "had_52w_high_recent",
    "last_52w_high_date",
    "last_52w_high_price",
    "bars_since_52w_high",
    "pullback_low_since_52w_high",
    "pullback_from_52w_high_pct",
    "pullback_pass",
    "atr",
    "ema_fast",
    "ema_slow",
    "prev_ema_fast",
    "prev_ema_slow",
    "ema_cross_up",
    "ema_cross_down",
    "ema_cross_recent",
    "ema_cross_delay_days",
    "ema_already_above_on_52w_high",
    "ema_entry_pass",
    "entry_gap_pct",
    "entry_gap_pass",
    "planned_entry_atr",
    "volume_sma50",
    "volume_sma50_pass",
    "volume_feed_pass",
    "volume_pass",
    "ibkr_category_breadth_pct",
    "ibkr_category_breadth_on",
    "ibkr_category_breadth_pass",
    "entry_signal",
    "planned_entry_date",
    "planned_entry_open",
]


def compute_signals(
    prices: pd.DataFrame,
    universe: pd.DataFrame,
    fundamentals: pd.DataFrame,
    cfg: Config,
    start,
    end,
) -> pd.DataFrame:
    """Return one row per actionable signal.

    Indicators are computed through the signal close. The trade entry is planned
    for a later trading day's open using only information known at the prior close.
    """
    if prices.empty:
        return pd.DataFrame(columns=SIGNAL_COLUMNS)

    close_m = prices.pivot(index="date", columns="symbol", values="close").sort_index()
    category_breadth = group_filter.compute_category_breadth(close_m, universe, cfg)
    revenue = fundamental_filter.compute_revenue_growth(
        fundamentals, close_m.index, close_m.columns, cfg
    )
    frames: list[pd.DataFrame] = []
    for symbol, sub in prices.groupby("symbol", sort=True):
        sub = sub.sort_values("date").copy()
        if len(sub) < max(cfg.high_lookback_days, cfg.ema_slow_days, cfg.volume_sma_days) + 1:
            continue

        high = sub["high"]
        close = sub["close"]
        volume = sub["volume"]
        alpaca_feed = sub["alpaca_price_feed"].astype("string").str.lower().str.strip()
        rolling_high = high.rolling(
            cfg.high_lookback_days,
            min_periods=cfg.high_lookback_days,
        ).max()
        is_52w_high = (high >= rolling_high) & rolling_high.notna()
        high_state = _last_high_state(sub, is_52w_high)
        had_high_recent = high_state["bars_since_52w_high"].le(cfg.high_recent_days)
        pullback_pass = high_state["pullback_from_52w_high_pct"].ge(cfg.min_pullback_pct)
        atr = _atr(sub, cfg.atr_days)
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
        crossed_down = (prev_ema_fast >= prev_ema_slow) & (ema_fast < ema_slow)
        crossed_any = crossed_up | crossed_down
        if cfg.ema_cross_lookback_days > 0:
            ema_cross_recent = (
                crossed_any.rolling(cfg.ema_cross_lookback_days, min_periods=1)
                .max()
                .fillna(False)
                .astype(bool)
            )
        else:
            ema_cross_recent = pd.Series(False, index=sub.index)
        ema_already_above_on_52w_high = (
            is_52w_high
            & (ema_fast > ema_slow)
            & (prev_ema_fast > prev_ema_slow)
        )
        ema_entry_pass = crossed_up
        volume_sma = volume.rolling(
            cfg.volume_sma_days,
            min_periods=cfg.volume_sma_days,
        ).mean()
        volume_sma50_pass = volume > volume_sma
        volume_feed_pass = alpaca_feed.eq("sip").fillna(False)
        volume_pass = volume_feed_pass & (volume_sma50_pass if cfg.volume_filter_enable else True)
        dates = pd.DatetimeIndex(sub["date"])
        if symbol in revenue["revenue_pass"].columns:
            revenue_pass = revenue["revenue_pass"][symbol].reindex(dates).fillna(False)
            revenue_yoy = revenue["revenue_yoy"][symbol].reindex(dates)
            revenue_ttm = revenue["revenue_ttm"][symbol].reindex(dates)
            prev_revenue_ttm = revenue["prev_revenue_ttm"][symbol].reindex(dates)
        else:
            revenue_pass = pd.Series(False, index=dates)
            revenue_yoy = pd.Series(float("nan"), index=dates)
            revenue_ttm = pd.Series(float("nan"), index=dates)
            prev_revenue_ttm = pd.Series(float("nan"), index=dates)
        if symbol in category_breadth["ibkr_category_breadth_pass"].columns:
            category_breadth_raw = category_breadth["ibkr_category_breadth"][symbol].reindex(dates)
            category_breadth_on = category_breadth["ibkr_category_breadth_on"][symbol].reindex(dates).fillna(False)
            category_breadth_pass = (
                category_breadth["ibkr_category_breadth_pass"][symbol]
                .reindex(dates)
                .fillna(False)
            )
        else:
            category_breadth_raw = pd.Series(float("nan"), index=dates)
            category_breadth_on = pd.Series(False, index=dates)
            category_breadth_pass = pd.Series(not cfg.ibkr_category_breadth_filter_enable, index=dates)

        sub["rolling_52w_high"] = rolling_high
        sub["is_52w_high"] = is_52w_high
        sub["had_52w_high_recent"] = had_high_recent
        sub["last_52w_high_date"] = high_state["last_52w_high_date"]
        sub["last_52w_high_price"] = high_state["last_52w_high_price"]
        sub["bars_since_52w_high"] = high_state["bars_since_52w_high"]
        sub["pullback_low_since_52w_high"] = high_state["pullback_low_since_52w_high"]
        sub["pullback_from_52w_high_pct"] = high_state["pullback_from_52w_high_pct"]
        sub["pullback_pass"] = pullback_pass
        sub["atr"] = atr
        sub["ema_fast"] = ema_fast
        sub["ema_slow"] = ema_slow
        sub["prev_ema_fast"] = prev_ema_fast
        sub["prev_ema_slow"] = prev_ema_slow
        sub["ema_cross_up"] = crossed_up
        sub["ema_cross_down"] = crossed_down
        sub["ema_cross_recent"] = ema_cross_recent
        sub["ema_already_above_on_52w_high"] = ema_already_above_on_52w_high
        sub["ema_entry_pass"] = ema_entry_pass
        sub["volume_sma50"] = volume_sma
        sub["volume_sma50_pass"] = volume_sma50_pass
        sub["volume_feed_pass"] = volume_feed_pass
        sub["volume_pass"] = volume_pass
        sub["ibkr_category_breadth_pct"] = (category_breadth_raw.to_numpy(dtype=float) * 100).round(4)
        sub["ibkr_category_breadth_on"] = category_breadth_on.to_numpy(dtype=bool)
        sub["ibkr_category_breadth_pass"] = category_breadth_pass.to_numpy(dtype=bool)
        sub["entry_signal"] = (
            had_high_recent
            & pullback_pass
            & ema_entry_pass
            & volume_pass
            & sub["ibkr_category_breadth_pass"]
        )
        _apply_delayed_entry_plan(sub, cfg)
        sub["market_cap_pass"] = (
            sub["entry_ref_market_cap_usd"].notna()
            & (sub["entry_ref_market_cap_usd"] >= cfg.min_market_cap_usd)
            & sub["entry_ref_market_cap_currency"].eq("USD")
        )
        sub["revenue_ttm"] = revenue_ttm.to_numpy(dtype=float)
        sub["prev_revenue_ttm"] = prev_revenue_ttm.to_numpy(dtype=float)
        sub["revenue_yoy"] = revenue_yoy.to_numpy(dtype=float)
        sub["revenue_pass"] = revenue_pass.to_numpy(dtype=bool)

        planned_entry_ts = pd.to_datetime(sub["planned_entry_date"])
        selected = sub[
            sub["entry_signal"]
            & (sub["date"].dt.date >= start)
            & (sub["date"].dt.date <= end)
            & (planned_entry_ts <= pd.Timestamp(end))
            & (sub["close"] >= cfg.min_price)
            & sub["planned_entry_open"].notna()
            & (sub["planned_entry_open"] >= cfg.min_price)
            & sub["market_cap_pass"]
            & sub["revenue_pass"]
            & sub["entry_gap_pass"]
        ].copy()
        if selected.empty:
            continue
        selected["period_end_date"] = selected["date"].dt.date
        selected["planned_entry_date"] = pd.to_datetime(selected["planned_entry_date"]).dt.date
        selected["last_52w_high_date"] = pd.to_datetime(selected["last_52w_high_date"]).dt.date
        selected["bars_since_52w_high"] = selected["bars_since_52w_high"].astype("int64")
        selected["ema_cross_delay_days"] = selected["ema_cross_delay_days"].astype("int64")
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


def _last_high_state(sub: pd.DataFrame, is_52w_high: pd.Series) -> pd.DataFrame:
    last_high_dates: list[pd.Timestamp | pd.NaT] = []
    last_high_prices: list[float] = []
    bars_since_values: list[int | None] = []
    pullback_lows: list[float] = []
    pullback_depths: list[float] = []

    last_date = pd.NaT
    last_price = float("nan")
    pullback_low = float("nan")
    bars_since: int | None = None

    for is_high, row in zip(is_52w_high.to_numpy(dtype=bool), sub.itertuples(index=False)):
        high_price = float(row.high)
        low_price = float(row.low)
        if is_high:
            last_date = pd.Timestamp(row.date)
            last_price = high_price
            pullback_low = low_price
            bars_since = 0
        elif bars_since is not None:
            bars_since += 1
            if low_price < pullback_low:
                pullback_low = low_price

        last_high_dates.append(last_date)
        last_high_prices.append(last_price)
        bars_since_values.append(bars_since)
        pullback_lows.append(pullback_low)
        if last_price > 0 and pd.notna(pullback_low):
            pullback_depths.append((last_price - pullback_low) / last_price)
        else:
            pullback_depths.append(float("nan"))

    return pd.DataFrame(
        {
            "last_52w_high_date": last_high_dates,
            "last_52w_high_price": last_high_prices,
            "bars_since_52w_high": bars_since_values,
            "pullback_low_since_52w_high": pullback_lows,
            "pullback_from_52w_high_pct": pullback_depths,
        },
        index=sub.index,
    )


def _atr(sub: pd.DataFrame, days: int) -> pd.Series:
    high = sub["high"].astype(float)
    low = sub["low"].astype(float)
    close = sub["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(days, min_periods=days).mean()


def _apply_delayed_entry_plan(sub: pd.DataFrame, cfg: Config) -> None:
    """Plan entries after recent EMA crosses have fallen out of the lookback."""
    signal_idx = sub.index[sub["entry_signal"].fillna(False)].tolist()
    date_values = sub["date"].to_numpy()
    open_values = sub["open"].to_numpy()
    close_values = sub["close"].to_numpy()
    market_cap_values = sub["market_cap_usd"].to_numpy()
    market_cap_currency_values = sub["market_cap_currency"].to_numpy()
    atr_values = sub["atr"].to_numpy()
    recent_cross_values = sub["ema_cross_recent"].fillna(False).to_numpy(dtype=bool)

    sub["planned_entry_date"] = pd.NaT
    sub["planned_entry_open"] = pd.NA
    sub["entry_ref_market_cap_usd"] = pd.NA
    sub["entry_ref_market_cap_currency"] = pd.NA
    sub["entry_gap_pct"] = pd.NA
    sub["entry_gap_pass"] = False
    sub["planned_entry_atr"] = pd.NA
    sub["ema_cross_delay_days"] = pd.NA

    index_to_position = {idx: pos for pos, idx in enumerate(sub.index)}
    for idx in signal_idx:
        signal_pos = index_to_position[idx]
        eval_pos = signal_pos
        while eval_pos < len(sub) - 1 and recent_cross_values[eval_pos]:
            eval_pos += 1
        if eval_pos >= len(sub) - 1 or recent_cross_values[eval_pos]:
            continue

        entry_pos = eval_pos + 1
        ref_close = float(close_values[eval_pos])
        entry_open = float(open_values[entry_pos])
        entry_gap_pct = (entry_open / ref_close) - 1.0 if ref_close > 0 else float("nan")
        sub.at[idx, "planned_entry_date"] = date_values[entry_pos]
        sub.at[idx, "planned_entry_open"] = entry_open
        sub.at[idx, "entry_ref_market_cap_usd"] = market_cap_values[eval_pos]
        sub.at[idx, "entry_ref_market_cap_currency"] = market_cap_currency_values[eval_pos]
        sub.at[idx, "entry_gap_pct"] = entry_gap_pct
        sub.at[idx, "entry_gap_pass"] = pd.notna(entry_gap_pct) and entry_gap_pct <= cfg.max_entry_gap_pct
        sub.at[idx, "planned_entry_atr"] = atr_values[eval_pos]
        sub.at[idx, "ema_cross_delay_days"] = eval_pos - signal_pos
