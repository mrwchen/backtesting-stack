"""Trend pullback reclaim swing model.

Model idea:
  - LONG: established uptrend, controlled pullback into the fast average, reclaim.
  - SHORT: established downtrend, controlled bounce into the fast average, rejection.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, IntentEvaluation, TradeIntent
from backtest_shared import (
    clamp,
    compute_rsi,
    directional_fundamental_score,
    env_bool,
    env_float,
    env_int,
    env_optional_float,
    env_str,
    mean,
)


@dataclass
class IntentConfig:
    min_bars: int = 320
    long_min_pullback: float = 3.0
    long_max_pullback: float = 18.0
    long_ideal_pullback: float = 8.0
    long_max_rsi: float = 68.0
    short_min_bounce: float = 3.0
    short_max_bounce: float = 18.0
    short_ideal_bounce: float = 8.0
    short_min_rsi: float = 30.0
    short_max_rsi: float = 66.0
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.50
    fundamental_abs_weight: float = 0.50
    long_min_absolute_score: Optional[float] = 50.0
    short_max_absolute_score: Optional[float] = 50.0
    price_lookback_bars: int = 260
    trend_lookback_bars: int = 180
    fast_ma_bars: int = 35
    slow_ma_bars: int = 140
    reclaim_bars: int = 8
    max_distance_to_fast_ma_pct: float = 4.0
    min_trend_return_pct: float = 6.0


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        long_min_pullback=env_float("LONG_MIN_PULLBACK", d.long_min_pullback),
        long_max_pullback=env_float("LONG_MAX_PULLBACK", d.long_max_pullback),
        long_ideal_pullback=env_float("LONG_IDEAL_PULLBACK", d.long_ideal_pullback),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        short_min_bounce=env_float("SHORT_MIN_BOUNCE", d.short_min_bounce),
        short_max_bounce=env_float("SHORT_MAX_BOUNCE", d.short_max_bounce),
        short_ideal_bounce=env_float("SHORT_IDEAL_BOUNCE", d.short_ideal_bounce),
        short_min_rsi=env_float("SHORT_MIN_RSI", d.short_min_rsi),
        short_max_rsi=env_float("SHORT_MAX_RSI", d.short_max_rsi),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", d.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", d.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", d.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", d.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", d.short_max_absolute_score),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        trend_lookback_bars=env_int("TREND_LOOKBACK_BARS", d.trend_lookback_bars),
        fast_ma_bars=env_int("FAST_MA_BARS", d.fast_ma_bars),
        slow_ma_bars=env_int("SLOW_MA_BARS", d.slow_ma_bars),
        reclaim_bars=env_int("RECLAIM_BARS", d.reclaim_bars),
        max_distance_to_fast_ma_pct=env_float("MAX_DISTANCE_TO_FAST_MA_PCT", d.max_distance_to_fast_ma_pct),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", d.min_trend_return_pct),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.trend_lookback_bars + 1,
        cfg.slow_ma_bars,
        cfg.fast_ma_bars,
        cfg.reclaim_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=trend_pullback_reclaim_v1", "summary": {}}


def _sma(values: list[float], n: int) -> float:
    return mean(values[-n:]) if values and n > 0 else 0.0


def _ret_pct(closes: list[float], lookback: int) -> float:
    base = closes[-lookback] if len(closes) > lookback and closes[-lookback] > 0.0 else closes[0]
    return (closes[-1] / base - 1.0) * 100.0


def _fund(f: FundamentalRow, cfg: IntentConfig, short: bool) -> float:
    return directional_fundamental_score(
        f,
        short=short,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    close = closes[-1]
    fast_ma = _sma(closes, cfg.fast_ma_bars)
    slow_ma = _sma(closes, cfg.slow_ma_bars)
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    lookback_high = max(highs[-cfg.price_lookback_bars:])
    pullback = (lookback_high - close) / lookback_high * 100.0 if lookback_high > 0.0 else 999.0
    recent_low = min(lows[-cfg.reclaim_bars:])
    touched_fast_ma = fast_ma > 0.0 and abs(recent_low / fast_ma - 1.0) * 100.0 <= cfg.max_distance_to_fast_ma_pct
    reclaimed = fast_ma > 0.0 and close > fast_ma
    rsi = compute_rsi(closes[-50:])
    if close <= slow_ma or fast_ma <= slow_ma or trend < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "uptrend_not_confirmed", f"Trend {trend:.2f}% or moving-average structure not confirmed.")
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "pullback_outside_range", f"Pullback {pullback:.2f}% outside range.")
    if not touched_fast_ma or not reclaimed:
        return IntentEvaluation(None, "rejected", "fast_ma_reclaim_missing", "Pullback did not touch and reclaim the fast moving average.")
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.")
    entry_score = (
        clamp(trend / 22.0, 0.0, 1.0) * 0.30
        + clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / max(cfg.long_ideal_pullback, 0.01), 0.0, 1.0) * 0.35
        + clamp((close / fast_ma - 1.0) * 100.0 / 3.0, 0.0, 1.0) * 0.20
        + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=False) * 0.40 + entry_score * 0.60) * 10.0
    reason = f"Trend pullback {pullback:.1f}% | trend {trend:.1f}% | fast MA reclaim"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason), "intent", "trend_pullback_reclaim_long_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    close = closes[-1]
    fast_ma = _sma(closes, cfg.fast_ma_bars)
    slow_ma = _sma(closes, cfg.slow_ma_bars)
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    bounce = (close - lookback_low) / lookback_low * 100.0 if lookback_low > 0.0 else 999.0
    recent_high = max(highs[-cfg.reclaim_bars:])
    touched_fast_ma = fast_ma > 0.0 and abs(recent_high / fast_ma - 1.0) * 100.0 <= cfg.max_distance_to_fast_ma_pct
    rejected = fast_ma > 0.0 and close < fast_ma
    rsi = compute_rsi(closes[-50:])
    if close >= slow_ma or fast_ma >= slow_ma or trend > -cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "downtrend_not_confirmed", f"Trend {trend:.2f}% or moving-average structure not confirmed.")
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return IntentEvaluation(None, "rejected", "bounce_outside_range", f"Bounce {bounce:.2f}% outside range.")
    if not touched_fast_ma or not rejected:
        return IntentEvaluation(None, "rejected", "fast_ma_rejection_missing", "Bounce did not touch and reject the fast moving average.")
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside short range.")
    entry_score = (
        clamp(abs(trend) / 22.0, 0.0, 1.0) * 0.30
        + clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / max(cfg.short_ideal_bounce, 0.01), 0.0, 1.0) * 0.35
        + clamp((fast_ma / close - 1.0) * 100.0 / 3.0, 0.0, 1.0) * 0.20
        + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=True) * 0.40 + entry_score * 0.60) * 10.0
    reason = f"Trend bounce {bounce:.1f}% | trend {trend:.1f}% | fast MA rejection"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason), "intent", "trend_pullback_reclaim_short_passed", reason)
