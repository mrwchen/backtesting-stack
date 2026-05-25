"""Volatility contraction breakout swing model.

Model idea:
  - LONG: constructive trend, tight volatility contraction, close near breakout high.
  - SHORT: weak trend, tight contraction, close near breakdown low.
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
    min_bars: int = 260
    long_min_pullback: float = 0.0
    long_max_pullback: float = 9.0
    long_ideal_pullback: float = 2.0
    long_max_rsi: float = 76.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 9.0
    short_ideal_bounce: float = 2.0
    short_min_rsi: float = 24.0
    short_max_rsi: float = 62.0
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.25
    fundamental_abs_weight: float = 0.75
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None
    price_lookback_bars: int = 240
    trend_lookback_bars: int = 120
    contraction_bars: int = 24
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    max_contraction_range_pct: float = 8.0
    max_contraction_atr_pct: float = 2.8
    breakout_tolerance_pct: float = 1.2
    min_trend_return_pct: float = 4.0
    min_volume_ratio: float = 0.75


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
        contraction_bars=env_int("CONTRACTION_BARS", d.contraction_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        max_contraction_range_pct=env_float("MAX_CONTRACTION_RANGE_PCT", d.max_contraction_range_pct),
        max_contraction_atr_pct=env_float("MAX_CONTRACTION_ATR_PCT", d.max_contraction_atr_pct),
        breakout_tolerance_pct=env_float("BREAKOUT_TOLERANCE_PCT", d.breakout_tolerance_pct),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", d.min_trend_return_pct),
        min_volume_ratio=env_float("MIN_VOLUME_RATIO", d.min_volume_ratio),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.trend_lookback_bars + 1,
        cfg.contraction_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=volatility_contraction_breakout_v1", "summary": {}}


def _ret_pct(closes: list[float], lookback: int) -> float:
    base = closes[-lookback] if len(closes) > lookback and closes[-lookback] > 0.0 else closes[0]
    return (closes[-1] / base - 1.0) * 100.0


def _range_pct(highs: list[float], lows: list[float], close: float) -> float:
    return (max(highs) - min(lows)) / close * 100.0 if close > 0.0 else 999.0


def _atr_pct(highs: list[float], lows: list[float], close: float) -> float:
    ranges = [h - l for h, l in zip(highs, lows)]
    return mean(ranges) / close * 100.0 if close > 0.0 else 999.0


def _vol_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0.0 else 1.0


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
    volumes = [float(b.volume) for b in bars]
    close = closes[-1]
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    contraction_range = _range_pct(highs[-cfg.contraction_bars:], lows[-cfg.contraction_bars:], close)
    contraction_atr = _atr_pct(highs[-cfg.contraction_bars:], lows[-cfg.contraction_bars:], close)
    breakout_high = max(highs[-cfg.price_lookback_bars:])
    breakout_gap = (breakout_high - close) / breakout_high * 100.0 if breakout_high > 0.0 else 999.0
    volume_ratio = _vol_ratio(volumes, cfg)
    rsi = compute_rsi(closes[-50:])
    if trend < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "trend_too_weak", f"Trend return {trend:.2f}% below minimum.")
    if contraction_range > cfg.max_contraction_range_pct or contraction_atr > cfg.max_contraction_atr_pct:
        return IntentEvaluation(None, "rejected", "volatility_not_contracted", f"Range {contraction_range:.2f}% ATR {contraction_atr:.2f}% not contracted.")
    if breakout_gap > cfg.breakout_tolerance_pct:
        return IntentEvaluation(None, "rejected", "not_near_breakout", f"Close is {breakout_gap:.2f}% below breakout high.")
    if volume_ratio < cfg.min_volume_ratio:
        return IntentEvaluation(None, "rejected", "volume_confirmation_missing", f"Volume ratio {volume_ratio:.2f} below minimum.")
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.")
    entry_score = (
        clamp(trend / 20.0, 0.0, 1.0) * 0.25
        + clamp(1.0 - contraction_range / max(cfg.max_contraction_range_pct, 0.01), 0.0, 1.0) * 0.30
        + clamp(1.0 - breakout_gap / max(cfg.breakout_tolerance_pct, 0.01), 0.0, 1.0) * 0.30
        + clamp((volume_ratio - cfg.min_volume_ratio) / 0.8, 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=False) * 0.40 + entry_score * 0.60) * 10.0
    reason = f"VCB long trend {trend:.1f}% | range {contraction_range:.1f}% | gap {breakout_gap:.1f}% | vol {volume_ratio:.2f}x"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason), "intent", "volatility_contraction_long_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    close = closes[-1]
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    contraction_range = _range_pct(highs[-cfg.contraction_bars:], lows[-cfg.contraction_bars:], close)
    contraction_atr = _atr_pct(highs[-cfg.contraction_bars:], lows[-cfg.contraction_bars:], close)
    breakdown_low = min(lows[-cfg.price_lookback_bars:])
    breakdown_gap = (close - breakdown_low) / breakdown_low * 100.0 if breakdown_low > 0.0 else 999.0
    volume_ratio = _vol_ratio(volumes, cfg)
    rsi = compute_rsi(closes[-50:])
    if trend > -cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "downtrend_too_weak", f"Trend return {trend:.2f}% not negative enough.")
    if contraction_range > cfg.max_contraction_range_pct or contraction_atr > cfg.max_contraction_atr_pct:
        return IntentEvaluation(None, "rejected", "volatility_not_contracted", f"Range {contraction_range:.2f}% ATR {contraction_atr:.2f}% not contracted.")
    if breakdown_gap > cfg.breakout_tolerance_pct:
        return IntentEvaluation(None, "rejected", "not_near_breakdown", f"Close is {breakdown_gap:.2f}% above breakdown low.")
    if volume_ratio < cfg.min_volume_ratio:
        return IntentEvaluation(None, "rejected", "volume_confirmation_missing", f"Volume ratio {volume_ratio:.2f} below minimum.")
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside short range.")
    entry_score = (
        clamp(abs(trend) / 20.0, 0.0, 1.0) * 0.25
        + clamp(1.0 - contraction_range / max(cfg.max_contraction_range_pct, 0.01), 0.0, 1.0) * 0.30
        + clamp(1.0 - breakdown_gap / max(cfg.breakout_tolerance_pct, 0.01), 0.0, 1.0) * 0.30
        + clamp((volume_ratio - cfg.min_volume_ratio) / 0.8, 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=True) * 0.40 + entry_score * 0.60) * 10.0
    reason = f"VCB short trend {trend:.1f}% | range {contraction_range:.1f}% | gap {breakdown_gap:.1f}% | vol {volume_ratio:.2f}x"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason), "intent", "volatility_contraction_short_passed", reason)
