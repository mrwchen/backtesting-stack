"""Quality breakout fundamental swing model.

Model idea:
  - LONG: strong fundamentals plus a close near/new breakout high.
  - SHORT: weak fundamentals plus a close near/new breakdown low.
  - Direction is selected by the generic runner from the world-regime score.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, TradeIntent, IntentEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, mean


@dataclass
class IntentConfig:
    """Parameters for quality_breakout_fundamental_v1."""

    min_bars: int = 180

    long_min_pullback: float = 0.0
    long_max_pullback: float = 8.0
    long_ideal_pullback: float = 2.0
    long_max_rsi: float = 72.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 8.0
    short_ideal_bounce: float = 2.0
    short_min_rsi: float = 28.0
    short_max_rsi: float = 58.0

    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25

    price_lookback_bars: int = 240
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    breakout_tolerance_pct: float = 1.5
    trend_lookback_bars: int = 80
    min_trend_return_pct: float = 4.0
    min_volume_ratio: float = 0.8


def intent_config_from_env() -> IntentConfig:
    defaults = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", defaults.min_bars),
        long_min_pullback=env_float("LONG_MIN_PULLBACK", defaults.long_min_pullback),
        long_max_pullback=env_float("LONG_MAX_PULLBACK", defaults.long_max_pullback),
        long_ideal_pullback=env_float("LONG_IDEAL_PULLBACK", defaults.long_ideal_pullback),
        long_max_rsi=env_float("LONG_MAX_RSI", defaults.long_max_rsi),
        short_min_bounce=env_float("SHORT_MIN_BOUNCE", defaults.short_min_bounce),
        short_max_bounce=env_float("SHORT_MAX_BOUNCE", defaults.short_max_bounce),
        short_ideal_bounce=env_float("SHORT_IDEAL_BOUNCE", defaults.short_ideal_bounce),
        short_min_rsi=env_float("SHORT_MIN_RSI", defaults.short_min_rsi),
        short_max_rsi=env_float("SHORT_MAX_RSI", defaults.short_max_rsi),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", defaults.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", defaults.mispricing_weight),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", defaults.price_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", defaults.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", defaults.vol_long_bars),
        breakout_tolerance_pct=env_float("BREAKOUT_TOLERANCE_PCT", defaults.breakout_tolerance_pct),
        trend_lookback_bars=env_int("TREND_LOOKBACK_BARS", defaults.trend_lookback_bars),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", defaults.min_trend_return_pct),
        min_volume_ratio=env_float("MIN_VOLUME_RATIO", defaults.min_volume_ratio),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.trend_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {
        "config": dataclasses.replace(base_cfg),
        "notes": "grid model=quality_breakout_fundamental_v1",
        "summary": {},
    }


def _volume_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    vol_short = mean(volumes[-cfg.vol_short_bars:])
    vol_long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else vol_short
    return (vol_short / vol_long) if vol_long > 0 else 1.0


def _fundamental_raw(fundamental: FundamentalRow, cfg: IntentConfig, short: bool = False) -> float:
    base = fundamental.composite_score
    mispricing = fundamental.mispricing_score
    if cfg.use_mispricing_score and mispricing is not None:
        base = base * (1.0 - cfg.mispricing_weight) + mispricing * cfg.mispricing_weight
    return (100.0 - base if short else base) / 100.0


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry_price = closes[-1]
    lookback_high = max(highs[-cfg.price_lookback_bars:])
    breakout_gap_pct = (lookback_high - entry_price) / lookback_high * 100.0 if lookback_high > 0 else 999.0
    if breakout_gap_pct > cfg.breakout_tolerance_pct:
        return IntentEvaluation(None, "rejected", "not_near_breakout_high", f"Close is {breakout_gap_pct:.2f}% below breakout high.")

    trend_base = closes[-cfg.trend_lookback_bars] if len(closes) > cfg.trend_lookback_bars and closes[-cfg.trend_lookback_bars] > 0 else closes[0]
    trend_return_pct = (entry_price / trend_base - 1.0) * 100.0
    if trend_return_pct < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "trend_too_weak", f"Trend return {trend_return_pct:.2f}% is below minimum {cfg.min_trend_return_pct:.2f}%.")

    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} is above maximum {cfg.long_max_rsi:.2f}.")

    vol_ratio = _volume_ratio(volumes, cfg)
    if vol_ratio < cfg.min_volume_ratio:
        return IntentEvaluation(None, "rejected", "volume_confirmation_missing", f"Volume ratio {vol_ratio:.2f} is below minimum {cfg.min_volume_ratio:.2f}.")

    breakout_score = clamp(1.0 - breakout_gap_pct / max(cfg.breakout_tolerance_pct, 0.01), 0.0, 1.0)
    trend_score = clamp(trend_return_pct / 20.0, 0.0, 1.0)
    rsi_score = clamp((cfg.long_max_rsi - rsi) / 25.0, 0.0, 1.0)
    vol_score = clamp((vol_ratio - cfg.min_volume_ratio) / 0.8, 0.0, 1.0)
    entry_score = breakout_score * 0.45 + trend_score * 0.25 + rsi_score * 0.15 + vol_score * 0.15
    combined = (_fundamental_raw(fundamental, cfg) * 0.45 + entry_score * 0.55) * 10.0
    reason = f"Breakout gap {breakout_gap_pct:.1f}% | Trend {trend_return_pct:.1f}% | Vol {vol_ratio:.2f}x"
    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "quality_breakout_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry_price = closes[-1]
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    breakdown_gap_pct = (entry_price - lookback_low) / lookback_low * 100.0 if lookback_low > 0 else 999.0
    if breakdown_gap_pct > cfg.breakout_tolerance_pct:
        return IntentEvaluation(None, "rejected", "not_near_breakdown_low", f"Close is {breakdown_gap_pct:.2f}% above breakdown low.")

    trend_base = closes[-cfg.trend_lookback_bars] if len(closes) > cfg.trend_lookback_bars and closes[-cfg.trend_lookback_bars] > 0 else closes[0]
    trend_return_pct = (entry_price / trend_base - 1.0) * 100.0
    if trend_return_pct > -cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "downtrend_too_weak", f"Trend return {trend_return_pct:.2f}% is not negative enough.")

    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_range", f"RSI {rsi:.2f} outside short range {cfg.short_min_rsi:.2f}-{cfg.short_max_rsi:.2f}.")

    vol_ratio = _volume_ratio(volumes, cfg)
    if vol_ratio < cfg.min_volume_ratio:
        return IntentEvaluation(None, "rejected", "volume_confirmation_missing", f"Volume ratio {vol_ratio:.2f} is below minimum {cfg.min_volume_ratio:.2f}.")

    breakdown_score = clamp(1.0 - breakdown_gap_pct / max(cfg.breakout_tolerance_pct, 0.01), 0.0, 1.0)
    trend_score = clamp(abs(trend_return_pct) / 20.0, 0.0, 1.0)
    rsi_mid = (cfg.short_min_rsi + cfg.short_max_rsi) / 2.0
    rsi_score = clamp(1.0 - abs(rsi - rsi_mid) / 20.0, 0.0, 1.0)
    vol_score = clamp((vol_ratio - cfg.min_volume_ratio) / 0.8, 0.0, 1.0)
    entry_score = breakdown_score * 0.45 + trend_score * 0.25 + rsi_score * 0.15 + vol_score * 0.15
    combined = (_fundamental_raw(fundamental, cfg, short=True) * 0.45 + entry_score * 0.55) * 10.0
    reason = f"Breakdown gap {breakdown_gap_pct:.1f}% | Trend {trend_return_pct:.1f}% | Vol {vol_ratio:.2f}x"
    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "weak_breakdown_passed", reason)
