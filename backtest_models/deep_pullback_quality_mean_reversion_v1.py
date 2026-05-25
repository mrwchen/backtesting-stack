"""Deep pullback quality mean-reversion swing model.

Model idea:
  - LONG: buy high-quality stocks after a deep but not catastrophic pullback.
  - SHORT: short weak stocks after an extended relief bounce.
  - Uses only information available in the provided PIT fundamental row and bars.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, TradeIntent, IntentEvaluation
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
    min_bars: int = 220

    long_min_pullback: float = 15.0
    long_max_pullback: float = 35.0
    long_ideal_pullback: float = 22.0
    long_max_rsi: float = 45.0
    short_min_bounce: float = 12.0
    short_max_bounce: float = 32.0
    short_ideal_bounce: float = 20.0
    short_min_rsi: float = 45.0
    short_max_rsi: float = 75.0

    use_mispricing_score: bool = True
    mispricing_weight: float = 0.35
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.30
    fundamental_abs_weight: float = 0.70
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None

    price_lookback_bars: int = 420
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    stabilization_bars: int = 3
    min_reclaim_pct: float = 0.5


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
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        stabilization_bars=env_int("STABILIZATION_BARS", d.stabilization_bars),
        min_reclaim_pct=env_float("MIN_RECLAIM_PCT", d.min_reclaim_pct),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        cfg.stabilization_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=deep_pullback_quality_mean_reversion_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund_score(f: FundamentalRow, cfg: IntentConfig, short: bool = False) -> float:
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
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    pullback = (high - entry) / high * 100.0 if high > 0 else 999.0
    if pullback < cfg.long_min_pullback:
        return IntentEvaluation(None, "rejected", "pullback_not_deep_enough", f"Pullback {pullback:.2f}% is below minimum {cfg.long_min_pullback:.2f}%.")
    if pullback > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "pullback_too_deep", f"Pullback {pullback:.2f}% is above maximum {cfg.long_max_pullback:.2f}%.")
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_not_oversold", f"RSI {rsi:.2f} is above maximum {cfg.long_max_rsi:.2f}.")
    recent_low = min(lows[-max(1, cfg.stabilization_bars):])
    reclaim = (entry / recent_low - 1.0) * 100.0 if recent_low > 0 else 0.0
    if reclaim < cfg.min_reclaim_pct:
        return IntentEvaluation(None, "rejected", "no_stabilization_reclaim", f"Close reclaimed only {reclaim:.2f}% from recent low.")
    vol_ratio = _vol_ratio(volumes, cfg)
    pullback_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / cfg.long_ideal_pullback, 0.0, 1.0)
    rsi_score = clamp((cfg.long_max_rsi - rsi) / 25.0, 0.0, 1.0)
    reclaim_score = clamp(reclaim / 4.0, 0.0, 1.0)
    entry_score = pullback_score * 0.45 + rsi_score * 0.30 + reclaim_score * 0.20 + clamp(1.2 - vol_ratio, 0.0, 1.0) * 0.05
    combined = (_fund_score(fundamental, cfg) * 0.50 + entry_score * 0.50) * 10.0
    reason = f"Deep pullback {pullback:.1f}% | RSI {rsi:.0f} | Reclaim {reclaim:.1f}%"
    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "deep_quality_reversion_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (entry - low) / low * 100.0 if low > 0 else 999.0
    if bounce < cfg.short_min_bounce:
        return IntentEvaluation(None, "rejected", "bounce_not_extended_enough", f"Bounce {bounce:.2f}% is below minimum {cfg.short_min_bounce:.2f}%.")
    if bounce > cfg.short_max_bounce:
        return IntentEvaluation(None, "rejected", "bounce_too_extended", f"Bounce {bounce:.2f}% is above maximum {cfg.short_max_bounce:.2f}%.")
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_not_in_relief_zone", f"RSI {rsi:.2f} outside short relief range.")
    recent_high = max(highs[-max(1, cfg.stabilization_bars):])
    rejection = (recent_high / entry - 1.0) * 100.0 if entry > 0 else 0.0
    if rejection < cfg.min_reclaim_pct:
        return IntentEvaluation(None, "rejected", "no_bounce_rejection", f"Close rejected only {rejection:.2f}% from recent high.")
    vol_ratio = _vol_ratio(volumes, cfg)
    bounce_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / cfg.short_ideal_bounce, 0.0, 1.0)
    rsi_score = clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0)
    rejection_score = clamp(rejection / 4.0, 0.0, 1.0)
    entry_score = bounce_score * 0.45 + rsi_score * 0.25 + rejection_score * 0.25 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.05
    combined = (_fund_score(fundamental, cfg, short=True) * 0.50 + entry_score * 0.50) * 10.0
    reason = f"Relief bounce {bounce:.1f}% | RSI {rsi:.0f} | Rejection {rejection:.1f}%"
    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "weak_relief_reversion_passed", reason)
