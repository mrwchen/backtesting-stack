"""Weak fundamental breakdown swing model.

Model idea:
  - LONG side is conservative and only buys quality reclaim breakouts.
  - SHORT side is the primary edge: weak fundamentals plus price breakdown.
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
    min_bars: int = 180
    long_min_pullback: float = 0.0
    long_max_pullback: float = 10.0
    long_ideal_pullback: float = 3.0
    long_max_rsi: float = 68.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 5.0
    short_ideal_bounce: float = 1.0
    short_min_rsi: float = 20.0
    short_max_rsi: float = 55.0
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.30
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.25
    fundamental_abs_weight: float = 0.75
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None
    price_lookback_bars: int = 260
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    breakdown_tolerance_pct: float = 1.2
    min_downtrend_pct: float = 6.0
    min_volume_ratio: float = 0.9


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
        breakdown_tolerance_pct=env_float("BREAKDOWN_TOLERANCE_PCT", d.breakdown_tolerance_pct),
        min_downtrend_pct=env_float("MIN_DOWNTREND_PCT", d.min_downtrend_pct),
        min_volume_ratio=env_float("MIN_VOLUME_RATIO", d.min_volume_ratio),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=weak_fundamental_breakdown_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund_raw(f: FundamentalRow, cfg: IntentConfig, short: bool) -> float:
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
    gap = (high - entry) / high * 100.0 if high > 0 else 999.0
    if gap > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "quality_reclaim_not_close_to_high", f"Close is {gap:.2f}% below high.")
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.")
    vol_ratio = _vol_ratio(volumes, cfg)
    entry_score = clamp(1.0 - gap / max(cfg.long_max_pullback, 0.01), 0.0, 1.0) * 0.55 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.20 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.25
    combined = (_fund_raw(fundamental, cfg, short=False) * 0.45 + entry_score * 0.55) * 10.0
    reason = f"Quality reclaim gap {gap:.1f}% | RSI {rsi:.0f}"
    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "quality_reclaim_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    low_gap = (entry - lookback_low) / lookback_low * 100.0 if lookback_low > 0 else 999.0
    if low_gap > cfg.breakdown_tolerance_pct:
        return IntentEvaluation(None, "rejected", "not_breaking_down", f"Close is {low_gap:.2f}% above lookback low.")
    trend_base = closes[-cfg.price_lookback_bars] if len(closes) > cfg.price_lookback_bars and closes[-cfg.price_lookback_bars] > 0 else closes[0]
    downtrend = (1.0 - entry / trend_base) * 100.0
    if downtrend < cfg.min_downtrend_pct:
        return IntentEvaluation(None, "rejected", "downtrend_too_small", f"Downtrend {downtrend:.2f}% below minimum {cfg.min_downtrend_pct:.2f}%.")
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_breakdown_range", f"RSI {rsi:.2f} outside range.")
    vol_ratio = _vol_ratio(volumes, cfg)
    if vol_ratio < cfg.min_volume_ratio:
        return IntentEvaluation(None, "rejected", "breakdown_volume_too_low", f"Volume ratio {vol_ratio:.2f} below minimum.")
    breakdown_score = clamp(1.0 - low_gap / max(cfg.breakdown_tolerance_pct, 0.01), 0.0, 1.0)
    trend_score = clamp(downtrend / 25.0, 0.0, 1.0)
    vol_score = clamp(vol_ratio / 1.8, 0.0, 1.0)
    entry_score = breakdown_score * 0.45 + trend_score * 0.35 + vol_score * 0.20
    combined = (_fund_raw(fundamental, cfg, short=True) * 0.50 + entry_score * 0.50) * 10.0
    reason = f"Breakdown gap {low_gap:.1f}% | Downtrend {downtrend:.1f}% | Vol {vol_ratio:.2f}x"
    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "weak_fundamental_breakdown_passed", reason)
