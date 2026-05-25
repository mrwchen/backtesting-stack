"""Failed breakout reversal swing model.

Model idea:
  - LONG: failed breakdown/bear trap in a fundamentally acceptable stock.
  - SHORT: failed breakout/bull trap in a fundamentally weak stock.
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
    min_bars: int = 220
    long_min_pullback: float = 4.0
    long_max_pullback: float = 28.0
    long_ideal_pullback: float = 12.0
    long_max_rsi: float = 66.0
    short_min_bounce: float = 4.0
    short_max_bounce: float = 28.0
    short_ideal_bounce: float = 12.0
    short_min_rsi: float = 34.0
    short_max_rsi: float = 74.0
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.50
    fundamental_abs_weight: float = 0.50
    long_min_absolute_score: Optional[float] = 45.0
    short_max_absolute_score: Optional[float] = 55.0
    price_lookback_bars: int = 180
    failure_window_bars: int = 10
    min_break_pct: float = 1.0
    min_reclaim_pct: float = 0.25
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    min_failure_volume_ratio: float = 0.8


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
        failure_window_bars=env_int("FAILURE_WINDOW_BARS", d.failure_window_bars),
        min_break_pct=env_float("MIN_BREAK_PCT", d.min_break_pct),
        min_reclaim_pct=env_float("MIN_RECLAIM_PCT", d.min_reclaim_pct),
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        min_failure_volume_ratio=env_float("MIN_FAILURE_VOLUME_RATIO", d.min_failure_volume_ratio),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars + cfg.failure_window_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=failed_breakout_reversal_v1", "summary": {}}


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
    prior_lows = lows[-(cfg.price_lookback_bars + cfg.failure_window_bars):-cfg.failure_window_bars]
    if not prior_lows:
        return IntentEvaluation(None, "rejected", "insufficient_failure_context", "No prior low window was available.")
    prior_low = min(prior_lows)
    failure_low = min(lows[-cfg.failure_window_bars:])
    close = closes[-1]
    break_pct = (prior_low - failure_low) / prior_low * 100.0 if prior_low > 0.0 else 0.0
    reclaim_pct = (close - prior_low) / prior_low * 100.0 if prior_low > 0.0 else 0.0
    lookback_high = max(highs[-cfg.price_lookback_bars:])
    pullback = (lookback_high - close) / lookback_high * 100.0 if lookback_high > 0.0 else 999.0
    volume_ratio = _vol_ratio(volumes, cfg)
    rsi = compute_rsi(closes[-50:])
    if break_pct < cfg.min_break_pct:
        return IntentEvaluation(None, "rejected", "breakdown_failure_missing", f"Break below prior low {break_pct:.2f}% below minimum.")
    if reclaim_pct < cfg.min_reclaim_pct:
        return IntentEvaluation(None, "rejected", "breakdown_reclaim_missing", f"Reclaim above prior low {reclaim_pct:.2f}% below minimum.")
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "long_failure_pullback_outside_range", f"Pullback {pullback:.2f}% outside range.")
    if volume_ratio < cfg.min_failure_volume_ratio:
        return IntentEvaluation(None, "rejected", "failure_volume_too_low", f"Volume ratio {volume_ratio:.2f} below minimum.")
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.")
    entry_score = (
        clamp(break_pct / 4.0, 0.0, 1.0) * 0.25
        + clamp(reclaim_pct / 3.0, 0.0, 1.0) * 0.35
        + clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / max(cfg.long_ideal_pullback, 0.01), 0.0, 1.0) * 0.25
        + clamp(volume_ratio / 1.8, 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=False) * 0.45 + entry_score * 0.55) * 10.0
    reason = f"Failed breakdown break {break_pct:.1f}% | reclaim {reclaim_pct:.1f}% | pullback {pullback:.1f}%"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason), "intent", "failed_breakdown_reversal_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    prior_highs = highs[-(cfg.price_lookback_bars + cfg.failure_window_bars):-cfg.failure_window_bars]
    if not prior_highs:
        return IntentEvaluation(None, "rejected", "insufficient_failure_context", "No prior high window was available.")
    prior_high = max(prior_highs)
    failure_high = max(highs[-cfg.failure_window_bars:])
    close = closes[-1]
    break_pct = (failure_high - prior_high) / prior_high * 100.0 if prior_high > 0.0 else 0.0
    reclaim_pct = (prior_high - close) / prior_high * 100.0 if prior_high > 0.0 else 0.0
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    bounce = (close - lookback_low) / lookback_low * 100.0 if lookback_low > 0.0 else 999.0
    volume_ratio = _vol_ratio(volumes, cfg)
    rsi = compute_rsi(closes[-50:])
    if break_pct < cfg.min_break_pct:
        return IntentEvaluation(None, "rejected", "breakout_failure_missing", f"Break above prior high {break_pct:.2f}% below minimum.")
    if reclaim_pct < cfg.min_reclaim_pct:
        return IntentEvaluation(None, "rejected", "breakout_rejection_missing", f"Rejection below prior high {reclaim_pct:.2f}% below minimum.")
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return IntentEvaluation(None, "rejected", "short_failure_bounce_outside_range", f"Bounce {bounce:.2f}% outside range.")
    if volume_ratio < cfg.min_failure_volume_ratio:
        return IntentEvaluation(None, "rejected", "failure_volume_too_low", f"Volume ratio {volume_ratio:.2f} below minimum.")
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside short range.")
    entry_score = (
        clamp(break_pct / 4.0, 0.0, 1.0) * 0.25
        + clamp(reclaim_pct / 3.0, 0.0, 1.0) * 0.35
        + clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / max(cfg.short_ideal_bounce, 0.01), 0.0, 1.0) * 0.25
        + clamp(volume_ratio / 1.8, 0.0, 1.0) * 0.15
    )
    combined = (_fund(fundamental, cfg, short=True) * 0.45 + entry_score * 0.55) * 10.0
    reason = f"Failed breakout break {break_pct:.1f}% | rejection {reclaim_pct:.1f}% | bounce {bounce:.1f}%"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason), "intent", "failed_breakout_reversal_passed", reason)
