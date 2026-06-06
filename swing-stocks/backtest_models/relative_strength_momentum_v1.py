"""Relative strength momentum swing model.

Model idea:
  - LONG: strong multi-month price momentum that has not recently broken down.
  - SHORT: weak multi-month price momentum that has not recently reversed up.
  - Fundamental quality/weakness is used as a context score, not as the signal core.
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
)


@dataclass
class IntentConfig:
    min_bars: int = 780
    long_min_pullback: float = 0.0
    long_max_pullback: float = 16.0
    long_ideal_pullback: float = 5.0
    long_max_rsi: float = 76.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 16.0
    short_ideal_bounce: float = 5.0
    short_min_rsi: float = 24.0
    short_max_rsi: float = 62.0
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.20
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.50
    fundamental_abs_weight: float = 0.50
    long_min_absolute_score: Optional[float] = 50.0
    short_max_absolute_score: Optional[float] = 50.0
    price_lookback_bars: int = 780
    momentum_lookback_bars: int = 520
    skip_recent_bars: int = 35
    confirmation_bars: int = 65
    min_long_momentum_pct: float = 8.0
    max_short_momentum_pct: float = -8.0
    min_long_confirmation_pct: float = -3.0
    max_short_confirmation_pct: float = 3.0


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
        momentum_lookback_bars=env_int("MOMENTUM_LOOKBACK_BARS", d.momentum_lookback_bars),
        skip_recent_bars=env_int("SKIP_RECENT_BARS", d.skip_recent_bars),
        confirmation_bars=env_int("CONFIRMATION_BARS", d.confirmation_bars),
        min_long_momentum_pct=env_float("MIN_LONG_MOMENTUM_PCT", d.min_long_momentum_pct),
        max_short_momentum_pct=env_float("MAX_SHORT_MOMENTUM_PCT", d.max_short_momentum_pct),
        min_long_confirmation_pct=env_float("MIN_LONG_CONFIRMATION_PCT", d.min_long_confirmation_pct),
        max_short_confirmation_pct=env_float("MAX_SHORT_CONFIRMATION_PCT", d.max_short_confirmation_pct),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.momentum_lookback_bars + cfg.skip_recent_bars + 1,
        cfg.confirmation_bars + 1,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=relative_strength_momentum_v1", "summary": {}}


def _ret_pct(closes: list[float], start_idx: int, end_idx: int = -1) -> float:
    start = closes[start_idx]
    end = closes[end_idx]
    return (end / start - 1.0) * 100.0 if start > 0.0 else 0.0


def _trend_score(value: float, threshold: float, span: float, short: bool) -> float:
    if short:
        return clamp((threshold - value) / max(span, 0.01), 0.0, 1.0)
    return clamp((value - threshold) / max(span, 0.01), 0.0, 1.0)


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
    old_idx = -(cfg.momentum_lookback_bars + cfg.skip_recent_bars)
    recent_idx = -cfg.skip_recent_bars
    momentum = _ret_pct(closes, old_idx, recent_idx)
    confirmation = _ret_pct(closes, -cfg.confirmation_bars)
    high = max(highs[-cfg.price_lookback_bars:])
    drawdown = (high - closes[-1]) / high * 100.0 if high > 0.0 else 999.0
    rsi = compute_rsi(closes[-50:])
    if momentum < cfg.min_long_momentum_pct:
        return IntentEvaluation(None, "rejected", "long_momentum_below_min", f"Skipped momentum {momentum:.2f}% below minimum.")
    if confirmation < cfg.min_long_confirmation_pct:
        return IntentEvaluation(None, "rejected", "recent_confirmation_too_weak", f"Recent confirmation {confirmation:.2f}% below minimum.")
    if drawdown > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "momentum_drawdown_too_deep", f"Drawdown from lookback high {drawdown:.2f}% above maximum.")
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.")
    entry_score = (
        _trend_score(momentum, cfg.min_long_momentum_pct, 30.0, short=False) * 0.45
        + _trend_score(confirmation, cfg.min_long_confirmation_pct, 12.0, short=False) * 0.25
        + clamp(1.0 - drawdown / max(cfg.long_max_pullback, 0.01), 0.0, 1.0) * 0.20
        + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.10
    )
    combined = (_fund(fundamental, cfg, short=False) * 0.30 + entry_score * 0.70) * 10.0
    reason = f"RS momentum {momentum:.1f}% skip {cfg.skip_recent_bars} bars | Confirm {confirmation:.1f}% | DD {drawdown:.1f}%"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason), "intent", "relative_strength_long_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    closes = [b.close for b in bars]
    lows = [b.low for b in bars]
    old_idx = -(cfg.momentum_lookback_bars + cfg.skip_recent_bars)
    recent_idx = -cfg.skip_recent_bars
    momentum = _ret_pct(closes, old_idx, recent_idx)
    confirmation = _ret_pct(closes, -cfg.confirmation_bars)
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (closes[-1] - low) / low * 100.0 if low > 0.0 else 999.0
    rsi = compute_rsi(closes[-50:])
    if momentum > cfg.max_short_momentum_pct:
        return IntentEvaluation(None, "rejected", "short_momentum_above_max", f"Skipped momentum {momentum:.2f}% above maximum.")
    if confirmation > cfg.max_short_confirmation_pct:
        return IntentEvaluation(None, "rejected", "recent_short_confirmation_too_strong", f"Recent confirmation {confirmation:.2f}% above maximum.")
    if bounce > cfg.short_max_bounce:
        return IntentEvaluation(None, "rejected", "weak_momentum_bounce_too_large", f"Bounce from lookback low {bounce:.2f}% above maximum.")
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside short range.")
    entry_score = (
        _trend_score(momentum, cfg.max_short_momentum_pct, 30.0, short=True) * 0.45
        + _trend_score(confirmation, cfg.max_short_confirmation_pct, 12.0, short=True) * 0.25
        + clamp(1.0 - bounce / max(cfg.short_max_bounce, 0.01), 0.0, 1.0) * 0.20
        + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.10
    )
    combined = (_fund(fundamental, cfg, short=True) * 0.30 + entry_score * 0.70) * 10.0
    reason = f"RS weakness {momentum:.1f}% skip {cfg.skip_recent_bars} bars | Confirm {confirmation:.1f}% | Bounce {bounce:.1f}%"
    return IntentEvaluation(TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason), "intent", "relative_weakness_short_passed", reason)
