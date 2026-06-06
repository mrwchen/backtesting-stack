"""Pullback/Bounce fundamental swing model.

Model idea:
  - LONG: strong fundamentals plus pullback from recent high.
  - SHORT: weak fundamentals plus bounce from recent low.
  - Direction is selected by the generic runner from the world-regime score.
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
    """Parameters for pullback_bounce_fundamental_v1."""

    # Intent limits
    min_bars: int = 150

    # Entry filters — LONG
    long_min_pullback: float = 5.0
    long_max_pullback: float = 25.0
    long_ideal_pullback: float = 12.5
    long_max_rsi: float = 50.0

    # Entry filters — SHORT
    short_min_bounce: float = 3.0
    short_max_bounce: float = 20.0
    short_ideal_bounce: float = 8.5
    short_min_rsi: float = 35.0
    short_max_rsi: float = 65.0

    # Mispricing score blending
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.3
    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = 55.0
    short_max_absolute_score: Optional[float] = 45.0

    # Lookback windows
    price_lookback_bars: int = 320
    vol_short_bars: int = 5
    vol_long_bars: int = 25


def intent_config_from_env() -> IntentConfig:
    """Build a model config from environment variables."""
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
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", defaults.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", defaults.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", defaults.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", defaults.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", defaults.short_max_absolute_score),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", defaults.price_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", defaults.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", defaults.vol_long_bars),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(
    base_cfg: IntentConfig,
    parse_grid_vals,
    parse_hold_grid_vals,
):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=pullback_bounce_fundamental_v1", "summary": {}}


def compute_long_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]

    entry_price = closes[-1]
    lookback_highs = highs[-cfg.price_lookback_bars:]
    high_20d = max(lookback_highs) if lookback_highs else entry_price

    if high_20d <= 0 or entry_price <= 0:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="invalid_price",
            reason_text="Entry price or lookback high is not positive.",
        )

    pullback_pct = (high_20d - entry_price) / high_20d * 100.0
    if pullback_pct < cfg.long_min_pullback:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="pullback_below_min",
            reason_text=f"Pullback {pullback_pct:.2f}% is below minimum {cfg.long_min_pullback:.2f}%.",
        )
    if pullback_pct > cfg.long_max_pullback:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="pullback_above_max",
            reason_text=f"Pullback {pullback_pct:.2f}% is above maximum {cfg.long_max_pullback:.2f}%.",
        )

    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="rsi_above_max",
            reason_text=f"RSI {rsi:.2f} is above maximum {cfg.long_max_rsi:.2f}.",
        )

    vol_short = mean(volumes[-cfg.vol_short_bars:])
    vol_long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else vol_short
    vol_ratio = (vol_short / vol_long) if vol_long > 0 else 1.0

    pullback_score = clamp(1.0 - abs((pullback_pct - cfg.long_ideal_pullback) / cfg.long_ideal_pullback), 0.0, 1.0)
    rsi_score = clamp((40.0 - rsi) / 20.0, 0.0, 1.0)
    vol_score = clamp((1.0 - vol_ratio) / 0.5, 0.0, 1.0)
    entry_score = pullback_score * 0.5 + rsi_score * 0.35 + vol_score * 0.15

    fund_raw = directional_fundamental_score(
        fundamental,
        short=False,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )

    combined = (fund_raw * 0.375 + entry_score * 0.625) * 10.0

    reason = f"Pullback {pullback_pct:.1f}% | RSI {rsi:.0f} | Vol {vol_ratio:.2f}x"

    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(
        intent=intent,
        decision="intent",
        reason_code="intent_passed",
        reason_text=reason,
    )


def compute_short_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]

    entry_price = closes[-1]
    lookback_lows = lows[-cfg.price_lookback_bars:]
    low_20d = min(lookback_lows) if lookback_lows else entry_price

    if low_20d <= 0 or entry_price <= 0:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="invalid_price",
            reason_text="Entry price or lookback low is not positive.",
        )

    bounce_pct = (entry_price - low_20d) / low_20d * 100.0
    if bounce_pct < cfg.short_min_bounce:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="bounce_below_min",
            reason_text=f"Bounce {bounce_pct:.2f}% is below minimum {cfg.short_min_bounce:.2f}%.",
        )
    if bounce_pct > cfg.short_max_bounce:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="bounce_above_max",
            reason_text=f"Bounce {bounce_pct:.2f}% is above maximum {cfg.short_max_bounce:.2f}%.",
        )

    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="rsi_below_min",
            reason_text=f"RSI {rsi:.2f} is below minimum {cfg.short_min_rsi:.2f}.",
        )
    if rsi > cfg.short_max_rsi:
        return IntentEvaluation(
            intent=None,
            decision="rejected",
            reason_code="rsi_above_max",
            reason_text=f"RSI {rsi:.2f} is above maximum {cfg.short_max_rsi:.2f}.",
        )

    vol_short = mean(volumes[-cfg.vol_short_bars:])
    vol_long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else vol_short
    vol_ratio = (vol_short / vol_long) if vol_long > 0 else 1.0

    bounce_score = clamp(1.0 - abs((bounce_pct - cfg.short_ideal_bounce) / cfg.short_ideal_bounce), 0.0, 1.0)
    rsi_ideal = (cfg.short_min_rsi + cfg.short_max_rsi) / 2.0
    rsi_score = clamp(1.0 - abs((rsi - rsi_ideal) / 15.0), 0.0, 1.0)
    vol_score = clamp((1.0 - vol_ratio) / 0.5, 0.0, 1.0)
    entry_score = bounce_score * 0.5 + rsi_score * 0.35 + vol_score * 0.15

    inv_fund = directional_fundamental_score(
        fundamental,
        short=True,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )

    combined = (inv_fund * 0.375 + entry_score * 0.625) * 10.0

    reason = f"Bounce {bounce_pct:.1f}% | RSI {rsi:.0f} | Vol {vol_ratio:.2f}x"

    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(
        intent=intent,
        decision="intent",
        reason_code="intent_passed",
        reason_text=reason,
    )
