"""Momentum pullback long model.

Model idea:
  - LONG only.
  - Use the fundamental scorer as a broad universe filter.
  - Use price-momentum leadership plus a controlled pullback as the entry edge.
  - Avoid near-high breakout entries; the forward research showed those are weak
    swing entries despite looking attractive as momentum.
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
    env_list,
    env_optional_float,
    env_str,
    mean,
)


@dataclass
class IntentConfig:
    min_bars: int = 320
    enable_shorts: bool = False

    allowed_world_regime_labels: tuple[str, ...] = ("CONSTRUCTIVE",)
    require_world_regime_label: bool = True

    min_long_composite_score: float = 50.0
    min_long_price_momentum_score: float = 75.0
    min_long_momentum_score: float = 55.0
    shallow_pullback_min_price_momentum_score: float = 85.0

    long_min_pullback_pct: float = 5.0
    long_max_pullback_pct: float = 10.0
    allow_shallow_pullback: bool = True
    shallow_min_pullback_pct: float = 2.0
    shallow_max_pullback_pct: float = 5.0

    price_lookback_bars: int = 260
    trend_lookback_bars: int = 260
    confirmation_bars: int = 13
    rebound_bars: int = 13
    fast_ma_bars: int = 65
    slow_ma_bars: int = 260
    atr_bars: int = 65

    min_trend_return_pct: float = 5.0
    min_confirmation_pct: float = -4.0
    min_rebound_from_recent_low_pct: float = 0.5
    long_min_rsi: float = 35.0
    long_max_rsi: float = 72.0
    max_atr_pct: float = 8.0

    use_mispricing_score: bool = False
    mispricing_weight: float = 0.0
    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        enable_shorts=env_bool("ENABLE_SHORTS", d.enable_shorts),
        allowed_world_regime_labels=tuple(
            label.strip().upper()
            for label in env_list("ALLOWED_WORLD_REGIME_LABELS", d.allowed_world_regime_labels)
            if label.strip()
        ),
        require_world_regime_label=env_bool("REQUIRE_WORLD_REGIME_LABEL", d.require_world_regime_label),
        min_long_composite_score=env_float("MIN_LONG_COMPOSITE_SCORE", d.min_long_composite_score),
        min_long_price_momentum_score=env_float("MIN_LONG_PRICE_MOMENTUM_SCORE", d.min_long_price_momentum_score),
        min_long_momentum_score=env_float("MIN_LONG_MOMENTUM_SCORE", d.min_long_momentum_score),
        shallow_pullback_min_price_momentum_score=env_float(
            "SHALLOW_PULLBACK_MIN_PRICE_MOMENTUM_SCORE",
            d.shallow_pullback_min_price_momentum_score,
        ),
        long_min_pullback_pct=env_float("LONG_MIN_PULLBACK_PCT", d.long_min_pullback_pct),
        long_max_pullback_pct=env_float("LONG_MAX_PULLBACK_PCT", d.long_max_pullback_pct),
        allow_shallow_pullback=env_bool("ALLOW_SHALLOW_PULLBACK", d.allow_shallow_pullback),
        shallow_min_pullback_pct=env_float("SHALLOW_MIN_PULLBACK_PCT", d.shallow_min_pullback_pct),
        shallow_max_pullback_pct=env_float("SHALLOW_MAX_PULLBACK_PCT", d.shallow_max_pullback_pct),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        trend_lookback_bars=env_int("TREND_LOOKBACK_BARS", d.trend_lookback_bars),
        confirmation_bars=env_int("CONFIRMATION_BARS", d.confirmation_bars),
        rebound_bars=env_int("REBOUND_BARS", d.rebound_bars),
        fast_ma_bars=env_int("FAST_MA_BARS", d.fast_ma_bars),
        slow_ma_bars=env_int("SLOW_MA_BARS", d.slow_ma_bars),
        atr_bars=env_int("ATR_BARS", d.atr_bars),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", d.min_trend_return_pct),
        min_confirmation_pct=env_float("MIN_CONFIRMATION_PCT", d.min_confirmation_pct),
        min_rebound_from_recent_low_pct=env_float(
            "MIN_REBOUND_FROM_RECENT_LOW_PCT",
            d.min_rebound_from_recent_low_pct,
        ),
        long_min_rsi=env_float("LONG_MIN_RSI", d.long_min_rsi),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        max_atr_pct=env_float("MAX_ATR_PCT", d.max_atr_pct),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", d.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", d.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", d.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", d.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", d.short_max_absolute_score),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.trend_lookback_bars + 1,
        cfg.confirmation_bars + 1,
        cfg.rebound_bars,
        cfg.slow_ma_bars,
        cfg.fast_ma_bars,
        cfg.atr_bars + 1,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=momentum_pullback_long_v1", "summary": {}}


def _score01(value: Optional[float], fallback: float = 50.0) -> float:
    return clamp(float(value if value is not None else fallback) / 100.0, 0.0, 1.0)


def _ret_pct(closes: list[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    base = closes[-lookback]
    return (closes[-1] / base - 1.0) * 100.0 if base > 0.0 else 0.0


def _atr_pct(bars: list[Bar], lookback: int) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    ranges: list[float] = []
    subset = bars[-(lookback + 1):]
    for prev, cur in zip(subset, subset[1:]):
        ranges.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    close = bars[-1].close
    return mean(ranges) / close * 100.0 if close > 0.0 else 0.0


def _sma(values: list[float], n: int) -> float:
    return mean(values[-n:]) if values and n > 0 else 0.0


def _world_regime_label(cfg: IntentConfig) -> str:
    return str(getattr(cfg, "daily_policy_world_regime_label", "") or "").strip().upper()


def _allowed_regime_label(cfg: IntentConfig) -> bool:
    label = _world_regime_label(cfg)
    if not label:
        return not cfg.require_world_regime_label
    allowed = {item.strip().upper() for item in cfg.allowed_world_regime_labels if item.strip()}
    return not allowed or label in allowed


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
    if not _allowed_regime_label(cfg):
        return IntentEvaluation(
            None,
            "rejected",
            "world_regime_not_allowed",
            f"World regime label {_world_regime_label(cfg) or '-'} is not allowed.",
        )

    composite = float(fundamental.composite_score)
    price_momentum = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else composite)
    momentum = float(fundamental.momentum_score if fundamental.momentum_score is not None else composite)

    if composite < cfg.min_long_composite_score:
        return IntentEvaluation(None, "rejected", "composite_below_min", f"Composite {composite:.1f} below minimum.")
    if price_momentum < cfg.min_long_price_momentum_score:
        return IntentEvaluation(None, "rejected", "price_momentum_below_min", f"Price momentum {price_momentum:.1f} below minimum.")
    if momentum < cfg.min_long_momentum_score:
        return IntentEvaluation(None, "rejected", "momentum_below_min", f"Momentum {momentum:.1f} below minimum.")

    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    close = closes[-1]
    if close <= 0.0:
        return IntentEvaluation(None, "rejected", "invalid_price", "Close is not positive.")

    lookback_high = max(highs[-cfg.price_lookback_bars:])
    pullback = (lookback_high - close) / lookback_high * 100.0 if lookback_high > 0.0 else 999.0
    primary_pullback = cfg.long_min_pullback_pct <= pullback <= cfg.long_max_pullback_pct
    shallow_pullback = (
        cfg.allow_shallow_pullback
        and cfg.shallow_min_pullback_pct <= pullback < cfg.shallow_max_pullback_pct
        and price_momentum >= cfg.shallow_pullback_min_price_momentum_score
    )
    if not primary_pullback and not shallow_pullback:
        return IntentEvaluation(
            None,
            "rejected",
            "pullback_outside_researched_edge",
            f"Pullback {pullback:.2f}% is outside researched edge.",
        )

    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    if trend < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "trend_below_min", f"Trend {trend:.2f}% below minimum.")

    confirmation = _ret_pct(closes, cfg.confirmation_bars)
    if confirmation < cfg.min_confirmation_pct:
        return IntentEvaluation(
            None,
            "rejected",
            "confirmation_too_weak",
            f"Recent confirmation {confirmation:.2f}% below minimum.",
        )

    recent_low = min(lows[-cfg.rebound_bars:])
    rebound = (close / recent_low - 1.0) * 100.0 if recent_low > 0.0 else 0.0
    if rebound < cfg.min_rebound_from_recent_low_pct:
        return IntentEvaluation(
            None,
            "rejected",
            "rebound_from_recent_low_too_small",
            f"Rebound from recent low {rebound:.2f}% below minimum.",
        )

    slow_ma = _sma(closes, cfg.slow_ma_bars)
    fast_ma = _sma(closes, cfg.fast_ma_bars)
    if slow_ma > 0.0 and close <= slow_ma:
        return IntentEvaluation(None, "rejected", "below_slow_ma", "Close is below slow moving average.")

    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.long_min_rsi or rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_range", f"RSI {rsi:.2f} outside long range.")

    atr = _atr_pct(bars, cfg.atr_bars)
    if atr > cfg.max_atr_pct:
        return IntentEvaluation(None, "rejected", "atr_above_max", f"ATR {atr:.2f}% above maximum.")

    fund_score = directional_fundamental_score(
        fundamental,
        short=False,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )
    pullback_ideal = (cfg.long_min_pullback_pct + cfg.long_max_pullback_pct) / 2.0
    pullback_width = max((cfg.long_max_pullback_pct - cfg.long_min_pullback_pct) / 2.0, 0.01)
    pullback_score = clamp(1.0 - abs(pullback - pullback_ideal) / pullback_width, 0.0, 1.0)
    if shallow_pullback:
        pullback_score *= 0.80
    price_momentum_score = clamp((price_momentum - cfg.min_long_price_momentum_score) / 25.0, 0.0, 1.0)
    trend_score = clamp((trend - cfg.min_trend_return_pct) / 30.0, 0.0, 1.0)
    confirmation_score = clamp((confirmation - cfg.min_confirmation_pct) / 12.0, 0.0, 1.0)
    rebound_score = clamp(rebound / 5.0, 0.0, 1.0)
    rsi_score = clamp(1.0 - max(0.0, rsi - 55.0) / max(cfg.long_max_rsi - 55.0, 1.0), 0.0, 1.0)
    ma_score = 1.0 if fast_ma > slow_ma and close > fast_ma else (0.6 if fast_ma > slow_ma else 0.3)

    combined = (
        price_momentum_score * 0.24
        + pullback_score * 0.24
        + fund_score * 0.18
        + trend_score * 0.12
        + confirmation_score * 0.08
        + rebound_score * 0.08
        + rsi_score * 0.03
        + ma_score * 0.03
    ) * 10.0

    reason = (
        f"Momentum pullback long | regime {_world_regime_label(cfg) or '-'} | "
        f"PM {price_momentum:.1f} | composite {composite:.1f} | pullback {pullback:.1f}% | "
        f"trend {trend:.1f}% | confirmation {confirmation:.1f}% | rebound {rebound:.1f}% | "
        f"RSI {rsi:.0f} | ATR {atr:.1f}%"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason),
        "intent",
        "momentum_pullback_long_passed",
        reason,
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
    if not cfg.enable_shorts:
        return IntentEvaluation(None, "rejected", "shorts_disabled", "Short side is disabled for this model.")
    return IntentEvaluation(None, "rejected", "shorts_not_implemented", "Short side is intentionally not implemented.")
