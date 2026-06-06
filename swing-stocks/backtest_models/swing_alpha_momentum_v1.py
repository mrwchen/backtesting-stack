"""Swing alpha momentum model.

Model idea:
  - LONG: use fundamental scorer subscores only as a universe/rank input, then
    require confirmed price leadership, shallow drawdown, and stable trend.
  - SHORT: disabled by default. Current short scorer flags are not reliable
    naked-short signals in the observed sample.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, IntentEvaluation, TradeIntent
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int


@dataclass
class IntentConfig:
    min_bars: int = 780
    enable_shorts: bool = False

    min_long_composite_score: float = 60.0
    min_long_scorer_alpha: float = 0.68
    min_long_swing_alpha: float = 0.66
    min_long_momentum_score: float = 70.0
    min_long_price_momentum_score: float = 70.0
    min_long_leadership_score: float = 65.0

    trend_lookback_bars: int = 520
    intermediate_lookback_bars: int = 260
    confirmation_bars: int = 65
    breakout_lookback_bars: int = 260
    fast_ma_bars: int = 65
    slow_ma_bars: int = 260
    atr_bars: int = 65

    min_trend_return_pct: float = 8.0
    min_intermediate_return_pct: float = 3.0
    min_confirmation_pct: float = -1.0
    long_max_drawdown_pct: float = 14.0
    long_max_breakout_gap_pct: float = 8.0
    long_min_rsi: float = 42.0
    long_max_rsi: float = 78.0
    max_atr_pct: float = 8.0

    max_short_composite_score: float = 35.0
    max_short_scorer_alpha: float = 0.38
    min_short_swing_alpha: float = 0.66
    max_short_momentum_score: float = 42.0
    max_short_price_momentum_score: float = 45.0
    max_short_leadership_score: float = 45.0
    max_short_trend_return_pct: float = -8.0
    max_short_confirmation_pct: float = 1.0
    short_max_bounce_pct: float = 10.0
    short_min_rsi: float = 24.0
    short_max_rsi: float = 62.0


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        enable_shorts=env_bool("ENABLE_SHORTS", d.enable_shorts),
        min_long_composite_score=env_float("MIN_LONG_COMPOSITE_SCORE", d.min_long_composite_score),
        min_long_scorer_alpha=env_float("MIN_LONG_SCORER_ALPHA", d.min_long_scorer_alpha),
        min_long_swing_alpha=env_float("MIN_LONG_SWING_ALPHA", d.min_long_swing_alpha),
        min_long_momentum_score=env_float("MIN_LONG_MOMENTUM_SCORE", d.min_long_momentum_score),
        min_long_price_momentum_score=env_float("MIN_LONG_PRICE_MOMENTUM_SCORE", d.min_long_price_momentum_score),
        min_long_leadership_score=env_float("MIN_LONG_LEADERSHIP_SCORE", d.min_long_leadership_score),
        trend_lookback_bars=env_int("TREND_LOOKBACK_BARS", d.trend_lookback_bars),
        intermediate_lookback_bars=env_int("INTERMEDIATE_LOOKBACK_BARS", d.intermediate_lookback_bars),
        confirmation_bars=env_int("CONFIRMATION_BARS", d.confirmation_bars),
        breakout_lookback_bars=env_int("BREAKOUT_LOOKBACK_BARS", d.breakout_lookback_bars),
        fast_ma_bars=env_int("FAST_MA_BARS", d.fast_ma_bars),
        slow_ma_bars=env_int("SLOW_MA_BARS", d.slow_ma_bars),
        atr_bars=env_int("ATR_BARS", d.atr_bars),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", d.min_trend_return_pct),
        min_intermediate_return_pct=env_float("MIN_INTERMEDIATE_RETURN_PCT", d.min_intermediate_return_pct),
        min_confirmation_pct=env_float("MIN_CONFIRMATION_PCT", d.min_confirmation_pct),
        long_max_drawdown_pct=env_float("LONG_MAX_DRAWDOWN_PCT", d.long_max_drawdown_pct),
        long_max_breakout_gap_pct=env_float("LONG_MAX_BREAKOUT_GAP_PCT", d.long_max_breakout_gap_pct),
        long_min_rsi=env_float("LONG_MIN_RSI", d.long_min_rsi),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        max_atr_pct=env_float("MAX_ATR_PCT", d.max_atr_pct),
        max_short_composite_score=env_float("MAX_SHORT_COMPOSITE_SCORE", d.max_short_composite_score),
        max_short_scorer_alpha=env_float("MAX_SHORT_SCORER_ALPHA", d.max_short_scorer_alpha),
        min_short_swing_alpha=env_float("MIN_SHORT_SWING_ALPHA", d.min_short_swing_alpha),
        max_short_momentum_score=env_float("MAX_SHORT_MOMENTUM_SCORE", d.max_short_momentum_score),
        max_short_price_momentum_score=env_float("MAX_SHORT_PRICE_MOMENTUM_SCORE", d.max_short_price_momentum_score),
        max_short_leadership_score=env_float("MAX_SHORT_LEADERSHIP_SCORE", d.max_short_leadership_score),
        max_short_trend_return_pct=env_float("MAX_SHORT_TREND_RETURN_PCT", d.max_short_trend_return_pct),
        max_short_confirmation_pct=env_float("MAX_SHORT_CONFIRMATION_PCT", d.max_short_confirmation_pct),
        short_max_bounce_pct=env_float("SHORT_MAX_BOUNCE_PCT", d.short_max_bounce_pct),
        short_min_rsi=env_float("SHORT_MIN_RSI", d.short_min_rsi),
        short_max_rsi=env_float("SHORT_MAX_RSI", d.short_max_rsi),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.trend_lookback_bars + 1,
        cfg.intermediate_lookback_bars + 1,
        cfg.confirmation_bars + 1,
        cfg.breakout_lookback_bars,
        cfg.slow_ma_bars,
        cfg.atr_bars + 1,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=swing_alpha_momentum_v1", "summary": {}}


def _score01(value: Optional[float], fallback: float) -> float:
    return clamp(float(value if value is not None else fallback) / 100.0, 0.0, 1.0)


def _ret_pct(closes: list[float], lookback: int) -> float:
    if len(closes) <= lookback:
        return 0.0
    start = closes[-lookback]
    end = closes[-1]
    return (end / start - 1.0) * 100.0 if start > 0.0 else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _atr_pct(bars: list[Bar], lookback: int) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    ranges: list[float] = []
    subset = bars[-(lookback + 1):]
    for prev, cur in zip(subset, subset[1:]):
        true_range = max(
            cur.high - cur.low,
            abs(cur.high - prev.close),
            abs(cur.low - prev.close),
        )
        ranges.append(true_range)
    close = bars[-1].close
    return (_mean(ranges) / close) * 100.0 if close > 0.0 else 0.0


def _long_scorer_alpha(f: FundamentalRow) -> float:
    composite = _score01(f.composite_score, 50.0)
    momentum = _score01(f.momentum_score, f.composite_score)
    price_momentum = _score01(f.price_momentum_score, f.momentum_score or f.composite_score)
    leadership = _score01(f.leadership_score, f.composite_score)
    fundamental_momentum = _score01(f.fundamental_momentum_score, f.composite_score)
    return clamp(
        composite * 0.20
        + momentum * 0.25
        + price_momentum * 0.25
        + leadership * 0.20
        + fundamental_momentum * 0.10,
        0.0,
        1.0,
    )


def _short_scorer_alpha(f: FundamentalRow) -> float:
    composite = 1.0 - _score01(f.composite_score, 50.0)
    momentum = 1.0 - _score01(f.momentum_score, f.composite_score)
    price_momentum = 1.0 - _score01(f.price_momentum_score, f.momentum_score or f.composite_score)
    leadership = 1.0 - _score01(f.leadership_score, f.composite_score)
    fundamental_momentum = 1.0 - _score01(f.fundamental_momentum_score, f.composite_score)
    return clamp(
        composite * 0.20
        + momentum * 0.25
        + price_momentum * 0.25
        + leadership * 0.20
        + fundamental_momentum * 0.10,
        0.0,
        1.0,
    )


def _long_price_alpha(bars: list[Bar], cfg: IntentConfig) -> tuple[float, dict[str, float]]:
    closes = [b.close for b in bars]
    close = closes[-1]
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    intermediate = _ret_pct(closes, cfg.intermediate_lookback_bars)
    confirmation = _ret_pct(closes, cfg.confirmation_bars)
    breakout_high = max(b.high for b in bars[-cfg.breakout_lookback_bars:])
    drawdown = (breakout_high - close) / breakout_high * 100.0 if breakout_high > 0.0 else 999.0
    breakout_gap = drawdown
    fast_ma = _mean(closes[-cfg.fast_ma_bars:])
    slow_ma = _mean(closes[-cfg.slow_ma_bars:])
    rsi = compute_rsi(closes[-50:])
    atr = _atr_pct(bars, cfg.atr_bars)

    trend_score = clamp((trend - cfg.min_trend_return_pct) / 35.0, 0.0, 1.0)
    intermediate_score = clamp((intermediate - cfg.min_intermediate_return_pct) / 25.0, 0.0, 1.0)
    confirmation_score = clamp((confirmation - cfg.min_confirmation_pct) / 12.0, 0.0, 1.0)
    drawdown_score = clamp(1.0 - drawdown / max(cfg.long_max_drawdown_pct, 0.01), 0.0, 1.0)
    breakout_score = clamp(1.0 - breakout_gap / max(cfg.long_max_breakout_gap_pct, 0.01), 0.0, 1.0)
    ma_score = 1.0 if close > fast_ma > slow_ma else (0.5 if close > slow_ma else 0.0)
    atr_score = clamp(1.0 - atr / max(cfg.max_atr_pct, 0.01), 0.0, 1.0)

    alpha = clamp(
        trend_score * 0.24
        + intermediate_score * 0.18
        + confirmation_score * 0.18
        + drawdown_score * 0.14
        + breakout_score * 0.12
        + ma_score * 0.09
        + atr_score * 0.05,
        0.0,
        1.0,
    )
    metrics = {
        "trend": trend,
        "intermediate": intermediate,
        "confirmation": confirmation,
        "drawdown": drawdown,
        "breakout_gap": breakout_gap,
        "rsi": rsi,
        "atr": atr,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
    }
    return alpha, metrics


def _short_price_alpha(bars: list[Bar], cfg: IntentConfig) -> tuple[float, dict[str, float]]:
    closes = [b.close for b in bars]
    close = closes[-1]
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    confirmation = _ret_pct(closes, cfg.confirmation_bars)
    lookback_low = min(b.low for b in bars[-cfg.breakout_lookback_bars:])
    bounce = (close - lookback_low) / lookback_low * 100.0 if lookback_low > 0.0 else 999.0
    fast_ma = _mean(closes[-cfg.fast_ma_bars:])
    slow_ma = _mean(closes[-cfg.slow_ma_bars:])
    rsi = compute_rsi(closes[-50:])
    atr = _atr_pct(bars, cfg.atr_bars)

    trend_score = clamp((cfg.max_short_trend_return_pct - trend) / 35.0, 0.0, 1.0)
    confirmation_score = clamp((cfg.max_short_confirmation_pct - confirmation) / 12.0, 0.0, 1.0)
    bounce_score = clamp(1.0 - bounce / max(cfg.short_max_bounce_pct, 0.01), 0.0, 1.0)
    ma_score = 1.0 if close < fast_ma < slow_ma else (0.5 if close < slow_ma else 0.0)
    atr_score = clamp(1.0 - atr / max(cfg.max_atr_pct, 0.01), 0.0, 1.0)
    alpha = clamp(trend_score * 0.35 + confirmation_score * 0.25 + bounce_score * 0.25 + ma_score * 0.10 + atr_score * 0.05, 0.0, 1.0)
    metrics = {"trend": trend, "confirmation": confirmation, "bounce": bounce, "rsi": rsi, "atr": atr, "fast_ma": fast_ma, "slow_ma": slow_ma}
    return alpha, metrics


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    scorer_alpha = _long_scorer_alpha(fundamental)
    price_alpha, m = _long_price_alpha(bars, cfg)
    swing_alpha = scorer_alpha * 0.45 + price_alpha * 0.55

    composite = float(fundamental.composite_score)
    momentum_score = float(fundamental.momentum_score if fundamental.momentum_score is not None else composite)
    price_momentum_score = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else momentum_score)
    leadership_score = float(fundamental.leadership_score if fundamental.leadership_score is not None else composite)

    if composite < cfg.min_long_composite_score:
        return IntentEvaluation(None, "rejected", "composite_below_min", f"Composite {composite:.1f} below minimum.")
    if momentum_score < cfg.min_long_momentum_score:
        return IntentEvaluation(None, "rejected", "momentum_score_below_min", f"Momentum score {momentum_score:.1f} below minimum.")
    if price_momentum_score < cfg.min_long_price_momentum_score:
        return IntentEvaluation(None, "rejected", "price_momentum_score_below_min", f"Price momentum score {price_momentum_score:.1f} below minimum.")
    if leadership_score < cfg.min_long_leadership_score:
        return IntentEvaluation(None, "rejected", "leadership_score_below_min", f"Leadership score {leadership_score:.1f} below minimum.")
    if scorer_alpha < cfg.min_long_scorer_alpha:
        return IntentEvaluation(None, "rejected", "scorer_alpha_below_min", f"Scorer alpha {scorer_alpha:.3f} below minimum.")
    if m["trend"] < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "trend_return_below_min", f"Trend return {m['trend']:.2f}% below minimum.")
    if m["intermediate"] < cfg.min_intermediate_return_pct:
        return IntentEvaluation(None, "rejected", "intermediate_return_below_min", f"Intermediate return {m['intermediate']:.2f}% below minimum.")
    if m["confirmation"] < cfg.min_confirmation_pct:
        return IntentEvaluation(None, "rejected", "confirmation_below_min", f"Confirmation return {m['confirmation']:.2f}% below minimum.")
    if m["drawdown"] > cfg.long_max_drawdown_pct:
        return IntentEvaluation(None, "rejected", "drawdown_above_max", f"Drawdown {m['drawdown']:.2f}% above maximum.")
    if m["breakout_gap"] > cfg.long_max_breakout_gap_pct:
        return IntentEvaluation(None, "rejected", "breakout_gap_above_max", f"Breakout gap {m['breakout_gap']:.2f}% above maximum.")
    if not (m["fast_ma"] > m["slow_ma"] and bars[-1].close > m["slow_ma"]):
        return IntentEvaluation(None, "rejected", "ma_structure_not_bullish", "Fast/slow moving-average structure is not bullish.")
    if m["rsi"] < cfg.long_min_rsi or m["rsi"] > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_range", f"RSI {m['rsi']:.2f} outside long range.")
    if m["atr"] > cfg.max_atr_pct:
        return IntentEvaluation(None, "rejected", "atr_pct_above_max", f"ATR {m['atr']:.2f}% above maximum.")
    if swing_alpha < cfg.min_long_swing_alpha:
        return IntentEvaluation(None, "rejected", "swing_alpha_below_min", f"Swing alpha {swing_alpha:.3f} below minimum.")

    reason = (
        f"SwingAlpha {swing_alpha:.3f} scorer {scorer_alpha:.3f} price {price_alpha:.3f} | "
        f"trend {m['trend']:.1f}% inter {m['intermediate']:.1f}% confirm {m['confirmation']:.1f}% "
        f"DD {m['drawdown']:.1f}% RSI {m['rsi']:.0f} | "
        f"scores C {composite:.0f} M {momentum_score:.0f} PM {price_momentum_score:.0f} L {leadership_score:.0f}"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "LONG", round(swing_alpha * 10.0, 4), reason),
        "intent",
        "swing_alpha_long_passed",
        reason,
    )


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    if not cfg.enable_shorts:
        return IntentEvaluation(None, "rejected", "shorts_disabled_by_model", "Swing-alpha shorts are disabled by default.")

    scorer_alpha = _short_scorer_alpha(fundamental)
    price_alpha, m = _short_price_alpha(bars, cfg)
    swing_alpha = scorer_alpha * 0.45 + price_alpha * 0.55

    composite = float(fundamental.composite_score)
    momentum_score = float(fundamental.momentum_score if fundamental.momentum_score is not None else composite)
    price_momentum_score = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else momentum_score)
    leadership_score = float(fundamental.leadership_score if fundamental.leadership_score is not None else composite)

    if composite > cfg.max_short_composite_score:
        return IntentEvaluation(None, "rejected", "composite_above_short_max", f"Composite {composite:.1f} above short maximum.")
    if momentum_score > cfg.max_short_momentum_score:
        return IntentEvaluation(None, "rejected", "momentum_score_above_short_max", f"Momentum score {momentum_score:.1f} above short maximum.")
    if price_momentum_score > cfg.max_short_price_momentum_score:
        return IntentEvaluation(None, "rejected", "price_momentum_score_above_short_max", f"Price momentum score {price_momentum_score:.1f} above short maximum.")
    if leadership_score > cfg.max_short_leadership_score:
        return IntentEvaluation(None, "rejected", "leadership_score_above_short_max", f"Leadership score {leadership_score:.1f} above short maximum.")
    if scorer_alpha < cfg.max_short_scorer_alpha:
        return IntentEvaluation(None, "rejected", "short_scorer_alpha_below_min", f"Short scorer alpha {scorer_alpha:.3f} below minimum.")
    if m["trend"] > cfg.max_short_trend_return_pct:
        return IntentEvaluation(None, "rejected", "short_trend_not_weak_enough", f"Trend return {m['trend']:.2f}% not weak enough.")
    if m["confirmation"] > cfg.max_short_confirmation_pct:
        return IntentEvaluation(None, "rejected", "short_confirmation_too_strong", f"Confirmation {m['confirmation']:.2f}% too strong.")
    if m["bounce"] > cfg.short_max_bounce_pct:
        return IntentEvaluation(None, "rejected", "short_bounce_above_max", f"Bounce {m['bounce']:.2f}% above maximum.")
    if m["rsi"] < cfg.short_min_rsi or m["rsi"] > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "short_rsi_outside_range", f"RSI {m['rsi']:.2f} outside short range.")
    if m["atr"] > cfg.max_atr_pct:
        return IntentEvaluation(None, "rejected", "short_atr_pct_above_max", f"ATR {m['atr']:.2f}% above maximum.")
    if swing_alpha < cfg.min_short_swing_alpha:
        return IntentEvaluation(None, "rejected", "short_swing_alpha_below_min", f"Swing alpha {swing_alpha:.3f} below minimum.")

    reason = (
        f"ShortSwingAlpha {swing_alpha:.3f} scorer {scorer_alpha:.3f} price {price_alpha:.3f} | "
        f"trend {m['trend']:.1f}% confirm {m['confirmation']:.1f}% bounce {m['bounce']:.1f}% RSI {m['rsi']:.0f}"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "SHORT", round(swing_alpha * 10.0, 4), reason),
        "intent",
        "swing_alpha_short_passed",
        reason,
    )
