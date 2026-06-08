"""Long scorer selector with a small regime-gated short hedge.

The long sleeve delegates to scorer_decile_long_selector_v2. The short sleeve
is intentionally narrow:
  - active only in configured stress regimes,
  - short max-hold is controlled by the central execution config,
  - candidates must be fundamentally weak and technically failing.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Optional

from backtest_shared import Bar, FundamentalRow, IntentEvaluation, TradeIntent
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, env_list, mean


MODEL_NAME = "scorer_decile_long_with_short_hedge_v1"

_LONG_PATH = Path(__file__).with_name("scorer_decile_long_selector_v2.py")
_SPEC = importlib.util.spec_from_file_location("_scorer_decile_long_selector_v2_base", _LONG_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load long selector model from {_LONG_PATH}")
_LONG = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _LONG
_SPEC.loader.exec_module(_LONG)

BENCHMARK_SYMBOL = _LONG.BENCHMARK_SYMBOL
BENCHMARK_SYMBOLS = _LONG.BENCHMARK_SYMBOLS
BENCHMARK_BAR_LOOKBACK = max(_LONG.BENCHMARK_BAR_LOOKBACK, 260)
IntentConfig = _LONG.IntentConfig


def _parse_csv_upper(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(value.strip().upper() for value in env_list(name, default) if value.strip())


def _parse_weekdays(name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    weekdays: set[int] = set()
    for raw in env_list(name, [str(value) for value in default]):
        value = int(raw)
        if value < 0 or value > 6:
            raise ValueError(f"{name} values must be integers from 0 Monday through 6 Sunday")
        weekdays.add(value)
    return tuple(sorted(weekdays))


def intent_config_from_env() -> IntentConfig:
    cfg = _LONG.intent_config_from_env()
    cfg.enable_shorts = env_bool("ENABLE_SHORTS", True)
    cfg.short_entry_weekdays = _parse_weekdays("SHORT_ENTRY_WEEKDAYS", (0, 1, 2, 3))
    cfg.short_allowed_world_regime_labels = _parse_csv_upper("SHORT_ALLOWED_WORLD_REGIME_LABELS", ("NEUTRAL", "DEFENSIVE"))
    cfg.short_allowed_daily_policy_phases = _parse_csv_upper(
        "SHORT_ALLOWED_DAILY_POLICY_PHASES",
        ("STRESS_BUILDING", "STRESS_SIDEWAYS", "STRESS_HIGH"),
    )
    cfg.short_allowed_regime_phase_pairs = _parse_csv_upper("SHORT_ALLOWED_REGIME_PHASE_PAIRS", ())
    cfg.short_require_benchmark_context = env_bool("SHORT_REQUIRE_BENCHMARK_CONTEXT", True)
    cfg.short_benchmark_lookback_bars = env_int("SHORT_BENCHMARK_LOOKBACK_BARS", 65)
    cfg.short_benchmark_confirmation_bars = env_int("SHORT_BENCHMARK_CONFIRMATION_BARS", 20)
    cfg.short_max_benchmark_return_pct = env_float("SHORT_MAX_BENCHMARK_RETURN_PCT", 3.0)
    cfg.short_max_benchmark_confirmation_pct = env_float("SHORT_MAX_BENCHMARK_CONFIRMATION_PCT", 2.0)

    cfg.max_short_composite_score = env_float("MAX_SHORT_COMPOSITE_SCORE", 38.0)
    cfg.max_short_momentum_score = env_float("MAX_SHORT_MOMENTUM_SCORE", 48.0)
    cfg.max_short_price_momentum_score = env_float("MAX_SHORT_PRICE_MOMENTUM_SCORE", 55.0)
    cfg.max_short_leadership_score = env_float("MAX_SHORT_LEADERSHIP_SCORE", 52.0)
    cfg.max_short_quality_score = env_float("MAX_SHORT_QUALITY_SCORE", 58.0)
    cfg.min_short_weakness_alpha = env_float("MIN_SHORT_WEAKNESS_ALPHA", 0.58)
    cfg.min_short_intent_score = env_float("MIN_SHORT_INTENT_SCORE", 5.4)

    cfg.short_trend_lookback_bars = env_int("SHORT_TREND_LOOKBACK_BARS", 130)
    cfg.short_swing_lookback_bars = env_int("SHORT_SWING_LOOKBACK_BARS", 65)
    cfg.short_confirmation_bars = env_int("SHORT_CONFIRMATION_BARS", 20)
    cfg.short_fast_ma_bars = env_int("SHORT_FAST_MA_BARS", 65)
    cfg.short_slow_ma_bars = env_int("SHORT_SLOW_MA_BARS", 130)
    cfg.short_breakdown_lookback_bars = env_int("SHORT_BREAKDOWN_LOOKBACK_BARS", 65)
    cfg.short_atr_bars = env_int("SHORT_ATR_BARS", 65)

    cfg.max_short_trend_return_pct = env_float("MAX_SHORT_TREND_RETURN_PCT", 6.0)
    cfg.max_short_swing_return_pct = env_float("MAX_SHORT_SWING_RETURN_PCT", 3.0)
    cfg.max_short_confirmation_return_pct = env_float("MAX_SHORT_CONFIRMATION_RETURN_PCT", 1.5)
    cfg.short_max_rsi = env_float("SHORT_MAX_RSI", 62.0)
    cfg.short_min_rsi = env_float("SHORT_MIN_RSI", 22.0)
    cfg.short_max_atr_pct = env_float("SHORT_MAX_ATR_PCT", 8.0)
    cfg.short_max_bounce_from_low_pct = env_float("SHORT_MAX_BOUNCE_FROM_LOW_PCT", 12.0)
    cfg.short_min_breakdown_from_high_pct = env_float("SHORT_MIN_BREAKDOWN_FROM_HIGH_PCT", 3.0)
    return cfg


def required_bar_lookback(cfg: IntentConfig) -> int:
    short_lookback = max(
        int(getattr(cfg, "short_trend_lookback_bars", 130)) + 1,
        int(getattr(cfg, "short_swing_lookback_bars", 65)) + 1,
        int(getattr(cfg, "short_confirmation_bars", 20)) + 1,
        int(getattr(cfg, "short_fast_ma_bars", 65)),
        int(getattr(cfg, "short_slow_ma_bars", 130)),
        int(getattr(cfg, "short_breakdown_lookback_bars", 65)),
        int(getattr(cfg, "short_atr_bars", 65)) + 1,
        int(getattr(cfg, "short_benchmark_lookback_bars", 65)) + 1,
        50,
    )
    return max(_LONG.required_bar_lookback(cfg), short_lookback)


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": copy.copy(base_cfg), "notes": f"grid model={MODEL_NAME}", "summary": {}}


def set_market_context(cfg: IntentConfig, as_of_ts, bars_by_symbol) -> None:
    _LONG.set_market_context(cfg, as_of_ts, bars_by_symbol)


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now, cfg: IntentConfig) -> IntentEvaluation:
    return _LONG.evaluate_long_intent(bars, fundamental, now, cfg)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def _score01(value: Optional[float], fallback: float = 50.0) -> float:
    return clamp(float(value if value is not None else fallback) / 100.0, 0.0, 1.0)


def _ret_pct(closes: list[float], lookback: int) -> float:
    if lookback <= 0 or len(closes) <= lookback:
        return 0.0
    base = closes[-lookback]
    return (closes[-1] / base - 1.0) * 100.0 if base > 0.0 else 0.0


def _sma(values: list[float], n: int) -> float:
    return mean(values[-n:]) if values and n > 0 and len(values) >= n else 0.0


def _atr_pct(bars: list[Bar], lookback: int) -> float:
    if len(bars) < lookback + 1:
        return 0.0
    ranges: list[float] = []
    subset = bars[-(lookback + 1):]
    for prev, cur in zip(subset, subset[1:]):
        ranges.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    close = bars[-1].close
    return mean(ranges) / close * 100.0 if close > 0.0 else 0.0


def _world_regime_label(cfg: IntentConfig) -> str:
    return str(getattr(cfg, "daily_policy_world_regime_label", "") or "").strip().upper()


def _daily_policy_phase(cfg: IntentConfig) -> str:
    return str(getattr(cfg, "daily_policy_phase", "") or "").strip().upper()


def _benchmark_bars(cfg: IntentConfig) -> list[Bar]:
    by_symbol = getattr(cfg, "market_context_bars_by_symbol", {}) or {}
    return list(by_symbol.get(BENCHMARK_SYMBOL, []))


def _short_regime_allowed(cfg: IntentConfig) -> tuple[bool, str, str]:
    label = _world_regime_label(cfg)
    phase = _daily_policy_phase(cfg)
    pair = f"{label}:{phase}" if label and phase else ""
    allowed_pairs = set(getattr(cfg, "short_allowed_regime_phase_pairs", ()))
    if pair and pair in allowed_pairs:
        return True, "", ""
    if label not in set(getattr(cfg, "short_allowed_world_regime_labels", ())):
        return False, "short_world_regime_not_allowed", f"World regime {label or '-'} is not allowed for hedge shorts."
    if phase not in set(getattr(cfg, "short_allowed_daily_policy_phases", ())):
        return False, "short_daily_policy_phase_not_allowed", f"Daily policy phase {phase or '-'} is not allowed for hedge shorts."
    return True, "", ""


def _benchmark_short_state(cfg: IntentConfig) -> tuple[bool, str, str, dict[str, float]]:
    bars = _benchmark_bars(cfg)
    required = max(
        int(getattr(cfg, "short_benchmark_lookback_bars", 65)) + 1,
        int(getattr(cfg, "short_benchmark_confirmation_bars", 20)) + 1,
    )
    if len(bars) < required:
        if bool(getattr(cfg, "short_require_benchmark_context", True)):
            return False, "short_benchmark_context_unavailable", "Benchmark bars are unavailable for hedge shorts.", {}
        return True, "", "", {}

    closes = [bar.close for bar in bars]
    ret = _ret_pct(closes, int(getattr(cfg, "short_benchmark_lookback_bars", 65)))
    confirmation = _ret_pct(closes, int(getattr(cfg, "short_benchmark_confirmation_bars", 20)))
    max_ret = float(getattr(cfg, "short_max_benchmark_return_pct", 3.0))
    max_confirmation = float(getattr(cfg, "short_max_benchmark_confirmation_pct", 2.0))
    state = {"benchmark_return": ret, "benchmark_confirmation": confirmation}
    if ret > max_ret:
        return False, "short_benchmark_return_too_strong", f"Benchmark return {ret:.2f}% above short maximum.", state
    if confirmation > max_confirmation:
        return False, "short_benchmark_confirmation_too_strong", f"Benchmark confirmation {confirmation:.2f}% above short maximum.", state
    return True, "", "", state


def _short_weakness_alpha(fundamental: FundamentalRow) -> float:
    composite_weak = 1.0 - _score01(fundamental.composite_score, 50.0)
    momentum_weak = 1.0 - _score01(fundamental.momentum_score, fundamental.composite_score)
    price_momentum_weak = 1.0 - _score01(
        fundamental.price_momentum_score,
        fundamental.momentum_score or fundamental.composite_score,
    )
    leadership_weak = 1.0 - _score01(fundamental.leadership_score, fundamental.composite_score)
    quality_weak = 1.0 - _score01(fundamental.quality_score, fundamental.composite_score)
    return clamp(
        composite_weak * 0.24
        + momentum_weak * 0.24
        + price_momentum_weak * 0.24
        + leadership_weak * 0.16
        + quality_weak * 0.12,
        0.0,
        1.0,
    )


def _short_price_alpha(bars: list[Bar], cfg: IntentConfig) -> tuple[float, dict[str, float]]:
    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    close = closes[-1]
    trend = _ret_pct(closes, int(getattr(cfg, "short_trend_lookback_bars", 130)))
    swing = _ret_pct(closes, int(getattr(cfg, "short_swing_lookback_bars", 65)))
    confirmation = _ret_pct(closes, int(getattr(cfg, "short_confirmation_bars", 20)))
    fast_ma = _sma(closes, int(getattr(cfg, "short_fast_ma_bars", 65)))
    slow_ma = _sma(closes, int(getattr(cfg, "short_slow_ma_bars", 130)))
    breakdown_high = max(highs[-int(getattr(cfg, "short_breakdown_lookback_bars", 65)):])
    recent_low = min(lows[-int(getattr(cfg, "short_breakdown_lookback_bars", 65)):])
    breakdown = (breakdown_high - close) / breakdown_high * 100.0 if breakdown_high > 0.0 else 0.0
    bounce = (close / recent_low - 1.0) * 100.0 if recent_low > 0.0 else 999.0
    rsi = compute_rsi(closes[-50:])
    atr = _atr_pct(bars, int(getattr(cfg, "short_atr_bars", 65)))

    trend_score = clamp((float(getattr(cfg, "max_short_trend_return_pct", 6.0)) - trend) / 24.0, 0.0, 1.0)
    swing_score = clamp((float(getattr(cfg, "max_short_swing_return_pct", 3.0)) - swing) / 16.0, 0.0, 1.0)
    confirmation_score = clamp(
        (float(getattr(cfg, "max_short_confirmation_return_pct", 1.5)) - confirmation) / 8.0,
        0.0,
        1.0,
    )
    ma_score = 1.0 if slow_ma > 0.0 and close < fast_ma < slow_ma else (0.6 if slow_ma > 0.0 and close < slow_ma else 0.0)
    breakdown_score = clamp(
        (breakdown - float(getattr(cfg, "short_min_breakdown_from_high_pct", 3.0))) / 18.0,
        0.0,
        1.0,
    )
    bounce_score = clamp(1.0 - bounce / max(float(getattr(cfg, "short_max_bounce_from_low_pct", 12.0)), 0.01), 0.0, 1.0)
    atr_score = clamp(1.0 - atr / max(float(getattr(cfg, "short_max_atr_pct", 8.0)), 0.01), 0.0, 1.0)

    alpha = clamp(
        trend_score * 0.20
        + swing_score * 0.20
        + confirmation_score * 0.18
        + ma_score * 0.16
        + breakdown_score * 0.12
        + bounce_score * 0.08
        + atr_score * 0.06,
        0.0,
        1.0,
    )
    return alpha, {
        "trend": trend,
        "swing": swing,
        "confirmation": confirmation,
        "breakdown": breakdown,
        "bounce": bounce,
        "rsi": rsi,
        "atr": atr,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
    }


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now, cfg: IntentConfig) -> IntentEvaluation:
    if not bool(getattr(cfg, "enable_shorts", False)):
        return IntentEvaluation(None, "rejected", "shorts_disabled_by_model", "Short hedge sleeve is disabled.")
    if str(fundamental.symbol or "").strip().upper() == BENCHMARK_SYMBOL:
        return IntentEvaluation(None, "rejected", "benchmark_symbol_context_only", "QQQ is market context only.")
    if getattr(cfg, "short_entry_weekdays", ()) and now.weekday() not in set(getattr(cfg, "short_entry_weekdays", ())):
        return IntentEvaluation(None, "rejected", "short_not_entry_weekday", f"Weekday {now.weekday()} is not a short entry weekday.")

    regime_ok, regime_code, regime_text = _short_regime_allowed(cfg)
    if not regime_ok:
        return IntentEvaluation(None, "rejected", regime_code, regime_text)
    benchmark_ok, benchmark_code, benchmark_text, benchmark = _benchmark_short_state(cfg)
    if not benchmark_ok:
        return IntentEvaluation(None, "rejected", benchmark_code, benchmark_text)

    composite = float(fundamental.composite_score)
    momentum = float(fundamental.momentum_score if fundamental.momentum_score is not None else composite)
    price_momentum = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else momentum)
    leadership = float(fundamental.leadership_score if fundamental.leadership_score is not None else composite)
    quality = float(fundamental.quality_score if fundamental.quality_score is not None else composite)

    if composite > float(getattr(cfg, "max_short_composite_score", 38.0)):
        return IntentEvaluation(None, "rejected", "short_composite_above_max", f"Composite {composite:.1f} above short maximum.")
    if momentum > float(getattr(cfg, "max_short_momentum_score", 48.0)):
        return IntentEvaluation(None, "rejected", "short_momentum_above_max", f"Momentum {momentum:.1f} above short maximum.")
    if price_momentum > float(getattr(cfg, "max_short_price_momentum_score", 55.0)):
        return IntentEvaluation(None, "rejected", "short_price_momentum_above_max", f"Price momentum {price_momentum:.1f} above short maximum.")
    if leadership > float(getattr(cfg, "max_short_leadership_score", 52.0)):
        return IntentEvaluation(None, "rejected", "short_leadership_above_max", f"Leadership {leadership:.1f} above short maximum.")
    if quality > float(getattr(cfg, "max_short_quality_score", 58.0)):
        return IntentEvaluation(None, "rejected", "short_quality_above_max", f"Quality {quality:.1f} above short maximum.")

    weakness_alpha = _short_weakness_alpha(fundamental)
    if weakness_alpha < float(getattr(cfg, "min_short_weakness_alpha", 0.58)):
        return IntentEvaluation(None, "rejected", "short_weakness_alpha_below_min", f"Weakness alpha {weakness_alpha:.3f} below minimum.")

    price_alpha, m = _short_price_alpha(bars, cfg)
    if m["trend"] > float(getattr(cfg, "max_short_trend_return_pct", 6.0)):
        return IntentEvaluation(None, "rejected", "short_trend_too_strong", f"Trend return {m['trend']:.2f}% above short maximum.")
    if m["swing"] > float(getattr(cfg, "max_short_swing_return_pct", 3.0)):
        return IntentEvaluation(None, "rejected", "short_swing_too_strong", f"Swing return {m['swing']:.2f}% above short maximum.")
    if m["confirmation"] > float(getattr(cfg, "max_short_confirmation_return_pct", 1.5)):
        return IntentEvaluation(None, "rejected", "short_confirmation_too_strong", f"Confirmation {m['confirmation']:.2f}% above short maximum.")
    if m["rsi"] < float(getattr(cfg, "short_min_rsi", 22.0)) or m["rsi"] > float(getattr(cfg, "short_max_rsi", 62.0)):
        return IntentEvaluation(None, "rejected", "short_rsi_outside_range", f"RSI {m['rsi']:.2f} outside short range.")
    if m["atr"] > float(getattr(cfg, "short_max_atr_pct", 8.0)):
        return IntentEvaluation(None, "rejected", "short_atr_above_max", f"ATR {m['atr']:.2f}% above maximum.")
    if m["breakdown"] < float(getattr(cfg, "short_min_breakdown_from_high_pct", 3.0)):
        return IntentEvaluation(None, "rejected", "short_breakdown_below_min", f"Breakdown {m['breakdown']:.2f}% below minimum.")
    if m["bounce"] > float(getattr(cfg, "short_max_bounce_from_low_pct", 12.0)):
        return IntentEvaluation(None, "rejected", "short_bounce_above_max", f"Bounce {m['bounce']:.2f}% above maximum.")

    benchmark_weakness = clamp(
        (
            float(getattr(cfg, "short_max_benchmark_return_pct", 3.0))
            - float(benchmark.get("benchmark_return", 0.0))
        ) / 12.0,
        0.0,
        1.0,
    )
    combined = (weakness_alpha * 0.42 + price_alpha * 0.43 + benchmark_weakness * 0.15) * 10.0
    if combined < float(getattr(cfg, "min_short_intent_score", 5.4)):
        return IntentEvaluation(None, "rejected", "short_intent_score_below_min", f"Short intent score {combined:.2f} below minimum.")

    reason = (
        f"ShortHedge alpha {combined / 10.0:.3f} weakness {weakness_alpha:.3f} price {price_alpha:.3f} "
        f"bench {benchmark_weakness:.3f} | scores C {composite:.1f} M {momentum:.1f} "
        f"PM {price_momentum:.1f} L {leadership:.1f} Q {quality:.1f} | "
        f"trend {m['trend']:.1f}% swing {m['swing']:.1f}% confirm {m['confirmation']:.1f}% "
        f"breakdown {m['breakdown']:.1f}% bounce {m['bounce']:.1f}% RSI {m['rsi']:.0f} ATR {m['atr']:.1f}% | "
        f"QQQ {benchmark.get('benchmark_return', 0.0):.1f}%/{benchmark.get('benchmark_confirmation', 0.0):.1f}% | "
        f"regime {_world_regime_label(cfg) or '-'} phase {_daily_policy_phase(cfg) or '-'}"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason),
        "intent",
        "short_hedge_passed",
        reason,
    )


def evaluate_position_exit(pos, ts, open_, high, low, close, total_bars, cfg: IntentConfig, *, exit_active: bool):
    return _LONG.evaluate_position_exit(
        pos,
        ts,
        open_,
        high,
        low,
        close,
        total_bars,
        cfg,
        exit_active=exit_active,
    )
