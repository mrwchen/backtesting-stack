"""V2 long-only scorer selector.

This version keeps the v1 selector mechanics, but uses additional config
guards for the two issues observed in the first live backtest:
  - overextended 130-hour momentum,
  - score saturation at price_momentum_score=100.

V2 also adds model-local ranking and exit guards for the second study:
  - QQQ-relative strength must be acceptable before a stock can take a slot,
  - sector and leadership constraints can block weak profiles,
  - slot priority favors QQQ-relative leaders,
  - dead-money exits can react to QQQ-relative underperformance.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path


MODEL_NAME = "scorer_decile_long_selector_v2"

_BASE_PATH = Path(__file__).with_name("scorer_decile_long_selector_v1.py")
_SPEC = importlib.util.spec_from_file_location("_scorer_decile_long_selector_v1_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load base scorer selector model from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

BENCHMARK_SYMBOL = _BASE.BENCHMARK_SYMBOL
BENCHMARK_SYMBOLS = _BASE.BENCHMARK_SYMBOLS
BENCHMARK_BAR_LOOKBACK = _BASE.BENCHMARK_BAR_LOOKBACK
IntentConfig = _BASE.IntentConfig


def _label_set(name: str, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    return tuple(value.strip().upper() for value in _BASE.env_list(name, default) if value.strip())


def intent_config_from_env() -> IntentConfig:
    cfg = _BASE.intent_config_from_env()
    setattr(cfg, "v2_relative_strength_enabled", _BASE.env_bool("V2_RELATIVE_STRENGTH_ENABLED", True))
    setattr(cfg, "v2_require_relative_context", _BASE.env_bool("V2_REQUIRE_RELATIVE_CONTEXT", True))
    setattr(
        cfg,
        "v2_min_qqq_relative_short_return_pct",
        _BASE.env_float("V2_MIN_QQQ_RELATIVE_SHORT_RETURN_PCT", -1.0),
    )
    setattr(
        cfg,
        "v2_min_qqq_relative_mid_return_pct",
        _BASE.env_float("V2_MIN_QQQ_RELATIVE_MID_RETURN_PCT", -2.0),
    )
    setattr(
        cfg,
        "v2_min_qqq_relative_long_return_pct",
        _BASE.env_optional_float("V2_MIN_QQQ_RELATIVE_LONG_RETURN_PCT", None),
    )
    setattr(
        cfg,
        "v2_max_qqq_relative_decay_pct",
        _BASE.env_optional_float("V2_MAX_QQQ_RELATIVE_DECAY_PCT", 8.0),
    )
    setattr(
        cfg,
        "v2_strong_qqq_short_return_pct",
        _BASE.env_float("V2_STRONG_QQQ_SHORT_RETURN_PCT", 3.0),
    )
    setattr(
        cfg,
        "v2_strong_qqq_mid_return_pct",
        _BASE.env_float("V2_STRONG_QQQ_MID_RETURN_PCT", 6.0),
    )
    setattr(
        cfg,
        "v2_strong_qqq_min_relative_short_return_pct",
        _BASE.env_float("V2_STRONG_QQQ_MIN_RELATIVE_SHORT_RETURN_PCT", 0.0),
    )
    setattr(
        cfg,
        "v2_strong_qqq_min_relative_mid_return_pct",
        _BASE.env_float("V2_STRONG_QQQ_MIN_RELATIVE_MID_RETURN_PCT", -1.0),
    )
    setattr(
        cfg,
        "v2_min_leadership_score",
        _BASE.env_optional_float("V2_MIN_LEADERSHIP_SCORE", None),
    )
    setattr(
        cfg,
        "v2_strong_qqq_min_leadership_score",
        _BASE.env_optional_float("V2_STRONG_QQQ_MIN_LEADERSHIP_SCORE", 58.0),
    )
    setattr(cfg, "v2_allowed_long_sectors", _label_set("V2_ALLOWED_LONG_SECTORS"))
    setattr(cfg, "v2_blocked_long_sectors", _label_set("V2_BLOCKED_LONG_SECTORS"))
    setattr(
        cfg,
        "v2_slot_priority_weight",
        _BASE.env_float("V2_SLOT_PRIORITY_WEIGHT", 0.35),
    )
    setattr(cfg, "v2_short_require_stress_context", _BASE.env_bool("V2_SHORT_REQUIRE_STRESS_CONTEXT", True))
    setattr(
        cfg,
        "v2_short_min_world_regime_ma_score",
        _BASE.env_optional_float("V2_SHORT_MIN_WORLD_REGIME_MA_SCORE", 60.0),
    )
    setattr(
        cfg,
        "v2_short_allowed_daily_policy_phases",
        _label_set("V2_SHORT_ALLOWED_DAILY_POLICY_PHASES", ("STRESS_HIGH", "STRESS_BUILDING", "STRESS_SIDEWAYS")),
    )
    setattr(
        cfg,
        "v2_short_max_qqq_relative_short_return_pct",
        _BASE.env_float("V2_SHORT_MAX_QQQ_RELATIVE_SHORT_RETURN_PCT", 0.0),
    )
    setattr(
        cfg,
        "v2_short_max_qqq_relative_mid_return_pct",
        _BASE.env_float("V2_SHORT_MAX_QQQ_RELATIVE_MID_RETURN_PCT", 2.0),
    )
    setattr(
        cfg,
        "v2_short_confirm_relative_short_return_pct",
        _BASE.env_float("V2_SHORT_CONFIRM_RELATIVE_SHORT_RETURN_PCT", -3.0),
    )
    setattr(
        cfg,
        "v2_short_confirm_relative_mid_return_pct",
        _BASE.env_float("V2_SHORT_CONFIRM_RELATIVE_MID_RETURN_PCT", -2.0),
    )
    setattr(
        cfg,
        "v2_short_max_stock_short_return_pct",
        _BASE.env_float("V2_SHORT_MAX_STOCK_SHORT_RETURN_PCT", 2.0),
    )
    setattr(
        cfg,
        "v2_short_max_stock_mid_return_pct",
        _BASE.env_float("V2_SHORT_MAX_STOCK_MID_RETURN_PCT", 10.0),
    )
    setattr(
        cfg,
        "v2_short_confirm_stock_short_return_pct",
        _BASE.env_float("V2_SHORT_CONFIRM_STOCK_SHORT_RETURN_PCT", -3.0),
    )
    setattr(cfg, "v2_short_min_drawdown_pct", _BASE.env_float("V2_SHORT_MIN_DRAWDOWN_PCT", 3.0))
    setattr(cfg, "v2_short_max_drawdown_pct", _BASE.env_float("V2_SHORT_MAX_DRAWDOWN_PCT", 35.0))
    setattr(cfg, "v2_short_max_bounce_from_low_pct", _BASE.env_float("V2_SHORT_MAX_BOUNCE_FROM_LOW_PCT", 14.0))
    setattr(cfg, "v2_short_min_rsi", _BASE.env_float("V2_SHORT_MIN_RSI", 25.0))
    setattr(cfg, "v2_short_max_rsi", _BASE.env_float("V2_SHORT_MAX_RSI", 68.0))
    setattr(cfg, "v2_short_max_atr_pct", _BASE.env_float("V2_SHORT_MAX_ATR_PCT", 8.0))
    setattr(
        cfg,
        "v2_short_strong_qqq_min_underperformance_pct",
        _BASE.env_float("V2_SHORT_STRONG_QQQ_MIN_UNDERPERFORMANCE_PCT", 4.0),
    )
    setattr(
        cfg,
        "v2_short_strong_qqq_max_stock_short_return_pct",
        _BASE.env_float("V2_SHORT_STRONG_QQQ_MAX_STOCK_SHORT_RETURN_PCT", 0.0),
    )
    setattr(cfg, "v2_short_min_intent_score", _BASE.env_float("V2_SHORT_MIN_INTENT_SCORE", 5.8))
    setattr(
        cfg,
        "v2_relative_dead_money_exit_enabled",
        _BASE.env_bool("V2_RELATIVE_DEAD_MONEY_EXIT_ENABLED", True),
    )
    setattr(
        cfg,
        "v2_relative_dead_money_exit_min_bars",
        _BASE.env_int("V2_RELATIVE_DEAD_MONEY_EXIT_MIN_BARS", 130),
    )
    setattr(
        cfg,
        "v2_relative_dead_money_max_return_pct",
        _BASE.env_float("V2_RELATIVE_DEAD_MONEY_MAX_RETURN_PCT", 3.0),
    )
    setattr(
        cfg,
        "v2_relative_dead_money_min_qqq_return_pct",
        _BASE.env_float("V2_RELATIVE_DEAD_MONEY_MIN_QQQ_RETURN_PCT", 2.0),
    )
    setattr(
        cfg,
        "v2_relative_dead_money_underperformance_pct",
        _BASE.env_float("V2_RELATIVE_DEAD_MONEY_UNDERPERFORMANCE_PCT", 7.0),
    )
    setattr(
        cfg,
        "v2_relative_dead_money_mfe_cap_pct",
        _BASE.env_optional_float("V2_RELATIVE_DEAD_MONEY_MFE_CAP_PCT", 8.0),
    )
    return cfg


def required_bar_lookback(cfg: IntentConfig) -> int:
    return _BASE.required_bar_lookback(cfg)


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": f"grid model={MODEL_NAME}", "summary": {}}


def set_market_context(cfg: IntentConfig, as_of_ts, bars_by_symbol) -> None:
    _BASE.set_market_context(cfg, as_of_ts, bars_by_symbol)


def compute_long_intent(bars, fundamental, now, cfg):
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars, fundamental, now, cfg):
    base_eval = _BASE.evaluate_long_intent(bars, fundamental, now, cfg)
    if base_eval.intent is None:
        return base_eval

    sector = str(getattr(fundamental, "sector", "") or "").strip().upper()
    allowed_sectors = tuple(getattr(cfg, "v2_allowed_long_sectors", ()) or ())
    blocked_sectors = tuple(getattr(cfg, "v2_blocked_long_sectors", ()) or ())
    if allowed_sectors and not _matches_any_sector(sector, allowed_sectors):
        return _reject("v2_sector_not_allowed", f"Sector {sector or '-'} is not in the v2 allowed sector list.")
    if blocked_sectors and _matches_any_sector(sector, blocked_sectors):
        return _reject("v2_sector_blocked", f"Sector {sector or '-'} is blocked by v2 sector filter.")

    leadership = float(
        getattr(fundamental, "leadership_score", None)
        if getattr(fundamental, "leadership_score", None) is not None
        else getattr(fundamental, "composite_score", 0.0)
    )
    min_leadership = getattr(cfg, "v2_min_leadership_score", None)
    if min_leadership is not None and leadership < float(min_leadership):
        return _reject("v2_leadership_below_min", f"Leadership {leadership:.1f} below v2 minimum.")

    profile = _relative_profile(bars, cfg)
    if bool(getattr(cfg, "v2_relative_strength_enabled", True)):
        if profile is None:
            if bool(getattr(cfg, "v2_require_relative_context", True)):
                return _reject("v2_relative_context_unavailable", "QQQ-relative context is unavailable.")
        else:
            blocked = _relative_filter_decision(profile, leadership, cfg)
            if blocked is not None:
                return blocked

    if profile is None:
        return base_eval

    score, reason = _adjusted_intent_score_and_reason(base_eval.intent, profile, leadership, cfg)
    if score < cfg.min_long_intent_score:
        return _reject(
            "v2_adjusted_intent_score_below_min",
            f"V2 adjusted intent score {score:.2f} below minimum.",
        )
    return _BASE.IntentEvaluation(
        dataclasses.replace(base_eval.intent, score=round(score, 4), reason=reason),
        "intent",
        "scorer_decile_long_selector_v2_passed",
        reason,
    )


def compute_short_intent(bars, fundamental, now, cfg):
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars, fundamental, now, cfg):
    if not bool(getattr(cfg, "enable_shorts", False)):
        return _reject("shorts_disabled_by_model", "V2 short side is disabled.")
    symbol = str(getattr(fundamental, "symbol", "") or "").strip().upper()
    if symbol == cfg.benchmark_symbol.strip().upper():
        return _reject("benchmark_symbol_context_only", "QQQ is market context only.")
    if cfg.entry_weekdays and now.weekday() not in cfg.entry_weekdays:
        return _reject("short_not_rebalance_weekday", f"Weekday {now.weekday()} is not a short entry weekday.")

    stress_ok, stress_code, stress_text, stress_score = _short_stress_state(cfg)
    if not stress_ok:
        return _reject(stress_code, stress_text)

    market_cap = float(getattr(fundamental, "market_cap_m", None) or 0.0)
    if market_cap < cfg.min_market_cap_m:
        return _reject("short_market_cap_below_model_min", f"Market cap {market_cap:.0f}m below model minimum.")

    profile = _relative_profile(bars, cfg)
    if profile is None:
        if bool(getattr(cfg, "v2_require_relative_context", True)):
            return _reject("v2_short_relative_context_unavailable", "QQQ-relative context is unavailable.")
        return _reject("v2_short_relative_context_required", "V2 shorts require QQQ-relative context.")

    price = _short_price_profile(bars, cfg)
    blocked = _short_price_filter_decision(profile, price, cfg)
    if blocked is not None:
        return blocked

    composite = float(getattr(fundamental, "composite_score", 0.0) or 0.0)
    momentum = float(
        getattr(fundamental, "momentum_score", None)
        if getattr(fundamental, "momentum_score", None) is not None
        else composite
    )
    price_momentum = float(
        getattr(fundamental, "price_momentum_score", None)
        if getattr(fundamental, "price_momentum_score", None) is not None
        else momentum
    )
    leadership = float(
        getattr(fundamental, "leadership_score", None)
        if getattr(fundamental, "leadership_score", None) is not None
        else composite
    )
    valuation_pressure = _short_valuation_pressure(fundamental)
    combined, components = _short_intent_score(profile, price, valuation_pressure, stress_score, cfg)
    if combined < float(getattr(cfg, "v2_short_min_intent_score", 5.8)):
        return _reject("v2_short_intent_score_below_min", f"V2 short intent score {combined:.2f} below minimum.")

    valuation_label = str(getattr(fundamental, "valuation_label", "") or "-").strip() or "-"
    reason = (
        f"ScorerSelectorShort alpha {combined / 10.0:.3f} rel {components['relative']:.3f} "
        f"price {components['price']:.3f} stress {components['stress']:.3f} val {valuation_pressure:.3f} | "
        f"relS {profile['rel_short']:.1f}% relM {profile['rel_mid']:.1f}% "
        f"stockS {profile['stock_short']:.1f}% stockM {profile['stock_mid']:.1f}% "
        f"qS {profile['qqq_short']:.1f}% qM {profile['qqq_mid']:.1f}% | "
        f"DD {price['drawdown']:.1f}% bounce {price['bounce']:.1f}% RSI {price['rsi']:.0f} "
        f"ATR {price['atr']:.1f}% C {composite:.1f} M {momentum:.1f} PM {price_momentum:.1f} "
        f"L {leadership:.1f} val {valuation_label} cap {market_cap:.0f}m | "
        f"regime {_BASE._world_regime_label(cfg) or '-'} phase {_BASE._daily_policy_phase(cfg) or '-'}"
    )
    return _BASE.IntentEvaluation(
        _BASE.TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason),
        "intent",
        "scorer_decile_long_selector_v2_short_passed",
        reason,
    )


def evaluate_position_exit(pos, ts, open_, high, low, close, total_bars, cfg, *, exit_active: bool):
    v2_exit = _relative_dead_money_exit(pos, ts, high, low, close, total_bars, cfg, exit_active=exit_active)
    if v2_exit is not None:
        return v2_exit
    return _BASE.evaluate_position_exit(
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


def _reject(reason_code: str, reason_text: str):
    return _BASE.IntentEvaluation(None, "rejected", reason_code, reason_text)


def _matches_any_sector(sector: str, patterns: tuple[str, ...]) -> bool:
    sector = sector.strip().upper()
    return any(sector == pattern or sector.startswith(pattern) for pattern in patterns)


def _daily_policy_world_regime_ma_score(cfg) -> float | None:
    value = getattr(cfg, "daily_policy_world_regime_ma_score", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_stress_state(cfg) -> tuple[bool, str, str, float]:
    phase = _BASE._daily_policy_phase(cfg)
    allowed_phases = tuple(getattr(cfg, "v2_short_allowed_daily_policy_phases", ()) or ())
    if allowed_phases and phase and phase not in set(allowed_phases):
        return False, "v2_short_daily_policy_phase_not_allowed", f"Daily policy phase {phase} is not allowed for shorts.", 0.0

    ma_score = _daily_policy_world_regime_ma_score(cfg)
    min_ma = getattr(cfg, "v2_short_min_world_regime_ma_score", 60.0)
    if ma_score is None:
        if bool(getattr(cfg, "v2_short_require_stress_context", True)):
            return False, "v2_short_stress_context_unavailable", "Daily policy stress context is unavailable.", 0.0
        return True, "", "", 0.0
    if min_ma is not None and ma_score < float(min_ma):
        return (
            False,
            "v2_short_stress_below_min",
            f"World-regime MA score {ma_score:.2f} below v2 short stress minimum.",
            _BASE.clamp(ma_score / 100.0, 0.0, 1.0),
        )
    return True, "", "", _BASE.clamp(ma_score / 100.0, 0.0, 1.0)


def _benchmark_returns(cfg) -> dict[str, float] | None:
    bars = _BASE._benchmark_bars(cfg)
    required = max(cfg.short_lookback_bars, cfg.mid_lookback_bars, cfg.long_lookback_bars) + 1
    if len(bars) < required:
        return None
    closes = [float(bar.close) for bar in bars]
    return {
        "short": _BASE._ret_pct(closes, cfg.short_lookback_bars),
        "mid": _BASE._ret_pct(closes, cfg.mid_lookback_bars),
        "long": _BASE._ret_pct(closes, cfg.long_lookback_bars),
    }


def _relative_profile(bars, cfg) -> dict[str, float] | None:
    benchmark = _benchmark_returns(cfg)
    if benchmark is None:
        return None
    price_alpha, price = _BASE._price_alpha(bars, cfg)
    rel_short = float(price["short_return"]) - benchmark["short"]
    rel_mid = float(price["mid_return"]) - benchmark["mid"]
    rel_long = float(price["long_return"]) - benchmark["long"]
    return {
        "price_alpha": float(price_alpha),
        "stock_short": float(price["short_return"]),
        "stock_mid": float(price["mid_return"]),
        "stock_long": float(price["long_return"]),
        "qqq_short": benchmark["short"],
        "qqq_mid": benchmark["mid"],
        "qqq_long": benchmark["long"],
        "rel_short": rel_short,
        "rel_mid": rel_mid,
        "rel_long": rel_long,
        "rel_decay": rel_mid - rel_short,
    }


def _short_price_profile(bars, cfg) -> dict[str, float]:
    price_alpha, price = _BASE._price_alpha(bars, cfg)
    lows = [float(bar.low) for bar in bars]
    close = float(bars[-1].close)
    bounce_lookback = max(1, min(len(lows), int(getattr(cfg, "mid_lookback_bars", 130))))
    recent_low = min(lows[-bounce_lookback:])
    bounce = (close / recent_low - 1.0) * 100.0 if recent_low > 0.0 else 999.0
    price = dict(price)
    price["price_alpha"] = float(price_alpha)
    price["close"] = close
    price["bounce"] = bounce
    return price


def _short_price_filter_decision(profile: dict[str, float], price: dict[str, float], cfg):
    max_rel_short = float(getattr(cfg, "v2_short_max_qqq_relative_short_return_pct", 0.0))
    if profile["rel_short"] > max_rel_short:
        return _reject(
            "v2_short_relative_short_above_max",
            f"QQQ-relative short return {profile['rel_short']:.2f}% above v2 short maximum.",
        )
    max_rel_mid = float(getattr(cfg, "v2_short_max_qqq_relative_mid_return_pct", 2.0))
    if profile["rel_mid"] > max_rel_mid:
        return _reject(
            "v2_short_relative_mid_above_max",
            f"QQQ-relative mid return {profile['rel_mid']:.2f}% above v2 short maximum.",
        )
    max_stock_short = float(getattr(cfg, "v2_short_max_stock_short_return_pct", 2.0))
    if profile["stock_short"] > max_stock_short:
        return _reject(
            "v2_short_stock_short_return_above_max",
            f"Stock short return {profile['stock_short']:.2f}% above v2 short maximum.",
        )
    max_stock_mid = float(getattr(cfg, "v2_short_max_stock_mid_return_pct", 10.0))
    if profile["stock_mid"] > max_stock_mid:
        return _reject(
            "v2_short_stock_mid_return_above_max",
            f"Stock mid return {profile['stock_mid']:.2f}% above v2 short maximum.",
        )

    confirmed = (
        profile["rel_short"] <= float(getattr(cfg, "v2_short_confirm_relative_short_return_pct", -3.0))
        or profile["rel_mid"] <= float(getattr(cfg, "v2_short_confirm_relative_mid_return_pct", -2.0))
        or profile["stock_short"] <= float(getattr(cfg, "v2_short_confirm_stock_short_return_pct", -3.0))
        or (
            price["slow_ma"] > 0.0
            and price["fast_ma"] > 0.0
            and price["close"] < price["fast_ma"] < price["slow_ma"]
        )
    )
    if not confirmed:
        return _reject("v2_short_confirmation_missing", "V2 short requires confirmed relative or absolute weakness.")

    min_drawdown = float(getattr(cfg, "v2_short_min_drawdown_pct", 3.0))
    if price["drawdown"] < min_drawdown:
        return _reject("v2_short_drawdown_below_min", f"Drawdown {price['drawdown']:.2f}% below v2 short minimum.")
    max_drawdown = float(getattr(cfg, "v2_short_max_drawdown_pct", 35.0))
    if price["drawdown"] > max_drawdown:
        return _reject("v2_short_drawdown_above_max", f"Drawdown {price['drawdown']:.2f}% above v2 short maximum.")
    max_bounce = float(getattr(cfg, "v2_short_max_bounce_from_low_pct", 14.0))
    if price["bounce"] > max_bounce:
        return _reject("v2_short_bounce_above_max", f"Bounce {price['bounce']:.2f}% above v2 short maximum.")
    min_rsi = float(getattr(cfg, "v2_short_min_rsi", 25.0))
    max_rsi = float(getattr(cfg, "v2_short_max_rsi", 68.0))
    if price["rsi"] < min_rsi or price["rsi"] > max_rsi:
        return _reject("v2_short_rsi_outside_range", f"RSI {price['rsi']:.2f} outside v2 short range.")
    max_atr = float(getattr(cfg, "v2_short_max_atr_pct", 8.0))
    if price["atr"] > max_atr:
        return _reject("v2_short_atr_above_max", f"ATR {price['atr']:.2f}% above v2 short maximum.")

    strong_qqq = (
        profile["qqq_short"] >= float(getattr(cfg, "v2_strong_qqq_short_return_pct", 3.0))
        or profile["qqq_mid"] >= float(getattr(cfg, "v2_strong_qqq_mid_return_pct", 6.0))
    )
    if strong_qqq:
        required_underperformance = float(getattr(cfg, "v2_short_strong_qqq_min_underperformance_pct", 4.0))
        if profile["rel_short"] > -required_underperformance:
            return _reject(
                "v2_short_strong_qqq_underperformance_below_min",
                f"QQQ-relative short underperformance {-profile['rel_short']:.2f}% below strong-QQQ minimum.",
            )
        strong_max_stock_short = float(getattr(cfg, "v2_short_strong_qqq_max_stock_short_return_pct", 0.0))
        if profile["stock_short"] > strong_max_stock_short:
            return _reject(
                "v2_short_strong_qqq_stock_short_return_above_max",
                f"Stock short return {profile['stock_short']:.2f}% above strong-QQQ short maximum.",
            )
    return None


def _short_valuation_pressure(fundamental) -> float:
    label = str(getattr(fundamental, "valuation_label", "") or "").strip().lower()
    label_score = {
        "overvalued_weak": 1.00,
        "overvalued": 0.82,
        "expensive_quality": 0.55,
        "fairly_valued": 0.28,
        "speculative": 0.70,
    }.get(label, 0.15)
    mispricing = getattr(fundamental, "mispricing_score", None)
    mispricing_pressure = 0.5
    if mispricing is not None:
        mispricing_pressure = _BASE.clamp((55.0 - float(mispricing)) / 35.0, 0.0, 1.0)
    composite = float(getattr(fundamental, "composite_score", 50.0) or 50.0)
    fundamental_pressure = _BASE.clamp((65.0 - composite) / 40.0, 0.0, 1.0)
    return _BASE.clamp(label_score * 0.50 + mispricing_pressure * 0.35 + fundamental_pressure * 0.15, 0.0, 1.0)


def _score_below_ceiling(value: float, ceiling: float, span: float) -> float:
    return _BASE.clamp((ceiling - value) / max(span, 0.01), 0.0, 1.0)


def _short_intent_score(
    profile: dict[str, float],
    price: dict[str, float],
    valuation_pressure: float,
    stress_score: float,
    cfg,
) -> tuple[float, dict[str, float]]:
    rel_short_score = _score_below_ceiling(
        profile["rel_short"],
        float(getattr(cfg, "v2_short_max_qqq_relative_short_return_pct", 0.0)),
        12.0,
    )
    rel_mid_score = _score_below_ceiling(
        profile["rel_mid"],
        float(getattr(cfg, "v2_short_max_qqq_relative_mid_return_pct", 2.0)),
        16.0,
    )
    stock_short_score = _score_below_ceiling(
        profile["stock_short"],
        float(getattr(cfg, "v2_short_max_stock_short_return_pct", 2.0)),
        12.0,
    )
    stock_mid_score = _score_below_ceiling(
        profile["stock_mid"],
        float(getattr(cfg, "v2_short_max_stock_mid_return_pct", 10.0)),
        24.0,
    )
    relative_score = _BASE.clamp(rel_short_score * 0.58 + rel_mid_score * 0.42, 0.0, 1.0)
    ma_score = 1.0 if price["slow_ma"] > 0.0 and price["fast_ma"] < price["slow_ma"] else 0.0
    drawdown_score = _BASE.clamp(
        (price["drawdown"] - float(getattr(cfg, "v2_short_min_drawdown_pct", 3.0))) / 18.0,
        0.0,
        1.0,
    )
    bounce_score = _BASE.clamp(
        1.0 - price["bounce"] / max(float(getattr(cfg, "v2_short_max_bounce_from_low_pct", 14.0)), 0.01),
        0.0,
        1.0,
    )
    atr_score = _BASE.clamp(1.0 - price["atr"] / max(float(getattr(cfg, "v2_short_max_atr_pct", 8.0)), 0.01), 0.0, 1.0)
    price_score = _BASE.clamp(
        stock_short_score * 0.32
        + stock_mid_score * 0.22
        + ma_score * 0.18
        + drawdown_score * 0.14
        + bounce_score * 0.08
        + atr_score * 0.06,
        0.0,
        1.0,
    )
    qqq_strength = _BASE.clamp((profile["qqq_short"] + 4.0) / 16.0, 0.0, 1.0)
    combined = (
        relative_score * 0.38
        + price_score * 0.34
        + valuation_pressure * 0.12
        + stress_score * 0.10
        + qqq_strength * 0.06
    ) * 10.0
    return _BASE.clamp(combined, 0.0, 10.0), {
        "relative": relative_score,
        "price": price_score,
        "stress": stress_score,
    }


def _relative_filter_decision(profile: dict[str, float], leadership: float, cfg):
    if profile["rel_short"] < float(getattr(cfg, "v2_min_qqq_relative_short_return_pct", -1.0)):
        return _reject(
            "v2_relative_short_below_min",
            f"QQQ-relative short return {profile['rel_short']:.2f}% below v2 minimum.",
        )
    if profile["rel_mid"] < float(getattr(cfg, "v2_min_qqq_relative_mid_return_pct", -2.0)):
        return _reject(
            "v2_relative_mid_below_min",
            f"QQQ-relative mid return {profile['rel_mid']:.2f}% below v2 minimum.",
        )
    min_rel_long = getattr(cfg, "v2_min_qqq_relative_long_return_pct", None)
    if min_rel_long is not None and profile["rel_long"] < float(min_rel_long):
        return _reject(
            "v2_relative_long_below_min",
            f"QQQ-relative long return {profile['rel_long']:.2f}% below v2 minimum.",
        )

    max_decay = getattr(cfg, "v2_max_qqq_relative_decay_pct", None)
    if max_decay is not None and profile["rel_decay"] > float(max_decay):
        return _reject(
            "v2_relative_decay_above_max",
            f"QQQ-relative momentum decay {profile['rel_decay']:.2f}% above v2 maximum.",
        )

    strong_qqq = (
        profile["qqq_short"] >= float(getattr(cfg, "v2_strong_qqq_short_return_pct", 3.0))
        or profile["qqq_mid"] >= float(getattr(cfg, "v2_strong_qqq_mid_return_pct", 6.0))
    )
    if not strong_qqq:
        return None

    min_strong_leadership = getattr(cfg, "v2_strong_qqq_min_leadership_score", None)
    if min_strong_leadership is not None and leadership < float(min_strong_leadership):
        return _reject(
            "v2_strong_qqq_leadership_below_min",
            f"Leadership {leadership:.1f} below v2 strong-QQQ minimum.",
        )
    min_strong_rel_short = float(getattr(cfg, "v2_strong_qqq_min_relative_short_return_pct", 0.0))
    if profile["rel_short"] < min_strong_rel_short:
        return _reject(
            "v2_strong_qqq_relative_short_below_min",
            f"QQQ-relative short return {profile['rel_short']:.2f}% below strong-QQQ minimum.",
        )
    min_strong_rel_mid = float(getattr(cfg, "v2_strong_qqq_min_relative_mid_return_pct", -1.0))
    if profile["rel_mid"] < min_strong_rel_mid:
        return _reject(
            "v2_strong_qqq_relative_mid_below_min",
            f"QQQ-relative mid return {profile['rel_mid']:.2f}% below strong-QQQ minimum.",
        )
    return None


def _score_above_floor(value: float, floor: float, span: float) -> float:
    return _BASE.clamp((value - floor) / max(span, 0.01), 0.0, 1.0)


def _adjusted_intent_score_and_reason(intent, profile: dict[str, float], leadership: float, cfg) -> tuple[float, str]:
    base_score = float(intent.score)
    short_score = _score_above_floor(
        profile["rel_short"],
        float(getattr(cfg, "v2_min_qqq_relative_short_return_pct", -1.0)),
        14.0,
    )
    mid_score = _score_above_floor(
        profile["rel_mid"],
        float(getattr(cfg, "v2_min_qqq_relative_mid_return_pct", -2.0)),
        20.0,
    )
    decay_cap = getattr(cfg, "v2_max_qqq_relative_decay_pct", 8.0)
    decay_cap = 8.0 if decay_cap is None else float(decay_cap)
    acceleration_score = _BASE.clamp((decay_cap - profile["rel_decay"]) / max(decay_cap + 8.0, 0.01), 0.0, 1.0)
    leadership_floor = getattr(cfg, "v2_min_leadership_score", None)
    leadership_floor = 55.0 if leadership_floor is None else float(leadership_floor)
    leadership_score = _score_above_floor(leadership, leadership_floor, 35.0)
    base_component = _BASE.clamp(base_score / 10.0, 0.0, 1.0)
    priority_score = (
        short_score * 0.32
        + mid_score * 0.25
        + acceleration_score * 0.18
        + leadership_score * 0.15
        + base_component * 0.10
    ) * 10.0
    weight = _BASE.clamp(float(getattr(cfg, "v2_slot_priority_weight", 0.35)), 0.0, 1.0)
    adjusted = _BASE.clamp(base_score * (1.0 - weight) + priority_score * weight, 0.0, 10.0)
    reason = (
        f"{intent.reason} | v2 relS {profile['rel_short']:.1f}% relM {profile['rel_mid']:.1f}% "
        f"relL {profile['rel_long']:.1f}% qS {profile['qqq_short']:.1f}% qM {profile['qqq_mid']:.1f}% "
        f"lead {leadership:.1f} slot {adjusted:.2f}"
    )
    return adjusted, reason


def _benchmark_close_at_or_before(cfg, ts) -> float | None:
    if ts is None:
        return None
    for bar in reversed(_BASE._benchmark_bars(cfg)):
        if bar.ts <= ts:
            close = float(bar.close)
            return close if close > 0.0 else None
    return None


def _relative_dead_money_exit(pos, ts, high, low, close, total_bars, cfg, *, exit_active: bool):
    if (
        not exit_active
        or not bool(getattr(cfg, "v2_relative_dead_money_exit_enabled", True))
        or str(pos.direction).upper() != "LONG"
        or bool(getattr(pos, "trailing_activated", False))
    ):
        return None
    if total_bars < int(getattr(cfg, "v2_relative_dead_money_exit_min_bars", 130)):
        return None

    entry = float(pos.entry_price)
    if entry <= 0.0:
        return None

    high_return = (float(high) / entry - 1.0) * 100.0
    low_return = (float(low) / entry - 1.0) * 100.0
    current_return = (float(close) / entry - 1.0) * 100.0
    mfe = max(float(getattr(pos, "model_mfe_pct", high_return)), high_return)
    mae = min(float(getattr(pos, "model_mae_pct", low_return)), low_return)
    setattr(pos, "model_mfe_pct", mfe)
    setattr(pos, "model_mae_pct", mae)

    max_return = float(getattr(cfg, "v2_relative_dead_money_max_return_pct", 3.0))
    if current_return > max_return:
        return None
    mfe_cap = getattr(cfg, "v2_relative_dead_money_mfe_cap_pct", 8.0)
    if mfe_cap is not None and mfe > float(mfe_cap):
        return None

    entry_qqq = _benchmark_close_at_or_before(cfg, getattr(pos, "entry_ts", None))
    current_qqq = _benchmark_close_at_or_before(cfg, ts)
    if entry_qqq is None or current_qqq is None:
        return None

    qqq_return = (current_qqq / entry_qqq - 1.0) * 100.0
    if qqq_return < float(getattr(cfg, "v2_relative_dead_money_min_qqq_return_pct", 2.0)):
        return None
    underperformance = qqq_return - current_return
    if underperformance < float(getattr(cfg, "v2_relative_dead_money_underperformance_pct", 7.0)):
        return None

    return {
        "exit": True,
        "status": "MODEL_SELECTOR_QQQ_RELATIVE_DEAD_MONEY",
        "price": float(close),
        "reason": (
            f"QQQ-relative dead-money exit return {current_return:.2f}% vs QQQ {qqq_return:.2f}% "
            f"after {total_bars} bars; underperformance {underperformance:.2f}%, "
            f"MFE {mfe:.2f}%, MAE {mae:.2f}%."
        ),
    }
