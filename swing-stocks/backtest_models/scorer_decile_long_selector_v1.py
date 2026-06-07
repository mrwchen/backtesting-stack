"""Long-only scorer decile selector model.

Research premise:
  - The forward study showed the usable edge on the long side in
    price_momentum_score, momentum_score, and scorer_alpha over 20-60 days.
  - QQQ is only market context. This model never trades QQQ and does not rank
    stocks by QQQ-relative strength.
  - The model should behave like a quiet selector portfolio, not like a tight
    entry/exit swing system.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, IntentEvaluation, TradeIntent
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, env_list, env_optional_float, env_str, mean


MODEL_NAME = "scorer_decile_long_selector_v1"
BENCHMARK_SYMBOL = "QQQ"
BENCHMARK_SYMBOLS = (BENCHMARK_SYMBOL,)
BENCHMARK_BAR_LOOKBACK = 520


@dataclass
class IntentConfig:
    min_bars: int = 520
    enable_shorts: bool = False
    entry_weekdays: tuple[int, ...] = (0, 2)

    benchmark_symbol: str = BENCHMARK_SYMBOL
    require_benchmark_context: bool = True
    benchmark_short_lookback_bars: int = 65
    benchmark_mid_lookback_bars: int = 260
    benchmark_slow_ma_bars: int = 260
    benchmark_drawdown_bars: int = 260
    min_benchmark_market_score: float = 4.5
    min_benchmark_short_return_pct: float = -8.0
    min_benchmark_mid_return_pct: float = -12.0
    max_benchmark_drawdown_pct: float = 24.0

    allowed_world_regime_labels: tuple[str, ...] = ()
    blocked_world_regime_labels: tuple[str, ...] = ()
    require_world_regime_label: bool = False
    blocked_daily_policy_phases: tuple[str, ...] = ()
    blocked_regime_phase_pairs: tuple[str, ...] = ()

    min_market_cap_m: float = 10000.0
    min_long_composite_score: float = 50.0
    min_long_momentum_score: float = 72.0
    min_long_price_momentum_score: float = 78.0
    max_long_price_momentum_score: Optional[float] = None
    min_long_leadership_score: float = 55.0
    min_long_selector_alpha: float = 0.72
    min_long_intent_score: float = 6.8

    long_lookback_bars: int = 390
    mid_lookback_bars: int = 130
    short_lookback_bars: int = 65
    fast_ma_bars: int = 65
    slow_ma_bars: int = 260
    drawdown_lookback_bars: int = 390
    atr_bars: int = 65

    min_long_return_pct: float = -3.0
    min_mid_return_pct: float = -4.0
    min_short_return_pct: float = -6.0
    max_long_return_pct: float = 140.0
    max_mid_return_pct: Optional[float] = None
    max_drawdown_pct: float = 22.0
    max_below_slow_ma_pct: float = 5.0
    long_min_rsi: float = 35.0
    long_max_rsi: float = 84.0
    max_atr_pct: float = 7.5

    model_exit_enabled: bool = True
    regime_exit_enabled: bool = False
    regime_exit_min_bars: int = 26
    regime_exit_max_return_pct: float = 4.0
    hard_loss_exit_min_bars: int = 130
    hard_loss_return_pct: float = -11.0
    hard_loss_mfe_cap_pct: float = 3.0
    dead_money_exit_min_bars: int = 195
    dead_money_max_return_pct: float = 1.0
    dead_money_mfe_cap_pct: float = 4.0
    mfe_fade_exit_min_bars: int = 260
    mfe_fade_min_mfe_pct: float = 14.0
    mfe_fade_giveback_pct: float = 11.0

    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None


def _parse_weekdays(default: tuple[int, ...]) -> tuple[int, ...]:
    values = env_list("ENTRY_WEEKDAYS", [str(value) for value in default])
    weekdays: set[int] = set()
    for value in values:
        day = int(value)
        if day < 0 or day > 6:
            raise ValueError("ENTRY_WEEKDAYS values must be integers from 0 Monday through 6 Sunday")
        weekdays.add(day)
    return tuple(sorted(weekdays))


def _parse_label_set(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(label.strip().upper() for label in env_list(name, default) if label.strip())


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        enable_shorts=env_bool("ENABLE_SHORTS", d.enable_shorts),
        entry_weekdays=_parse_weekdays(d.entry_weekdays),
        benchmark_symbol=env_str("BENCHMARK_SYMBOL", d.benchmark_symbol).upper(),
        require_benchmark_context=env_bool("REQUIRE_BENCHMARK_CONTEXT", d.require_benchmark_context),
        benchmark_short_lookback_bars=env_int("BENCHMARK_SHORT_LOOKBACK_BARS", d.benchmark_short_lookback_bars),
        benchmark_mid_lookback_bars=env_int("BENCHMARK_MID_LOOKBACK_BARS", d.benchmark_mid_lookback_bars),
        benchmark_slow_ma_bars=env_int("BENCHMARK_SLOW_MA_BARS", d.benchmark_slow_ma_bars),
        benchmark_drawdown_bars=env_int("BENCHMARK_DRAWDOWN_BARS", d.benchmark_drawdown_bars),
        min_benchmark_market_score=env_float("MIN_BENCHMARK_MARKET_SCORE", d.min_benchmark_market_score),
        min_benchmark_short_return_pct=env_float("MIN_BENCHMARK_SHORT_RETURN_PCT", d.min_benchmark_short_return_pct),
        min_benchmark_mid_return_pct=env_float("MIN_BENCHMARK_MID_RETURN_PCT", d.min_benchmark_mid_return_pct),
        max_benchmark_drawdown_pct=env_float("MAX_BENCHMARK_DRAWDOWN_PCT", d.max_benchmark_drawdown_pct),
        allowed_world_regime_labels=_parse_label_set("ALLOWED_WORLD_REGIME_LABELS", d.allowed_world_regime_labels),
        blocked_world_regime_labels=_parse_label_set("BLOCKED_WORLD_REGIME_LABELS", d.blocked_world_regime_labels),
        require_world_regime_label=env_bool("REQUIRE_WORLD_REGIME_LABEL", d.require_world_regime_label),
        blocked_daily_policy_phases=_parse_label_set("BLOCKED_DAILY_POLICY_PHASES", d.blocked_daily_policy_phases),
        blocked_regime_phase_pairs=_parse_label_set("BLOCKED_REGIME_PHASE_PAIRS", d.blocked_regime_phase_pairs),
        min_market_cap_m=env_float("MIN_MARKET_CAP_USD_M", d.min_market_cap_m),
        min_long_composite_score=env_float("MIN_LONG_COMPOSITE_SCORE", d.min_long_composite_score),
        min_long_momentum_score=env_float("MIN_LONG_MOMENTUM_SCORE", d.min_long_momentum_score),
        min_long_price_momentum_score=env_float("MIN_LONG_PRICE_MOMENTUM_SCORE", d.min_long_price_momentum_score),
        max_long_price_momentum_score=env_optional_float(
            "MAX_LONG_PRICE_MOMENTUM_SCORE",
            d.max_long_price_momentum_score,
        ),
        min_long_leadership_score=env_float("MIN_LONG_LEADERSHIP_SCORE", d.min_long_leadership_score),
        min_long_selector_alpha=env_float("MIN_LONG_SELECTOR_ALPHA", d.min_long_selector_alpha),
        min_long_intent_score=env_float("MIN_LONG_INTENT_SCORE", d.min_long_intent_score),
        long_lookback_bars=env_int("LONG_LOOKBACK_BARS", d.long_lookback_bars),
        mid_lookback_bars=env_int("MID_LOOKBACK_BARS", d.mid_lookback_bars),
        short_lookback_bars=env_int("SHORT_LOOKBACK_BARS", d.short_lookback_bars),
        fast_ma_bars=env_int("FAST_MA_BARS", d.fast_ma_bars),
        slow_ma_bars=env_int("SLOW_MA_BARS", d.slow_ma_bars),
        drawdown_lookback_bars=env_int("DRAWDOWN_LOOKBACK_BARS", d.drawdown_lookback_bars),
        atr_bars=env_int("ATR_BARS", d.atr_bars),
        min_long_return_pct=env_float("MIN_LONG_RETURN_PCT", d.min_long_return_pct),
        min_mid_return_pct=env_float("MIN_MID_RETURN_PCT", d.min_mid_return_pct),
        min_short_return_pct=env_float("MIN_SHORT_RETURN_PCT", d.min_short_return_pct),
        max_long_return_pct=env_float("MAX_LONG_RETURN_PCT", d.max_long_return_pct),
        max_mid_return_pct=env_optional_float("MAX_MID_RETURN_PCT", d.max_mid_return_pct),
        max_drawdown_pct=env_float("MAX_DRAWDOWN_PCT", d.max_drawdown_pct),
        max_below_slow_ma_pct=env_float("MAX_BELOW_SLOW_MA_PCT", d.max_below_slow_ma_pct),
        long_min_rsi=env_float("LONG_MIN_RSI", d.long_min_rsi),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        max_atr_pct=env_float("MAX_ATR_PCT", d.max_atr_pct),
        model_exit_enabled=env_bool("MODEL_EXIT_ENABLED", d.model_exit_enabled),
        regime_exit_enabled=env_bool("REGIME_EXIT_ENABLED", d.regime_exit_enabled),
        regime_exit_min_bars=env_int("REGIME_EXIT_MIN_BARS", d.regime_exit_min_bars),
        regime_exit_max_return_pct=env_float("REGIME_EXIT_MAX_RETURN_PCT", d.regime_exit_max_return_pct),
        hard_loss_exit_min_bars=env_int("HARD_LOSS_EXIT_MIN_BARS", d.hard_loss_exit_min_bars),
        hard_loss_return_pct=env_float("HARD_LOSS_RETURN_PCT", d.hard_loss_return_pct),
        hard_loss_mfe_cap_pct=env_float("HARD_LOSS_MFE_CAP_PCT", d.hard_loss_mfe_cap_pct),
        dead_money_exit_min_bars=env_int("DEAD_MONEY_EXIT_MIN_BARS", d.dead_money_exit_min_bars),
        dead_money_max_return_pct=env_float("DEAD_MONEY_MAX_RETURN_PCT", d.dead_money_max_return_pct),
        dead_money_mfe_cap_pct=env_float("DEAD_MONEY_MFE_CAP_PCT", d.dead_money_mfe_cap_pct),
        mfe_fade_exit_min_bars=env_int("MFE_FADE_EXIT_MIN_BARS", d.mfe_fade_exit_min_bars),
        mfe_fade_min_mfe_pct=env_float("MFE_FADE_MIN_MFE_PCT", d.mfe_fade_min_mfe_pct),
        mfe_fade_giveback_pct=env_float("MFE_FADE_GIVEBACK_PCT", d.mfe_fade_giveback_pct),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", d.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", d.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", d.fundamental_abs_weight),
        long_min_absolute_score=None,
        short_max_absolute_score=None,
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.long_lookback_bars + 1,
        cfg.mid_lookback_bars + 1,
        cfg.short_lookback_bars + 1,
        cfg.fast_ma_bars,
        cfg.slow_ma_bars,
        cfg.drawdown_lookback_bars,
        cfg.atr_bars + 1,
        cfg.benchmark_mid_lookback_bars + 1,
        cfg.benchmark_slow_ma_bars,
        cfg.benchmark_drawdown_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": f"grid model={MODEL_NAME}", "summary": {}}


def set_market_context(cfg: IntentConfig, as_of_ts: datetime, bars_by_symbol: dict[str, list[Bar]]) -> None:
    setattr(cfg, "market_context_as_of_ts", as_of_ts)
    setattr(cfg, "market_context_bars_by_symbol", bars_by_symbol)


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


def _explicit_regime_block(cfg: IntentConfig) -> tuple[bool, str, str]:
    label = _world_regime_label(cfg)
    phase = _daily_policy_phase(cfg)
    if label and label in set(cfg.blocked_world_regime_labels):
        return True, "world_regime_blocked", f"World regime label {label} is blocked."
    if phase and phase in set(cfg.blocked_daily_policy_phases):
        return True, "daily_policy_phase_blocked", f"Daily policy phase {phase} is blocked."
    if label and phase:
        pair = f"{label}:{phase}"
        if pair in set(cfg.blocked_regime_phase_pairs):
            return True, "regime_phase_pair_blocked", f"Regime/phase pair {pair} is blocked."
    return False, "", ""


def _regime_is_tradeable(cfg: IntentConfig) -> tuple[bool, str, str]:
    label = _world_regime_label(cfg)
    if not label and cfg.require_world_regime_label:
        return False, "world_regime_missing", "World regime label is missing."
    blocked, code, text = _explicit_regime_block(cfg)
    if blocked:
        return False, code, text
    if cfg.allowed_world_regime_labels and label and label not in set(cfg.allowed_world_regime_labels):
        return False, "world_regime_not_allowed", f"World regime label {label} is not allowed."
    return True, "", ""


def _benchmark_bars(cfg: IntentConfig) -> list[Bar]:
    by_symbol = getattr(cfg, "market_context_bars_by_symbol", {}) or {}
    return list(by_symbol.get(cfg.benchmark_symbol.strip().upper(), []))


def _market_state(cfg: IntentConfig) -> dict[str, float | bool]:
    bars = _benchmark_bars(cfg)
    required = max(cfg.benchmark_mid_lookback_bars + 1, cfg.benchmark_slow_ma_bars, cfg.benchmark_drawdown_bars)
    if len(bars) < required:
        return {"available": False, "score": 0.0}

    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    close = closes[-1]
    ret_short = _ret_pct(closes, cfg.benchmark_short_lookback_bars)
    ret_mid = _ret_pct(closes, cfg.benchmark_mid_lookback_bars)
    slow_ma = _sma(closes, cfg.benchmark_slow_ma_bars)
    high = max(highs[-cfg.benchmark_drawdown_bars:])
    drawdown = (high - close) / high * 100.0 if high > 0.0 else 999.0

    score = 0.0
    score += clamp((ret_short - cfg.min_benchmark_short_return_pct) / 16.0, 0.0, 1.0) * 2.8
    score += clamp((ret_mid - cfg.min_benchmark_mid_return_pct) / 28.0, 0.0, 1.0) * 2.8
    score += clamp((cfg.max_benchmark_drawdown_pct - drawdown) / cfg.max_benchmark_drawdown_pct, 0.0, 1.0) * 2.4
    score += (1.0 if slow_ma > 0.0 and close >= slow_ma else 0.0) * 2.0
    return {
        "available": True,
        "score": clamp(score, 0.0, 10.0),
        "ret_short": ret_short,
        "ret_mid": ret_mid,
        "drawdown": drawdown,
        "above_slow_ma": slow_ma > 0.0 and close >= slow_ma,
    }


def _market_is_tradeable(cfg: IntentConfig) -> tuple[bool, str, str, dict[str, float | bool]]:
    state = _market_state(cfg)
    if not state.get("available"):
        if cfg.require_benchmark_context:
            return False, "benchmark_context_unavailable", "Benchmark bars are unavailable.", state
        return True, "", "", state
    if float(state["score"]) < cfg.min_benchmark_market_score:
        return False, "benchmark_market_score_below_min", f"Benchmark market score {float(state['score']):.2f} below minimum.", state
    if float(state["ret_short"]) < cfg.min_benchmark_short_return_pct:
        return False, "benchmark_short_return_too_weak", f"Benchmark short return {float(state['ret_short']):.2f}% below minimum.", state
    if float(state["ret_mid"]) < cfg.min_benchmark_mid_return_pct:
        return False, "benchmark_mid_return_too_weak", f"Benchmark mid return {float(state['ret_mid']):.2f}% below minimum.", state
    if float(state["drawdown"]) > cfg.max_benchmark_drawdown_pct:
        return False, "benchmark_drawdown_too_deep", f"Benchmark drawdown {float(state['drawdown']):.2f}% above maximum.", state
    return True, "", "", state


def _selector_alpha(f: FundamentalRow) -> float:
    composite = _score01(f.composite_score, 50.0)
    momentum = _score01(f.momentum_score, f.composite_score)
    price_momentum = _score01(f.price_momentum_score, f.momentum_score or f.composite_score)
    leadership = _score01(f.leadership_score, f.composite_score)
    fundamental_momentum = _score01(f.fundamental_momentum_score, f.composite_score)
    return clamp(
        price_momentum * 0.42
        + momentum * 0.28
        + leadership * 0.14
        + fundamental_momentum * 0.10
        + composite * 0.06,
        0.0,
        1.0,
    )


def _price_alpha(bars: list[Bar], cfg: IntentConfig) -> tuple[float, dict[str, float]]:
    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    close = closes[-1]
    long_return = _ret_pct(closes, cfg.long_lookback_bars)
    mid_return = _ret_pct(closes, cfg.mid_lookback_bars)
    short_return = _ret_pct(closes, cfg.short_lookback_bars)
    fast_ma = _sma(closes, cfg.fast_ma_bars)
    slow_ma = _sma(closes, cfg.slow_ma_bars)
    lookback_high = max(highs[-cfg.drawdown_lookback_bars:])
    drawdown = (lookback_high - close) / lookback_high * 100.0 if lookback_high > 0.0 else 999.0
    rsi = compute_rsi(closes[-50:])
    atr = _atr_pct(bars, cfg.atr_bars)

    trend_score = clamp((long_return - cfg.min_long_return_pct) / 45.0, 0.0, 1.0)
    mid_score = clamp((mid_return - cfg.min_mid_return_pct) / 26.0, 0.0, 1.0)
    short_score = clamp((short_return - cfg.min_short_return_pct) / 16.0, 0.0, 1.0)
    ma_score = 1.0 if close > fast_ma > slow_ma else (0.65 if slow_ma > 0.0 and close > slow_ma else 0.0)
    drawdown_score = clamp(1.0 - drawdown / max(cfg.max_drawdown_pct, 0.01), 0.0, 1.0)
    atr_score = clamp(1.0 - atr / max(cfg.max_atr_pct, 0.01), 0.0, 1.0)
    rsi_score = 1.0 if rsi <= 68.0 else clamp((cfg.long_max_rsi - rsi) / max(cfg.long_max_rsi - 68.0, 1.0), 0.0, 1.0)

    alpha = clamp(
        trend_score * 0.26
        + mid_score * 0.24
        + short_score * 0.14
        + ma_score * 0.14
        + drawdown_score * 0.10
        + atr_score * 0.07
        + rsi_score * 0.05,
        0.0,
        1.0,
    )
    return alpha, {
        "long_return": long_return,
        "mid_return": mid_return,
        "short_return": short_return,
        "fast_ma": fast_ma,
        "slow_ma": slow_ma,
        "drawdown": drawdown,
        "rsi": rsi,
        "atr": atr,
    }


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
    symbol = str(fundamental.symbol or "").strip().upper()
    if symbol == cfg.benchmark_symbol.strip().upper():
        return IntentEvaluation(None, "rejected", "benchmark_symbol_context_only", "QQQ is market context only.")
    if cfg.entry_weekdays and now.weekday() not in cfg.entry_weekdays:
        return IntentEvaluation(None, "rejected", "not_rebalance_weekday", f"Weekday {now.weekday()} is not an entry weekday.")

    regime_ok, regime_code, regime_text = _regime_is_tradeable(cfg)
    if not regime_ok:
        return IntentEvaluation(None, "rejected", regime_code, regime_text)

    market_ok, market_code, market_text, market = _market_is_tradeable(cfg)
    if not market_ok:
        return IntentEvaluation(None, "rejected", market_code, market_text)

    market_cap = float(fundamental.market_cap_m or 0.0)
    if market_cap < cfg.min_market_cap_m:
        return IntentEvaluation(None, "rejected", "market_cap_below_model_min", f"Market cap {market_cap:.0f}m below model minimum.")

    composite = float(fundamental.composite_score)
    momentum = float(fundamental.momentum_score if fundamental.momentum_score is not None else composite)
    price_momentum = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else momentum)
    leadership = float(fundamental.leadership_score if fundamental.leadership_score is not None else composite)
    selector_alpha = _selector_alpha(fundamental)

    if composite < cfg.min_long_composite_score:
        return IntentEvaluation(None, "rejected", "composite_below_floor", f"Composite {composite:.1f} below floor.")
    if momentum < cfg.min_long_momentum_score:
        return IntentEvaluation(None, "rejected", "momentum_below_selector_min", f"Momentum {momentum:.1f} below selector minimum.")
    if price_momentum < cfg.min_long_price_momentum_score:
        return IntentEvaluation(None, "rejected", "price_momentum_below_selector_min", f"Price momentum {price_momentum:.1f} below selector minimum.")
    if cfg.max_long_price_momentum_score is not None and price_momentum > cfg.max_long_price_momentum_score:
        return IntentEvaluation(
            None,
            "rejected",
            "price_momentum_above_selector_max",
            f"Price momentum {price_momentum:.1f} above selector maximum.",
        )
    if leadership < cfg.min_long_leadership_score:
        return IntentEvaluation(None, "rejected", "leadership_below_floor", f"Leadership {leadership:.1f} below floor.")
    if selector_alpha < cfg.min_long_selector_alpha:
        return IntentEvaluation(None, "rejected", "selector_alpha_below_min", f"Selector alpha {selector_alpha:.3f} below minimum.")

    price_alpha, m = _price_alpha(bars, cfg)
    close = bars[-1].close
    if m["long_return"] < cfg.min_long_return_pct:
        return IntentEvaluation(None, "rejected", "long_return_below_min", f"Long return {m['long_return']:.2f}% below minimum.")
    if m["mid_return"] < cfg.min_mid_return_pct:
        return IntentEvaluation(None, "rejected", "mid_return_below_min", f"Mid return {m['mid_return']:.2f}% below minimum.")
    if m["short_return"] < cfg.min_short_return_pct:
        return IntentEvaluation(None, "rejected", "short_return_below_min", f"Short return {m['short_return']:.2f}% below minimum.")
    if m["long_return"] > cfg.max_long_return_pct:
        return IntentEvaluation(None, "rejected", "long_return_too_extended", f"Long return {m['long_return']:.2f}% above maximum.")
    if cfg.max_mid_return_pct is not None and m["mid_return"] > cfg.max_mid_return_pct:
        return IntentEvaluation(None, "rejected", "mid_return_too_extended", f"Mid return {m['mid_return']:.2f}% above maximum.")
    if m["drawdown"] > cfg.max_drawdown_pct:
        return IntentEvaluation(None, "rejected", "drawdown_above_max", f"Drawdown {m['drawdown']:.2f}% above maximum.")
    if m["slow_ma"] > 0.0 and close < m["slow_ma"] * (1.0 - cfg.max_below_slow_ma_pct / 100.0):
        return IntentEvaluation(None, "rejected", "too_far_below_slow_ma", "Close is too far below slow MA.")
    if m["rsi"] < cfg.long_min_rsi or m["rsi"] > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_range", f"RSI {m['rsi']:.2f} outside range.")
    if m["atr"] > cfg.max_atr_pct:
        return IntentEvaluation(None, "rejected", "atr_above_max", f"ATR {m['atr']:.2f}% above maximum.")

    market_score = clamp(float(market.get("score", 0.0)) / 10.0, 0.0, 1.0)
    combined = (selector_alpha * 0.58 + price_alpha * 0.32 + market_score * 0.10) * 10.0
    if combined < cfg.min_long_intent_score:
        return IntentEvaluation(None, "rejected", "intent_score_below_min", f"Intent score {combined:.2f} below minimum.")

    reason = (
        f"ScorerSelector alpha {selector_alpha:.3f} price {price_alpha:.3f} market {market_score:.3f} | "
        f"scores PM {price_momentum:.1f} M {momentum:.1f} L {leadership:.1f} C {composite:.1f} | "
        f"ret{cfg.long_lookback_bars} {m['long_return']:.1f}% ret{cfg.mid_lookback_bars} {m['mid_return']:.1f}% "
        f"ret{cfg.short_lookback_bars} {m['short_return']:.1f}% DD {m['drawdown']:.1f}% "
        f"RSI {m['rsi']:.0f} ATR {m['atr']:.1f}% cap {market_cap:.0f}m | "
        f"regime {_world_regime_label(cfg) or '-'}"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason),
        "intent",
        "scorer_decile_long_selector_passed",
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
    return IntentEvaluation(None, "rejected", "shorts_disabled_by_model", "This selector test is long-only.")


def evaluate_position_exit(
    pos,
    ts: datetime,
    open_: float,
    high: float,
    low: float,
    close: float,
    total_bars: int,
    cfg: IntentConfig,
    *,
    exit_active: bool,
):
    if not cfg.model_exit_enabled or not exit_active or str(pos.direction).upper() != "LONG":
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

    if cfg.regime_exit_enabled and total_bars >= cfg.regime_exit_min_bars:
        blocked, code, text = _explicit_regime_block(cfg)
        if blocked and current_return <= cfg.regime_exit_max_return_pct:
            return {
                "exit": True,
                "status": "MODEL_SELECTOR_REGIME_EXIT",
                "price": float(close),
                "reason": (
                    f"Selector regime exit {code}: {text} Current return {current_return:.2f}% "
                    f"with MFE {mfe:.2f}% and MAE {mae:.2f}% after {total_bars} bars."
                ),
            }

    if total_bars >= cfg.hard_loss_exit_min_bars:
        if current_return <= cfg.hard_loss_return_pct and mfe <= cfg.hard_loss_mfe_cap_pct:
            return {
                "exit": True,
                "status": "MODEL_SELECTOR_HARD_LOSS",
                "price": float(close),
                "reason": (
                    f"Selector hard-loss exit return {current_return:.2f}% with MFE {mfe:.2f}% "
                    f"after {total_bars} bars; MAE {mae:.2f}%."
                ),
            }

    if total_bars >= cfg.dead_money_exit_min_bars:
        if current_return <= cfg.dead_money_max_return_pct and mfe <= cfg.dead_money_mfe_cap_pct:
            return {
                "exit": True,
                "status": "MODEL_SELECTOR_DEAD_MONEY",
                "price": float(close),
                "reason": (
                    f"Selector dead-money exit return {current_return:.2f}% with MFE {mfe:.2f}% "
                    f"after {total_bars} bars; MAE {mae:.2f}%."
                ),
            }

    if total_bars >= cfg.mfe_fade_exit_min_bars:
        if mfe >= cfg.mfe_fade_min_mfe_pct and mfe - current_return >= cfg.mfe_fade_giveback_pct:
            return {
                "exit": True,
                "status": "MODEL_SELECTOR_MFE_FADE",
                "price": float(close),
                "reason": (
                    f"Selector MFE fade from {mfe:.2f}% to {current_return:.2f}% "
                    f"after {total_bars} bars; MAE {mae:.2f}%."
                ),
            }

    return None
