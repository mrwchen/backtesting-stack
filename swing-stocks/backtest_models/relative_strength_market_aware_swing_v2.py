"""Market-aware relative-strength swing model.

Model idea:
  - LONG only.
  - QQQ is only the market/exposure benchmark; the model never buys QQQ.
  - Fundamental score is a quality floor, not the ranking edge.
  - Stock entries must pass one concrete alpha setup instead of accumulating
    many weak partial scores.
  - A model exit closes trades that fail to produce early MFE.
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


MODEL_NAME = "relative_strength_market_aware_swing_v2"
BENCHMARK_SYMBOL = "QQQ"
BENCHMARK_SYMBOLS = (BENCHMARK_SYMBOL,)
BENCHMARK_BAR_LOOKBACK = 520
DIRECT_CANDIDATE_SYMBOLS = ()
DIRECT_CANDIDATE_MODE = "append"
DIRECT_CANDIDATE_REQUIRE_BROKER_ELIGIBILITY = False


@dataclass
class IntentConfig:
    min_bars: int = 320
    enable_shorts: bool = False

    benchmark_symbol: str = BENCHMARK_SYMBOL
    require_benchmark_context: bool = True
    benchmark_short_lookback_bars: int = 13
    benchmark_mid_lookback_bars: int = 65
    benchmark_long_lookback_bars: int = 260
    benchmark_fast_ma_bars: int = 65
    benchmark_slow_ma_bars: int = 260
    benchmark_drawdown_bars: int = 260
    min_benchmark_market_score: float = 6.5
    min_benchmark_short_return_pct: float = -3.0
    min_benchmark_mid_return_pct: float = -6.0
    max_benchmark_drawdown_pct: float = 14.0

    allowed_world_regime_labels: tuple[str, ...] = ("CONSTRUCTIVE", "NEUTRAL")
    blocked_world_regime_labels: tuple[str, ...] = ("DEFENSIVE",)
    require_world_regime_label: bool = True
    blocked_daily_policy_phases: tuple[str, ...] = ("STRESS_HIGH",)

    min_long_composite_score: float = 50.0
    min_long_price_momentum_score: float = 55.0
    min_long_intent_score: float = 6.25
    use_mispricing_score: bool = False
    mispricing_weight: float = 0.0
    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None

    price_lookback_bars: int = 260
    trend_lookback_bars: int = 260
    swing_lookback_bars: int = 65
    confirmation_bars: int = 13
    rebound_bars: int = 13
    fast_ma_bars: int = 65
    slow_ma_bars: int = 260
    atr_bars: int = 65

    min_trend_return_pct: float = 0.0
    min_swing_return_pct: float = 0.0
    min_confirmation_pct: float = 1.0
    min_relative_trend_pct: float = -2.0
    max_relative_trend_pct: float = 55.0
    min_relative_swing_pct: float = 0.0
    min_relative_confirmation_pct: float = 0.5
    min_rebound_from_recent_low_pct: float = 1.0
    long_min_pullback_pct: float = 0.0
    long_max_pullback_pct: float = 10.0
    long_ideal_pullback_pct: float = 4.0
    long_min_rsi: float = 52.0
    long_max_rsi: float = 71.0
    max_atr_pct: float = 3.0
    max_below_slow_ma_pct: float = 0.5

    min_breakout_confirmation_pct: float = 5.0
    min_breakout_relative_swing_pct: float = 3.0
    min_breakout_relative_confirmation_pct: float = 1.5
    max_breakout_pullback_pct: float = 4.0
    min_reclaim_pullback_pct: float = 5.0
    min_reclaim_rebound_pct: float = 3.0
    min_reclaim_confirmation_pct: float = 2.0
    min_reclaim_relative_swing_pct: float = 0.0
    max_chase_rsi: float = 68.0

    failure_fast_enabled: bool = True
    failure_fast_min_bars: int = 28
    failure_fast_min_mfe_pct: float = 1.75
    failure_fast_max_return_pct: float = 0.0
    relative_failure_min_bars: int = 45
    relative_failure_loss_pct: float = 2.5
    relative_failure_mfe_cap_pct: float = 2.75


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        enable_shorts=env_bool("ENABLE_SHORTS", d.enable_shorts),
        benchmark_symbol=env_str("BENCHMARK_SYMBOL", d.benchmark_symbol).upper(),
        require_benchmark_context=env_bool("REQUIRE_BENCHMARK_CONTEXT", d.require_benchmark_context),
        benchmark_short_lookback_bars=env_int("BENCHMARK_SHORT_LOOKBACK_BARS", d.benchmark_short_lookback_bars),
        benchmark_mid_lookback_bars=env_int("BENCHMARK_MID_LOOKBACK_BARS", d.benchmark_mid_lookback_bars),
        benchmark_long_lookback_bars=env_int("BENCHMARK_LONG_LOOKBACK_BARS", d.benchmark_long_lookback_bars),
        benchmark_fast_ma_bars=env_int("BENCHMARK_FAST_MA_BARS", d.benchmark_fast_ma_bars),
        benchmark_slow_ma_bars=env_int("BENCHMARK_SLOW_MA_BARS", d.benchmark_slow_ma_bars),
        benchmark_drawdown_bars=env_int("BENCHMARK_DRAWDOWN_BARS", d.benchmark_drawdown_bars),
        min_benchmark_market_score=env_float("MIN_BENCHMARK_MARKET_SCORE", d.min_benchmark_market_score),
        min_benchmark_short_return_pct=env_float("MIN_BENCHMARK_SHORT_RETURN_PCT", d.min_benchmark_short_return_pct),
        min_benchmark_mid_return_pct=env_float("MIN_BENCHMARK_MID_RETURN_PCT", d.min_benchmark_mid_return_pct),
        max_benchmark_drawdown_pct=env_float("MAX_BENCHMARK_DRAWDOWN_PCT", d.max_benchmark_drawdown_pct),
        allowed_world_regime_labels=tuple(
            label.strip().upper()
            for label in env_list("ALLOWED_WORLD_REGIME_LABELS", d.allowed_world_regime_labels)
            if label.strip()
        ),
        blocked_world_regime_labels=tuple(
            label.strip().upper()
            for label in env_list("BLOCKED_WORLD_REGIME_LABELS", d.blocked_world_regime_labels)
            if label.strip()
        ),
        require_world_regime_label=env_bool("REQUIRE_WORLD_REGIME_LABEL", d.require_world_regime_label),
        blocked_daily_policy_phases=tuple(
            phase.strip().upper()
            for phase in env_list("BLOCKED_DAILY_POLICY_PHASES", d.blocked_daily_policy_phases)
            if phase.strip()
        ),
        min_long_composite_score=env_float("MIN_LONG_COMPOSITE_SCORE", d.min_long_composite_score),
        min_long_price_momentum_score=env_float("MIN_LONG_PRICE_MOMENTUM_SCORE", d.min_long_price_momentum_score),
        min_long_intent_score=env_float("MIN_LONG_INTENT_SCORE", d.min_long_intent_score),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", d.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", d.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", d.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", d.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", d.short_max_absolute_score),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        trend_lookback_bars=env_int("TREND_LOOKBACK_BARS", d.trend_lookback_bars),
        swing_lookback_bars=env_int("SWING_LOOKBACK_BARS", d.swing_lookback_bars),
        confirmation_bars=env_int("CONFIRMATION_BARS", d.confirmation_bars),
        rebound_bars=env_int("REBOUND_BARS", d.rebound_bars),
        fast_ma_bars=env_int("FAST_MA_BARS", d.fast_ma_bars),
        slow_ma_bars=env_int("SLOW_MA_BARS", d.slow_ma_bars),
        atr_bars=env_int("ATR_BARS", d.atr_bars),
        min_trend_return_pct=env_float("MIN_TREND_RETURN_PCT", d.min_trend_return_pct),
        min_swing_return_pct=env_float("MIN_SWING_RETURN_PCT", d.min_swing_return_pct),
        min_confirmation_pct=env_float("MIN_CONFIRMATION_PCT", d.min_confirmation_pct),
        min_relative_trend_pct=env_float("MIN_RELATIVE_TREND_PCT", d.min_relative_trend_pct),
        max_relative_trend_pct=env_float("MAX_RELATIVE_TREND_PCT", d.max_relative_trend_pct),
        min_relative_swing_pct=env_float("MIN_RELATIVE_SWING_PCT", d.min_relative_swing_pct),
        min_relative_confirmation_pct=env_float(
            "MIN_RELATIVE_CONFIRMATION_PCT",
            d.min_relative_confirmation_pct,
        ),
        min_rebound_from_recent_low_pct=env_float(
            "MIN_REBOUND_FROM_RECENT_LOW_PCT",
            d.min_rebound_from_recent_low_pct,
        ),
        long_min_pullback_pct=env_float("LONG_MIN_PULLBACK_PCT", d.long_min_pullback_pct),
        long_max_pullback_pct=env_float("LONG_MAX_PULLBACK_PCT", d.long_max_pullback_pct),
        long_ideal_pullback_pct=env_float("LONG_IDEAL_PULLBACK_PCT", d.long_ideal_pullback_pct),
        long_min_rsi=env_float("LONG_MIN_RSI", d.long_min_rsi),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        max_atr_pct=env_float("MAX_ATR_PCT", d.max_atr_pct),
        max_below_slow_ma_pct=env_float("MAX_BELOW_SLOW_MA_PCT", d.max_below_slow_ma_pct),
        min_breakout_confirmation_pct=env_float(
            "MIN_BREAKOUT_CONFIRMATION_PCT",
            d.min_breakout_confirmation_pct,
        ),
        min_breakout_relative_swing_pct=env_float(
            "MIN_BREAKOUT_RELATIVE_SWING_PCT",
            d.min_breakout_relative_swing_pct,
        ),
        min_breakout_relative_confirmation_pct=env_float(
            "MIN_BREAKOUT_RELATIVE_CONFIRMATION_PCT",
            d.min_breakout_relative_confirmation_pct,
        ),
        max_breakout_pullback_pct=env_float("MAX_BREAKOUT_PULLBACK_PCT", d.max_breakout_pullback_pct),
        min_reclaim_pullback_pct=env_float("MIN_RECLAIM_PULLBACK_PCT", d.min_reclaim_pullback_pct),
        min_reclaim_rebound_pct=env_float("MIN_RECLAIM_REBOUND_PCT", d.min_reclaim_rebound_pct),
        min_reclaim_confirmation_pct=env_float("MIN_RECLAIM_CONFIRMATION_PCT", d.min_reclaim_confirmation_pct),
        min_reclaim_relative_swing_pct=env_float(
            "MIN_RECLAIM_RELATIVE_SWING_PCT",
            d.min_reclaim_relative_swing_pct,
        ),
        max_chase_rsi=env_float("MAX_CHASE_RSI", d.max_chase_rsi),
        failure_fast_enabled=env_bool("FAILURE_FAST_ENABLED", d.failure_fast_enabled),
        failure_fast_min_bars=env_int("FAILURE_FAST_MIN_BARS", d.failure_fast_min_bars),
        failure_fast_min_mfe_pct=env_float("FAILURE_FAST_MIN_MFE_PCT", d.failure_fast_min_mfe_pct),
        failure_fast_max_return_pct=env_float("FAILURE_FAST_MAX_RETURN_PCT", d.failure_fast_max_return_pct),
        relative_failure_min_bars=env_int("RELATIVE_FAILURE_MIN_BARS", d.relative_failure_min_bars),
        relative_failure_loss_pct=env_float("RELATIVE_FAILURE_LOSS_PCT", d.relative_failure_loss_pct),
        relative_failure_mfe_cap_pct=env_float("RELATIVE_FAILURE_MFE_CAP_PCT", d.relative_failure_mfe_cap_pct),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.trend_lookback_bars + 1,
        cfg.swing_lookback_bars + 1,
        cfg.confirmation_bars + 1,
        cfg.rebound_bars,
        cfg.fast_ma_bars,
        cfg.slow_ma_bars,
        cfg.atr_bars + 1,
        cfg.benchmark_long_lookback_bars + 1,
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


def _benchmark_bars(cfg: IntentConfig) -> list[Bar]:
    by_symbol = getattr(cfg, "market_context_bars_by_symbol", {}) or {}
    return list(by_symbol.get(cfg.benchmark_symbol.strip().upper(), []))


def _market_state(cfg: IntentConfig) -> dict[str, float | bool]:
    bars = _benchmark_bars(cfg)
    if len(bars) < max(cfg.benchmark_mid_lookback_bars + 1, cfg.benchmark_fast_ma_bars):
        return {"available": False, "score": 0.0}

    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    close = closes[-1]
    ret_short = _ret_pct(closes, cfg.benchmark_short_lookback_bars)
    ret_mid = _ret_pct(closes, cfg.benchmark_mid_lookback_bars)
    ret_long = _ret_pct(closes, cfg.benchmark_long_lookback_bars)
    high = max(highs[-min(len(highs), cfg.benchmark_drawdown_bars):])
    drawdown = (high - close) / high * 100.0 if high > 0.0 else 999.0
    fast_ma = _sma(closes, cfg.benchmark_fast_ma_bars)
    slow_ma = _sma(closes, cfg.benchmark_slow_ma_bars)
    rsi = compute_rsi(closes[-50:])

    score = 0.0
    score += clamp((ret_short - cfg.min_benchmark_short_return_pct) / 8.0, 0.0, 1.0) * 2.0
    score += clamp((ret_mid - cfg.min_benchmark_mid_return_pct) / 16.0, 0.0, 1.0) * 2.2
    score += clamp((ret_long + 8.0) / 24.0, 0.0, 1.0) * 2.0
    score += (1.0 if fast_ma > 0.0 and close > fast_ma else 0.0) * 1.2
    score += (1.0 if slow_ma > 0.0 and close > slow_ma else 0.0) * 1.2
    score += clamp((cfg.max_benchmark_drawdown_pct - drawdown) / cfg.max_benchmark_drawdown_pct, 0.0, 1.0) * 1.4

    return {
        "available": True,
        "score": clamp(score, 0.0, 10.0),
        "ret_short": ret_short,
        "ret_mid": ret_mid,
        "ret_long": ret_long,
        "drawdown": drawdown,
        "rsi": rsi,
        "above_fast_ma": fast_ma > 0.0 and close > fast_ma,
        "above_slow_ma": slow_ma > 0.0 and close > slow_ma,
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


def _regime_is_tradeable(cfg: IntentConfig) -> tuple[bool, str, str]:
    label = _world_regime_label(cfg)
    phase = _daily_policy_phase(cfg)
    blocked_labels = {item.strip().upper() for item in cfg.blocked_world_regime_labels if item.strip()}
    allowed_labels = {item.strip().upper() for item in cfg.allowed_world_regime_labels if item.strip()}
    blocked_phases = {item.strip().upper() for item in cfg.blocked_daily_policy_phases if item.strip()}

    if not label and cfg.require_world_regime_label:
        return False, "world_regime_missing", "World regime label is missing."
    if label and label in blocked_labels:
        return False, "world_regime_blocked", f"World regime label {label} is blocked."
    if allowed_labels and label and label not in allowed_labels:
        return False, "world_regime_not_allowed", f"World regime label {label} is not allowed."
    if phase and phase in blocked_phases:
        return False, "daily_policy_phase_blocked", f"Daily policy phase {phase} is blocked."
    return True, "", ""


def _fund(fundamental: FundamentalRow, cfg: IntentConfig, short: bool = False) -> float:
    return directional_fundamental_score(
        fundamental,
        short=short,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )


def _pullback_score(pullback: float, cfg: IntentConfig) -> float:
    if pullback < cfg.long_min_pullback_pct or pullback > cfg.long_max_pullback_pct:
        return 0.0
    width = max(cfg.long_max_pullback_pct - cfg.long_min_pullback_pct, 0.01)
    ideal_width = max(width / 2.0, 0.01)
    return clamp(1.0 - abs(pullback - cfg.long_ideal_pullback_pct) / ideal_width, 0.0, 1.0)


def _evaluate_benchmark_long(fundamental: FundamentalRow, cfg: IntentConfig) -> IntentEvaluation:
    return IntentEvaluation(
        None,
        "rejected",
        "benchmark_symbol_context_only",
        f"{cfg.benchmark_symbol} is used only as benchmark context; this model does not trade it.",
    )


def _stock_setup(
    *,
    confirmation: float,
    relative_swing: float,
    relative_confirmation: float,
    pullback: float,
    rebound: float,
    rsi: float,
    atr: float,
    cfg: IntentConfig,
) -> tuple[Optional[str], str]:
    breakout = (
        confirmation >= cfg.min_breakout_confirmation_pct
        and relative_swing >= cfg.min_breakout_relative_swing_pct
        and relative_confirmation >= cfg.min_breakout_relative_confirmation_pct
        and pullback <= cfg.max_breakout_pullback_pct
        and rsi <= cfg.max_chase_rsi
    )
    if breakout:
        return "BREAKOUT_CONFIRMATION", "Strong confirmation while still close to the lookback high."

    reclaim = (
        pullback >= cfg.min_reclaim_pullback_pct
        and rebound >= cfg.min_reclaim_rebound_pct
        and confirmation >= cfg.min_reclaim_confirmation_pct
        and relative_swing >= cfg.min_reclaim_relative_swing_pct
        and rsi <= cfg.long_max_rsi
    )
    if reclaim:
        return "PULLBACK_RECLAIM", "Controlled pullback with rebound and positive swing-relative strength."

    reason = (
        f"No stock alpha setup passed: confirmation {confirmation:.2f}%, relative swing "
        f"{relative_swing:.2f}%, relative confirmation {relative_confirmation:.2f}%, "
        f"pullback {pullback:.2f}%, rebound {rebound:.2f}%, RSI {rsi:.1f}, ATR {atr:.2f}%."
    )
    return None, reason


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
    candidate_symbol = str(fundamental.symbol or "").strip().upper()
    if candidate_symbol == cfg.benchmark_symbol.strip().upper():
        return _evaluate_benchmark_long(fundamental, cfg)

    regime_ok, regime_code, regime_text = _regime_is_tradeable(cfg)
    if not regime_ok:
        return IntentEvaluation(None, "rejected", regime_code, regime_text)

    market_ok, market_code, market_text, market = _market_is_tradeable(cfg)
    if not market_ok:
        return IntentEvaluation(None, "rejected", market_code, market_text)

    composite = float(fundamental.composite_score)
    price_momentum = float(fundamental.price_momentum_score if fundamental.price_momentum_score is not None else composite)
    if composite < cfg.min_long_composite_score:
        return IntentEvaluation(None, "rejected", "composite_below_quality_floor", f"Composite {composite:.1f} below quality floor.")
    if price_momentum < cfg.min_long_price_momentum_score:
        return IntentEvaluation(None, "rejected", "price_momentum_below_floor", f"Price momentum {price_momentum:.1f} below floor.")

    closes = [bar.close for bar in bars]
    highs = [bar.high for bar in bars]
    lows = [bar.low for bar in bars]
    close = closes[-1]
    if close <= 0.0:
        return IntentEvaluation(None, "rejected", "invalid_price", "Close is not positive.")

    benchmark = _benchmark_bars(cfg)
    benchmark_closes = [bar.close for bar in benchmark]
    trend = _ret_pct(closes, cfg.trend_lookback_bars)
    swing = _ret_pct(closes, cfg.swing_lookback_bars)
    confirmation = _ret_pct(closes, cfg.confirmation_bars)
    benchmark_trend = _ret_pct(benchmark_closes, cfg.trend_lookback_bars)
    benchmark_swing = _ret_pct(benchmark_closes, cfg.swing_lookback_bars)
    benchmark_confirmation = _ret_pct(benchmark_closes, cfg.confirmation_bars)
    relative_trend = trend - benchmark_trend
    relative_swing = swing - benchmark_swing
    relative_confirmation = confirmation - benchmark_confirmation

    if trend < cfg.min_trend_return_pct:
        return IntentEvaluation(None, "rejected", "trend_below_min", f"Trend {trend:.2f}% below minimum.")
    if swing < cfg.min_swing_return_pct:
        return IntentEvaluation(None, "rejected", "swing_return_below_min", f"Swing return {swing:.2f}% below minimum.")
    if confirmation < cfg.min_confirmation_pct:
        return IntentEvaluation(None, "rejected", "confirmation_below_min", f"Confirmation {confirmation:.2f}% below minimum.")
    if relative_trend < cfg.min_relative_trend_pct:
        return IntentEvaluation(None, "rejected", "relative_trend_below_min", f"Relative trend {relative_trend:.2f}% below minimum.")
    if relative_trend > cfg.max_relative_trend_pct:
        return IntentEvaluation(
            None,
            "rejected",
            "relative_trend_too_extended",
            f"Relative trend {relative_trend:.2f}% above maximum; avoid late crowded momentum.",
        )
    if relative_swing < cfg.min_relative_swing_pct:
        return IntentEvaluation(None, "rejected", "relative_swing_below_min", f"Relative swing {relative_swing:.2f}% below minimum.")
    if relative_confirmation < cfg.min_relative_confirmation_pct:
        return IntentEvaluation(
            None,
            "rejected",
            "relative_confirmation_below_min",
            f"Relative confirmation {relative_confirmation:.2f}% below minimum.",
        )

    lookback_high = max(highs[-cfg.price_lookback_bars:])
    pullback = (lookback_high - close) / lookback_high * 100.0 if lookback_high > 0.0 else 999.0
    if pullback < cfg.long_min_pullback_pct or pullback > cfg.long_max_pullback_pct:
        return IntentEvaluation(None, "rejected", "pullback_outside_range", f"Pullback {pullback:.2f}% outside range.")

    recent_low = min(lows[-cfg.rebound_bars:])
    rebound = (close / recent_low - 1.0) * 100.0 if recent_low > 0.0 else 0.0
    if rebound < cfg.min_rebound_from_recent_low_pct:
        return IntentEvaluation(None, "rejected", "rebound_too_small", f"Rebound {rebound:.2f}% below minimum.")

    slow_ma = _sma(closes, cfg.slow_ma_bars)
    fast_ma = _sma(closes, cfg.fast_ma_bars)
    if slow_ma > 0.0 and close < slow_ma * (1.0 - cfg.max_below_slow_ma_pct / 100.0):
        return IntentEvaluation(None, "rejected", "too_far_below_slow_ma", "Close is too far below slow MA.")

    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.long_min_rsi or rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_range", f"RSI {rsi:.2f} outside range.")

    atr = _atr_pct(bars, cfg.atr_bars)
    if atr > cfg.max_atr_pct:
        return IntentEvaluation(None, "rejected", "atr_above_max", f"ATR {atr:.2f}% above maximum.")

    setup_name, setup_text = _stock_setup(
        confirmation=confirmation,
        relative_swing=relative_swing,
        relative_confirmation=relative_confirmation,
        pullback=pullback,
        rebound=rebound,
        rsi=rsi,
        atr=atr,
        cfg=cfg,
    )
    if setup_name is None:
        return IntentEvaluation(None, "rejected", "no_stock_alpha_setup", setup_text)

    relative_score = (
        clamp((relative_swing - cfg.min_relative_swing_pct) / 12.0, 0.0, 1.0) * 0.50
        + clamp((relative_confirmation - cfg.min_relative_confirmation_pct) / 6.0, 0.0, 1.0) * 0.30
        + clamp((relative_trend - cfg.min_relative_trend_pct) / 24.0, 0.0, 1.0) * 0.20
    )
    timing_score = (
        clamp((confirmation - cfg.min_confirmation_pct) / 7.0, 0.0, 1.0) * 0.30
        + clamp(rebound / 6.0, 0.0, 1.0) * 0.20
        + _pullback_score(pullback, cfg) * 0.18
        + clamp((rsi - cfg.long_min_rsi) / max(66.0 - cfg.long_min_rsi, 1.0), 0.0, 1.0) * 0.12
        + (1.0 if fast_ma >= slow_ma else 0.4) * 0.20
    )
    market_score = clamp(float(market.get("score", 0.0)) / 10.0, 0.0, 1.0)
    quality_score = _fund(fundamental, cfg, short=False)
    price_momentum_score = _score01(price_momentum)
    volatility_score = clamp((cfg.max_atr_pct - atr) / max(cfg.max_atr_pct, 0.01), 0.0, 1.0)
    setup_score = 1.0 if setup_name == "BREAKOUT_CONFIRMATION" else 0.82

    combined = (
        setup_score * 0.24
        + relative_score * 0.24
        + timing_score * 0.22
        + market_score * 0.12
        + price_momentum_score * 0.08
        + quality_score * 0.06
        + volatility_score * 0.04
    ) * 10.0
    if combined < cfg.min_long_intent_score:
        return IntentEvaluation(None, "rejected", "intent_score_below_min", f"Intent score {combined:.2f} below minimum.")

    reason = (
        f"RSv2 stock | setup {setup_name} | market {float(market.get('score', 0.0)):.2f} | "
        f"rel{cfg.trend_lookback_bars} {relative_trend:.1f}% | "
        f"rel{cfg.swing_lookback_bars} {relative_swing:.1f}% | "
        f"rel{cfg.confirmation_bars} {relative_confirmation:.1f}% | "
        f"trend {trend:.1f}% | swing {swing:.1f}% | confirmation {confirmation:.1f}% | "
        f"pullback {pullback:.1f}% | rebound {rebound:.1f}% | RSI {rsi:.0f} | ATR {atr:.1f}% | "
        f"PM {price_momentum:.1f} | composite {composite:.1f} | regime {_world_regime_label(cfg) or '-'}"
    )
    return IntentEvaluation(
        TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason),
        "intent",
        "relative_strength_market_aware_long_passed",
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
    if not cfg.failure_fast_enabled or str(pos.direction).upper() != "LONG":
        return None

    entry = float(pos.entry_price)
    if entry <= 0.0:
        return None

    high_return = (float(high) / entry - 1.0) * 100.0
    low_return = (float(low) / entry - 1.0) * 100.0
    mfe = max(float(getattr(pos, "model_mfe_pct", high_return)), high_return)
    mae = min(float(getattr(pos, "model_mae_pct", low_return)), low_return)
    setattr(pos, "model_mfe_pct", mfe)
    setattr(pos, "model_mae_pct", mae)

    if not exit_active or bool(getattr(pos, "trailing_activated", False)):
        return None

    current_return = (float(close) / entry - 1.0) * 100.0
    if total_bars >= cfg.failure_fast_min_bars:
        if mfe < cfg.failure_fast_min_mfe_pct and current_return <= cfg.failure_fast_max_return_pct:
            return {
                "exit": True,
                "status": "MODEL_FAILURE_FAST",
                "price": float(close),
                "reason": (
                    f"MFE {mfe:.2f}% below {cfg.failure_fast_min_mfe_pct:.2f}% "
                    f"after {total_bars} bars; current return {current_return:.2f}%."
                ),
            }

    if total_bars >= cfg.relative_failure_min_bars:
        if current_return <= -cfg.relative_failure_loss_pct and mfe < cfg.relative_failure_mfe_cap_pct:
            return {
                "exit": True,
                "status": "MODEL_FAILURE_FAST",
                "price": float(close),
                "reason": (
                    f"Return {current_return:.2f}% with capped MFE {mfe:.2f}% "
                    f"after {total_bars} bars; MAE {mae:.2f}%."
                ),
            }

    return None
