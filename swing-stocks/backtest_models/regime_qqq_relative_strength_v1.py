"""Regime-aware relative strength model against QQQ.

The model is point-in-time by construction: it only sees completed 1h bars up
to the runner cutoff plus the previous-day world-regime state supplied by the
daily position policy.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime

from backtest_shared import Bar, CandidateRow, IntentEvaluation, TradeIntent
from backtest_shared import clamp, env_bool, env_float, env_int, env_list, env_str, mean


MODEL_NAME = "regime_qqq_relative_strength_v1"
BENCHMARK_SYMBOL = "QQQ"
BENCHMARK_SYMBOLS = (BENCHMARK_SYMBOL,)
BENCHMARK_BAR_LOOKBACK = 520
DIRECT_CANDIDATE_SYMBOLS = ()
DIRECT_CANDIDATE_MODE = "append"
DIRECT_CANDIDATE_REQUIRE_BROKER_ELIGIBILITY = True


@dataclass
class IntentConfig:
    min_bars: int = 420
    enable_longs: bool = True
    enable_shorts: bool = True

    benchmark_symbol: str = BENCHMARK_SYMBOL
    require_benchmark_context: bool = True

    low_stress_max_score: float = 55.0
    high_stress_min_score: float = 65.0
    allow_mid_stress_longs: bool = False
    allow_mid_stress_shorts: bool = False

    long_primary_lookback_bars: int = 130
    long_persistence_lookback_bars: int = 390
    long_confirmation_lookback_bars: int = 39
    long_fast_ma_bars: int = 65
    long_slow_ma_bars: int = 260
    long_max_drawdown_pct: float = 18.0
    min_long_stock_primary_return_pct: float = 0.0
    min_long_excess_primary_pct: float = 1.0
    min_long_excess_persistence_pct: float = 0.0
    min_long_excess_confirmation_pct: float = -0.5
    min_long_intent_score: float = 6.0

    short_fast_lookback_bars: int = 39
    short_primary_lookback_bars: int = 130
    short_fast_ma_bars: int = 39
    short_slow_ma_bars: int = 130
    max_short_stock_fast_return_pct: float = -2.0
    max_short_excess_fast_pct: float = -4.0
    max_short_excess_primary_pct: float = -3.0
    min_short_drawdown_pct: float = 8.0
    min_short_intent_score: float = 6.0

    blocked_world_regime_labels: tuple[str, ...] = ()


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        enable_longs=env_bool("ENABLE_LONGS", d.enable_longs),
        enable_shorts=env_bool("ENABLE_SHORTS", d.enable_shorts),
        benchmark_symbol=env_str("BENCHMARK_SYMBOL", d.benchmark_symbol).upper(),
        require_benchmark_context=env_bool("REQUIRE_BENCHMARK_CONTEXT", d.require_benchmark_context),
        low_stress_max_score=env_float("LOW_STRESS_MAX_SCORE", d.low_stress_max_score),
        high_stress_min_score=env_float("HIGH_STRESS_MIN_SCORE", d.high_stress_min_score),
        allow_mid_stress_longs=env_bool("ALLOW_MID_STRESS_LONGS", d.allow_mid_stress_longs),
        allow_mid_stress_shorts=env_bool("ALLOW_MID_STRESS_SHORTS", d.allow_mid_stress_shorts),
        long_primary_lookback_bars=env_int("LONG_PRIMARY_LOOKBACK_BARS", d.long_primary_lookback_bars),
        long_persistence_lookback_bars=env_int(
            "LONG_PERSISTENCE_LOOKBACK_BARS",
            d.long_persistence_lookback_bars,
        ),
        long_confirmation_lookback_bars=env_int(
            "LONG_CONFIRMATION_LOOKBACK_BARS",
            d.long_confirmation_lookback_bars,
        ),
        long_fast_ma_bars=env_int("LONG_FAST_MA_BARS", d.long_fast_ma_bars),
        long_slow_ma_bars=env_int("LONG_SLOW_MA_BARS", d.long_slow_ma_bars),
        long_max_drawdown_pct=env_float("LONG_MAX_DRAWDOWN_PCT", d.long_max_drawdown_pct),
        min_long_stock_primary_return_pct=env_float(
            "MIN_LONG_STOCK_PRIMARY_RETURN_PCT",
            d.min_long_stock_primary_return_pct,
        ),
        min_long_excess_primary_pct=env_float("MIN_LONG_EXCESS_PRIMARY_PCT", d.min_long_excess_primary_pct),
        min_long_excess_persistence_pct=env_float(
            "MIN_LONG_EXCESS_PERSISTENCE_PCT",
            d.min_long_excess_persistence_pct,
        ),
        min_long_excess_confirmation_pct=env_float(
            "MIN_LONG_EXCESS_CONFIRMATION_PCT",
            d.min_long_excess_confirmation_pct,
        ),
        min_long_intent_score=env_float("MIN_LONG_INTENT_SCORE", d.min_long_intent_score),
        short_fast_lookback_bars=env_int("SHORT_FAST_LOOKBACK_BARS", d.short_fast_lookback_bars),
        short_primary_lookback_bars=env_int("SHORT_PRIMARY_LOOKBACK_BARS", d.short_primary_lookback_bars),
        short_fast_ma_bars=env_int("SHORT_FAST_MA_BARS", d.short_fast_ma_bars),
        short_slow_ma_bars=env_int("SHORT_SLOW_MA_BARS", d.short_slow_ma_bars),
        max_short_stock_fast_return_pct=env_float(
            "MAX_SHORT_STOCK_FAST_RETURN_PCT",
            d.max_short_stock_fast_return_pct,
        ),
        max_short_excess_fast_pct=env_float("MAX_SHORT_EXCESS_FAST_PCT", d.max_short_excess_fast_pct),
        max_short_excess_primary_pct=env_float(
            "MAX_SHORT_EXCESS_PRIMARY_PCT",
            d.max_short_excess_primary_pct,
        ),
        min_short_drawdown_pct=env_float("MIN_SHORT_DRAWDOWN_PCT", d.min_short_drawdown_pct),
        min_short_intent_score=env_float("MIN_SHORT_INTENT_SCORE", d.min_short_intent_score),
        blocked_world_regime_labels=tuple(
            label.strip().upper()
            for label in env_list("BLOCKED_WORLD_REGIME_LABELS", d.blocked_world_regime_labels)
            if label.strip()
        ),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.long_primary_lookback_bars + 1,
        cfg.long_persistence_lookback_bars + 1,
        cfg.long_confirmation_lookback_bars + 1,
        cfg.long_slow_ma_bars,
        cfg.short_fast_lookback_bars + 1,
        cfg.short_primary_lookback_bars + 1,
        cfg.short_slow_ma_bars,
        80,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": f"grid model={MODEL_NAME}", "summary": {}}


def set_market_context(cfg: IntentConfig, as_of_ts: datetime, bars_by_symbol: dict[str, list[Bar]]) -> None:
    setattr(cfg, "market_context_as_of_ts", as_of_ts)
    setattr(cfg, "market_context_bars_by_symbol", bars_by_symbol)


def _benchmark_bars(cfg: IntentConfig) -> list[Bar]:
    by_symbol = getattr(cfg, "market_context_bars_by_symbol", {}) or {}
    return list(by_symbol.get(cfg.benchmark_symbol.strip().upper(), []))


def _regime_score(cfg: IntentConfig) -> float | None:
    value = getattr(cfg, "daily_policy_world_regime_ma_score", None)
    return float(value) if value is not None else None


def _regime_label(cfg: IntentConfig) -> str:
    return str(getattr(cfg, "daily_policy_world_regime_label", "") or "").strip().upper()


def _daily_phase(cfg: IntentConfig) -> str:
    return str(getattr(cfg, "daily_policy_phase", "") or "").strip().upper()


def _regime_is_blocked(cfg: IntentConfig) -> bool:
    label = _regime_label(cfg)
    blocked = {item.strip().upper() for item in cfg.blocked_world_regime_labels if item.strip()}
    return bool(label and label in blocked)


def _is_low_stress(cfg: IntentConfig) -> bool:
    score = _regime_score(cfg)
    phase = _daily_phase(cfg)
    if score is None:
        return phase == "LOW_STRESS"
    if score <= cfg.low_stress_max_score:
        return True
    return cfg.allow_mid_stress_longs and score < cfg.high_stress_min_score


def _is_high_stress(cfg: IntentConfig) -> bool:
    score = _regime_score(cfg)
    phase = _daily_phase(cfg)
    if score is None:
        return phase == "STRESS_HIGH"
    if score >= cfg.high_stress_min_score:
        return True
    return cfg.allow_mid_stress_shorts and score > cfg.low_stress_max_score


def _ret_pct(closes: list[float], lookback: int) -> float:
    if lookback <= 0 or len(closes) <= lookback:
        return 0.0
    base = closes[-lookback]
    return (closes[-1] / base - 1.0) * 100.0 if base > 0.0 else 0.0


def _sma(values: list[float], n: int) -> float:
    return mean(values[-n:]) if values and n > 0 and len(values) >= n else 0.0


def _drawdown_pct(bars: list[Bar], lookback: int) -> float:
    if not bars:
        return 0.0
    subset = bars[-min(len(bars), lookback):]
    high = max(float(bar.high) for bar in subset)
    close = float(bars[-1].close)
    return (high - close) / high * 100.0 if high > 0.0 else 0.0


def _benchmark_rets(cfg: IntentConfig, lookbacks: tuple[int, ...]) -> dict[int, float] | None:
    bars = _benchmark_bars(cfg)
    if len(bars) <= max(lookbacks):
        return None
    closes = [bar.close for bar in bars]
    return {lookback: _ret_pct(closes, lookback) for lookback in lookbacks}


def _benchmark_unavailable(cfg: IntentConfig) -> IntentEvaluation | None:
    if not cfg.require_benchmark_context:
        return None
    return IntentEvaluation(None, "rejected", "benchmark_context_unavailable", "QQQ benchmark bars are unavailable.")


def _reject_benchmark(candidate: CandidateRow, cfg: IntentConfig) -> IntentEvaluation | None:
    if candidate.symbol.strip().upper() != cfg.benchmark_symbol.strip().upper():
        return None
    return IntentEvaluation(
        None,
        "rejected",
        "benchmark_symbol_context_only",
        f"{cfg.benchmark_symbol} is used only as benchmark context.",
    )


def compute_long_intent(
    bars: list[Bar],
    candidate: CandidateRow,
    now: datetime,
    cfg: IntentConfig,
) -> TradeIntent | None:
    return evaluate_long_intent(bars, candidate, now, cfg).intent


def evaluate_long_intent(
    bars: list[Bar],
    candidate: CandidateRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    benchmark_reject = _reject_benchmark(candidate, cfg)
    if benchmark_reject is not None:
        return benchmark_reject
    if not cfg.enable_longs:
        return IntentEvaluation(None, "rejected", "longs_disabled", "Long side is disabled.")
    if _regime_is_blocked(cfg):
        return IntentEvaluation(None, "rejected", "world_regime_label_blocked", "World regime label is blocked.")
    if not _is_low_stress(cfg):
        return IntentEvaluation(
            None,
            "rejected",
            "world_regime_not_low_stress",
            f"World regime score {_regime_score(cfg)} is not low-stress enough for longs.",
        )

    lookbacks = (
        cfg.long_primary_lookback_bars,
        cfg.long_persistence_lookback_bars,
        cfg.long_confirmation_lookback_bars,
    )
    qqq = _benchmark_rets(cfg, lookbacks)
    if qqq is None:
        unavailable = _benchmark_unavailable(cfg)
        if unavailable is not None:
            return unavailable
        qqq = {lookback: 0.0 for lookback in lookbacks}

    closes = [bar.close for bar in bars]
    close = closes[-1]
    if close <= 0.0:
        return IntentEvaluation(None, "rejected", "invalid_price", "Close is not positive.")

    primary = _ret_pct(closes, cfg.long_primary_lookback_bars)
    persistence = _ret_pct(closes, cfg.long_persistence_lookback_bars)
    confirmation = _ret_pct(closes, cfg.long_confirmation_lookback_bars)
    excess_primary = primary - qqq[cfg.long_primary_lookback_bars]
    excess_persistence = persistence - qqq[cfg.long_persistence_lookback_bars]
    excess_confirmation = confirmation - qqq[cfg.long_confirmation_lookback_bars]

    if primary < cfg.min_long_stock_primary_return_pct:
        return IntentEvaluation(None, "rejected", "long_stock_return_below_min", f"Primary return {primary:.2f}% below minimum.")
    if excess_primary < cfg.min_long_excess_primary_pct:
        return IntentEvaluation(None, "rejected", "long_excess_primary_below_min", f"Primary QQQ-excess {excess_primary:.2f}% below minimum.")
    if excess_persistence < cfg.min_long_excess_persistence_pct:
        return IntentEvaluation(None, "rejected", "long_excess_persistence_below_min", f"Persistent QQQ-excess {excess_persistence:.2f}% below minimum.")
    if excess_confirmation < cfg.min_long_excess_confirmation_pct:
        return IntentEvaluation(None, "rejected", "long_excess_confirmation_below_min", f"Confirmation QQQ-excess {excess_confirmation:.2f}% below minimum.")

    fast_ma = _sma(closes, cfg.long_fast_ma_bars)
    slow_ma = _sma(closes, cfg.long_slow_ma_bars)
    if fast_ma <= 0.0 or slow_ma <= 0.0 or close < slow_ma or fast_ma < slow_ma:
        return IntentEvaluation(None, "rejected", "long_trend_filter_failed", "Long trend filter failed.")

    drawdown = _drawdown_pct(bars, cfg.long_slow_ma_bars)
    if drawdown > cfg.long_max_drawdown_pct:
        return IntentEvaluation(None, "rejected", "long_drawdown_above_max", f"Drawdown {drawdown:.2f}% above maximum.")

    quiet_score = clamp((cfg.low_stress_max_score - (_regime_score(cfg) or cfg.low_stress_max_score)) / 20.0, 0.0, 1.0)
    relative_score = (
        clamp((excess_primary - cfg.min_long_excess_primary_pct) / 12.0, 0.0, 1.0) * 0.45
        + clamp((excess_persistence - cfg.min_long_excess_persistence_pct) / 18.0, 0.0, 1.0) * 0.35
        + clamp((excess_confirmation - cfg.min_long_excess_confirmation_pct) / 6.0, 0.0, 1.0) * 0.20
    )
    trend_score = (
        clamp((close / slow_ma - 1.0) * 10.0, 0.0, 1.0) * 0.45
        + clamp((fast_ma / slow_ma - 1.0) * 12.0, 0.0, 1.0) * 0.35
        + clamp((cfg.long_max_drawdown_pct - drawdown) / cfg.long_max_drawdown_pct, 0.0, 1.0) * 0.20
    )
    combined = (relative_score * 0.62 + trend_score * 0.28 + quiet_score * 0.10) * 10.0
    if combined < cfg.min_long_intent_score:
        return IntentEvaluation(None, "rejected", "long_intent_score_below_min", f"Intent score {combined:.2f} below minimum.")

    reason = (
        f"Regime QQQ RS long | regime {_regime_score(cfg)} {_daily_phase(cfg) or '-'} | "
        f"excess{cfg.long_primary_lookback_bars} {excess_primary:.1f}% | "
        f"excess{cfg.long_persistence_lookback_bars} {excess_persistence:.1f}% | "
        f"excess{cfg.long_confirmation_lookback_bars} {excess_confirmation:.1f}% | "
        f"stock primary {primary:.1f}% | drawdown {drawdown:.1f}%"
    )
    return IntentEvaluation(
        TradeIntent(candidate.symbol, "LONG", round(combined, 4), reason),
        "intent",
        "regime_qqq_relative_strength_long_passed",
        reason,
    )


def compute_short_intent(
    bars: list[Bar],
    candidate: CandidateRow,
    now: datetime,
    cfg: IntentConfig,
) -> TradeIntent | None:
    return evaluate_short_intent(bars, candidate, now, cfg).intent


def evaluate_short_intent(
    bars: list[Bar],
    candidate: CandidateRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    benchmark_reject = _reject_benchmark(candidate, cfg)
    if benchmark_reject is not None:
        return benchmark_reject
    if not cfg.enable_shorts:
        return IntentEvaluation(None, "rejected", "shorts_disabled", "Short side is disabled.")
    if _regime_is_blocked(cfg):
        return IntentEvaluation(None, "rejected", "world_regime_label_blocked", "World regime label is blocked.")
    if not _is_high_stress(cfg):
        return IntentEvaluation(
            None,
            "rejected",
            "world_regime_not_high_stress",
            f"World regime score {_regime_score(cfg)} is not high-stress enough for shorts.",
        )

    lookbacks = (cfg.short_fast_lookback_bars, cfg.short_primary_lookback_bars)
    qqq = _benchmark_rets(cfg, lookbacks)
    if qqq is None:
        unavailable = _benchmark_unavailable(cfg)
        if unavailable is not None:
            return unavailable
        qqq = {lookback: 0.0 for lookback in lookbacks}

    closes = [bar.close for bar in bars]
    close = closes[-1]
    if close <= 0.0:
        return IntentEvaluation(None, "rejected", "invalid_price", "Close is not positive.")

    fast_return = _ret_pct(closes, cfg.short_fast_lookback_bars)
    primary = _ret_pct(closes, cfg.short_primary_lookback_bars)
    excess_fast = fast_return - qqq[cfg.short_fast_lookback_bars]
    excess_primary = primary - qqq[cfg.short_primary_lookback_bars]

    if fast_return > cfg.max_short_stock_fast_return_pct:
        return IntentEvaluation(None, "rejected", "short_stock_fast_return_too_strong", f"Fast return {fast_return:.2f}% above short maximum.")
    if excess_fast > cfg.max_short_excess_fast_pct:
        return IntentEvaluation(None, "rejected", "short_fast_excess_not_weak_enough", f"Fast QQQ-excess {excess_fast:.2f}% above maximum.")
    if excess_primary > cfg.max_short_excess_primary_pct:
        return IntentEvaluation(None, "rejected", "short_primary_excess_not_weak_enough", f"Primary QQQ-excess {excess_primary:.2f}% above maximum.")

    fast_ma = _sma(closes, cfg.short_fast_ma_bars)
    slow_ma = _sma(closes, cfg.short_slow_ma_bars)
    if fast_ma <= 0.0 or slow_ma <= 0.0 or close > fast_ma or fast_ma > slow_ma:
        return IntentEvaluation(None, "rejected", "short_downtrend_filter_failed", "Short downtrend filter failed.")

    drawdown = _drawdown_pct(bars, cfg.short_primary_lookback_bars)
    if drawdown < cfg.min_short_drawdown_pct:
        return IntentEvaluation(None, "rejected", "short_drawdown_below_min", f"Drawdown {drawdown:.2f}% below minimum.")

    stress_score = clamp(((_regime_score(cfg) or cfg.high_stress_min_score) - cfg.high_stress_min_score) / 20.0, 0.0, 1.0)
    relative_score = (
        clamp((cfg.max_short_excess_fast_pct - excess_fast) / 12.0, 0.0, 1.0) * 0.55
        + clamp((cfg.max_short_excess_primary_pct - excess_primary) / 18.0, 0.0, 1.0) * 0.45
    )
    trend_score = (
        clamp((slow_ma / max(close, 0.01) - 1.0) * 5.0, 0.0, 1.0) * 0.45
        + clamp((slow_ma / max(fast_ma, 0.01) - 1.0) * 8.0, 0.0, 1.0) * 0.30
        + clamp((drawdown - cfg.min_short_drawdown_pct) / 18.0, 0.0, 1.0) * 0.25
    )
    combined = (relative_score * 0.62 + trend_score * 0.28 + stress_score * 0.10) * 10.0
    if combined < cfg.min_short_intent_score:
        return IntentEvaluation(None, "rejected", "short_intent_score_below_min", f"Intent score {combined:.2f} below minimum.")

    reason = (
        f"Regime QQQ RS short | regime {_regime_score(cfg)} {_daily_phase(cfg) or '-'} | "
        f"excess{cfg.short_fast_lookback_bars} {excess_fast:.1f}% | "
        f"excess{cfg.short_primary_lookback_bars} {excess_primary:.1f}% | "
        f"stock fast {fast_return:.1f}% | stock primary {primary:.1f}% | drawdown {drawdown:.1f}%"
    )
    return IntentEvaluation(
        TradeIntent(candidate.symbol, "SHORT", round(combined, 4), reason),
        "intent",
        "regime_qqq_relative_strength_short_passed",
        reason,
    )


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
    return None
