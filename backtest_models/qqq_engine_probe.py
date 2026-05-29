"""QQQ engine probe model.

This diagnostic model is intentionally simple: it emits QQQ intents with no
fundamental or technical opinion. The runner still owns point-in-time data,
entry timing, stops, sizing, broker rules, regime policy, and exits.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, IntentEvaluation, TradeIntent
from backtest_shared import env_bool, env_float, env_int, env_optional_float, env_str


MODEL_NAME = "QQQ-Engine-Probe"
BENCHMARK_SYMBOL = "QQQ"
DIRECT_CANDIDATE_SYMBOLS = (BENCHMARK_SYMBOL,)
DIRECT_CANDIDATE_MODE = "replace"
DIRECT_CANDIDATE_REQUIRE_BROKER_ELIGIBILITY = False
ALLOW_MULTIPLE_POSITIONS_PER_INSTRUMENT = False


@dataclass
class IntentConfig:
    """Parameters for qqq_engine_probe."""

    min_bars: int = 1
    intent_score: float = 10.0
    direction_mode: str = "long_only"
    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None
    price_lookback_bars: int = 1
    reject_non_qqq: bool = True


def intent_config_from_env() -> IntentConfig:
    defaults = IntentConfig()
    return IntentConfig(
        min_bars=env_int("MIN_BARS", defaults.min_bars),
        intent_score=env_float("INTENT_SCORE", defaults.intent_score),
        direction_mode=env_str("DIRECTION_MODE", defaults.direction_mode).lower(),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", defaults.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", defaults.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", defaults.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", defaults.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", defaults.short_max_absolute_score),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", defaults.price_lookback_bars),
        reject_non_qqq=env_bool("REJECT_NON_QQQ", defaults.reject_non_qqq),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(1, cfg.min_bars, cfg.price_lookback_bars)


def iter_grid_search_configs(base_cfg: IntentConfig, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=QQQ-Engine-Probe", "summary": {}}


def compute_long_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def compute_short_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    return _evaluate(fundamental, cfg, "LONG")


def evaluate_short_intent(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: IntentConfig,
) -> IntentEvaluation:
    return _evaluate(fundamental, cfg, "SHORT")


def _evaluate(fundamental: FundamentalRow, cfg: IntentConfig, direction: str) -> IntentEvaluation:
    mode = str(cfg.direction_mode or "").strip().lower()
    if mode not in {"long_only", "short_only", "long_short"}:
        return IntentEvaluation(
            None,
            "rejected",
            "invalid_direction_mode",
            f"{MODEL_NAME} DIRECTION_MODE must be long_only, short_only, or long_short.",
        )
    if direction == "LONG" and mode == "short_only":
        return IntentEvaluation(
            None,
            "rejected",
            "probe_long_disabled",
            f"{MODEL_NAME} direction mode {mode} disables long intents.",
        )
    if direction == "SHORT" and mode == "long_only":
        return IntentEvaluation(
            None,
            "rejected",
            "probe_short_disabled",
            f"{MODEL_NAME} direction mode {mode} disables short intents.",
        )

    candidate_symbol = str(fundamental.symbol or "").strip().upper()
    if cfg.reject_non_qqq and candidate_symbol != BENCHMARK_SYMBOL:
        return IntentEvaluation(
            None,
            "rejected",
            "probe_symbol_mismatch",
            f"{MODEL_NAME} only emits {BENCHMARK_SYMBOL}; candidate was {candidate_symbol or '-'}.",
        )

    reason = f"{MODEL_NAME} {direction.lower()} probe intent for {BENCHMARK_SYMBOL} with runner regime policy."
    return IntentEvaluation(
        TradeIntent(BENCHMARK_SYMBOL, direction, cfg.intent_score, reason),
        "intent",
        "qqq_engine_probe_passed",
        reason,
    )
