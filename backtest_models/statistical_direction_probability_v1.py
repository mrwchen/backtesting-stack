"""Statistical direction/fundamental probability model.

Model idea:
  - The generic runner chooses direction exposure from world-regime score.
  - LONG side: good fundamentals, only when a
    point-in-time empirical event study estimates high next-day up probability.
  - SHORT side: weak fundamentals, only when a
    point-in-time empirical event study estimates high next-day down probability.

The model intentionally uses no RSI, EMA, moving-average crossover, chart
patterns, or other technical-analysis indicators. It estimates probabilities
from past same-symbol daily return events that look statistically similar to
today's recent drawdown event.
"""

import dataclasses
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from backtest_shared import Bar, FundamentalRow, TradeIntent, IntentEvaluation
from backtest_shared import clamp, env_bool, env_float, env_int, mean


@dataclass
class IntentConfig:
    """Parameters for statistical_direction_probability_v1."""

    # Keep enough 1h history so the model can build daily event samples.
    min_bars: int = 650
    price_lookback_bars: int = 2200

    # Legacy runner/run-result fields. They are not used as TA inputs here.
    long_min_pullback: float = 0.0
    long_max_pullback: float = 30.0
    long_ideal_pullback: float = 5.0
    long_max_rsi: float = 100.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 30.0
    short_ideal_bounce: float = 5.0
    short_min_rsi: float = 0.0
    short_max_rsi: float = 100.0

    use_mispricing_score: bool = True
    mispricing_weight: float = 0.30

    session_tz: str = "America/New_York"
    event_lookback_days: int = 5
    volatility_lookback_days: int = 20
    min_daily_observations: int = 90
    min_consecutive_down_days: int = 2
    min_event_drop_pct: float = 2.0

    min_analog_events: int = 25
    max_analog_events: int = 120
    full_weight_analog_events: int = 80
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    return_similarity_scale_pct: float = 2.5
    vol_similarity_scale_pct: float = 1.0
    analog_same_event_direction: bool = True

    long_min_probability: float = 0.56
    short_min_probability: float = 0.56
    min_edge_pct_points: float = 2.0
    edge_score_scale_pct_points: float = 12.0


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    shared_min_bars = env_int("MIN_BARS", d.min_bars)
    shared_lookback_bars = env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars)
    return IntentConfig(
        min_bars=env_int("PROBABILITY_MIN_BARS", max(d.min_bars, shared_min_bars)),
        price_lookback_bars=env_int("PROBABILITY_HISTORY_BARS", max(d.price_lookback_bars, shared_lookback_bars)),
        long_min_pullback=d.long_min_pullback,
        long_max_pullback=d.long_max_pullback,
        long_ideal_pullback=d.long_ideal_pullback,
        long_max_rsi=d.long_max_rsi,
        short_min_bounce=d.short_min_bounce,
        short_max_bounce=d.short_max_bounce,
        short_ideal_bounce=d.short_ideal_bounce,
        short_min_rsi=d.short_min_rsi,
        short_max_rsi=d.short_max_rsi,
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        session_tz=os.getenv("PROBABILITY_SESSION_TZ", d.session_tz).strip() or d.session_tz,
        event_lookback_days=env_int("EVENT_LOOKBACK_DAYS", d.event_lookback_days),
        volatility_lookback_days=env_int("VOLATILITY_LOOKBACK_DAYS", d.volatility_lookback_days),
        min_daily_observations=env_int("MIN_DAILY_OBSERVATIONS", d.min_daily_observations),
        min_consecutive_down_days=env_int("MIN_CONSECUTIVE_DOWN_DAYS", d.min_consecutive_down_days),
        min_event_drop_pct=env_float("MIN_EVENT_DROP_PCT", d.min_event_drop_pct),
        min_analog_events=env_int("MIN_ANALOG_EVENTS", d.min_analog_events),
        max_analog_events=env_int("MAX_ANALOG_EVENTS", d.max_analog_events),
        full_weight_analog_events=env_int("FULL_WEIGHT_ANALOG_EVENTS", d.full_weight_analog_events),
        prior_alpha=env_float("PRIOR_ALPHA", d.prior_alpha),
        prior_beta=env_float("PRIOR_BETA", d.prior_beta),
        return_similarity_scale_pct=env_float("RETURN_SIMILARITY_SCALE_PCT", d.return_similarity_scale_pct),
        vol_similarity_scale_pct=env_float("VOL_SIMILARITY_SCALE_PCT", d.vol_similarity_scale_pct),
        analog_same_event_direction=env_bool("ANALOG_SAME_EVENT_DIRECTION", d.analog_same_event_direction),
        long_min_probability=env_float("LONG_MIN_PROBABILITY", d.long_min_probability),
        short_min_probability=env_float("SHORT_MIN_PROBABILITY", d.short_min_probability),
        min_edge_pct_points=env_float("MIN_EDGE_PCT_POINTS", d.min_edge_pct_points),
        edge_score_scale_pct_points=env_float("EDGE_SCORE_SCALE_PCT_POINTS", d.edge_score_scale_pct_points),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(cfg.min_bars, cfg.price_lookback_bars)


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {
        "config": dataclasses.replace(base_cfg),
        "notes": "grid model=statistical_direction_probability_v1",
        "summary": {},
    }


@dataclass(frozen=True)
class ProbabilityEstimate:
    p_up: float
    p_down: float
    baseline_up: float
    baseline_down: float
    analog_count: int
    baseline_count: int
    event_return_pct: float
    last_return_pct: float
    down_streak: int
    up_streak: int
    realized_vol_pct: float


def _daily_closes(bars: list[Bar], cfg: IntentConfig) -> list[tuple[object, float]]:
    zone = ZoneInfo(cfg.session_tz)
    latest_by_day: dict[object, tuple[datetime, float]] = {}
    for bar in bars:
        if bar.close <= 0:
            continue
        ts = bar.ts
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        local_ts = ts.astimezone(zone)
        day = local_ts.date()
        current = latest_by_day.get(day)
        if current is None or local_ts > current[0]:
            latest_by_day[day] = (local_ts, float(bar.close))
    return [(day, close) for day, (_ts, close) in sorted(latest_by_day.items())]


def _pct_return(new_value: float, old_value: float) -> float:
    if old_value <= 0:
        return 0.0
    return (new_value / old_value - 1.0) * 100.0


def _streak(returns_pct: list[float], end_exclusive: int, positive: bool) -> int:
    count = 0
    idx = end_exclusive - 1
    while idx >= 0:
        value = returns_pct[idx]
        if positive and value > 0:
            count += 1
        elif not positive and value < 0:
            count += 1
        else:
            break
        idx -= 1
    return count


def _stddev(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _event_features(closes: list[float], returns_pct: list[float], close_idx: int, cfg: IntentConfig) -> tuple[float, float, int, int, float]:
    lookback = cfg.event_lookback_days
    event_return = _pct_return(closes[close_idx], closes[close_idx - lookback])
    last_return = returns_pct[close_idx - 1] if close_idx > 0 else 0.0
    trailing = returns_pct[max(0, close_idx - cfg.volatility_lookback_days):close_idx]
    realized_vol = _stddev(trailing)
    down_streak = _streak(returns_pct, close_idx, positive=False)
    up_streak = _streak(returns_pct, close_idx, positive=True)
    return event_return, last_return, down_streak, up_streak, realized_vol


def _posterior_rate(successes: int, count: int, cfg: IntentConfig) -> float:
    return (successes + cfg.prior_alpha) / (count + cfg.prior_alpha + cfg.prior_beta)


def _blend_rate(analog_rate: float, baseline_rate: float, analog_count: int, cfg: IntentConfig) -> float:
    full_weight = max(cfg.full_weight_analog_events, 1)
    weight = clamp(analog_count / full_weight, 0.0, 1.0)
    return analog_rate * weight + baseline_rate * (1.0 - weight)


def _estimate_probability(bars: list[Bar], cfg: IntentConfig) -> Optional[ProbabilityEstimate]:
    daily = _daily_closes(bars, cfg)
    if len(daily) < max(cfg.min_daily_observations, cfg.event_lookback_days + 3):
        return None

    closes = [close for _day, close in daily]
    returns_pct = [_pct_return(closes[idx], closes[idx - 1]) for idx in range(1, len(closes))]
    current_idx = len(closes) - 1
    if current_idx <= cfg.event_lookback_days:
        return None

    cur_event_return, cur_last_return, cur_down_streak, cur_up_streak, cur_vol = _event_features(
        closes,
        returns_pct,
        current_idx,
        cfg,
    )

    if cur_event_return > -cfg.min_event_drop_pct:
        return ProbabilityEstimate(
            p_up=0.0,
            p_down=0.0,
            baseline_up=0.0,
            baseline_down=0.0,
            analog_count=0,
            baseline_count=0,
            event_return_pct=cur_event_return,
            last_return_pct=cur_last_return,
            down_streak=cur_down_streak,
            up_streak=cur_up_streak,
            realized_vol_pct=cur_vol,
        )
    if cur_down_streak < cfg.min_consecutive_down_days:
        return ProbabilityEstimate(
            p_up=0.0,
            p_down=0.0,
            baseline_up=0.0,
            baseline_down=0.0,
            analog_count=0,
            baseline_count=0,
            event_return_pct=cur_event_return,
            last_return_pct=cur_last_return,
            down_streak=cur_down_streak,
            up_streak=cur_up_streak,
            realized_vol_pct=cur_vol,
        )

    event_rows: list[tuple[float, float]] = []
    baseline_targets: list[float] = []
    return_scale = max(cfg.return_similarity_scale_pct, abs(cur_vol) * math.sqrt(max(cfg.event_lookback_days, 1)), 0.5)
    last_return_scale = max(return_scale / max(cfg.event_lookback_days, 1), 0.25)
    vol_scale = max(cfg.vol_similarity_scale_pct, cur_vol, 0.5)
    streak_scale = max(float(cfg.event_lookback_days), 1.0)

    for close_idx in range(cfg.event_lookback_days, len(closes) - 1):
        target_return = returns_pct[close_idx]
        if not math.isfinite(target_return):
            continue
        event_return, last_return, down_streak, _up_streak, realized_vol = _event_features(
            closes,
            returns_pct,
            close_idx,
            cfg,
        )
        baseline_targets.append(target_return)
        if cfg.analog_same_event_direction and cur_event_return < 0.0 and event_return > 0.0:
            continue
        distance = (
            abs(event_return - cur_event_return) / return_scale
            + 0.45 * abs(last_return - cur_last_return) / last_return_scale
            + 0.70 * abs(realized_vol - cur_vol) / vol_scale
            + 0.65 * abs(down_streak - cur_down_streak) / streak_scale
        )
        event_rows.append((distance, target_return))

    if len(baseline_targets) < cfg.min_analog_events or not event_rows:
        return None

    event_rows.sort(key=lambda row: row[0])
    analog_targets = [target for _distance, target in event_rows[:cfg.max_analog_events]]
    if len(analog_targets) < cfg.min_analog_events:
        return ProbabilityEstimate(
            p_up=0.0,
            p_down=0.0,
            baseline_up=0.0,
            baseline_down=0.0,
            analog_count=len(analog_targets),
            baseline_count=len(baseline_targets),
            event_return_pct=cur_event_return,
            last_return_pct=cur_last_return,
            down_streak=cur_down_streak,
            up_streak=cur_up_streak,
            realized_vol_pct=cur_vol,
        )

    analog_up = sum(1 for value in analog_targets if value > 0.0)
    analog_down = sum(1 for value in analog_targets if value < 0.0)
    baseline_up = sum(1 for value in baseline_targets if value > 0.0)
    baseline_down = sum(1 for value in baseline_targets if value < 0.0)

    analog_p_up = _posterior_rate(analog_up, len(analog_targets), cfg)
    analog_p_down = _posterior_rate(analog_down, len(analog_targets), cfg)
    base_p_up = _posterior_rate(baseline_up, len(baseline_targets), cfg)
    base_p_down = _posterior_rate(baseline_down, len(baseline_targets), cfg)

    return ProbabilityEstimate(
        p_up=_blend_rate(analog_p_up, base_p_up, len(analog_targets), cfg),
        p_down=_blend_rate(analog_p_down, base_p_down, len(analog_targets), cfg),
        baseline_up=base_p_up,
        baseline_down=base_p_down,
        analog_count=len(analog_targets),
        baseline_count=len(baseline_targets),
        event_return_pct=cur_event_return,
        last_return_pct=cur_last_return,
        down_streak=cur_down_streak,
        up_streak=cur_up_streak,
        realized_vol_pct=cur_vol,
    )


def _fundamental_score(fundamental: FundamentalRow, cfg: IntentConfig, short: bool) -> float:
    score = fundamental.composite_score
    if cfg.use_mispricing_score and fundamental.mispricing_score is not None:
        score = score * (1.0 - cfg.mispricing_weight) + fundamental.mispricing_score * cfg.mispricing_weight
    return clamp((100.0 - score if short else score) / 100.0, 0.0, 1.0)


def _build_intent(
    fundamental: FundamentalRow,
    cfg: IntentConfig,
    prob: ProbabilityEstimate,
    short: bool,
) -> TradeIntent:
    success_prob = prob.p_down if short else prob.p_up
    baseline_prob = prob.baseline_down if short else prob.baseline_up
    edge_pct_points = (success_prob - baseline_prob) * 100.0
    edge_score = clamp(edge_pct_points / max(cfg.edge_score_scale_pct_points, 0.01), 0.0, 1.0)
    fund_score = _fundamental_score(fundamental, cfg, short)
    combined = (success_prob * 0.60 + edge_score * 0.20 + fund_score * 0.20) * 10.0
    if short:
        direction = "SHORT"
        reason_code = "p_down"
    else:
        direction = "LONG"
        reason_code = "p_up"

    reason = (
        f"{reason_code}={success_prob * 100.0:.1f}% "
        f"baseline={baseline_prob * 100.0:.1f}% "
        f"edge={edge_pct_points:.1f}pp "
        f"event={prob.event_return_pct:.1f}%/{cfg.event_lookback_days}d "
        f"down_streak={prob.down_streak} "
        f"analogs={prob.analog_count}"
    )
    return TradeIntent(fundamental.symbol, direction, round(combined, 4), reason)


def _evaluate(
    bars: list[Bar],
    fundamental: FundamentalRow,
    cfg: IntentConfig,
    short: bool,
) -> IntentEvaluation:
    prob = _estimate_probability(bars, cfg)
    if prob is None:
        return IntentEvaluation(
            None,
            "rejected",
            "insufficient_probability_history",
            "Not enough point-in-time daily history was available to estimate a next-day probability.",
        )
    if prob.baseline_count == 0:
        return IntentEvaluation(
            None,
            "rejected",
            "recent_drop_event_missing",
            (
                f"Recent event return {prob.event_return_pct:.2f}% and down streak {prob.down_streak} "
                f"did not meet drop filters: <= -{cfg.min_event_drop_pct:.2f}% over "
                f"{cfg.event_lookback_days}d and at least {cfg.min_consecutive_down_days} consecutive down days."
            ),
        )
    if prob.analog_count < cfg.min_analog_events:
        return IntentEvaluation(
            None,
            "rejected",
            "too_few_similar_events",
            f"Only {prob.analog_count} similar historical events were available; minimum is {cfg.min_analog_events}.",
        )

    success_prob = prob.p_down if short else prob.p_up
    baseline_prob = prob.baseline_down if short else prob.baseline_up
    min_probability = cfg.short_min_probability if short else cfg.long_min_probability
    edge_pct_points = (success_prob - baseline_prob) * 100.0
    direction_name = "down" if short else "up"

    if success_prob < min_probability:
        return IntentEvaluation(
            None,
            "rejected",
            f"{direction_name}_probability_below_min",
            (
                f"Estimated P(next-day {direction_name}) {success_prob * 100.0:.2f}% "
                f"is below minimum {min_probability * 100.0:.2f}%."
            ),
        )
    if edge_pct_points < cfg.min_edge_pct_points:
        return IntentEvaluation(
            None,
            "rejected",
            "probability_edge_below_min",
            (
                f"Estimated edge {edge_pct_points:.2f} percentage points over same-symbol baseline "
                f"is below minimum {cfg.min_edge_pct_points:.2f}."
            ),
        )

    intent = _build_intent(fundamental, cfg, prob, short)
    reason_code = "statistical_short_probability_passed" if short else "statistical_long_probability_passed"
    return IntentEvaluation(
        intent,
        "intent",
        reason_code,
        intent.reason,
    )


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    return _evaluate(bars, fundamental, cfg, short=False)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    return _evaluate(bars, fundamental, cfg, short=True)
