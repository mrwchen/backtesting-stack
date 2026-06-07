"""Technical setup candidates for the scalping decision layer.

The decision model is a filter, not the signal generator. This layer creates
causal bar-close setups first; the classifier then decides whether a setup is
worth trading at the next bar open.
"""

from dataclasses import dataclass

import numpy as np

from . import config

LONG_REGIME0_PULLBACK_RECLAIM = "LONG_REGIME0_PULLBACK_RECLAIM"
LONG_REGIME0_TREND_CONTINUATION = "LONG_REGIME0_TREND_CONTINUATION"
SHORT_REGIME0_PULLBACK_FADE = "SHORT_REGIME0_PULLBACK_FADE"
SHORT_REGIME0_LEVEL_REJECT = "SHORT_REGIME0_LEVEL_REJECT"
SHORT_REGIME0_OPENING_RANGE_REJECT = "SHORT_REGIME0_OPENING_RANGE_REJECT"
SHORT_REGIME0_MOMENTUM_ROLLOVER = "SHORT_REGIME0_MOMENTUM_ROLLOVER"
SHORT_REGIME1_BOUNCE_REJECT = "SHORT_REGIME1_BOUNCE_REJECT"
SHORT_REGIME1_WEAK_BOUNCE_REJECT = "SHORT_REGIME1_WEAK_BOUNCE_REJECT"
SHORT_REGIME1_STRONG_BOUNCE_REJECT = "SHORT_REGIME1_STRONG_BOUNCE_REJECT"
SHORT_REGIME1_FAILED_RECLAIM = "SHORT_REGIME1_FAILED_RECLAIM"
SHORT_REGIME1_OPENING_RANGE_REJECT = "SHORT_REGIME1_OPENING_RANGE_REJECT"
SHORT_REGIME1_LEVEL_REJECT = "SHORT_REGIME1_LEVEL_REJECT"
SHORT_REGIME1_CONTINUATION = "SHORT_REGIME1_CONTINUATION"

SETUP_IDS = (
    LONG_REGIME0_PULLBACK_RECLAIM,
    LONG_REGIME0_TREND_CONTINUATION,
    SHORT_REGIME0_PULLBACK_FADE,
    SHORT_REGIME0_LEVEL_REJECT,
    SHORT_REGIME0_OPENING_RANGE_REJECT,
    SHORT_REGIME0_MOMENTUM_ROLLOVER,
    SHORT_REGIME1_BOUNCE_REJECT,
    SHORT_REGIME1_WEAK_BOUNCE_REJECT,
    SHORT_REGIME1_STRONG_BOUNCE_REJECT,
    SHORT_REGIME1_FAILED_RECLAIM,
    SHORT_REGIME1_OPENING_RANGE_REJECT,
    SHORT_REGIME1_LEVEL_REJECT,
    SHORT_REGIME1_CONTINUATION,
)
SETUP_FEATURE_COLUMNS = tuple(f"setup_{setup_id.lower()}" for setup_id in SETUP_IDS)
SETUP_QUALITY_FEATURE_COLUMNS = (
    "setup_score",
    "setup_bounce_quality",
    "setup_reject_quality",
    "setup_close_weakness",
    "setup_downtrend_quality",
    "setup_rsi_quality",
    "setup_momentum_quality",
    "setup_time_quality",
)
_SETUP_INDEX = {setup_id: idx for idx, setup_id in enumerate(SETUP_IDS)}


@dataclass(frozen=True)
class SetupCandidate:
    setup_id: str
    direction: str
    score: float
    quality_features: tuple[float, ...]


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _safe_basis(cur_price: float, sigma_ret: float, atr_pts: float) -> float:
    sigma_pts = sigma_ret * cur_price if np.isfinite(sigma_ret) and sigma_ret > 0.0 else 0.0
    atr_value = atr_pts if np.isfinite(atr_pts) and atr_pts > 0.0 else 0.0
    return max(sigma_pts, atr_value, 1e-6)


def _body_position(cur_high: float, cur_low: float, cur_close: float) -> float:
    span = max(cur_high - cur_low, 1e-9)
    return _clip01((cur_close - cur_low) / span)


def _enabled(setup_id: str) -> bool:
    return config.CANDIDATE_SETUP_ENABLED.get(setup_id, True)


def _score_allowed(candidate: SetupCandidate, minutes_from_open: float) -> bool:
    gate = config.CANDIDATE_SETUP_SCORE_GATES.get(candidate.setup_id)
    if gate is None:
        return candidate.score >= config.MIN_CANDIDATE_SCORE
    if candidate.score < gate.min_score:
        return False
    if gate.max_score is not None and candidate.score >= gate.max_score:
        return False
    if gate.entry_minute_ranges:
        if not np.isfinite(minutes_from_open):
            return False
        if not any(start <= minutes_from_open < end for start, end in gate.entry_minute_ranges):
            return False
    return True


def setup_feature_vector(setup_id: str) -> np.ndarray:
    vector = np.zeros(len(SETUP_IDS), dtype=np.float64)
    setup_idx = _SETUP_INDEX.get(setup_id)
    if setup_idx is not None:
        vector[setup_idx] = 1.0
    return vector


def candidate_feature_row(base_features: np.ndarray, candidate: SetupCandidate) -> np.ndarray:
    quality = np.asarray((candidate.score, *candidate.quality_features), dtype=np.float64)
    if quality.shape[0] != len(SETUP_QUALITY_FEATURE_COLUMNS):
        raise ValueError(f"Setup candidate {candidate.setup_id} has {quality.shape[0]} quality features")
    return np.concatenate([
        base_features.astype(np.float64, copy=False),
        setup_feature_vector(candidate.setup_id),
        quality,
    ])


def _candidate(
    setup_id: str,
    score: float,
    bounce: float,
    reject: float,
    close_weakness: float,
    trend: float,
    rsi_quality: float,
    momentum_quality: float,
    time_quality: float,
) -> SetupCandidate:
    return SetupCandidate(
        setup_id=setup_id,
        direction="SHORT",
        score=float(score),
        quality_features=(
            float(bounce),
            float(reject),
            float(close_weakness),
            float(trend),
            float(rsi_quality),
            float(momentum_quality),
            float(time_quality),
        ),
    )


def build_setup_candidates(
    idx: int,
    states: np.ndarray,
    high_vol: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    levels: np.ndarray,
    slopes: np.ndarray,
    sigma_ret: np.ndarray,
    atr: np.ndarray,
    momentum: np.ndarray,
    rsi: np.ndarray,
    opening_high: np.ndarray,
    minutes_from_entry_start: np.ndarray,
) -> list[SetupCandidate]:
    if idx <= 0:
        return []
    if bool(high_vol[idx]) and not config.CANDIDATE_ALLOW_HIGH_VOL:
        return []

    state = int(states[idx])
    cur_close = float(close[idx])
    cur_high = float(high[idx])
    cur_low = float(low[idx])
    level = float(levels[idx])
    prev_level = float(levels[idx - 1])
    basis = _safe_basis(cur_close, float(sigma_ret[idx]), float(atr[idx]))
    prev_basis = _safe_basis(float(close[idx - 1]), float(sigma_ret[idx - 1]), float(atr[idx - 1]))
    dev = (cur_close - level) / basis
    prev_dev = (float(close[idx - 1]) - prev_level) / prev_basis
    high_dev = (cur_high - level) / basis
    low_dev = (cur_low - level) / basis
    slope = float(slopes[idx]) / basis
    mom = float(momentum[idx])
    rsi_value = float(rsi[idx])
    body_pos = _body_position(cur_high, cur_low, cur_close)
    minutes_from_open = float(minutes_from_entry_start[idx])
    opening_high_value = float(opening_high[idx]) if np.isfinite(opening_high[idx]) else np.nan
    bounce = _clip01((high_dev + 0.10) / 1.25)
    reject = _clip01((max(prev_dev, high_dev) - dev + 0.05) / 1.00)
    close_weakness = _clip01((0.62 - body_pos) / 0.62)
    trend_ok = _clip01((0.20 - slope) / 0.65)
    rsi_short_quality = _clip01((rsi_value - 30.0) / 48.0)
    momentum_down = _clip01((0.00025 - mom) / 0.0015)
    time_quality = _clip01((210.0 - minutes_from_open) / 210.0)
    candidates: list[SetupCandidate] = []

    if state == 0:
        if (
            _enabled(LONG_REGIME0_PULLBACK_RECLAIM)
            and 34.0 <= rsi_value <= 68.0
            and slope >= -0.20
            and low_dev <= 0.20
            and dev >= -0.55
            and (dev > prev_dev or body_pos >= 0.55)
        ):
            pullback = _clip01((-low_dev + 0.25) / 1.25)
            reclaim = _clip01((dev - min(prev_dev, low_dev) + 0.05) / 0.95)
            close_strength = _clip01((body_pos - 0.42) / 0.48)
            trend_ok = _clip01((slope + 0.20) / 0.65)
            rsi_room = _clip01((68.0 - rsi_value) / 34.0)
            score = 0.30 * pullback + 0.25 * reclaim + 0.20 * close_strength + 0.15 * trend_ok + 0.10 * rsi_room
            candidates.append(SetupCandidate(
                LONG_REGIME0_PULLBACK_RECLAIM,
                "LONG",
                score,
                (pullback, reclaim, close_strength, trend_ok, rsi_room, _clip01(mom / 0.0015), time_quality),
            ))

        if (
            _enabled(LONG_REGIME0_TREND_CONTINUATION)
            and 42.0 <= rsi_value <= 70.0
            and -0.25 <= dev <= 1.30
            and slope >= -0.05
            and mom >= -0.0002
            and (cur_close >= float(close[idx - 1]) or dev >= prev_dev)
            and body_pos >= 0.45
        ):
            trend = _clip01((slope + 0.05) / 0.55)
            continuation = _clip01((dev + 0.25) / 1.55)
            close_strength = _clip01((body_pos - 0.45) / 0.45)
            rsi_room = _clip01((70.0 - rsi_value) / 28.0)
            score = 0.35 * trend + 0.25 * continuation + 0.25 * close_strength + 0.15 * rsi_room
            candidates.append(SetupCandidate(
                LONG_REGIME0_TREND_CONTINUATION,
                "LONG",
                score,
                (continuation, trend, close_strength, trend, rsi_room, _clip01(mom / 0.0015), time_quality),
            ))

        if (
            _enabled(SHORT_REGIME0_PULLBACK_FADE)
            and 42.0 <= rsi_value <= 78.0
            and high_dev >= 0.25
            and dev <= 1.10
            and high_dev - dev >= 0.10
            and slope <= 0.35
            and body_pos <= 0.60
        ):
            fade_extension = _clip01((high_dev - 0.15) / 1.35)
            fade_reject = _clip01((high_dev - dev) / 1.05)
            rsi_pressure = _clip01((rsi_value - 42.0) / 36.0)
            score = (
                0.27 * fade_extension
                + 0.27 * fade_reject
                + 0.20 * close_weakness
                + 0.11 * trend_ok
                + 0.09 * rsi_pressure
                + 0.06 * momentum_down
            )
            candidates.append(_candidate(
                SHORT_REGIME0_PULLBACK_FADE,
                score,
                fade_extension,
                fade_reject,
                close_weakness,
                trend_ok,
                rsi_pressure,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME0_LEVEL_REJECT)
            and 38.0 <= rsi_value <= 76.0
            and high_dev >= -0.02
            and dev <= 0.38
            and high_dev - dev >= 0.06
            and body_pos <= 0.62
            and slope <= 0.32
        ):
            level_test = _clip01((high_dev + 0.02) / 0.82)
            level_reject = _clip01((high_dev - dev) / 0.82)
            rsi_pressure = _clip01((rsi_value - 38.0) / 38.0)
            score = (
                0.27 * level_test
                + 0.29 * level_reject
                + 0.18 * close_weakness
                + 0.11 * trend_ok
                + 0.08 * momentum_down
                + 0.07 * rsi_pressure
            )
            candidates.append(_candidate(
                SHORT_REGIME0_LEVEL_REJECT,
                score,
                level_test,
                level_reject,
                close_weakness,
                trend_ok,
                rsi_pressure,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME0_OPENING_RANGE_REJECT)
            and np.isfinite(opening_high_value)
            and 12.0 <= minutes_from_open <= 240.0
            and cur_high >= opening_high_value - 0.22 * basis
            and cur_close <= opening_high_value + 0.16 * basis
            and body_pos <= 0.64
            and slope <= 0.38
        ):
            range_retest = _clip01((cur_high - opening_high_value + 0.28 * basis) / max(1e-6, 1.35 * basis))
            range_reject = _clip01((cur_high - cur_close) / max(1e-6, 1.15 * basis))
            rsi_pressure = _clip01((rsi_value - 36.0) / 42.0)
            score = (
                0.27 * range_retest
                + 0.27 * range_reject
                + 0.18 * close_weakness
                + 0.11 * trend_ok
                + 0.09 * momentum_down
                + 0.08 * time_quality
            )
            candidates.append(_candidate(
                SHORT_REGIME0_OPENING_RANGE_REJECT,
                score,
                range_retest,
                range_reject,
                close_weakness,
                trend_ok,
                rsi_pressure,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME0_MOMENTUM_ROLLOVER)
            and 40.0 <= rsi_value <= 74.0
            and -0.20 <= dev <= 1.20
            and slope <= 0.30
            and mom <= 0.00015
            and cur_close < float(close[idx - 1])
            and body_pos <= 0.50
        ):
            rollover_location = _clip01((dev + 0.20) / 1.40)
            rollover_close = _clip01((0.50 - body_pos) / 0.50)
            rsi_pressure = _clip01((rsi_value - 40.0) / 34.0)
            score = (
                0.23 * rollover_location
                + 0.27 * rollover_close
                + 0.18 * close_weakness
                + 0.12 * trend_ok
                + 0.12 * momentum_down
                + 0.08 * rsi_pressure
            )
            candidates.append(_candidate(
                SHORT_REGIME0_MOMENTUM_ROLLOVER,
                score,
                rollover_location,
                rollover_close,
                close_weakness,
                trend_ok,
                rsi_pressure,
                momentum_down,
                time_quality,
            ))

    if state == 1:
        if (
            _enabled(SHORT_REGIME1_BOUNCE_REJECT)
            and 32.0 <= rsi_value <= 76.0
            and high_dev >= -0.05
            and dev <= 0.55
            and slope <= 0.20
            and (dev < prev_dev or body_pos <= 0.55)
        ):
            score = (
                0.24 * bounce
                + 0.24 * reject
                + 0.20 * close_weakness
                + 0.14 * trend_ok
                + 0.10 * rsi_short_quality
                + 0.08 * momentum_down
            )
            candidates.append(_candidate(
                SHORT_REGIME1_BOUNCE_REJECT,
                score,
                bounce,
                reject,
                close_weakness,
                trend_ok,
                rsi_short_quality,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_WEAK_BOUNCE_REJECT)
            and 30.0 <= rsi_value <= 70.0
            and -0.10 <= high_dev <= 0.45
            and dev <= 0.25
            and slope <= 0.15
            and body_pos <= 0.60
            and (dev < prev_dev or cur_close < float(close[idx - 1]))
        ):
            weak_bounce = _clip01((high_dev + 0.10) / 0.55)
            weak_reject = _clip01((max(prev_dev, high_dev) - dev + 0.03) / 0.70)
            score = (
                0.22 * weak_bounce
                + 0.28 * weak_reject
                + 0.22 * close_weakness
                + 0.13 * trend_ok
                + 0.10 * momentum_down
                + 0.05 * time_quality
            )
            candidates.append(_candidate(
                SHORT_REGIME1_WEAK_BOUNCE_REJECT,
                score,
                weak_bounce,
                weak_reject,
                close_weakness,
                trend_ok,
                rsi_short_quality,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_STRONG_BOUNCE_REJECT)
            and 38.0 <= rsi_value <= 80.0
            and high_dev >= 0.25
            and dev <= 0.65
            and high_dev - dev >= 0.18
            and slope <= 0.25
            and body_pos <= 0.55
        ):
            strong_bounce = _clip01((high_dev - 0.15) / 1.35)
            strong_reject = _clip01((high_dev - dev) / 1.05)
            rsi_pressure = _clip01((rsi_value - 38.0) / 42.0)
            score = (
                0.26 * strong_bounce
                + 0.28 * strong_reject
                + 0.18 * close_weakness
                + 0.12 * trend_ok
                + 0.10 * rsi_pressure
                + 0.06 * momentum_down
            )
            candidates.append(_candidate(
                SHORT_REGIME1_STRONG_BOUNCE_REJECT,
                score,
                strong_bounce,
                strong_reject,
                close_weakness,
                trend_ok,
                rsi_pressure,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_FAILED_RECLAIM)
            and 30.0 <= rsi_value <= 76.0
            and high_dev >= 0.00
            and dev <= 0.05
            and prev_dev <= 0.25
            and body_pos <= 0.55
            and slope <= 0.18
        ):
            reclaim_attempt = _clip01((high_dev + 0.05) / 0.90)
            failed_close = _clip01((0.08 - dev) / 0.85)
            score = (
                0.26 * reclaim_attempt
                + 0.28 * failed_close
                + 0.18 * close_weakness
                + 0.14 * trend_ok
                + 0.08 * momentum_down
                + 0.06 * rsi_short_quality
            )
            candidates.append(_candidate(
                SHORT_REGIME1_FAILED_RECLAIM,
                score,
                reclaim_attempt,
                failed_close,
                close_weakness,
                trend_ok,
                rsi_short_quality,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_OPENING_RANGE_REJECT)
            and np.isfinite(opening_high_value)
            and 15.0 <= minutes_from_open <= 180.0
            and cur_high >= opening_high_value - 0.15 * basis
            and cur_close <= opening_high_value + 0.05 * basis
            and body_pos <= 0.58
            and slope <= 0.25
        ):
            range_retest = _clip01((cur_high - opening_high_value + 0.25 * basis) / max(1e-6, 1.25 * basis))
            range_reject = _clip01((cur_high - cur_close) / max(1e-6, 1.10 * basis))
            score = (
                0.28 * range_retest
                + 0.28 * range_reject
                + 0.18 * close_weakness
                + 0.12 * trend_ok
                + 0.08 * momentum_down
                + 0.06 * time_quality
            )
            candidates.append(_candidate(
                SHORT_REGIME1_OPENING_RANGE_REJECT,
                score,
                range_retest,
                range_reject,
                close_weakness,
                trend_ok,
                rsi_short_quality,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_LEVEL_REJECT)
            and 30.0 <= rsi_value <= 76.0
            and high_dev >= -0.08
            and dev <= 0.18
            and high_dev - dev >= 0.08
            and body_pos <= 0.57
            and slope <= 0.18
        ):
            level_test = _clip01((high_dev + 0.08) / 0.88)
            level_reject = _clip01((high_dev - dev) / 0.85)
            score = (
                0.27 * level_test
                + 0.29 * level_reject
                + 0.18 * close_weakness
                + 0.14 * trend_ok
                + 0.07 * momentum_down
                + 0.05 * rsi_short_quality
            )
            candidates.append(_candidate(
                SHORT_REGIME1_LEVEL_REJECT,
                score,
                level_test,
                level_reject,
                close_weakness,
                trend_ok,
                rsi_short_quality,
                momentum_down,
                time_quality,
            ))

        if (
            _enabled(SHORT_REGIME1_CONTINUATION)
            and 24.0 <= rsi_value <= 62.0
            and -1.80 <= dev <= 0.35
            and slope <= 0.05
            and mom <= 0.0002
            and body_pos <= 0.62
        ):
            trend = _clip01((0.05 - slope) / 0.55)
            continuation = _clip01((0.35 - dev) / 2.15)
            close_weakness = _clip01((0.62 - body_pos) / 0.62)
            rsi_room = _clip01((rsi_value - 24.0) / 38.0)
            score = 0.35 * trend + 0.25 * continuation + 0.25 * close_weakness + 0.15 * rsi_room
            candidates.append(_candidate(
                SHORT_REGIME1_CONTINUATION,
                score,
                continuation,
                trend,
                close_weakness,
                trend,
                rsi_room,
                momentum_down,
                time_quality,
            ))

    candidates = [c for c in candidates if _score_allowed(c, minutes_from_open)]
    candidates.sort(key=lambda c: (c.score, c.direction == "SHORT"), reverse=True)
    return candidates[: config.MAX_CANDIDATES_PER_BAR]
