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
SHORT_REGIME1_BOUNCE_REJECT = "SHORT_REGIME1_BOUNCE_REJECT"
SHORT_REGIME1_CONTINUATION = "SHORT_REGIME1_CONTINUATION"

SETUP_IDS = (
    LONG_REGIME0_PULLBACK_RECLAIM,
    LONG_REGIME0_TREND_CONTINUATION,
    SHORT_REGIME1_BOUNCE_REJECT,
    SHORT_REGIME1_CONTINUATION,
)
SETUP_FEATURE_COLUMNS = tuple(f"setup_{setup_id.lower()}" for setup_id in SETUP_IDS)
_SETUP_INDEX = {setup_id: idx for idx, setup_id in enumerate(SETUP_IDS)}


@dataclass(frozen=True)
class SetupCandidate:
    setup_id: str
    direction: str
    score: float


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


def setup_feature_vector(setup_id: str) -> np.ndarray:
    vector = np.zeros(len(SETUP_IDS), dtype=np.float64)
    setup_idx = _SETUP_INDEX.get(setup_id)
    if setup_idx is not None:
        vector[setup_idx] = 1.0
    return vector


def candidate_feature_row(base_features: np.ndarray, setup_id: str) -> np.ndarray:
    return np.concatenate([base_features.astype(np.float64, copy=False), setup_feature_vector(setup_id)])


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
            candidates.append(SetupCandidate(LONG_REGIME0_PULLBACK_RECLAIM, "LONG", score))

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
            candidates.append(SetupCandidate(LONG_REGIME0_TREND_CONTINUATION, "LONG", score))

    if state == 1:
        if (
            _enabled(SHORT_REGIME1_BOUNCE_REJECT)
            and 32.0 <= rsi_value <= 76.0
            and high_dev >= -0.05
            and dev <= 0.55
            and slope <= 0.20
            and (dev < prev_dev or body_pos <= 0.55)
        ):
            bounce = _clip01((high_dev + 0.10) / 1.25)
            reject = _clip01((max(prev_dev, high_dev) - dev + 0.05) / 1.00)
            close_weakness = _clip01((0.58 - body_pos) / 0.58)
            trend_ok = _clip01((0.20 - slope) / 0.65)
            rsi_room = _clip01((rsi_value - 32.0) / 44.0)
            score = 0.30 * bounce + 0.25 * reject + 0.20 * close_weakness + 0.15 * trend_ok + 0.10 * rsi_room
            candidates.append(SetupCandidate(SHORT_REGIME1_BOUNCE_REJECT, "SHORT", score))

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
            candidates.append(SetupCandidate(SHORT_REGIME1_CONTINUATION, "SHORT", score))

    candidates = [c for c in candidates if c.score >= config.MIN_CANDIDATE_SCORE]
    candidates.sort(key=lambda c: (c.score, c.direction == "SHORT"), reverse=True)
    return candidates[: config.MAX_CANDIDATES_PER_BAR]
