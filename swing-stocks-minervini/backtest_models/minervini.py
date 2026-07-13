"""Daily, causal Minervini-style setup recognition.

This module owns chart-pattern semantics.  Eligibility, fundamental ranking,
portfolio constraints and execution remain model-independent runner concerns.
The detector intentionally uses no breakout-day close or volume confirmation:
it emits a pre-session setup whose pivot can be used by a daily stop-buy model.

The pattern rules favour robust shape properties over exact textbook geometry:

* every setup requires a material prior advance;
* a VCP may contain one modestly wider contraction, but must finish tight;
* an ambiguous outside bar is ignored rather than erasing prior structure;
* bases may last up to roughly six months;
* lower dry-up ratios always receive a higher score; and
* the most recent bars must form a genuinely tight area below the pivot.

All observations used for a setup detected on D are known at the close of D.
Swing extrema are only added after ``swing_window`` later sessions confirm them.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd


SetupType = Literal["vcp", "flat_base", "power_play", "tight_shelf"]
Swing = tuple[int, float, Literal["H", "L"]]
MODEL_VERSION = "minervini_daily_v7"

# Classification is score-first.  The order is only a deterministic tie
# breaker for nested patterns with identical quality: a contraction structure
# carries more shape information than a short range, while a flat base is the
# broadest classification.
_CLASSIFICATION_SPECIFICITY: dict[SetupType, int] = {
    "vcp": 4,
    "power_play": 3,
    "tight_shelf": 2,
    "flat_base": 1,
}


@dataclass(frozen=True)
class Setup:
    """One chart setup available for an order on the following session."""

    symbol: str
    setup_type: SetupType
    price_continuity_segment: int
    detect_date: date
    pivot: float
    last_low: float
    stop_level: float
    base_start_date: date
    base_days: int
    n_contractions: int
    contraction_depths: tuple[float, ...]
    base_count: int
    dryup_ratio: float
    setup_score: float
    prior_advance_pct: float
    final_tightness_pct: float
    structure_quality_score: float
    volume_dryup_score: float
    tightness_score: float
    pivot_proximity_score: float
    prior_advance_score: float
    close: float
    valid_until: date


@dataclass(frozen=True)
class GoldCase:
    """Human-review control, never an optimisation target.

    Positive cases are well-known historical leaders.  Negative cases are
    failed entries from the frozen 2020-2023 market-filter development run.
    A control says "inspect this chart"; it does not assert that the detector
    must or must not emit a setup on one exact day.
    """

    symbol: str
    role: Literal["positive", "negative"]
    reference_date: date
    rationale: str
    source: str
    lookback_days: int = 90
    forward_days: int = 5


GOLD_CASES: tuple[GoldCase, ...] = (
    GoldCase("NVDA", "positive", date(2023, 5, 25), "earnings gap and leadership continuation", "historical leader review"),
    GoldCase("SMCI", "positive", date(2023, 5, 24), "powerful advance with continuation bases", "historical leader review"),
    GoldCase("CELH", "positive", date(2021, 5, 10), "multi-stage leader with repeated tight shelves", "historical leader review"),
    GoldCase("ELF", "positive", date(2023, 2, 2), "persistent leader with compact continuation structures", "historical leader review"),
    GoldCase("CROX", "positive", date(2021, 4, 30), "strong trend and constructive bases", "historical leader review"),
    GoldCase("MOD", "positive", date(2023, 5, 25), "strong advance followed by continuation structure", "historical leader review"),
    GoldCase("AXON", "positive", date(2023, 3, 1), "leadership trend with tight consolidations", "historical leader review"),
    GoldCase("JEF", "negative", date(2021, 11, 2), "pivot entry stopped within six sessions", "dev_market_ablation_v1 run 14"),
    GoldCase("IR", "negative", date(2023, 5, 18), "pivot entry stopped within four sessions", "dev_market_ablation_v1 run 14"),
    GoldCase("ATI", "negative", date(2023, 7, 27), "pivot entry stopped within four sessions", "dev_market_ablation_v1 run 14"),
    GoldCase("MTRN", "negative", date(2023, 4, 18), "pivot entry stopped within three sessions", "dev_market_ablation_v1 run 14"),
    GoldCase("REGN", "negative", date(2020, 7, 20), "pivot entry stopped within four sessions", "dev_market_ablation_v1 run 14"),
    GoldCase("LSCC", "negative", date(2023, 7, 18), "pivot entry stopped within two sessions", "dev_market_ablation_v1 run 14"),
)


@dataclass(frozen=True)
class _Rules:
    swing_window: int
    base_min_days: int
    base_max_days: int
    contractions_min: int
    contractions_max: int
    base_depth_max: float
    final_depth_max: float
    pivot_below_high_max: float
    setup_valid_days: int
    stop_max_pct: float
    prior_advance_min: float
    dryup_score_zero: float
    widening_relative_tolerance: float = 0.25
    widening_absolute_tolerance: float = 0.025
    max_widening_steps: int = 1
    final_tight_days: int = 5
    final_tight_max: float = 0.08

    @classmethod
    def from_config(cls, cfg: Any) -> "_Rules":
        return cls(
            swing_window=int(cfg.swing_window),
            base_min_days=int(cfg.base_min_days),
            base_max_days=int(cfg.base_max_days),
            contractions_min=int(cfg.contractions_min),
            contractions_max=int(cfg.contractions_max),
            base_depth_max=float(cfg.base_depth_max),
            final_depth_max=float(cfg.final_depth_max),
            pivot_below_high_max=float(cfg.pivot_below_base_high_max),
            setup_valid_days=int(cfg.setup_valid_days),
            stop_max_pct=float(cfg.stop_max_pct),
            prior_advance_min=float(cfg.prior_advance_min),
            dryup_score_zero=float(cfg.dryup_score_zero_ratio),
        )


def _candidate_at(high: np.ndarray, low: np.ndarray, i: int, k: int) -> Swing | None:
    if i < k or i + k >= len(high):
        return None
    high_window = high[i - k : i + k + 1]
    low_window = low[i - k : i + k + 1]
    if not np.isfinite(high_window).all() or not np.isfinite(low_window).all():
        return None
    high_i = float(high[i])
    low_i = float(low[i])
    is_high = high_i > float(np.max(high[i - k : i])) and high_i >= float(np.max(high[i + 1 : i + k + 1]))
    is_low = low_i < float(np.min(low[i - k : i])) and low_i <= float(np.min(low[i + 1 : i + k + 1]))
    # Daily data cannot order the two extremes.  Ignoring the ambiguous bar
    # keeps us causal without destroying months of already confirmed structure.
    if is_high == is_low:
        return None
    return (i, high_i, "H") if is_high else (i, low_i, "L")


def _apply_candidate(swings: list[Swing], event: Swing) -> None:
    if not swings or swings[-1][2] != event[2]:
        swings.append(event)
        return
    previous = swings[-1]
    if (event[2] == "H" and event[1] > previous[1]) or (event[2] == "L" and event[1] < previous[1]):
        swings[-1] = event


def find_swings(high: np.ndarray, low: np.ndarray, k: int) -> list[Swing]:
    """Return alternating, causally confirmed swings for a complete series."""
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    if len(high) != len(low):
        raise ValueError("high and low must have equal length")
    if k < 1:
        raise ValueError("swing window must be >= 1")
    swings: list[Swing] = []
    for i in range(k, len(high) - k):
        event = _candidate_at(high, low, i, k)
        if event is not None:
            _apply_candidate(swings, event)
    return swings


def _prior_advance(
    high: np.ndarray,
    low: np.ndarray,
    base_start: int,
    segment_start: int,
    lookback: int = 126,
) -> float:
    start = max(segment_start, base_start - lookback)
    history = low[start : base_start + 1]
    if len(history) < 10 or not np.isfinite(history).all():
        return 0.0
    prior_low = float(np.min(history))
    return float(high[base_start] / prior_low - 1.0) if prior_low > 0 else 0.0


def _power_play_thrust_volume_ratio(
    volume: np.ndarray,
    base_start: int,
    segment_start: int,
    lookback: int = 40,
) -> float | None:
    """Return causal volume expansion in the thrust immediately before a base.

    A Power Play is an exceptional price-and-demand event, not merely a short
    base after an ordinary uptrend.  The comparison uses only the sessions at
    or before ``base_start``.  The median of the same fixed thrust window is a
    robust local baseline; the mean of its three largest prints represents the
    institutional demand burst without letting a single bad tick qualify it.
    """
    start = max(segment_start, base_start - lookback + 1)
    observed = volume[start : base_start + 1]
    valid = observed[np.isfinite(observed) & (observed > 0)]
    if len(valid) < 20:
        return None
    baseline = float(np.median(valid))
    if baseline <= 0:
        return None
    leaders = np.partition(valid, -min(3, len(valid)))[-min(3, len(valid)) :]
    return float(np.mean(leaders) / baseline)


def _power_play_has_leadership_structure(
    t: int,
    pivot: float,
    high: np.ndarray,
    close: np.ndarray,
    segment_start: int,
) -> bool:
    """Return whether a Power Play is a causal Stage-2 continuation.

    Doubling from a crash low is not leadership.  The pivot must still be near
    the highest price seen in the trailing year and price must be above its
    medium-term trend.  Mature histories additionally have to satisfy the
    classic 50/150/200-session moving-average order; the 200-session average
    must be rising once enough prior observations exist to measure that.

    Young issues are intentionally allowed after 50 sessions.  In that case
    only the moving averages that can be computed from the current continuity
    segment are applied.  Every slice ends at ``t`` (or earlier), so the check
    cannot see breakout-day or future observations.
    """
    available = t - segment_start + 1
    if available < 50 or pivot <= 0:
        return False

    history_start = max(segment_start, t - 251)
    trailing_highs = high[history_start : t + 1]
    if not np.isfinite(trailing_highs).all():
        return False
    trailing_high = float(np.max(trailing_highs))
    if trailing_high <= 0 or pivot < 0.85 * trailing_high:
        return False

    sma50 = float(np.mean(close[t - 49 : t + 1]))
    if not np.isfinite(sma50) or float(close[t]) <= sma50:
        return False

    sma150: float | None = None
    if available >= 150:
        sma150 = float(np.mean(close[t - 149 : t + 1]))
        if not np.isfinite(sma150) or sma50 <= sma150:
            return False

    if available >= 200:
        assert sma150 is not None
        sma200 = float(np.mean(close[t - 199 : t + 1]))
        if (
            not np.isfinite(sma200)
            or float(close[t]) <= sma200
            or sma150 <= sma200
        ):
            return False
        if available >= 220:
            prior_sma200 = float(np.mean(close[t - 219 : t - 19]))
            if not np.isfinite(prior_sma200) or sma200 <= prior_sma200:
                return False

    return True


def _close_dispersion(
    close: np.ndarray,
    start: int,
    end: int,
    pivot: float,
) -> float:
    values = close[start:end]
    if pivot <= 0 or len(values) == 0 or not np.isfinite(values).all():
        return float("inf")
    return float((np.max(values) - np.min(values)) / pivot)


def _final_tight_area(
    t: int,
    pivot: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    rules: _Rules,
    *,
    max_tightness: float | None = None,
    require_high_below_pivot: bool = False,
) -> tuple[float, float] | None:
    start = t - rules.final_tight_days + 1
    if start < 0:
        return None
    h = high[start : t + 1]
    l = low[start : t + 1]
    c = close[start : t + 1]
    if not (np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(c).all()) or pivot <= 0:
        return None
    tightness = float((np.max(h) - np.min(l)) / pivot)
    proximity = float((pivot - close[t]) / pivot)
    tightness_limit = rules.final_tight_max if max_tightness is None else max_tightness
    if tightness > tightness_limit or proximity < 0 or proximity > rules.final_depth_max:
        return None
    # A range setup ceases to be pre-breakout as soon as the established
    # resistance trades. VCP performs the equivalent check from its last low.
    if require_high_below_pivot and float(np.max(h)) > pivot:
        return None
    if float(np.max(c)) > pivot:
        return None
    return tightness, proximity


def _volume_dryup(t: int, volume: np.ndarray, segment_start: int) -> float | None:
    recent_start = max(segment_start, t - 4)
    baseline_end = recent_start
    baseline_start = max(segment_start, baseline_end - 50)
    baseline = volume[baseline_start:baseline_end]
    recent = volume[recent_start : t + 1]
    if len(baseline) < 20 or len(recent) < 3:
        return None
    baseline_valid = baseline[np.isfinite(baseline) & (baseline > 0)]
    recent_valid = recent[np.isfinite(recent) & (recent > 0)]
    if (
        len(baseline_valid) < max(16, int(np.ceil(0.8 * len(baseline))))
        or len(recent_valid) < max(3, int(np.ceil(0.8 * len(recent))))
    ):
        return None
    return float(np.mean(recent_valid) / np.mean(baseline_valid))


def _score(
    *,
    structure_quality: float,
    final_tightness: float,
    proximity: float,
    dryup: float,
    prior_advance: float,
    rules: _Rules,
) -> tuple[float, float, float, float, float]:
    contraction_score = 25.0 * float(np.clip(structure_quality, 0.0, 1.0))
    tightness_score = 20.0 * float(np.clip(1.0 - final_tightness / rules.final_tight_max, 0.0, 1.0))
    # Strictly monotone: less recent volume can never reduce this component.
    volume_score = (
        15.0 * float(np.clip(1.0 - dryup / rules.dryup_score_zero, 0.0, 1.0))
        if np.isfinite(dryup) and dryup > 0
        else 0.0
    )
    proximity_score = 15.0 * float(np.clip(1.0 - proximity / rules.final_depth_max, 0.0, 1.0))
    prior_score = 25.0 * float(np.clip(prior_advance / 0.60, 0.0, 1.0))
    return (
        contraction_score + tightness_score + volume_score + proximity_score + prior_score,
        contraction_score,
        volume_score,
        tightness_score,
        proximity_score,
    )


def _vcp_candidate(
    t: int,
    swings: list[Swing],
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_positions: np.ndarray,
    segment_start: int,
    rules: _Rules,
) -> dict[str, Any] | None:
    if len(swings) < 4 or swings[-1][2] != "L":
        return None
    pairs: list[tuple[Swing, Swing]] = []
    j = len(swings) - 1
    while j >= 1 and len(pairs) < rules.contractions_max:
        lo, hi = swings[j], swings[j - 1]
        if lo[2] != "L" or hi[2] != "H":
            break
        pairs.append((hi, lo))
        j -= 2
    pairs.reverse()
    for count in range(min(len(pairs), rules.contractions_max), rules.contractions_min - 1, -1):
        selected = pairs[-count:]
        base_start = selected[0][0][0]
        base_days = int(session_positions[t] - session_positions[base_start])
        if not rules.base_min_days <= base_days <= rules.base_max_days:
            continue
        depths = np.asarray([(hi[1] - lo[1]) / hi[1] for hi, lo in selected], dtype=float)
        if (
            np.any(depths <= 0)
            or depths[0] > rules.base_depth_max
            or depths[-1] > min(rules.final_depth_max, 0.08)
        ):
            continue
        widen = depths[1:] - depths[:-1]
        allowed = np.maximum(rules.widening_absolute_tolerance, depths[:-1] * rules.widening_relative_tolerance)
        if int(np.sum(widen > allowed)) > rules.max_widening_steps:
            continue
        # The base must contract materially overall even when one intermediate
        # step is noisy. This is deliberately structural, not outcome-fitted.
        if depths[-1] >= depths[0] * 0.75:
            continue
        pivot = float(selected[-1][0][1])
        base_high = max(float(pair[0][1]) for pair in selected)
        if pivot < base_high * (1.0 - rules.pivot_below_high_max):
            continue
        last_low = float(selected[-1][1][1])
        last_low_idx = int(selected[-1][1][0])
        if not last_low < close[t] < pivot:
            continue
        later_low = low[last_low_idx + 1 : t + 1]
        if len(later_low) and (
            not np.isfinite(later_low).all()
            or float(np.min(later_low)) < last_low
        ):
            continue
        if float(np.max(high[last_low_idx : t + 1])) > pivot:
            continue
        prior = _prior_advance(high, low, base_start, segment_start)
        if prior < rules.prior_advance_min:
            continue
        final = _final_tight_area(t, pivot, high, low, close, rules)
        dryup = _volume_dryup(t, volume, segment_start)
        if final is None:
            continue
        if dryup is not None and dryup > 1.10:
            continue
        dryup = float("nan") if dryup is None else dryup
        tightness, proximity = final
        shrink = float(np.clip(1.0 - depths[-1] / depths[0], 0.0, 1.0))
        tolerated_steps = float(np.mean(widen <= allowed)) if len(widen) else 1.0
        structure_quality = 0.65 * shrink + 0.35 * tolerated_steps
        score = _score(
            structure_quality=structure_quality,
            final_tightness=tightness,
            proximity=proximity,
            dryup=dryup,
            prior_advance=prior,
            rules=rules,
        )
        return {
            "setup_type": "vcp",
            # The confirmed resistance bar is stable while this structure
            # matures and is shared by classifications that use the same
            # pivot.  Setup type is deliberately not part of the identity.
            "identity": (int(selected[-1][0][0]), round(pivot, 8)),
            "pivot_idx": int(selected[-1][0][0]),
            "structure_end": t,
            "base_start": base_start,
            "base_days": base_days,
            "pivot": pivot,
            "last_low": last_low,
            "depths": tuple(round(float(x), 4) for x in depths),
            "dryup": dryup,
            "prior": prior,
            "tightness": tightness,
            "score": score,
        }
    return None


def _range_candidate(
    setup_type: SetupType,
    t: int,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    session_positions: np.ndarray,
    segment_start: int,
    rules: _Rules,
) -> dict[str, Any] | None:
    if setup_type == "power_play":
        # A Power Play is the exceptional case: at least a 100% thrust in no
        # more than 40 sessions, followed by 10 to 30 elapsed sessions of rest.
        # ``length`` is the number of observations, while ``base_days`` is the
        # session distance from base start to detection, hence the +1 bounds.
        lengths, max_depth, required_advance = (
            range(11, 32),
            0.20,
            max(1.00, rules.prior_advance_min),
        )
    elif setup_type == "tight_shelf":
        lengths, max_depth, required_advance = (
            range(10, 22),
            0.06,
            rules.prior_advance_min,
        )
    elif setup_type == "flat_base":
        lengths, max_depth, required_advance = range(25, 71), 0.15, rules.prior_advance_min
    else:
        raise ValueError(f"unsupported range setup {setup_type!r}")

    # Prefer the longest qualifying structure so base_start remains stable as
    # the pattern matures. Duplicate suppression handles later overlapping days.
    for length in reversed(tuple(lengths)):
        start = t - length + 1
        if start < segment_start:
            continue
        h = high[start : t + 1]
        l = low[start : t + 1]
        if not (np.isfinite(h).all() and np.isfinite(l).all()):
            continue
        final_start = t - rules.final_tight_days + 1
        resistance = high[start:final_start]
        if len(resistance) < 3 or not np.isfinite(resistance).all():
            continue
        pivot = float(np.max(resistance))
        pivot_idx = start + int(np.argmax(resistance))
        last_low = float(np.min(l))
        depth = (pivot - last_low) / pivot if pivot > 0 else np.inf
        if depth > max_depth:
            continue
        if setup_type == "power_play" and not _power_play_has_leadership_structure(
            t,
            pivot,
            high,
            close,
            segment_start,
        ):
            continue
        prior = _prior_advance(
            high,
            low,
            start,
            segment_start,
            lookback=40 if setup_type == "power_play" else 126,
        )
        if prior < required_advance:
            continue
        thrust_volume_ratio: float | None = None
        if setup_type == "power_play":
            thrust_volume_ratio = _power_play_thrust_volume_ratio(
                volume, start, segment_start
            )
            if thrust_volume_ratio is None or thrust_volume_ratio < 1.50:
                continue
        if not last_low < close[t] < pivot:
            continue
        tightness_limit = (
            0.08
            if setup_type == "power_play"
            else 0.03 if setup_type == "tight_shelf" else None
        )
        final = _final_tight_area(
            t,
            pivot,
            high,
            low,
            close,
            rules,
            max_tightness=tightness_limit,
            require_high_below_pivot=True,
        )
        dryup = _volume_dryup(t, volume, segment_start)
        if final is None:
            continue
        dryup = float("nan") if dryup is None else dryup
        tightness, proximity = final
        final_close_dispersion = _close_dispersion(
            close, final_start, t + 1, pivot
        )
        if setup_type == "power_play":
            if final_close_dispersion > 0.04:
                continue
            if depth > 0.10:
                early_end = max(start + 5, final_start)
                early_high = high[start:early_end]
                early_low = low[start:early_end]
                if (
                    len(early_high) < 5
                    or not np.isfinite(early_high).all()
                    or not np.isfinite(early_low).all()
                ):
                    continue
                early_range = float(
                    (np.max(early_high) - np.min(early_low)) / pivot
                )
                if tightness > 0.80 * early_range:
                    continue
        elif setup_type == "tight_shelf":
            # Kept as a research label only.  Detection itself is strict so
            # its independent-mode evidence remains interpretable.
            if (
                final_close_dispersion > 0.03
                or not np.isfinite(dryup)
                or dryup > 1.00
                or float(np.mean(l[-rules.final_tight_days :]))
                + 1e-12
                < float(np.mean(l[: rules.final_tight_days]))
            ):
                continue
        structure_quality = float(np.clip(1.0 - depth / max_depth, 0.0, 1.0))
        if setup_type == "power_play":
            assert thrust_volume_ratio is not None
            structure_quality = (
                0.40 * structure_quality
                + 0.35 * float(np.clip(prior / 1.50, 0.0, 1.0))
                + 0.25
                * float(np.clip((thrust_volume_ratio - 1.0) / 1.5, 0.0, 1.0))
            )
        score = _score(
            structure_quality=structure_quality,
            final_tightness=tightness,
            proximity=proximity,
            dryup=dryup,
            prior_advance=prior,
            rules=rules,
        )
        return {
            "setup_type": setup_type,
            # Anchor identity to established resistance, not the sliding range
            # start or the chosen setup label.
            "identity": (pivot_idx, round(pivot, 8)),
            "pivot_idx": pivot_idx,
            "structure_end": t,
            "base_start": start,
            "base_days": int(session_positions[t] - session_positions[start]),
            "pivot": pivot,
            "last_low": last_low,
            "depths": (round(float(depth), 4),),
            "dryup": dryup,
            "prior": prior,
            "tightness": tightness,
            "score": score,
        }
    return None


def _same_structure(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two causal candidates describe one price structure.

    An exact resistance anchor is definitive.  Different detectors can choose
    neighbouring resistance bars, so near-equal pivots are also considered the
    same structure when their observed base intervals substantially overlap.
    """
    if left["identity"] == right["identity"]:
        return True
    left_pivot = float(left["pivot"])
    right_pivot = float(right["pivot"])
    if left_pivot <= 0 or right_pivot <= 0:
        return False
    relative_pivot_gap = abs(left_pivot - right_pivot) / min(
        left_pivot, right_pivot
    )
    if relative_pivot_gap > 0.03 + 1e-12:
        return False

    left_start = int(left["base_start"])
    right_start = int(right["base_start"])
    left_end = int(left["structure_end"])
    right_end = int(right["structure_end"])
    overlap = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    shorter = min(left_end - left_start + 1, right_end - right_start + 1)
    return shorter > 0 and overlap / shorter >= 0.60


def _classification_key(candidate: dict[str, Any]) -> tuple[float, int, float, int, int, str]:
    """Return a total, deterministic order for competing classifications."""
    total = float(candidate["score"][0])
    if not np.isfinite(total):
        total = float("-inf")
    setup_type: SetupType = candidate["setup_type"]
    prior = float(candidate["prior"])
    if not np.isfinite(prior):
        prior = float("-inf")
    return (
        total,
        _CLASSIFICATION_SPECIFICITY[setup_type],
        prior,
        int(candidate["base_days"]),
        -int(candidate["pivot_idx"]),
        setup_type,
    )


def _collapse_overlapping_classifications(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep one best label per mutually overlapping structure cluster.

    Complete-linkage prevents a bridge candidate from merging two pivots that
    do not overlap each other. Sorting first makes cluster construction and the
    returned representatives independent of detector call order.
    """
    components: list[list[dict[str, Any]]] = []
    ordered = sorted(candidates, key=_classification_key, reverse=True)
    for candidate in ordered:
        for component in components:
            if all(_same_structure(candidate, member) for member in component):
                component.append(candidate)
                break
        else:
            components.append([candidate])
    return [max(component, key=_classification_key) for component in components]


def _validate_inputs(
    dates: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    continuity_segment: np.ndarray,
    continuity_break: np.ndarray,
    trading_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(dates)
    if not (
        len(high)
        == len(low)
        == len(close)
        == len(volume)
        == len(continuity_segment)
        == len(continuity_break)
        == n
    ):
        raise ValueError(
            "dates, OHLCV and price continuity arrays must have equal length"
        )
    if trading_dates.empty or not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be non-empty, sorted and unique")
    session_positions = trading_dates.get_indexer(dates)
    if np.any(session_positions < 0):
        raise ValueError("every symbol date must exist in trading_dates")
    segment_numeric = pd.to_numeric(
        pd.Series(continuity_segment), errors="coerce"
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(segment_numeric).all()
        or not np.equal(segment_numeric, np.floor(segment_numeric)).all()
        or np.any(segment_numeric <= 0)
    ):
        raise ValueError(
            "price_continuity_segment must contain positive finite integers"
        )
    break_values = pd.Series(continuity_break, dtype="boolean")
    if break_values.isna().any():
        raise ValueError("price_continuity_break must contain booleans")
    price_complete = np.isfinite(high) & np.isfinite(low) & np.isfinite(close)
    return (
        session_positions,
        price_complete,
        segment_numeric.astype(np.int64),
        break_values.to_numpy(dtype=bool),
    )


def find_setups(
    symbol: str,
    dates: pd.DatetimeIndex,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    continuity_segment: np.ndarray,
    continuity_break: np.ndarray,
    pass_idx: np.ndarray,
    cfg: Any,
    *,
    trading_dates: pd.DatetimeIndex,
) -> list[Setup]:
    """Detect all supported setup classes behind one runner-facing API.

    ``pass_idx`` is the model-independent candidate slate. Pattern state still
    advances on every symbol bar, which keeps swing confirmation causal and
    decouples setup recognition from fundamental/group ranking dates.
    """
    dates = pd.DatetimeIndex(dates)
    trading_dates = pd.DatetimeIndex(trading_dates)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    volume = np.asarray(volume, dtype=float)
    continuity_segment = np.asarray(continuity_segment)
    continuity_break = np.asarray(continuity_break)
    rules = _Rules.from_config(cfg)
    if rules.swing_window < 1:
        raise ValueError("swing_window must be >= 1")
    session_positions, complete, segments, segment_breaks = _validate_inputs(
        dates,
        high,
        low,
        close,
        volume,
        continuity_segment,
        continuity_break,
        trading_dates,
    )
    pass_days = {int(i) for i in np.asarray(pass_idx, dtype=int) if 0 <= int(i) < len(dates)}
    if not pass_days:
        return []

    setups: list[Setup] = []
    swings: list[Swing] = []
    emission_history: list[dict[str, Any]] = []
    base_count = 0
    previous_pivot: float | None = None
    segment_start = 0
    k = rules.swing_window

    for t in range(max(pass_days) + 1):
        explicit_boundary = (
            t == 0
            or segment_breaks[t]
            or segments[t] != segments[t - 1]
        )
        gap = t > 0 and session_positions[t] != session_positions[t - 1] + 1
        if explicit_boundary or gap or not complete[t]:
            swings.clear()
            emission_history.clear()
            base_count = 0
            previous_pivot = None
            segment_start = t + (0 if complete[t] else 1)
            if not complete[t]:
                continue
        candidate_idx = t - k
        if candidate_idx >= segment_start + k:
            window = slice(candidate_idx - k, candidate_idx + k + 1)
            if complete[window].all() and np.all(np.diff(session_positions[window]) == 1):
                event = _candidate_at(high, low, candidate_idx, k)
                if event is not None:
                    _apply_candidate(swings, event)
        if t not in pass_days or t - segment_start < 25:
            continue

        candidates = [
            _range_candidate("power_play", t, high, low, close, volume, session_positions, segment_start, rules),
            _vcp_candidate(t, swings, high, low, close, volume, session_positions, segment_start, rules),
            _range_candidate("tight_shelf", t, high, low, close, volume, session_positions, segment_start, rules),
            _range_candidate("flat_base", t, high, low, close, volume, session_positions, segment_start, rules),
        ]
        observed = [candidate for candidate in candidates if candidate is not None]
        if not observed:
            continue

        # Collapse only alternate labels of the same structure. Distinct pivots
        # on the same day remain separate first-touch research candidates; the
        # portfolio layer can still nominate at most one per symbol.
        representatives = sorted(
            _collapse_overlapping_classifications(observed),
            key=_classification_key,
            reverse=True,
        )
        for found in representatives:
            if any(_same_structure(found, prior) for prior in emission_history):
                # A base is one causal structure, not a fresh setup every time
                # its tightness or volume score changes. A clearly higher
                # pivot falls outside ``_same_structure`` and remains eligible
                # as a genuinely new continuation base.
                continue
            setup_type: SetupType = found["setup_type"]
            emission_history = [
                prior
                for prior in emission_history
                if not _same_structure(found, prior)
            ]
            emission_history.append(found)

            if previous_pivot is None or found["pivot"] >= previous_pivot * 1.25:
                base_count = 1
            else:
                base_count += 1
            previous_pivot = found["pivot"]
            detect_session = int(trading_dates.searchsorted(dates[t]))
            valid_session = min(
                detect_session + rules.setup_valid_days,
                len(trading_dates) - 1,
            )
            total, contraction, volume_score, tight_score, proximity_score = (
                found["score"]
            )
            prior_score = 25.0 * float(
                np.clip(found["prior"] / 0.60, 0.0, 1.0)
            )
            setups.append(
                Setup(
                    symbol=symbol,
                    setup_type=setup_type,
                    price_continuity_segment=int(segments[t]),
                    detect_date=dates[t].date(),
                    pivot=round(float(found["pivot"]), 8),
                    last_low=round(float(found["last_low"]), 8),
                    stop_level=round(
                        max(
                            float(found["last_low"]),
                            float(found["pivot"]) * (1.0 - rules.stop_max_pct),
                        ),
                        8,
                    ),
                    base_start_date=dates[int(found["base_start"])].date(),
                    base_days=int(found["base_days"]),
                    n_contractions=(
                        len(found["depths"]) if setup_type == "vcp" else 0
                    ),
                    contraction_depths=found["depths"],
                    base_count=base_count,
                    dryup_ratio=round(float(found["dryup"]), 4),
                    setup_score=round(float(total), 4),
                    prior_advance_pct=round(float(found["prior"]), 4),
                    final_tightness_pct=round(float(found["tightness"]), 4),
                    structure_quality_score=round(float(contraction), 4),
                    volume_dryup_score=round(float(volume_score), 4),
                    tightness_score=round(float(tight_score), 4),
                    pivot_proximity_score=round(float(proximity_score), 4),
                    prior_advance_score=round(float(prior_score), 4),
                    close=round(float(close[t]), 8),
                    valid_until=trading_dates[valid_session].date(),
                )
            )
    return setups
