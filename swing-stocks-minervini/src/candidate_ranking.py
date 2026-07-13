"""Causal candidate quality, fill and slate-priority estimates.

The three public estimates deliberately have different meanings:

``quality_score``
    Setup-type-specific posterior expected R-multiple.  Its raw signal uses
    technical and point-in-time fundamental information only.  Pivot distance
    and readiness are explicitly excluded.  The posterior is exposed only
    after its stored, purged walk-forward predictions demonstrate a robust
    monotone outcome lift within the same setup class; before that it is the
    explicit neutral prior.
``fill_probability``
    Setup-type-specific posterior probability that an order touches.  This is
    the only model that may use the readiness/pivot-distance signal.
``slate_priority``
    ``quality_score * fill_probability``.  It is an expected-R contribution,
    not another independently tuned score.

Both calibrators are deterministic, dependency-free kernel smoothers with
explicit neutral priors.  Every label carries the date on which its outcome
became known.  A snapshot for session ``t`` can therefore consume only labels
whose ``available_date`` is no later than the information close ``t - 1``.
Calibration is always isolated by setup class; observations are never pooled
across VCP, flat-base, power-play, tight-shelf or unknown classes.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from bisect import bisect_right
from dataclasses import dataclass, field, replace
from hashlib import blake2b
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd


QUALITY_FEATURES = frozenset(
    {
        "trend",
        "rs",
        "dryup",
        "structure",
        "tightness",
        "prior_advance",
        "fundamental",
    }
)

# These profiles create a within-class raw ordering signal.  They are not
# treated as comparable return forecasts across setup classes.  Only the
# class-local, outcome-calibrated ``quality_score`` is comparable.
DEFAULT_QUALITY_WEIGHTS: dict[str, dict[str, float]] = {
    "vcp": {
        "trend": 0.10,
        "rs": 0.22,
        "dryup": 0.23,
        "structure": 0.22,
        "tightness": 0.13,
        "prior_advance": 0.03,
        "fundamental": 0.07,
    },
    "flat_base": {
        "trend": 0.10,
        "rs": 0.23,
        "dryup": 0.11,
        "structure": 0.21,
        "tightness": 0.20,
        "prior_advance": 0.05,
        "fundamental": 0.10,
    },
    "power_play": {
        "trend": 0.10,
        "rs": 0.23,
        "dryup": 0.05,
        "structure": 0.13,
        "tightness": 0.09,
        "prior_advance": 0.31,
        "fundamental": 0.09,
    },
    "tight_shelf": {
        "trend": 0.10,
        "rs": 0.20,
        "dryup": 0.14,
        "structure": 0.16,
        "tightness": 0.27,
        "prior_advance": 0.04,
        "fundamental": 0.09,
    },
    "default": {
        "trend": 0.10,
        "rs": 0.22,
        "dryup": 0.14,
        "structure": 0.19,
        "tightness": 0.18,
        "prior_advance": 0.08,
        "fundamental": 0.09,
    },
}

# Quality validation uses non-overlapping, two-year calendar review epochs.
# Each of the eight outcome-availability quarters is an independent regime
# block, while completion dates are equal-weight clusters inside a block.  A
# review happens once at the next epoch boundary and remains frozen for the
# full following epoch; results are therefore never repeatedly peeked as
# individual trades close.  These are predeclared model-contract constants,
# not parameters tuned on the current backtest.
QUALITY_VALIDATION_GROUPS = 4
QUALITY_VALIDATION_QUARTERS_PER_REVIEW = 8
QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER = 40
QUALITY_VALIDATION_MIN_DAYS_PER_GROUP = 5
QUALITY_VALIDATION_BLOCK_CONFIDENCE_T = 2.365
QUALITY_VALIDATION_EPOCH_ANCHOR_YEAR = 2000
QUALITY_VALIDATION_MIN_LABELS = (
    QUALITY_VALIDATION_QUARTERS_PER_REVIEW
    * QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER
)


def _day(value: Any) -> pd.Timestamp:
    """Return a timezone-neutral normalized timestamp for causal comparisons."""
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("calibration dates must not be missing")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp.normalize()


@dataclass(frozen=True)
class QualityCalibrationLabel:
    """A completed trade outcome and when that outcome became observable.

    ``raw_quality_score`` must be the raw score from the entry-decision
    snapshot, not a value recomputed at exit.  One independently executable
    first-touch setup contributes one quality label only after its complete
    R-multiple is known.  ``walk_forward_quality_score`` is the base posterior
    that was produced at that original snapshot using only outcomes already
    available then.  It is therefore a purged out-of-sample prediction rather
    than an in-sample refit.  Portfolio selection must not define this training
    set, otherwise the calibrator would learn its own selection bias.
    """

    setup_type: str
    information_date: pd.Timestamp
    available_date: pd.Timestamp
    raw_quality_score: float
    realized_r_multiple: float
    walk_forward_quality_score: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        information_date = _day(self.information_date)
        available_date = _day(self.available_date)
        raw_quality_score = float(self.raw_quality_score)
        realized_r_multiple = float(self.realized_r_multiple)
        weight = float(self.weight)
        walk_forward_quality_score = float(self.walk_forward_quality_score)
        if available_date <= information_date:
            raise ValueError("quality label must become available after its snapshot")
        if not isfinite(raw_quality_score) or not 0.0 <= raw_quality_score <= 100.0:
            raise ValueError("raw_quality_score must be finite and between 0 and 100")
        if not isfinite(realized_r_multiple):
            raise ValueError("realized_r_multiple must be finite")
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("label weight must be finite and positive")
        if not isfinite(walk_forward_quality_score):
            raise ValueError("walk_forward_quality_score must be finite")
        object.__setattr__(self, "setup_type", str(self.setup_type))
        object.__setattr__(self, "information_date", information_date)
        object.__setattr__(self, "available_date", available_date)
        object.__setattr__(self, "raw_quality_score", raw_quality_score)
        object.__setattr__(self, "realized_r_multiple", realized_r_multiple)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(
            self, "walk_forward_quality_score", walk_forward_quality_score
        )


@dataclass(frozen=True)
class FillCalibrationLabel:
    """One order-session executable-fill outcome known after session close.

    ``filled`` describes whether the setup's own stop-buy and buy-zone rules
    would have executed.  Portfolio capacity, market gates and whether the
    candidate happened to be selected must not turn that counterfactual market
    outcome into a negative label.
    """

    setup_type: str
    information_date: pd.Timestamp
    available_date: pd.Timestamp
    readiness_signal: float
    filled: bool
    weight: float = 1.0

    def __post_init__(self) -> None:
        information_date = _day(self.information_date)
        available_date = _day(self.available_date)
        readiness_signal = float(self.readiness_signal)
        weight = float(self.weight)
        if available_date <= information_date:
            raise ValueError("fill label must become available after its snapshot")
        if not isfinite(readiness_signal) or not 0.0 <= readiness_signal <= 100.0:
            raise ValueError("readiness_signal must be finite and between 0 and 100")
        if not isinstance(self.filled, (bool, np.bool_)):
            raise TypeError("filled must be boolean")
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("label weight must be finite and positive")
        object.__setattr__(self, "setup_type", str(self.setup_type))
        object.__setattr__(self, "information_date", information_date)
        object.__setattr__(self, "available_date", available_date)
        object.__setattr__(self, "readiness_signal", readiness_signal)
        object.__setattr__(self, "filled", bool(self.filled))
        object.__setattr__(self, "weight", weight)


@dataclass(frozen=True)
class CandidateSnapshot:
    """All information and causal estimates available before one order session."""

    setup_key: int
    setup_id: int | None
    symbol: str
    setup_type: str
    session_date: pd.Timestamp
    information_date: pd.Timestamp
    setup_age_sessions: int
    pivot: float | None
    prior_close: float | None
    distance_to_pivot_pct: float | None
    readiness_score: float | None
    current_rs_rating: float | None
    dynamic_dryup_ratio: float | None
    dryup_source: str
    dryup_coverage: float
    fundamental_score: float | None
    fundamental_coverage: float
    eps_yoy: float | None
    revenue_yoy: float | None
    structure_quality_score: float | None
    tightness_score: float | None
    prior_advance_score: float | None
    raw_quality_score: float
    quality_feature_coverage: float
    walk_forward_quality_score: float
    quality_score: float
    quality_model_validated: bool
    fill_probability: float
    slate_priority: float
    quality_rank: int | None
    quality_calibration_count: int
    quality_effective_samples: float
    quality_validation_count: int
    fill_calibration_count: int
    fill_effective_samples: float
    context_values: tuple[tuple[str, float | None], ...]

    def context_value(self, name: str) -> float | None:
        """Return one named as-of context value, or ``None`` when absent."""
        return dict(self.context_values).get(name)


@dataclass(frozen=True)
class _Estimate:
    value: float
    count: int
    effective_samples: float


@dataclass(frozen=True)
class _QualityValidation:
    passed: bool
    count: int
    review_epoch_start: int | None


@dataclass
class _BinnedCalibrationState:
    """Incremental as-of aggregates for one setup class."""

    as_of: pd.Timestamp | None = None
    next_label: int = 0
    count: int = 0
    weights: np.ndarray = field(default_factory=lambda: np.zeros(101, dtype=float))
    weighted_signals: np.ndarray = field(
        default_factory=lambda: np.zeros(101, dtype=float)
    )
    weighted_targets: np.ndarray = field(
        default_factory=lambda: np.zeros(101, dtype=float)
    )
    validation_scores: list[float] = field(default_factory=list)
    validation_targets: list[float] = field(default_factory=list)
    validation_weights: list[float] = field(default_factory=list)
    validation_dates: list[pd.Timestamp] = field(default_factory=list)
    validation_result: _QualityValidation | None = None

    def reset(self) -> None:
        self.as_of = None
        self.next_label = 0
        self.count = 0
        self.weights.fill(0.0)
        self.weighted_signals.fill(0.0)
        self.weighted_targets.fill(0.0)
        self.validation_scores.clear()
        self.validation_targets.clear()
        self.validation_weights.clear()
        self.validation_dates.clear()
        self.validation_result = None


class CandidateRanker:
    """Build, calibrate and order strictly pre-session candidate snapshots.

    Labels can be supplied up front or registered online.  Supplying a future
    label up front is safe: every estimate filters by ``available_date``.  For
    a realistic simulation, register a fill label after every eligible order
    session and a quality label only when the associated trade has closed.
    """

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
        close: pd.DataFrame,
        *,
        continuity_segment: pd.DataFrame,
        volume: pd.DataFrame | None = None,
        rs_rating: pd.DataFrame | None = None,
        context: Mapping[str, pd.DataFrame] | None = None,
        quality_type_weights: Mapping[str, Mapping[str, float]] | None = None,
        quality_labels: Iterable[QualityCalibrationLabel] = (),
        fill_labels: Iterable[FillCalibrationLabel] = (),
        quality_prior_r_multiple: float = 0.0,
        fill_prior_probability: float = 0.5,
        quality_prior_strength: float = 12.0,
        fill_prior_strength: float = 12.0,
        quality_kernel_bandwidth: float = 20.0,
        fill_kernel_bandwidth: float = 20.0,
        dryup_recent_sessions: int = 5,
        dryup_baseline_sessions: int = 50,
        dryup_zero_ratio: float = 1.25,
    ) -> None:
        self.dates = pd.DatetimeIndex(dates)
        self.symbols = pd.Index(symbols)
        if not self.dates.is_monotonic_increasing or not self.dates.is_unique:
            raise ValueError("dates must be unique and monotonically increasing")
        if not self.symbols.is_unique:
            raise ValueError("symbols must be unique")

        self.close = self._validate_matrix("close", close)
        self.continuity_segment = self._validate_matrix(
            "continuity_segment", continuity_segment
        )
        if self.continuity_segment is None:  # pragma: no cover - signature enforces it
            raise TypeError("continuity_segment is required")
        self.volume = self._validate_matrix("volume", volume)
        self.rs_rating = self._validate_matrix("rs_rating", rs_rating)
        self.context = {
            str(name): self._validate_matrix(f"context[{name}]", matrix)
            for name, matrix in sorted((context or {}).items())
        }
        self._symbol_col = {str(symbol): col for col, symbol in enumerate(self.symbols)}

        if dryup_recent_sessions < 3:
            raise ValueError("dryup_recent_sessions must be at least 3")
        if dryup_baseline_sessions < 20:
            raise ValueError("dryup_baseline_sessions must be at least 20")
        self.dryup_recent_sessions = int(dryup_recent_sessions)
        self.dryup_baseline_sessions = int(dryup_baseline_sessions)
        self.dryup_zero_ratio = self._positive("dryup_zero_ratio", dryup_zero_ratio)
        self.quality_prior_r_multiple = self._finite(
            "quality_prior_r_multiple", quality_prior_r_multiple
        )
        self.fill_prior_probability = self._finite(
            "fill_prior_probability", fill_prior_probability
        )
        if not 0.0 <= self.fill_prior_probability <= 1.0:
            raise ValueError("fill_prior_probability must be between 0 and 1")
        self.quality_prior_strength = self._positive(
            "quality_prior_strength", quality_prior_strength
        )
        self.fill_prior_strength = self._positive(
            "fill_prior_strength", fill_prior_strength
        )
        self.quality_kernel_bandwidth = self._positive(
            "quality_kernel_bandwidth", quality_kernel_bandwidth
        )
        self.fill_kernel_bandwidth = self._positive(
            "fill_kernel_bandwidth", fill_kernel_bandwidth
        )
        self.quality_type_weights = self._validate_weights(
            quality_type_weights or DEFAULT_QUALITY_WEIGHTS
        )
        self._quality_labels = [self._quality_label(label) for label in quality_labels]
        self._fill_labels = [self._fill_label(label) for label in fill_labels]
        self._quality_by_type: dict[str, list[QualityCalibrationLabel]] = {}
        self._fill_by_type: dict[str, list[FillCalibrationLabel]] = {}
        for label in sorted(self._quality_labels, key=self._quality_sort_key):
            self._quality_by_type.setdefault(label.setup_type, []).append(label)
        for label in sorted(self._fill_labels, key=self._fill_sort_key):
            self._fill_by_type.setdefault(label.setup_type, []).append(label)
        self._quality_states: dict[str, _BinnedCalibrationState] = {}
        self._fill_states: dict[str, _BinnedCalibrationState] = {}

    @property
    def quality_labels(self) -> tuple[QualityCalibrationLabel, ...]:
        """Immutable copy suitable for a second, causally filtered simulation."""
        return tuple(sorted(self._quality_labels, key=self._quality_export_sort_key))

    @property
    def fill_labels(self) -> tuple[FillCalibrationLabel, ...]:
        """Immutable copy suitable for a second, causally filtered simulation."""
        return tuple(sorted(self._fill_labels, key=self._fill_export_sort_key))

    @staticmethod
    def _finite(name: str, value: Any) -> float:
        numeric = float(value)
        if not isfinite(numeric):
            raise ValueError(f"{name} must be finite")
        return numeric

    @classmethod
    def _positive(cls, name: str, value: Any) -> float:
        numeric = cls._finite(name, value)
        if numeric <= 0.0:
            raise ValueError(f"{name} must be positive")
        return numeric

    @staticmethod
    def _quality_label(label: QualityCalibrationLabel) -> QualityCalibrationLabel:
        if not isinstance(label, QualityCalibrationLabel):
            raise TypeError("quality_labels must contain QualityCalibrationLabel values")
        return label

    @staticmethod
    def _fill_label(label: FillCalibrationLabel) -> FillCalibrationLabel:
        if not isinstance(label, FillCalibrationLabel):
            raise TypeError("fill_labels must contain FillCalibrationLabel values")
        return label

    @staticmethod
    def _quality_sort_key(label: QualityCalibrationLabel) -> tuple:
        return (
            label.available_date,
            label.information_date,
            label.raw_quality_score,
            label.realized_r_multiple,
            label.weight,
            label.walk_forward_quality_score,
        )

    @classmethod
    def _quality_export_sort_key(cls, label: QualityCalibrationLabel) -> tuple:
        return (label.setup_type, *cls._quality_sort_key(label))

    @staticmethod
    def _fill_sort_key(label: FillCalibrationLabel) -> tuple:
        return (
            label.available_date,
            label.information_date,
            label.readiness_signal,
            label.filled,
            label.weight,
        )

    @classmethod
    def _fill_export_sort_key(cls, label: FillCalibrationLabel) -> tuple:
        return (label.setup_type, *cls._fill_sort_key(label))

    def _validate_matrix(
        self, name: str, matrix: pd.DataFrame | None
    ) -> pd.DataFrame | None:
        if matrix is None:
            return None
        if not isinstance(matrix, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        if not pd.DatetimeIndex(matrix.index).equals(self.dates):
            raise ValueError(f"{name} must have exactly the configured dates")
        if not matrix.columns.equals(self.symbols):
            raise ValueError(f"{name} must have exactly the configured symbols")
        return matrix

    @staticmethod
    def _validate_weights(
        weights: Mapping[str, Mapping[str, float]],
    ) -> dict[str, dict[str, float]]:
        if "default" not in weights:
            raise ValueError("quality_type_weights must contain a default profile")
        validated: dict[str, dict[str, float]] = {}
        for setup_type, profile in weights.items():
            unknown = set(profile) - QUALITY_FEATURES
            if unknown:
                raise ValueError(
                    "non-whitelisted quality features: " + ", ".join(sorted(unknown))
                )
            clean: dict[str, float] = {}
            for name, weight in profile.items():
                numeric = float(weight)
                if not isfinite(numeric) or numeric < 0.0:
                    raise ValueError("quality weights must be finite and non-negative")
                clean[name] = numeric
            if sum(clean.values()) <= 0.0:
                raise ValueError("each setup-type profile needs positive total weight")
            validated[str(setup_type)] = clean
        return validated

    @staticmethod
    def _attr(setup: object, name: str, default: Any = None) -> Any:
        if isinstance(setup, Mapping):
            return setup.get(name, default)
        return getattr(setup, name, default)

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        return numeric if isfinite(numeric) else None

    @staticmethod
    def _scaled(value: float | None, maximum: float) -> float | None:
        if value is None:
            return None
        return float(np.clip(value / maximum * 100.0, 0.0, 100.0))

    def _matrix_value(
        self, matrix: pd.DataFrame | None, row: int, col: int
    ) -> float | None:
        if matrix is None:
            return None
        return self._optional_float(matrix.iat[row, col])

    def _dynamic_dryup(
        self, information_idx: int, col: int
    ) -> tuple[float | None, float]:
        if self.volume is None:
            return None, 0.0
        current_segment = self.continuity_segment.iat[information_idx, col]
        if pd.isna(current_segment):
            return None, 0.0

        # A segment id is meaningful only as one contiguous price history.
        # Walk back at most the configured feature horizon; no older row can
        # contribute to either the recent or baseline window.
        horizon_start = max(
            0,
            information_idx
            - self.dryup_recent_sessions
            - self.dryup_baseline_sessions
            + 1,
        )
        segment_start = information_idx
        for row in range(information_idx - 1, horizon_start - 1, -1):
            previous_segment = self.continuity_segment.iat[row, col]
            if pd.isna(previous_segment) or previous_segment != current_segment:
                break
            segment_start = row

        recent_start = information_idx - self.dryup_recent_sessions + 1
        baseline_end = recent_start
        if recent_start < segment_start:
            return None, 0.0
        baseline_start = max(
            segment_start, baseline_end - self.dryup_baseline_sessions
        )
        # A fresh listing/reorganisation needs at least 20 complete baseline
        # sessions plus the configured recent window before dry-up is known.
        if baseline_end - baseline_start < 20:
            return None, 0.0
        baseline = pd.to_numeric(
            self.volume.iloc[baseline_start:baseline_end, col], errors="coerce"
        ).to_numpy(dtype=float)
        recent = pd.to_numeric(
            self.volume.iloc[recent_start : information_idx + 1, col], errors="coerce"
        ).to_numpy(dtype=float)
        baseline = baseline[np.isfinite(baseline) & (baseline > 0)]
        recent = recent[np.isfinite(recent) & (recent > 0)]
        coverage = min(
            min(1.0, len(baseline) / float(self.dryup_baseline_sessions)),
            min(1.0, len(recent) / float(self.dryup_recent_sessions)),
        )
        if len(baseline) < 20 or len(recent) < 3:
            return None, 0.0
        return float(np.mean(recent) / np.mean(baseline)), coverage

    def _first_order_idx(self, setup: object) -> int:
        start_idx = self._optional_float(self._attr(setup, "start_idx"))
        if start_idx is not None:
            return int(start_idx)
        detect_date = self._attr(setup, "detect_date")
        if detect_date is None:
            raise ValueError("a setup needs detect_date or start_idx")
        return int(self.dates.searchsorted(pd.Timestamp(detect_date), side="right"))

    @staticmethod
    def _signal_bin(signal: float) -> int:
        return min(100, max(0, int(signal)))

    @staticmethod
    def _state_estimate(
        state: _BinnedCalibrationState,
        *,
        signal: float,
        bandwidth: float,
        prior_value: float,
        prior_strength: float,
    ) -> _Estimate:
        populated = state.weights > 0.0
        if not np.any(populated):
            return _Estimate(prior_value, state.count, 0.0)
        weights = state.weights[populated]
        centers = state.weighted_signals[populated] / weights
        kernels = np.exp(-0.5 * np.square((centers - signal) / bandwidth))
        kernel_weights = kernels * weights
        effective_samples = float(np.sum(kernel_weights))
        numerator = prior_strength * prior_value + float(
            np.sum(kernels * state.weighted_targets[populated])
        )
        value = numerator / (prior_strength + effective_samples)
        return _Estimate(value, state.count, effective_samples)

    def _quality_state(
        self, setup_type: str, information_date: pd.Timestamp
    ) -> _BinnedCalibrationState:
        state = self._quality_states.setdefault(
            setup_type, _BinnedCalibrationState()
        )
        if state.as_of is not None and information_date < state.as_of:
            state.reset()
        labels = self._quality_by_type.get(setup_type, [])
        while (
            state.next_label < len(labels)
            and labels[state.next_label].available_date <= information_date
        ):
            label = labels[state.next_label]
            index = self._signal_bin(label.raw_quality_score)
            state.weights[index] += label.weight
            state.weighted_signals[index] += label.weight * label.raw_quality_score
            state.weighted_targets[index] += (
                label.weight * label.realized_r_multiple
            )
            state.validation_scores.append(label.walk_forward_quality_score)
            state.validation_targets.append(label.realized_r_multiple)
            state.validation_weights.append(label.weight)
            state.validation_dates.append(label.available_date)
            state.next_label += 1
            state.count += 1
        state.as_of = information_date
        return state

    @staticmethod
    def _quarter_ordinal(value: Any) -> int:
        date = _day(value)
        return date.year * 4 + date.quarter - 1

    @classmethod
    def _completed_review_epoch(
        cls, information_date: pd.Timestamp
    ) -> tuple[int, int] | None:
        """Return the latest fixed eight-quarter epoch completed before ``t``."""
        anchor = QUALITY_VALIDATION_EPOCH_ANCHOR_YEAR * 4
        current_quarter = cls._quarter_ordinal(information_date)
        completed_epochs = (current_quarter - anchor) // (
            QUALITY_VALIDATION_QUARTERS_PER_REVIEW
        )
        if completed_epochs <= 0:
            return None
        start = anchor + (
            completed_epochs - 1
        ) * QUALITY_VALIDATION_QUARTERS_PER_REVIEW
        return start, start + QUALITY_VALIDATION_QUARTERS_PER_REVIEW - 1

    @staticmethod
    def _quarter_group_means(
        scores: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        dates: np.ndarray,
    ) -> tuple[float, ...] | None:
        """Return score-group means with completion dates as equal clusters."""
        if len(scores) < QUALITY_VALIDATION_MIN_LABELS_PER_QUARTER:
            return None
        quantiles = np.linspace(
            0.0, 1.0, QUALITY_VALIDATION_GROUPS + 1, dtype=float
        )[1:-1]
        boundaries = np.quantile(scores, quantiles, method="linear")
        # Never split equal predictions into artificial ordered groups.
        if len(np.unique(boundaries)) != QUALITY_VALIDATION_GROUPS - 1:
            return None
        group_ids = np.searchsorted(boundaries, scores, side="right")
        means: list[float] = []
        for group_id in range(QUALITY_VALIDATION_GROUPS):
            group_mask = group_ids == group_id
            group_dates = dates[group_mask]
            unique_dates = np.unique(group_dates)
            if len(unique_dates) < QUALITY_VALIDATION_MIN_DAYS_PER_GROUP:
                return None
            group_targets = targets[group_mask]
            group_weights = weights[group_mask]
            daily_means = []
            for date in unique_dates:
                day_mask = group_dates == date
                daily_means.append(
                    float(
                        np.average(
                            group_targets[day_mask], weights=group_weights[day_mask]
                        )
                    )
                )
            # A day with many correlated exits remains one observation.
            means.append(float(np.mean(daily_means)))
        return tuple(means)

    @classmethod
    def _quality_validation(
        cls,
        state: _BinnedCalibrationState,
        information_date: pd.Timestamp,
    ) -> _QualityValidation:
        """Review one frozen, non-overlapping two-year OOF epoch.

        Each outcome-availability quarter supplies four local score groups.
        Quarter-level top-minus-bottom lifts are the independent observations
        for the confidence bound, and aggregate group means must be monotone.
        No group itself must be profitable.
        """
        epoch = cls._completed_review_epoch(information_date)
        epoch_start = epoch[0] if epoch is not None else None
        if (
            state.validation_result is not None
            and state.validation_result.review_epoch_start == epoch_start
        ):
            return state.validation_result
        if epoch is None:
            result = _QualityValidation(False, 0, None)
            state.validation_result = result
            return result

        quarter_ordinals = np.asarray(
            [cls._quarter_ordinal(date) for date in state.validation_dates],
            dtype=int,
        )
        epoch_mask = (quarter_ordinals >= epoch[0]) & (
            quarter_ordinals <= epoch[1]
        )
        count = int(np.sum(epoch_mask))
        if count < QUALITY_VALIDATION_MIN_LABELS:
            result = _QualityValidation(False, count, epoch_start)
            state.validation_result = result
            return result

        scores = np.asarray(state.validation_scores, dtype=float)[epoch_mask]
        targets = np.asarray(state.validation_targets, dtype=float)[epoch_mask]
        weights = np.asarray(state.validation_weights, dtype=float)[epoch_mask]
        dates = np.asarray(state.validation_dates, dtype="datetime64[D]")[epoch_mask]
        selected_quarters = quarter_ordinals[epoch_mask]
        block_means: list[tuple[float, ...]] = []
        for quarter in range(epoch[0], epoch[1] + 1):
            block_mask = selected_quarters == quarter
            means = cls._quarter_group_means(
                scores[block_mask],
                targets[block_mask],
                weights[block_mask],
                dates[block_mask],
            )
            if means is None:
                result = _QualityValidation(False, count, epoch_start)
                state.validation_result = result
                return result
            block_means.append(means)

        blocks = np.asarray(block_means, dtype=float)
        aggregate_means = np.mean(blocks, axis=0)
        monotone = bool(np.all(np.diff(aggregate_means) >= 0.0))
        block_lifts = blocks[:, -1] - blocks[:, 0]
        mean_lift = float(np.mean(block_lifts))
        block_standard_error = float(
            np.std(block_lifts, ddof=1) / np.sqrt(len(block_lifts))
        )
        robust_lift = (
            mean_lift
            - QUALITY_VALIDATION_BLOCK_CONFIDENCE_T * block_standard_error
            > 0.0
        )
        result = _QualityValidation(monotone and robust_lift, count, epoch_start)
        state.validation_result = result
        return result

    def _fill_state(
        self, setup_type: str, information_date: pd.Timestamp
    ) -> _BinnedCalibrationState:
        state = self._fill_states.setdefault(setup_type, _BinnedCalibrationState())
        if state.as_of is not None and information_date < state.as_of:
            state.reset()
        labels = self._fill_by_type.get(setup_type, [])
        while (
            state.next_label < len(labels)
            and labels[state.next_label].available_date <= information_date
        ):
            label = labels[state.next_label]
            index = self._signal_bin(label.readiness_signal)
            state.weights[index] += label.weight
            state.weighted_signals[index] += label.weight * label.readiness_signal
            state.weighted_targets[index] += label.weight * float(label.filled)
            state.next_label += 1
            state.count += 1
        state.as_of = information_date
        return state

    def _quality_estimate(
        self, setup_type: str, raw_score: float, information_date: pd.Timestamp
    ) -> tuple[_Estimate, _QualityValidation]:
        state = self._quality_state(setup_type, information_date)
        estimate = self._state_estimate(
            state,
            signal=raw_score,
            bandwidth=self.quality_kernel_bandwidth,
            prior_value=self.quality_prior_r_multiple,
            prior_strength=self.quality_prior_strength,
        )
        value = estimate.value
        if not isfinite(value):
            raise RuntimeError("quality calibration produced a non-finite estimate")
        return estimate, self._quality_validation(state, information_date)

    def _fill_estimate(
        self, setup_type: str, readiness_signal: float, information_date: pd.Timestamp
    ) -> _Estimate:
        state = self._fill_state(setup_type, information_date)
        estimate = self._state_estimate(
            state,
            signal=readiness_signal,
            bandwidth=self.fill_kernel_bandwidth,
            prior_value=self.fill_prior_probability,
            prior_strength=self.fill_prior_strength,
        )
        return replace(estimate, value=float(np.clip(estimate.value, 0.0, 1.0)))

    def add_quality_label(
        self,
        snapshot: CandidateSnapshot,
        *,
        available_date: Any,
        realized_r_multiple: float,
        weight: float = 1.0,
    ) -> QualityCalibrationLabel:
        """Register a closed outcome using the original decision snapshot."""
        label = QualityCalibrationLabel(
            setup_type=snapshot.setup_type,
            information_date=snapshot.information_date,
            available_date=available_date,
            raw_quality_score=snapshot.raw_quality_score,
            realized_r_multiple=realized_r_multiple,
            weight=weight,
            walk_forward_quality_score=snapshot.walk_forward_quality_score,
        )
        self._quality_labels.append(label)
        labels = self._quality_by_type.setdefault(label.setup_type, [])
        if labels and label.available_date < labels[-1].available_date:
            position = bisect_right(
                [item.available_date for item in labels], label.available_date
            )
            labels.insert(position, label)
        else:
            labels.append(label)
        state = self._quality_states.get(label.setup_type)
        if (
            state is not None
            and state.as_of is not None
            and label.available_date <= state.as_of
        ):
            state.reset()
        return label

    def add_fill_label(
        self,
        snapshot: CandidateSnapshot,
        *,
        available_date: Any,
        filled: bool,
        weight: float = 1.0,
    ) -> FillCalibrationLabel:
        """Register one completed, capacity-independent fill outcome."""
        label = FillCalibrationLabel(
            setup_type=snapshot.setup_type,
            information_date=snapshot.information_date,
            available_date=available_date,
            readiness_signal=(
                snapshot.readiness_score
                if snapshot.readiness_score is not None
                else 50.0
            ),
            filled=filled,
            weight=weight,
        )
        self._fill_labels.append(label)
        labels = self._fill_by_type.setdefault(label.setup_type, [])
        if labels and label.available_date < labels[-1].available_date:
            position = bisect_right(
                [item.available_date for item in labels], label.available_date
            )
            labels.insert(position, label)
        else:
            labels.append(label)
        state = self._fill_states.get(label.setup_type)
        if (
            state is not None
            and state.as_of is not None
            and label.available_date <= state.as_of
        ):
            state.reset()
        return label

    def snapshot(
        self,
        setup: object,
        session_idx: int,
        *,
        setup_key: int | None = None,
    ) -> CandidateSnapshot:
        """Build one snapshot using no market observation newer than ``t - 1``."""
        if not 0 < session_idx < len(self.dates):
            raise ValueError("session_idx must have a preceding session")
        first_order_idx = self._first_order_idx(setup)
        if first_order_idx > session_idx:
            raise ValueError("setup is not yet known for the requested session")

        symbol = str(self._attr(setup, "symbol", ""))
        if symbol not in self._symbol_col:
            raise KeyError(f"unknown setup symbol {symbol!r}")
        col = self._symbol_col[symbol]
        information_idx = session_idx - 1
        information_date = _day(self.dates[information_idx])
        pivot = self._optional_float(self._attr(setup, "pivot"))
        if pivot is not None and pivot <= 0.0:
            pivot = None
        prior_close = self._matrix_value(self.close, information_idx, col)
        if pivot is None or prior_close is None:
            distance_fraction = None
            distance_pct = None
            readiness = None
        else:
            distance_fraction = (pivot - prior_close) / pivot
            distance_pct = 100.0 * distance_fraction
            scale = 0.10 if distance_fraction >= 0.0 else 0.02
            readiness = float(
                np.clip(
                    100.0 * (1.0 - abs(distance_fraction) / scale), 0.0, 100.0
                )
            )

        current_rs = self._matrix_value(self.rs_rating, information_idx, col)
        if current_rs is None:
            current_rs = self._optional_float(self._attr(setup, "rs_rating"))

        dynamic_dryup, dynamic_dryup_coverage = self._dynamic_dryup(
            information_idx, col
        )
        if self.volume is not None:
            dryup_ratio = dynamic_dryup
            dryup_coverage = dynamic_dryup_coverage
            dryup_source = (
                "dynamic"
                if dynamic_dryup is not None and dryup_coverage >= 1.0 - 1e-12
                else "dynamic_partial"
                if dynamic_dryup is not None
                else "missing"
            )
        else:
            dryup_ratio = self._optional_float(self._attr(setup, "dryup_ratio"))
            if dryup_ratio is not None and dryup_ratio > 0.0:
                dryup_source = "setup"
                dryup_coverage = 1.0
            else:
                dryup_ratio = None
                dryup_source = "missing"
                dryup_coverage = 0.0

        context_values = tuple(
            (name, self._matrix_value(matrix, information_idx, col))
            for name, matrix in self.context.items()
        )
        current_context = dict(context_values)

        def context_or_setup(name: str) -> float | None:
            value = current_context.get(name)
            return (
                value
                if value is not None
                else self._optional_float(self._attr(setup, name))
            )

        fundamental_score = context_or_setup("fundamental_score")
        raw_fundamental_coverage = context_or_setup("fundamental_coverage")
        fundamental_coverage = (
            float(np.clip(raw_fundamental_coverage, 0.0, 1.0))
            if raw_fundamental_coverage is not None
            else 1.0
            if fundamental_score is not None
            else 0.0
        )
        eps_yoy = context_or_setup("eps_yoy")
        revenue_yoy = context_or_setup("revenue_yoy")
        trend_template_pass = context_or_setup("trend_template_pass")
        structure = self._optional_float(
            self._attr(setup, "structure_quality_score")
        )
        tightness = self._optional_float(self._attr(setup, "tightness_score"))
        prior_advance = self._optional_float(
            self._attr(setup, "prior_advance_score")
        )
        setup_age = session_idx - first_order_idx

        dryup_component = (
            float(
                np.clip(
                    100.0 * (1.0 - dryup_ratio / self.dryup_zero_ratio),
                    0.0,
                    100.0,
                )
            )
            if dryup_ratio is not None
            else None
        )
        fundamental_observed_count = 6.0 * fundamental_coverage
        fundamental_component = (
            self._scaled(fundamental_score, fundamental_observed_count)
            if fundamental_observed_count > 0.0
            else None
        )
        components = {
            "trend": (
                float(np.clip(trend_template_pass * 100.0, 0.0, 100.0))
                if trend_template_pass is not None
                else None
            ),
            "rs": self._scaled(current_rs - 1.0, 98.0)
            if current_rs is not None
            else None,
            "dryup": dryup_component,
            "structure": self._scaled(structure, 25.0),
            "tightness": self._scaled(tightness, 20.0),
            "prior_advance": self._scaled(prior_advance, 25.0),
            "fundamental": fundamental_component,
        }
        component_coverage = {
            name: (
                dryup_coverage
                if name == "dryup"
                else fundamental_coverage
                if name == "fundamental"
                else 1.0
                if value is not None
                else 0.0
            )
            for name, value in components.items()
        }

        def effective_component(name: str) -> float:
            value = components.get(name)
            coverage = component_coverage.get(name, 0.0)
            observed_value = value if value is not None else 50.0
            return coverage * observed_value + (1.0 - coverage) * 50.0

        setup_type = str(self._attr(setup, "setup_type", "unknown"))
        profile = self.quality_type_weights.get(
            setup_type, self.quality_type_weights["default"]
        )
        total_weight = sum(profile.values())
        observed_weight = sum(
            weight * component_coverage.get(name, 0.0)
            for name, weight in profile.items()
        )
        raw_quality_score = (
            sum(weight * effective_component(name) for name, weight in profile.items())
            / total_weight
        )
        readiness_signal = readiness if readiness is not None else 50.0
        quality_estimate, quality_validation = self._quality_estimate(
            setup_type, raw_quality_score, information_date
        )
        fill_estimate = self._fill_estimate(
            setup_type, readiness_signal, information_date
        )
        exposed_quality_score = (
            quality_estimate.value
            if quality_validation.passed
            else self.quality_prior_r_multiple
        )
        # Fill probability must not create a hidden ordering while outcome
        # quality is unvalidated.  With a validated model the product regains
        # its expected-R-contribution meaning.
        slate_priority = (
            exposed_quality_score * fill_estimate.value
            if quality_validation.passed
            else 0.0
        )
        if not isfinite(slate_priority):
            raise RuntimeError("slate priority produced a non-finite estimate")

        raw_setup_id = self._optional_float(self._attr(setup, "setup_id"))
        setup_id = int(raw_setup_id) if raw_setup_id is not None else None
        inferred_key = self._optional_float(self._attr(setup, "sim_setup_key"))
        resolved_key = (
            int(setup_key)
            if setup_key is not None
            else int(inferred_key)
            if inferred_key is not None
            else setup_id
            if setup_id is not None
            else -1
        )
        return CandidateSnapshot(
            setup_key=resolved_key,
            setup_id=setup_id,
            symbol=symbol,
            setup_type=setup_type,
            session_date=self.dates[session_idx],
            information_date=self.dates[information_idx],
            setup_age_sessions=setup_age,
            pivot=pivot,
            prior_close=prior_close,
            distance_to_pivot_pct=distance_pct,
            readiness_score=readiness,
            current_rs_rating=current_rs,
            dynamic_dryup_ratio=dryup_ratio,
            dryup_source=dryup_source,
            dryup_coverage=dryup_coverage,
            fundamental_score=fundamental_score,
            fundamental_coverage=fundamental_coverage,
            eps_yoy=eps_yoy,
            revenue_yoy=revenue_yoy,
            structure_quality_score=structure,
            tightness_score=tightness,
            prior_advance_score=prior_advance,
            raw_quality_score=raw_quality_score,
            quality_feature_coverage=observed_weight / total_weight,
            walk_forward_quality_score=quality_estimate.value,
            quality_score=exposed_quality_score,
            quality_model_validated=quality_validation.passed,
            fill_probability=fill_estimate.value,
            slate_priority=slate_priority,
            quality_rank=None,
            quality_calibration_count=quality_estimate.count,
            quality_effective_samples=quality_estimate.effective_samples,
            quality_validation_count=quality_validation.count,
            fill_calibration_count=fill_estimate.count,
            fill_effective_samples=fill_estimate.effective_samples,
            context_values=context_values,
        )

    @staticmethod
    def _neutral_tie_key(snapshot: CandidateSnapshot) -> int:
        """Stable session lottery for candidates lacking validated quality.

        Cryptographic hashing avoids Python's process-randomized ``hash`` and
        removes persistent lexical symbol preference.  Every input is known
        before the order session and the result is independent of input order.
        """
        identity = "\x1f".join(
            (
                _day(snapshot.session_date).date().isoformat(),
                snapshot.setup_type,
                snapshot.symbol,
                (
                    "missing"
                    if snapshot.pivot is None
                    else format(snapshot.pivot, ".8f")
                ),
                str(snapshot.setup_age_sessions),
            )
        ).encode("utf-8")
        digest = blake2b(
            identity, digest_size=8, person=b"minervini-v6"
        ).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def rank(
        self,
        active_setups: Mapping[int, object] | Iterable[object],
        session_idx: int,
    ) -> tuple[CandidateSnapshot, ...]:
        """Return candidates in slate order and attach their pure-quality rank.

        The returned order follows ``slate_priority`` because that is what the
        order allocator consumes. ``quality_rank`` deliberately ignores fill
        probability and therefore remains an honest rank of expected R.
        """
        if isinstance(active_setups, Mapping):
            snapshots = [
                self.snapshot(setup, session_idx, setup_key=int(key))
                for key, setup in active_setups.items()
            ]
        else:
            snapshots = [self.snapshot(setup, session_idx) for setup in active_setups]
        quality_order = sorted(
            snapshots,
            key=lambda item: (
                -item.quality_score,
                item.setup_type,
                -item.raw_quality_score if item.quality_model_validated else 0.0,
                self._neutral_tie_key(item)
                if not item.quality_model_validated
                else 0,
                item.symbol,
                item.setup_id if item.setup_id is not None else 2**63 - 1,
                item.setup_key,
            ),
        )
        quality_ranks: dict[int, int] = {}
        previous_quality: float | None = None
        current_rank = 0
        for position, snapshot in enumerate(quality_order, start=1):
            if previous_quality is None or snapshot.quality_score != previous_quality:
                current_rank = position
                previous_quality = snapshot.quality_score
            quality_ranks[id(snapshot)] = current_rank

        # Equal posterior estimates contain no evidence for a cross-class
        # preference. A lexical setup-type tie break would silently create one.
        # Interleave every exact posterior tie by class and rotate the first
        # class by session. Inside an unvalidated class a stable session hash
        # provides an input-order-independent neutral lottery; raw quality and
        # fill may break ties only after quality validation has passed.
        posterior_groups: dict[
            tuple[float, float], list[CandidateSnapshot]
        ] = {}
        for snapshot in snapshots:
            posterior_groups.setdefault(
                (snapshot.slate_priority, snapshot.quality_score), []
            ).append(snapshot)
        slate_order: list[CandidateSnapshot] = []
        for group_number, posterior in enumerate(
            sorted(posterior_groups, key=lambda value: (-value[0], -value[1]))
        ):
            by_type: dict[str, list[CandidateSnapshot]] = {}
            for snapshot in posterior_groups[posterior]:
                by_type.setdefault(snapshot.setup_type, []).append(snapshot)
            for candidates in by_type.values():
                candidates.sort(
                    key=lambda item: (
                        (
                            -item.raw_quality_score
                            if item.quality_model_validated
                            else 0.0
                        ),
                        (
                            -item.fill_probability
                            if item.quality_model_validated
                            else 0.0
                        ),
                        (
                            self._neutral_tie_key(item)
                            if not item.quality_model_validated
                            else 0
                        ),
                        item.symbol,
                        item.setup_id if item.setup_id is not None else 2**63 - 1,
                        item.setup_key,
                    )
                )
            setup_types = sorted(by_type)
            # Calendar-date rotation is stable even when a separate OOS run
            # loads a different amount of warm-up history before this session.
            session_ordinal = _day(self.dates[session_idx]).date().toordinal()
            rotation = (session_ordinal + group_number) % len(setup_types)
            setup_types = setup_types[rotation:] + setup_types[:rotation]
            for index in range(max(len(values) for values in by_type.values())):
                for setup_type in setup_types:
                    candidates = by_type[setup_type]
                    if index < len(candidates):
                        slate_order.append(candidates[index])
        return tuple(
            replace(snapshot, quality_rank=quality_ranks[id(snapshot)])
            for snapshot in slate_order
        )
