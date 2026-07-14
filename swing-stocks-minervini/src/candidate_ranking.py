"""Causal candidate quality, fill and slate-priority estimates.

The three public estimates deliberately have different meanings:

``quality_score``
    Setup-type-specific causal posterior expected net R-multiple.  Its raw
    signal uses technical and point-in-time fundamental information only.
    Pivot distance and readiness are explicitly excluded.  In
    ``relative_quality`` mode the posterior determines only the relative slate
    order; its absolute level is never a cash-versus-trade gate.
``fill_probability``
    Setup-type-specific posterior probability that an order touches.  This is
    the only model that may use the readiness/pivot-distance signal.
``slate_priority``
    The effective ``quality_score`` used for ordering.  Fill probability is a
    diagnostic only and can never change slate order.

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

# These profiles create a within-class raw ordering signal.  The resulting
# calibrated posteriors all target the same net R-multiple unit and are
# therefore globally comparable after calibration.
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
    fill_probability: float
    slate_priority: float
    quality_rank: int | None
    quality_calibration_count: int
    quality_effective_samples: float
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

    def reset(self) -> None:
        self.as_of = None
        self.next_label = 0
        self.count = 0
        self.weights.fill(0.0)
        self.weighted_signals.fill(0.0)
        self.weighted_targets.fill(0.0)


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
        neutral_rank_salt: str = "v9-relative-quality-shadow",
        ranking_mode: str = "relative_quality",
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
        if not isinstance(neutral_rank_salt, str):
            raise TypeError("neutral_rank_salt must be a string")
        if not neutral_rank_salt:
            raise ValueError("neutral_rank_salt must not be empty")
        self.neutral_rank_salt = neutral_rank_salt
        if not isinstance(ranking_mode, str):
            raise TypeError("ranking_mode must be a string")
        if ranking_mode not in {"neutral", "relative_quality"}:
            raise ValueError(
                "ranking_mode must be neutral or relative_quality"
            )
        self.ranking_mode = ranking_mode
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
        normalized: dict[str, dict[str, float]] = {}
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
            normalized[str(setup_type)] = clean
        return normalized

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
            state.next_label += 1
            state.count += 1
        state.as_of = information_date
        return state

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
    ) -> _Estimate:
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
        return estimate

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
        quality_estimate = self._quality_estimate(
            setup_type, raw_quality_score, information_date
        )
        fill_estimate = self._fill_estimate(
            setup_type, readiness_signal, information_date
        )
        exposed_quality_score = (
            0.0
            if self.ranking_mode == "neutral"
            else quality_estimate.value
        )
        # Fill probability remains diagnostic-only. Readiness
        # can therefore never improve or damage a candidate's slate order.
        slate_priority = exposed_quality_score
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
            fill_probability=fill_estimate.value,
            slate_priority=slate_priority,
            quality_rank=None,
            quality_calibration_count=quality_estimate.count,
            quality_effective_samples=quality_estimate.effective_samples,
            fill_calibration_count=fill_estimate.count,
            fill_effective_samples=fill_estimate.effective_samples,
            context_values=context_values,
        )

    def _neutral_tie_key(self, snapshot: CandidateSnapshot) -> int:
        """Stable session lottery for neutral candidates and quality ties.

        Cryptographic hashing avoids Python's process-randomized ``hash`` and
        removes persistent lexical symbol preference.  Every input is known
        before the order session and the result is independent of input order.
        """
        identity = "\x1f".join(
            (
                self.neutral_rank_salt,
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
            identity, digest_size=8, person=b"minervini-v9"
        ).digest()
        return int.from_bytes(digest, byteorder="big", signed=False)

    def rank(
        self,
        active_setups: Mapping[int, object] | Iterable[object],
        session_idx: int,
    ) -> tuple[CandidateSnapshot, ...]:
        """Return one global expected-net-R order and its global rank.

        All calibrated posteriors share the same net R-multiple target.  The
        neutral mode uses only the stable salted hash. ``relative_quality``
        sorts globally by expected R, including negative values; fill
        probability is diagnostic and never participates in ordering.
        """
        if isinstance(active_setups, Mapping):
            snapshots = [
                self.snapshot(setup, session_idx, setup_key=int(key))
                for key, setup in active_setups.items()
            ]
        else:
            snapshots = [self.snapshot(setup, session_idx) for setup in active_setups]

        def stable_identity(item: CandidateSnapshot) -> tuple:
            return (
                self._neutral_tie_key(item),
                item.symbol,
                item.setup_id if item.setup_id is not None else 2**63 - 1,
                item.setup_key,
            )

        if self.ranking_mode == "neutral":
            slate_order = sorted(snapshots, key=stable_identity)
        else:
            slate_order = sorted(
                snapshots,
                key=lambda item: (-item.quality_score, *stable_identity(item)),
            )

        ranked: list[CandidateSnapshot] = []
        for position, snapshot in enumerate(slate_order, start=1):
            ranked.append(replace(snapshot, quality_rank=position))
        return tuple(ranked)
