"""Causal, deterministic ranking of active pre-session setup candidates.

``CandidateRanker`` is deliberately independent of portfolio state.  For an
order session at index ``t`` it reads dynamic market data only at ``t - 1`` or
from windows ending at ``t - 1``.  Detection-time values on the setup row are
safe fallbacks when a dynamic matrix is unavailable.

The rank is continuous and uses a separate feature-weight profile for each
setup class.  Missing features receive a neutral score instead of silently
becoming either perfect or failed observations; ``feature_coverage`` exposes
how much of the configured weight was actually observed.  Only the explicit
technical/fundamental whitelist below can enter the score.  In particular,
current IBKR taxonomy/group fields and raw institutional/13F counts are not
ranking inputs.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import exp, isfinite, log
from typing import Any

import numpy as np
import pandas as pd


SCORE_FEATURES = frozenset(
    {
        "readiness",
        "rs",
        "dryup",
        "structure",
        "tightness",
        "prior_advance",
        "fundamental",
        "freshness",
    }
)

# All component values are scaled to [0, 100] before these weights are
# applied.  The profiles intentionally express structural differences rather
# than an empirically fitted setup-type return premium.
DEFAULT_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "vcp": {
        "readiness": 0.26,
        "rs": 0.20,
        "dryup": 0.19,
        "structure": 0.18,
        "tightness": 0.09,
        "prior_advance": 0.03,
        "fundamental": 0.03,
        "freshness": 0.02,
    },
    "flat_base": {
        "readiness": 0.30,
        "rs": 0.20,
        "dryup": 0.10,
        "structure": 0.13,
        "tightness": 0.14,
        "prior_advance": 0.05,
        "fundamental": 0.05,
        "freshness": 0.03,
    },
    "power_play": {
        "readiness": 0.24,
        "rs": 0.25,
        "dryup": 0.07,
        "structure": 0.08,
        "tightness": 0.08,
        "prior_advance": 0.22,
        "fundamental": 0.04,
        "freshness": 0.02,
    },
    "tight_shelf": {
        "readiness": 0.30,
        "rs": 0.21,
        "dryup": 0.12,
        "structure": 0.10,
        "tightness": 0.19,
        "prior_advance": 0.03,
        "fundamental": 0.03,
        "freshness": 0.02,
    },
    "default": {
        "readiness": 0.28,
        "rs": 0.22,
        "dryup": 0.15,
        "structure": 0.14,
        "tightness": 0.08,
        "prior_advance": 0.06,
        "fundamental": 0.04,
        "freshness": 0.03,
    },
}


@dataclass(frozen=True)
class CandidateSnapshot:
    """Information available immediately before one order session.

    Higher ``ranking_score`` is better.  ``context_values`` is sorted by name
    and immutable so snapshots remain deterministic and easy to persist as
    ordinary scalar columns chosen by the caller (no JSON is required).
    """

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
    context_score: float
    ranking_score: float
    feature_coverage: float
    context_values: tuple[tuple[str, float | None], ...]

    @property
    def total_rank(self) -> float:
        """Alias describing the score's role when persisting research data."""
        return self.ranking_score

    def context_value(self, name: str) -> float | None:
        """Return one named as-of context value, or ``None`` when absent."""
        return dict(self.context_values).get(name)


class CandidateRanker:
    """Build and order causal snapshots from aligned daily matrices.

    Parameters mirror the simulator's matrix API.  ``close`` is required;
    ``volume``, ``rs_rating`` and named context matrices are optional but, when
    supplied, must have exactly the same dates and symbols.  Named context is
    captured for attribution.  Of it, only ``fundamental_score`` is on the
    score whitelist; ``eps_yoy`` and ``revenue_yoy`` are exposed as context.
    """

    def __init__(
        self,
        dates: pd.DatetimeIndex,
        symbols: pd.Index,
        close: pd.DataFrame,
        *,
        volume: pd.DataFrame | None = None,
        rs_rating: pd.DataFrame | None = None,
        context: Mapping[str, pd.DataFrame] | None = None,
        type_weights: Mapping[str, Mapping[str, float]] | None = None,
        dryup_recent_sessions: int = 5,
        dryup_baseline_sessions: int = 50,
        dryup_zero_ratio: float = 1.25,
        age_half_life_sessions: float = 10.0,
    ) -> None:
        self.dates = pd.DatetimeIndex(dates)
        self.symbols = pd.Index(symbols)
        if not self.dates.is_monotonic_increasing or not self.dates.is_unique:
            raise ValueError("dates must be unique and monotonically increasing")
        if not self.symbols.is_unique:
            raise ValueError("symbols must be unique")

        self.close = self._validate_matrix("close", close)
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
        if not isfinite(dryup_zero_ratio) or dryup_zero_ratio <= 0:
            raise ValueError("dryup_zero_ratio must be finite and positive")
        if not isfinite(age_half_life_sessions) or age_half_life_sessions <= 0:
            raise ValueError("age_half_life_sessions must be finite and positive")
        self.dryup_recent_sessions = int(dryup_recent_sessions)
        self.dryup_baseline_sessions = int(dryup_baseline_sessions)
        self.dryup_zero_ratio = float(dryup_zero_ratio)
        self.age_half_life_sessions = float(age_half_life_sessions)
        self.type_weights = self._validate_weights(type_weights or DEFAULT_TYPE_WEIGHTS)

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
            raise ValueError("type_weights must contain a default profile")
        validated: dict[str, dict[str, float]] = {}
        for setup_type, profile in weights.items():
            unknown = set(profile) - SCORE_FEATURES
            if unknown:
                raise ValueError(
                    "non-whitelisted ranking features: " + ", ".join(sorted(unknown))
                )
            clean: dict[str, float] = {}
            for name, weight in profile.items():
                numeric = float(weight)
                if not isfinite(numeric) or numeric < 0:
                    raise ValueError("ranking weights must be finite and non-negative")
                clean[name] = numeric
            if sum(clean.values()) <= 0:
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
        recent_start = information_idx - self.dryup_recent_sessions + 1
        baseline_end = recent_start
        baseline_start = baseline_end - self.dryup_baseline_sessions
        if recent_start < 0 or baseline_end < 20:
            return None, 0.0
        baseline_start = max(0, baseline_start)
        baseline = pd.to_numeric(
            self.volume.iloc[baseline_start:baseline_end, col], errors="coerce"
        ).to_numpy(dtype=float)
        recent = pd.to_numeric(
            self.volume.iloc[recent_start : information_idx + 1, col], errors="coerce"
        ).to_numpy(dtype=float)
        baseline = baseline[np.isfinite(baseline) & (baseline > 0)]
        recent = recent[np.isfinite(recent) & (recent > 0)]
        baseline_coverage = min(
            1.0, len(baseline) / float(self.dryup_baseline_sessions)
        )
        recent_coverage = min(
            1.0, len(recent) / float(self.dryup_recent_sessions)
        )
        coverage = min(baseline_coverage, recent_coverage)
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

    def snapshot(
        self,
        setup: object,
        session_idx: int,
        *,
        setup_key: int | None = None,
    ) -> CandidateSnapshot:
        """Build one snapshot using no observation newer than ``t - 1``."""
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
        pivot = self._optional_float(self._attr(setup, "pivot"))
        if pivot is not None and pivot <= 0:
            pivot = None
        prior_close = self._matrix_value(self.close, information_idx, col)
        if pivot is None or prior_close is None:
            distance = None
            readiness = None
        else:
            distance = (pivot - prior_close) / pivot
            # A close just below the pivot is ideal.  Being far below the
            # pivot decays over 10%; already being above it decays faster.
            scale = 0.10 if distance >= 0 else 0.02
            readiness = float(np.clip(100.0 * (1.0 - abs(distance) / scale), 0.0, 100.0))

        current_rs = self._matrix_value(self.rs_rating, information_idx, col)
        if current_rs is None:
            current_rs = self._optional_float(self._attr(setup, "rs_rating"))

        dynamic_dryup, dynamic_dryup_coverage = self._dynamic_dryup(
            information_idx, col
        )
        if self.volume is not None:
            # Once a daily volume source is part of the run contract, an
            # incomplete current window is unknown. Falling back to the
            # detect-day value would quietly reintroduce stale ranking context.
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
            if dryup_ratio is not None and dryup_ratio > 0:
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
            return value if value is not None else self._optional_float(self._attr(setup, name))

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
        structure = self._optional_float(self._attr(setup, "structure_quality_score"))
        tightness = self._optional_float(self._attr(setup, "tightness_score"))
        prior_advance = self._optional_float(self._attr(setup, "prior_advance_score"))
        setup_age = session_idx - first_order_idx

        dryup_component = (
            float(np.clip(100.0 * (1.0 - dryup_ratio / self.dryup_zero_ratio), 0.0, 100.0))
            if dryup_ratio is not None
            else None
        )
        fundamental_observed_count = 6.0 * fundamental_coverage
        fundamental_component = (
            self._scaled(fundamental_score, fundamental_observed_count)
            if fundamental_observed_count > 0
            else None
        )
        components = {
            "readiness": readiness,
            "rs": self._scaled(current_rs - 1.0, 98.0) if current_rs is not None else None,
            "dryup": dryup_component,
            "structure": self._scaled(structure, 25.0),
            "tightness": self._scaled(tightness, 20.0),
            "prior_advance": self._scaled(prior_advance, 25.0),
            "fundamental": fundamental_component,
            "freshness": 100.0 * exp(-log(2.0) * setup_age / self.age_half_life_sessions),
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
        profile = self.type_weights.get(setup_type, self.type_weights["default"])
        total_weight = sum(profile.values())
        observed_weight = sum(
            weight * component_coverage.get(name, 0.0)
            for name, weight in profile.items()
        )
        # Missing information gets a neutral prior.  This avoids both the old
        # zero-as-failure behaviour and a renormalisation that could make a
        # sparsely observed candidate look artificially perfect.
        weighted_score = sum(
            weight * effective_component(name)
            for name, weight in profile.items()
        )
        context_weights = {
            "rs": profile.get("rs", 0.0),
            "fundamental": profile.get("fundamental", 0.0),
        }
        context_weight = sum(context_weights.values())
        context_score = (
            sum(
                weight
                * effective_component(name)
                for name, weight in context_weights.items()
            )
            / context_weight
            if context_weight > 0
            else 50.0
        )

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
            distance_to_pivot_pct=distance,
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
            context_score=context_score,
            ranking_score=weighted_score / total_weight,
            feature_coverage=observed_weight / total_weight,
            context_values=context_values,
        )

    def rank(
        self,
        active_setups: Mapping[int, object] | Iterable[object],
        session_idx: int,
    ) -> tuple[CandidateSnapshot, ...]:
        """Return snapshots best-first with stable, data-independent tie breaks."""
        if isinstance(active_setups, Mapping):
            snapshots = [
                self.snapshot(setup, session_idx, setup_key=int(key))
                for key, setup in active_setups.items()
            ]
        else:
            snapshots = [self.snapshot(setup, session_idx) for setup in active_setups]
        return tuple(
            sorted(
                snapshots,
                key=lambda item: (
                    -item.ranking_score,
                    item.setup_type,
                    item.symbol,
                    item.setup_id if item.setup_id is not None else 2**63 - 1,
                    item.setup_key,
                ),
            )
        )
