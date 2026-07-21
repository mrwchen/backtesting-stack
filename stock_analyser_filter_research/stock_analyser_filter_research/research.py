from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import ceil
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    EARLY_CUT_COLUMNS,
    EARLY_CUT_FEATURE_GROUPS,
    EARLY_CUT_LANDMARK_DAYS,
    ENTRY_FEATURE_GROUPS,
    IDENTITY_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)


POLICY_KEY_COLUMNS = ("signal_date", *IDENTITY_COLUMNS)


@dataclass(frozen=True)
class ObjectiveSpec:
    decision_family: str
    objective: str
    protected_outcome: str
    landmark_day: int | None = None

    @property
    def date_column(self) -> str:
        return (
            "signal_date" if self.decision_family == "entry_filter" else "landmark_date"
        )

    @property
    def label_end_column(self) -> str:
        return (
            "forward_5d_label_end_date"
            if self.decision_family == "entry_filter"
            else "horizon_end_date"
        )


ENTRY_SPECS = (
    ObjectiveSpec("entry_filter", "weak_5d", "strong_first_5d"),
    ObjectiveSpec("entry_filter", "loss_first_5d", "strong_first_5d"),
)


def early_specs(landmark_day: int) -> tuple[ObjectiveSpec, ...]:
    return (
        ObjectiveSpec(
            "early_cut", "stagnant_to_day5", "strong_first_to_day5", landmark_day
        ),
        ObjectiveSpec(
            "early_cut", "loss_first_to_day5", "strong_first_to_day5", landmark_day
        ),
    )


SEQUENTIAL_EARLY_SPEC = ObjectiveSpec(
    "early_cut", "bad_to_day5", "strong_first_to_day5", 1
)


@dataclass(frozen=True)
class ConditionTemplate:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    quantile: float
    minimum_threshold: float | None = None


@dataclass(frozen=True)
class Condition:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    threshold: float
    quantile: float
    threshold_fit_end_date: date | None = None

    @property
    def text(self) -> str:
        symbol = "<=" if self.operator == "le" else ">="
        return f"{self.feature_name} {symbol} {self.threshold:.10g}"

    def matches(self, frame: pd.DataFrame) -> pd.Series:
        if self.feature_name not in frame:
            return pd.Series(False, index=frame.index, dtype=bool)
        values = _finite_numeric(frame, self.feature_name)
        if self.operator == "le":
            return values.notna() & values.le(self.threshold)
        if self.operator == "ge":
            return values.notna() & values.ge(self.threshold)
        raise ValueError(f"unsupported condition operator {self.operator!r}")


@dataclass(frozen=True)
class PatternTemplate:
    rule_id: str
    feature_group: str
    pattern_name: str
    clauses: tuple[ConditionTemplate, ...]


@dataclass(frozen=True)
class CompositeCondition:
    rule_id: str
    feature_group: str
    pattern_name: str
    clauses: tuple[Condition, ...]

    @property
    def text(self) -> str:
        return f"{self.pattern_name} (" + " AND ".join(
            clause.text for clause in self.clauses
        ) + ")"

    @property
    def threshold_fit_end_date(self) -> date | None:
        dates = [
            clause.threshold_fit_end_date
            for clause in self.clauses
            if clause.threshold_fit_end_date is not None
        ]
        return max(dates) if dates else None

    def matches(self, frame: pd.DataFrame) -> pd.Series:
        result = pd.Series(True, index=frame.index, dtype=bool)
        if not self.clauses:
            return pd.Series(False, index=frame.index, dtype=bool)
        for clause in self.clauses:
            result &= clause.matches(frame)
        return result


CandidateTemplate = ConditionTemplate | PatternTemplate
ExecutableCondition = Condition | CompositeCondition


@dataclass(frozen=True)
class WalkForwardFold:
    year: int
    start_date: date
    end_date: date
    threshold_fit_end_date: date


@dataclass
class FoldResult:
    fold: WalkForwardFold
    conditions: tuple[ExecutableCondition, ...]
    metrics: dict[str, object]


@dataclass
class CandidateEvaluation:
    templates: tuple[CandidateTemplate, ...]
    fold_results: tuple[FoldResult, ...]
    pooled_metrics: dict[str, object]
    final_conditions: tuple[ExecutableCondition, ...]
    development_metrics: dict[str, object]
    passes_development_gates: bool
    passes_stability_gates: bool
    stability: dict[str, object]
    selection_score: float | None

    @property
    def rule_id(self) -> str:
        return (
            "__OR__".join(template.rule_id for template in self.templates)
            or "NO_FILTER"
        )


@dataclass
class ObjectiveSelection:
    spec: ObjectiveSpec
    selected: CandidateEvaluation
    candidates: tuple[CandidateEvaluation, ...]
    prefixes: tuple[CandidateEvaluation, ...]


@dataclass
class SelectedSummary:
    entry: dict[str, ObjectiveSelection]
    early_cut: dict[int, dict[str, ObjectiveSelection]]
    entry_prefix_lengths: dict[str, int]
    early_cut_prefix_lengths: dict[int, dict[str, int]]

    @property
    def condition_count(self) -> int:
        return sum(self.entry_prefix_lengths.values()) + sum(
            sum(values.values()) for values in self.early_cut_prefix_lengths.values()
        )

    @property
    def summary_text(self) -> str:
        parts: list[str] = []
        for objective, selection in self.entry.items():
            length = self.entry_prefix_lengths.get(objective, 0)
            text = (
                " OR ".join(
                    condition.text
                    for condition in selection.selected.final_conditions[:length]
                )
                or "no filter"
            )
            parts.append(f"entry/{objective}: {text}")
        for day in sorted(self.early_cut):
            for objective, selection in self.early_cut[day].items():
                length = self.early_cut_prefix_lengths.get(day, {}).get(objective, 0)
                text = (
                    " OR ".join(
                        condition.text
                        for condition in selection.selected.final_conditions[:length]
                    )
                    or "no filter"
                )
                parts.append(f"early/day{day}/{objective}: {text}")
        return "; ".join(parts)


@dataclass
class ResearchResult:
    signals: pd.DataFrame
    early_cuts: pd.DataFrame
    rules: pd.DataFrame
    selected: SelectedSummary
    selected_condition_count: int
    selected_text: str


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype("boolean").fillna(False).astype(bool)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _metrics_from_counts(counts: Mapping[str, int]) -> dict[str, object]:
    result = {name: int(value) for name, value in counts.items()}
    population = result["population_count"]
    sample = result["sample_count"]
    matched = result["matched_count"]
    matched_labeled = result["matched_labeled_count"]
    objective = result["objective_count"]
    protected = result["protected_count"]
    matched_objective = result["matched_objective_count"]
    matched_protected = result["matched_protected_count"]
    retained_count = sample - matched_labeled
    population_rate = _safe_ratio(objective, sample)
    matched_rate = _safe_ratio(matched_objective, matched_labeled)
    protected_rejection = _safe_ratio(matched_protected, protected)
    result.update(
        {
            "label_coverage_rate": _safe_ratio(sample, population),
            "matched_label_coverage_rate": _safe_ratio(matched_labeled, matched),
            "match_rate": _safe_ratio(matched_labeled, sample),
            "population_objective_rate": population_rate,
            "matched_objective_rate": matched_rate,
            "objective_capture_rate": _safe_ratio(matched_objective, objective),
            "objective_lift": (
                None
                if population_rate in (None, 0) or matched_rate is None
                else matched_rate / population_rate
            ),
            "protected_rejection_rate": protected_rejection,
            "protected_retention_rate": (
                None if protected_rejection is None else 1.0 - protected_rejection
            ),
            "retained_objective_rate": _safe_ratio(
                objective - matched_objective, retained_count
            ),
            "retained_protected_rate": _safe_ratio(
                protected - matched_protected, retained_count
            ),
        }
    )
    return result


def objective_metrics(
    frame: pd.DataFrame,
    matched: pd.Series,
    objective: str,
    protected_outcome: str,
    label_available: pd.Series | None = None,
) -> dict[str, object]:
    matched = (
        matched.reindex(frame.index, fill_value=False)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    labeled = frame[objective].notna() & frame[protected_outcome].notna()
    if label_available is not None:
        labeled &= (
            label_available.reindex(frame.index, fill_value=False)
            .astype("boolean")
            .fillna(False)
            .astype(bool)
        )
    objective_values = _truthy(frame[objective])
    protected_values = _truthy(frame[protected_outcome])
    matched_labeled = matched & labeled
    counts = {
        "population_count": len(frame),
        "sample_count": int(labeled.sum()),
        "unlabeled_count": int((~labeled).sum()),
        "matched_count": int(matched.sum()),
        "matched_labeled_count": int(matched_labeled.sum()),
        "matched_unlabeled_count": int((matched & ~labeled).sum()),
        "objective_count": int((labeled & objective_values).sum()),
        "protected_count": int((labeled & protected_values).sum()),
        "matched_objective_count": int((matched_labeled & objective_values).sum()),
        "matched_protected_count": int((matched_labeled & protected_values).sum()),
    }
    return _metrics_from_counts(counts)


def _aggregate_metrics(items: Iterable[Mapping[str, object]]) -> dict[str, object]:
    names = (
        "population_count",
        "sample_count",
        "unlabeled_count",
        "matched_count",
        "matched_labeled_count",
        "matched_unlabeled_count",
        "objective_count",
        "protected_count",
        "matched_objective_count",
        "matched_protected_count",
    )
    counts = {name: 0 for name in names}
    for item in items:
        for name in names:
            counts[name] += int(item[name])
    return _metrics_from_counts(counts)


def _baseline_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return _metrics_from_counts(
        {
            "population_count": int(metrics["population_count"]),
            "sample_count": int(metrics["sample_count"]),
            "unlabeled_count": int(metrics["unlabeled_count"]),
            "matched_count": 0,
            "matched_labeled_count": 0,
            "matched_unlabeled_count": 0,
            "objective_count": int(metrics["objective_count"]),
            "protected_count": int(metrics["protected_count"]),
            "matched_objective_count": 0,
            "matched_protected_count": 0,
        }
    )


def _combined_mask(
    frame: pd.DataFrame, conditions: Iterable[ExecutableCondition]
) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for condition in conditions:
        result |= condition.matches(frame)
    return result


def build_candidate_templates(
    decision_family: str,
    landmark_day: int | None = None,
    quantile_count: int = 20,
) -> tuple[CandidateTemplate, ...]:
    if decision_family == "entry_filter":
        groups = ENTRY_FEATURE_GROUPS
        prefix = "ENTRY"
    elif decision_family == "early_cut":
        if landmark_day not in EARLY_CUT_LANDMARK_DAYS:
            raise ValueError("early-cut templates require landmark day 1, 2, or 3")
        groups = EARLY_CUT_FEATURE_GROUPS
        prefix = f"EARLY_D{landmark_day}"
    else:
        raise ValueError(f"unsupported decision family {decision_family!r}")
    grid = np.linspace(0.0, 1.0, quantile_count + 1)[1:-1]
    lower = tuple(float(value) for value in grid if value <= 0.30 + 1e-12)
    upper = tuple(float(value) for value in grid if value >= 0.70 - 1e-12)
    templates: list[CandidateTemplate] = []
    for group, features in groups.items():
        for feature in features:
            for quantile, operator in (
                *((value, "le") for value in lower),
                *((value, "ge") for value in upper),
            ):
                side = "LE" if operator == "le" else "GE"
                tag = int(round(quantile * 100))
                templates.append(
                    ConditionTemplate(
                        f"{prefix}_{group}_{feature}_{side}_Q{tag:02d}",
                        group,
                        feature,
                        operator,
                        quantile,
                    )
                )
    if decision_family == "entry_filter":
        templates.extend(_entry_pattern_templates())
    return tuple(sorted(templates, key=lambda item: item.rule_id))


def _entry_pattern_templates() -> tuple[PatternTemplate, ...]:
    def pattern(
        name: str,
        clauses: tuple[tuple[str, str, float, float | None], ...],
    ) -> PatternTemplate:
        rule_id = f"ENTRY_D_PATTERN_{name}"
        return PatternTemplate(
            rule_id,
            "D",
            name.lower(),
            tuple(
                ConditionTemplate(
                    f"{rule_id}_C{number}",
                    "D",
                    feature_name,
                    operator,
                    quantile,
                    minimum_threshold,
                )
                for number, (
                    feature_name,
                    operator,
                    quantile,
                    minimum_threshold,
                ) in enumerate(clauses, start=1)
            ),
        )

    return (
        pattern(
            "FLAT_BASE",
            (
                ("prior_base_width_20_pct", "le", 0.30, None),
                ("prior_trend_efficiency_20", "le", 0.30, None),
                ("prior_range_compression_10_vs_10_ratio", "le", 0.30, None),
            ),
        ),
        pattern(
            "ORDERED_UPTREND",
            (
                ("prior_trend_slope_20_pct_per_session", "ge", 0.70, None),
                ("prior_trend_r2_20", "ge", 0.70, None),
                ("prior_positive_return_share_20", "ge", 0.70, None),
                ("prior_max_drawdown_21d_pct", "ge", 0.70, None),
            ),
        ),
        pattern(
            "PULLBACK_FROM_HIGH",
            (
                ("prior_pullback_from_40d_high_pct", "le", 0.30, None),
                ("prior_peak_age_40_sessions", "ge", 0.30, None),
                ("signal_close_vs_prior_20d_high_pct", "ge", 0.70, None),
            ),
        ),
        pattern(
            "V_RECOVERY",
            (
                ("prior_drawdown_to_trough_40_pct", "le", 0.30, None),
                ("prior_v_recovery_fraction_40", "ge", 0.70, None),
                ("prior_trough_age_40_sessions", "ge", 0.30, None),
            ),
        ),
        pattern(
            "VOLUME_DRY_UP_BREAKOUT",
            (
                ("prior_volume_sma5_vs21_ratio", "le", 0.20, None),
                ("prior_range_compression_10_vs_10_ratio", "le", 0.30, None),
                ("adjusted_volume_vs_sma21_prior_ratio", "ge", 0.70, None),
                ("signal_close_vs_prior_20d_high_pct", "ge", 0.70, None),
            ),
        ),
        pattern(
            "DISTRIBUTION_TOP",
            (
                ("prior_distribution_day_count_20", "ge", 0.80, 1.0),
                ("prior_failed_breakout_count_20", "ge", 0.80, 1.0),
                ("prior_close_vs_63d_high_pct", "ge", 0.70, None),
            ),
        ),
    )


def build_walk_forward_folds(cfg: Config) -> tuple[WalkForwardFold, ...]:
    folds: list[WalkForwardFold] = []
    for year in range(cfg.walk_forward_first_year, cfg.validation_end_date.year + 1):
        start = date(year, 1, 1)
        end = min(date(year, 12, 31), cfg.validation_end_date)
        if start <= end:
            folds.append(WalkForwardFold(year, start, end, start - timedelta(days=1)))
    return tuple(folds)


def fit_template(
    frame: pd.DataFrame,
    template: ConditionTemplate,
    date_column: str,
    fit_end_date: date,
) -> Condition | None:
    dates = pd.to_datetime(frame[date_column], errors="coerce").dt.date
    values = _finite_numeric(
        frame.loc[dates.le(fit_end_date)], template.feature_name
    ).dropna()
    if values.empty:
        return None
    threshold = float(
        np.quantile(values.to_numpy(), template.quantile, method="nearest")
    )
    if not np.isfinite(threshold):
        return None
    if template.minimum_threshold is not None:
        threshold = max(threshold, template.minimum_threshold)
    return Condition(
        template.rule_id,
        template.feature_group,
        template.feature_name,
        template.operator,
        threshold,
        template.quantile,
        fit_end_date,
    )


def _universe(frame: pd.DataFrame, spec: ObjectiveSpec) -> pd.DataFrame:
    result = frame
    if spec.decision_family == "early_cut":
        result = result.loc[
            pd.to_numeric(result["landmark_day"], errors="coerce").eq(spec.landmark_day)
        ]
        result = result.loc[_truthy(result["eligible_at_landmark"])]
    return result.copy()


def _period_frame(
    frame: pd.DataFrame, spec: ObjectiveSpec, start: date | None, end: date | None
) -> pd.DataFrame:
    dates = pd.to_datetime(frame[spec.date_column], errors="coerce").dt.date
    mask = dates.notna()
    if start is not None:
        mask &= dates.ge(start)
    if end is not None:
        mask &= dates.le(end)
    return frame.loc[mask]


def _available(frame: pd.DataFrame, spec: ObjectiveSpec, through: date) -> pd.Series:
    ends = pd.to_datetime(frame[spec.label_end_column], errors="coerce").dt.date
    result = ends.notna() & ends.le(through)
    if spec.decision_family == "early_cut":
        result &= _truthy(frame["full_outcome_available"])
    return result


def _development_gates(metrics: Mapping[str, object], cfg: Config) -> bool:
    sample = int(metrics["sample_count"])
    matched = int(metrics["matched_labeled_count"])
    minimum = max(50, int(ceil(sample * cfg.min_candidate_match_pct)))
    return bool(
        sample > 0
        and matched >= minimum
        and metrics["match_rate"] is not None
        and float(metrics["match_rate"]) <= cfg.max_candidate_match_pct
        and metrics["protected_retention_rate"] is not None
        and float(metrics["protected_retention_rate"])
        >= cfg.min_protected_retention_pct
        and metrics["matched_label_coverage_rate"] is not None
        and float(metrics["matched_label_coverage_rate"])
        >= cfg.min_matched_label_coverage_pct
        and metrics["objective_lift"] is not None
        and float(metrics["objective_lift"]) >= cfg.min_objective_lift
        and metrics["objective_capture_rate"] is not None
        and float(metrics["objective_capture_rate"]) > 0
    )


def _safety_gates(metrics: Mapping[str, object], cfg: Config) -> bool:
    return bool(
        int(metrics["sample_count"]) > 0
        and metrics["match_rate"] is not None
        and float(metrics["match_rate"]) <= cfg.max_candidate_match_pct
        and metrics["protected_retention_rate"] is not None
        and float(metrics["protected_retention_rate"])
        >= cfg.min_protected_retention_pct
        and metrics["matched_label_coverage_rate"] is not None
        and float(metrics["matched_label_coverage_rate"])
        >= cfg.min_matched_label_coverage_pct
    )


def _selection_score(metrics: Mapping[str, object]) -> float | None:
    capture = metrics.get("objective_capture_rate")
    protected_rejection = metrics.get("protected_rejection_rate")
    if capture is None or protected_rejection is None:
        return None
    return float(capture) - float(protected_rejection)


def _stability(
    folds: Iterable[FoldResult], cfg: Config
) -> tuple[bool, dict[str, object]]:
    eligible = [
        item.metrics
        for item in folds
        if int(item.metrics["sample_count"]) >= cfg.min_fold_sample_count
        and int(item.metrics["objective_count"]) >= cfg.min_fold_objective_count
    ]
    lifts = [
        float(item["objective_lift"])
        for item in eligible
        if item["objective_lift"] is not None
    ]
    positive = sum(
        item["objective_lift"] is not None
        and float(item["objective_lift"]) >= cfg.min_fold_objective_lift
        for item in eligible
    )
    fraction = _safe_ratio(positive, len(eligible))
    retentions = [
        float(item["protected_retention_rate"])
        for item in eligible
        if item["protected_retention_rate"] is not None
    ]
    match_rates = [
        float(item["match_rate"]) for item in eligible if item["match_rate"] is not None
    ]
    passes = bool(
        len(eligible) >= cfg.min_walk_forward_folds
        and fraction is not None
        and fraction >= cfg.min_stable_fold_fraction
        and len(retentions) == len(eligible)
        and min(retentions) >= cfg.min_fold_protected_retention_pct
        and len(match_rates) == len(eligible)
        and max(match_rates) <= cfg.max_fold_match_pct
    )
    stats: dict[str, object] = {
        "eligible_fold_count": len(eligible),
        "positive_lift_fold_count": positive,
        "positive_lift_fold_fraction": fraction,
        "median_fold_objective_lift": float(np.median(lifts)) if lifts else None,
        "min_fold_objective_lift": min(lifts) if lifts else None,
        "min_fold_protected_retention_rate": min(retentions) if retentions else None,
        "max_fold_match_rate": max(match_rates) if match_rates else None,
    }
    return passes, stats


def _fold_safety_gates(folds: Iterable[FoldResult], cfg: Config) -> bool:
    """Apply only target-neutral match and protected-outcome fold gates."""
    eligible = [
        item.metrics
        for item in folds
        if int(item.metrics["sample_count"]) >= cfg.min_fold_sample_count
    ]
    retentions = [
        float(item["protected_retention_rate"])
        for item in eligible
        if item["protected_retention_rate"] is not None
    ]
    match_rates = [
        float(item["match_rate"])
        for item in eligible
        if item["match_rate"] is not None
    ]
    return bool(
        len(eligible) >= cfg.min_walk_forward_folds
        and len(retentions) == len(eligible)
        and min(retentions) >= cfg.min_fold_protected_retention_pct
        and len(match_rates) == len(eligible)
        and max(match_rates) <= cfg.max_fold_match_pct
    )


def _prepared_metrics(
    matched: pd.Series,
    labeled: pd.Series,
    objective_values: pd.Series,
    protected_values: pd.Series,
) -> dict[str, object]:
    matched_labeled = matched & labeled
    return _metrics_from_counts(
        {
            "population_count": len(labeled),
            "sample_count": int(labeled.sum()),
            "unlabeled_count": int((~labeled).sum()),
            "matched_count": int(matched.sum()),
            "matched_labeled_count": int(matched_labeled.sum()),
            "matched_unlabeled_count": int((matched & ~labeled).sum()),
            "objective_count": int((labeled & objective_values).sum()),
            "protected_count": int((labeled & protected_values).sum()),
            "matched_objective_count": int((matched_labeled & objective_values).sum()),
            "matched_protected_count": int((matched_labeled & protected_values).sum()),
        }
    )


class _Evaluator:
    def __init__(
        self,
        frame: pd.DataFrame,
        spec: ObjectiveSpec,
        templates: tuple[CandidateTemplate, ...],
        cfg: Config,
    ):
        self.frame = _universe(frame, spec)
        self.spec = spec
        self.templates = templates
        self.atomic_templates = tuple(
            clause
            for template in templates
            for clause in (
                template.clauses
                if isinstance(template, PatternTemplate)
                else (template,)
            )
        )
        self.cfg = cfg
        self.folds = build_walk_forward_folds(cfg)
        self._fit_cache: dict[tuple[int | str, str], Condition | None] = {}
        self._compiled_cache: dict[
            tuple[int | str, str], ExecutableCondition | None
        ] = {}
        self._eval_cache: dict[tuple[str, ...], CandidateEvaluation] = {}
        self._dates = pd.to_datetime(
            self.frame[self.spec.date_column], errors="coerce"
        ).dt.date
        self._numeric_cache: dict[str, pd.Series] = {}
        self._threshold_cache: dict[tuple[int | str, str], dict[float, float]] = {}
        self._fold_data: dict[
            int, tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]
        ] = {}
        self._match_cache: dict[tuple[int, str], pd.Series] = {}
        for fold in self.folds:
            evaluation = _period_frame(
                self.frame, self.spec, fold.start_date, fold.end_date
            )
            labeled = (
                evaluation[self.spec.objective].notna()
                & evaluation[self.spec.protected_outcome].notna()
                & _available(evaluation, self.spec, fold.end_date)
            )
            self._fold_data[fold.year] = (
                evaluation,
                labeled,
                _truthy(evaluation[self.spec.objective]),
                _truthy(evaluation[self.spec.protected_outcome]),
            )

    def _fit(
        self, template: ConditionTemplate, fold: WalkForwardFold | None
    ) -> Condition | None:
        key: tuple[int | str, str] = (
            (fold.year if fold else "final"),
            template.rule_id,
        )
        if key not in self._fit_cache:
            end = fold.threshold_fit_end_date if fold else self.cfg.validation_end_date
            period_key: int | str = fold.year if fold else "final"
            threshold_key = (period_key, template.feature_name)
            if threshold_key not in self._threshold_cache:
                if template.feature_name not in self._numeric_cache:
                    self._numeric_cache[template.feature_name] = _finite_numeric(
                        self.frame, template.feature_name
                    )
                values = (
                    self._numeric_cache[template.feature_name]
                    .loc[self._dates.le(end)]
                    .dropna()
                )
                feature_templates = tuple(
                    item
                    for item in self.atomic_templates
                    if item.feature_name == template.feature_name
                )
                quantiles = tuple(sorted({item.quantile for item in feature_templates}))
                if values.empty or not quantiles:
                    thresholds: dict[float, float] = {}
                else:
                    calculated = np.atleast_1d(
                        np.quantile(values.to_numpy(), quantiles, method="nearest")
                    )
                    thresholds = {
                        quantile: float(value)
                        for quantile, value in zip(quantiles, calculated)
                        if np.isfinite(value)
                    }
                self._threshold_cache[threshold_key] = thresholds
            threshold = self._threshold_cache[threshold_key].get(template.quantile)
            if threshold is not None and template.minimum_threshold is not None:
                threshold = max(threshold, template.minimum_threshold)
            self._fit_cache[key] = (
                None
                if threshold is None
                else Condition(
                    template.rule_id,
                    template.feature_group,
                    template.feature_name,
                    template.operator,
                    threshold,
                    template.quantile,
                    end,
                )
            )
        return self._fit_cache[key]

    def _compile(
        self, template: CandidateTemplate, fold: WalkForwardFold | None
    ) -> ExecutableCondition | None:
        key: tuple[int | str, str] = (
            (fold.year if fold else "final"),
            template.rule_id,
        )
        if key not in self._compiled_cache:
            if isinstance(template, ConditionTemplate):
                compiled: ExecutableCondition | None = self._fit(template, fold)
            else:
                clauses = tuple(self._fit(clause, fold) for clause in template.clauses)
                compiled = (
                    None
                    if any(clause is None for clause in clauses)
                    else CompositeCondition(
                        template.rule_id,
                        template.feature_group,
                        template.pattern_name,
                        tuple(
                            clause for clause in clauses if clause is not None
                        ),
                    )
                )
            self._compiled_cache[key] = compiled
        return self._compiled_cache[key]

    def evaluate(
        self, templates: tuple[CandidateTemplate, ...]
    ) -> CandidateEvaluation:
        key = tuple(item.rule_id for item in templates)
        if key in self._eval_cache:
            return self._eval_cache[key]
        fold_results: list[FoldResult] = []
        for fold in self.folds:
            evaluation, labeled, objective_values, protected_values = self._fold_data[
                fold.year
            ]
            conditions = tuple(
                condition
                for template in templates
                if (condition := self._compile(template, fold)) is not None
            )
            matched = pd.Series(False, index=evaluation.index, dtype=bool)
            for condition in conditions:
                match_key = (fold.year, condition.rule_id)
                if match_key not in self._match_cache:
                    self._match_cache[match_key] = condition.matches(evaluation)
                matched |= self._match_cache[match_key]
            metrics = _prepared_metrics(
                matched, labeled, objective_values, protected_values
            )
            fold_results.append(FoldResult(fold, conditions, metrics))
        pooled = _aggregate_metrics(item.metrics for item in fold_results)
        final_conditions = tuple(
            condition
            for template in templates
            if (condition := self._compile(template, None)) is not None
        )
        stability_pass, stability = _stability(fold_results, self.cfg)
        # Selection is exclusively based on causal walk-forward folds.  The
        # fixed-threshold development metric is emitted as a diagnostic only.
        passes_development = _development_gates(pooled, self.cfg)
        result = CandidateEvaluation(
            templates,
            tuple(fold_results),
            pooled,
            final_conditions,
            {},
            passes_development,
            stability_pass,
            stability,
            _selection_score(pooled),
        )
        self._eval_cache[key] = result
        return result


def _rank(candidate: CandidateEvaluation) -> tuple[object, ...]:
    metrics = candidate.pooled_metrics
    stability = candidate.stability

    def desc(value: object) -> float:
        return float("inf") if value is None else -float(value)

    return (
        desc(candidate.selection_score),
        desc(metrics["objective_capture_rate"]),
        desc(stability["positive_lift_fold_fraction"]),
        desc(stability["median_fold_objective_lift"]),
        desc(metrics["protected_retention_rate"]),
        desc(metrics["objective_lift"]),
        float(metrics["match_rate"] if metrics["match_rate"] is not None else 1.0),
        candidate.rule_id,
    )


def select_objective(
    frame: pd.DataFrame,
    spec: ObjectiveSpec,
    templates: tuple[CandidateTemplate, ...],
    cfg: Config,
) -> ObjectiveSelection:
    evaluator = _Evaluator(frame, spec, templates, cfg)
    empty = evaluator.evaluate(())
    singles = [evaluator.evaluate((template,)) for template in templates]
    eligible_singles = [
        item
        for item in singles
        if item.passes_development_gates and item.passes_stability_gates
    ]
    eligible_singles.sort(key=_rank)
    beam = eligible_singles[: cfg.rule_search_beam_width]
    pairs: list[CandidateEvaluation] = []
    seen: set[tuple[str, str]] = set()
    single_by_id = {item.templates[0].rule_id: item for item in singles}
    pair_bases = beam if cfg.max_conditions_per_objective >= 2 else []
    for base in pair_bases:
        first = base.templates[0]
        for added_evaluation in beam:
            added = added_evaluation.templates[0]
            if added.rule_id == first.rule_id or (
                isinstance(added, ConditionTemplate)
                and isinstance(first, ConditionTemplate)
                and added.feature_name == first.feature_name
                and added.operator == first.operator
            ):
                continue
            pair_key = tuple(sorted((first.rule_id, added.rule_id)))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            other = single_by_id[added.rule_id]
            first_eval, second_eval = (
                (base, other) if _rank(base) <= _rank(other) else (other, base)
            )
            if not (
                first_eval.passes_development_gates
                and first_eval.passes_stability_gates
            ):
                continue
            pair = evaluator.evaluate(
                (first_eval.templates[0], second_eval.templates[0])
            )
            base_score = first_eval.selection_score
            pair_score = pair.selection_score
            if (
                pair.passes_development_gates
                and pair.passes_stability_gates
                and base_score is not None
                and pair_score is not None
                and pair_score
                >= base_score + cfg.min_selection_score_improvement
            ):
                pairs.append(pair)
    eligible = [*eligible_singles, *pairs]
    selected = min(eligible, key=_rank) if eligible else empty
    prefixes: list[CandidateEvaluation] = [empty]
    if selected.templates:
        prefixes.append(evaluator.evaluate((selected.templates[0],)))
    if len(selected.templates) == 2:
        prefixes.append(selected)
    candidates = tuple(sorted([*singles, *pairs], key=_rank))
    return ObjectiveSelection(spec, selected, candidates, tuple(prefixes))


def _choose_prefixes(
    frame: pd.DataFrame,
    selections: Mapping[str, ObjectiveSelection],
    cfg: Config,
    *,
    require_cross_objective_lift: bool = True,
) -> dict[str, int]:
    objectives = tuple(selections)
    best_rank: tuple[object, ...] | None = None
    best = {objective: 0 for objective in objectives}
    ranges = [range(len(selections[objective].prefixes)) for objective in objectives]
    for lengths in __import__("itertools").product(*ranges):
        total_conditions = sum(
            len(selections[objective].prefixes[length].templates)
            for objective, length in zip(objectives, lengths)
        )
        fold_metrics: dict[str, list[dict[str, object]]] = {
            objective: [] for objective in objectives
        }
        first_selection = selections[objectives[0]]
        for fold_index, fold_result in enumerate(
            first_selection.prefixes[0].fold_results
        ):
            fold = fold_result.fold
            first_spec = first_selection.spec
            evaluation = _period_frame(
                _universe(frame, first_spec),
                first_spec,
                fold.start_date,
                fold.end_date,
            )
            union = pd.Series(False, index=evaluation.index, dtype=bool)
            for objective, length in zip(objectives, lengths):
                prefix = selections[objective].prefixes[length]
                conditions = prefix.fold_results[fold_index].conditions
                union |= _combined_mask(evaluation, conditions)
            for objective in objectives:
                spec = selections[objective].spec
                fold_metrics[objective].append(
                    objective_metrics(
                        evaluation,
                        union,
                        spec.objective,
                        spec.protected_outcome,
                        _available(evaluation, spec, fold.end_date),
                    )
                )
        metrics_by_objective = [
            _aggregate_metrics(fold_metrics[objective]) for objective in objectives
        ]
        safe = total_conditions == 0
        if total_conditions:
            pooled_safe = all(
                _safety_gates(metrics, cfg) for metrics in metrics_by_objective
            )
            stable = True
            for objective in objectives:
                reference_folds = selections[objective].prefixes[0].fold_results
                union_folds = tuple(
                    FoldResult(reference.fold, (), metrics)
                    for reference, metrics in zip(
                        reference_folds, fold_metrics[objective]
                    )
                )
                stable &= (
                    _stability(union_folds, cfg)[0]
                    if require_cross_objective_lift
                    else _fold_safety_gates(union_folds, cfg)
                )
            safe = pooled_safe and stable
        if not safe:
            continue
        captures = [
            float(item["objective_capture_rate"] or 0.0)
            for item in metrics_by_objective
        ]
        retentions = [
            float(item["protected_retention_rate"] or 0.0)
            for item in metrics_by_objective
        ]
        match_rates = [
            float(item["match_rate"] or 0.0) for item in metrics_by_objective
        ]
        rank = (
            -float(np.mean(captures)),
            -min(retentions),
            max(match_rates),
            total_conditions,
            tuple(lengths),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = dict(zip(objectives, lengths))
    return best


def _gate_refit_prefixes(
    frame: pd.DataFrame,
    selections: Mapping[str, ObjectiveSelection],
    prefix_lengths: Mapping[str, int],
    cfg: Config,
) -> dict[str, int]:
    """Keep only prefixes whose final, fixed thresholds still pass all gates.

    Walk-forward selection uses thresholds fitted before each fold.  The final
    conditions are subsequently refitted through ``validation_end_date`` and
    can therefore have materially different match and protected-rejection
    rates.  Only the development and validation scopes may veto that refit;
    diagnostic and holdout data are deliberately not inspected here.
    """
    objectives = tuple(selections)
    if not objectives:
        return {}
    capped = {
        objective: min(
            max(int(prefix_lengths.get(objective, 0)), 0),
            len(selections[objective].prefixes) - 1,
        )
        for objective in objectives
    }
    combinations = list(
        __import__("itertools").product(
            *(range(capped[objective] + 1) for objective in objectives)
        )
    )
    combinations.sort(
        key=lambda lengths: (
            -sum(
                len(selections[objective].prefixes[length].templates)
                for objective, length in zip(objectives, lengths)
            ),
            tuple(-length for length in lengths),
        )
    )
    scopes = (
        (cfg.signal_start_date, cfg.validation_end_date),
        (cfg.discovery_end_date + timedelta(days=1), cfg.validation_end_date),
    )
    for lengths in combinations:
        chosen = {
            objective: selections[objective].prefixes[length]
            for objective, length in zip(objectives, lengths)
        }
        valid = True
        for objective, candidate in chosen.items():
            if not candidate.templates:
                continue
            if len(candidate.final_conditions) != len(candidate.templates):
                valid = False
                break
            spec = selections[objective].spec
            for start, end in scopes:
                _, metrics = _scope_metrics(
                    frame,
                    spec,
                    candidate.final_conditions,
                    start,
                    end,
                    cfg.validation_end_date,
                )
                if not _development_gates(metrics, cfg):
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        union_conditions = _unique_conditions(
            condition
            for candidate in chosen.values()
            for condition in candidate.final_conditions
        )
        if union_conditions:
            for objective in objectives:
                spec = selections[objective].spec
                for start, end in scopes:
                    _, metrics = _scope_metrics(
                        frame,
                        spec,
                        union_conditions,
                        start,
                        end,
                        cfg.validation_end_date,
                    )
                    if not _safety_gates(metrics, cfg):
                        valid = False
                        break
                if not valid:
                    break
        if valid:
            return dict(zip(objectives, lengths))
    return {objective: 0 for objective in objectives}


def _policy_keys(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        list(frame.loc[:, POLICY_KEY_COLUMNS].itertuples(index=False, name=None)),
        index=frame.index,
        dtype="object",
    )


def _policy_period_mask(
    frame: pd.DataFrame, start: date | None, end: date | None
) -> pd.Series:
    dates = pd.to_datetime(frame["landmark_date"], errors="coerce").dt.date
    mask = dates.notna()
    if start is not None:
        mask &= dates.ge(start)
    if end is not None:
        mask &= dates.le(end)
    return mask


def _sequential_policy_state(
    frame: pd.DataFrame,
    conditions_by_day: Mapping[int, tuple[ExecutableCondition, ...]],
    *,
    start: date | None = None,
    end: date | None = None,
    complete_only: bool = False,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    dict[int, pd.Series],
    dict[int, pd.Series],
    dict[tuple[object, ...], int],
]:
    """Evaluate one D+1 -> D+2 -> D+3 policy from a D+1 anchor cohort.

    ``eligible_at_landmark`` remains the intrinsic chart/path eligibility.
    Active state is rebuilt for every candidate and every fold, so final-policy
    state persisted in the result table can never leak into selection.
    """
    day_values = pd.to_numeric(frame["landmark_day"], errors="coerce")
    period = _policy_period_mask(frame, start, end)
    anchor_mask = day_values.eq(1) & _truthy(frame["eligible_at_landmark"]) & period
    if complete_only:
        horizon_end = pd.to_datetime(frame["horizon_end_date"], errors="coerce").dt.date
        anchor_mask &= _truthy(frame["full_outcome_available"])
        if end is not None:
            anchor_mask &= horizon_end.notna() & horizon_end.le(end)
    anchor = frame.loc[anchor_mask].copy()
    anchor_keys = _policy_keys(anchor)
    active_keys = set(anchor_keys.tolist())
    cut_day_by_key: dict[tuple[object, ...], int] = {}
    active_masks = {
        day: pd.Series(False, index=frame.index, dtype=bool)
        for day in EARLY_CUT_LANDMARK_DAYS
    }
    cut_masks = {
        day: pd.Series(False, index=frame.index, dtype=bool)
        for day in EARLY_CUT_LANDMARK_DAYS
    }
    for day in EARLY_CUT_LANDMARK_DAYS:
        row_mask = (
            day_values.eq(day)
            & _truthy(frame["eligible_at_landmark"])
            & period
        )
        local = frame.loc[row_mask]
        if local.empty or not active_keys:
            continue
        local_keys = _policy_keys(local)
        active = local_keys.isin(active_keys)
        active_masks[day].loc[local.index] = active.to_numpy()
        matched = _combined_mask(
            local, conditions_by_day.get(day, ())
        ) & active
        cut_masks[day].loc[local.index] = matched.to_numpy()
        if matched.any():
            for key in local_keys.loc[matched].tolist():
                cut_day_by_key[key] = day
                active_keys.discard(key)
    anchor_matched = anchor_keys.isin(cut_day_by_key)
    return anchor, anchor_matched, active_masks, cut_masks, cut_day_by_key


def _local_objective_keys_by_day(
    frame: pd.DataFrame, objectives: Iterable[str]
) -> dict[str, dict[int, set[object]]]:
    """Return intrinsic, landmark-local positive labels keyed by signal."""
    objective_names = tuple(objectives)
    result = {
        objective: {day: set() for day in EARLY_CUT_LANDMARK_DAYS}
        for objective in objective_names
    }
    day_values = pd.to_numeric(frame["landmark_day"], errors="coerce")
    eligible = _truthy(frame["eligible_at_landmark"])
    for day in EARLY_CUT_LANDMARK_DAYS:
        local = frame.loc[day_values.eq(day) & eligible]
        if local.empty:
            continue
        keys = _policy_keys(local)
        for objective in objective_names:
            result[objective][day] = set(
                keys.loc[_truthy(local[objective])].tolist()
            )
    return result


def _anchored_actual_metrics_from_sets(
    *,
    population_count: int,
    anchor_keys: set[object],
    labeled_keys: set[object],
    objective_keys: set[object],
    protected_keys: set[object],
    cut_day_by_key: Mapping[object, int],
    local_objective_keys_by_day: Mapping[int, set[object]],
) -> dict[str, object]:
    """Score one cut per D+1 anchor against its actual cut-landmark label.

    Population, labels and objective/protected denominators remain fixed at
    D+1.  A cut receives objective credit only when that same objective is
    still true at the first cut landmark.  Any cut of a D+1 protected signal
    remains a protected rejection, irrespective of its later local label.
    """
    matched_keys = set(cut_day_by_key) & anchor_keys
    matched_labeled = matched_keys & labeled_keys
    matched_objective = {
        key
        for key in matched_labeled & objective_keys
        if key in local_objective_keys_by_day.get(cut_day_by_key[key], set())
    }
    matched_protected = matched_labeled & protected_keys
    return _metrics_from_counts(
        {
            "population_count": population_count,
            "sample_count": len(labeled_keys),
            "unlabeled_count": population_count - len(labeled_keys),
            "matched_count": len(matched_keys),
            "matched_labeled_count": len(matched_labeled),
            "matched_unlabeled_count": len(matched_keys - labeled_keys),
            "objective_count": len(objective_keys),
            "protected_count": len(protected_keys),
            "matched_objective_count": len(matched_objective),
            "matched_protected_count": len(matched_protected),
        }
    )


def _sequential_policy_metrics(
    frame: pd.DataFrame,
    conditions_by_day: Mapping[int, tuple[ExecutableCondition, ...]],
    *,
    start: date | None,
    end: date | None,
    available_through: date,
    complete_only: bool,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    anchor, _, _, _, cut_day_by_key = _sequential_policy_state(
        frame,
        conditions_by_day,
        start=start,
        end=end,
        complete_only=complete_only,
    )
    specs = {
        spec.objective: spec
        for spec in (*early_specs(1), SEQUENTIAL_EARLY_SPEC)
    }
    anchor_key_series = _policy_keys(anchor)
    anchor_keys = set(anchor_key_series.tolist())
    local_keys = _local_objective_keys_by_day(frame, specs)
    metrics: dict[str, dict[str, object]] = {}
    for objective, spec in specs.items():
        available = _available(anchor, spec, available_through)
        labeled = (
            anchor[objective].notna()
            & anchor[spec.protected_outcome].notna()
            & available
        )
        objective_values = labeled & _truthy(anchor[objective])
        protected_values = labeled & _truthy(anchor[spec.protected_outcome])
        metrics[objective] = _anchored_actual_metrics_from_sets(
            population_count=len(anchor),
            anchor_keys=anchor_keys,
            labeled_keys=set(anchor_key_series.loc[labeled].tolist()),
            objective_keys=set(anchor_key_series.loc[objective_values].tolist()),
            protected_keys=set(anchor_key_series.loc[protected_values].tolist()),
            cut_day_by_key=cut_day_by_key,
            local_objective_keys_by_day=local_keys[objective],
        )
    bad_metrics = metrics.pop(SEQUENTIAL_EARLY_SPEC.objective)
    return metrics, bad_metrics


def _early_conditions_by_day(
    selections: Mapping[int, Mapping[str, ObjectiveSelection]],
    prefix_lengths: Mapping[int, Mapping[str, int]],
    *,
    fold_index: int | None = None,
) -> dict[int, tuple[ExecutableCondition, ...]]:
    result: dict[int, tuple[ExecutableCondition, ...]] = {}
    for day in EARLY_CUT_LANDMARK_DAYS:
        conditions: list[ExecutableCondition] = []
        for objective, selection in selections[day].items():
            prefix = selection.prefixes[prefix_lengths[day][objective]]
            selected_conditions = (
                prefix.final_conditions
                if fold_index is None
                else prefix.fold_results[fold_index].conditions
            )
            conditions.extend(selected_conditions)
        result[day] = _unique_conditions(conditions)
    return result


def _choose_sequential_early_prefixes(
    frame: pd.DataFrame,
    selections: Mapping[int, Mapping[str, ObjectiveSelection]],
    maximum_lengths: Mapping[int, Mapping[str, int]],
    cfg: Config,
) -> dict[int, dict[str, int]]:
    """Choose one cumulative early policy under pooled, fold and refit gates."""
    slots = tuple(
        (day, objective)
        for day in EARLY_CUT_LANDMARK_DAYS
        for objective in selections[day]
    )
    ranges = tuple(
        range(int(maximum_lengths[day][objective]) + 1)
        for day, objective in slots
    )
    folds = build_walk_forward_folds(cfg)
    policy_specs = (*early_specs(1), SEQUENTIAL_EARLY_SPEC)
    local_objective_keys = _local_objective_keys_by_day(
        frame, (spec.objective for spec in policy_specs)
    )

    def build_context(
        start: date,
        end: date,
        available_through: date,
        fold_index: int | None,
    ) -> dict[str, object]:
        anchor, _, _, _, _ = _sequential_policy_state(
            frame,
            {day: () for day in EARLY_CUT_LANDMARK_DAYS},
            start=start,
            end=end,
            complete_only=True,
        )
        keys = _policy_keys(anchor)
        if keys.duplicated().any():
            raise RuntimeError("sequential policy anchor contains duplicate signals")
        key_sets: dict[str, tuple[set[object], set[object], set[object]]] = {}
        for spec in policy_specs:
            available = _available(anchor, spec, available_through)
            labeled = (
                anchor[spec.objective].notna()
                & anchor[spec.protected_outcome].notna()
                & available
            )
            objective = labeled & _truthy(anchor[spec.objective])
            protected = labeled & _truthy(anchor[spec.protected_outcome])
            key_sets[spec.objective] = (
                set(keys.loc[labeled].tolist()),
                set(keys.loc[objective].tolist()),
                set(keys.loc[protected].tolist()),
            )
        matches: dict[tuple[int, str, int], set[object]] = {}
        for day, objective in slots:
            maximum = int(maximum_lengths[day][objective])
            matches[(day, objective, 0)] = set()
            for length in range(1, maximum + 1):
                prefix = selections[day][objective].prefixes[length]
                conditions = (
                    prefix.final_conditions
                    if fold_index is None
                    else prefix.fold_results[fold_index].conditions
                )
                day_conditions = {
                    item: (() if item != day else conditions)
                    for item in EARLY_CUT_LANDMARK_DAYS
                }
                _, _, _, _, cut_days = _sequential_policy_state(
                    frame,
                    day_conditions,
                    start=start,
                    end=end,
                    complete_only=True,
                )
                matches[(day, objective, length)] = set(cut_days)
        return {
            "population_count": len(anchor),
            "anchor_keys": set(keys.tolist()),
            "key_sets": key_sets,
            "matches": matches,
        }

    def context_metrics(
        context: Mapping[str, object],
        objective: str,
        cut_day_by_key: Mapping[object, int],
    ) -> dict[str, object]:
        labeled_keys, objective_keys, protected_keys = context["key_sets"][
            objective
        ]
        return _anchored_actual_metrics_from_sets(
            population_count=int(context["population_count"]),
            anchor_keys=context["anchor_keys"],
            labeled_keys=labeled_keys,
            objective_keys=objective_keys,
            protected_keys=protected_keys,
            cut_day_by_key=cut_day_by_key,
            local_objective_keys_by_day=local_objective_keys[objective],
        )

    def context_cut_days(
        context: Mapping[str, object], values: tuple[int, ...]
    ) -> dict[object, int]:
        """Resolve the first selected cut day without rescanning the frame."""
        result: dict[object, int] = {}
        for day in EARLY_CUT_LANDMARK_DAYS:
            matches: set[object] = set()
            for (slot_day, objective), value in zip(slots, values):
                if slot_day == day:
                    matches.update(context["matches"][(day, objective, value)])
            for key in matches:
                result.setdefault(key, day)
        return result

    fold_contexts = [
        build_context(
            fold.start_date,
            fold.end_date,
            fold.end_date,
            fold_index,
        )
        for fold_index, fold in enumerate(folds)
    ]
    fixed_contexts = [
        build_context(start, end, cfg.validation_end_date, None)
        for start, end in (
            (cfg.signal_start_date, cfg.validation_end_date),
            (cfg.discovery_end_date + timedelta(days=1), cfg.validation_end_date),
        )
    ]
    best_rank: tuple[object, ...] | None = None
    best = {
        day: {objective: 0 for objective in selections[day]}
        for day in EARLY_CUT_LANDMARK_DAYS
    }
    for values in __import__("itertools").product(*ranges):
        lengths = {
            day: {objective: 0 for objective in selections[day]}
            for day in EARLY_CUT_LANDMARK_DAYS
        }
        for (day, objective), value in zip(slots, values):
            lengths[day][objective] = value
        total_conditions = sum(
            len(selections[day][objective].prefixes[value].templates)
            for (day, objective), value in zip(slots, values)
        )
        objective_folds: dict[str, list[FoldResult]] = {
            spec.objective: [] for spec in early_specs(1)
        }
        bad_folds: list[FoldResult] = []
        for fold, context in zip(folds, fold_contexts):
            cut_day_by_key = context_cut_days(context, values)
            for objective in objective_folds:
                metrics = context_metrics(context, objective, cut_day_by_key)
                objective_folds[objective].append(
                    FoldResult(fold, (), metrics)
                )
            bad_folds.append(
                FoldResult(
                    fold,
                    (),
                    context_metrics(
                        context,
                        SEQUENTIAL_EARLY_SPEC.objective,
                        cut_day_by_key,
                    ),
                )
            )
        pooled_by_objective = {
            objective: _aggregate_metrics(item.metrics for item in results)
            for objective, results in objective_folds.items()
        }
        pooled_bad = _aggregate_metrics(item.metrics for item in bad_folds)
        valid = total_conditions == 0
        if total_conditions:
            valid = (
                _development_gates(pooled_bad, cfg)
                and _stability(bad_folds, cfg)[0]
                and _fold_safety_gates(bad_folds, cfg)
            )
            for context in fixed_contexts:
                cut_day_by_key = context_cut_days(context, values)
                fixed_bad = context_metrics(
                    context,
                    SEQUENTIAL_EARLY_SPEC.objective,
                    cut_day_by_key,
                )
                valid &= _development_gates(fixed_bad, cfg)
                if not valid:
                    break
        if not valid:
            continue
        score = _selection_score(pooled_bad) or 0.0
        captures = [
            float(metrics["objective_capture_rate"] or 0.0)
            for metrics in pooled_by_objective.values()
        ]
        retention = float(pooled_bad["protected_retention_rate"] or 0.0)
        match_rate = float(pooled_bad["match_rate"] or 0.0)
        complexity_adjusted_score = score - (
            total_conditions * cfg.min_selection_score_improvement
        )
        rank = (
            -complexity_adjusted_score,
            -score,
            -float(np.mean(captures)),
            -retention,
            match_rate,
            total_conditions,
            tuple(values),
        )
        if best_rank is None or rank < best_rank:
            best_rank = rank
            best = lengths
    return best


def _row_matches(
    frame: pd.DataFrame, conditions: tuple[ExecutableCondition, ...]
) -> tuple[pd.Series, list[str | None], list[str | None]]:
    masks = [condition.matches(frame) for condition in conditions]
    combined = pd.Series(False, index=frame.index, dtype=bool)
    ids: list[str | None] = []
    reasons: list[str | None] = []
    for mask in masks:
        combined |= mask
    for position in range(len(frame)):
        hits = [
            condition
            for condition, mask in zip(conditions, masks)
            if bool(mask.iloc[position])
        ]
        ids.append(",".join(item.rule_id for item in hits) or None)
        reasons.append(" OR ".join(item.text for item in hits) or None)
    return combined, ids, reasons


def apply_entry_decisions(
    signals: pd.DataFrame, selected: SelectedSummary
) -> pd.DataFrame:
    result = signals.copy()
    weak_selection = selected.entry["weak_5d"]
    loss_selection = selected.entry["loss_first_5d"]
    weak_len = selected.entry_prefix_lengths["weak_5d"]
    loss_len = selected.entry_prefix_lengths["loss_first_5d"]
    weak_conditions = weak_selection.selected.final_conditions[:weak_len]
    loss_conditions = loss_selection.selected.final_conditions[:loss_len]
    weak_mask, weak_ids, weak_reasons = _row_matches(result, weak_conditions)
    loss_mask, loss_ids, loss_reasons = _row_matches(result, loss_conditions)
    final_mask = weak_mask | loss_mask
    result["include_weak_filter"] = ~weak_mask
    result["include_loss_first_filter"] = ~loss_mask
    result["include_final"] = ~final_mask
    result["weak_matched_rule_ids"] = weak_ids
    result["loss_first_matched_rule_ids"] = loss_ids
    result["matched_rule_ids"] = [
        ",".join(filter(None, (weak_id, loss_id))) or None
        for weak_id, loss_id in zip(weak_ids, loss_ids)
    ]
    result["exclusion_reason"] = [
        "; ".join(
            part
            for part in (
                f"weak_5d: {weak_reason}" if weak_reason else "",
                f"loss_first_5d: {loss_reason}" if loss_reason else "",
            )
            if part
        )
        or None
        for weak_reason, loss_reason in zip(weak_reasons, loss_reasons)
    ]
    result["filter_decision"] = np.where(final_mask, "exclude", "include")
    return result.loc[:, SIGNAL_COLUMNS]


def apply_early_decisions(
    early_cuts: pd.DataFrame, selected: SelectedSummary
) -> pd.DataFrame:
    result = early_cuts.copy()
    result["include_stagnation_filter"] = False
    result["include_loss_filter"] = False
    result["include_final"] = False
    result["stagnation_matched_rule_ids"] = pd.NA
    result["loss_matched_rule_ids"] = pd.NA
    result["matched_rule_ids"] = pd.NA
    result["cut_reason"] = pd.NA
    result["active_at_landmark"] = False
    result["prior_policy_cut_day"] = pd.NA
    prior_state = result["cut_decision"].astype("string")
    preserved = prior_state.isin(("not_eligible", "not_evaluable"))
    result["cut_decision"] = prior_state.where(preserved, "not_evaluable")
    eligible = _truthy(result["eligible_at_landmark"])
    result.loc[eligible, "cut_decision"] = "hold"
    conditions_by_day = _early_conditions_by_day(
        selected.early_cut, selected.early_cut_prefix_lengths
    )
    _, _, active_masks, policy_cut_masks, cut_day_by_key = (
        _sequential_policy_state(result, conditions_by_day)
    )
    for day, objective_selections in selected.early_cut.items():
        day_rows = pd.to_numeric(result["landmark_day"], errors="coerce").eq(day)
        day_indices = result.index[day_rows]
        if len(day_indices):
            day_keys = _policy_keys(result.loc[day_indices])
            prior_days = [
                (
                    cut_day_by_key.get(key)
                    if cut_day_by_key.get(key, day) < day
                    else pd.NA
                )
                for key in day_keys
            ]
            result.loc[day_indices, "prior_policy_cut_day"] = prior_days
            prior_cut = pd.Series(
                [not pd.isna(value) for value in prior_days],
                index=day_indices,
                dtype=bool,
            )
            result.loc[prior_cut.index[prior_cut], "cut_decision"] = "not_active"
        rows = day_rows & active_masks[day]
        result.loc[rows, "active_at_landmark"] = True
        result.loc[rows, "include_stagnation_filter"] = True
        result.loc[rows, "include_loss_filter"] = True
        result.loc[rows, "include_final"] = True
        local = result.loc[rows]
        if local.empty:
            continue
        stagnant = objective_selections["stagnant_to_day5"]
        loss = objective_selections["loss_first_to_day5"]
        lengths = selected.early_cut_prefix_lengths[day]
        stagnant_mask, stagnant_ids, stagnant_reasons = _row_matches(
            local, stagnant.selected.final_conditions[: lengths["stagnant_to_day5"]]
        )
        loss_mask, loss_ids, loss_reasons = _row_matches(
            local, loss.selected.final_conditions[: lengths["loss_first_to_day5"]]
        )
        cut = stagnant_mask | loss_mask
        expected_cut = policy_cut_masks[day].loc[local.index]
        if not cut.equals(expected_cut):
            raise AssertionError("sequential early-cut union mask is inconsistent")
        result.loc[rows, "include_stagnation_filter"] = (~stagnant_mask).to_numpy()
        result.loc[rows, "include_loss_filter"] = (~loss_mask).to_numpy()
        result.loc[rows, "include_final"] = (~cut).to_numpy()
        result.loc[rows, "stagnation_matched_rule_ids"] = stagnant_ids
        result.loc[rows, "loss_matched_rule_ids"] = loss_ids
        result.loc[rows, "matched_rule_ids"] = [
            ",".join(filter(None, (left, right))) or None
            for left, right in zip(stagnant_ids, loss_ids)
        ]
        reasons = [
            "; ".join(
                part
                for part in (
                    f"stagnant_to_day5: {left}" if left else "",
                    f"loss_first_to_day5: {right}" if right else "",
                )
                if part
            )
            or None
            for left, right in zip(stagnant_reasons, loss_reasons)
        ]
        cut_indices = local.index[cut]
        result.loc[cut_indices, "cut_decision"] = "cut"
        result.loc[rows, "cut_reason"] = reasons
        result.loc[local.index[~cut], "cut_reason"] = pd.NA
    expected_active = eligible & result["prior_policy_cut_day"].isna()
    if not _truthy(result["active_at_landmark"]).equals(expected_active):
        raise AssertionError("sequential early-cut active state is inconsistent")
    cuts = result.loc[
        result["cut_decision"].astype("string").eq("cut"),
        [*POLICY_KEY_COLUMNS, "landmark_day"],
    ]
    cut_lookup = {
        (*key, int(day))
        for *key, day in cuts.itertuples(index=False, name=None)
    }
    prior_rows = result.loc[result["prior_policy_cut_day"].notna()]
    for row in prior_rows.loc[
        :, [*POLICY_KEY_COLUMNS, "prior_policy_cut_day"]
    ].itertuples(index=False, name=None):
        *key, prior_day = row
        if (*key, int(prior_day)) not in cut_lookup:
            raise AssertionError(
                "prior_policy_cut_day does not reference an earlier policy cut"
            )
    return result.loc[:, EARLY_CUT_COLUMNS]


def _rule_text(conditions: tuple[ExecutableCondition, ...]) -> str:
    return " OR ".join(item.text for item in conditions) or "no exclusion condition"


def _sequential_rule_text(
    conditions_by_day: Mapping[int, tuple[ExecutableCondition, ...]],
) -> str:
    parts = [
        f"D+{day}: " + " OR ".join(condition.text for condition in conditions)
        for day, conditions in sorted(conditions_by_day.items())
        if conditions
    ]
    return "; then ".join(parts) or "no sequential cut condition"


def _sequential_template_text(
    templates_by_day: Mapping[int, tuple[CandidateTemplate, ...]],
) -> str:
    parts = [
        f"D+{day}: {_template_text(templates)}"
        for day, templates in sorted(templates_by_day.items())
        if templates
    ]
    return "; then ".join(parts) or "no sequential cut condition"


def _feature_fields(
    conditions: tuple[ExecutableCondition, ...],
) -> tuple[str, str | None, str | None, float | None, float | None]:
    if not conditions:
        return "none", None, None, None, None
    if len(conditions) == 1:
        item = conditions[0]
        if isinstance(item, CompositeCondition):
            return item.feature_group, None, None, None, None
        return (
            item.feature_group,
            item.feature_name,
            item.operator,
            item.quantile,
            item.threshold,
        )
    groups = {item.feature_group for item in conditions}
    return (
        (next(iter(groups)) if len(groups) == 1 else "multiple"),
        None,
        None,
        None,
        None,
    )


def _template_fields(
    templates: tuple[CandidateTemplate, ...],
) -> tuple[str, str | None, str | None, float | None, float | None]:
    if not templates:
        return "none", None, None, None, None
    if len(templates) == 1:
        item = templates[0]
        if isinstance(item, PatternTemplate):
            return item.feature_group, None, None, None, None
        return item.feature_group, item.feature_name, item.operator, item.quantile, None
    groups = {item.feature_group for item in templates}
    return (
        next(iter(groups)) if len(groups) == 1 else "multiple",
        None,
        None,
        None,
        None,
    )


def _template_text(templates: tuple[CandidateTemplate, ...]) -> str:
    parts = []
    for item in templates:
        clauses = item.clauses if isinstance(item, PatternTemplate) else (item,)
        clause_text = []
        for clause in clauses:
            symbol = "<=" if clause.operator == "le" else ">="
            threshold_text = f"causal Q{clause.quantile:.4g} threshold"
            if clause.minimum_threshold is not None:
                threshold_text = (
                    f"max({threshold_text}, {clause.minimum_threshold:.10g})"
                )
            clause_text.append(f"{clause.feature_name} {symbol} {threshold_text}")
        text = " AND ".join(clause_text)
        if isinstance(item, PatternTemplate):
            text = f"{item.pattern_name} ({text})"
        parts.append(text)
    return " OR ".join(parts) or "no exclusion condition"


def _rule_row(
    *,
    rule_id: str,
    result_kind: str,
    spec: ObjectiveSpec,
    conditions: tuple[ExecutableCondition, ...],
    evaluation_scope: str,
    scope_year: int | None,
    period_start: date | None,
    period_end: date | None,
    metrics: Mapping[str, object],
    is_selected: bool,
    is_final_filter: bool,
    passes_holdout: bool | None = None,
    passes_development_gates: bool | None = None,
    passes_stability_gates: bool | None = None,
    stability: Mapping[str, object] | None = None,
    selection_order: int | None = None,
    threshold_fit_end_date: date | None = None,
    templates: tuple[CandidateTemplate, ...] = (),
    rule_text_override: str | None = None,
) -> dict[str, object]:
    feature_group, feature_name, operator, quantile, threshold = (
        _feature_fields(conditions) if conditions else _template_fields(templates)
    )
    stats = stability or {}
    row: dict[str, object] = {
        "rule_id": rule_id,
        "result_kind": result_kind,
        "decision_family": spec.decision_family,
        "objective": spec.objective,
        "protected_outcome": spec.protected_outcome,
        "landmark_day": spec.landmark_day,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "operator": operator,
        "quantile_value": quantile,
        "threshold_value": threshold,
        "rule_text": rule_text_override
        or (_rule_text(conditions) if conditions else _template_text(templates)),
        "selection_order": selection_order,
        "evaluation_scope": evaluation_scope,
        "scope_year": scope_year,
        "period_start": period_start,
        "period_end": period_end,
        "threshold_fit_end_date": threshold_fit_end_date,
        "is_selected": is_selected,
        "is_final_filter": is_final_filter,
        "passes_holdout": passes_holdout,
        "passes_development_gates": passes_development_gates,
        "passes_stability_gates": passes_stability_gates,
        "component_count": sum(
            len(item.clauses)
            if isinstance(item, (CompositeCondition, PatternTemplate))
            else 1
            for item in (conditions if conditions else templates)
        ),
        **metrics,
        "eligible_fold_count": int(stats.get("eligible_fold_count", 0) or 0),
        "positive_lift_fold_count": int(stats.get("positive_lift_fold_count", 0) or 0),
        "positive_lift_fold_fraction": stats.get("positive_lift_fold_fraction"),
        "median_fold_objective_lift": stats.get("median_fold_objective_lift"),
        "min_fold_objective_lift": stats.get("min_fold_objective_lift"),
        "min_fold_protected_retention_rate": stats.get(
            "min_fold_protected_retention_rate"
        ),
        "max_fold_match_rate": stats.get("max_fold_match_rate"),
        "selection_score": _selection_score(metrics),
    }
    return row


def _scope_metrics(
    frame: pd.DataFrame,
    spec: ObjectiveSpec,
    conditions: tuple[ExecutableCondition, ...],
    start: date | None,
    end: date | None,
    available_through: date,
) -> tuple[pd.DataFrame, dict[str, object]]:
    population = _period_frame(_universe(frame, spec), spec, start, end)
    return population, objective_metrics(
        population,
        _combined_mask(population, conditions),
        spec.objective,
        spec.protected_outcome,
        _available(population, spec, available_through),
    )


def _unique_conditions(
    items: Iterable[ExecutableCondition],
) -> tuple[ExecutableCondition, ...]:
    result: list[ExecutableCondition] = []
    seen: set[str] = set()
    for item in items:
        if item.rule_id not in seen:
            seen.add(item.rule_id)
            result.append(item)
    return tuple(result)


def _unique_templates(
    items: Iterable[CandidateTemplate],
) -> tuple[CandidateTemplate, ...]:
    result: list[CandidateTemplate] = []
    seen: set[str] = set()
    for item in items:
        if item.rule_id not in seen:
            seen.add(item.rule_id)
            result.append(item)
    return tuple(result)


def _append_final_union_rows(
    rows: list[dict[str, object]],
    source: pd.DataFrame,
    selections: Mapping[str, ObjectiveSelection],
    prefix_lengths: Mapping[str, int],
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
    *,
    mark_final: bool = True,
) -> None:
    objectives = tuple(selections)
    if not objectives:
        return
    chosen = {
        objective: selections[objective].prefixes[prefix_lengths[objective]]
        for objective in objectives
    }
    final_conditions = _unique_conditions(
        condition
        for objective in objectives
        for condition in chosen[objective].final_conditions
    )
    final_templates = _unique_templates(
        template for objective in objectives for template in chosen[objective].templates
    )
    first = selections[objectives[0]].spec
    base_id = (
        "ENTRY_FILTER_FINAL"
        if first.decision_family == "entry_filter"
        else f"EARLY_CUT_D{first.landmark_day}_POLICY_COMPONENT"
    )
    holdout_dates = trading_dates[trading_dates.date > cfg.holdout_cutoff_date]
    holdout_start = holdout_dates[0].date() if len(holdout_dates) else None
    as_of = trading_dates[-1].date() if len(trading_dates) else cfg.holdout_cutoff_date
    union_fold_conditions: list[tuple[ExecutableCondition, ...]] = []
    reference = chosen[objectives[0]].fold_results
    for fold_index in range(len(reference)):
        union_fold_conditions.append(
            _unique_conditions(
                condition
                for objective in objectives
                for condition in chosen[objective].fold_results[fold_index].conditions
            )
        )
    for objective in objectives:
        selection = selections[objective]
        spec = selection.spec
        union_fold_results: list[FoldResult] = []
        for reference_fold, conditions in zip(reference, union_fold_conditions):
            fold = reference_fold.fold
            population = _period_frame(
                _universe(source, spec), spec, fold.start_date, fold.end_date
            )
            metrics = objective_metrics(
                population,
                _combined_mask(population, conditions),
                spec.objective,
                spec.protected_outcome,
                _available(population, spec, fold.end_date),
            )
            union_fold_results.append(FoldResult(fold, conditions, metrics))
            rows.append(
                _rule_row(
                    rule_id=base_id,
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=conditions,
                    evaluation_scope="walk_forward_year",
                    scope_year=fold.year,
                    period_start=fold.start_date,
                    period_end=fold.end_date,
                    metrics=metrics,
                    is_selected=True,
                    is_final_filter=mark_final,
                    threshold_fit_end_date=(
                        fold.threshold_fit_end_date if conditions else None
                    ),
                )
            )
        pooled = _aggregate_metrics(item.metrics for item in union_fold_results)
        objective_stability_pass, stability = _stability(union_fold_results, cfg)
        stability_pass = (
            objective_stability_pass
            if first.decision_family != "entry_filter"
            else _fold_safety_gates(union_fold_results, cfg)
        )
        rows.append(
            _rule_row(
                rule_id=base_id,
                result_kind="selected_filter",
                spec=spec,
                conditions=(),
                templates=final_templates,
                evaluation_scope="walk_forward_pooled",
                scope_year=None,
                period_start=date(cfg.walk_forward_first_year, 1, 1),
                period_end=cfg.validation_end_date,
                metrics=pooled,
                is_selected=True,
                is_final_filter=mark_final,
                passes_development_gates=_safety_gates(pooled, cfg),
                passes_stability_gates=stability_pass,
                stability=stability,
            )
        )
        rows.append(
            _rule_row(
                rule_id=f"BASELINE_{base_id}_{spec.objective.upper()}",
                result_kind="baseline",
                spec=spec,
                conditions=(),
                evaluation_scope="walk_forward_pooled",
                scope_year=None,
                period_start=date(cfg.walk_forward_first_year, 1, 1),
                period_end=cfg.validation_end_date,
                metrics=_baseline_metrics(pooled),
                is_selected=False,
                is_final_filter=False,
            )
        )
        scopes = (
            (
                "development",
                cfg.signal_start_date,
                cfg.validation_end_date,
                cfg.validation_end_date,
            ),
            (
                "validation",
                cfg.discovery_end_date + timedelta(days=1),
                cfg.validation_end_date,
                cfg.validation_end_date,
            ),
            (
                "diagnostic",
                cfg.validation_end_date + timedelta(days=1),
                cfg.holdout_cutoff_date,
                cfg.holdout_cutoff_date,
            ),
            ("holdout", holdout_start, None, as_of),
            ("all_signals", cfg.signal_start_date, None, as_of),
        )
        for scope, start, end, available_through in scopes:
            if scope == "holdout" and holdout_start is None:
                population = _universe(source, spec).iloc[0:0]
                metrics = objective_metrics(
                    population,
                    pd.Series(dtype=bool),
                    spec.objective,
                    spec.protected_outcome,
                )
            else:
                population, metrics = _scope_metrics(
                    source,
                    spec,
                    final_conditions,
                    start,
                    end,
                    available_through,
                )
            holdout_pass = None
            if (
                scope == "holdout"
                and int(metrics["sample_count"]) >= cfg.min_holdout_sample_count
            ):
                holdout_pass = _development_gates(metrics, cfg)
            dates = pd.to_datetime(population[spec.date_column], errors="coerce")
            period_start = (
                start
                if start is not None
                else (dates.min().date() if len(population) else None)
            )
            period_end = (
                None
                if scope == "holdout" and holdout_start is None
                else (end if end is not None else as_of)
            )
            rows.append(
                _rule_row(
                    rule_id=base_id,
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=final_conditions,
                    evaluation_scope=scope,
                    scope_year=None,
                    period_start=period_start,
                    period_end=period_end,
                    metrics=metrics,
                    is_selected=True,
                    is_final_filter=mark_final,
                    passes_holdout=holdout_pass,
                    passes_development_gates=(
                        _safety_gates(metrics, cfg) if scope == "development" else None
                    ),
                    passes_stability_gates=(
                        stability_pass if scope == "development" else None
                    ),
                    stability=(stability if scope == "development" else None),
                    threshold_fit_end_date=(
                        cfg.validation_end_date
                        if final_conditions
                        and not (scope == "holdout" and holdout_start is None)
                        else None
                    ),
                )
            )
            if scope in {"development", "diagnostic", "holdout"}:
                rows.append(
                    _rule_row(
                        rule_id=f"BASELINE_{base_id}_{spec.objective.upper()}",
                        result_kind="baseline",
                        spec=spec,
                        conditions=(),
                        evaluation_scope=scope,
                        scope_year=None,
                        period_start=period_start,
                        period_end=period_end,
                        metrics=_baseline_metrics(metrics),
                        is_selected=False,
                        is_final_filter=False,
                    )
                )


def _append_sequential_policy_rows(
    rows: list[dict[str, object]],
    source: pd.DataFrame,
    selections: Mapping[int, Mapping[str, ObjectiveSelection]],
    prefix_lengths: Mapping[int, Mapping[str, int]],
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> None:
    base_id = "EARLY_CUT_D1_D3_SEQUENTIAL_D1_ANCHOR_FINAL"
    specs = (SEQUENTIAL_EARLY_SPEC, *early_specs(1))
    fold_results: dict[str, list[FoldResult]] = {
        spec.objective: [] for spec in specs
    }
    folds = build_walk_forward_folds(cfg)
    for fold_index, fold in enumerate(folds):
        conditions_by_day = _early_conditions_by_day(
            selections, prefix_lengths, fold_index=fold_index
        )
        target_metrics, bad_metrics = _sequential_policy_metrics(
            source,
            conditions_by_day,
            start=fold.start_date,
            end=fold.end_date,
            available_through=fold.end_date,
            complete_only=True,
        )
        metrics_by_objective = {
            **target_metrics,
            SEQUENTIAL_EARLY_SPEC.objective: bad_metrics,
        }
        conditions = _unique_conditions(
            condition
            for day_conditions in conditions_by_day.values()
            for condition in day_conditions
        )
        policy_text = _sequential_rule_text(conditions_by_day)
        for spec in specs:
            metrics = metrics_by_objective[spec.objective]
            fold_results[spec.objective].append(
                FoldResult(fold, conditions, metrics)
            )
            rows.append(
                _rule_row(
                    rule_id=base_id,
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=conditions,
                    evaluation_scope="walk_forward_year",
                    scope_year=fold.year,
                    period_start=fold.start_date,
                    period_end=fold.end_date,
                    metrics=metrics,
                    is_selected=True,
                    is_final_filter=True,
                    threshold_fit_end_date=(
                        fold.threshold_fit_end_date if conditions else None
                    ),
                    rule_text_override=policy_text,
                )
            )
    target_stability_pass, policy_stability = _stability(
        fold_results[SEQUENTIAL_EARLY_SPEC.objective], cfg
    )
    policy_stability_pass = target_stability_pass and _fold_safety_gates(
        fold_results[SEQUENTIAL_EARLY_SPEC.objective], cfg
    )
    final_conditions_by_day = _early_conditions_by_day(
        selections, prefix_lengths
    )
    final_conditions = _unique_conditions(
        condition
        for day_conditions in final_conditions_by_day.values()
        for condition in day_conditions
    )
    final_policy_text = _sequential_rule_text(final_conditions_by_day)
    final_templates_by_day = {
        day: _unique_templates(
            template
            for objective, selection in selections[day].items()
            for template in selection.prefixes[
                prefix_lengths[day][objective]
            ].templates
        )
        for day in EARLY_CUT_LANDMARK_DAYS
    }
    final_templates = _unique_templates(
        template
        for day_templates in final_templates_by_day.values()
        for template in day_templates
    )
    final_policy_template_text = _sequential_template_text(
        final_templates_by_day
    )
    for spec in specs:
        pooled = _aggregate_metrics(
            item.metrics for item in fold_results[spec.objective]
        )
        _, objective_stability = _stability(
            fold_results[spec.objective], cfg
        )
        rows.append(
            _rule_row(
                rule_id=base_id,
                result_kind="selected_filter",
                spec=spec,
                conditions=(),
                templates=final_templates,
                evaluation_scope="walk_forward_pooled",
                scope_year=None,
                period_start=date(cfg.walk_forward_first_year, 1, 1),
                period_end=cfg.validation_end_date,
                metrics=pooled,
                is_selected=True,
                is_final_filter=True,
                passes_development_gates=(
                    _development_gates(pooled, cfg)
                    if spec == SEQUENTIAL_EARLY_SPEC
                    else _safety_gates(pooled, cfg)
                ),
                passes_stability_gates=policy_stability_pass,
                stability=(
                    policy_stability
                    if spec == SEQUENTIAL_EARLY_SPEC
                    else objective_stability
                ),
                rule_text_override=final_policy_template_text,
            )
        )
        rows.append(
            _rule_row(
                rule_id=f"BASELINE_{base_id}_{spec.objective.upper()}",
                result_kind="baseline",
                spec=spec,
                conditions=(),
                evaluation_scope="walk_forward_pooled",
                scope_year=None,
                period_start=date(cfg.walk_forward_first_year, 1, 1),
                period_end=cfg.validation_end_date,
                metrics=_baseline_metrics(pooled),
                is_selected=False,
                is_final_filter=False,
            )
        )
    holdout_dates = trading_dates[trading_dates.date > cfg.holdout_cutoff_date]
    holdout_start = holdout_dates[0].date() if len(holdout_dates) else None
    as_of = trading_dates[-1].date() if len(trading_dates) else cfg.holdout_cutoff_date
    scopes = (
        (
            "development",
            cfg.signal_start_date,
            cfg.validation_end_date,
            cfg.validation_end_date,
        ),
        (
            "validation",
            cfg.discovery_end_date + timedelta(days=1),
            cfg.validation_end_date,
            cfg.validation_end_date,
        ),
        (
            "diagnostic",
            cfg.validation_end_date + timedelta(days=1),
            cfg.holdout_cutoff_date,
            cfg.holdout_cutoff_date,
        ),
        ("holdout", holdout_start, None, as_of),
        ("all_signals", cfg.signal_start_date, None, as_of),
    )
    for scope, start, end, available_through in scopes:
        if scope == "holdout" and holdout_start is None:
            metrics_by_objective = {
                spec.objective: objective_metrics(
                    source.iloc[0:0],
                    pd.Series(dtype=bool),
                    spec.objective,
                    spec.protected_outcome,
                )
                for spec in specs
            }
        else:
            target_metrics, bad_metrics = _sequential_policy_metrics(
                source,
                final_conditions_by_day,
                start=start,
                end=end,
                available_through=available_through,
                complete_only=False,
            )
            metrics_by_objective = {
                **target_metrics,
                SEQUENTIAL_EARLY_SPEC.objective: bad_metrics,
            }
        for spec in specs:
            metrics = metrics_by_objective[spec.objective]
            holdout_pass = None
            if (
                scope == "holdout"
                and int(metrics["sample_count"]) >= cfg.min_holdout_sample_count
            ):
                holdout_pass = (
                    _development_gates(metrics, cfg)
                    if spec == SEQUENTIAL_EARLY_SPEC
                    else _safety_gates(metrics, cfg)
                )
            rows.append(
                _rule_row(
                    rule_id=base_id,
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=final_conditions,
                    evaluation_scope=scope,
                    scope_year=None,
                    period_start=(
                        None if scope == "holdout" and holdout_start is None else start
                    ),
                    period_end=(
                        None
                        if scope == "holdout" and holdout_start is None
                        else (end if end is not None else as_of)
                    ),
                    metrics=metrics,
                    is_selected=True,
                    is_final_filter=True,
                    passes_holdout=holdout_pass,
                    passes_development_gates=(
                        (
                            _development_gates(metrics, cfg)
                            if spec == SEQUENTIAL_EARLY_SPEC
                            else _safety_gates(metrics, cfg)
                        )
                        if scope == "development"
                        else None
                    ),
                    passes_stability_gates=(
                        policy_stability_pass if scope == "development" else None
                    ),
                    stability=(
                        policy_stability if scope == "development" else None
                    ),
                    threshold_fit_end_date=(
                        cfg.validation_end_date
                        if final_conditions
                        and not (scope == "holdout" and holdout_start is None)
                        else None
                    ),
                    rule_text_override=final_policy_text,
                )
            )
            if scope in {"development", "diagnostic", "holdout"}:
                rows.append(
                    _rule_row(
                        rule_id=f"BASELINE_{base_id}_{spec.objective.upper()}",
                        result_kind="baseline",
                        spec=spec,
                        conditions=(),
                        evaluation_scope=scope,
                        scope_year=None,
                        period_start=(
                            None
                            if scope == "holdout" and holdout_start is None
                            else start
                        ),
                        period_end=(
                            None
                            if scope == "holdout" and holdout_start is None
                            else (end if end is not None else as_of)
                        ),
                        metrics=_baseline_metrics(metrics),
                        is_selected=False,
                        is_final_filter=False,
                    )
                )


def build_rule_results(
    signals: pd.DataFrame,
    early_cuts: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    selected: SelectedSummary,
    cfg: Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    holdout_dates = trading_dates[trading_dates.date > cfg.holdout_cutoff_date]
    holdout_start = holdout_dates[0].date() if len(holdout_dates) else None
    as_of = trading_dates[-1].date() if len(trading_dates) else cfg.holdout_cutoff_date
    portfolios: list[tuple[pd.DataFrame, ObjectiveSelection, int]] = []
    for objective, selection in selected.entry.items():
        portfolios.append(
            (signals, selection, selected.entry_prefix_lengths[objective])
        )
    for day, selections in selected.early_cut.items():
        for objective, selection in selections.items():
            portfolios.append(
                (
                    early_cuts,
                    selection,
                    selected.early_cut_prefix_lengths[day][objective],
                )
            )
    for source, selection, prefix_length in portfolios:
        spec = selection.spec
        chosen = selection.prefixes[prefix_length]
        conditions = chosen.final_conditions
        base_id = f"{spec.decision_family.upper()}_{spec.objective.upper()}" + (
            f"_D{spec.landmark_day}" if spec.landmark_day else ""
        )
        # Singleton rows keep every A/B/C/D/E candidate visible without
        # leaking diagnostic or holdout results into selection.
        for candidate in selection.candidates:
            if len(candidate.templates) != 1:
                continue
            rows.append(
                _rule_row(
                    rule_id=candidate.rule_id,
                    result_kind="candidate_rule",
                    spec=spec,
                    conditions=(),
                    evaluation_scope="walk_forward_pooled",
                    scope_year=None,
                    period_start=date(cfg.walk_forward_first_year, 1, 1),
                    period_end=cfg.validation_end_date,
                    metrics=candidate.pooled_metrics,
                    is_selected=candidate.templates[0] in chosen.templates,
                    is_final_filter=False,
                    passes_development_gates=candidate.passes_development_gates,
                    passes_stability_gates=candidate.passes_stability_gates,
                    stability=candidate.stability,
                    selection_order=(
                        chosen.templates.index(candidate.templates[0]) + 1
                        if candidate.templates[0] in chosen.templates
                        else None
                    ),
                    templates=candidate.templates,
                )
            )
        for order, condition in enumerate(conditions, start=1):
            population, component_metrics = _scope_metrics(
                source,
                spec,
                (condition,),
                None,
                cfg.validation_end_date,
                cfg.validation_end_date,
            )
            component_dates = pd.to_datetime(
                population[spec.date_column], errors="coerce"
            )
            rows.append(
                _rule_row(
                    rule_id=condition.rule_id,
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=(condition,),
                    evaluation_scope="development",
                    scope_year=None,
                    period_start=(
                        component_dates.min().date() if len(population) else None
                    ),
                    period_end=cfg.validation_end_date,
                    metrics=component_metrics,
                    is_selected=True,
                    is_final_filter=False,
                    selection_order=order,
                    threshold_fit_end_date=cfg.validation_end_date,
                )
            )
        for fold_result in chosen.fold_results:
            rows.append(
                _rule_row(
                    rule_id=f"{base_id}_FILTER",
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=fold_result.conditions,
                    evaluation_scope="walk_forward_year",
                    scope_year=fold_result.fold.year,
                    period_start=fold_result.fold.start_date,
                    period_end=fold_result.fold.end_date,
                    metrics=fold_result.metrics,
                    is_selected=True,
                    is_final_filter=False,
                    passes_development_gates=None,
                    passes_stability_gates=None,
                    threshold_fit_end_date=(
                        fold_result.fold.threshold_fit_end_date
                        if fold_result.conditions
                        else None
                    ),
                )
            )
        rows.append(
            _rule_row(
                rule_id=f"{base_id}_FILTER",
                result_kind="selected_filter",
                spec=spec,
                conditions=(),
                evaluation_scope="walk_forward_pooled",
                scope_year=None,
                period_start=date(cfg.walk_forward_first_year, 1, 1),
                period_end=cfg.validation_end_date,
                metrics=chosen.pooled_metrics,
                is_selected=True,
                is_final_filter=False,
                passes_development_gates=chosen.passes_development_gates,
                passes_stability_gates=chosen.passes_stability_gates,
                stability=chosen.stability,
                templates=chosen.templates,
            )
        )
        scopes = (
            (
                "development",
                None,
                cfg.signal_start_date,
                cfg.validation_end_date,
                cfg.validation_end_date,
            ),
            (
                "validation",
                None,
                cfg.discovery_end_date + timedelta(days=1),
                cfg.validation_end_date,
                cfg.validation_end_date,
            ),
            (
                "diagnostic",
                None,
                cfg.validation_end_date + timedelta(days=1),
                cfg.holdout_cutoff_date,
                cfg.holdout_cutoff_date,
            ),
            ("holdout", None, holdout_start, None, as_of),
            ("all_signals", None, cfg.signal_start_date, None, as_of),
        )
        for scope, year, start, end, available_through in scopes:
            if scope == "holdout" and holdout_start is None:
                population = _universe(source, spec).iloc[0:0]
                metrics = objective_metrics(
                    population,
                    pd.Series(dtype=bool),
                    spec.objective,
                    spec.protected_outcome,
                )
            else:
                population, metrics = _scope_metrics(
                    source, spec, conditions, start, end, available_through
                )
            holdout_pass = None
            if (
                scope == "holdout"
                and int(metrics["sample_count"]) >= cfg.min_holdout_sample_count
            ):
                holdout_pass = _development_gates(metrics, cfg)
            rows.append(
                _rule_row(
                    rule_id=f"{base_id}_FILTER",
                    result_kind="selected_filter",
                    spec=spec,
                    conditions=conditions,
                    evaluation_scope=scope,
                    scope_year=year,
                    period_start=(
                        start
                        if start is not None
                        else (
                            pd.to_datetime(
                                population[spec.date_column], errors="coerce"
                            )
                            .min()
                            .date()
                            if len(population)
                            else None
                        )
                    ),
                    period_end=(
                        None
                        if scope == "holdout" and holdout_start is None
                        else (end if end is not None else as_of)
                    ),
                    metrics=metrics,
                    is_selected=True,
                    is_final_filter=False,
                    passes_holdout=holdout_pass,
                    passes_development_gates=(
                        _development_gates(metrics, cfg)
                        if scope == "development"
                        else None
                    ),
                    passes_stability_gates=(
                        chosen.passes_stability_gates
                        if scope == "development"
                        else None
                    ),
                    stability=(chosen.stability if scope == "development" else None),
                    threshold_fit_end_date=(
                        cfg.validation_end_date
                        if conditions
                        and not (scope == "holdout" and holdout_start is None)
                        else None
                    ),
                )
            )
    _append_final_union_rows(
        rows,
        signals,
        selected.entry,
        selected.entry_prefix_lengths,
        trading_dates,
        cfg,
    )
    for day, day_selections in selected.early_cut.items():
        _append_final_union_rows(
            rows,
            early_cuts,
            day_selections,
            selected.early_cut_prefix_lengths[day],
            trading_dates,
            cfg,
            mark_final=False,
        )
    _append_sequential_policy_rows(
        rows,
        early_cuts,
        selected.early_cut,
        selected.early_cut_prefix_lengths,
        trading_dates,
        cfg,
    )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=RULE_COLUMNS)
    missing = sorted(set(RULE_COLUMNS) - set(result.columns))
    if missing:
        raise AssertionError("rule results missing columns: " + ", ".join(missing))
    return (
        result.loc[:, RULE_COLUMNS]
        .sort_values(
            [
                "decision_family",
                "landmark_day",
                "objective",
                "result_kind",
                "rule_id",
                "evaluation_scope",
                "scope_year",
            ],
            kind="stable",
            na_position="first",
        )
        .reset_index(drop=True)
    )


def run_research(
    signals: pd.DataFrame,
    early_cuts: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> ResearchResult:
    if signals.empty:
        raise RuntimeError("no false-to-true signals were found")
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")
    signals = signals.sort_values(
        ["signal_date", "symbol", "exchange", "cik"], kind="stable"
    ).reset_index(drop=True)
    if signals.duplicated(["signal_date", "symbol", "exchange", "cik"]).any():
        raise RuntimeError("signal results contain duplicate primary keys")
    early_cuts = early_cuts.sort_values(
        ["signal_date", "landmark_day", "symbol", "exchange", "cik"], kind="stable"
    ).reset_index(drop=True)
    if (
        not early_cuts.empty
        and early_cuts.duplicated(
            ["signal_date", "landmark_day", "symbol", "exchange", "cik"]
        ).any()
    ):
        raise RuntimeError("early-cut results contain duplicate primary keys")
    entry_templates = build_candidate_templates(
        "entry_filter", quantile_count=cfg.quantile_count
    )
    entry = {
        spec.objective: select_objective(signals, spec, entry_templates, cfg)
        for spec in ENTRY_SPECS
    }
    entry_frame = _period_frame(signals, ENTRY_SPECS[0], None, cfg.validation_end_date)
    entry_lengths = _choose_prefixes(
        entry_frame,
        entry,
        cfg,
        require_cross_objective_lift=False,
    )
    entry_lengths = _gate_refit_prefixes(signals, entry, entry_lengths, cfg)
    early: dict[int, dict[str, ObjectiveSelection]] = {}
    early_lengths: dict[int, dict[str, int]] = {}
    for day in EARLY_CUT_LANDMARK_DAYS:
        templates = build_candidate_templates(
            "early_cut", day, quantile_count=cfg.quantile_count
        )
        selections = {
            spec.objective: select_objective(early_cuts, spec, templates, cfg)
            for spec in early_specs(day)
        }
        early[day] = selections
        day_frame = _period_frame(
            _universe(early_cuts, early_specs(day)[0]),
            early_specs(day)[0],
            None,
            cfg.validation_end_date,
        )
        early_lengths[day] = _choose_prefixes(
            day_frame,
            selections,
            cfg,
            require_cross_objective_lift=False,
        )
        early_lengths[day] = _gate_refit_prefixes(
            early_cuts, selections, early_lengths[day], cfg
        )
    early_lengths = _choose_sequential_early_prefixes(
        early_cuts, early, early_lengths, cfg
    )
    summary = SelectedSummary(entry, early, entry_lengths, early_lengths)
    decided_signals = apply_entry_decisions(signals, summary)
    decided_early = apply_early_decisions(early_cuts, summary)
    rules = build_rule_results(
        decided_signals, decided_early, trading_dates, summary, cfg
    )
    return ResearchResult(
        decided_signals,
        decided_early,
        rules,
        summary,
        summary.condition_count,
        summary.summary_text,
    )
