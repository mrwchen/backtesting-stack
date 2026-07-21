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
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)


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


@dataclass(frozen=True)
class ConditionTemplate:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    quantile: float


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
class WalkForwardFold:
    year: int
    start_date: date
    end_date: date
    threshold_fit_end_date: date


@dataclass
class FoldResult:
    fold: WalkForwardFold
    conditions: tuple[Condition, ...]
    metrics: dict[str, object]


@dataclass
class CandidateEvaluation:
    templates: tuple[ConditionTemplate, ...]
    fold_results: tuple[FoldResult, ...]
    pooled_metrics: dict[str, object]
    final_conditions: tuple[Condition, ...]
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


def _combined_mask(frame: pd.DataFrame, conditions: Iterable[Condition]) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for condition in conditions:
        result |= condition.matches(frame)
    return result


def build_candidate_templates(
    decision_family: str,
    landmark_day: int | None = None,
    quantile_count: int = 20,
) -> tuple[ConditionTemplate, ...]:
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
    templates: list[ConditionTemplate] = []
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
    return tuple(sorted(templates, key=lambda item: item.rule_id))


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
        templates: tuple[ConditionTemplate, ...],
        cfg: Config,
    ):
        self.frame = _universe(frame, spec)
        self.spec = spec
        self.templates = templates
        self.cfg = cfg
        self.folds = build_walk_forward_folds(cfg)
        self._fit_cache: dict[tuple[int | str, str], Condition | None] = {}
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
                    for item in self.templates
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

    def evaluate(self, templates: tuple[ConditionTemplate, ...]) -> CandidateEvaluation:
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
                if (condition := self._fit(template, fold)) is not None
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
            if (condition := self._fit(template, None)) is not None
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
            pooled["objective_capture_rate"],
        )
        self._eval_cache[key] = result
        return result


def _rank(candidate: CandidateEvaluation) -> tuple[object, ...]:
    metrics = candidate.pooled_metrics
    stability = candidate.stability

    def desc(value: object) -> float:
        return float("inf") if value is None else -float(value)

    return (
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
    templates: tuple[ConditionTemplate, ...],
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
                added.feature_name == first.feature_name
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
            base_capture = float(
                first_eval.pooled_metrics["objective_capture_rate"] or 0.0
            )
            pair_capture = float(pair.pooled_metrics["objective_capture_rate"] or 0.0)
            if (
                pair.passes_development_gates
                and pair.passes_stability_gates
                and pair_capture >= base_capture + cfg.min_selection_capture_improvement
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
                stable &= _stability(union_folds, cfg)[0]
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


def _row_matches(
    frame: pd.DataFrame, conditions: tuple[Condition, ...]
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
    prior_state = result["cut_decision"].astype("string")
    preserved = prior_state.isin(("not_eligible", "not_evaluable"))
    result["cut_decision"] = prior_state.where(preserved, "not_evaluable")
    eligible = _truthy(result["eligible_at_landmark"])
    result.loc[eligible, "cut_decision"] = "hold"
    result.loc[eligible, "include_stagnation_filter"] = True
    result.loc[eligible, "include_loss_filter"] = True
    result.loc[eligible, "include_final"] = True
    for day, objective_selections in selected.early_cut.items():
        rows = pd.to_numeric(result["landmark_day"], errors="coerce").eq(day) & eligible
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
    return result.loc[:, EARLY_CUT_COLUMNS]


def _rule_text(conditions: tuple[Condition, ...]) -> str:
    return " OR ".join(item.text for item in conditions) or "no exclusion condition"


def _feature_fields(
    conditions: tuple[Condition, ...],
) -> tuple[str, str | None, str | None, float | None, float | None]:
    if not conditions:
        return "none", None, None, None, None
    if len(conditions) == 1:
        item = conditions[0]
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
    templates: tuple[ConditionTemplate, ...],
) -> tuple[str, str | None, str | None, float | None, float | None]:
    if not templates:
        return "none", None, None, None, None
    if len(templates) == 1:
        item = templates[0]
        return item.feature_group, item.feature_name, item.operator, item.quantile, None
    groups = {item.feature_group for item in templates}
    return (
        next(iter(groups)) if len(groups) == 1 else "multiple",
        None,
        None,
        None,
        None,
    )


def _template_text(templates: tuple[ConditionTemplate, ...]) -> str:
    parts = []
    for item in templates:
        symbol = "<=" if item.operator == "le" else ">="
        parts.append(
            f"{item.feature_name} {symbol} causal Q{item.quantile:.4g} threshold"
        )
    return " OR ".join(parts) or "no exclusion condition"


def _rule_row(
    *,
    rule_id: str,
    result_kind: str,
    spec: ObjectiveSpec,
    conditions: tuple[Condition, ...],
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
    templates: tuple[ConditionTemplate, ...] = (),
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
        "rule_text": (
            _rule_text(conditions) if conditions else _template_text(templates)
        ),
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
        "component_count": len(conditions) if conditions else len(templates),
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
        "selection_score": metrics.get("objective_capture_rate"),
    }
    return row


def _scope_metrics(
    frame: pd.DataFrame,
    spec: ObjectiveSpec,
    conditions: tuple[Condition, ...],
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


def _unique_conditions(items: Iterable[Condition]) -> tuple[Condition, ...]:
    result: list[Condition] = []
    seen: set[str] = set()
    for item in items:
        if item.rule_id not in seen:
            seen.add(item.rule_id)
            result.append(item)
    return tuple(result)


def _unique_templates(
    items: Iterable[ConditionTemplate],
) -> tuple[ConditionTemplate, ...]:
    result: list[ConditionTemplate] = []
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
        else f"EARLY_CUT_D{first.landmark_day}_FINAL"
    )
    holdout_dates = trading_dates[trading_dates.date > cfg.holdout_cutoff_date]
    holdout_start = holdout_dates[0].date() if len(holdout_dates) else None
    as_of = trading_dates[-1].date() if len(trading_dates) else cfg.holdout_cutoff_date
    union_fold_conditions: list[tuple[Condition, ...]] = []
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
                    is_final_filter=True,
                    threshold_fit_end_date=(
                        fold.threshold_fit_end_date if conditions else None
                    ),
                )
            )
        pooled = _aggregate_metrics(item.metrics for item in union_fold_results)
        stability_pass, stability = _stability(union_fold_results, cfg)
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
                    is_final_filter=True,
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
        # Candidate singleton pooled rows keep every A/B/C/E template visible without leaking diagnostics into selection.
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
    entry_lengths = _choose_prefixes(entry_frame, entry, cfg)
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
        early_lengths[day] = _choose_prefixes(day_frame, selections, cfg)
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
