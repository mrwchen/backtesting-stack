from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import logging
from math import ceil
import re
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    EARLY_CUT_COLUMNS,
    EARLY_CUT_FEATURE_GROUPS,
    EARLY_CUT_LANDMARK_DAYS,
    MANAGEMENT_LANDMARK_DAYS,
    POSITION_LANDMARK_DAYS,
    ENTRY_FEATURE_GROUPS,
    IDENTITY_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)


log = logging.getLogger(__name__)


POLICY_KEY_COLUMNS = ("signal_date", *IDENTITY_COLUMNS)


@dataclass(frozen=True)
class ObjectiveSpec:
    decision_family: str
    objective: str
    protected_outcome: str
    landmark_day: int | None = None
    label_end_column_name: str | None = None

    @property
    def date_column(self) -> str:
        return (
            "signal_date"
            if self.decision_family.startswith("entry_")
            else "landmark_date"
        )

    @property
    def label_end_column(self) -> str:
        if self.label_end_column_name is not None:
            return self.label_end_column_name
        return (
            "forward_5d_label_end_date"
            if self.decision_family.startswith("entry_")
            else "horizon_end_date"
        )


ENTRY_EXCLUSION_SPECS = (
    ObjectiveSpec("entry_filter", "weak_5d", "strong_first_5d"),
    ObjectiveSpec("entry_filter", "loss_first_5d", "strong_first_5d"),
    ObjectiveSpec("entry_filter", "terminal_stagnant_5d", "terminal_winner_5d"),
    ObjectiveSpec(
        "entry_filter",
        "stagnant_5d",
        "terminal_winner_20d",
        label_end_column_name="forward_20d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_filter",
        "hard_stop_10pct_5d",
        "runner_60d",
        label_end_column_name="forward_60d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_filter",
        "terminal_nonpositive_20d",
        "terminal_winner_20d",
        label_end_column_name="forward_20d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_filter",
        "terminal_nonpositive_30d",
        "terminal_winner_30d",
        label_end_column_name="forward_30d_label_end_date",
    ),
)

ENTRY_CONFIRMATION_SPECS = (
    ObjectiveSpec("entry_confirmation", "strong_first_5d", "bad_5d"),
    ObjectiveSpec(
        "entry_confirmation", "terminal_winner_5d", "terminal_stagnant_5d"
    ),
    ObjectiveSpec(
        "entry_confirmation",
        "terminal_winner_20d",
        "terminal_nonpositive_20d",
        label_end_column_name="forward_20d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_confirmation",
        "terminal_winner_30d",
        "terminal_nonpositive_30d",
        label_end_column_name="forward_30d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_confirmation",
        "runner_60d",
        "terminal_nonpositive_30d",
        label_end_column_name="forward_60d_label_end_date",
    ),
    ObjectiveSpec(
        "entry_confirmation",
        "runner_90d",
        "terminal_nonpositive_30d",
        label_end_column_name="forward_90d_label_end_date",
    ),
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


def management_specs(landmark_day: int) -> tuple[ObjectiveSpec, ...]:
    if landmark_day == 5:
        horizons = (20,)
    elif landmark_day in (20, 30):
        horizons = (40, 60, 90)
    else:
        raise ValueError("position-management landmark must be D+5, D+20 or D+30")
    return tuple(
        ObjectiveSpec(
            "position_management",
            f"take_profit_better_to_day{horizon}",
            f"continue_winner_to_day{horizon}",
            landmark_day,
            f"day{horizon}_end_date",
        )
        for horizon in horizons
    )


@dataclass(frozen=True)
class ConditionTemplate:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    quantile: float | None
    minimum_threshold: float | None = None
    fixed_threshold: float | None = None


@dataclass(frozen=True)
class Condition:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    threshold: float
    quantile: float | None
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
    minimum_clause_count: int | None = None

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError("pattern template requires at least one clause")
        if self.minimum_clause_count is not None and not (
            2 <= self.minimum_clause_count <= len(self.clauses)
        ):
            raise ValueError("pattern minimum clause count is invalid")


@dataclass(frozen=True)
class CompositeCondition:
    rule_id: str
    feature_group: str
    pattern_name: str
    clauses: tuple[Condition, ...]
    minimum_clause_count: int | None = None

    def __post_init__(self) -> None:
        if not self.clauses:
            raise ValueError("composite condition requires at least one clause")
        if self.minimum_clause_count is not None and not (
            2 <= self.minimum_clause_count <= len(self.clauses)
        ):
            raise ValueError("composite minimum clause count is invalid")

    @property
    def text(self) -> str:
        if self.minimum_clause_count is None:
            expression = " AND ".join(clause.text for clause in self.clauses)
        else:
            expression = (
                f"at least {self.minimum_clause_count} of {len(self.clauses)}: "
                + "; ".join(clause.text for clause in self.clauses)
            )
        return f"{self.pattern_name} ({expression})"

    @property
    def threshold_fit_end_date(self) -> date | None:
        dates = [
            clause.threshold_fit_end_date
            for clause in self.clauses
            if clause.threshold_fit_end_date is not None
        ]
        return max(dates) if dates else None

    def matches(self, frame: pd.DataFrame) -> pd.Series:
        masks = [clause.matches(frame) for clause in self.clauses]
        available = pd.Series(True, index=frame.index, dtype=bool)
        for clause in self.clauses:
            available &= _finite_numeric(frame, clause.feature_name).notna()
        if self.minimum_clause_count is None:
            result = pd.Series(True, index=frame.index, dtype=bool)
            for mask in masks:
                result &= mask
            return available & result
        matched_count = sum(mask.astype(np.int16) for mask in masks)
        return available & matched_count.ge(self.minimum_clause_count)


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
    multiple_testing_candidate_count: int | None = None
    permutation_trial_count: int | None = None
    max_stat_permutation_p_value: float | None = None
    passes_multiple_testing: bool | None = None

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
    confirmation: dict[str, ObjectiveSelection]
    early_cut: dict[int, dict[str, ObjectiveSelection]]
    management: dict[int, dict[str, ObjectiveSelection]]
    entry_prefix_lengths: dict[str, int]
    confirmation_prefix_lengths: dict[str, int]
    early_cut_prefix_lengths: dict[int, dict[str, int]]
    management_prefix_lengths: dict[int, dict[str, int]]

    @property
    def condition_count(self) -> int:
        return (
            sum(self.entry_prefix_lengths.values())
            + sum(self.confirmation_prefix_lengths.values())
            + sum(
            sum(values.values()) for values in self.early_cut_prefix_lengths.values()
            )
            + sum(
                sum(values.values())
                for values in self.management_prefix_lengths.values()
            )
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
        for objective, selection in self.confirmation.items():
            length = self.confirmation_prefix_lengths.get(objective, 0)
            text = (
                " OR ".join(
                    condition.text
                    for condition in selection.selected.final_conditions[:length]
                )
                or "no confirmation"
            )
            parts.append(f"confirmation/{objective}: {text}")
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
        for day in sorted(self.management):
            for objective, selection in self.management[day].items():
                length = self.management_prefix_lengths.get(day, {}).get(
                    objective, 0
                )
                text = (
                    " OR ".join(
                        condition.text
                        for condition in selection.selected.final_conditions[:length]
                    )
                    or "no filter"
                )
                parts.append(f"management/day{day}/{objective}: {text}")
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
    if decision_family in {"entry_filter", "entry_confirmation"}:
        groups = ENTRY_FEATURE_GROUPS
        prefix = "ENTRY" if decision_family == "entry_filter" else "CONFIRM"
    elif decision_family == "early_cut":
        if landmark_day not in EARLY_CUT_LANDMARK_DAYS:
            raise ValueError("early-cut templates require landmark day 1, 2, or 3")
        groups = EARLY_CUT_FEATURE_GROUPS
        prefix = f"EARLY_D{landmark_day}"
    elif decision_family == "position_management":
        if landmark_day not in MANAGEMENT_LANDMARK_DAYS:
            raise ValueError(
                "position-management templates require landmark day 5, 20, or 30"
            )
        groups = EARLY_CUT_FEATURE_GROUPS
        prefix = f"MANAGE_D{landmark_day}"
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
    activity_features = {
        feature
        for features in groups.values()
        for feature in features
        if (
            "volume_vs_sma" in feature
            or "notional_vs_sma" in feature
        )
        and any(f"sma{window}" in feature for window in (7, 14, 21, 50, 100))
    }
    fixed_grid = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
    feature_group_by_name = {
        feature: group for group, features in groups.items() for feature in features
    }
    for feature in sorted(activity_features):
        group = feature_group_by_name[feature]
        for threshold in fixed_grid:
            threshold_tag = str(threshold).replace(".", "P")
            for operator in ("le", "ge"):
                side = "LE" if operator == "le" else "GE"
                templates.append(
                    ConditionTemplate(
                        f"{prefix}_{group}_{feature}_{side}_FIXED_{threshold_tag}",
                        group,
                        feature,
                        operator,
                        None,
                        fixed_threshold=threshold,
                    )
                )
    score_features = {
        feature
        for features in groups.values()
        for feature in features
        if feature.startswith("pattern_") and "_score" in feature
    }
    for feature in sorted(score_features):
        group = feature_group_by_name[feature]
        for threshold in (40.0, 50.0, 60.0, 70.0, 80.0, 90.0):
            for operator in ("le", "ge"):
                side = "LE" if operator == "le" else "GE"
                templates.append(
                    ConditionTemplate(
                        f"{prefix}_{group}_{feature}_{side}_FIXED_{int(threshold)}",
                        group,
                        feature,
                        operator,
                        None,
                        fixed_threshold=threshold,
                    )
                )
    if decision_family in {"entry_filter", "entry_confirmation"}:
        templates.extend(_entry_pattern_templates())
        templates.extend(_interaction_pattern_templates(prefix))
    return tuple(sorted(templates, key=lambda item: item.rule_id))


def _relaxed_pattern_variants(
    patterns: tuple[PatternTemplate, ...],
) -> tuple[PatternTemplate, ...]:
    """Keep exact patterns and add causal k-of-n sensitivity variants."""

    variants: list[PatternTemplate] = []
    for strict in patterns:
        variants.append(strict)
        clause_count = len(strict.clauses)
        for required in range(max(2, clause_count - 2), clause_count):
            variants.append(
                PatternTemplate(
                    f"{strict.rule_id}__K{required}_OF_{clause_count}",
                    strict.feature_group,
                    f"{strict.pattern_name}_k{required}_of_{clause_count}",
                    strict.clauses,
                    minimum_clause_count=required,
                )
            )
    return tuple(variants)


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

    chart_patterns = (
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
    return (
        _relaxed_pattern_variants(chart_patterns)
        + _fundamental_pattern_templates()
        + _market_cap_pattern_templates()
        + _relaxed_pattern_variants(_trader_pattern_templates())
    )


def _trader_pattern_templates() -> tuple[PatternTemplate, ...]:
    """Predeclared Minervini/Ryan/O'Neil/Zanger/Weinstein-style templates."""

    def build(
        group: str,
        name: str,
        clauses: tuple[
            tuple[str, str, float | None, float | None], ...
        ],
    ) -> PatternTemplate:
        rule_id = f"ENTRY_{group}_PATTERN_{name}"
        return PatternTemplate(
            rule_id,
            group,
            name.lower(),
            tuple(
                ConditionTemplate(
                    f"{rule_id}_C{number}",
                    group,
                    feature,
                    operator,
                    quantile,
                    fixed_threshold=fixed,
                )
                for number, (feature, operator, quantile, fixed) in enumerate(
                    clauses, start=1
                )
            ),
        )

    return (
        build(
            "T",
            "VCP",
            (
                ("prior_atr_5_vs21_ratio", "le", 0.20, None),
                ("prior_contraction_count_40", "ge", 0.80, None),
                ("prior_volume_sma5_vs50_ratio", "le", 0.20, None),
                ("prior_close_vs_63d_high_pct", "ge", 0.80, None),
            ),
        ),
        build(
            "T",
            "HIGH_TIGHT_FLAG",
            (
                ("prior_return_42d_pct", "ge", 0.90, None),
                ("prior_base_width_10_pct", "le", 0.20, None),
                ("prior_tight_close_range_5_pct", "le", 0.20, None),
                ("prior_close_vs_63d_high_pct", "ge", 0.80, None),
            ),
        ),
        build(
            "T",
            "BULL_FLAG",
            (
                ("prior_return_21d_pct", "ge", 0.80, None),
                ("prior_base_width_10_pct", "le", 0.30, None),
                ("prior_pullback_from_40d_high_pct", "ge", 0.30, None),
                ("prior_close_vs_63d_high_pct", "ge", 0.70, None),
            ),
        ),
        build(
            "T",
            "DARVAS_BOX",
            (
                ("prior_base_width_20_pct", "le", 0.20, None),
                ("prior_high_test_count_20", "ge", 0.70, None),
                ("prior_tight_close_range_10_pct", "le", 0.30, None),
                ("signal_close_vs_prior_20d_high_pct", "ge", 0.80, None),
            ),
        ),
        build(
            "T",
            "ASCENDING_TRIANGLE",
            (
                ("prior_high_slope_20_pct_per_session", "le", 0.30, None),
                ("prior_low_slope_20_pct_per_session", "ge", 0.70, None),
                ("prior_high_test_count_20", "ge", 0.70, None),
                ("signal_close_vs_prior_20d_high_pct", "ge", 0.80, None),
            ),
        ),
        build(
            "T",
            "CUP_WITH_HANDLE",
            (
                ("prior_max_drawdown_126d_pct", "le", 0.20, None),
                ("prior_close_vs_126d_high_pct", "ge", 0.70, None),
                ("prior_tight_close_range_15_pct", "le", 0.30, None),
                ("prior_volume_sma5_vs50_ratio", "le", 0.30, None),
            ),
        ),
        build(
            "T",
            "THREE_WEEKS_TIGHT",
            (
                ("prior_tight_close_range_15_pct", "le", 0.20, None),
                ("prior_close_vs_63d_high_pct", "ge", 0.80, None),
                ("prior_volume_sma5_vs50_ratio", "le", 0.30, None),
            ),
        ),
        build(
            "S",
            "POCKET_PIVOT",
            (
                (
                    "signal_volume_vs_prior_10d_max_down_volume_ratio",
                    "ge",
                    None,
                    1.0,
                ),
                ("daily_price_change_pct", "ge", None, 0.0),
                ("signal_close_location_value", "ge", 0.70, None),
            ),
        ),
        build(
            "R",
            "RS_LEADER",
            (
                ("rs_rating", "ge", None, 90.0),
                ("prior_rs_rating_change_21d", "ge", 0.70, None),
                ("relative_return_vs_spy_63d_pct_points", "ge", 0.70, None),
                ("prior_close_vs_63d_high_pct", "ge", 0.80, None),
            ),
        ),
        build(
            "T",
            "WEINSTEIN_STAGE2",
            (
                ("distance_to_ma150_pct", "ge", 0.70, None),
                ("ma150_slope_21d_pct", "ge", 0.70, None),
                ("ma200_slope_63d_pct", "ge", 0.70, None),
                ("cross_sectional_rs_126d_pct_rank", "ge", 0.80, None),
            ),
        ),
        build(
            "S",
            "ZANGER_VOLUME_BREAKOUT",
            (
                ("adjusted_volume_vs_sma21_prior_ratio", "ge", 0.80, None),
                ("signal_close_location_value", "ge", 0.70, None),
                ("signal_close_vs_prior_20d_high_pct", "ge", 0.80, None),
                ("prior_volume_dryup_share10", "ge", 0.70, None),
            ),
        ),
        build(
            "F",
            "GROWTH_LEADER",
            (
                ("fundamental_quarterly_revenue_yoy_growth_ratio", "ge", 0.80, None),
                ("fundamental_quarterly_eps_yoy_growth_ratio", "ge", 0.80, None),
                ("fundamental_quarterly_revenue_growth_acceleration", "ge", 0.70, None),
                ("fundamental_roe_ttm_ratio", "ge", 0.70, None),
            ),
        ),
        build(
            "F",
            "QUALITY_GROWTH",
            (
                ("fundamental_roe_ttm_ratio", "ge", 0.70, None),
                ("fundamental_cash_conversion_ttm_ratio", "ge", 0.70, None),
                ("fundamental_debt_to_assets_ratio", "le", 0.30, None),
                ("fundamental_accruals_ratio", "le", 0.30, None),
            ),
        ),
        build(
            "N",
            "POST_EARNINGS_POWER",
            (
                ("earnings_event_age_days", "le", None, 5.0),
                ("signal_gap_pct", "ge", 0.70, None),
                ("adjusted_volume_vs_sma21_prior_ratio", "ge", 0.80, None),
                ("signal_close_location_value", "ge", 0.70, None),
            ),
        ),
        build(
            "R",
            "MARKET_CONFIRMED_LEADER",
            (
                ("market_breadth_above_ma50_ratio", "ge", 0.70, None),
                ("market_breadth_trend_template_ratio", "ge", 0.70, None),
                ("cross_sectional_rs_63d_pct_rank", "ge", 0.80, None),
                ("market_vix_to_vix3m_ratio", "le", 0.30, None),
            ),
        ),
    )


def _interaction_pattern_templates(prefix: str) -> tuple[PatternTemplate, ...]:
    """Generate all tail combinations for predeclared causal feature pairs."""

    pairs = (
        ("log_market_cap_usd", "prior_atr_14d_pct"),
        ("log_market_cap_usd", "adjusted_volume_vs_sma21_prior_ratio"),
        ("log_market_cap_usd", "daily_traded_notional_vs_sma21_prior_ratio"),
        ("log_market_cap_usd", "cross_sectional_rs_63d_pct_rank"),
        ("prior_atr_14d_pct", "adjusted_volume_vs_sma21_prior_ratio"),
        ("prior_atr_5_vs21_ratio", "prior_volume_sma5_vs50_ratio"),
        ("prior_atr_5_vs21_ratio", "prior_base_width_20_pct"),
        ("prior_base_width_20_pct", "prior_volume_dryup_share10"),
        ("prior_base_width_20_pct", "signal_close_vs_prior_20d_high_pct"),
        ("prior_tight_close_range_5_pct", "prior_close_vs_63d_high_pct"),
        ("prior_contraction_count_40", "prior_volume_dryup_share20"),
        ("prior_distribution_day_count_20", "prior_failed_breakout_count_20"),
        ("fundamental_quarterly_revenue_yoy_growth_ratio", "cross_sectional_rs_63d_pct_rank"),
        ("fundamental_quarterly_eps_yoy_growth_ratio", "cross_sectional_rs_63d_pct_rank"),
        ("fundamental_quarterly_revenue_growth_acceleration", "adjusted_volume_vs_sma21_prior_ratio"),
        ("fundamental_roe_ttm_ratio", "fundamental_cash_conversion_ttm_ratio"),
        ("fundamental_debt_to_assets_ratio", "fundamental_fcf_margin_ttm_ratio"),
        ("earnings_event_age_days", "adjusted_volume_vs_sma21_prior_ratio"),
        ("earnings_event_age_days", "signal_gap_pct"),
        ("market_breadth_above_ma50_ratio", "cross_sectional_rs_63d_pct_rank"),
        ("market_breadth_trend_template_ratio", "prior_trend_slope_20_pct_per_session"),
        ("market_vix_level", "prior_atr_14d_pct"),
        ("market_vix_to_vix3m_ratio", "prior_base_width_20_pct"),
        ("relative_return_vs_spy_63d_pct_points", "adjusted_volume_vs_sma21_prior_ratio"),
        ("signal_turnover_ratio", "signal_close_location_value"),
    )
    templates: list[PatternTemplate] = []
    for pair_number, (left, right) in enumerate(pairs, start=1):
        for left_operator, left_quantile, left_tag in (
            ("le", 0.20, "LL"),
            ("ge", 0.80, "LH"),
        ):
            for right_operator, right_quantile, right_tag in (
                ("le", 0.20, "RL"),
                ("ge", 0.80, "RH"),
            ):
                rule_id = (
                    f"{prefix}_I_PAIR_{pair_number:02d}_{left_tag}_{right_tag}"
                )
                templates.append(
                    PatternTemplate(
                        rule_id,
                        "I",
                        f"{left}_{left_operator}_{right}_{right_operator}",
                        (
                            ConditionTemplate(
                                f"{rule_id}_C1",
                                "I",
                                left,
                                left_operator,
                                left_quantile,
                            ),
                            ConditionTemplate(
                                f"{rule_id}_C2",
                                "I",
                                right,
                                right_operator,
                                right_quantile,
                            ),
                        ),
                    )
                )
    return tuple(templates)


def _fundamental_pattern_templates() -> tuple[PatternTemplate, ...]:
    def pattern(
        name: str,
        clauses: tuple[tuple[str, str, float], ...],
    ) -> PatternTemplate:
        rule_id = f"ENTRY_F_PATTERN_{name}"
        return PatternTemplate(
            rule_id,
            "F",
            name.lower(),
            tuple(
                ConditionTemplate(
                    f"{rule_id}_C{number}",
                    "F",
                    feature_name,
                    operator,
                    quantile,
                )
                for number, (feature_name, operator, quantile) in enumerate(
                    clauses, start=1
                )
            ),
        )

    return (
        pattern(
            "EARNINGS_DETERIORATION",
            (
                ("fundamental_quarterly_revenue_yoy_growth_ratio", "le", 0.30),
                ("fundamental_quarterly_eps_yoy_change_ratio", "le", 0.30),
                (
                    "fundamental_quarterly_operating_margin_yoy_change",
                    "le",
                    0.30,
                ),
            ),
        ),
        pattern(
            "MARGIN_COMPRESSION",
            (
                ("fundamental_operating_margin_ttm_ratio", "le", 0.30),
                ("fundamental_fcf_margin_ttm_ratio", "le", 0.30),
                (
                    "fundamental_quarterly_operating_margin_yoy_change",
                    "le",
                    0.30,
                ),
            ),
        ),
        pattern(
            "BALANCE_SHEET_STRESS",
            (
                ("fundamental_debt_to_capital_ratio", "ge", 0.70),
                ("fundamental_cash_to_assets_ratio", "le", 0.30),
                ("fundamental_current_ratio", "le", 0.30),
            ),
        ),
        pattern(
            "CASHFLOW_WEAKNESS",
            (
                ("fundamental_fcf_margin_ttm_ratio", "le", 0.30),
                (
                    "fundamental_fcf_sbc_adjusted_margin_ttm_ratio",
                    "le",
                    0.30,
                ),
                ("fundamental_accruals_ratio", "ge", 0.70),
            ),
        ),
    )


def _market_cap_pattern_templates() -> tuple[PatternTemplate, ...]:
    def clause(
        rule_id: str,
        number: int,
        feature_name: str,
        operator: str,
        *,
        quantile: float | None = None,
        fixed_threshold: float | None = None,
    ) -> ConditionTemplate:
        return ConditionTemplate(
            f"{rule_id}_C{number}",
            "M",
            feature_name,
            operator,
            quantile,
            fixed_threshold=fixed_threshold,
        )

    large_cap_rule = "ENTRY_M_PATTERN_LARGE_CAP_LOW_ATR_STAGNATION"
    small_cap_rule = "ENTRY_M_PATTERN_SMALL_CAP_BEARISH_HIGH_VOLUME"
    confirmation_rule = "ENTRY_M_PATTERN_HIGH_VOLUME_STRONG_CLOSE"
    legacy_patterns = (
        PatternTemplate(
            large_cap_rule,
            "M",
            "large_cap_low_atr_stagnation",
            (
                clause(
                    large_cap_rule,
                    1,
                    "market_cap_usd",
                    "ge",
                    fixed_threshold=10_000_000_000.0,
                ),
                clause(
                    large_cap_rule,
                    2,
                    "prior_atr_14d_pct",
                    "le",
                    quantile=0.30,
                ),
            ),
        ),
        PatternTemplate(
            small_cap_rule,
            "M",
            "small_cap_bearish_high_volume",
            (
                clause(
                    small_cap_rule,
                    1,
                    "market_cap_usd",
                    "le",
                    fixed_threshold=3_000_000_000.0,
                ),
                clause(
                    small_cap_rule,
                    2,
                    "adjusted_volume_vs_sma21_prior_ratio",
                    "ge",
                    quantile=0.70,
                ),
                clause(
                    small_cap_rule,
                    3,
                    "daily_price_change_pct",
                    "le",
                    fixed_threshold=0.0,
                ),
                clause(
                    small_cap_rule,
                    4,
                    "signal_close_location_value",
                    "le",
                    fixed_threshold=0.35,
                ),
            ),
        ),
        PatternTemplate(
            confirmation_rule,
            "M",
            "high_volume_strong_close",
            (
                clause(
                    confirmation_rule,
                    1,
                    "adjusted_volume_vs_sma21_prior_ratio",
                    "ge",
                    quantile=0.70,
                ),
                clause(
                    confirmation_rule,
                    2,
                    "daily_price_change_pct",
                    "ge",
                    fixed_threshold=0.0,
                ),
                clause(
                    confirmation_rule,
                    3,
                    "signal_close_location_value",
                    "ge",
                    fixed_threshold=0.65,
                ),
            ),
        ),
    )
    bands = (
        ("MICRO_CAP", None, 299_999_999.0),
        ("SMALL_CAP", 300_000_000.0, 1_999_999_999.0),
        ("MID_CAP", 2_000_000_000.0, 9_999_999_999.0),
        ("LARGE_CAP", 10_000_000_000.0, 199_999_999_999.0),
        ("MEGA_CAP", 200_000_000_000.0, None),
    )
    band_patterns: list[PatternTemplate] = []
    for name, lower, upper in bands:
        rule_id = f"ENTRY_M_PATTERN_{name}_RANGE"
        clauses: list[ConditionTemplate] = []
        if lower is not None:
            clauses.append(
                clause(
                    rule_id,
                    len(clauses) + 1,
                    "market_cap_usd",
                    "ge",
                    fixed_threshold=lower,
                )
            )
        if upper is not None:
            clauses.append(
                clause(
                    rule_id,
                    len(clauses) + 1,
                    "market_cap_usd",
                    "le",
                    fixed_threshold=upper,
                )
            )
        band_patterns.append(
            PatternTemplate(
                rule_id,
                "M",
                f"{name.lower()}_range",
                tuple(clauses),
            )
        )
        for activity_name, activity_feature in (
            ("VOLUME21", "adjusted_volume_vs_sma21_prior_ratio"),
            ("NOTIONAL21", "daily_traded_notional_vs_sma21_prior_ratio"),
        ):
            for operator, quantile, tail in (
                ("le", 0.20, "LOW"),
                ("ge", 0.80, "HIGH"),
            ):
                interaction_id = (
                    f"ENTRY_I_PATTERN_{name}_{activity_name}_{tail}"
                )
                interaction_clauses = [
                    ConditionTemplate(
                        f"{interaction_id}_C{position}",
                        "I",
                        item.feature_name,
                        item.operator,
                        item.quantile,
                        fixed_threshold=item.fixed_threshold,
                    )
                    for position, item in enumerate(clauses, start=1)
                ]
                interaction_clauses.append(
                    ConditionTemplate(
                        f"{interaction_id}_C{len(interaction_clauses) + 1}",
                        "I",
                        activity_feature,
                        operator,
                        quantile,
                    )
                )
                band_patterns.append(
                    PatternTemplate(
                        interaction_id,
                        "I",
                        (
                            f"{name.lower()}_range_"
                            f"{activity_name.lower()}_{tail.lower()}"
                        ),
                        tuple(interaction_clauses),
                    )
                )
    return (*legacy_patterns, *band_patterns)


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
    if template.fixed_threshold is not None:
        threshold = float(template.fixed_threshold)
    elif template.quantile is not None:
        threshold = float(
            np.quantile(values.to_numpy(), template.quantile, method="nearest")
        )
    else:
        return None
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
    if spec.decision_family in {"early_cut", "position_management"}:
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
    ends = pd.to_datetime(frame[spec.label_end_column], errors="coerce")
    result = ends.notna() & ends.le(pd.Timestamp(through))
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
            if template.feature_name not in self._numeric_cache:
                self._numeric_cache[template.feature_name] = _finite_numeric(
                    self.frame, template.feature_name
                )
            values = (
                self._numeric_cache[template.feature_name]
                .loc[self._dates.le(end)]
                .dropna()
            )
            if template.fixed_threshold is not None:
                threshold = (
                    float(template.fixed_threshold) if not values.empty else None
                )
            else:
                threshold_key = (period_key, template.feature_name)
                if threshold_key not in self._threshold_cache:
                    feature_templates = tuple(
                        item
                        for item in self.atomic_templates
                        if item.feature_name == template.feature_name
                        and item.quantile is not None
                        and item.fixed_threshold is None
                    )
                    quantiles = tuple(
                        sorted({item.quantile for item in feature_templates})
                    )
                    if values.empty or not quantiles:
                        thresholds: dict[float, float] = {}
                    else:
                        calculated = np.atleast_1d(
                            np.quantile(
                                values.to_numpy(), quantiles, method="nearest"
                            )
                        )
                        thresholds = {
                            quantile: float(value)
                            for quantile, value in zip(quantiles, calculated)
                            if np.isfinite(value)
                        }
                    self._threshold_cache[threshold_key] = thresholds
                threshold = self._threshold_cache[threshold_key].get(
                    template.quantile
                )
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
                        template.minimum_clause_count,
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


def _apply_max_stat_permutation_gate(
    frame: pd.DataFrame,
    spec: ObjectiveSpec,
    candidates: Iterable[CandidateEvaluation],
    cfg: Config,
) -> None:
    """Apply a deterministic family-wise max-statistic gate on validation data.

    The joint objective/protected label state is permuted within calendar
    years.  This preserves all four possible states, including observations
    that belong to both classes.  Every candidate is compared with the maximum
    null score across the complete evaluable candidate family, so adding many
    features or interactions cannot silently increase the false-positive
    budget.
    """

    candidate_list = list(candidates)
    evaluable = [
        candidate
        for candidate in candidate_list
        if candidate.templates
        and len(candidate.final_conditions) == len(candidate.templates)
    ]
    candidate_count = len(evaluable)
    for candidate in candidate_list:
        candidate.multiple_testing_candidate_count = candidate_count or None
        candidate.permutation_trial_count = (
            cfg.permutation_trial_count if candidate_count else None
        )
        candidate.max_stat_permutation_p_value = None
        candidate.passes_multiple_testing = None
    if not evaluable:
        return

    validation = _period_frame(
        _universe(frame, spec),
        spec,
        cfg.discovery_end_date + timedelta(days=1),
        cfg.validation_end_date,
    )
    available = _available(validation, spec, cfg.validation_end_date)
    validation = validation.loc[available].copy()
    objective = _truthy(validation[spec.objective]).to_numpy(dtype=bool)
    protected = _truthy(validation[spec.protected_outcome]).to_numpy(dtype=bool)
    if not len(validation) or not objective.any() or not protected.any():
        for candidate in evaluable:
            candidate.passes_multiple_testing = False
        return
    masks = np.column_stack(
        [
            _combined_mask(validation, candidate.final_conditions)
            .to_numpy(dtype=bool)
            for candidate in evaluable
        ]
    )
    mask_matrix = masks.astype(np.float32, copy=False)
    objective_count = float(objective.sum())
    protected_count = float(protected.sum())
    observed = (
        mask_matrix.T @ objective.astype(np.float32) / objective_count
        - mask_matrix.T @ protected.astype(np.float32) / protected_count
    )

    # Bit 0 represents objective membership and bit 1 protected membership.
    # Permuting this joint code keeps overlap prevalence and its association
    # structure intact instead of forcing a logically false disjoint split.
    label_codes = objective.astype(np.uint8) + (
        protected.astype(np.uint8) * np.uint8(2)
    )
    years = pd.to_datetime(validation[spec.date_column], errors="raise").dt.year
    strata = [
        np.flatnonzero(years.to_numpy() == year)
        for year in sorted(years.unique())
    ]
    seed_text = (
        f"{spec.decision_family}|{spec.objective}|{spec.protected_outcome}|"
        f"{spec.landmark_day or 0}"
    ).encode("utf-8")
    seed_offset = sum(
        (position + 1) * value for position, value in enumerate(seed_text)
    ) % (2**32)
    rng = np.random.default_rng(
        (cfg.permutation_random_seed + seed_offset) % (2**32)
    )
    null_maxima = np.empty(cfg.permutation_trial_count, dtype=float)
    batch_size = min(25, cfg.permutation_trial_count)
    for start in range(0, cfg.permutation_trial_count, batch_size):
        stop = min(start + batch_size, cfg.permutation_trial_count)
        trial_count = stop - start
        objective_permutations = np.zeros(
            (len(validation), trial_count), dtype=np.float32
        )
        protected_permutations = np.zeros_like(objective_permutations)
        for trial in range(trial_count):
            permuted = label_codes.copy()
            for positions in strata:
                permuted[positions] = rng.permutation(permuted[positions])
            objective_permutations[:, trial] = (permuted & 1) != 0
            protected_permutations[:, trial] = (permuted & 2) != 0
        null_scores = (
            mask_matrix.T @ objective_permutations / objective_count
            - mask_matrix.T @ protected_permutations / protected_count
        )
        null_maxima[start:stop] = np.max(null_scores, axis=0)

    for position, candidate in enumerate(evaluable):
        p_value = float(
            (1 + np.count_nonzero(null_maxima >= observed[position] - 1e-12))
            / (cfg.permutation_trial_count + 1)
        )
        candidate.max_stat_permutation_p_value = p_value
        candidate.passes_multiple_testing = bool(
            p_value <= cfg.max_stat_permutation_p_value
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
    atomic_beam = eligible_singles[: cfg.rule_search_beam_width]
    tested: list[CandidateEvaluation] = list(singles)
    viable: list[CandidateEvaluation] = list(eligible_singles)
    frontier = atomic_beam
    seen = {tuple(item.rule_id for item in candidate.templates) for candidate in singles}
    for size in range(2, cfg.max_conditions_per_objective + 1):
        expanded: list[CandidateEvaluation] = []
        for base in frontier:
            existing_ids = {item.rule_id for item in base.templates}
            existing_atomic = {
                (item.feature_name, item.operator)
                for item in base.templates
                if isinstance(item, ConditionTemplate)
            }
            for added_evaluation in atomic_beam:
                added = added_evaluation.templates[0]
                if added.rule_id in existing_ids or (
                    isinstance(added, ConditionTemplate)
                    and (added.feature_name, added.operator) in existing_atomic
                ):
                    continue
                combined = (*base.templates, added)
                key = tuple(sorted(item.rule_id for item in combined))
                if key in seen:
                    continue
                seen.add(key)
                candidate = evaluator.evaluate(combined)
                tested.append(candidate)
                base_score = base.selection_score
                candidate_score = candidate.selection_score
                if (
                    candidate.passes_development_gates
                    and candidate.passes_stability_gates
                    and base_score is not None
                    and candidate_score is not None
                    and candidate_score
                    >= base_score + cfg.min_selection_score_improvement
                ):
                    expanded.append(candidate)
        expanded.sort(key=_rank)
        viable.extend(expanded)
        frontier = expanded[: cfg.rule_search_beam_width]
        if not frontier:
            break
    _apply_max_stat_permutation_gate(frame, spec, tested, cfg)
    eligible = [
        item for item in viable if item.passes_multiple_testing is True
    ]
    selected = min(eligible, key=_rank) if eligible else empty
    prefixes: list[CandidateEvaluation] = [empty]
    for length in range(1, len(selected.templates) + 1):
        prefixes.append(evaluator.evaluate(selected.templates[:length]))
    candidates = tuple(sorted(tested, key=_rank))
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
        evaluable_positions = [
            position
            for position, metrics in enumerate(metrics_by_objective)
            if int(metrics["sample_count"]) > 0
        ]
        if not evaluable_positions:
            if total_conditions:
                continue
            rank = (0.0, 0.0, 0.0, 0, tuple(lengths))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best = dict(zip(objectives, lengths))
            continue
        safe = total_conditions == 0
        if total_conditions:
            pooled_safe = all(
                _safety_gates(metrics_by_objective[position], cfg)
                for position in evaluable_positions
            )
            stable = True
            for position in evaluable_positions:
                objective = objectives[position]
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
            float(metrics_by_objective[position]["objective_capture_rate"] or 0.0)
            for position in evaluable_positions
        ]
        retentions = [
            float(metrics_by_objective[position]["protected_retention_rate"] or 0.0)
            for position in evaluable_positions
        ]
        match_rates = [
            float(metrics_by_objective[position]["match_rate"] or 0.0)
            for position in evaluable_positions
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
                    if int(metrics["sample_count"]) > 0 and not _safety_gates(
                        metrics, cfg
                    ):
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
    anchor_key_set = set(anchor_keys.tolist())
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
        if local.empty or not anchor_key_set:
            continue
        local_keys = _policy_keys(local)
        # Use the same exact Python tuple-key semantics for active state,
        # first-cut storage and later prior-cut lookup. Pandas ``isin`` over a
        # large object Series of tuples can otherwise disagree with the dict
        # lookup used when decisions are persisted.
        active = pd.Series(
            [
                key in anchor_key_set and key not in cut_day_by_key
                for key in local_keys.tolist()
            ],
            index=local.index,
            dtype=bool,
        )
        active_masks[day].loc[local.index] = active.to_numpy()
        matched = _combined_mask(
            local, conditions_by_day.get(day, ())
        ) & active
        cut_masks[day].loc[local.index] = matched.to_numpy()
        if matched.any():
            for key in local_keys.loc[matched].tolist():
                cut_day_by_key.setdefault(key, day)
    anchor_matched = pd.Series(
        [key in cut_day_by_key for key in anchor_keys.tolist()],
        index=anchor.index,
        dtype=bool,
    )
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
    exclusion_results: dict[
        str, tuple[pd.Series, list[str | None], list[str | None]]
    ] = {}
    for objective, selection in selected.entry.items():
        length = selected.entry_prefix_lengths[objective]
        exclusion_results[objective] = _row_matches(
            result, selection.selected.final_conditions[:length]
        )
    weak_mask, weak_ids, _ = exclusion_results["weak_5d"]
    loss_mask, loss_ids, _ = exclusion_results["loss_first_5d"]
    final_mask = pd.Series(False, index=result.index, dtype=bool)
    for mask, _, _ in exclusion_results.values():
        final_mask |= mask
    result["include_weak_filter"] = ~weak_mask
    result["include_loss_first_filter"] = ~loss_mask
    result["include_final"] = ~final_mask
    result["weak_matched_rule_ids"] = weak_ids
    result["loss_first_matched_rule_ids"] = loss_ids
    result["matched_rule_ids"] = [
        ",".join(
            filter(
                None,
                (exclusion_results[objective][1][position] for objective in selected.entry),
            )
        )
        or None
        for position in range(len(result))
    ]
    result["exclusion_reason"] = [
        "; ".join(
            f"{objective}: {exclusion_results[objective][2][position]}"
            for objective in selected.entry
            if exclusion_results[objective][2][position]
        )
        or None
        for position in range(len(result))
    ]
    result["filter_decision"] = np.where(final_mask, "exclude", "include")

    confirmation_results: dict[
        str, tuple[pd.Series, list[str | None], list[str | None]]
    ] = {}
    confirmation_mask = pd.Series(False, index=result.index, dtype=bool)
    for objective, selection in selected.confirmation.items():
        length = selected.confirmation_prefix_lengths[objective]
        evaluated = _row_matches(
            result, selection.selected.final_conditions[:length]
        )
        confirmation_results[objective] = evaluated
        confirmation_mask |= evaluated[0]
    result["strong_confirmation"] = confirmation_mask
    result["confirmation_matched_rule_ids"] = [
        ",".join(
            filter(
                None,
                (
                    confirmation_results[objective][1][position]
                    for objective in selected.confirmation
                ),
            )
        )
        or None
        for position in range(len(result))
    ]
    result["confirmation_reason"] = [
        "; ".join(
            f"{objective}: {confirmation_results[objective][2][position]}"
            for objective in selected.confirmation
            if confirmation_results[objective][2][position]
        )
        or None
        for position in range(len(result))
    ]
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
    early_rows = pd.to_numeric(
        result["landmark_day"], errors="coerce"
    ).isin(EARLY_CUT_LANDMARK_DAYS)
    eligible = _truthy(result["eligible_at_landmark"]) & early_rows
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
    policy_active = pd.Series(False, index=result.index, dtype=bool)
    for day in EARLY_CUT_LANDMARK_DAYS:
        policy_active |= active_masks[day]
    if not _truthy(result["active_at_landmark"]).equals(policy_active):
        raise AssertionError("sequential early-cut active state is inconsistent")
    active_with_prior_cut = policy_active & result[
        "prior_policy_cut_day"
    ].notna()
    if active_with_prior_cut.any():
        log.warning(
            "Corrected %d active D1-D3 policy rows carrying a contradictory prior cut",
            int(active_with_prior_cut.sum()),
        )
        result.loc[active_with_prior_cut, "prior_policy_cut_day"] = pd.NA
    result.loc[policy_active, "eligible_at_landmark"] = True
    active_holds = policy_active & _truthy(result["include_final"])
    active_cuts = policy_active & ~_truthy(result["include_final"])
    result.loc[active_holds, "cut_decision"] = "hold"
    result.loc[active_holds, "cut_reason"] = pd.NA
    result.loc[active_cuts, "cut_decision"] = "cut"
    # A later landmark can be intrinsically evaluable even when its D+1 row
    # was not part of the anchor cohort (for example around sparse identity
    # history). Such a row is not eligible for the sequential D+1 -> D+3
    # policy. Persist that policy eligibility explicitly instead of treating
    # the later row as active without an anchor.
    non_anchored = (
        eligible
        & ~policy_active
        & result["prior_policy_cut_day"].isna()
    )
    if non_anchored.any():
        log.info(
            "Excluded %d intrinsically eligible D2/D3 rows without an active D1 policy anchor",
            int(non_anchored.sum()),
        )
        result.loc[non_anchored, "eligible_at_landmark"] = False
        result.loc[non_anchored, "cut_decision"] = "not_eligible"
        result.loc[non_anchored, "include_stagnation_filter"] = False
        result.loc[non_anchored, "include_loss_filter"] = False
        result.loc[non_anchored, "include_final"] = False
        result.loc[non_anchored, "stagnation_matched_rule_ids"] = pd.NA
        result.loc[non_anchored, "loss_matched_rule_ids"] = pd.NA
        result.loc[non_anchored, "matched_rule_ids"] = pd.NA
        result.loc[non_anchored, "cut_reason"] = pd.NA
    expected_active = (
        _truthy(result["eligible_at_landmark"])
        & early_rows
        & result["prior_policy_cut_day"].isna()
    )
    if not np.array_equal(
        policy_active.to_numpy(dtype=bool),
        expected_active.to_numpy(dtype=bool),
    ):
        raise AssertionError(
            "sequential early-cut policy eligibility is inconsistent"
        )
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


def apply_management_decisions(
    landmarks: pd.DataFrame, selected: SelectedSummary
) -> pd.DataFrame:
    """Apply independent D5/D20/D30 continue-versus-exit research rules."""

    result = landmarks.copy()
    result["management_include_final"] = result[
        "management_include_final"
    ].astype("boolean")
    result["management_decision"] = result["management_decision"].astype("string")
    management_rows = pd.to_numeric(
        result["landmark_day"], errors="coerce"
    ).isin(MANAGEMENT_LANDMARK_DAYS)
    result.loc[management_rows, "management_include_final"] = False
    result.loc[management_rows, "management_matched_rule_ids"] = pd.NA
    result.loc[management_rows, "management_reason"] = pd.NA
    for day, selections in selected.management.items():
        rows = (
            pd.to_numeric(result["landmark_day"], errors="coerce").eq(day)
            & _truthy(result["eligible_at_landmark"])
        )
        local = result.loc[rows]
        if local.empty:
            continue
        evaluated: dict[
            str, tuple[pd.Series, list[str | None], list[str | None]]
        ] = {}
        exit_mask = pd.Series(False, index=local.index, dtype=bool)
        for objective, selection in selections.items():
            length = selected.management_prefix_lengths[day][objective]
            match = _row_matches(
                local, selection.selected.final_conditions[:length]
            )
            evaluated[objective] = match
            exit_mask |= match[0]
        result.loc[rows, "management_include_final"] = (~exit_mask).to_numpy()
        result.loc[rows, "management_matched_rule_ids"] = [
            ",".join(
                filter(
                    None,
                    (evaluated[objective][1][position] for objective in selections),
                )
            )
            or None
            for position in range(len(local))
        ]
        result.loc[rows, "management_reason"] = [
            "; ".join(
                f"{objective}: {evaluated[objective][2][position]}"
                for objective in selections
                if evaluated[objective][2][position]
            )
            or None
            for position in range(len(local))
        ]
        result.loc[rows, "management_decision"] = np.where(
            exit_mask,
            "cut_stagnation" if day == 5 else "take_profit",
            "hold",
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
        return (
            item.feature_group,
            item.feature_name,
            item.operator,
            item.quantile,
            item.fixed_threshold,
        )
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
            if clause.fixed_threshold is not None:
                threshold_text = f"fixed {clause.fixed_threshold:.10g} threshold"
            elif clause.quantile is not None:
                threshold_text = f"causal Q{clause.quantile:.4g} threshold"
            else:
                threshold_text = "unavailable threshold"
            if clause.minimum_threshold is not None:
                threshold_text = (
                    f"max({threshold_text}, {clause.minimum_threshold:.10g})"
                )
            clause_text.append(f"{clause.feature_name} {symbol} {threshold_text}")
        if (
            isinstance(item, PatternTemplate)
            and item.minimum_clause_count is not None
        ):
            text = (
                f"at least {item.minimum_clause_count} of {len(item.clauses)}: "
                + "; ".join(clause_text)
            )
        else:
            text = " AND ".join(clause_text)
        if isinstance(item, PatternTemplate):
            text = f"{item.pattern_name} ({text})"
        parts.append(text)
    return " OR ".join(parts) or "no exclusion condition"


def _pattern_fields(
    items: tuple[CandidateTemplate | ExecutableCondition, ...],
) -> tuple[
    str | None,
    str | None,
    int | None,
    int | None,
    int | None,
    float | None,
]:
    if len(items) != 1:
        return None, None, None, None, None, None
    item = items[0]
    if isinstance(item, (PatternTemplate, CompositeCondition)):
        total = len(item.clauses)
        required = item.minimum_clause_count or total
        mode = "k_of_n" if item.minimum_clause_count is not None else "all"
        return item.pattern_name, mode, total, required, None, None
    feature_name = item.feature_name
    if not feature_name.startswith("pattern_") or "_score" not in feature_name:
        return None, None, None, None, None, None
    name = feature_name.removeprefix("pattern_")
    name = re.sub(r"_(?:setup|trigger)_score$", "", name)
    window_match = re.search(r"_score_(\d+)d$", name)
    window = int(window_match.group(1)) if window_match else None
    name = re.sub(r"_score_\d+d$", "", name)
    threshold = (
        item.threshold if isinstance(item, Condition) else item.fixed_threshold
    )
    return name, "score_threshold", None, None, window, threshold


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
    passes_multiple_testing: bool | None = None,
    multiple_testing_candidate_count: int | None = None,
    permutation_trial_count: int | None = None,
    max_stat_permutation_p_value: float | None = None,
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
    pattern_items: tuple[CandidateTemplate | ExecutableCondition, ...] = (
        conditions if conditions else templates
    )
    (
        pattern_name,
        pattern_match_mode,
        pattern_total_clause_count,
        pattern_required_clause_count,
        pattern_score_window_sessions,
        pattern_score_threshold_pct,
    ) = _pattern_fields(pattern_items)
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
        "pattern_name": pattern_name,
        "pattern_match_mode": pattern_match_mode,
        "pattern_total_clause_count": pattern_total_clause_count,
        "pattern_required_clause_count": pattern_required_clause_count,
        "pattern_score_window_sessions": pattern_score_window_sessions,
        "pattern_score_threshold_pct": pattern_score_threshold_pct,
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
        "passes_multiple_testing": passes_multiple_testing,
        "multiple_testing_candidate_count": multiple_testing_candidate_count,
        "permutation_trial_count": permutation_trial_count,
        "max_stat_permutation_p_value": max_stat_permutation_p_value,
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
    if first.decision_family == "entry_filter":
        base_id = "ENTRY_FILTER_FINAL"
    elif first.decision_family == "entry_confirmation":
        base_id = "ENTRY_CONFIRMATION_FINAL"
    else:
        base_id = f"EARLY_CUT_D{first.landmark_day}_POLICY_COMPONENT"
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
            if not first.decision_family.startswith("entry_")
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
    for objective, selection in selected.confirmation.items():
        portfolios.append(
            (
                signals,
                selection,
                selected.confirmation_prefix_lengths[objective],
            )
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
    for day, selections in selected.management.items():
        for objective, selection in selections.items():
            portfolios.append(
                (
                    early_cuts,
                    selection,
                    selected.management_prefix_lengths[day][objective],
                )
            )
    for source, selection, prefix_length in portfolios:
        spec = selection.spec
        chosen = selection.prefixes[prefix_length]
        conditions = chosen.final_conditions
        base_id = f"{spec.decision_family.upper()}_{spec.objective.upper()}" + (
            f"_D{spec.landmark_day}" if spec.landmark_day else ""
        )
        # Persist every evaluated atomic, composite and OR-pair candidate.
        # Diagnostic and holdout outcomes remain outside candidate selection.
        for candidate in selection.candidates:
            selected_candidate = candidate.templates == chosen.templates or (
                len(candidate.templates) == 1
                and candidate.templates[0] in chosen.templates
            )
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
                    is_selected=selected_candidate,
                    is_final_filter=False,
                    passes_development_gates=candidate.passes_development_gates,
                    passes_stability_gates=candidate.passes_stability_gates,
                    passes_multiple_testing=candidate.passes_multiple_testing,
                    multiple_testing_candidate_count=(
                        candidate.multiple_testing_candidate_count
                    ),
                    permutation_trial_count=candidate.permutation_trial_count,
                    max_stat_permutation_p_value=(
                        candidate.max_stat_permutation_p_value
                    ),
                    stability=candidate.stability,
                    selection_order=(
                        chosen.templates.index(candidate.templates[0]) + 1
                        if len(candidate.templates) == 1
                        and candidate.templates[0] in chosen.templates
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
                passes_multiple_testing=chosen.passes_multiple_testing,
                multiple_testing_candidate_count=(
                    chosen.multiple_testing_candidate_count
                ),
                permutation_trial_count=chosen.permutation_trial_count,
                max_stat_permutation_p_value=(
                    chosen.max_stat_permutation_p_value
                ),
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
    _append_final_union_rows(
        rows,
        signals,
        selected.confirmation,
        selected.confirmation_prefix_lengths,
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
    for day, day_selections in selected.management.items():
        _append_final_union_rows(
            rows,
            early_cuts,
            day_selections,
            selected.management_prefix_lengths[day],
            trading_dates,
            cfg,
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
        for spec in ENTRY_EXCLUSION_SPECS
    }
    entry_frame = _period_frame(
        signals, ENTRY_EXCLUSION_SPECS[0], None, cfg.validation_end_date
    )
    entry_lengths = _choose_prefixes(
        entry_frame,
        entry,
        cfg,
        require_cross_objective_lift=False,
    )
    entry_lengths = _gate_refit_prefixes(signals, entry, entry_lengths, cfg)
    confirmation_templates = build_candidate_templates(
        "entry_confirmation", quantile_count=cfg.quantile_count
    )
    confirmation = {
        spec.objective: select_objective(
            signals, spec, confirmation_templates, cfg
        )
        for spec in ENTRY_CONFIRMATION_SPECS
    }
    confirmation_frame = _period_frame(
        signals,
        ENTRY_CONFIRMATION_SPECS[0],
        None,
        cfg.validation_end_date,
    )
    confirmation_lengths = _choose_prefixes(
        confirmation_frame,
        confirmation,
        cfg,
        require_cross_objective_lift=False,
    )
    confirmation_lengths = _gate_refit_prefixes(
        signals, confirmation, confirmation_lengths, cfg
    )
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
    management: dict[int, dict[str, ObjectiveSelection]] = {}
    management_lengths: dict[int, dict[str, int]] = {}
    for day in MANAGEMENT_LANDMARK_DAYS:
        templates = build_candidate_templates(
            "position_management", day, quantile_count=cfg.quantile_count
        )
        selections = {
            spec.objective: select_objective(early_cuts, spec, templates, cfg)
            for spec in management_specs(day)
        }
        management[day] = selections
        day_frame = _period_frame(
            _universe(early_cuts, management_specs(day)[0]),
            management_specs(day)[0],
            None,
            cfg.validation_end_date,
        )
        management_lengths[day] = _choose_prefixes(
            day_frame,
            selections,
            cfg,
            require_cross_objective_lift=False,
        )
        management_lengths[day] = _gate_refit_prefixes(
            early_cuts, selections, management_lengths[day], cfg
        )
    summary = SelectedSummary(
        entry,
        confirmation,
        early,
        management,
        entry_lengths,
        confirmation_lengths,
        early_lengths,
        management_lengths,
    )
    decided_signals = apply_entry_decisions(signals, summary)
    decided_early = apply_early_decisions(early_cuts, summary)
    decided_early = apply_management_decisions(decided_early, summary)
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
