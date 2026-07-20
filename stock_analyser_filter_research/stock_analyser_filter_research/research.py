from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    FEATURE_GROUPS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)


LOWER_QUANTILES = (0.05, 0.10, 0.20, 0.30)
UPPER_QUANTILES = (0.70, 0.80, 0.90, 0.95)


@dataclass(frozen=True)
class Condition:
    rule_id: str
    feature_group: str
    feature_name: str
    operator: str
    threshold: float
    quantile: float

    @property
    def text(self) -> str:
        symbol = "<=" if self.operator == "le" else ">="
        return f"{self.feature_name} {symbol} {self.threshold:.10g}"

    def matches(self, frame: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(frame[self.feature_name], errors="coerce")
        if self.operator == "le":
            return values.notna() & values.le(self.threshold)
        if self.operator == "ge":
            return values.notna() & values.ge(self.threshold)
        raise ValueError(f"unsupported condition operator {self.operator!r}")


@dataclass(frozen=True)
class SelectedRules:
    stage_a: tuple[Condition, ...]
    stage_ab: tuple[Condition, ...]
    stage_abc: tuple[Condition, ...]

    @property
    def final(self) -> tuple[Condition, ...]:
        return self.stage_abc


@dataclass
class ResearchResult:
    signals: pd.DataFrame
    rules: pd.DataFrame
    selected: SelectedRules


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.replace([np.inf, -np.inf], np.nan)


def _labeled(frame: pd.DataFrame) -> pd.DataFrame:
    required = ("weak_5d", "strong_5d", "deep_loss_5d", "bad_5d")
    available = frame.loc[:, list(required)].notna().all(axis=1)
    return frame.loc[available]


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _metrics(frame: pd.DataFrame, excluded: pd.Series) -> dict[str, object]:
    all_excluded = (
        excluded.reindex(frame.index, fill_value=False)
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    sample = _labeled(frame)
    excluded = all_excluded.reindex(sample.index, fill_value=False).astype(bool)
    weak = sample["weak_5d"].astype(bool)
    deep = sample["deep_loss_5d"].astype(bool)
    bad = sample["bad_5d"].astype(bool)
    strong = sample["strong_5d"].astype(bool)
    late10 = (
        sample["late_strong_10d"].astype("boolean").fillna(False).astype(bool)
    )
    late20 = (
        sample["late_strong_20d"].astype("boolean").fillna(False).astype(bool)
    )
    retained = ~excluded

    signal_count = int(len(frame))
    sample_count = int(len(sample))
    unlabeled_count = signal_count - sample_count
    matched_signal_count = int(all_excluded.sum())
    weak_count = int(weak.sum())
    deep_count = int(deep.sum())
    bad_count = int(bad.sum())
    strong_count = int(strong.sum())
    late10_count = int(late10.sum())
    late20_count = int(late20.sum())
    excluded_count = int(excluded.sum())
    excluded_weak = int((excluded & weak).sum())
    excluded_deep = int((excluded & deep).sum())
    excluded_bad = int((excluded & bad).sum())
    excluded_strong = int((excluded & strong).sum())
    excluded_late10 = int((excluded & late10).sum())
    excluded_late20 = int((excluded & late20).sum())
    matched_unlabeled_count = matched_signal_count - excluded_count
    retained_count = sample_count - excluded_count

    population_bad_rate = _safe_ratio(bad_count, sample_count)
    excluded_bad_rate = _safe_ratio(excluded_bad, excluded_count)
    bad_lift = (
        None
        if population_bad_rate in (None, 0) or excluded_bad_rate is None
        else excluded_bad_rate / population_bad_rate
    )
    strong_rejection = _safe_ratio(excluded_strong, strong_count)

    return {
        "signal_count": signal_count,
        "sample_count": sample_count,
        "unlabeled_count": unlabeled_count,
        "matched_signal_count": matched_signal_count,
        "matched_unlabeled_count": matched_unlabeled_count,
        "weak_count": weak_count,
        "deep_loss_count": deep_count,
        "bad_count": bad_count,
        "strong_count": strong_count,
        "late_strong_10d_count": late10_count,
        "late_strong_20d_count": late20_count,
        "excluded_count": excluded_count,
        "excluded_weak_count": excluded_weak,
        "excluded_deep_loss_count": excluded_deep,
        "excluded_bad_count": excluded_bad,
        "excluded_strong_count": excluded_strong,
        "excluded_late_strong_10d_count": excluded_late10,
        "excluded_late_strong_20d_count": excluded_late20,
        "label_coverage_rate": _safe_ratio(sample_count, signal_count),
        "matched_label_coverage_rate": _safe_ratio(
            excluded_count, matched_signal_count
        ),
        "exclusion_rate": _safe_ratio(excluded_count, sample_count),
        "weak_capture_rate": _safe_ratio(excluded_weak, weak_count),
        "deep_loss_capture_rate": _safe_ratio(excluded_deep, deep_count),
        "bad_capture_rate": _safe_ratio(excluded_bad, bad_count),
        "strong_rejection_rate": strong_rejection,
        "strong_retention_rate": (
            None if strong_rejection is None else 1.0 - strong_rejection
        ),
        "excluded_bad_rate": excluded_bad_rate,
        "bad_lift": bad_lift,
        "retained_bad_rate": _safe_ratio(int((retained & bad).sum()), retained_count),
        "retained_strong_rate": _safe_ratio(
            int((retained & strong).sum()), retained_count
        ),
        "late_strong_10d_rejection_rate": _safe_ratio(
            excluded_late10, late10_count
        ),
        "late_strong_20d_rejection_rate": _safe_ratio(
            excluded_late20, late20_count
        ),
    }


def _combined_mask(
    frame: pd.DataFrame, conditions: tuple[Condition, ...]
) -> pd.Series:
    result = pd.Series(False, index=frame.index, dtype=bool)
    for condition in conditions:
        result |= condition.matches(frame)
    return result


def _condition_sort_key(condition: Condition) -> tuple[object, ...]:
    return (
        condition.feature_group,
        condition.feature_name,
        condition.operator,
        condition.threshold,
        condition.rule_id,
    )


def build_candidates(signals: pd.DataFrame) -> tuple[Condition, ...]:
    # Thresholds depend only on point-in-time features. Requiring an outcome
    # here would let future horizon availability (for example a later segment
    # break or delisting) influence the feature threshold itself.
    discovery = signals.loc[signals["analysis_split"].eq("discovery")]
    candidates: list[Condition] = []
    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            values = _finite_numeric(discovery, feature).dropna()
            if values.empty:
                continue
            observed: set[tuple[str, float]] = set()
            for quantile, operator in (
                *((value, "le") for value in LOWER_QUANTILES),
                *((value, "ge") for value in UPPER_QUANTILES),
            ):
                threshold = float(
                    np.quantile(values.to_numpy(), quantile, method="nearest")
                )
                if not np.isfinite(threshold):
                    continue
                key = (operator, threshold)
                if key in observed:
                    continue
                observed.add(key)
                side = "LE" if operator == "le" else "GE"
                quantile_tag = int(round(quantile * 100))
                candidates.append(
                    Condition(
                        rule_id=f"CAND_{group}_{feature}_{side}_Q{quantile_tag:02d}",
                        feature_group=group,
                        feature_name=feature,
                        operator=operator,
                        threshold=threshold,
                        quantile=quantile,
                    )
                )
    return tuple(sorted(candidates, key=_condition_sort_key))


def _admissible(metrics: dict[str, object], cfg: Config) -> bool:
    sample_count = int(metrics["sample_count"])
    excluded_count = int(metrics["excluded_count"])
    minimum_count = max(
        50, int(ceil(sample_count * cfg.min_candidate_match_pct))
    )
    exclusion_rate = metrics["exclusion_rate"]
    strong_retention = metrics["strong_retention_rate"]
    bad_lift = metrics["bad_lift"]
    bad_capture = metrics["bad_capture_rate"]
    matched_coverage = metrics["matched_label_coverage_rate"]
    return bool(
        sample_count > 0
        and excluded_count >= minimum_count
        and exclusion_rate is not None
        and float(exclusion_rate) <= cfg.max_candidate_match_pct
        and strong_retention is not None
        and float(strong_retention) >= cfg.min_strong_retention_pct
        and matched_coverage is not None
        and float(matched_coverage) >= cfg.min_matched_label_coverage_pct
        and bad_lift is not None
        and float(bad_lift) >= cfg.min_bad_lift
        and bad_capture is not None
        and float(bad_capture) > 0
    )


def select_rules(
    signals: pd.DataFrame,
    candidates: tuple[Condition, ...],
    cfg: Config,
) -> SelectedRules:
    discovery = signals.loc[signals["analysis_split"].eq("discovery")]
    validation = signals.loc[signals["analysis_split"].eq("validation")]
    selected: tuple[Condition, ...] = ()
    stages: dict[str, tuple[Condition, ...]] = {}

    for group in ("A", "B", "C"):
        if len(selected) < cfg.max_rule_conditions:
            base_discovery = _metrics(
                discovery, _combined_mask(discovery, selected)
            )
            base_validation = _metrics(
                validation, _combined_mask(validation, selected)
            )
            base_discovery_capture = float(
                base_discovery["bad_capture_rate"] or 0.0
            )
            base_validation_capture = float(
                base_validation["bad_capture_rate"] or 0.0
            )
            eligible: list[
                tuple[tuple[object, ...], Condition, tuple[Condition, ...]]
            ] = []
            for candidate in candidates:
                if candidate.feature_group != group:
                    continue
                combined = (*selected, candidate)
                discovery_metrics = _metrics(
                    discovery, _combined_mask(discovery, combined)
                )
                validation_metrics = _metrics(
                    validation, _combined_mask(validation, combined)
                )
                discovery_capture = float(
                    discovery_metrics["bad_capture_rate"] or 0.0
                )
                validation_capture = float(
                    validation_metrics["bad_capture_rate"] or 0.0
                )
                if not (
                    _admissible(discovery_metrics, cfg)
                    and _admissible(validation_metrics, cfg)
                    and discovery_capture
                    >= base_discovery_capture
                    + cfg.min_selection_score_improvement
                    and validation_capture
                    >= base_validation_capture
                    + cfg.min_selection_score_improvement
                ):
                    continue
                rank = (
                    -validation_capture,
                    -float(validation_metrics["strong_retention_rate"]),
                    -float(validation_metrics["bad_lift"]),
                    float(validation_metrics["exclusion_rate"]),
                    candidate.text,
                )
                eligible.append((rank, candidate, combined))
            if eligible:
                eligible.sort(key=lambda item: item[0])
                selected = eligible[0][2]
        stages[group] = selected

    return SelectedRules(
        stage_a=stages["A"],
        stage_ab=stages["B"],
        stage_abc=stages["C"],
    )


def apply_selected_rules(
    signals: pd.DataFrame, selected: SelectedRules
) -> pd.DataFrame:
    result = signals.copy()
    masks = {
        "include_stage_a": _combined_mask(result, selected.stage_a),
        "include_stage_ab": _combined_mask(result, selected.stage_ab),
        "include_stage_abc": _combined_mask(result, selected.stage_abc),
    }
    for column, excluded in masks.items():
        result[column] = ~excluded

    final_conditions = selected.final
    final_masks = [condition.matches(result) for condition in final_conditions]
    matched_ids: list[str | None] = []
    reasons: list[str | None] = []
    for row_position in range(len(result)):
        matched = [
            condition
            for condition, mask in zip(final_conditions, final_masks)
            if bool(mask.iloc[row_position])
        ]
        matched_ids.append(
            ",".join(condition.rule_id for condition in matched) or None
        )
        reasons.append(" OR ".join(condition.text for condition in matched) or None)
    result["matched_rule_ids"] = matched_ids
    result["exclusion_reason"] = reasons
    result["filter_decision"] = np.where(
        result["include_stage_abc"], "include", "exclude"
    )
    return result.loc[:, SIGNAL_COLUMNS]


def _scope_frames(signals: pd.DataFrame) -> list[tuple[str, int | None, pd.DataFrame]]:
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [
        (
            split,
            None,
            signals.loc[signals["analysis_split"].eq(split)],
        )
        for split in ("discovery", "validation", "test")
    ]
    scopes.append(("all_signals", None, signals))
    years = pd.to_datetime(signals["signal_date"]).dt.year
    for year in sorted(years.dropna().unique()):
        scopes.append(
            ("calendar_year", int(year), signals.loc[years.eq(year)])
        )
    return scopes


def _rule_row(
    *,
    rule_id: str,
    result_kind: str,
    stage: str,
    feature_group: str,
    feature_name: str | None,
    operator: str | None,
    threshold_value: float | None,
    bin_number: int | None,
    bin_lower_bound: float | None,
    bin_upper_bound: float | None,
    rule_text: str,
    selection_order: int | None,
    evaluation_scope: str,
    scope_year: int | None,
    frame: pd.DataFrame,
    excluded: pd.Series,
    is_selected: bool,
    is_final_filter: bool,
    passes_holdout: bool | None,
    component_count: int,
) -> dict[str, object]:
    if frame.empty:
        period_start = None
        period_end = None
    else:
        dates = pd.to_datetime(frame["signal_date"])
        period_start = dates.min().date()
        period_end = dates.max().date()
    row: dict[str, object] = {
        "rule_id": rule_id,
        "result_kind": result_kind,
        "stage": stage,
        "feature_group": feature_group,
        "feature_name": feature_name,
        "operator": operator,
        "threshold_value": threshold_value,
        "bin_number": bin_number,
        "bin_lower_bound": bin_lower_bound,
        "bin_upper_bound": bin_upper_bound,
        "rule_text": rule_text,
        "selection_order": selection_order,
        "evaluation_scope": evaluation_scope,
        "scope_year": scope_year,
        "period_start": period_start,
        "period_end": period_end,
        "is_selected": is_selected,
        "is_final_filter": is_final_filter,
        "passes_holdout": passes_holdout,
        "component_count": component_count,
        **_metrics(frame, excluded),
    }
    return row


def _quantile_boundaries(
    discovery: pd.DataFrame, feature: str, quantile_count: int
) -> tuple[float, ...]:
    values = _finite_numeric(discovery, feature).dropna().to_numpy()
    if not len(values):
        return ()
    quantiles = np.linspace(0.0, 1.0, quantile_count + 1)[1:-1]
    internal = [
        float(np.quantile(values, value, method="nearest"))
        for value in quantiles
    ]
    return tuple(sorted(set(value for value in internal if np.isfinite(value))))


def build_rule_results(
    signals: pd.DataFrame,
    candidates: tuple[Condition, ...],
    selected: SelectedRules,
    cfg: Config,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scopes = _scope_frames(signals)
    discovery = signals.loc[signals["analysis_split"].eq("discovery")]
    selected_order = {
        condition.rule_id: index + 1
        for index, condition in enumerate(selected.final)
    }

    for scope, year, frame in scopes:
        rows.append(
            _rule_row(
                rule_id="BASELINE_INCLUDE_ALL",
                result_kind="baseline",
                stage="baseline",
                feature_group="none",
                feature_name=None,
                operator=None,
                threshold_value=None,
                bin_number=None,
                bin_lower_bound=None,
                bin_upper_bound=None,
                rule_text="include all eight-criterion signals",
                selection_order=None,
                evaluation_scope=scope,
                scope_year=year,
                frame=frame,
                excluded=pd.Series(False, index=frame.index),
                is_selected=False,
                is_final_filter=False,
                passes_holdout=None,
                component_count=0,
            )
        )

    for group, features in FEATURE_GROUPS.items():
        for feature in features:
            boundaries = _quantile_boundaries(
                discovery, feature, cfg.quantile_count
            )
            if not boundaries:
                continue
            edges = (-np.inf, *boundaries, np.inf)
            for bin_index in range(len(edges) - 1):
                lower = edges[bin_index]
                upper = edges[bin_index + 1]
                lower_text = "-inf" if not np.isfinite(lower) else f"{lower:.10g}"
                upper_text = "inf" if not np.isfinite(upper) else f"{upper:.10g}"
                for scope, year, frame in scopes:
                    values = _finite_numeric(frame, feature)
                    in_bin = values.notna() & values.gt(lower) & values.le(upper)
                    rows.append(
                        _rule_row(
                            rule_id=f"BIN_{group}_{feature}_{bin_index + 1:02d}",
                            result_kind="quantile_bin",
                            stage=group,
                            feature_group=group,
                            feature_name=feature,
                            operator=None,
                            threshold_value=None,
                            bin_number=bin_index + 1,
                            bin_lower_bound=(
                                None if not np.isfinite(lower) else float(lower)
                            ),
                            bin_upper_bound=(
                                None if not np.isfinite(upper) else float(upper)
                            ),
                            rule_text=(
                                f"{lower_text} < {feature} <= {upper_text}"
                            ),
                            selection_order=None,
                            evaluation_scope=scope,
                            scope_year=year,
                            frame=frame,
                            excluded=in_bin,
                            is_selected=False,
                            is_final_filter=False,
                            passes_holdout=None,
                            component_count=1,
                        )
                    )

    for condition in candidates:
        for scope in ("discovery", "validation"):
            frame = signals.loc[signals["analysis_split"].eq(scope)]
            rows.append(
                _rule_row(
                    rule_id=condition.rule_id,
                    result_kind="candidate_rule",
                    stage=condition.feature_group,
                    feature_group=condition.feature_group,
                    feature_name=condition.feature_name,
                    operator=condition.operator,
                    threshold_value=condition.threshold,
                    bin_number=None,
                    bin_lower_bound=None,
                    bin_upper_bound=None,
                    rule_text=condition.text,
                    selection_order=selected_order.get(condition.rule_id),
                    evaluation_scope=scope,
                    scope_year=None,
                    frame=frame,
                    excluded=condition.matches(frame),
                    is_selected=condition.rule_id in selected_order,
                    is_final_filter=False,
                    passes_holdout=None,
                    component_count=1,
                )
            )

    selected_stages = (
        ("FILTER_A", "A", selected.stage_a),
        ("FILTER_A_B", "A_B", selected.stage_ab),
        ("FILTER_A_B_C", "A_B_C", selected.stage_abc),
    )
    final_test_metrics = _metrics(
        signals.loc[signals["analysis_split"].eq("test")],
        _combined_mask(
            signals.loc[signals["analysis_split"].eq("test")], selected.final
        ),
    )
    final_passes_holdout = bool(
        selected.final and _admissible(final_test_metrics, cfg)
    )
    for rule_id, stage, conditions in selected_stages:
        text = (
            " OR ".join(condition.text for condition in conditions)
            or "include all eight-criterion signals"
        )
        groups = {condition.feature_group for condition in conditions}
        feature_group = (
            "none"
            if not groups
            else next(iter(groups)) if len(groups) == 1 else "multiple"
        )
        for scope, year, frame in scopes:
            is_final = stage == "A_B_C"
            rows.append(
                _rule_row(
                    rule_id=rule_id,
                    result_kind="selected_filter",
                    stage=stage,
                    feature_group=feature_group,
                    feature_name=None,
                    operator=None,
                    threshold_value=None,
                    bin_number=None,
                    bin_lower_bound=None,
                    bin_upper_bound=None,
                    rule_text=text,
                    selection_order=None,
                    evaluation_scope=scope,
                    scope_year=year,
                    frame=frame,
                    excluded=_combined_mask(frame, conditions),
                    is_selected=True,
                    is_final_filter=is_final,
                    passes_holdout=(
                        final_passes_holdout
                        if is_final and scope == "test"
                        else None
                    ),
                    component_count=len(conditions),
                )
            )

    result = pd.DataFrame(rows)
    missing = sorted(set(RULE_COLUMNS) - set(result.columns))
    if missing:
        raise AssertionError("rule results missing columns: " + ", ".join(missing))
    return result.loc[:, RULE_COLUMNS].sort_values(
        ["result_kind", "rule_id", "evaluation_scope", "scope_year"],
        kind="stable",
        na_position="first",
    ).reset_index(drop=True)


def run_research(signals: pd.DataFrame, cfg: Config) -> ResearchResult:
    if signals.empty:
        raise RuntimeError("no false-to-true signals were found")
    signals = signals.sort_values(
        ["signal_date", "symbol", "exchange", "cik"], kind="stable"
    ).reset_index(drop=True)
    if signals.duplicated(["signal_date", "symbol", "exchange", "cik"]).any():
        raise RuntimeError("signal results contain duplicate primary keys")
    candidates = build_candidates(signals)
    selected = select_rules(signals, candidates, cfg)
    decided = apply_selected_rules(signals, selected)
    rule_results = build_rule_results(decided, candidates, selected, cfg)
    return ResearchResult(signals=decided, rules=rule_results, selected=selected)
