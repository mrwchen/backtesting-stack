from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research.contracts import (
    FEATURE_GROUPS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)
from stock_analyser_filter_research.research import (
    Condition,
    _admissible,
    _metrics,
    _scope_frames,
    apply_selected_rules,
    build_candidates,
    run_research,
    select_rules,
)


A_FEATURE = "adjusted_volume_vs_sma21_prior_ratio"
B_FEATURE = "prior_return_5d_pct"
C_FEATURE = "prior_volume_sma5_vs21_ratio"


def _blank_signals(size: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series([pd.NA] * size, dtype="object") for column in SIGNAL_COLUMNS}
    )
    frame["signal_date"] = pd.date_range("2020-01-01", periods=size, freq="D")
    frame["previous_session_date"] = frame["signal_date"] - pd.Timedelta(days=1)
    frame["symbol"] = [f"S{index:06d}" for index in range(size)]
    frame["exchange"] = "NYSE"
    frame["cik"] = np.arange(1, size + 1)
    frame["price_continuity_segment"] = 1
    frame["currency"] = "USD"
    frame["include_stage_a"] = True
    frame["include_stage_ab"] = True
    frame["include_stage_abc"] = True
    frame["filter_decision"] = "include"
    return frame


def _set_outcomes(frame: pd.DataFrame, bad: np.ndarray) -> None:
    frame["weak_5d"] = bad
    frame["deep_loss_5d"] = False
    frame["bad_5d"] = bad
    frame["strong_5d"] = ~bad
    frame["late_strong_10d"] = False
    frame["late_strong_20d"] = False


def _learning_signals(per_split: int = 200, bad_count: int = 60) -> pd.DataFrame:
    splits = ("discovery", "validation", "test")
    frame = _blank_signals(per_split * len(splits))
    frame["analysis_split"] = np.repeat(splits, per_split)
    values: list[np.ndarray] = []
    bad_parts: list[np.ndarray] = []
    for _ in splits:
        bad = np.arange(per_split) < bad_count
        bad_parts.append(bad)
        values.append(
            np.concatenate(
                [np.arange(bad_count), 100.0 + np.arange(per_split - bad_count)]
            )
        )
    frame[A_FEATURE] = np.concatenate(values)
    _set_outcomes(frame, np.concatenate(bad_parts))
    return frame


def _condition_signature(conditions: tuple[Condition, ...]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        (
            item.rule_id,
            item.feature_group,
            item.feature_name,
            item.operator,
            item.threshold,
            item.quantile,
        )
        for item in conditions
    )


def test_candidate_thresholds_are_discovery_only_and_observed_values() -> None:
    signals = _learning_signals()
    candidates = build_candidates(signals)
    a_candidates = tuple(item for item in candidates if item.feature_name == A_FEATURE)
    discovery_values = set(
        signals.loc[signals["analysis_split"].eq("discovery"), A_FEATURE]
    )

    mutated = signals.copy()
    outside_discovery = ~mutated["analysis_split"].eq("discovery")
    mutated.loc[outside_discovery, A_FEATURE] = np.linspace(
        -10**12, 10**12, outside_discovery.sum()
    )
    changed_candidates = build_candidates(mutated)
    changed_a_candidates = tuple(
        item for item in changed_candidates if item.feature_name == A_FEATURE
    )

    assert _condition_signature(a_candidates) == _condition_signature(
        changed_a_candidates
    )
    assert a_candidates
    assert {item.threshold for item in a_candidates} <= discovery_values


def test_candidate_thresholds_do_not_depend_on_future_label_availability() -> None:
    signals = _learning_signals()
    discovery = signals["analysis_split"].eq("discovery")
    censored = discovery & signals[A_FEATURE].lt(60)
    for column in ("weak_5d", "strong_5d", "deep_loss_5d", "bad_5d"):
        signals[column] = signals[column].astype("boolean")
    signals.loc[
        censored,
        ("weak_5d", "strong_5d", "deep_loss_5d", "bad_5d"),
    ] = pd.NA

    candidates = build_candidates(signals)
    lower_five = next(
        item
        for item in candidates
        if item.feature_name == A_FEATURE
        and item.operator == "le"
        and item.quantile == 0.05
    )
    all_discovery_values = signals.loc[discovery, A_FEATURE].to_numpy()
    expected = float(
        np.quantile(all_discovery_values, 0.05, method="nearest")
    )

    assert lower_five.threshold == expected


def test_evaluation_scopes_retain_unlabeled_signals_for_coverage() -> None:
    signals = _learning_signals()
    test_rows = signals["analysis_split"].eq("test")
    censored_index = signals.index[test_rows][0]
    for column in ("weak_5d", "strong_5d", "deep_loss_5d", "bad_5d"):
        signals[column] = signals[column].astype("boolean")
        signals.loc[censored_index, column] = pd.NA

    scopes = {
        (scope, year): frame
        for scope, year, frame in _scope_frames(signals)
    }

    assert len(scopes[("test", None)]) == test_rows.sum()
    assert len(scopes[("all_signals", None)]) == len(signals)
    test_metrics = _metrics(
        scopes[("test", None)],
        pd.Series(False, index=scopes[("test", None)].index),
    )
    assert test_metrics["unlabeled_count"] == 1


def test_rule_admissibility_requires_matched_label_coverage(cfg_factory) -> None:
    cfg = cfg_factory(min_matched_label_coverage_pct=0.90)
    metrics = {
        "sample_count": 1_000,
        "excluded_count": 100,
        "exclusion_rate": 0.10,
        "strong_retention_rate": 0.95,
        "matched_label_coverage_rate": 0.89,
        "bad_lift": 1.20,
        "bad_capture_rate": 0.20,
    }

    assert not _admissible(metrics, cfg)
    metrics["matched_label_coverage_rate"] = 0.90
    assert _admissible(metrics, cfg)


def test_test_split_mutation_cannot_change_selected_rules(cfg_factory) -> None:
    cfg = cfg_factory()
    signals = _learning_signals()
    candidates = build_candidates(signals)
    selected = select_rules(signals, candidates, cfg)
    assert selected.stage_a

    mutated = signals.copy()
    test_rows = mutated["analysis_split"].eq("test")
    mutated.loc[test_rows, A_FEATURE] = -10**9
    mutated.loc[test_rows, "weak_5d"] = False
    mutated.loc[test_rows, "deep_loss_5d"] = True
    mutated.loc[test_rows, "bad_5d"] = True
    mutated.loc[test_rows, "strong_5d"] = True

    changed_candidates = build_candidates(mutated)
    changed_selected = select_rules(mutated, changed_candidates, cfg)

    assert _condition_signature(candidates) == _condition_signature(
        changed_candidates
    )
    assert selected == changed_selected


@pytest.mark.parametrize("operator", ["le", "ge"])
def test_condition_never_matches_null_or_non_numeric(operator: str) -> None:
    frame = pd.DataFrame({A_FEATURE: [None, np.nan, pd.NA, "bad", 1.0, 2.0]})
    condition = Condition(
        rule_id="NULL_TEST",
        feature_group="A",
        feature_name=A_FEATURE,
        operator=operator,
        threshold=1.5,
        quantile=0.5,
    )

    matches = condition.matches(frame)

    assert matches.iloc[:4].tolist() == [False, False, False, False]
    assert matches.iloc[4:].tolist() == (
        [True, False] if operator == "le" else [False, True]
    )


def _three_group_signals(per_split: int = 400) -> pd.DataFrame:
    frame = _blank_signals(per_split * 3)
    frame["analysis_split"] = np.repeat(
        ("discovery", "validation", "test"), per_split
    )
    local_position = np.tile(np.arange(per_split), 3)
    bad = local_position < 180
    _set_outcomes(frame, bad)
    frame[A_FEATURE] = np.where(local_position < 60, 0.0, 1.0)
    frame[B_FEATURE] = np.where(
        (local_position >= 60) & (local_position < 120), 0.0, 1.0
    )
    frame[C_FEATURE] = np.where(
        (local_position >= 120) & (local_position < 180), 0.0, 1.0
    )
    frame["local_position"] = local_position
    return frame


def _three_conditions() -> tuple[Condition, ...]:
    return (
        Condition("RULE_A", "A", A_FEATURE, "le", 0.5, 0.3),
        Condition("RULE_B", "B", B_FEATURE, "le", 0.5, 0.3),
        Condition("RULE_C", "C", C_FEATURE, "le", 0.5, 0.3),
    )


def test_selection_adds_at_most_one_rule_per_group_and_applies_or(
    cfg_factory,
) -> None:
    signals = _three_group_signals()
    conditions = _three_conditions()
    cfg = cfg_factory(max_candidate_match_pct=0.60)

    selected = select_rules(signals, conditions, cfg)

    assert [item.feature_group for item in selected.stage_a] == ["A"]
    assert [item.feature_group for item in selected.stage_ab] == ["A", "B"]
    assert [item.feature_group for item in selected.stage_abc] == ["A", "B", "C"]
    assert len(selected.final) <= cfg.max_rule_conditions
    assert all(
        sum(item.feature_group == group for item in selected.final) <= 1
        for group in ("A", "B", "C")
    )

    decided = apply_selected_rules(
        signals.drop(columns="local_position"), selected
    )
    local_position = signals["local_position"].to_numpy()
    expected_excluded = local_position < 180
    assert decided["include_stage_abc"].tolist() == (~expected_excluded).tolist()
    assert decided["filter_decision"].tolist() == np.where(
        expected_excluded, "exclude", "include"
    ).tolist()
    assert decided.loc[local_position < 60, "matched_rule_ids"].eq("RULE_A").all()
    assert decided.loc[
        (local_position >= 60) & (local_position < 120), "matched_rule_ids"
    ].eq("RULE_B").all()
    assert decided.loc[
        (local_position >= 120) & (local_position < 180), "matched_rule_ids"
    ].eq("RULE_C").all()
    assert decided.loc[local_position >= 180, "matched_rule_ids"].isna().all()

    capped = select_rules(
        signals,
        conditions,
        cfg_factory(max_candidate_match_pct=0.60, max_rule_conditions=2),
    )
    assert [item.feature_group for item in capped.final] == ["A", "B"]


def test_metrics_use_exact_denominators_and_return_none_for_zero_denominators() -> None:
    frame = pd.DataFrame(
        {
            "weak_5d": [True, False, False, False, pd.NA],
            "deep_loss_5d": [False, True, False, False, pd.NA],
            "bad_5d": [True, True, False, False, pd.NA],
            "strong_5d": [False, False, True, False, pd.NA],
            "late_strong_10d": [True, False, False, False, pd.NA],
            "late_strong_20d": [False, True, False, False, pd.NA],
        }
    )
    excluded = pd.Series([True, True, True, False, True])

    metrics = _metrics(frame, excluded)

    assert metrics["signal_count"] == 5
    assert metrics["sample_count"] == 4
    assert metrics["unlabeled_count"] == 1
    assert metrics["matched_signal_count"] == 4
    assert metrics["matched_unlabeled_count"] == 1
    assert metrics["weak_count"] == 1
    assert metrics["deep_loss_count"] == 1
    assert metrics["bad_count"] == 2
    assert metrics["strong_count"] == 1
    assert metrics["excluded_count"] == 3
    assert metrics["excluded_bad_count"] == 2
    assert metrics["excluded_strong_count"] == 1
    assert metrics["label_coverage_rate"] == pytest.approx(4 / 5)
    assert metrics["matched_label_coverage_rate"] == pytest.approx(3 / 4)
    assert metrics["exclusion_rate"] == pytest.approx(3 / 4)
    assert metrics["weak_capture_rate"] == pytest.approx(1.0)
    assert metrics["deep_loss_capture_rate"] == pytest.approx(1.0)
    assert metrics["bad_capture_rate"] == pytest.approx(1.0)
    assert metrics["strong_rejection_rate"] == pytest.approx(1.0)
    assert metrics["strong_retention_rate"] == pytest.approx(0.0)
    assert metrics["excluded_bad_rate"] == pytest.approx(2 / 3)
    assert metrics["bad_lift"] == pytest.approx(4 / 3)
    assert metrics["retained_bad_rate"] == pytest.approx(0.0)
    assert metrics["retained_strong_rate"] == pytest.approx(0.0)
    assert metrics["late_strong_10d_rejection_rate"] == pytest.approx(1.0)
    assert metrics["late_strong_20d_rejection_rate"] == pytest.approx(1.0)

    empty = frame.iloc[0:0]
    zero = _metrics(empty, pd.Series(dtype=bool))
    for name in (
        "exclusion_rate",
        "weak_capture_rate",
        "deep_loss_capture_rate",
        "bad_capture_rate",
        "strong_rejection_rate",
        "strong_retention_rate",
        "excluded_bad_rate",
        "bad_lift",
        "retained_bad_rate",
        "retained_strong_rate",
        "late_strong_10d_rejection_rate",
        "late_strong_20d_rejection_rate",
    ):
        assert zero[name] is None


def test_tied_rule_selection_is_deterministic_across_input_order(
    cfg_factory,
) -> None:
    signals = _learning_signals()
    second_feature = "daily_traded_notional_vs_sma21_prior_ratio"
    signals[second_feature] = signals[A_FEATURE]
    candidates = (
        Condition("RULE_Z", "A", second_feature, "le", 59.0, 0.3),
        Condition("RULE_A", "A", A_FEATURE, "le", 59.0, 0.3),
    )
    cfg = cfg_factory(max_rule_conditions=1)

    selected_forward = select_rules(signals, candidates, cfg)
    selected_reverse = select_rules(
        signals.sample(frac=1.0, random_state=17),
        tuple(reversed(candidates)),
        cfg,
    )

    assert selected_forward == selected_reverse
    assert selected_forward.final[0].rule_id == "RULE_A"
    assert set(FEATURE_GROUPS) == {"A", "B", "C"}


def test_research_end_to_end_emits_both_complete_table_contracts(
    cfg_factory,
) -> None:
    signals = _learning_signals()

    result = run_research(signals, cfg_factory())

    assert tuple(result.signals.columns) == SIGNAL_COLUMNS
    assert tuple(result.rules.columns) == RULE_COLUMNS
    assert len(result.signals) == len(signals)
    baseline_all = result.rules.loc[
        result.rules["rule_id"].eq("BASELINE_INCLUDE_ALL")
        & result.rules["evaluation_scope"].eq("all_signals")
    ]
    assert len(baseline_all) == 1
    assert baseline_all.iloc[0]["signal_count"] == len(signals)
    assert baseline_all.iloc[0]["sample_count"] == len(signals)
