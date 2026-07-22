from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from stock_analyser_filter_research import db
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)
from stock_analyser_filter_research.research import (
    CompositeCondition,
    CandidateEvaluation,
    Condition,
    ConditionTemplate,
    ENTRY_CONFIRMATION_SPECS,
    ENTRY_EXCLUSION_SPECS,
    FoldResult,
    ObjectiveSelection,
    PatternTemplate,
    SelectedSummary,
    _Evaluator,
    _apply_max_stat_permutation_gate,
    _choose_sequential_early_prefixes,
    _rule_row,
    _sequential_policy_metrics,
    _sequential_policy_state,
    _template_text,
    apply_early_decisions,
    apply_management_decisions,
    build_candidate_templates,
    build_walk_forward_folds,
    fit_template,
    management_specs,
    objective_metrics,
    run_research,
    select_objective,
)


A_FEATURE = "adjusted_volume_vs_sma21_prior_ratio"
B_FEATURE = "prior_return_5d_pct"
C_FEATURE = "prior_volume_sma5_vs21_ratio"
E_FEATURE = "close_return_from_signal_pct"


def _cfg(cfg_factory, **overrides):
    return cfg_factory(signal_start_date=date(2016, 1, 1), **overrides)


def _blank(columns: tuple[str, ...], size: int) -> pd.DataFrame:
    return pd.DataFrame(
        {column: pd.Series([pd.NA] * size, dtype="object") for column in columns}
    )


def _signals(rows_per_year: int = 100) -> pd.DataFrame:
    dates = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.bdate_range(f"{year}-01-02", periods=rows_per_year).to_numpy()
                for year in range(2016, 2027)
            ]
        )
    )
    frame = _blank(SIGNAL_COLUMNS, len(dates))
    frame["signal_date"] = dates
    frame["previous_session_date"] = dates - pd.offsets.BDay(1)
    frame["forward_5d_label_end_date"] = dates + pd.offsets.BDay(5)
    frame["symbol"] = [f"S{index:06d}" for index in range(len(frame))]
    frame["exchange"] = "NYSE"
    frame["cik"] = np.arange(1, len(frame) + 1)
    frame["price_continuity_segment"] = 1
    frame["currency"] = "USD"
    normalized_dates = pd.Series(dates).dt.date
    frame["analysis_split"] = np.select(
        [
            normalized_dates.le(date(2022, 12, 31)),
            normalized_dates.le(date(2024, 12, 31)),
            normalized_dates.le(date(2026, 7, 20)),
        ],
        ["discovery", "validation", "diagnostic"],
        default="holdout",
    )
    local = np.tile(np.arange(rows_per_year), 11)
    weak = local < 30
    loss = (local >= 30) & (local < 60)
    protected = (local >= 60) & (local < 80)
    frame["weak_5d"] = weak
    frame["loss_first_5d"] = loss
    frame["strong_first_5d"] = protected
    frame["terminal_stagnant_5d"] = weak
    frame["terminal_winner_5d"] = protected
    frame["strong_5d"] = protected
    frame["deep_loss_5d"] = loss
    frame["bad_5d"] = weak | loss
    frame["late_strong_10d"] = False
    frame["late_strong_20d"] = False
    frame[A_FEATURE] = np.where(weak, 0.0, 1.0)
    frame[B_FEATURE] = np.where(loss, 0.0, 1.0)
    frame[C_FEATURE] = 1.0
    frame["include_weak_filter"] = True
    frame["include_loss_first_filter"] = True
    frame["include_final"] = True
    frame["filter_decision"] = "include"
    return frame


def _empty_early() -> pd.DataFrame:
    return _blank(EARLY_CUT_COLUMNS, 0)


def _early_from_signals(signals: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for position, signal in signals.reset_index(drop=True).iterrows():
        local = position % 100
        for day in (1, 2, 3):
            rows.append(
                {
                    "signal_date": signal["signal_date"],
                    "landmark_day": day,
                    "landmark_date": pd.Timestamp(signal["signal_date"])
                    + pd.offsets.BDay(day),
                    "effective_session_date": pd.Timestamp(signal["signal_date"])
                    + pd.offsets.BDay(day + 1),
                    "horizon_end_date": pd.Timestamp(signal["signal_date"])
                    + pd.offsets.BDay(5),
                    "symbol": signal["symbol"],
                    "exchange": signal["exchange"],
                    "cik": signal["cik"],
                    "price_continuity_segment": 1,
                    "currency": "USD",
                    "landmark_observed": True,
                    "same_continuity_segment": True,
                    "eligible_at_landmark": True,
                    "active_at_landmark": True,
                    "prior_policy_cut_day": pd.NA,
                    "full_outcome_available": True,
                    "stagnant_to_day5": local < 15,
                    "loss_first_to_day5": 15 <= local < 30,
                    "strong_first_to_day5": 60 <= local < 80,
                    "bad_to_day5": local < 30,
                    "analysis_split": signal["analysis_split"],
                    "include_stagnation_filter": True,
                    "include_loss_filter": True,
                    "include_final": True,
                    "cut_decision": "hold",
                    E_FEATURE: 0.0 if local < 30 else 1.0,
                }
            )
    return pd.DataFrame(rows, columns=EARLY_CUT_COLUMNS)


def _template(rule_id: str, group: str, feature: str, quantile: float = 0.20):
    return ConditionTemplate(rule_id, group, feature, "le", quantile)


def test_management_decision_applies_exit_rule_and_preserves_hard_stop() -> None:
    spec = management_specs(20)[0]
    condition = Condition(
        "MANAGE_LOW_RETURN",
        "E",
        E_FEATURE,
        "le",
        0.0,
        None,
    )
    candidate = CandidateEvaluation(
        templates=(_template("MANAGE_LOW_RETURN", "E", E_FEATURE),),
        fold_results=(),
        pooled_metrics={},
        final_conditions=(condition,),
        development_metrics={},
        passes_development_gates=True,
        passes_stability_gates=True,
        stability={},
        selection_score=1.0,
    )
    selection = ObjectiveSelection(spec, candidate, (candidate,), (candidate,))
    summary = SelectedSummary(
        entry={},
        confirmation={},
        early_cut={},
        management={20: {spec.objective: selection}},
        entry_prefix_lengths={},
        confirmation_prefix_lengths={},
        early_cut_prefix_lengths={},
        management_prefix_lengths={20: {spec.objective: 1}},
    )
    landmarks = _blank(EARLY_CUT_COLUMNS, 3)
    landmarks["landmark_day"] = [5, 20, 20]
    landmarks["eligible_at_landmark"] = [False, True, True]
    landmarks["management_include_final"] = [False, True, True]
    landmarks["management_decision"] = ["hard_stop", "hold", "hold"]
    landmarks[E_FEATURE] = [-5.0, -1.0, 2.0]

    result = apply_management_decisions(landmarks, summary)

    assert result.loc[0, "management_decision"] == "hard_stop"
    assert not bool(result.loc[0, "management_include_final"])
    assert result.loc[1, "management_decision"] == "take_profit"
    assert not bool(result.loc[1, "management_include_final"])
    assert result.loc[1, "management_matched_rule_ids"] == "MANAGE_LOW_RETURN"
    assert result.loc[2, "management_decision"] == "hold"
    assert bool(result.loc[2, "management_include_final"])


def test_objective_metrics_have_exact_target_specific_denominators() -> None:
    frame = pd.DataFrame(
        {
            "target": [True, False, False, pd.NA],
            "protected": [False, True, False, pd.NA],
        }
    )
    metrics = objective_metrics(
        frame,
        pd.Series([True, True, False, True]),
        "target",
        "protected",
    )
    assert metrics["population_count"] == 4
    assert metrics["sample_count"] == 3
    assert metrics["matched_count"] == 3
    assert metrics["matched_labeled_count"] == 2
    assert metrics["objective_count"] == 1
    assert metrics["matched_objective_count"] == 1
    assert metrics["protected_count"] == 1
    assert metrics["matched_protected_count"] == 1
    assert metrics["matched_label_coverage_rate"] == pytest.approx(2 / 3)
    assert metrics["objective_capture_rate"] == 1.0
    assert metrics["objective_lift"] == pytest.approx(1.5)
    assert metrics["protected_retention_rate"] == 0.0


def test_objective_metrics_allow_overlapping_objective_and_protected_labels() -> None:
    frame = pd.DataFrame(
        {
            "target": [True, True, False],
            "protected": [True, False, True],
        }
    )
    metrics = objective_metrics(
        frame,
        pd.Series([True, True, True]),
        "target",
        "protected",
    )
    assert metrics["sample_count"] == 3
    assert metrics["objective_count"] == 2
    assert metrics["protected_count"] == 2
    assert metrics["matched_labeled_count"] == 3
    assert metrics["matched_objective_count"] == 2
    assert metrics["matched_protected_count"] == 2
    assert metrics["objective_count"] + metrics["protected_count"] > metrics["sample_count"]
    assert (
        metrics["matched_objective_count"] + metrics["matched_protected_count"]
        > metrics["matched_labeled_count"]
    )


def test_fold_threshold_uses_only_rows_before_fold_year(cfg_factory) -> None:
    cfg = _cfg(cfg_factory)
    signals = _signals()
    template = _template("RULE_A", "A", A_FEATURE)
    fold = build_walk_forward_folds(cfg)[0]
    condition = fit_template(
        signals, template, "signal_date", fold.threshold_fit_end_date
    )
    assert condition is not None
    mutated = signals.copy()
    mutated.loc[
        pd.to_datetime(mutated["signal_date"]).dt.year >= fold.year, A_FEATURE
    ] = -1_000_000.0
    changed = fit_template(
        mutated, template, "signal_date", fold.threshold_fit_end_date
    )
    assert changed == condition
    training_values = set(
        signals.loc[
            pd.to_datetime(signals["signal_date"]).dt.date
            <= fold.threshold_fit_end_date,
            A_FEATURE,
        ]
    )
    assert condition.threshold in training_values


def test_quantile_count_controls_template_grid() -> None:
    coarse = build_candidate_templates("entry_filter", quantile_count=4)
    fine = build_candidate_templates("entry_filter", quantile_count=20)
    assert len(coarse) < len(fine)
    atomic = [item for item in coarse if isinstance(item, ConditionTemplate)]
    patterns = [item for item in coarse if isinstance(item, PatternTemplate)]
    assert {item.quantile for item in atomic if item.quantile is not None} == {
        0.25,
        0.75,
    }
    assert any(item.fixed_threshold == 1.0 for item in atomic)
    assert any(
        item.feature_group == "P"
        and item.feature_name == "pattern_pullback_score_40d"
        and item.fixed_threshold == 70.0
        and item.operator == "ge"
        for item in atomic
    )
    expected_original_patterns = {
        "flat_base",
        "ordered_uptrend",
        "pullback_from_high",
        "v_recovery",
        "volume_dry_up_breakout",
        "distribution_top",
        "earnings_deterioration",
        "margin_compression",
        "balance_sheet_stress",
        "cashflow_weakness",
        "large_cap_low_atr_stagnation",
        "small_cap_bearish_high_volume",
        "high_volume_strong_close",
    }
    pattern_names = {item.pattern_name for item in patterns}
    assert expected_original_patterns <= pattern_names
    assert {
        "flat_base_k2_of_3",
        "ordered_uptrend_k2_of_4",
        "ordered_uptrend_k3_of_4",
        "v_recovery_k2_of_3",
    } <= pattern_names
    assert {
        "vcp",
        "high_tight_flag",
        "bull_flag",
        "darvas_box",
        "ascending_triangle",
        "cup_with_handle",
        "three_weeks_tight",
        "pocket_pivot",
        "rs_leader",
        "weinstein_stage2",
        "zanger_volume_breakout",
        "growth_leader",
        "quality_growth",
        "post_earnings_power",
        "market_confirmed_leader",
    } <= pattern_names
    assert {
        "micro_cap_range",
        "small_cap_range",
        "mid_cap_range",
        "large_cap_range",
        "mega_cap_range",
        "mid_cap_range_volume21_high",
        "large_cap_range_notional21_low",
    } <= pattern_names
    assert all(
        isinstance(item, ConditionTemplate)
        for item in build_candidate_templates("early_cut", 1, quantile_count=4)
    )


def test_max_stat_permutation_gate_is_deterministic_and_family_wise(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory, permutation_trial_count=49)
    dates = pd.bdate_range("2023-01-02", periods=200)
    frame = pd.DataFrame(
        {
            "signal_date": dates,
            "forward_5d_label_end_date": dates + pd.offsets.BDay(5),
            "weak_5d": np.arange(200) % 4 == 0,
            "strong_first_5d": np.arange(200) % 4 == 1,
            "balanced_a": np.isin(np.arange(200) % 4, [0, 1]).astype(float),
            "balanced_b": np.isin(np.arange(200) % 4, [0, 1]).astype(float),
            "perfect": (np.arange(200) % 4 != 0).astype(float),
            "bad_5d": np.isin(np.arange(200) % 4, [0, 1]),
            "confirmation": (np.arange(200) % 4 != 1).astype(float),
        }
    )

    def candidate(rule_id: str, feature: str) -> CandidateEvaluation:
        template = ConditionTemplate(rule_id, "I", feature, "le", None)
        condition = Condition(rule_id, "I", feature, "le", 0.5, None)
        return CandidateEvaluation(
            (template,), (), {}, (condition,), {}, True, True, {}, 0.0
        )

    null_candidates = [
        candidate("NULL_A", "balanced_a"),
        candidate("NULL_B", "balanced_b"),
    ]
    _apply_max_stat_permutation_gate(
        frame, ENTRY_EXCLUSION_SPECS[0], null_candidates, cfg
    )
    assert all(item.passes_multiple_testing is False for item in null_candidates)
    assert all(
        item.max_stat_permutation_p_value > cfg.max_stat_permutation_p_value
        for item in null_candidates
    )

    perfect = candidate("PERFECT", "perfect")
    _apply_max_stat_permutation_gate(frame, ENTRY_EXCLUSION_SPECS[0], [perfect], cfg)
    assert perfect.passes_multiple_testing is True
    assert perfect.max_stat_permutation_p_value == pytest.approx(1 / 50)

    overlapping_first = candidate("OVERLAP_FIRST", "confirmation")
    overlapping_second = candidate("OVERLAP_SECOND", "confirmation")
    _apply_max_stat_permutation_gate(
        frame, ENTRY_CONFIRMATION_SPECS[0], [overlapping_first], cfg
    )
    _apply_max_stat_permutation_gate(
        frame, ENTRY_CONFIRMATION_SPECS[0], [overlapping_second], cfg
    )
    assert overlapping_first.max_stat_permutation_p_value is not None
    assert overlapping_first.max_stat_permutation_p_value == (
        overlapping_second.max_stat_permutation_p_value
    )


def test_pattern_candidate_uses_and_inside_pattern_and_causal_thresholds(
    cfg_factory,
) -> None:
    frame = _signals()
    local = np.tile(np.arange(100), 11)
    frame["pattern_left"] = np.where(local < 45, 0.0, 1.0)
    frame["pattern_right"] = np.where((local >= 20) & (local < 65), 0.0, 1.0)
    template = PatternTemplate(
        "ENTRY_D_PATTERN_TEST",
        "D",
        "test_pattern",
        (
            ConditionTemplate(
                "ENTRY_D_PATTERN_TEST_C1", "D", "pattern_left", "le", 0.30
            ),
            ConditionTemplate(
                "ENTRY_D_PATTERN_TEST_C2", "D", "pattern_right", "le", 0.30
            ),
        ),
    )
    evaluator = _Evaluator(
        frame, ENTRY_EXCLUSION_SPECS[0], (template,), _cfg(cfg_factory)
    )

    candidate = evaluator.evaluate((template,))

    assert len(candidate.final_conditions) == 1
    compiled = candidate.final_conditions[0]
    assert isinstance(compiled, CompositeCondition)
    expected = compiled.clauses[0].matches(frame) & compiled.clauses[1].matches(frame)
    union = compiled.clauses[0].matches(frame) | compiled.clauses[1].matches(frame)
    pd.testing.assert_series_equal(compiled.matches(frame), expected)
    assert not expected.equals(union)
    assert " AND " in compiled.text
    assert all(
        condition.threshold_fit_end_date == date(2024, 12, 31)
        for condition in compiled.clauses
    )


def test_relaxed_pattern_requires_k_of_n_and_complete_inputs() -> None:
    frame = pd.DataFrame(
        {
            "left": [0.0, 0.0, 0.0],
            "middle": [0.0, 1.0, 0.0],
            "right": [1.0, 1.0, np.nan],
        }
    )
    clauses = tuple(
        Condition(name.upper(), "D", name, "le", 0.5, None)
        for name in ("left", "middle", "right")
    )
    condition = CompositeCondition(
        "RELAXED",
        "D",
        "relaxed_test",
        clauses,
        minimum_clause_count=2,
    )

    assert condition.matches(frame).tolist() == [True, False, False]
    assert "at least 2 of 3" in condition.text


def test_rule_rows_expose_pattern_mode_window_and_score_threshold() -> None:
    metrics = objective_metrics(
        pd.DataFrame(
            {
                "strong_first_5d": [True, False],
                "bad_5d": [False, True],
            }
        ),
        pd.Series([True, False]),
        "strong_first_5d",
        "bad_5d",
    )
    score = Condition(
        "SCORE",
        "P",
        "pattern_pullback_score_40d",
        "ge",
        70.0,
        None,
    )
    row = _rule_row(
        rule_id="SCORE",
        result_kind="selected_filter",
        spec=ENTRY_CONFIRMATION_SPECS[0],
        conditions=(score,),
        evaluation_scope="validation",
        scope_year=None,
        period_start=date(2023, 1, 1),
        period_end=date(2024, 12, 31),
        metrics=metrics,
        is_selected=True,
        is_final_filter=True,
    )

    assert tuple(row) == RULE_COLUMNS
    assert row["pattern_name"] == "pullback"
    assert row["pattern_match_mode"] == "score_threshold"
    assert row["pattern_score_window_sessions"] == 40
    assert row["pattern_score_threshold_pct"] == 70.0


def test_structural_minimum_prevents_zero_count_distribution_clause(
    cfg_factory,
) -> None:
    frame = _signals()
    frame["distribution_count"] = 0.0
    template = ConditionTemplate(
        "COUNT_RULE", "D", "distribution_count", "ge", 0.80, 1.0
    )

    condition = fit_template(
        frame,
        template,
        "signal_date",
        date(2024, 12, 31),
    )

    assert condition is not None
    assert condition.threshold == 1.0
    assert not condition.matches(frame).any()


def test_market_cap_patterns_include_fixed_disjoint_ranges_and_interactions() -> None:
    patterns = [
        item
        for item in build_candidate_templates("entry_filter")
        if isinstance(item, PatternTemplate) and item.feature_group == "M"
    ]

    assert {item.pattern_name for item in patterns} == {
        "high_volume_strong_close",
        "large_cap_low_atr_stagnation",
        "small_cap_bearish_high_volume",
        "micro_cap_range",
        "small_cap_range",
        "mid_cap_range",
        "large_cap_range",
        "mega_cap_range",
    }
    clauses = {
        pattern.pattern_name: {
            clause.feature_name: clause for clause in pattern.clauses
        }
        for pattern in patterns
    }
    assert clauses["large_cap_low_atr_stagnation"][
        "market_cap_usd"
    ].fixed_threshold == 10_000_000_000.0
    assert clauses["small_cap_bearish_high_volume"][
        "market_cap_usd"
    ].fixed_threshold == 3_000_000_000.0
    assert clauses["small_cap_bearish_high_volume"][
        "adjusted_volume_vs_sma21_prior_ratio"
    ].quantile == 0.70
    assert clauses["high_volume_strong_close"][
        "signal_close_location_value"
    ].fixed_threshold == 0.65
    assert clauses["micro_cap_range"]["market_cap_usd"].fixed_threshold == (
        299_999_999.0
    )
    mid_cap_range = next(
        item for item in patterns if item.pattern_name == "mid_cap_range"
    )
    assert {
        item.operator: item.fixed_threshold for item in mid_cap_range.clauses
    } == {"ge": 2_000_000_000.0, "le": 9_999_999_999.0}
    cap_activity_interactions = [
        item
        for item in build_candidate_templates("entry_filter")
        if isinstance(item, PatternTemplate)
        and item.feature_group == "I"
        and "_cap_range_" in item.pattern_name
    ]
    assert len(cap_activity_interactions) == 20


def test_fixed_threshold_is_not_refit_and_is_identified_in_rule_text() -> None:
    frame = _signals()
    frame["fixed_feature"] = np.arange(len(frame), dtype=float)
    template = ConditionTemplate(
        "FIXED_RULE",
        "M",
        "fixed_feature",
        "ge",
        None,
        fixed_threshold=10_000_000_000.0,
    )

    first = fit_template(frame, template, "signal_date", date(2020, 12, 31))
    second = fit_template(frame, template, "signal_date", date(2024, 12, 31))

    assert first is not None
    assert second is not None
    assert first.threshold == second.threshold == 10_000_000_000.0
    assert first.quantile is None
    assert "fixed 1e+10 threshold" in _template_text((template,))


def test_evaluator_compiles_fixed_and_quantile_clauses_together(
    cfg_factory,
) -> None:
    frame = _signals()
    frame["fixed_feature"] = 20.0
    frame["quantile_feature"] = np.tile(np.arange(100), 11)
    template = PatternTemplate(
        "ENTRY_M_PATTERN_TEST",
        "M",
        "fixed_quantile_test",
        (
            ConditionTemplate(
                "ENTRY_M_PATTERN_TEST_C1",
                "M",
                "fixed_feature",
                "ge",
                None,
                fixed_threshold=10.0,
            ),
            ConditionTemplate(
                "ENTRY_M_PATTERN_TEST_C2",
                "M",
                "quantile_feature",
                "le",
                0.30,
            ),
        ),
    )

    candidate = _Evaluator(
        frame,
        ENTRY_EXCLUSION_SPECS[0],
        (template,),
        _cfg(cfg_factory),
    ).evaluate((template,))

    compiled = candidate.final_conditions[0]
    assert isinstance(compiled, CompositeCondition)
    assert compiled.clauses[0].threshold == 10.0
    assert compiled.clauses[0].quantile is None
    assert compiled.clauses[1].quantile == 0.30


def test_a_b_c_templates_compete_globally_and_future_diagnostic_is_frozen(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory)
    signals = _signals()
    templates = (
        _template("RULE_C", "C", C_FEATURE),
        _template("RULE_B", "B", B_FEATURE),
        _template("RULE_A", "A", A_FEATURE),
    )
    selected = select_objective(signals, ENTRY_EXCLUSION_SPECS[0], templates, cfg)
    assert selected.selected.templates
    assert selected.selected.templates[0].feature_group == "A"

    mutated = signals.copy()
    diagnostic = (
        pd.to_datetime(mutated["signal_date"]).dt.date > cfg.validation_end_date
    )
    mutated.loc[diagnostic, A_FEATURE] = -1e12
    mutated.loc[diagnostic, "weak_5d"] = False
    changed = select_objective(
        mutated, ENTRY_EXCLUSION_SPECS[0], tuple(reversed(templates)), cfg
    )
    assert tuple(item.rule_id for item in selected.selected.templates) == tuple(
        item.rule_id for item in changed.selected.templates
    )
    assert tuple(
        item.threshold for item in selected.selected.final_conditions
    ) == tuple(item.threshold for item in changed.selected.final_conditions)


def test_beam_search_can_select_two_conditions_with_one_point_gain(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory, min_selection_score_improvement=0.01)
    signals = _signals()
    local = np.tile(np.arange(100), 11)
    signals["weak_5d"] = local < 30
    signals[A_FEATURE] = np.where(local < 15, 0.0, 1.0)
    signals[C_FEATURE] = np.where((local >= 15) & (local < 30), 0.0, 1.0)
    templates = (
        _template("RULE_A", "A", A_FEATURE, 0.10),
        _template("RULE_C", "C", C_FEATURE, 0.10),
    )
    selected = select_objective(signals, ENTRY_EXCLUSION_SPECS[0], templates, cfg)
    assert len(selected.selected.templates) == 2
    assert {item.feature_group for item in selected.selected.templates} == {"A", "C"}
    assert selected.selected.pooled_metrics["objective_capture_rate"] == pytest.approx(
        1.0
    )

    capped = select_objective(
        signals,
        ENTRY_EXCLUSION_SPECS[0],
        templates,
        _cfg(cfg_factory, max_conditions_per_objective=1),
    )
    assert len(capped.selected.templates) == 1


def test_beam_search_can_select_three_complementary_conditions(
    cfg_factory,
) -> None:
    cfg = _cfg(
        cfg_factory,
        max_conditions_per_objective=3,
        rule_search_beam_width=10,
        min_selection_score_improvement=0.01,
    )
    signals = _signals(rows_per_year=200)
    local = np.tile(np.arange(200), 11)
    signals["weak_5d"] = local < 60
    signals["strong_first_5d"] = (local >= 120) & (local < 160)
    signals[A_FEATURE] = np.where(local < 20, 0.0, 1.0)
    signals[B_FEATURE] = np.where(
        (local >= 20) & (local < 40), 0.0, 1.0
    )
    signals[C_FEATURE] = np.where(
        (local >= 40) & (local < 60), 0.0, 1.0
    )
    templates = (
        ConditionTemplate(
            "RULE_A", "A", A_FEATURE, "le", None, fixed_threshold=0.0
        ),
        ConditionTemplate(
            "RULE_B", "B", B_FEATURE, "le", None, fixed_threshold=0.0
        ),
        ConditionTemplate(
            "RULE_C", "C", C_FEATURE, "le", None, fixed_threshold=0.0
        ),
    )

    selected = select_objective(
        signals, ENTRY_EXCLUSION_SPECS[0], templates, cfg
    )

    assert len(selected.selected.templates) == 3
    assert {item.rule_id for item in selected.selected.templates} == {
        "RULE_A",
        "RULE_B",
        "RULE_C",
    }
    assert selected.selected.pooled_metrics["objective_capture_rate"] == (
        pytest.approx(1.0)
    )


def test_entry_union_does_not_require_weak_rule_to_lift_loss_objective(
    cfg_factory,
    monkeypatch,
) -> None:
    cfg = _cfg(cfg_factory)
    signals = _signals()
    monkeypatch.setattr(
        "stock_analyser_filter_research.research.build_candidate_templates",
        lambda family, landmark_day=None, quantile_count=20: (
            (_template("RULE_WEAK_ONLY", "A", A_FEATURE),)
            if family == "entry_filter"
            else (_template(f"RULE_E_D{landmark_day}", "E", E_FEATURE),)
        ),
    )

    result = run_research(
        signals,
        _empty_early(),
        pd.bdate_range("2015-01-01", "2026-07-20"),
        cfg,
    )

    lengths = result.selected.entry_prefix_lengths
    assert lengths["loss_first_5d"] == 0
    # The same fitted rule explains both weak and terminal-stagnant rows. The
    # union deliberately persists it only once, independent of objective owner.
    assert lengths["weak_5d"] + lengths["terminal_stagnant_5d"] == 1
    weak_rows = result.signals["weak_5d"].astype("boolean").fillna(False)
    assert result.signals.loc[weak_rows, "filter_decision"].eq("exclude").all()
    assert result.signals.loc[~weak_rows, "filter_decision"].eq("include").all()


def test_pair_upgrade_requires_net_capture_minus_protected_damage(
    cfg_factory,
) -> None:
    cfg = _cfg(
        cfg_factory,
        min_protected_retention_pct=0.85,
        min_fold_protected_retention_pct=0.80,
    )
    signals = _signals(rows_per_year=200)
    local = np.tile(np.arange(200), 11)
    weak = local < 60
    loss = (local >= 60) & (local < 120)
    protected = (local >= 120) & (local < 160)
    signals["weak_5d"] = weak
    signals["loss_first_5d"] = loss
    signals["strong_first_5d"] = protected
    signals[A_FEATURE] = np.where(local < 30, 0.0, 1.0)
    # Adds seven target rows, but also five protected and nine other rows.
    # Capture rises by 7/60 while protected rejection rises by 5/40.
    signals[C_FEATURE] = np.where(
        ((local >= 30) & (local < 37))
        | ((local >= 60) & (local < 69))
        | ((local >= 120) & (local < 125)),
        0.0,
        1.0,
    )
    rule_a = _template("RULE_A", "A", A_FEATURE, 0.10)
    rule_c = _template("RULE_C", "C", C_FEATURE, 0.10)

    selected = select_objective(
        signals, ENTRY_EXCLUSION_SPECS[0], (rule_a, rule_c), cfg
    )

    assert selected.selected.templates == (rule_a,)
    assert selected.selected.selection_score == pytest.approx(0.5)
    tested_pair = next(
        candidate
        for candidate in selected.candidates
        if len(candidate.templates) == 2
    )
    assert tested_pair.selection_score < (
        selected.selected.selection_score + cfg.min_selection_score_improvement
    )


def test_final_refit_falls_back_to_safe_prefix_using_only_development_data(
    cfg_factory,
    monkeypatch,
) -> None:
    cfg = _cfg(
        cfg_factory,
        min_protected_retention_pct=0.90,
        min_fold_protected_retention_pct=0.80,
    )
    signals = _signals(rows_per_year=200)
    local = np.tile(np.arange(200), 11)
    years = pd.to_datetime(signals["signal_date"]).dt.year.to_numpy()
    weak = local < 60
    loss = (local >= 60) & (local < 120)
    protected = (local >= 120) & (local < 160)
    signals["weak_5d"] = weak
    signals["loss_first_5d"] = loss
    signals["strong_first_5d"] = protected
    signals[A_FEATURE] = np.where(local < 30, 0.0, 1.0)
    signals[C_FEATURE] = np.where(
        ((local >= 30) & (local < 60))
        | (
            np.isin(years, [2023, 2024])
            & (local >= 120)
            & (local < 125)
        ),
        0.0,
        1.0,
    )
    templates = (
        _template("RULE_A", "A", A_FEATURE, 0.10),
        _template("RULE_C", "C", C_FEATURE, 0.10),
    )
    before_refit_gate = select_objective(
        signals, ENTRY_EXCLUSION_SPECS[0], templates, cfg
    )
    assert len(before_refit_gate.selected.templates) == 2
    monkeypatch.setattr(
        "stock_analyser_filter_research.research.build_candidate_templates",
        lambda family, landmark_day=None, quantile_count=20: (
            templates
            if family == "entry_filter"
            else (_template(f"RULE_E_D{landmark_day}", "E", E_FEATURE),)
        ),
    )

    result = run_research(
        signals,
        _empty_early(),
        pd.bdate_range("2015-01-01", "2026-12-31"),
        cfg,
    )
    assert result.selected.entry_prefix_lengths["weak_5d"] == 1
    chosen = result.selected.entry["weak_5d"].prefixes[1]
    assert tuple(item.rule_id for item in chosen.templates) == ("RULE_A",)

    # Even an extreme post-validation mutation must not alter the fallback.
    changed = signals.copy()
    diagnostic = pd.to_datetime(changed["signal_date"]).dt.date > cfg.validation_end_date
    changed.loc[diagnostic, C_FEATURE] = -1e12
    changed_result = run_research(
        changed,
        _empty_early(),
        pd.bdate_range("2015-01-01", "2026-12-31"),
        cfg,
    )
    assert changed_result.selected.entry_prefix_lengths["weak_5d"] == 1
    assert tuple(
        condition.threshold
        for condition in changed_result.selected.entry["weak_5d"].prefixes[
            1
        ].final_conditions
    ) == tuple(condition.threshold for condition in chosen.final_conditions)


def test_condition_never_matches_null_or_non_numeric() -> None:
    frame = pd.DataFrame({A_FEATURE: [None, np.nan, pd.NA, "bad", 1.0, 2.0]})
    condition = Condition("RULE", "A", A_FEATURE, "le", 1.5, 0.3)
    assert condition.matches(frame).tolist() == [
        False,
        False,
        False,
        False,
        True,
        False,
    ]


def test_early_decision_states_are_causal_and_reason_only_exists_for_cut(
    cfg_factory,
    monkeypatch,
) -> None:
    cfg = _cfg(cfg_factory)
    signals = _signals()
    monkeypatch.setattr(
        "stock_analyser_filter_research.research.build_candidate_templates",
        lambda family, landmark_day=None, quantile_count=20: (
            (_template("RULE_A", "A", A_FEATURE),)
            if family == "entry_filter"
            else (_template(f"RULE_E_D{landmark_day}", "E", E_FEATURE),)
        ),
    )
    result = run_research(
        signals,
        _empty_early(),
        pd.bdate_range("2015-01-01", "2026-07-31"),
        cfg,
    )
    early = _blank(EARLY_CUT_COLUMNS, 3)
    early["signal_date"] = pd.Timestamp("2024-01-02")
    early["landmark_day"] = 1
    early["landmark_date"] = [
        pd.Timestamp("2024-01-03"),
        pd.Timestamp("2024-01-03"),
        pd.NaT,
    ]
    # The decision is made from landmark-close information; a missing next
    # session must not turn an otherwise observable landmark into ineligible.
    early["effective_session_date"] = [pd.NaT, pd.Timestamp("2024-01-04"), pd.NaT]
    early["landmark_observed"] = [True, True, False]
    early["same_continuity_segment"] = [True, True, False]
    early["eligible_at_landmark"] = [True, False, False]
    early["cut_decision"] = ["hold", "not_eligible", "not_evaluable"]
    early[E_FEATURE] = [1.0, 1.0, 1.0]
    decided = apply_early_decisions(early, result.selected)
    assert decided["cut_decision"].tolist() == ["hold", "not_eligible", "not_evaluable"]
    assert decided["cut_reason"].isna().all()


def test_early_decisions_are_sequential_across_days_and_identities() -> None:
    empty = SimpleNamespace(final_conditions=())

    def selection(condition: Condition):
        chosen = SimpleNamespace(final_conditions=(condition,))
        return SimpleNamespace(selected=chosen, prefixes=(empty, chosen))

    early_selections = {}
    early_lengths = {}
    for day in (1, 2, 3):
        stagnant = selection(
            Condition(f"CUT_D{day}", "E", E_FEATURE, "le", 0.0, 0.2)
        )
        loss = selection(
            Condition(f"LOSS_D{day}", "E", E_FEATURE, "ge", 99.0, 0.8)
        )
        early_selections[day] = {
            "stagnant_to_day5": stagnant,
            "loss_first_to_day5": loss,
        }
        early_lengths[day] = {
            "stagnant_to_day5": 1,
            "loss_first_to_day5": 0,
        }
    summary = SimpleNamespace(
        early_cut=early_selections,
        early_cut_prefix_lengths=early_lengths,
    )
    rows = []
    for symbol, cik, values in (
        ("AAA", 1, {1: -1.0, 2: -1.0, 3: -1.0}),
        ("BBB", 2, {1: 1.0, 2: -1.0, 3: -1.0}),
    ):
        for day in (1, 2, 3):
            rows.append(
                {
                    "signal_date": pd.Timestamp("2024-01-02"),
                    "landmark_day": day,
                    "landmark_date": pd.Timestamp("2024-01-02")
                    + pd.offsets.BDay(day),
                    "symbol": symbol,
                    "exchange": "NYSE",
                    "cik": cik,
                    "eligible_at_landmark": True,
                    "active_at_landmark": True,
                    "prior_policy_cut_day": pd.NA,
                    "cut_decision": "hold",
                    "include_stagnation_filter": True,
                    "include_loss_filter": True,
                    "include_final": True,
                    E_FEATURE: values[day],
                }
            )
    early = pd.DataFrame(rows, columns=EARLY_CUT_COLUMNS).sample(
        frac=1.0, random_state=7
    )

    decided = apply_early_decisions(early, summary).sort_values(
        ["symbol", "landmark_day"]
    )
    aaa = decided.loc[decided["symbol"].eq("AAA")]
    bbb = decided.loc[decided["symbol"].eq("BBB")]

    assert aaa["cut_decision"].tolist() == ["cut", "not_active", "not_active"]
    assert pd.isna(aaa["prior_policy_cut_day"].iloc[0])
    assert aaa["prior_policy_cut_day"].iloc[1:].astype(int).tolist() == [1, 1]
    assert bbb["cut_decision"].tolist() == ["hold", "cut", "not_active"]
    assert bbb["prior_policy_cut_day"].iloc[:2].isna().all()
    assert int(bbb["prior_policy_cut_day"].iloc[2]) == 2
    assert aaa["active_at_landmark"].tolist() == [True, False, False]
    assert bbb["active_at_landmark"].tolist() == [True, True, False]
    inactive = decided.loc[decided["cut_decision"].eq("not_active")]
    assert not inactive[
        ["include_stagnation_filter", "include_loss_filter", "include_final"]
    ].astype(bool).any().any()
    assert inactive["cut_reason"].isna().all()


def test_large_sequential_state_never_reactivates_a_cut_tuple_key() -> None:
    identity_count = 20_000
    frame = pd.DataFrame(
        {
            "signal_date": np.repeat(pd.Timestamp("2024-01-02"), identity_count * 3),
            "landmark_day": np.repeat([1, 2, 3], identity_count),
            "landmark_date": np.repeat(
                pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
                identity_count,
            ),
            "symbol": np.tile(
                [f"S{position:05d}" for position in range(identity_count)], 3
            ),
            "exchange": "NYSE",
            "cik": np.tile(np.arange(1, identity_count + 1), 3),
            "eligible_at_landmark": True,
            E_FEATURE: np.tile(
                np.where(np.arange(identity_count) < identity_count // 2, -1.0, 1.0),
                3,
            ),
        }
    )
    condition = Condition("CUT_HALF", "E", E_FEATURE, "le", 0.0, None)

    _, _, active, _, cut_days = _sequential_policy_state(
        frame,
        {1: (condition,), 2: (), 3: ()},
    )

    assert int(active[1].sum()) == identity_count
    assert int(active[2].sum()) == identity_count // 2
    assert int(active[3].sum()) == identity_count // 2
    assert len(cut_days) == identity_count // 2
    assert set(cut_days.values()) == {1}


def test_early_policy_normalizes_later_rows_without_d1_anchor_and_ignores_management() -> None:
    empty = SimpleNamespace(final_conditions=())
    selection = SimpleNamespace(selected=empty, prefixes=(empty,))
    summary = SimpleNamespace(
        early_cut={
            day: {
                "stagnant_to_day5": selection,
                "loss_first_to_day5": selection,
            }
            for day in (1, 2, 3)
        },
        early_cut_prefix_lengths={
            day: {
                "stagnant_to_day5": 0,
                "loss_first_to_day5": 0,
            }
            for day in (1, 2, 3)
        },
    )
    rows = []
    signal_date = pd.Timestamp("2024-01-02")
    for day in (1, 2, 3, 5, 20, 30):
        is_early = day <= 3
        rows.append(
            {
                "signal_date": signal_date,
                "landmark_day": day,
                "landmark_date": signal_date + pd.offsets.BDay(day),
                "symbol": "ORPHAN",
                "exchange": "NYSE",
                "cik": 3,
                "eligible_at_landmark": day != 1,
                "active_at_landmark": is_early and day != 1,
                "prior_policy_cut_day": pd.NA,
                "cut_decision": (
                    "not_eligible"
                    if day == 1
                    else "hold" if is_early else "not_evaluable"
                ),
                "include_stagnation_filter": is_early and day != 1,
                "include_loss_filter": is_early and day != 1,
                "include_final": is_early and day != 1,
                "management_include_final": not is_early,
                "management_decision": "not_evaluable" if is_early else "hold",
                E_FEATURE: 1.0,
            }
        )
    landmarks = pd.DataFrame(rows, columns=EARLY_CUT_COLUMNS)

    decided = apply_early_decisions(landmarks, summary)

    early = decided.loc[decided["landmark_day"].le(3)]
    assert not early["eligible_at_landmark"].astype(bool).any()
    assert not early["active_at_landmark"].astype(bool).any()
    assert early["cut_decision"].eq("not_eligible").all()
    management = decided.loc[decided["landmark_day"].isin((5, 20, 30))]
    assert management["eligible_at_landmark"].astype(bool).all()
    assert management["management_include_final"].astype(bool).all()
    assert management["management_decision"].eq("hold").all()
    assert management["cut_decision"].eq("not_evaluable").all()


def _actual_landmark_metric_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    signal_date = pd.Timestamp("2024-01-02")
    anchor_labels = {
        "A": (True, False, False),
        "B": (True, False, False),
        "C": (True, False, False),
        "D": (False, False, True),
    }
    day2_labels = {
        "A": (True, False, False),
        "B": (False, False, False),
        "C": (True, False, False),
        "D": (False, True, False),
    }
    for cik, symbol in enumerate(anchor_labels, start=1):
        for day in (1, 2, 3):
            stagnant, loss, strong = (
                anchor_labels[symbol]
                if day == 1
                else day2_labels[symbol]
                if day == 2
                else (False, False, False)
            )
            rows.append(
                {
                    "signal_date": signal_date,
                    "landmark_day": day,
                    "landmark_date": signal_date + pd.offsets.BDay(day),
                    "horizon_end_date": signal_date + pd.offsets.BDay(5),
                    "symbol": symbol,
                    "exchange": "NYSE",
                    "cik": cik,
                    "eligible_at_landmark": True,
                    "full_outcome_available": True,
                    "stagnant_to_day5": stagnant,
                    "loss_first_to_day5": loss,
                    "bad_to_day5": stagnant or loss,
                    "strong_first_to_day5": strong,
                    "d1_match": 0.0 if symbol == "A" and day == 1 else 1.0,
                    "d2_match": (
                        0.0 if day == 2 and symbol in {"A", "B", "D"} else 1.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_sequential_metrics_keep_d1_denominators_and_require_actual_cut_label() -> None:
    frame = _actual_landmark_metric_frame()
    d2 = Condition("D2", "E", "d2_match", "le", 0.0, 0.2)

    targets, bad = _sequential_policy_metrics(
        frame,
        {1: (), 2: (d2,), 3: ()},
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        available_through=date(2024, 12, 31),
        complete_only=True,
    )
    _, empty_bad = _sequential_policy_metrics(
        frame,
        {1: (), 2: (), 3: ()},
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        available_through=date(2024, 12, 31),
        complete_only=True,
    )

    # A is D1-bad and still bad at its D2 cut, so it receives credit. B was
    # D1-bad but is neutral at D2 and receives none. D is D1-protected and its
    # later bad label must never erase the protected rejection.
    assert bad["population_count"] == 4
    assert bad["sample_count"] == 4
    assert bad["objective_count"] == 3
    assert bad["protected_count"] == 1
    assert bad["matched_count"] == 3
    assert bad["matched_objective_count"] == 1
    assert bad["matched_protected_count"] == 1
    assert targets["stagnant_to_day5"]["matched_objective_count"] == 1
    assert targets["loss_first_to_day5"]["matched_objective_count"] == 0
    for name in ("population_count", "sample_count", "objective_count", "protected_count"):
        assert bad[name] == empty_bad[name]
    assert empty_bad["matched_count"] == 0


def test_sequential_metrics_use_only_the_first_cut_day_and_never_double_count() -> None:
    frame = _actual_landmark_metric_frame()
    d1 = Condition("D1", "E", "d1_match", "le", 0.0, 0.2)
    d2 = Condition("D2", "E", "d2_match", "le", 0.0, 0.2)

    _, _, _, _, cut_days = _sequential_policy_state(
        frame,
        {1: (d1,), 2: (d2,), 3: ()},
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        complete_only=True,
    )
    _, bad = _sequential_policy_metrics(
        frame,
        {1: (d1,), 2: (d2,), 3: ()},
        start=date(2024, 1, 1),
        end=date(2024, 12, 31),
        available_through=date(2024, 12, 31),
        complete_only=True,
    )

    assert cut_days == {
        (pd.Timestamp("2024-01-02"), "A", "NYSE", 1): 1,
        (pd.Timestamp("2024-01-02"), "B", "NYSE", 2): 2,
        (pd.Timestamp("2024-01-02"), "D", "NYSE", 4): 2,
    }
    assert bad["matched_count"] == 3
    assert bad["matched_objective_count"] == 1
    assert bad["matched_protected_count"] == 1
    assert bad["matched_objective_count"] <= bad["objective_count"]
    assert bad["matched_protected_count"] <= bad["protected_count"]


def test_sequential_selector_enforces_cumulative_strong_retention(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory)
    folds = build_walk_forward_folds(cfg)
    rows = []
    for year in range(2016, 2025):
        signal_dates = pd.bdate_range(f"{year}-01-02", periods=200)
        for position, signal_date in enumerate(signal_dates):
            for day in (1, 2, 3):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "landmark_day": day,
                        "landmark_date": signal_date + pd.offsets.BDay(day),
                        "horizon_end_date": signal_date + pd.offsets.BDay(5),
                        "symbol": f"S{year}{position:03d}",
                        "exchange": "NYSE",
                        "cik": year * 1_000 + position,
                        "eligible_at_landmark": True,
                        "full_outcome_available": True,
                        "stagnant_to_day5": position < 30,
                        "loss_first_to_day5": 30 <= position < 60,
                        "bad_to_day5": position < 60,
                        "strong_first_to_day5": 120 <= position < 160,
                        "d1_rule": (
                            0.0
                            if position < 25 or 120 <= position < 122
                            else 1.0
                        ),
                        "d2_rule": (
                            0.0
                            if 25 <= position < 50 or 122 <= position < 124
                            else 1.0
                        ),
                    }
                )
    frame = pd.DataFrame(rows)

    def prefixes(day: int, feature: str):
        template = ConditionTemplate(
            f"D{day}_{feature}", "E", feature, "le", 0.2
        )
        condition = Condition(
            template.rule_id, "E", feature, "le", 0.0, 0.2
        )
        empty = SimpleNamespace(
            templates=(),
            final_conditions=(),
            fold_results=tuple(FoldResult(fold, (), {}) for fold in folds),
        )
        chosen = SimpleNamespace(
            templates=(template,),
            final_conditions=(condition,),
            fold_results=tuple(
                FoldResult(fold, (condition,), {}) for fold in folds
            ),
        )
        return (empty, chosen)

    selections = {}
    maximums = {}
    for day in (1, 2, 3):
        feature = f"d{day}_rule" if day <= 2 else "d2_rule"
        active_prefixes = prefixes(day, feature)
        selections[day] = {
            "stagnant_to_day5": SimpleNamespace(prefixes=active_prefixes),
            "loss_first_to_day5": SimpleNamespace(
                prefixes=(active_prefixes[0],)
            ),
        }
        maximums[day] = {
            "stagnant_to_day5": 1 if day <= 2 else 0,
            "loss_first_to_day5": 0,
        }

    chosen = _choose_sequential_early_prefixes(
        frame, selections, maximums, cfg
    )

    assert sum(sum(day.values()) for day in chosen.values()) == 1
    assert not (
        chosen[1]["stagnant_to_day5"]
        and chosen[2]["stagnant_to_day5"]
    )


def test_sequential_selector_checks_safety_in_low_objective_folds(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory)
    folds = build_walk_forward_folds(cfg)
    rows = []
    for year in range(2016, 2025):
        signal_date = pd.Timestamp(f"{year}-06-01")
        for position in range(100):
            if year == 2024:
                bad = position < 4
                protected = 60 <= position < 80
                rule_match = (
                    bad
                    or 20 <= position < 23
                    or 60 <= position < 63
                )
            else:
                bad = position < 20
                protected = 60 <= position < 80
                rule_match = bad or (
                    year >= 2020 and 20 <= position < 40
                )
            for day in (1, 2, 3):
                rows.append(
                    {
                        "signal_date": signal_date,
                        "landmark_day": day,
                        "landmark_date": signal_date + pd.offsets.BDay(day),
                        "horizon_end_date": signal_date + pd.offsets.BDay(5),
                        "symbol": f"S{year}{position:03d}",
                        "exchange": "NYSE",
                        "cik": year * 1_000 + position,
                        "eligible_at_landmark": True,
                        "full_outcome_available": True,
                        "stagnant_to_day5": bad,
                        "loss_first_to_day5": False,
                        "bad_to_day5": bad,
                        "strong_first_to_day5": protected,
                        "d1_rule": (
                            0.0 if day == 1 and rule_match else 1.0
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    template = ConditionTemplate("D1_RULE", "E", "d1_rule", "le", 0.2)
    condition = Condition("D1_RULE", "E", "d1_rule", "le", 0.0, 0.2)
    empty = SimpleNamespace(
        templates=(),
        final_conditions=(),
        fold_results=tuple(FoldResult(fold, (), {}) for fold in folds),
    )
    active = SimpleNamespace(
        templates=(template,),
        final_conditions=(condition,),
        fold_results=tuple(
            FoldResult(fold, (condition,), {}) for fold in folds
        ),
    )
    selections = {
        day: {
            "stagnant_to_day5": SimpleNamespace(
                prefixes=(empty, active) if day == 1 else (empty,)
            ),
            "loss_first_to_day5": SimpleNamespace(prefixes=(empty,)),
        }
        for day in (1, 2, 3)
    }
    maximums = {
        day: {
            "stagnant_to_day5": 1 if day == 1 else 0,
            "loss_first_to_day5": 0,
        }
        for day in (1, 2, 3)
    }

    chosen = _choose_sequential_early_prefixes(
        frame, selections, maximums, cfg
    )

    # The 2024 fold has only four bad outcomes, so it is ineligible for the
    # lift-stability statistic.  Its 85% strong retention must nevertheless
    # veto the policy through the target-neutral fold-safety gate.
    assert sum(sum(day.values()) for day in chosen.values()) == 0


def test_sequential_selector_does_not_credit_d1_bad_after_neutral_d2_cut(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory)
    folds = build_walk_forward_folds(cfg)
    rows: list[dict[str, object]] = []
    for year in range(2016, 2025):
        for position, signal_date in enumerate(
            pd.bdate_range(f"{year}-01-02", periods=200)
        ):
            for day in (1, 2, 3):
                d1_bad = position < 60
                local_bad = d1_bad if day == 1 else False
                rows.append(
                    {
                        "signal_date": signal_date,
                        "landmark_day": day,
                        "landmark_date": signal_date + pd.offsets.BDay(day),
                        "horizon_end_date": signal_date + pd.offsets.BDay(5),
                        "symbol": f"S{year}{position:03d}",
                        "exchange": "NYSE",
                        "cik": year * 1_000 + position,
                        "eligible_at_landmark": True,
                        "full_outcome_available": True,
                        "stagnant_to_day5": local_bad,
                        "loss_first_to_day5": False,
                        "bad_to_day5": local_bad,
                        "strong_first_to_day5": 120 <= position < 160,
                        "d2_neutral_rule": (
                            0.0 if day == 2 and position < 25 else 1.0
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    template = ConditionTemplate(
        "D2_NEUTRAL", "E", "d2_neutral_rule", "le", 0.2
    )
    condition = Condition(
        "D2_NEUTRAL", "E", "d2_neutral_rule", "le", 0.0, 0.2
    )
    empty = SimpleNamespace(
        templates=(),
        final_conditions=(),
        fold_results=tuple(FoldResult(fold, (), {}) for fold in folds),
    )
    active = SimpleNamespace(
        templates=(template,),
        final_conditions=(condition,),
        fold_results=tuple(
            FoldResult(fold, (condition,), {}) for fold in folds
        ),
    )
    selections = {
        day: {
            "stagnant_to_day5": SimpleNamespace(
                prefixes=(empty, active) if day == 2 else (empty,)
            ),
            "loss_first_to_day5": SimpleNamespace(prefixes=(empty,)),
        }
        for day in (1, 2, 3)
    }
    maximums = {
        day: {
            "stagnant_to_day5": 1 if day == 2 else 0,
            "loss_first_to_day5": 0,
        }
        for day in (1, 2, 3)
    }

    chosen = _choose_sequential_early_prefixes(
        frame, selections, maximums, cfg
    )

    assert sum(sum(day.values()) for day in chosen.values()) == 0


def test_sequential_holdout_is_anchored_at_day1_not_later_landmarks() -> None:
    frame = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2026-07-17"),
                "landmark_day": day,
                "landmark_date": pd.Timestamp("2026-07-20")
                + pd.offsets.BDay(day - 1),
                "symbol": "AAA",
                "exchange": "NYSE",
                "cik": 1,
                "eligible_at_landmark": True,
                "full_outcome_available": True,
                E_FEATURE: -1.0,
            }
            for day in (1, 2, 3)
        ]
    )
    condition = Condition("D2", "E", E_FEATURE, "le", 0.0, 0.2)

    anchor, _, active, _, cut_days = _sequential_policy_state(
        frame,
        {1: (), 2: (condition,), 3: ()},
        start=date(2026, 7, 21),
        end=None,
    )

    assert anchor.empty
    assert not cut_days
    assert not any(mask.any() for mask in active.values())


def test_run_research_emits_exact_contracts_and_holdout_stays_null_until_minimum(
    cfg_factory,
    monkeypatch,
) -> None:
    cfg = _cfg(cfg_factory, min_holdout_sample_count=500)
    signals = _signals()
    trading_dates = pd.bdate_range("2015-01-01", "2026-07-20")
    monkeypatch.setattr(
        "stock_analyser_filter_research.research.build_candidate_templates",
        lambda family, landmark_day=None, quantile_count=20: (
            (
                _template("RULE_A", "A", A_FEATURE),
                _template("RULE_B", "B", B_FEATURE),
            )
            if family == "entry_filter"
            else (_template(f"RULE_E_D{landmark_day}", "E", E_FEATURE),)
        ),
    )
    result = run_research(
        signals,
        _empty_early(),
        trading_dates,
        cfg,
    )
    assert tuple(result.signals.columns) == SIGNAL_COLUMNS
    assert tuple(result.early_cuts.columns) == EARLY_CUT_COLUMNS
    assert tuple(result.rules.columns) == RULE_COLUMNS
    assert isinstance(result.selected_condition_count, int)
    assert isinstance(result.selected_text, str)
    assert set(result.rules["result_kind"]) <= {
        "baseline",
        "candidate_rule",
        "selected_filter",
    }
    assert set(result.rules["decision_family"]) == {
        "entry_filter",
        "entry_confirmation",
        "early_cut",
        "position_management",
    }
    assert result.rules["eligible_fold_count"].notna().all()
    assert result.rules["positive_lift_fold_count"].notna().all()
    assert result.rules["result_kind"].eq("baseline").any()
    assert (
        result.rules.loc[result.rules["period_end"].notna(), "period_start"]
        .notna()
        .all()
    )
    assert set(result.rules["evaluation_scope"]) <= {
        "development",
        "discovery",
        "validation",
        "diagnostic",
        "holdout",
        "all_signals",
        "calendar_year",
        "walk_forward_year",
        "walk_forward_pooled",
    }
    holdout = result.rules.loc[result.rules["evaluation_scope"].eq("holdout")]
    assert not holdout.empty
    assert holdout["passes_holdout"].isna().all()
    assert holdout["period_start"].isna().all()
    assert holdout["period_end"].isna().all()
    assert holdout["threshold_fit_end_date"].isna().all()
    assert (
        result.rules.loc[
            result.rules["evaluation_scope"].eq("walk_forward_year"), "scope_year"
        ]
        .dropna()
        .astype(int)
        .between(2020, 2024)
        .all()
    )
    point_in_time = result.rules.loc[
        result.rules["threshold_fit_end_date"].notna()
        & result.rules["period_end"].notna()
    ]
    assert (
        pd.to_datetime(point_in_time["threshold_fit_end_date"])
        <= pd.to_datetime(point_in_time["period_end"])
    ).all()
    strict = point_in_time.loc[
        point_in_time["evaluation_scope"].isin(
            ["walk_forward_year", "diagnostic", "holdout"]
        )
    ]
    assert strict["period_start"].notna().all()
    assert (
        pd.to_datetime(strict["threshold_fit_end_date"])
        < pd.to_datetime(strict["period_start"])
    ).all()
    components = result.rules.loc[
        result.rules["result_kind"].eq("selected_filter")
        & result.rules["is_selected"]
        & ~result.rules["is_final_filter"]
        & result.rules["selection_order"].notna()
    ]
    assert components["threshold_value"].notna().all()
    assert components["feature_name"].notna().all()
    required_rule_columns = [
        column
        for column, contract in db.RULE_COLUMN_CONTRACTS.items()
        if column != "result_id" and not contract[1]
    ]
    assert result.rules[required_rule_columns].notna().all().all()
    assert set(result.rules["objective"]) <= {
        "weak_5d",
        "loss_first_5d",
        "terminal_stagnant_5d",
        "stagnant_5d",
        "hard_stop_10pct_5d",
        "terminal_nonpositive_20d",
        "terminal_nonpositive_30d",
        "strong_first_5d",
        "terminal_winner_5d",
        "terminal_winner_20d",
        "terminal_winner_30d",
        "runner_60d",
        "runner_90d",
        "stagnant_to_day5",
        "loss_first_to_day5",
        "bad_to_day5",
        "take_profit_better_to_day20",
        "take_profit_better_to_day40",
        "take_profit_better_to_day60",
        "take_profit_better_to_day90",
    }
    assert set(result.rules["protected_outcome"]) <= {
        "strong_first_5d",
        "terminal_winner_5d",
        "terminal_winner_20d",
        "terminal_winner_30d",
        "runner_60d",
        "bad_5d",
        "terminal_stagnant_5d",
        "terminal_nonpositive_20d",
        "terminal_nonpositive_30d",
        "strong_first_to_day5",
        "continue_winner_to_day20",
        "continue_winner_to_day40",
        "continue_winner_to_day60",
        "continue_winner_to_day90",
    }
    permutation_rows = result.rules.loc[
        result.rules["result_kind"].eq("candidate_rule")
        & result.rules["max_stat_permutation_p_value"].notna()
    ]
    assert not permutation_rows.empty
    assert permutation_rows["multiple_testing_candidate_count"].gt(0).all()
    assert permutation_rows["permutation_trial_count"].eq(
        cfg.permutation_trial_count
    ).all()
    assert permutation_rows["max_stat_permutation_p_value"].between(0, 1).all()
    assert set(result.rules["feature_group"]) <= {
        "none",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "multiple",
    }
    assert set(result.rules["operator"].dropna()) <= {"le", "ge"}
    assert result.rules["is_final_filter"].astype(bool).any()


def test_run_research_emits_one_sequential_final_early_policy(
    cfg_factory,
    monkeypatch,
) -> None:
    cfg = _cfg(cfg_factory)
    signals = _signals()
    early = _early_from_signals(signals)
    monkeypatch.setattr(
        "stock_analyser_filter_research.research.build_candidate_templates",
        lambda family, landmark_day=None, quantile_count=20: (
            (_template("RULE_A", "A", A_FEATURE),)
            if family == "entry_filter"
            else (
                _template(
                    f"RULE_E_D{landmark_day}", "E", E_FEATURE
                ),
            )
        ),
    )

    result = run_research(
        signals,
        early,
        pd.bdate_range("2015-01-01", "2026-07-20"),
        cfg,
    )

    policy = result.rules.loc[
        result.rules["rule_id"].eq(
            "EARLY_CUT_D1_D3_SEQUENTIAL_D1_ANCHOR_FINAL"
        )
    ]
    assert not policy.empty
    assert policy["is_final_filter"].astype(bool).all()
    assert "bad_to_day5" in set(policy["objective"])
    pooled_policy = policy.loc[
        policy["evaluation_scope"].eq("walk_forward_pooled")
    ]
    assert pooled_policy["component_count"].gt(0).all()
    assert pooled_policy["threshold_value"].isna().all()
    assert pooled_policy["threshold_fit_end_date"].isna().all()
    assert pooled_policy["rule_text"].str.contains("D+", regex=False).all()
    assert pooled_policy["rule_text"].str.contains("causal Q", regex=False).all()
    fixed_policy = policy.loc[policy["evaluation_scope"].eq("validation")]
    assert fixed_policy["threshold_fit_end_date"].notna().all()
    assert not fixed_policy["rule_text"].str.contains(
        "causal Q", regex=False
    ).any()
    day_components = result.rules.loc[
        result.rules["rule_id"].str.contains("POLICY_COMPONENT", na=False)
        & result.rules["decision_family"].eq("early_cut")
    ]
    assert not day_components.empty
    assert not day_components["is_final_filter"].astype(bool).any()
    cuts_per_signal = (
        result.early_cuts["cut_decision"].eq("cut").groupby(
            [
                result.early_cuts["signal_date"],
                result.early_cuts["symbol"],
                result.early_cuts["exchange"],
                result.early_cuts["cik"],
            ]
        ).sum()
    )
    assert cuts_per_signal.le(1).all()
    inactive = result.early_cuts.loc[
        result.early_cuts["cut_decision"].eq("not_active")
    ]
    assert inactive["prior_policy_cut_day"].notna().all()
