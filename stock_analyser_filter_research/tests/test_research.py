from __future__ import annotations

from datetime import date

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
    Condition,
    ConditionTemplate,
    ENTRY_SPECS,
    apply_early_decisions,
    build_candidate_templates,
    build_walk_forward_folds,
    fit_template,
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


def _template(rule_id: str, group: str, feature: str, quantile: float = 0.20):
    return ConditionTemplate(rule_id, group, feature, "le", quantile)


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
    assert {item.quantile for item in coarse} == {0.25, 0.75}


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
    selected = select_objective(signals, ENTRY_SPECS[0], templates, cfg)
    assert selected.selected.templates
    assert selected.selected.templates[0].feature_group == "A"

    mutated = signals.copy()
    diagnostic = (
        pd.to_datetime(mutated["signal_date"]).dt.date > cfg.validation_end_date
    )
    mutated.loc[diagnostic, A_FEATURE] = -1e12
    mutated.loc[diagnostic, "weak_5d"] = False
    changed = select_objective(mutated, ENTRY_SPECS[0], tuple(reversed(templates)), cfg)
    assert tuple(item.rule_id for item in selected.selected.templates) == tuple(
        item.rule_id for item in changed.selected.templates
    )
    assert tuple(
        item.threshold for item in selected.selected.final_conditions
    ) == tuple(item.threshold for item in changed.selected.final_conditions)


def test_beam_search_can_select_two_conditions_with_one_point_gain(
    cfg_factory,
) -> None:
    cfg = _cfg(cfg_factory, min_selection_capture_improvement=0.01)
    signals = _signals()
    local = np.tile(np.arange(100), 11)
    signals["weak_5d"] = local < 30
    signals[A_FEATURE] = np.where(local < 15, 0.0, 1.0)
    signals[C_FEATURE] = np.where((local >= 15) & (local < 30), 0.0, 1.0)
    templates = (
        _template("RULE_A", "A", A_FEATURE, 0.10),
        _template("RULE_C", "C", C_FEATURE, 0.10),
    )
    selected = select_objective(signals, ENTRY_SPECS[0], templates, cfg)
    assert len(selected.selected.templates) == 2
    assert {item.feature_group for item in selected.selected.templates} == {"A", "C"}
    assert selected.selected.pooled_metrics["objective_capture_rate"] == pytest.approx(
        1.0
    )

    capped = select_objective(
        signals,
        ENTRY_SPECS[0],
        templates,
        _cfg(cfg_factory, max_conditions_per_objective=1),
    )
    assert len(capped.selected.templates) == 1


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
    assert set(result.rules["decision_family"]) == {"entry_filter", "early_cut"}
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
        "stagnant_to_day5",
        "loss_first_to_day5",
    }
    assert set(result.rules["protected_outcome"]) <= {
        "strong_first_5d",
        "strong_first_to_day5",
    }
    assert set(result.rules["feature_group"]) <= {
        "none",
        "A",
        "B",
        "C",
        "E",
        "multiple",
    }
    assert set(result.rules["operator"].dropna()) <= {"le", "ge"}
    assert result.rules["is_final_filter"].astype(bool).any()
