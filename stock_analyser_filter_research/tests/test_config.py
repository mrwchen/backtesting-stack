from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from stock_analyser_filter_research.config import Config


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"discovery_end_date": date(2024, 12, 31)}, "dates must satisfy"),
        ({"validation_end_date": date(2026, 7, 20)}, "dates must satisfy"),
        ({"signal_start_date": date(2023, 1, 1)}, "dates must satisfy"),
        (
            {"signal_end_date": date(2015, 12, 31)},
            "SIGNAL_END_DATE must be empty or not precede",
        ),
    ],
)
def test_v2_split_dates_must_be_strictly_ordered(cfg_factory, changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        replace(cfg_factory(), **changes).validate()


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"weak_5d_max_gain_pct": 0.0}, "WEAK_5D_MAX_GAIN_PCT"),
        ({"strong_5d_min_gain_pct": 2.0}, "must exceed"),
        ({"deep_loss_5d_max_loss_pct": 0.0}, "must be negative"),
        ({"quantile_count": 3}, "QUANTILE_COUNT"),
        ({"quantile_count": 21}, "QUANTILE_COUNT"),
        ({"max_conditions_per_objective": 0}, "MAX_CONDITIONS_PER_OBJECTIVE"),
        ({"max_conditions_per_objective": 4}, "MAX_CONDITIONS_PER_OBJECTIVE"),
        ({"hard_stop_5d_max_loss_pct": -4.0}, "HARD_STOP_5D_MAX_LOSS_PCT"),
        (
            {"continuation_winner_min_return_pct": 0.0},
            "CONTINUATION_WINNER_MIN_RETURN_PCT",
        ),
        ({"rule_search_beam_width": 0}, "RULE_SEARCH_BEAM_WIDTH"),
        ({"min_fold_sample_count": 0}, "fold sample/count"),
        ({"min_fold_objective_count": 0}, "fold sample/count"),
        ({"min_holdout_sample_count": 0}, "MIN_HOLDOUT_SAMPLE_COUNT"),
        (
            {"taxonomy_backcast_industry_min_members": 1},
            "TAXONOMY_BACKCAST_INDUSTRY_MIN_MEMBERS",
        ),
        (
            {"taxonomy_backcast_category_min_members": 1},
            "TAXONOMY_BACKCAST_CATEGORY_MIN_MEMBERS",
        ),
        (
            {"taxonomy_backcast_subcategory_min_members": 1},
            "TAXONOMY_BACKCAST_SUBCATEGORY_MIN_MEMBERS",
        ),
        ({"min_stable_fold_fraction": 0.0}, "MIN_STABLE_FOLD_FRACTION"),
        ({"min_fold_protected_retention_pct": 1.01}, "MIN_FOLD_PROTECTED"),
        ({"max_fold_match_pct": 0.0}, "MAX_FOLD_MATCH_PCT"),
        ({"min_protected_retention_pct": 1.01}, "MIN_PROTECTED_RETENTION"),
        (
            {"min_matched_label_coverage_pct": 0.0},
            "MIN_MATCHED_LABEL_COVERAGE_PCT",
        ),
        ({"min_candidate_match_pct": 0.0}, "candidate match percentages"),
        ({"max_candidate_match_pct": 1.0}, "candidate match percentages"),
        (
            {"min_candidate_match_pct": 0.40},
            "candidate match percentages",
        ),
        ({"min_objective_lift": 0.99}, "objective lift"),
        ({"min_fold_objective_lift": 0.99}, "objective lift"),
        (
            {"min_selection_score_improvement": -0.01},
            "MIN_SELECTION_SCORE_IMPROVEMENT",
        ),
        (
            {"min_selection_score_improvement": 1.01},
            "MIN_SELECTION_SCORE_IMPROVEMENT",
        ),
        ({"permutation_trial_count": 18}, "PERMUTATION_TRIAL_COUNT"),
        ({"max_stat_permutation_p_value": 0.0}, "MAX_STAT_PERMUTATION_P_VALUE"),
        ({"permutation_random_seed": -1}, "PERMUTATION_RANDOM_SEED"),
        ({"max_workers": 0}, "MAX_WORKERS"),
        ({"worker_identity_batch_size": 0}, "WORKER_IDENTITY_BATCH_SIZE"),
        ({"db_fetch_batch_size": 0}, "database batch sizes"),
        ({"db_copy_batch_size": 0}, "database batch sizes"),
    ],
)
def test_v2_research_and_runtime_bounds_are_enforced(
    cfg_factory, changes, message
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(cfg_factory(), **changes).validate()


@pytest.mark.parametrize(
    "changes",
    [
        {"walk_forward_first_year": 2016},
        {"walk_forward_first_year": 2025},
        {"min_walk_forward_folds": 6},
        {"min_walk_forward_folds": 0},
        {"signal_end_date": date(2022, 12, 31)},
    ],
)
def test_v2_walk_forward_window_must_fit_development_period(
    cfg_factory, changes
) -> None:
    with pytest.raises(ValueError, match="WALK_FORWARD|MIN_WALK_FORWARD"):
        replace(cfg_factory(), **changes).validate()


def test_v2_result_tables_must_be_owned_and_distinct(cfg_factory) -> None:
    cfg = cfg_factory()
    with pytest.raises(ValueError, match="must be distinct"):
        replace(
            cfg,
            early_cut_result_table=cfg.signal_result_table,
        ).validate()
    with pytest.raises(ValueError, match="must be distinct"):
        replace(
            cfg,
            early_cut_result_table=f"public.{cfg.signal_result_table}",
        ).validate()
    with pytest.raises(ValueError, match="EARLY_CUT_RESULT_TABLE must start"):
        replace(cfg, early_cut_result_table="foreign_early_cut").validate()
    with pytest.raises(ValueError, match="lowercase"):
        replace(
            cfg,
            early_cut_result_table=("stock_analyser_filter_research_EARLY_cut_results"),
        ).validate()


def test_v2_from_env_reads_holdout_early_cut_and_research_controls(
    monkeypatch,
) -> None:
    values = {
        "SIGNAL_START_DATE": "2017-01-01",
        "DISCOVERY_END_DATE": "2022-06-30",
        "VALIDATION_END_DATE": "2024-06-30",
        "HOLDOUT_CUTOFF_DATE": "2026-07-20",
        "EARLY_CUT_RESULT_TABLE": (
            "research.stock_analyser_filter_research_early_cut_v2"
        ),
        "MAX_CONDITIONS_PER_OBJECTIVE": "1",
        "RULE_SEARCH_BEAM_WIDTH": "7",
        "WALK_FORWARD_FIRST_YEAR": "2020",
        "MIN_WALK_FORWARD_FOLDS": "4",
        "MIN_MATCHED_LABEL_COVERAGE_PCT": "0.93",
        "MIN_SELECTION_SCORE_IMPROVEMENT": "0.02",
        "PERMUTATION_TRIAL_COUNT": "99",
        "MAX_STAT_PERMUTATION_P_VALUE": "0.04",
        "PERMUTATION_RANDOM_SEED": "31415",
        "MIN_HOLDOUT_SAMPLE_COUNT": "321",
        "WORKER_IDENTITY_BATCH_SIZE": "11",
        "DB_COPY_BATCH_SIZE": "1234",
        "MARKET_METRICS_TABLE": "research.stock_core_market_metrics_daily",
        "SECURITY_MASTER_CURRENT_TABLE": (
            "research.stock_core_security_master_current"
        ),
        "TAXONOMY_BACKCAST_INDUSTRY_MIN_MEMBERS": "30",
        "TAXONOMY_BACKCAST_CATEGORY_MIN_MEMBERS": "12",
        "TAXONOMY_BACKCAST_SUBCATEGORY_MIN_MEMBERS": "6",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    cfg = Config.from_env()

    assert cfg.signal_start_date == date(2017, 1, 1)
    assert cfg.discovery_end_date == date(2022, 6, 30)
    assert cfg.validation_end_date == date(2024, 6, 30)
    assert cfg.holdout_cutoff_date == date(2026, 7, 20)
    assert cfg.early_cut_result_table == values["EARLY_CUT_RESULT_TABLE"]
    assert cfg.max_conditions_per_objective == 1
    assert cfg.rule_search_beam_width == 7
    assert cfg.walk_forward_first_year == 2020
    assert cfg.min_walk_forward_folds == 4
    assert cfg.min_fold_protected_retention_pct == pytest.approx(0.90)
    assert cfg.min_protected_retention_pct == pytest.approx(0.92)
    assert cfg.min_matched_label_coverage_pct == pytest.approx(0.93)
    assert cfg.min_selection_score_improvement == pytest.approx(0.02)
    assert cfg.permutation_trial_count == 99
    assert cfg.max_stat_permutation_p_value == pytest.approx(0.04)
    assert cfg.permutation_random_seed == 31415
    assert cfg.min_holdout_sample_count == 321
    assert cfg.worker_identity_batch_size == 11
    assert cfg.db_copy_batch_size == 1234
    assert cfg.market_metrics_table == values["MARKET_METRICS_TABLE"]
    assert cfg.security_master_current_table == values[
        "SECURITY_MASTER_CURRENT_TABLE"
    ]
    assert cfg.taxonomy_backcast_industry_min_members == 30
    assert cfg.taxonomy_backcast_category_min_members == 12
    assert cfg.taxonomy_backcast_subcategory_min_members == 6


def test_v2_from_env_rejects_invalid_holdout_date(monkeypatch) -> None:
    monkeypatch.setenv("HOLDOUT_CUTOFF_DATE", "20-07-2026")

    with pytest.raises(ValueError, match="HOLDOUT_CUTOFF_DATE must use YYYY-MM-DD"):
        Config.from_env()
