from __future__ import annotations

from datetime import date

import pytest

from stock_analyser_filter_research.config import Config


@pytest.fixture
def cfg_factory():
    def factory(**overrides: object) -> Config:
        values: dict[str, object] = {
            "pg_host": "timescaledb",
            "pg_port": 5432,
            "pg_database": "postgres",
            "pg_user": "research",
            "pg_password": "research",
            "pg_app_name": "stock_analyser_filter_research_tests",
            "db_connect_timeout_seconds": 5,
            "db_statement_timeout_ms": 0,
            "source_table": "stock_analyser_trend_template_daily",
            "signal_result_table": ("stock_analyser_filter_research_signal_results"),
            "early_cut_result_table": (
                "stock_analyser_filter_research_early_cut_results"
            ),
            "rule_result_table": ("stock_analyser_filter_research_rule_results"),
            "log_level": "INFO",
            "signal_start_date": date(2016, 1, 1),
            "signal_end_date": None,
            "discovery_end_date": date(2022, 12, 31),
            "validation_end_date": date(2024, 12, 31),
            "holdout_cutoff_date": date(2026, 7, 20),
            "weak_5d_max_gain_pct": 2.0,
            "strong_5d_min_gain_pct": 5.0,
            "deep_loss_5d_max_loss_pct": -5.0,
            "quantile_count": 10,
            "max_conditions_per_objective": 2,
            "rule_search_beam_width": 10,
            "walk_forward_first_year": 2020,
            "min_walk_forward_folds": 4,
            "min_stable_fold_fraction": 0.75,
            "min_fold_objective_lift": 1.0,
            "min_fold_sample_count": 20,
            "min_fold_objective_count": 5,
            "min_fold_protected_retention_pct": 0.85,
            "max_fold_match_pct": 0.40,
            "min_candidate_match_pct": 0.01,
            "max_candidate_match_pct": 0.35,
            "min_protected_retention_pct": 0.90,
            "min_matched_label_coverage_pct": 0.90,
            "min_objective_lift": 1.05,
            "min_selection_capture_improvement": 0.01,
            "min_holdout_sample_count": 50,
            "max_workers": 1,
            "worker_identity_batch_size": 2,
            "db_fetch_batch_size": 100,
            "db_copy_batch_size": 100,
        }
        values.update(overrides)
        cfg = Config(**values)
        cfg.validate()
        return cfg

    return factory
