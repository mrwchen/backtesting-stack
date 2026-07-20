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
            "signal_result_table": (
                "stock_analyser_filter_research_signal_results"
            ),
            "rule_result_table": (
                "stock_analyser_filter_research_rule_results"
            ),
            "log_level": "INFO",
            "signal_start_date": date(2020, 1, 1),
            "signal_end_date": None,
            "discovery_end_date": date(2022, 12, 31),
            "validation_end_date": date(2024, 12, 31),
            "weak_5d_max_gain_pct": 2.0,
            "strong_5d_min_gain_pct": 5.0,
            "deep_loss_5d_max_loss_pct": -5.0,
            "quantile_count": 10,
            "max_rule_conditions": 3,
            "min_candidate_match_pct": 0.01,
            "max_candidate_match_pct": 0.35,
            "min_strong_retention_pct": 0.90,
            "min_matched_label_coverage_pct": 0.90,
            "min_bad_lift": 1.05,
            "min_selection_score_improvement": 0.01,
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
