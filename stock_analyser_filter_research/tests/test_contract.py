from __future__ import annotations

from dataclasses import replace
import logging
from math import nan
from pathlib import Path
import re

import pytest

from stock_analyser_filter_research import db
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_COLUMNS,
)
from stock_analyser_filter_research.logging_utils import configure_logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _create_table_block(sql_text: str, table: str) -> str:
    match = re.search(
        rf"CREATE TABLE IF NOT EXISTS {re.escape(table)} \((.*?)\n\);",
        sql_text,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def _declared_columns(create_block: str) -> set[str]:
    type_pattern = (
        r"BIGSERIAL|BIGINT|SMALLINT|INTEGER|NUMERIC(?:\([^)]*\))?" r"|BOOLEAN|TEXT|DATE"
    )
    return set(
        re.findall(
            rf"^\s{{4}}([a-z][a-z0-9_]*)\s+(?:{type_pattern})(?:\s|,)",
            create_block,
            flags=re.MULTILINE,
        )
    )


def _declared_column_order(create_block: str) -> tuple[str, ...]:
    type_pattern = (
        r"BIGSERIAL|BIGINT|SMALLINT|INTEGER|NUMERIC(?:\([^)]*\))?" r"|BOOLEAN|TEXT|DATE"
    )
    return tuple(
        re.findall(
            rf"^\s{{4}}([a-z][a-z0-9_]*)\s+(?:{type_pattern})(?:\s|,)",
            create_block,
            flags=re.MULTILINE,
        )
    )


def _declared_contracts(
    create_block: str,
) -> dict[str, tuple[str, bool, int | None, int | None]]:
    type_pattern = (
        r"BIGSERIAL|BIGINT|SMALLINT|INTEGER|NUMERIC(?:\((\d+),(\d+)\))?"
        r"|BOOLEAN|TEXT|DATE"
    )
    type_names = {
        "BIGSERIAL": "int8",
        "BIGINT": "int8",
        "SMALLINT": "int2",
        "INTEGER": "int4",
        "NUMERIC": "numeric",
        "BOOLEAN": "bool",
        "TEXT": "text",
        "DATE": "date",
    }
    contracts: dict[str, tuple[str, bool, int | None, int | None]] = {}
    for match in re.finditer(
        rf"^\s{{4}}([a-z][a-z0-9_]*)\s+({type_pattern})([^\n]*)$",
        create_block,
        flags=re.MULTILINE,
    ):
        column, sql_type, precision, scale, suffix = match.groups()
        base_type = sql_type.split("(", 1)[0]
        nullable = not (
            "NOT NULL" in suffix.upper()
            or "PRIMARY KEY" in suffix.upper()
            or base_type == "BIGSERIAL"
        )
        contracts[column] = (
            type_names[base_type],
            nullable,
            int(precision) if precision is not None else None,
            int(scale) if scale is not None else None,
        )
    return contracts


def _normalized_sql(sql_text: str) -> str:
    return " ".join(sql_text.split())


def test_init_sql_is_the_complete_runtime_contract() -> None:
    sql_text = (PROJECT_ROOT / "init" / "schema.sql").read_text(encoding="utf-8")
    signal_table = "stock_analyser_filter_research_signal_results"
    early_cut_table = "stock_analyser_filter_research_early_cut_results"
    rule_table = "stock_analyser_filter_research_rule_results"
    signal_block = _create_table_block(sql_text, signal_table)
    early_cut_block = _create_table_block(sql_text, early_cut_table)
    rule_block = _create_table_block(sql_text, rule_table)
    signal_columns = _declared_columns(signal_block)
    early_cut_columns = _declared_columns(early_cut_block)
    rule_columns = _declared_columns(rule_block)

    assert signal_columns == set(SIGNAL_COLUMNS)
    assert early_cut_columns == set(EARLY_CUT_COLUMNS)
    assert rule_columns == {*RULE_COLUMNS, "result_id"}
    assert _declared_column_order(signal_block) == SIGNAL_COLUMNS
    assert _declared_column_order(early_cut_block) == EARLY_CUT_COLUMNS
    assert _declared_column_order(rule_block) == ("result_id", *RULE_COLUMNS)
    assert _declared_contracts(signal_block) == db.SIGNAL_COLUMN_CONTRACTS
    assert _declared_contracts(early_cut_block) == db.EARLY_CUT_COLUMN_CONTRACTS
    assert _declared_contracts(rule_block) == db.RULE_COLUMN_CONTRACTS
    assert sql_text.count("CREATE TABLE IF NOT EXISTS") == 3
    assert "drop_all_stock_analyser_filter_research_tables_on_start" in sql_text
    assert (
        "\\if :{?drop_all_stock_analyser_filter_research_tables_on_start}" in sql_text
    )
    assert sql_text.count("chunk_time_interval => INTERVAL '365 days'") == 2
    assert sql_text.count("create_hypertable(") == 2
    assert "PRIMARY KEY (signal_date, symbol, exchange, cik, landmark_day)" in sql_text
    assert "DROP TABLE IF EXISTS " + early_cut_table in sql_text
    assert (
        "'cut', 'hold', 'not_active', 'not_eligible', 'not_evaluable'" in sql_text
    )
    assert "active_at_landmark" in sql_text
    assert "prior_policy_cut_day BETWEEN 1 AND landmark_day - 1" in sql_text
    assert (
        "include_final = (include_weak_filter AND include_loss_first_filter)"
        in sql_text
    )
    assert (
        "include_final = (include_stagnation_filter AND include_loss_filter)"
        in sql_text
    )
    assert "objective_count + protected_count <= sample_count" in sql_text
    assert (
        "matched_objective_count + matched_protected_count <= matched_labeled_count"
        in sql_text
    )
    assert (
        "active_at_landmark AND include_final AND cut_decision = 'hold'" in sql_text
    )
    assert "threshold_fit_end_date < period_start" in sql_text
    assert "json" not in sql_text.lower()
    lowered = sql_text.lower()
    assert "timescaledb.compress" not in lowered
    assert "add_compression_policy" not in lowered
    assert "add_columnstore_policy" not in lowered

    runtime_text = (
        (PROJECT_ROOT / "stock_analyser_filter_research" / "db.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "create table" not in runtime_text
    assert "alter table" not in runtime_text
    assert "drop table" not in runtime_text


def test_early_cut_outcome_check_enforces_exact_label_states() -> None:
    sql_text = (PROJECT_ROOT / "init" / "schema.sql").read_text(encoding="utf-8")
    early_cut_block = _normalized_sql(
        _create_table_block(
            sql_text, "stock_analyser_filter_research_early_cut_results"
        )
    )

    expected_states = (
        "WHEN continuation_outcome IS NULL THEN stagnant_to_day5 IS NULL "
        "AND loss_first_to_day5 IS NULL AND strong_first_to_day5 IS NULL "
        "AND bad_to_day5 IS NULL",
        "WHEN continuation_outcome = 'same_session_ambiguous' THEN "
        "stagnant_to_day5 IS NULL AND loss_first_to_day5 IS NULL "
        "AND strong_first_to_day5 IS NULL AND bad_to_day5 IS NULL",
        "WHEN continuation_outcome = 'loss_first' THEN stagnant_to_day5 IS FALSE "
        "AND loss_first_to_day5 IS TRUE AND strong_first_to_day5 IS FALSE "
        "AND bad_to_day5 IS TRUE",
        "WHEN continuation_outcome = 'strong_first' THEN stagnant_to_day5 IS FALSE "
        "AND loss_first_to_day5 IS FALSE AND strong_first_to_day5 IS TRUE "
        "AND bad_to_day5 IS FALSE",
        "WHEN continuation_outcome = 'stagnant' THEN stagnant_to_day5 IS TRUE "
        "AND loss_first_to_day5 IS FALSE AND strong_first_to_day5 IS FALSE "
        "AND bad_to_day5 IS TRUE",
        "WHEN continuation_outcome = 'neutral' THEN stagnant_to_day5 IS FALSE "
        "AND loss_first_to_day5 IS FALSE AND strong_first_to_day5 IS FALSE "
        "AND bad_to_day5 IS FALSE",
    )
    for expected_state in expected_states:
        assert expected_state in early_cut_block
    assert "ELSE FALSE END" in early_cut_block


def test_rule_objective_check_anchors_only_sequential_bad_metric_at_day_one() -> None:
    sql_text = (PROJECT_ROOT / "init" / "schema.sql").read_text(encoding="utf-8")
    rule_block = _normalized_sql(
        _create_table_block(
            sql_text, "stock_analyser_filter_research_rule_results"
        )
    )

    assert "landmark_day BETWEEN 1 AND 3" in rule_block
    assert (
        "objective IN ('stagnant_to_day5', 'loss_first_to_day5') OR "
        "(objective = 'bad_to_day5' AND landmark_day = 1)"
    ) in rule_block


def test_config_rejects_non_owned_target_names(cfg_factory) -> None:
    cfg = cfg_factory()
    with pytest.raises(ValueError, match="must start"):
        replace(cfg, signal_result_table="some_other_table").validate()
    with pytest.raises(ValueError, match="must start"):
        replace(cfg, early_cut_result_table="some_other_table").validate()


def test_config_rejects_non_finite_research_thresholds(cfg_factory) -> None:
    cfg = cfg_factory()
    with pytest.raises(ValueError, match="must be finite"):
        replace(cfg, weak_5d_max_gain_pct=nan).validate()


def test_logging_uses_compact_utc_positional_format(capsys) -> None:
    configure_logging("INFO")
    logging.getLogger("contract_test").info("compact message")

    output = capsys.readouterr().err.strip()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z INFO \S+ \S+ compact message",
        output,
    )
    assert "ts_utc=" not in output
    assert "loglevel=" not in output
