from __future__ import annotations

from dataclasses import replace
import logging
from math import nan
from pathlib import Path
import re

import pytest

from stock_analyser_filter_research.contracts import (
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
        r"BIGSERIAL|BIGINT|SMALLINT|INTEGER|NUMERIC(?:\([^)]*\))?"
        r"|BOOLEAN|TEXT|DATE"
    )
    return set(
        re.findall(
            rf"^\s{{4}}([a-z][a-z0-9_]*)\s+(?:{type_pattern})(?:\s|,)",
            create_block,
            flags=re.MULTILINE,
        )
    )


def test_init_sql_is_the_complete_runtime_contract() -> None:
    sql_text = (PROJECT_ROOT / "init" / "schema.sql").read_text(
        encoding="utf-8"
    )
    signal_table = "stock_analyser_filter_research_signal_results"
    rule_table = "stock_analyser_filter_research_rule_results"
    signal_columns = _declared_columns(
        _create_table_block(sql_text, signal_table)
    )
    rule_columns = _declared_columns(_create_table_block(sql_text, rule_table))

    assert signal_columns == set(SIGNAL_COLUMNS)
    assert rule_columns == {*RULE_COLUMNS, "result_id"}
    assert sql_text.count("CREATE TABLE IF NOT EXISTS") == 2
    assert "drop_all_stock_analyser_filter_research_tables_on_start" in sql_text
    assert "chunk_time_interval => INTERVAL '365 days'" in sql_text
    assert "create_hypertable(" in sql_text
    assert "json" not in sql_text.lower()
    lowered = sql_text.lower()
    assert "timescaledb.compress" not in lowered
    assert "add_compression_policy" not in lowered
    assert "add_columnstore_policy" not in lowered


def test_config_rejects_non_owned_target_names(cfg_factory) -> None:
    cfg = cfg_factory()
    with pytest.raises(ValueError, match="must start"):
        replace(cfg, signal_result_table="some_other_table").validate()


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
