from pathlib import Path

import pandas as pd
import pytest

from src import db


class _Cursor:
    def __init__(self):
        self.copied_sql = None
        self.copied_csv = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def copy_expert(self, sql, buf):
        self.copied_sql = sql
        self.copied_csv = buf.read()


class _Connection:
    def __init__(self):
        self.cursor_obj = _Cursor()
        self.committed = False

    def cursor(self):
        return self.cursor_obj

    def commit(self):
        self.committed = True


def test_copy_df_writes_infinite_numbers_as_null_csv_fields():
    conn = _Connection()
    df = pd.DataFrame(
        {
            "period_end_date": ["2026-07-03", "2026-07-04"],
            "symbol": ["AAA", "BBB"],
            "revenue_yoy": [float("inf"), float("-inf")],
        }
    )

    written = db.copy_df(
        conn,
        df,
        "backtesting_minervini_screen_daily",
        ["period_end_date", "symbol", "revenue_yoy"],
    )

    assert written == 2
    assert conn.committed is True
    assert conn.cursor_obj.copied_csv == "2026-07-03,AAA,\n2026-07-04,BBB,\n"


def test_breakout_events_schema_is_a_365_day_hypertable_created_only_in_init_sql():
    root = Path(__file__).parents[1]
    schema = (root / "init" / "schema.sql").read_text(encoding="utf-8")
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (root / "src").glob("*.py")
    )

    assert "CREATE TABLE IF NOT EXISTS backtesting_minervini_breakout_events" in schema
    assert "'backtesting_minervini_breakout_events'" in schema
    assert "chunk_time_interval => INTERVAL '365 days'" in schema
    assert "CREATE TABLE" not in source


def test_rebuilt_schema_has_typed_setups_and_no_legacy_vcp_columns():
    root = Path(__file__).parents[1]
    schema = (root / "init" / "schema.sql").read_text(encoding="utf-8")

    assert "setup_type          TEXT NOT NULL" in schema
    assert "setup_score         NUMERIC" in schema
    assert "model_version       TEXT NOT NULL" in schema
    assert "input_fingerprint   TEXT NOT NULL" in schema
    assert "vcp_score" not in schema
    assert "compression" not in schema.lower()
    assert "DROP TABLE IF EXISTS backtesting_minervini_stage_state" in schema


def test_runtime_schema_validation_fails_fast_on_legacy_result_tables(monkeypatch):
    rows = [
        {"table_name": table, "column_name": column}
        for table, columns in db.RESULT_SCHEMA_COLUMNS.items()
        for column in columns
        if not (table == "backtesting_minervini_setups" and column == "setup_score")
    ]
    rows.append(
        {
            "table_name": "backtesting_minervini_setups",
            "column_name": "vcp_score",
        }
    )
    monkeypatch.setattr(db, "read_df", lambda *_args, **_kwargs: pd.DataFrame(rows))

    with pytest.raises(RuntimeError, match="DROP_ALL_MINERVINI_TABLES_ON_START=true"):
        db.validate_result_schema(object())
