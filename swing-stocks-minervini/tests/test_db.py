from pathlib import Path

import pandas as pd
import pytest

from src import db, persistence


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


def test_copy_df_can_participate_in_a_caller_owned_transaction():
    conn = _Connection()

    db.copy_df(
        conn,
        pd.DataFrame({"symbol": ["AAA"]}),
        "backtesting_minervini_trades",
        ["symbol"],
        commit=False,
    )

    assert conn.committed is False


def test_screen_stage_reader_returns_all_daily_rows(monkeypatch):
    captured = {}

    def fake_read_df(conn, query, params):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(persistence.db, "read_df", fake_read_df)

    persistence.read_screen_stage_output(
        object(), pd.Timestamp("2024-01-01").date(), pd.Timestamp("2024-01-31").date()
    )

    assert "WHERE screen_pass" not in captured["query"]
    assert "WHERE period_end_date BETWEEN" in captured["query"]


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


def test_breakout_events_schema_and_copy_contract_include_candidate_snapshot():
    root = Path(__file__).parents[1]
    schema = (root / "init" / "schema.sql").read_text(encoding="utf-8")
    definitions = {
        "snapshot_date": "snapshot_date           DATE NOT NULL",
        "quality_score": "quality_score           NUMERIC NOT NULL",
        "fill_probability": "fill_probability        NUMERIC NOT NULL",
        "slate_priority": "slate_priority          NUMERIC NOT NULL",
        "setup_age_sessions": "setup_age_sessions      INTEGER NOT NULL",
        "distance_to_pivot_pct": "distance_to_pivot_pct   NUMERIC",
        "quality_rank": "quality_rank            INTEGER NOT NULL",
    }

    for column, definition in definitions.items():
        assert definition in schema
        assert column in persistence.BREAKOUT_EVENT_COLUMNS
    assert definitions.keys() <= db.RESULT_SCHEMA_COLUMNS[
        "backtesting_minervini_breakout_events"
    ]
    assert "CHECK (snapshot_date < breakout_date)" in schema
    assert "CHECK (quality_score NOT IN (" in schema
    assert "CHECK (fill_probability BETWEEN 0 AND 1)" in schema
    assert "CHECK (slate_priority NOT IN (" in schema
    assert "'NaN'::NUMERIC, 'Infinity'::NUMERIC, '-Infinity'::NUMERIC" in schema
    assert "CHECK (quality_rank > 0)" in schema
    assert "'setup_class_research_only', 'price_continuity_break'" in schema
    assert "'non_positive_quality'" in schema
    assert "'trend_template_not_passed'" not in schema
    assert "'bad_fundamentals'" not in schema
    assert "JSON" not in "\n".join(
        line
        for line in schema.splitlines()
        if any(column in line for column in definitions)
    ).upper()


def test_rebuilt_schema_has_typed_setups_and_no_legacy_vcp_columns():
    root = Path(__file__).parents[1]
    schema = (root / "init" / "schema.sql").read_text(encoding="utf-8")

    assert "setup_type          TEXT NOT NULL" in schema
    assert "price_continuity_segment INTEGER NOT NULL" in schema
    assert "CHECK (price_continuity_segment > 0)" in schema
    assert "setup_score         NUMERIC" in schema
    assert "model_version       TEXT NOT NULL" in schema
    assert "input_fingerprint   TEXT NOT NULL" in schema
    assert "vcp_score" not in schema
    assert "compression" not in schema.lower()
    assert "DROP TABLE IF EXISTS backtesting_minervini_stage_state" in schema
    assert "ON ALL TABLES IN SCHEMA public" not in schema
    assert "ON ALL SEQUENCES IN SCHEMA public" not in schema


def test_v5_schema_removes_v4_breakout_ranking_columns():
    root = Path(__file__).parents[1]
    schema = (root / "init" / "schema.sql").read_text(encoding="utf-8")
    breakout_schema = schema.split(
        "CREATE TABLE IF NOT EXISTS backtesting_minervini_breakout_events", 1
    )[1].split(");", 1)[0]

    for legacy in (
        "dynamic_setup_score", "readiness_score", "context_score", "candidate_rank"
    ):
        assert legacy not in breakout_schema
        assert legacy not in persistence.BREAKOUT_EVENT_COLUMNS


def test_read_setups_does_not_freeze_detect_date_screen_context(monkeypatch):
    captured = {}

    def fake_read_df(_conn, query, params):
        captured["query"] = query
        captured["params"] = params
        return pd.DataFrame()

    monkeypatch.setattr(db, "read_df", fake_read_df)

    persistence.read_setups(object(), "2024-01-01", "2024-12-31")

    query = captured["query"].lower()
    assert "left join" not in query
    assert "backtesting_minervini_screen_daily" not in query
    assert "fundamental_score" not in query
    assert "rs_rating" not in query
    assert "price_continuity_segment" in query
    assert captured["params"] == ("2024-01-01", "2024-12-31")


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


def test_runtime_schema_validation_rejects_v4_columns_even_if_v5_columns_exist(
    monkeypatch,
):
    rows = [
        {"table_name": table, "column_name": column}
        for table, columns in db.RESULT_SCHEMA_COLUMNS.items()
        for column in columns
    ]
    rows.append(
        {
            "table_name": "backtesting_minervini_breakout_events",
            "column_name": "dynamic_setup_score",
        }
    )
    monkeypatch.setattr(db, "read_df", lambda *_args, **_kwargs: pd.DataFrame(rows))

    with pytest.raises(RuntimeError, match="dynamic_setup_score"):
        db.validate_result_schema(object())
