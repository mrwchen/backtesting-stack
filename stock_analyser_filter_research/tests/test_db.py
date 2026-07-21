from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from psycopg2 import extensions

from stock_analyser_filter_research import db
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_BOOLEAN_COLUMNS,
    SIGNAL_COLUMNS,
    SIGNAL_INTEGER_COLUMNS,
)


class _Cursor:
    def __init__(self, connection) -> None:
        self.connection = connection

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, statement, parameters=None) -> None:
        self.connection.executed.append((str(statement), parameters))


class _IdleConnection:
    def __init__(self) -> None:
        self.executed: list[tuple[str, object]] = []
        self.commits = 0
        self.rollbacks = 0

    def get_transaction_status(self) -> int:
        return extensions.TRANSACTION_STATUS_IDLE

    def cursor(self):
        return _Cursor(self)

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _RowsCursor(_Cursor):
    def fetchall(self):
        return self.connection.rows


class _RowsConnection(_IdleConnection):
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.rows = rows

    def cursor(self):
        return _RowsCursor(self)


def test_application_name_is_explicit_and_utf8_byte_bounded(cfg_factory) -> None:
    cfg = cfg_factory(pg_app_name="research_" + "ae" * 40)

    name = db._application_name(cfg, "worker_01")

    assert name.startswith("research_")
    assert len(name.encode("utf-8")) <= 63
    assert db._application_name(cfg_factory(), None) == (
        "stock_analyser_filter_research_tests"
    )


def test_copy_csv_normalizes_nullable_integer_boolean_and_infinity() -> None:
    frame = pd.DataFrame(
        {
            "integer": [1.0, np.nan],
            "flag": [True, pd.NA],
            "numeric": [np.inf, 2.5],
            "text": ["alpha", None],
        }
    )

    csv = db._copy_csv_batch(
        frame,
        ("integer", "flag", "numeric", "text"),
        integer_columns=("integer",),
        boolean_columns=("flag",),
    )

    assert csv.splitlines() == ["1,True,\\N,alpha", "\\N,\\N,2.5,\\N"]
    with pytest.raises(ValueError, match="fractions"):
        db._copy_csv_batch(
            pd.DataFrame({"integer": [1.25]}),
            ("integer",),
            integer_columns=("integer",),
        )
    with pytest.raises(ValueError, match="reserved NULL token"):
        db._copy_csv_batch(
            pd.DataFrame({"text": [r"\N"]}),
            ("text",),
        )


def test_large_mixed_signal_frame_streams_across_copy_batches() -> None:
    row_count = 5_101
    frame = pd.DataFrame(index=pd.RangeIndex(row_count))
    positions = np.arange(row_count)
    for column in SIGNAL_COLUMNS:
        udt_name, nullable, _, _ = db.SIGNAL_COLUMN_CONTRACTS[column]
        if udt_name == "date":
            frame[column] = pd.Timestamp("2020-01-02")
        elif udt_name == "text":
            values = np.full(row_count, f"value_{column}", dtype=object)
            if nullable:
                values[positions % 23 == 0] = None
            frame[column] = pd.Series(values, dtype="string")
        elif udt_name in {"int2", "int4", "int8"}:
            values = pd.array(positions % 97 + 1, dtype="Int64")
            if nullable:
                values[positions % 29 == 0] = pd.NA
            frame[column] = values
        elif udt_name == "bool":
            values = pd.array(positions % 2 == 0, dtype="boolean")
            if nullable:
                values[positions % 31 == 0] = pd.NA
            frame[column] = values
        else:
            values = positions.astype(float) / 10.0
            values[positions % 37 == 0] = np.nan
            frame[column] = values
    frame.loc[4_645, "prior_atr_14d_pct"] = np.inf
    # COPY is positional; production slices can carry arbitrary source labels.
    frame.index = pd.RangeIndex(10_000, 10_000 + row_count)

    stream = db._CopyCsvStream(
        frame,
        SIGNAL_COLUMNS,
        5_000,
        integer_columns=SIGNAL_INTEGER_COLUMNS,
        boolean_columns=SIGNAL_BOOLEAN_COLUMNS,
    )
    csv = stream.read()

    assert csv.count("\n") == row_count
    assert "inf" not in csv.lower()


def test_copy_stream_reports_failing_positional_batch(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise KeyError(np.int64(4_645))

    monkeypatch.setattr(db, "_copy_csv_batch", fail)
    stream = db._CopyCsvStream(
        pd.DataFrame({"value": range(6_000)}),
        ("value",),
        5_000,
    )

    with pytest.raises(RuntimeError, match="COPY rows 0 through 4999") as error:
        stream.read()
    assert isinstance(error.value.__cause__, KeyError)


def test_database_column_contracts_cover_every_written_column(
    monkeypatch,
) -> None:
    assert set(db.SIGNAL_COLUMN_CONTRACTS) == set(SIGNAL_COLUMNS)
    assert set(db.EARLY_CUT_COLUMN_CONTRACTS) == set(EARLY_CUT_COLUMNS)
    assert set(db.RULE_COLUMN_CONTRACTS) == {"result_id", *RULE_COLUMNS}
    assert db.EARLY_CUT_COLUMN_CONTRACTS["active_at_landmark"] == (
        "bool",
        False,
        None,
        None,
    )
    assert db.EARLY_CUT_COLUMN_CONTRACTS["prior_policy_cut_day"] == (
        "int2",
        True,
        None,
        None,
    )
    wrong = dict(db.SIGNAL_COLUMN_CONTRACTS)
    wrong["signal_date"] = ("text", False, None, None)
    monkeypatch.setattr(db, "_column_definitions", lambda connection, table: wrong)

    with pytest.raises(RuntimeError, match="signal_date"):
        db._validate_column_definitions(object(), "target", db.SIGNAL_COLUMN_CONTRACTS)


class _HypertableCursor:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.current = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def execute(self, statement, parameters=None) -> None:
        self.current = self.responses.pop(0)

    def fetchone(self):
        return self.current

    def fetchall(self):
        return self.current


class _HypertableConnection:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses

    def cursor(self):
        return _HypertableCursor(self.responses)


def test_hypertable_validation_requires_365_days_and_no_compression() -> None:
    db._validate_hypertable(
        _HypertableConnection([(False,), [("signal_date", timedelta(days=365))]]),
        "stock_analyser_filter_research_signal_results",
        "signal_date",
    )
    with pytest.raises(RuntimeError, match="must not use compression"):
        db._validate_hypertable(
            _HypertableConnection([(True,)]),
            "stock_analyser_filter_research_signal_results",
            "signal_date",
        )
    with pytest.raises(RuntimeError, match="365-day"):
        db._validate_hypertable(
            _HypertableConnection([(False,), [("signal_date", timedelta(days=30))]]),
            "stock_analyser_filter_research_signal_results",
            "signal_date",
        )

    db._validate_hypertable(
        _HypertableConnection([(False,), [("signal_date", timedelta(days=365))]]),
        "stock_analyser_filter_research_early_cut_results",
        "signal_date",
    )


def test_source_calendar_and_workload_keep_post_signal_end_rows(
    cfg_factory,
) -> None:
    cfg = cfg_factory(signal_end_date=date(2024, 6, 30))
    calendar_connection = _RowsConnection([(date(2024, 6, 28),), (date(2024, 7, 1),)])

    calendar = db.load_trading_dates(calendar_connection, cfg)

    assert calendar[-1] == pd.Timestamp("2024-07-01")
    calendar_statement, calendar_parameters = calendar_connection.executed[0]
    assert "period_end_date <=" not in calendar_statement
    assert calendar_parameters is None

    workload_connection = _RowsConnection([("ABC", "NYSE", 123, 900)])
    workload = db.load_identity_work(workload_connection, cfg)

    assert workload == [("ABC", "NYSE", 123, 900)]
    workload_statement, workload_parameters = workload_connection.executed[0]
    assert " WHERE " not in workload_statement
    assert "HAVING bool_or" in workload_statement
    assert workload_parameters == [
        cfg.signal_start_date,
        cfg.signal_end_date,
        cfg.signal_end_date,
    ]


def test_empty_rebuild_guard_checks_all_three_targets(cfg_factory, monkeypatch) -> None:
    cfg = cfg_factory()
    visited: list[str] = []

    def target_empty(connection, table_name):
        visited.append(table_name)
        return table_name != cfg.early_cut_result_table

    monkeypatch.setattr(db, "_target_empty", target_empty)

    with pytest.raises(RuntimeError, match=cfg.early_cut_result_table):
        db.assert_targets_empty(object(), cfg)
    assert visited == [
        cfg.signal_result_table,
        cfg.early_cut_result_table,
        cfg.rule_result_table,
    ]


def test_atomic_copy_commits_all_or_rolls_back_all(cfg_factory, monkeypatch) -> None:
    cfg = cfg_factory()
    signals = pd.DataFrame(columns=SIGNAL_COLUMNS)
    early_cuts = pd.DataFrame(columns=EARLY_CUT_COLUMNS)
    rules = pd.DataFrame(columns=RULE_COLUMNS)
    copied: list[str] = []
    monkeypatch.setattr(db, "assert_targets_empty", lambda connection, cfg: None)

    def successful_copy(connection, table_name, *args, **kwargs):
        copied.append(table_name)

    monkeypatch.setattr(db, "_copy_frame", successful_copy)
    connection = _IdleConnection()

    assert db.write_results_atomic(connection, cfg, signals, early_cuts, rules) == (
        0,
        0,
        0,
    )
    assert copied == [
        cfg.signal_result_table,
        cfg.early_cut_result_table,
        cfg.rule_result_table,
    ]
    assert connection.commits == 1
    assert connection.rollbacks == 0

    copied.clear()

    def failing_copy(connection, table_name, *args, **kwargs):
        copied.append(table_name)
        if table_name == cfg.rule_result_table:
            raise RuntimeError("third COPY failed")

    monkeypatch.setattr(db, "_copy_frame", failing_copy)
    connection = _IdleConnection()
    with pytest.raises(RuntimeError, match="third COPY"):
        db.write_results_atomic(connection, cfg, signals, early_cuts, rules)
    assert copied == [
        cfg.signal_result_table,
        cfg.early_cut_result_table,
        cfg.rule_result_table,
    ]
    assert connection.commits == 0
    assert connection.rollbacks == 1


def test_atomic_copy_rejects_duplicate_early_cut_primary_keys(
    cfg_factory,
) -> None:
    cfg = cfg_factory()
    signals = pd.DataFrame(columns=SIGNAL_COLUMNS)
    rules = pd.DataFrame(columns=RULE_COLUMNS)
    early_cuts = pd.DataFrame(
        [
            {
                "signal_date": pd.Timestamp("2024-01-02"),
                "symbol": "ABC",
                "exchange": "NYSE",
                "cik": 1,
                "landmark_day": 1,
            },
            {
                "signal_date": pd.Timestamp("2024-01-02"),
                "symbol": "ABC",
                "exchange": "NYSE",
                "cik": 1,
                "landmark_day": 1,
            },
        ],
        columns=EARLY_CUT_COLUMNS,
    )

    with pytest.raises(ValueError, match="early-cut.*duplicate"):
        db.write_results_atomic(_IdleConnection(), cfg, signals, early_cuts, rules)
