from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from psycopg2 import extensions

from stock_analyser_filter_research import db
from stock_analyser_filter_research.contracts import (
    EARLY_CUT_COLUMNS,
    EARNINGS_EVENT_SOURCE_COLUMNS,
    FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS,
    MARKET_METRIC_SOURCE_COLUMNS,
    QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS,
    RULE_COLUMNS,
    SIGNAL_BOOLEAN_COLUMNS,
    SIGNAL_COLUMNS,
    SIGNAL_INTEGER_COLUMNS,
    WORLD_MARKET_SOURCE_COLUMNS,
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


class _StreamingCursor(_Cursor):
    def __init__(self, connection) -> None:
        super().__init__(connection)
        self.itersize = None

    def fetchmany(self, size):
        if self.connection.returned:
            return []
        self.connection.returned = True
        return self.connection.rows


class _StreamingConnection(_IdleConnection):
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        super().__init__()
        self.rows = rows
        self.returned = False
        self.cursor_names: list[str | None] = []

    def cursor(self, name=None):
        self.cursor_names.append(name)
        return _StreamingCursor(self)


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
    positions = np.arange(row_count)
    data: dict[str, object] = {}
    for column in SIGNAL_COLUMNS:
        udt_name, nullable, _, _ = db.SIGNAL_COLUMN_CONTRACTS[column]
        if udt_name == "date":
            data[column] = pd.Series(
                pd.Timestamp("2020-01-02"), index=pd.RangeIndex(row_count)
            )
        elif udt_name == "text":
            values = np.full(row_count, f"value_{column}", dtype=object)
            if nullable:
                values[positions % 23 == 0] = None
            data[column] = pd.Series(values, dtype="string")
        elif udt_name in {"int2", "int4", "int8"}:
            values = pd.array(positions % 97 + 1, dtype="Int64")
            if nullable:
                values[positions % 29 == 0] = pd.NA
            data[column] = values
        elif udt_name == "bool":
            values = pd.array(positions % 2 == 0, dtype="boolean")
            if nullable:
                values[positions % 31 == 0] = pd.NA
            data[column] = values
        else:
            values = positions.astype(float) / 10.0
            values[positions % 37 == 0] = np.nan
            data[column] = values
    frame = pd.DataFrame(data, index=pd.RangeIndex(row_count))
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
    assert set(db.FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMN_CONTRACTS) == set(
        FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS
    )
    assert set(db.QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMN_CONTRACTS) == set(
        QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS
    )
    assert set(db.MARKET_METRIC_SOURCE_COLUMN_CONTRACTS) == set(
        MARKET_METRIC_SOURCE_COLUMNS
    )
    assert db.MARKET_METRIC_SOURCE_COLUMN_CONTRACTS["market_cap"] == (
        "int8",
        True,
        None,
        None,
    )
    assert db.FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMN_CONTRACTS[
        "sec_operating_margin_ttm"
    ] == ("numeric", True, 18, 6)
    assert db.QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMN_CONTRACTS[
        "quarterly_revenue"
    ] == ("numeric", True, 28, 2)
    assert db.QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMN_CONTRACTS[
        "quarterly_operating_margin"
    ] == ("numeric", True, 18, 8)
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


def test_fundamental_loaders_are_identity_bounded_and_normalized(cfg_factory) -> None:
    cfg = cfg_factory()
    identity = ("ABC", "NYSE", 123)

    snapshot_values = {
        column: None for column in FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS
    }
    snapshot_values.update(
        {
            "symbol": "ABC",
            "exchange": "NYSE",
            "cik": 123,
            "period_end_date": date(2025, 12, 31),
            "sec_fundamental_currency": "usd",
            "sec_latest_period_end_date": date(2025, 12, 31),
            "sec_data_available_at": pd.Timestamp("2026-02-01T12:00:00Z"),
            "sec_revenue_ttm": 1000,
            "sec_operating_margin_ttm": 0.1,
        }
    )
    snapshot_connection = _StreamingConnection(
        [tuple(snapshot_values[column] for column in FUNDAMENTAL_SNAPSHOT_SOURCE_COLUMNS)]
    )
    snapshots = db.load_fundamental_snapshot_batch(
        snapshot_connection, cfg, [identity]
    )

    assert snapshots.loc[0, "sec_fundamental_currency"] == "USD"
    assert snapshots.loc[0, "sec_operating_margin_ttm"] == pytest.approx(0.1)
    snapshot_statement, snapshot_parameters = snapshot_connection.executed[0]
    assert cfg.fundamental_snapshot_table in snapshot_statement
    assert "unnest" in snapshot_statement
    assert snapshot_parameters == [["ABC"], ["NYSE"], [123]]
    assert snapshot_connection.cursor_names[0].startswith("safr_fund_snapshot_")

    event_values = {
        column: None for column in QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS
    }
    event_values.update(
        {
            "symbol": "ABC",
            "exchange": "NYSE",
            "cik": 123,
            "accession_number": "0001",
            "accepted_at": pd.Timestamp("2026-02-01T12:00:00Z"),
            "effective_date": date(2026, 2, 2),
            "fiscal_period_end_date": date(2025, 12, 31),
            "currency": "eur",
            "quarterly_revenue": 100.0,
        }
    )
    event_connection = _StreamingConnection(
        [
            tuple(
                event_values[column]
                for column in QUARTERLY_FUNDAMENTAL_EVENT_SOURCE_COLUMNS
            )
        ]
    )
    events = db.load_quarterly_fundamental_event_batch(
        event_connection, cfg, [identity]
    )

    assert events.loc[0, "currency"] == "EUR"
    assert events.loc[0, "quarterly_revenue"] == pytest.approx(100.0)
    event_statement, event_parameters = event_connection.executed[0]
    assert cfg.quarterly_fundamental_event_table in event_statement
    assert "unnest" in event_statement
    assert event_parameters == [["ABC"], ["NYSE"], [123]]
    assert event_connection.cursor_names[0].startswith("safr_fund_event_")

    earnings_values = {
        column: None for column in EARNINGS_EVENT_SOURCE_COLUMNS
    }
    earnings_values.update(
        {
            "symbol": "ABC",
            "exchange": "NYSE",
            "cik": 123,
            "earnings_date": date(2026, 2, 1),
            "announcement_ts": pd.Timestamp("2026-02-01T21:00:00Z"),
            "announcement_time_type": "after_market_close",
            "source": "sec_8k_item_2_02",
            "source_event_id": "0001-8k",
            "known_as_of_ts": pd.Timestamp("2026-02-01T21:00:00Z"),
            "is_confirmed": True,
        }
    )
    earnings_connection = _StreamingConnection(
        [
            tuple(
                earnings_values[column]
                for column in EARNINGS_EVENT_SOURCE_COLUMNS
            )
        ]
    )
    earnings = db.load_earnings_event_batch(
        earnings_connection, cfg, [identity]
    )

    assert earnings.loc[0, "source"] == "sec_8k_item_2_02"
    earnings_statement, earnings_parameters = earnings_connection.executed[0]
    assert cfg.earnings_event_table in earnings_statement
    assert "source.source = 'sec_8k_item_2_02'" in earnings_statement
    assert "source.is_confirmed = true" in earnings_statement
    assert earnings_parameters == [["ABC"], ["NYSE"], [123]]
    assert earnings_connection.cursor_names[0].startswith("safr_earnings_event_")


def test_market_metric_loader_is_exact_signal_key_bounded_and_normalized(
    cfg_factory,
) -> None:
    cfg = cfg_factory()
    signal_key = (date(2026, 7, 15), "ABC", "NYSE", 123)
    values = {
        "period_end_date": signal_key[0],
        "symbol": signal_key[1],
        "exchange": signal_key[2],
        "cik": signal_key[3],
        "market_cap": 12_345_678_901,
        "market_cap_currency": "usd",
        "shares_outstanding_staleness_days": 17,
        "adjusted_open": 100.25,
        "raw_volume": 1_250_000,
        "shares_outstanding": 123_456_789,
        "shares_outstanding_source": "sec_companyfacts",
    }
    connection = _StreamingConnection(
        [tuple(values[column] for column in MARKET_METRIC_SOURCE_COLUMNS)]
    )

    metrics = db.load_market_metrics_for_signals(connection, cfg, [signal_key])

    assert metrics.loc[0, "market_cap"] == 12_345_678_901
    assert metrics.loc[0, "market_cap_currency"] == "USD"
    assert metrics.loc[0, "shares_outstanding_staleness_days"] == 17
    statement, parameters = connection.executed[0]
    assert cfg.market_metrics_table in statement
    assert "unnest" in statement
    assert "source.period_end_date = selected.period_end_date" in statement
    assert parameters == [
        [date(2026, 7, 15)],
        ["ABC"],
        ["NYSE"],
        [123],
    ]
    assert connection.cursor_names[0].startswith("safr_market_metric_")


def test_market_metric_loader_rejects_duplicate_or_invalid_signal_keys(
    cfg_factory,
) -> None:
    cfg = cfg_factory()
    key = (date(2026, 7, 15), "ABC", "NYSE", 123)
    with pytest.raises(ValueError, match="duplicates"):
        db.load_market_metrics_for_signals(object(), cfg, [key, key])
    with pytest.raises(ValueError, match="valid date"):
        db.load_market_metrics_for_signals(
            object(), cfg, [(pd.NaT, "ABC", "NYSE", 123)]
        )


def test_global_market_loaders_are_bounded_and_point_in_time(cfg_factory) -> None:
    cfg = cfg_factory()
    breadth_connection = _RowsConnection(
        [
            (
                date(2026, 7, 15),
                0.60,
                0.55,
                0.50,
                0.12,
                0.40,
                0.10,
                0.52,
                0.25,
            )
        ]
    )

    breadth = db.load_market_breadth_daily(breadth_connection, cfg)

    assert breadth.loc[0, "market_breadth_above_ma50_ratio"] == pytest.approx(0.60)
    breadth_statement, breadth_parameters = breadth_connection.executed[0]
    assert cfg.source_table in breadth_statement
    assert "period_end_date >= %s" in breadth_statement
    assert breadth_parameters == (
        cfg.signal_start_date,
        cfg.signal_end_date,
        cfg.signal_end_date,
    )

    values = {
        "source": "twelve_data",
        "series_id": "SPY",
        "observation_time": pd.Timestamp("2026-07-15T20:00:00Z"),
        "value": 625.5,
        "available_at": pd.Timestamp("2026-07-15T20:30:00Z"),
        "asof_known_at": pd.Timestamp("2026-07-15T20:30:00Z"),
        "is_revision_prone": False,
        "is_final": True,
        "source_local_date": date(2026, 7, 15),
    }
    world_connection = _StreamingConnection(
        [tuple(values[column] for column in WORLD_MARKET_SOURCE_COLUMNS)]
    )

    world = db.load_world_market_observations(world_connection, cfg)

    assert world.loc[0, "series_id"] == "SPY"
    assert world.loc[0, "value"] == pytest.approx(625.5)
    statement, parameters = world_connection.executed[0]
    assert cfg.world_market_observation_table in statement
    assert "source.is_revision_prone = false" in statement
    assert "source.is_final = true" in statement
    assert len(parameters[0]) == len(parameters[1]) == 10
    assert parameters[2:] == (
        cfg.signal_start_date,
        cfg.signal_end_date,
        cfg.signal_end_date,
    )
    assert world_connection.cursor_names[0].startswith("safr_world_market_")


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
