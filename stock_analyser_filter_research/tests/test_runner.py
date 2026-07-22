from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_analyser_filter_research.computation import (
    CalculationBatchResult,
    empty_early_cut_frame,
    empty_signal_frame,
)
from stock_analyser_filter_research.contracts import MARKET_METRIC_SOURCE_COLUMNS
from stock_analyser_filter_research import runner


def test_identity_balancing_is_disjoint_deterministic_and_weight_aware() -> None:
    work = [
        ("A", "NYSE", 1, 100),
        ("B", "NYSE", 2, 90),
        ("C", "NASDAQ", 3, 20),
        ("D", "NASDAQ", 4, 10),
    ]

    partitions = runner.balance_identity_work(work, 2)
    flattened = [identity for partition in partitions for identity in partition]
    weights = {(s, e, c): rows for s, e, c, rows in work}
    loads = [sum(weights[identity] for identity in part) for part in partitions]

    assert sorted(flattened) == sorted(weights)
    assert len(flattened) == len(set(flattened))
    assert max(loads) - min(loads) <= 20
    assert partitions == runner.balance_identity_work(list(reversed(work)), 2)


class _WorkerConnection:
    def rollback(self) -> None:
        return None


def test_worker_imports_shared_snapshot_and_reads_bounded_batches(
    cfg_factory, monkeypatch
) -> None:
    cfg = cfg_factory(worker_identity_batch_size=2)
    events: list[object] = []

    @contextmanager
    def fake_connect(cfg, app_suffix=None):
        events.append(("connect", app_suffix))
        yield _WorkerConnection()

    monkeypatch.setattr(runner.db, "connect", fake_connect)
    monkeypatch.setattr(
        runner.db,
        "import_snapshot",
        lambda connection, snapshot: events.append(("snapshot", snapshot)),
    )

    def fake_load(connection, cfg, identities):
        events.append(("batch", tuple(identities)))
        return pd.DataFrame(index=range(len(identities)))

    monkeypatch.setattr(runner.db, "load_source_batch", fake_load)

    def fake_fundamental_snapshots(connection, cfg, identities):
        events.append(("fundamental_snapshots", tuple(identities)))
        return pd.DataFrame()

    def fake_quarterly_events(connection, cfg, identities):
        events.append(("quarterly_events", tuple(identities)))
        return pd.DataFrame()

    monkeypatch.setattr(
        runner.db,
        "load_fundamental_snapshot_batch",
        fake_fundamental_snapshots,
    )
    monkeypatch.setattr(
        runner.db,
        "load_quarterly_fundamental_event_batch",
        fake_quarterly_events,
    )
    monkeypatch.setattr(
        runner.db,
        "load_earnings_event_batch",
        lambda connection, cfg, identities: events.append(
            ("earnings_events", tuple(identities))
        )
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner,
        "calculate_signal_batch",
        lambda source, dates, cfg, **kwargs: CalculationBatchResult(
            signals=empty_signal_frame(),
            early_cut=empty_early_cut_frame(),
        ),
    )
    identities = (
        ("A", "NYSE", 1),
        ("B", "NYSE", 2),
        ("C", "NASDAQ", 3),
    )
    task = runner.WorkerTask(
        cfg=cfg,
        worker_number=2,
        identities=identities,
        trading_dates=pd.date_range("2020-01-01", periods=3),
        snapshot_id="snapshot-1",
    )

    result = runner._calculate_worker(task)

    assert events[:2] == [("connect", "worker_02"), ("snapshot", "snapshot-1")]
    assert [event[0] for event in events[2:]] == [
        "batch",
        "fundamental_snapshots",
        "quarterly_events",
        "earnings_events",
        "batch",
        "fundamental_snapshots",
        "quarterly_events",
        "earnings_events",
    ]
    assert [len(event[1]) for event in events[2:]] == [2, 2, 2, 2, 1, 1, 1, 1]
    assert result.loaded_rows == 3
    assert result.identity_count == 3
    assert result.signals.empty
    assert result.early_cuts.empty


def test_worker_preserves_all_six_landmarks_per_signal_across_batches(
    cfg_factory, monkeypatch
) -> None:
    cfg = cfg_factory(worker_identity_batch_size=2)

    @contextmanager
    def fake_connect(cfg, app_suffix=None):
        yield _WorkerConnection()

    monkeypatch.setattr(runner.db, "connect", fake_connect)
    monkeypatch.setattr(runner.db, "import_snapshot", lambda *args: None)
    monkeypatch.setattr(
        runner.db,
        "load_source_batch",
        lambda connection, cfg, identities: pd.DataFrame(
            {"identity": list(identities)}
        ),
    )
    monkeypatch.setattr(
        runner.db,
        "load_fundamental_snapshot_batch",
        lambda connection, cfg, identities: pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_quarterly_fundamental_event_batch",
        lambda connection, cfg, identities: pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_earnings_event_batch",
        lambda connection, cfg, identities: pd.DataFrame(),
    )

    loaded_keys = []

    def fake_market_metrics(connection, cfg, signal_keys):
        loaded_keys.extend(signal_keys)
        return pd.DataFrame(columns=MARKET_METRIC_SOURCE_COLUMNS)

    monkeypatch.setattr(
        runner.db,
        "load_market_metrics_for_signals",
        fake_market_metrics,
    )

    def fake_calculate(source, dates, cfg, **kwargs):
        signal_count = len(source)
        signals = empty_signal_frame().reindex(range(signal_count))
        identities = source["identity"].tolist()
        signals["signal_date"] = pd.Timestamp("2020-01-02")
        signals["symbol"] = [identity[0] for identity in identities]
        signals["exchange"] = [identity[1] for identity in identities]
        signals["cik"] = [identity[2] for identity in identities]
        early_cuts = empty_early_cut_frame().reindex(range(6 * signal_count))
        early_cuts["landmark_day"] = [1, 2, 3, 5, 20, 30] * signal_count
        return CalculationBatchResult(signals=signals, early_cut=early_cuts)

    monkeypatch.setattr(runner, "calculate_signal_batch", fake_calculate)
    task = runner.WorkerTask(
        cfg=cfg,
        worker_number=1,
        identities=(
            ("A", "NYSE", 1),
            ("B", "NYSE", 2),
            ("C", "NASDAQ", 3),
        ),
        trading_dates=pd.date_range("2020-01-01", periods=3),
        snapshot_id="snapshot-1",
    )

    result = runner._calculate_worker(task)

    assert result.loaded_rows == 3
    assert len(result.signals) == 3
    assert sorted(loaded_keys) == [
        (pd.Timestamp("2020-01-02").date(), "A", "NYSE", 1),
        (pd.Timestamp("2020-01-02").date(), "B", "NYSE", 2),
        (pd.Timestamp("2020-01-02").date(), "C", "NASDAQ", 3),
    ]
    assert len(result.early_cuts) == 18
    assert result.early_cuts["landmark_day"].tolist() == [
        1,
        2,
        3,
        5,
        20,
        30,
        1,
        2,
        3,
        5,
        20,
        30,
        1,
        2,
        3,
        5,
        20,
        30,
    ]


class _MainConnection:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def commit(self) -> None:
        self.events.append("commit")

    def rollback(self) -> None:
        self.events.append("rollback")


def test_run_uses_one_snapshot_then_one_atomic_main_process_write(
    cfg_factory, monkeypatch
) -> None:
    cfg = cfg_factory(max_workers=1)
    events: list[str] = []
    connection = _MainConnection(events)

    @contextmanager
    def fake_connect(cfg):
        events.append("connect")
        yield connection

    @contextmanager
    def fake_lock(connection):
        events.append("lock")
        yield
        events.append("unlock")

    monkeypatch.setattr(runner.db, "connect", fake_connect)
    monkeypatch.setattr(runner.db, "advisory_lock", fake_lock)
    monkeypatch.setattr(
        runner.db,
        "begin_exported_snapshot",
        lambda connection: events.append("export") or "snapshot-1",
    )
    monkeypatch.setattr(
        runner.db,
        "validate_schema",
        lambda connection, cfg: events.append("schema"),
    )
    monkeypatch.setattr(
        runner.db,
        "assert_targets_empty",
        lambda connection, cfg: events.append("empty"),
    )
    monkeypatch.setattr(
        runner.db,
        "load_trading_dates",
        lambda connection, cfg: events.append("dates")
        or pd.date_range("2020-01-01", periods=3),
    )
    monkeypatch.setattr(
        runner.db,
        "load_identity_work",
        lambda connection, cfg: events.append("work") or [("A", "NYSE", 1, 2)],
    )
    monkeypatch.setattr(
        runner.db,
        "load_market_breadth_daily",
        lambda connection, cfg: events.append("breadth") or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_world_market_observations",
        lambda connection, cfg: events.append("world") or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner,
        "build_global_market_context",
        lambda dates, breadth, world: events.append("market_context")
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_current_taxonomy_backcast",
        lambda connection, cfg: events.append("taxonomy") or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_current_taxonomy_backcast_group_context",
        lambda connection, cfg: events.append("taxonomy_groups")
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner,
        "build_current_taxonomy_backcast_context",
        lambda raw, dates, cfg: events.append("taxonomy_context")
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner.db,
        "load_current_taxonomy_backcast_member_ranks",
        lambda connection, cfg: events.append("taxonomy_ranks")
        or pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner,
        "enrich_global_features",
        lambda signals, early_cuts, context: SimpleNamespace(
            signals=signals, early_cut=early_cuts
        ),
    )
    monkeypatch.setattr(
        runner,
        "enrich_current_taxonomy_backcast_features",
        lambda signals, taxonomy, context, ranks: events.append(
            "taxonomy_enrich"
        )
        or signals,
    )
    worker_result = runner.WorkerResult(
        signals=empty_signal_frame(),
        early_cuts=empty_early_cut_frame(),
        loaded_rows=2,
        identity_count=1,
        elapsed_seconds=0.1,
    )

    def fake_workers(tasks):
        events.append("workers")
        assert tasks[0].snapshot_id == "snapshot-1"
        return [worker_result]

    monkeypatch.setattr(runner, "_execute_workers", fake_workers)
    research_result = SimpleNamespace(
        signals=pd.DataFrame(index=range(2)),
        early_cuts=pd.DataFrame(index=range(6)),
        rules=pd.DataFrame(index=range(3)),
        selected_condition_count=0,
        selected_text="",
    )
    monkeypatch.setattr(
        runner,
        "run_research",
        lambda signals, early_cuts, trading_dates, cfg: events.append("research")
        or research_result,
    )

    def fake_write(connection, observed_cfg, signals, early_cuts, rules):
        events.append("write")
        assert observed_cfg is cfg
        assert signals is research_result.signals
        assert early_cuts is research_result.early_cuts
        assert rules is research_result.rules
        return (2, 6, 3)

    monkeypatch.setattr(runner.db, "write_results_atomic", fake_write)

    assert runner.run(cfg) == (2, 6, 3)
    assert events == [
        "connect",
        "lock",
        "commit",
        "export",
        "schema",
        "empty",
        "dates",
        "work",
        "breadth",
        "world",
        "market_context",
        "workers",
        "taxonomy",
        "taxonomy_groups",
        "taxonomy_context",
        "taxonomy_ranks",
        "commit",
        "taxonomy_enrich",
        "research",
        "write",
        "unlock",
        "rollback",
    ]


@pytest.mark.parametrize(
    ("trading_dates", "identity_work"),
    [
        (pd.DatetimeIndex([]), [("A", "NYSE", 1, 2)]),
        (pd.date_range("2020-01-01", periods=3), []),
    ],
)
def test_run_with_no_source_work_returns_empty_without_research_or_write(
    cfg_factory,
    monkeypatch,
    trading_dates,
    identity_work,
) -> None:
    cfg = cfg_factory(max_workers=1)
    events: list[str] = []
    connection = _MainConnection(events)

    @contextmanager
    def fake_connect(cfg):
        events.append("connect")
        yield connection

    @contextmanager
    def fake_lock(connection):
        events.append("lock")
        yield
        events.append("unlock")

    monkeypatch.setattr(runner.db, "connect", fake_connect)
    monkeypatch.setattr(runner.db, "advisory_lock", fake_lock)
    monkeypatch.setattr(
        runner.db,
        "begin_exported_snapshot",
        lambda connection: "snapshot-1",
    )
    monkeypatch.setattr(runner.db, "validate_schema", lambda *args: None)
    monkeypatch.setattr(runner.db, "assert_targets_empty", lambda *args: None)
    monkeypatch.setattr(
        runner.db,
        "load_trading_dates",
        lambda connection, cfg: trading_dates,
    )
    monkeypatch.setattr(
        runner.db,
        "load_identity_work",
        lambda connection, cfg: identity_work,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("workers, research and writes must not run without source")

    monkeypatch.setattr(runner, "_execute_workers", forbidden)
    monkeypatch.setattr(runner, "run_research", forbidden)
    monkeypatch.setattr(runner.db, "write_results_atomic", forbidden)

    assert runner.run(cfg) == (0, 0, 0)
    assert events == [
        "connect",
        "lock",
        "commit",
        "rollback",
        "unlock",
        "rollback",
    ]


def test_run_rejects_any_landmark_count_other_than_six_per_signal(
    cfg_factory, monkeypatch
) -> None:
    cfg = cfg_factory(max_workers=1)
    connection = _MainConnection([])

    @contextmanager
    def fake_connect(cfg):
        yield connection

    @contextmanager
    def fake_lock(connection):
        yield

    monkeypatch.setattr(runner.db, "connect", fake_connect)
    monkeypatch.setattr(runner.db, "advisory_lock", fake_lock)
    monkeypatch.setattr(
        runner.db,
        "begin_exported_snapshot",
        lambda connection: "snapshot-1",
    )
    monkeypatch.setattr(runner.db, "validate_schema", lambda *args: None)
    monkeypatch.setattr(runner.db, "assert_targets_empty", lambda *args: None)
    monkeypatch.setattr(
        runner.db,
        "load_trading_dates",
        lambda *args: pd.date_range("2020-01-01", periods=3),
    )
    monkeypatch.setattr(
        runner.db,
        "load_identity_work",
        lambda *args: [("A", "NYSE", 1, 2)],
    )
    monkeypatch.setattr(
        runner.db, "load_market_breadth_daily", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        runner.db, "load_world_market_observations", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        runner, "build_global_market_context", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        runner.db, "load_current_taxonomy_backcast", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        runner.db,
        "load_current_taxonomy_backcast_group_context",
        lambda *args: pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner, "build_current_taxonomy_backcast_context", lambda *args: pd.DataFrame()
    )
    monkeypatch.setattr(
        runner.db,
        "load_current_taxonomy_backcast_member_ranks",
        lambda *args: pd.DataFrame(),
    )
    monkeypatch.setattr(
        runner,
        "enrich_global_features",
        lambda signals, early_cuts, context: SimpleNamespace(
            signals=signals, early_cut=early_cuts
        ),
    )
    monkeypatch.setattr(
        runner,
        "enrich_current_taxonomy_backcast_features",
        lambda signals, *args: signals,
    )
    worker_result = runner.WorkerResult(
        signals=empty_signal_frame().reindex(range(2)),
        early_cuts=empty_early_cut_frame().reindex(range(5)),
        loaded_rows=2,
        identity_count=1,
        elapsed_seconds=0.1,
    )
    monkeypatch.setattr(runner, "_execute_workers", lambda tasks: [worker_result])

    def forbidden(*args, **kwargs):
        pytest.fail("invalid landmark batches must not reach research or COPY")

    monkeypatch.setattr(runner, "run_research", forbidden)
    monkeypatch.setattr(runner.db, "write_results_atomic", forbidden)

    with pytest.raises(
        RuntimeError,
        match="expected 12 rows for 2 signals, received 5",
    ):
        runner.run(cfg)
