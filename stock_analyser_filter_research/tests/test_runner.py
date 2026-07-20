from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pandas as pd

from stock_analyser_filter_research.computation import empty_signal_frame
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
    monkeypatch.setattr(
        runner,
        "calculate_signal_batch",
        lambda source, dates, cfg: empty_signal_frame(),
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
    assert [event[0] for event in events[2:]] == ["batch", "batch"]
    assert [len(event[1]) for event in events[2:]] == [2, 1]
    assert result.loaded_rows == 3
    assert result.identity_count == 3
    assert result.signals.empty


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
        lambda connection, cfg: events.append("work")
        or [("A", "NYSE", 1, 2)],
    )
    worker_result = runner.WorkerResult(
        signals=empty_signal_frame(),
        loaded_rows=2,
        identity_count=1,
        elapsed_seconds=0.1,
    )

    def fake_workers(tasks):
        events.append("workers")
        assert tasks[0].snapshot_id == "snapshot-1"
        return [worker_result]

    monkeypatch.setattr(runner, "_execute_workers", fake_workers)
    selected = SimpleNamespace(final=())
    research_result = SimpleNamespace(
        signals=pd.DataFrame(index=range(2)),
        rules=pd.DataFrame(index=range(3)),
        selected=selected,
    )
    monkeypatch.setattr(
        runner,
        "run_research",
        lambda signals, cfg: events.append("research") or research_result,
    )
    monkeypatch.setattr(
        runner.db,
        "write_results_atomic",
        lambda connection, cfg, signals, rules: events.append("write") or (2, 3),
    )

    assert runner.run(cfg) == (2, 3)
    assert events == [
        "connect",
        "lock",
        "commit",
        "export",
        "schema",
        "empty",
        "dates",
        "work",
        "workers",
        "commit",
        "research",
        "write",
        "unlock",
        "rollback",
    ]

