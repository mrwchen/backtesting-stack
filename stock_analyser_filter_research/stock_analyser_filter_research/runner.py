from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import gc
import logging
import multiprocessing
import time

import pandas as pd

from . import db
from .computation import calculate_signal_batch, empty_signal_frame
from .config import Config
from .logging_utils import configure_logging
from .research import run_research


log = logging.getLogger(__name__)
StockIdentity = tuple[str, str, int]
IdentityWork = tuple[str, str, int, int]


@dataclass(frozen=True)
class WorkerTask:
    cfg: Config
    worker_number: int
    identities: tuple[StockIdentity, ...]
    trading_dates: pd.DatetimeIndex
    snapshot_id: str


@dataclass
class WorkerResult:
    signals: pd.DataFrame
    loaded_rows: int
    identity_count: int
    elapsed_seconds: float


def balance_identity_work(
    identity_work: list[IdentityWork], max_workers: int
) -> tuple[tuple[StockIdentity, ...], ...]:
    """Greedily balance disjoint stock identities by their source row counts."""
    if max_workers < 1:
        raise ValueError("max_workers must be >= 1")
    if not identity_work:
        return ()

    partition_count = min(max_workers, len(identity_work))
    partitions: list[list[StockIdentity]] = [
        [] for _ in range(partition_count)
    ]
    partition_loads = [0] * partition_count
    normalized = [
        ((str(symbol), str(exchange), int(cik)), int(row_count))
        for symbol, exchange, cik, row_count in identity_work
    ]
    for identity, row_count in sorted(
        normalized, key=lambda item: (-item[1], item[0])
    ):
        if row_count < 1:
            raise ValueError("identity row counts must be positive")
        partition_index = min(
            range(partition_count),
            key=lambda index: (partition_loads[index], index),
        )
        partitions[partition_index].append(identity)
        partition_loads[partition_index] += row_count
    return tuple(tuple(partition) for partition in partitions)


def _identity_batches(
    identities: tuple[StockIdentity, ...], batch_size: int
) -> tuple[tuple[StockIdentity, ...], ...]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    return tuple(
        identities[start : start + batch_size]
        for start in range(0, len(identities), batch_size)
    )


def _calculate_worker(task: WorkerTask) -> WorkerResult:
    configure_logging(task.cfg.log_level)
    started = time.monotonic()
    frames: list[pd.DataFrame] = []
    loaded_rows = 0
    app_suffix = f"worker_{task.worker_number:02d}"
    with db.connect(task.cfg, app_suffix=app_suffix) as connection:
        db.import_snapshot(connection, task.snapshot_id)
        for identities in _identity_batches(
            task.identities, task.cfg.worker_identity_batch_size
        ):
            source = db.load_source_batch(connection, task.cfg, identities)
            loaded_rows += len(source)
            signals = calculate_signal_batch(
                source, task.trading_dates, task.cfg
            )
            if not signals.empty:
                frames.append(signals)
            del source, signals
            gc.collect()
        # An imported snapshot is read-only. Rollback closes it without an
        # unnecessary commit and leaves no transaction behind on close.
        connection.rollback()

    result = (
        pd.concat(frames, ignore_index=True, copy=False)
        if frames
        else empty_signal_frame()
    )
    elapsed = time.monotonic() - started
    log.info(
        "Worker %d processed %d identities and %d source rows into %d signals in %.1f seconds",
        task.worker_number,
        len(task.identities),
        loaded_rows,
        len(result),
        elapsed,
    )
    return WorkerResult(
        signals=result,
        loaded_rows=loaded_rows,
        identity_count=len(task.identities),
        elapsed_seconds=elapsed,
    )


def _execute_workers(tasks: list[WorkerTask]) -> list[WorkerResult]:
    if len(tasks) == 1:
        return [_calculate_worker(tasks[0])]
    with ProcessPoolExecutor(
        max_workers=len(tasks),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        return list(executor.map(_calculate_worker, tasks))


def run(cfg: Config) -> tuple[int, int]:
    started = time.monotonic()
    with db.connect(cfg) as connection:
        try:
            with db.advisory_lock(connection):
                # The session-level lock survives this transaction boundary.
                # Exporting a snapshot must be the first operation in the next
                # transaction so all processes can import one consistent view.
                connection.commit()
                snapshot_id = db.begin_exported_snapshot(connection)
                db.validate_schema(connection, cfg)
                db.assert_targets_empty(connection, cfg)
                trading_dates = db.load_trading_dates(connection, cfg)
                identity_work = db.load_identity_work(connection, cfg)

                if trading_dates.empty or not identity_work:
                    connection.rollback()
                    log.info("No source rows available; no results written")
                    return (0, 0)

                partitions = balance_identity_work(
                    identity_work, cfg.max_workers
                )
                tasks = [
                    WorkerTask(
                        cfg=cfg,
                        worker_number=index + 1,
                        identities=partition,
                        trading_dates=trading_dates,
                        snapshot_id=snapshot_id,
                    )
                    for index, partition in enumerate(partitions)
                ]
                log.info(
                    "Discovered %d global sessions and %d stock identities; processing with %d worker processes",
                    len(trading_dates),
                    len(identity_work),
                    len(tasks),
                )
                worker_results = _execute_workers(tasks)

                expected_rows = sum(item[3] for item in identity_work)
                loaded_rows = sum(item.loaded_rows for item in worker_results)
                if loaded_rows != expected_rows:
                    raise RuntimeError(
                        "source row count changed inside the shared snapshot: "
                        f"expected {expected_rows}, loaded {loaded_rows}"
                    )
                # Workers have finished importing/reading the snapshot, so the
                # exporter may end its read-only transaction before research and
                # the atomic write transaction start.
                connection.commit()

                frames = [item.signals for item in worker_results]
                signals = (
                    pd.concat(frames, ignore_index=True, copy=False)
                    if frames
                    else empty_signal_frame()
                )
                del frames, worker_results
                gc.collect()

                research_started = time.monotonic()
                result = run_research(signals, cfg)
                research_seconds = time.monotonic() - research_started
                selected_text = (
                    " OR ".join(condition.text for condition in result.selected.final)
                    or "none"
                )
                log.info(
                    "Research evaluated %d signals and selected %d conditions in %.1f seconds: %s",
                    len(result.signals),
                    len(result.selected.final),
                    research_seconds,
                    selected_text,
                )
                signal_count, rule_count = db.write_results_atomic(
                    connection, cfg, result.signals, result.rules
                )
                log.info(
                    "Atomically stored %d signal rows and %d rule rows in %.1f seconds",
                    signal_count,
                    rule_count,
                    time.monotonic() - started,
                )
                return signal_count, rule_count
        finally:
            # Keep advisory-lock cleanup usable even after a failed statement.
            connection.rollback()


def main() -> None:
    try:
        cfg = Config.from_env()
        configure_logging(cfg.log_level)
        run(cfg)
    except Exception:
        # Config errors occur before logging setup; basicConfig is harmless when
        # logging is already configured and preserves the required format.
        if not logging.getLogger().handlers:
            configure_logging("INFO")
        log.exception("Stock analyser filter research failed")
        raise SystemExit(1) from None

