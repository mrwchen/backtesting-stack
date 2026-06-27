"""Range analysis orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import logging
import multiprocessing as mp
import time
from uuid import UUID, uuid4

from . import config, persistence, profile
from .config import AnalysisConfig
from .data import BarData, build_mid_bars, load_ticks, ns_to_datetime
from .db import connect_with_retry
from .profile import RangeProfileArrays, RangeProfileConfig, rolling_range_profile_arrays

log = logging.getLogger(__name__)

_WORKER_BARS: BarData | None = None
_WORKER_CFG: AnalysisConfig | None = None


@dataclass(frozen=True)
class LookbackResult:
    lookback_bars: int
    min_lookback_bars: int
    arrays: RangeProfileArrays


def main() -> None:
    started = time.time()
    cfg = config.active_analysis_config()
    lookbacks = cfg.lookback_values
    analysis_id = uuid4()
    created_at = datetime.now(timezone.utc)
    profile_max = (
        str(cfg.profile_max_lookback_seconds)
        if cfg.profile_max_lookback_seconds is not None
        else "per_lookback_bar_seconds"
    )
    log.info(
        "NAS100 hit-frequency range analysis start analysis_id %s symbol %s source %s start %s end %s bar_seconds %d lookbacks %s profile_max_lookback_seconds %s processes %d",
        analysis_id,
        cfg.symbol,
        cfg.source_table,
        cfg.start_ts_utc,
        cfg.end_ts_utc,
        cfg.bar_seconds,
        ",".join(str(v) for v in lookbacks),
        profile_max,
        cfg.analysis_processes,
    )

    read_conn = connect_with_retry()
    try:
        persistence.validate_schema(read_conn, cfg)
        raw_ticks = load_ticks(read_conn, cfg)
        data_start_ts = ns_to_datetime(int(raw_ticks.tick_time_ns[0]))
        data_end_ts = ns_to_datetime(int(raw_ticks.tick_time_ns[-1]))
        ticks_loaded = len(raw_ticks)
        bars = build_mid_bars(raw_ticks, cfg)
    finally:
        read_conn.close()

    del raw_ticks
    gc.collect()
    profile.warmup()

    write_conn = None
    total_rows = 0
    try:
        if cfg.analysis_processes > 1 and len(lookbacks) > 1:
            ctx = _multiprocessing_context()
            with ctx.Pool(
                processes=min(cfg.analysis_processes, len(lookbacks)),
                initializer=_init_worker,
                initargs=(bars, cfg),
            ) as pool:
                write_conn = connect_with_retry()
                persistence.validate_schema(write_conn, cfg)
                for result in pool.imap_unordered(_compute_worker_lookback, lookbacks, chunksize=1):
                    total_rows += _persist_result(
                        write_conn,
                        cfg,
                        analysis_id,
                        created_at,
                        data_start_ts,
                        data_end_ts,
                        ticks_loaded,
                        bars,
                        result,
                    )
                    del result
                    gc.collect()
        else:
            write_conn = connect_with_retry()
            persistence.validate_schema(write_conn, cfg)
            for lookback in lookbacks:
                result = compute_lookback(bars, cfg, lookback)
                total_rows += _persist_result(
                    write_conn,
                    cfg,
                    analysis_id,
                    created_at,
                    data_start_ts,
                    data_end_ts,
                    ticks_loaded,
                    bars,
                    result,
                )
                del result
                gc.collect()
    finally:
        if write_conn is not None:
            write_conn.close()

    elapsed = time.time() - started
    log.info(
        "NAS100 hit-frequency range analysis complete analysis_id %s lookbacks %d rows_inserted %d elapsed_seconds %.1f",
        analysis_id,
        len(lookbacks),
        total_rows,
        elapsed,
    )


def compute_lookback(bars: BarData, cfg: AnalysisConfig, lookback_bars: int) -> LookbackResult:
    min_lookback = min(cfg.min_lookback_bars, int(lookback_bars))
    log.info("Profile compute start lookback_bars %d min_lookback_bars %d bars %d", lookback_bars, min_lookback, len(bars))
    arrays = rolling_range_profile_arrays(
        bars,
        RangeProfileConfig(
            price_step=cfg.price_step,
            lookback_bars=int(lookback_bars),
            min_lookback_bars=min_lookback,
            bar_seconds=cfg.bar_seconds,
            profile_max_lookback_seconds=cfg.profile_max_lookback_seconds,
        ),
    )
    log.info("Profile compute complete lookback_bars %d bars %d", lookback_bars, len(bars))
    return LookbackResult(
        lookback_bars=int(lookback_bars),
        min_lookback_bars=min_lookback,
        arrays=arrays,
    )


def _persist_result(
    conn,
    cfg: AnalysisConfig,
    analysis_id: UUID,
    created_at: datetime,
    data_start_ts: datetime,
    data_end_ts: datetime,
    ticks_loaded: int,
    bars: BarData,
    result: LookbackResult,
) -> int:
    log.info("DB copy start lookback_bars %d bars %d", result.lookback_bars, len(bars))
    inserted = persistence.copy_profile_rows(
        conn,
        cfg,
        analysis_id,
        created_at,
        data_start_ts,
        data_end_ts,
        ticks_loaded,
        bars,
        result.lookback_bars,
        result.min_lookback_bars,
        result.arrays,
    )
    log.info("DB copy complete lookback_bars %d rows %d", result.lookback_bars, inserted)
    return inserted


def _multiprocessing_context() -> mp.context.BaseContext:
    methods = mp.get_all_start_methods()
    if "fork" in methods:
        return mp.get_context("fork")
    return mp.get_context()


def _init_worker(bars: BarData, cfg: AnalysisConfig) -> None:
    global _WORKER_BARS, _WORKER_CFG
    _WORKER_BARS = bars
    _WORKER_CFG = cfg
    profile.warmup()


def _compute_worker_lookback(lookback_bars: int) -> LookbackResult:
    if _WORKER_BARS is None or _WORKER_CFG is None:
        raise RuntimeError("Worker was not initialized")
    return compute_lookback(_WORKER_BARS, _WORKER_CFG, int(lookback_bars))

