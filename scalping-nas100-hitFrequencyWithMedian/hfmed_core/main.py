"""Orchestration: load ticks, build bars and run single or walk-forward mode."""

import logging
import gc
import time

from . import config, persistence
from .data import build_mid_bars, load_ticks
from .db import connect_with_retry
from .optimizer import run_single_backtest, run_walk_forward_optimizer

log = logging.getLogger(__name__)


def main() -> None:
    started = time.time()
    cfg = config.active_run_config()
    opt_cfg = config.active_optimizer_config()
    log.info(
        "NAS100 hit-frequency median start mode %s symbol %s source %s start %s end %s bar_seconds %d baseline_lookback %d profile_max_lookback_seconds %d long_cross_quantile %.4f short_cross_quantile %.4f sessions %s",
        config.RUN_MODE,
        cfg.symbol,
        cfg.source_table,
        cfg.start_ts_utc,
        cfg.end_ts_utc,
        cfg.bar_seconds,
        cfg.lookback_bars,
        cfg.profile_max_lookback_seconds or cfg.lookback_bars * cfg.bar_seconds,
        cfg.long_cross_quantile,
        cfg.short_cross_quantile,
        config.session_filter_summary(cfg),
    )

    conn = connect_with_retry()
    try:
        persistence.validate_schema(conn)
        raw_ticks = load_ticks(conn, cfg)
        ticks, bars = build_mid_bars(raw_ticks, cfg)
        del raw_ticks
        gc.collect()
        if config.RUN_MODE == "walk_forward":
            run_walk_forward_optimizer(conn, cfg, opt_cfg, ticks, bars, started)
        else:
            run_single_backtest(conn, cfg, opt_cfg, ticks, bars, started)
    finally:
        conn.close()
