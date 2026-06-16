"""Orchestration: load ticks, build bars and run single or walk-forward mode."""

import logging
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
        "NAS100 hit-frequency median start mode %s symbol %s source %s start %s end %s bar_seconds %d baseline_lookback %d",
        config.RUN_MODE,
        cfg.symbol,
        cfg.source_table,
        cfg.start_ts_utc,
        cfg.end_ts_utc,
        cfg.bar_seconds,
        cfg.lookback_bars,
    )

    conn = connect_with_retry()
    try:
        persistence.validate_schema(conn)
        ticks = load_ticks(conn, cfg)
        bars = build_mid_bars(ticks, cfg)
        if config.RUN_MODE == "walk_forward":
            run_walk_forward_optimizer(conn, cfg, opt_cfg, ticks, bars, started)
        else:
            run_single_backtest(conn, cfg, opt_cfg, ticks, bars, started)
    finally:
        conn.close()
