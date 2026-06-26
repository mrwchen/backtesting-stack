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
    opt_cfgs = config.active_optimizer_configs()
    first_opt_cfg = opt_cfgs[0]
    parameter_source = (
        first_opt_cfg.parameter_grid_path
        if config.RUN_MODE == "walk_forward"
        else first_opt_cfg.single_parameter_path
    )
    profile_max = (
        str(cfg.profile_max_lookback_seconds)
        if cfg.profile_max_lookback_seconds is not None
        else "per_parameter_lookback"
    )
    log.info(
        "NAS100 hit-frequency median start mode %s symbol %s source %s start %s end %s bar_seconds %d profile_max_lookback_seconds %s parameter_source %s sessions %s wf_runs %d",
        config.RUN_MODE,
        cfg.symbol,
        cfg.source_table,
        cfg.start_ts_utc,
        cfg.end_ts_utc,
        cfg.bar_seconds,
        profile_max,
        parameter_source,
        config.session_filter_summary(cfg),
        len(opt_cfgs) if config.RUN_MODE == "walk_forward" else 1,
    )

    conn = connect_with_retry()
    try:
        persistence.validate_schema(conn)
        raw_ticks = load_ticks(conn, cfg)
        ticks, bars = build_mid_bars(raw_ticks, cfg)
        del raw_ticks
        gc.collect()
        if config.RUN_MODE == "walk_forward":
            if len(opt_cfgs) > 1:
                log.info("Walk-forward matrix start runs %d", len(opt_cfgs))
            for index, opt_cfg in enumerate(opt_cfgs, start=1):
                run_started = time.time()
                log.info(
                    "Walk-forward matrix run %d/%d train_days %d test_days %d step_days %d",
                    index,
                    len(opt_cfgs),
                    opt_cfg.train_days,
                    opt_cfg.test_days,
                    opt_cfg.step_days,
                )
                run_walk_forward_optimizer(conn, cfg, opt_cfg, ticks, bars, run_started)
                gc.collect()
        else:
            run_single_backtest(conn, cfg, first_opt_cfg, ticks, bars, started)
    finally:
        conn.close()
