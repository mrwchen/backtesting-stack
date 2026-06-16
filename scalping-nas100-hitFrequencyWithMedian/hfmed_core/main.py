"""Orchestration: load ticks -> build bars -> simulate -> summarise -> persist."""

import logging
import time

from . import config, persistence
from .data import build_mid_bars, load_ticks
from .db import connect_with_retry
from .risk import run_monte_carlo, summarize_trades
from .simulation import run_simulation

log = logging.getLogger(__name__)


def main() -> None:
    started = time.time()
    cfg = config.active_run_config()
    log.info(
        "NAS100 hit-frequency median backtest start symbol %s source %s start %s end %s bar_seconds %d lookback %d stop %.2f tp %.2f",
        cfg.symbol, cfg.source_table, cfg.start_ts_utc, cfg.end_ts_utc,
        cfg.bar_seconds, cfg.lookback_bars, cfg.stop_points, cfg.take_profit_points,
    )

    conn = connect_with_retry()
    try:
        persistence.validate_schema(conn)
        ticks = load_ticks(conn)
        bars = build_mid_bars(ticks)
        data_start_ts = ticks["tick_time"].iloc[0].to_pydatetime()
        data_end_ts = ticks["tick_time"].iloc[-1].to_pydatetime()

        run_id = persistence.create_run(conn, cfg, data_start_ts, data_end_ts, len(ticks), len(bars))
        result = run_simulation(ticks, bars)
        summary = summarize_trades(result.trades, result.initial_equity, result.final_equity)
        mc = run_monte_carlo(result.trades, result.initial_equity)

        persistence.write_trades(conn, run_id, result.trades)
        persistence.write_monte_carlo(conn, run_id, mc)
        persistence.update_run_summary(
            conn,
            run_id,
            summary,
            result,
            run_duration_seconds=time.time() - started,
        )

        log.info(
            "Run %d complete trades %d final_equity %.2f return %.2f%% max_drawdown %.2f%% win_rate %.2f%%",
            run_id,
            summary["total_trades"],
            result.final_equity,
            summary["total_return_pct"],
            summary["max_drawdown_pct"],
            summary["win_rate_pct"],
        )
    finally:
        conn.close()

