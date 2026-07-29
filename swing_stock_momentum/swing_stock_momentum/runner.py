from __future__ import annotations

from datetime import datetime, timezone
import logging
from time import perf_counter
from uuid import uuid4

from .config import Config
from .db import (
    acquire_run_lock,
    build_run_row,
    connect,
    iter_market_days,
    read_snapshot_metadata,
    release_run_lock,
    validate_schema,
    write_result,
)
from .engine import BacktestProgress, run_backtest


log = logging.getLogger(__name__)


def run(cfg: Config) -> str:
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    run_started = perf_counter()
    log.info(
        "Backtest %s starting: requested start %s, capital %.2f USD, "
        "max %d new positions per day, %d open positions and a %d-session "
        "retrospective SEC earnings blackout",
        run_id,
        cfg.strategy.requested_start_date,
        cfg.strategy.starting_capital_usd,
        cfg.strategy.max_new_positions_per_day,
        cfg.strategy.max_positions,
        cfg.strategy.earnings_blackout_sessions,
    )
    with connect(cfg) as connection:
        acquired = False
        try:
            acquire_run_lock(connection)
            acquired = True
            log.info("Backtest %s acquired the database advisory lock", run_id)
            log.info("Backtest %s validating the database schema", run_id)
            validate_schema(connection, cfg)
            log.info(
                "Backtest %s validating source freshness and identity coverage",
                run_id,
            )
            metadata = read_snapshot_metadata(connection, cfg)
            log.info(
                "Backtest %s source snapshot is complete through %s with %d rows",
                run_id,
                metadata.end_date,
                metadata.source_row_count,
            )

            def report_progress(progress: BacktestProgress) -> None:
                if (
                    progress.sessions_processed == 1
                    or progress.sessions_processed
                    % cfg.progress_log_interval_sessions
                    == 0
                ):
                    log.info(
                        "Backtest %s progress: %d sessions through %s, equity %.2f USD, "
                        "%d candidates, %d selected, %d open and %d closed trades",
                        run_id,
                        progress.sessions_processed,
                        progress.valuation_date,
                        progress.total_equity_usd,
                        progress.signal_count,
                        progress.selected_signal_count,
                        progress.open_position_count,
                        progress.closed_trade_count,
                    )

            log.info("Backtest %s starting the sequential portfolio simulation", run_id)
            result = run_backtest(
                iter_market_days(connection, cfg, metadata),
                cfg.strategy,
                progress_callback=report_progress,
            )
            log.info(
                "Backtest %s simulation finished: %d sessions, %d signals and %d trades; "
                "preparing the atomic database write",
                run_id,
                len(result.equity_daily),
                len(result.signal_decisions),
                len(result.trades),
            )
            earnings_blackouts = sum(
                row["decision"] == "earnings_blackout"
                for row in result.signal_decisions
            )
            incomplete_earnings_horizons = sum(
                row["decision"] == "earnings_horizon_incomplete"
                for row in result.signal_decisions
            )
            log.info(
                "Backtest %s earnings filter rejected %d candidates for confirmed SEC "
                "earnings and %d candidates for an incomplete end horizon",
                run_id,
                earnings_blackouts,
                incomplete_earnings_horizons,
            )
            completed_at = datetime.now(timezone.utc)
            run_row = build_run_row(
                cfg,
                metadata,
                result,
                run_id,
                started_at,
                completed_at,
            )
            write_result(connection, cfg, run_row, result)
            log.info("Backtest %s committing all result tables atomically", run_id)
            connection.commit()
            log.info(
                "Backtest %s completed through %s: equity %.2f USD, return %.4f%%, "
                "%d closed and %d open trades in %.1f seconds",
                run_id,
                result.end_date,
                result.ending_equity_usd,
                result.total_return_pct,
                result.closed_trade_count,
                result.open_trade_count,
                perf_counter() - run_started,
            )
            return run_id
        except Exception:
            connection.rollback()
            raise
        finally:
            if acquired:
                try:
                    release_run_lock(connection)
                    connection.commit()
                except Exception:
                    connection.rollback()
                    log.exception("Failed to release the backtest advisory lock cleanly")
