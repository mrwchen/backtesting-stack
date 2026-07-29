from __future__ import annotations

from datetime import datetime, timezone
import logging
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
from .engine import run_backtest


log = logging.getLogger(__name__)


def run(cfg: Config) -> str:
    run_id = str(uuid4())
    started_at = datetime.now(timezone.utc)
    with connect(cfg) as connection:
        acquired = False
        try:
            acquire_run_lock(connection)
            acquired = True
            validate_schema(connection, cfg)
            metadata = read_snapshot_metadata(connection, cfg)
            log.info(
                "Backtest %s source snapshot is complete through %s with %d rows",
                run_id,
                metadata.end_date,
                metadata.source_row_count,
            )
            result = run_backtest(iter_market_days(connection, cfg, metadata), cfg.strategy)
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
            connection.commit()
            log.info(
                "Backtest %s completed through %s: equity %.2f USD, return %.4f%%, "
                "%d closed and %d open trades",
                run_id,
                result.end_date,
                result.ending_equity_usd,
                result.total_return_pct,
                result.closed_trade_count,
                result.open_trade_count,
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
