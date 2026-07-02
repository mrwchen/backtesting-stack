from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import BacktestConfig, load_config
from .db import (
    connect,
    create_run,
    finish_run,
    load_earnings_events,
    load_symbol_market_rows,
    load_universe,
    save_results,
    update_run_progress,
    validate_tables,
)
from .logging_utils import configure_logging
from .models import StockIdentity, SymbolBacktestResult
from .strategies import empty_strategy_results, run_strategies_for_symbol, strategies_for_config

log = logging.getLogger(__name__)


def run_symbol_worker(args: tuple[BacktestConfig, StockIdentity]) -> SymbolBacktestResult:
    cfg, identity = args
    configure_logging(cfg.log_level)
    try:
        with connect(cfg) as conn:
            rows = load_symbol_market_rows(conn, cfg, identity)
            events = load_earnings_events(conn, cfg, identity)
        results = run_strategies_for_symbol(identity, rows, events, cfg)
        return SymbolBacktestResult(identity=identity, status="ok", results=results)
    except Exception as exc:
        error_text = str(exc)[:4000]
        return SymbolBacktestResult(
            identity=identity,
            status="error",
            results=empty_strategy_results(identity, "error", error_text, strategies_for_config(cfg)),
            error_text=error_text,
        )


def run_backtest(cfg: BacktestConfig) -> int:
    strategies = strategies_for_config(cfg)
    with connect(cfg) as conn:
        validate_tables(conn, cfg)
        universe = load_universe(conn, cfg)
        log.info(
            "Backtest universe loaded: %d symbols. Date range: %s to %s. Strategies: %d.",
            len(universe),
            cfg.start_date,
            cfg.end_date,
            len(strategies),
        )
        run_id = create_run(conn, cfg, len(strategies), len(universe))

    if not universe:
        with connect(cfg) as conn:
            finish_run(conn, cfg, run_id, "ok", "No symbols matched the configured universe.")
        log.info("Backtest run %d finished without symbols.", run_id)
        return run_id

    processed = 0
    failed = 0
    run_error: str | None = None
    max_workers = min(cfg.process_parallelism, len(universe))

    try:
        with connect(cfg) as write_conn:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(run_symbol_worker, (cfg, identity)): identity for identity in universe}
                for future in as_completed(futures):
                    identity = futures[future]
                    try:
                        symbol_result = future.result()
                    except Exception as exc:
                        failed += 1
                        processed += 1
                        error_text = str(exc)[:4000]
                        log.exception("Symbol failed outside worker envelope: %s %s.", identity.symbol, error_text)
                        save_results(
                            write_conn,
                            cfg,
                            run_id,
                            empty_strategy_results(identity, "error", error_text, strategies),
                        )
                    else:
                        processed += 1
                        if symbol_result.status != "ok":
                            failed += 1
                            log.warning(
                                "Symbol backtest failed: %s %s %s",
                                symbol_result.identity.symbol,
                                symbol_result.identity.exchange,
                                symbol_result.error_text or "",
                            )
                        save_results(write_conn, cfg, run_id, symbol_result.results)

                    if processed % 25 == 0 or processed == len(universe):
                        update_run_progress(write_conn, cfg, run_id, processed, failed)
                        log.info(
                            "Backtest progress: %d/%d symbols processed; %d failed.",
                            processed,
                            len(universe),
                            failed,
                        )

            status = "ok" if failed == 0 else "error"
            if failed:
                run_error = f"{failed} symbols failed; successful symbols were still persisted."
            finish_run(write_conn, cfg, run_id, status, run_error)
    except Exception as exc:
        run_error = str(exc)[:4000]
        with connect(cfg) as conn:
            finish_run(conn, cfg, run_id, "error", run_error)
        raise

    log.info("Backtest run %d finished. Processed symbols: %d. Failed symbols: %d.", run_id, processed, failed)
    return run_id


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.log_level)
    log.info("Swing stocks IBKR backtester starting.")
    run_id = run_backtest(cfg)
    log.info("Swing stocks IBKR backtester done. Run id: %d.", run_id)


if __name__ == "__main__":
    main()
