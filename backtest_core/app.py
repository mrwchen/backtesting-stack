"""Single-model worker entry point and startup context logging."""

import logging
import signal
from pathlib import Path

from . import runtime
from .config import *
from .db import connect_with_retry, validate_result_schema, validate_source_schema
from .grid_search import _print_grid_summary, run_grid_search
from .logging_utils import set_log_process_name
from .market_data import clear_market_data_caches
from .model_loader import _validate_model_filename, load_model_module
from .simulation import run_backtest

log = logging.getLogger(__name__)


def _install_worker_shutdown_handler() -> None:
    def _handle_shutdown(signum: int, _frame: object) -> None:
        log.warning("Model worker shutdown requested signal %d model %s", signum, runtime.CURRENT_MODEL_FILE)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

def log_backtest_context(model_files: list[str]) -> None:
    log.info(
        "Source tables bars %s fundamentals %s world regime %s account profile %s pepperstone table %s",
        SOURCE_1H,
        SOURCE_FUNDAMENTAL,
        SOURCE_WORLD_REGIME,
        ACCOUNT_PROFILE,
        PEPPERSTONE_TABLE,
    )
    if ACCOUNT_PROFILE == "ps_acc":
        log.info(
            "Pepperstone account filter active — candidates must exist in %s with symbol_ps and trading enabled",
            PEPPERSTONE_TABLE,
        )
    log.info(
        "Backtesting model selection %s count %d files %s dir %s parallelism %d",
        MODEL_SELECTION,
        len(model_files),
        ",".join(model_files),
        MODEL_DIR,
        MODEL_PARALLELISM,
    )
    log.info("Account profile %s", ACCOUNT_PROFILE)
    if ACCOUNT_PROFILE == "ps_acc":
        log.info(
            "Pepperstone margin policy margin requirement %.2f stop-out margin level %.2f min entry margin level %.2f",
            MARGIN_REQUIREMENT_PCT,
            PS_MARGIN_STOP_OUT_LEVEL_PCT,
            PS_MIN_ENTRY_MARGIN_LEVEL_PCT,
        )
    else:
        log.info(
            "IBKR margin policy reads symbol margin percentages from %s",
            IBKR_MARGIN_REQUIREMENTS_TABLE,
        )
    log.info(
        "Execution model fractional shares %s spread bps %.2f slippage bps %.2f commission per order %.2f commission per share %.4f commission min %.2f commission max pct %.2f commission bps %.2f",
        ALLOW_FRACTIONAL_SHARES,
        SPREAD_BPS,
        SLIPPAGE_BPS,
        COMMISSION_PER_ORDER_USD,
        COMMISSION_PER_SHARE_USD,
        COMMISSION_MIN_PER_ORDER_USD,
        COMMISSION_MAX_PCT,
        COMMISSION_BPS,
    )
    if ACCOUNT_PROFILE == "ps_acc":
        log.info(
            "Pepperstone share CFD overnight model rollover tz America/New_York rollover time 17:00 ARR pct %.2f admin fee pct %.2f short borrow rate pct %.2f day count %.0f friday multiplier 3",
            PS_SHARE_CFD_ARR_PCT,
            PS_SHARE_CFD_ADMIN_FEE_PCT,
            PS_SHARE_CFD_SHORT_BORROW_RATE_PCT,
            PS_SHARE_CFD_OVERNIGHT_DAY_COUNT,
        )
    else:
        log.info(
            "Margin financing model rate pct %.2f",
            MARGIN_FINANCING_RATE_PCT,
        )
    log.info(
        "Entry window enabled %s tz %s start %s end %s",
        ENTRY_WINDOW_ENABLED,
        ENTRY_WINDOW_TZ,
        ENTRY_WINDOW_START,
        ENTRY_WINDOW_END,
    )
    log.info(
        "SL/TP window tz %s start %s end %s",
        SL_TP_WINDOW_TZ,
        SL_TP_WINDOW_START,
        SL_TP_WINDOW_END,
    )
    log.info(
        "Stop loss RTH guard — %s %s %s %s",
        STOP_LOSS_RTH_ONLY,
        STOP_LOSS_RTH_TZ,
        STOP_LOSS_RTH_START,
        STOP_LOSS_RTH_END,
    )
    log.info(
        "Holding rule long max hold days %.2f short max hold days %.2f SL/TP active from next 1h bar",
        LONG_MAX_HOLD_DAYS,
        SHORT_MAX_HOLD_DAYS,
    )
    log.info(
        "Candidate filter %.0f %s strict negative_earnings_long_filter %s negative_earnings_short_filter %s",
        MIN_MARKET_CAP_M,
        REQUIRE_USD_FUNDAMENTALS,
        FILTER_NEGATIVE_EARNINGS_LONG,
        FILTER_NEGATIVE_EARNINGS_SHORT,
    )
    log.info("Sector diversification enabled %s", SECTOR_DIVERSIFICATION_ENABLED)
    log.info("Grid search enabled %s", GRID_SEARCH_ENABLED)
    log.info(
        "Performance caches — trading days on, world regime on, candidates on, bars incremental PIT batches of %d symbols with %d warmup days",
        BAR_CACHE_BATCH_SIZE,
        BAR_CACHE_WARMUP_DAYS,
    )


def run_single_model_worker() -> None:
    model_file = MODEL_FILE
    _validate_model_filename(model_file)
    model_files = [model_file]
    runtime.CURRENT_MODEL_FILE = model_file
    set_log_process_name(f"bt-{Path(model_file).stem}")
    _install_worker_shutdown_handler()
    conn = connect_with_retry()
    try:
        log.info(
            "Connected starting backtest model %s %s to %s equity %.0f application name %s",
            runtime.CURRENT_MODEL_FILE,
            START_DATE,
            END_DATE,
            INITIAL_EQUITY,
            DB["application_name"],
        )
        validate_source_schema(conn)
        validate_result_schema(conn)
        log_backtest_context(model_files)
        runtime.MODEL_MODULE = load_model_module(model_file)
        cfg = runtime.MODEL_MODULE.signal_config_from_env()
        log.info(
            "Model worker file %s grid search %s",
            runtime.CURRENT_MODEL_FILE,
            GRID_SEARCH_ENABLED,
        )
        if GRID_SEARCH_ENABLED:
            results = run_grid_search(conn, cfg)
            _print_grid_summary(results)
        else:
            try:
                run_backtest(conn, cfg, LONG_MAX_HOLD_DAYS, SHORT_MAX_HOLD_DAYS, TP1_CLOSE_RATIO)
            finally:
                clear_market_data_caches("single_run")
    finally:
        conn.close()
