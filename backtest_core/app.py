"""Single-model worker entry point and startup context logging."""

import logging
import os
import signal
import time as _time
from pathlib import Path

import psycopg2

from . import runtime
from .config import *
from .db import connect_with_retry, validate_result_schema, validate_source_schema
from .grid_search import _print_grid_summary, run_grid_search
from .logging_utils import set_log_process_name
from .market_data import clear_market_data_caches, preload_candidate_timelines
from .model_loader import (
    _validate_model_filename,
    load_model_config_env,
    load_model_module,
    model_requires_fundamental_sources,
)
from .persistence import delete_run_results
from .policy import COMMON_POLICY, candidate_policy_kwargs
from .simulation import run_backtest

log = logging.getLogger(__name__)


def _install_worker_shutdown_handler() -> None:
    def _handle_shutdown(signum: int, _frame: object) -> None:
        log.warning("Model worker shutdown requested signal %d model %s", signum, runtime.CURRENT_MODEL_FILE)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)


def _reserved_run_id_from_env() -> int | None:
    raw = os.getenv("BACKTEST_RUN_ID", "").strip()
    if not raw:
        return None
    try:
        run_id = int(raw)
    except ValueError as exc:
        raise ValueError(f"BACKTEST_RUN_ID must be a positive integer, got {raw!r}") from exc
    if run_id <= 0:
        raise ValueError(f"BACKTEST_RUN_ID must be a positive integer, got {raw!r}")
    return run_id


def _close_connection(conn: psycopg2.extensions.connection | None) -> None:
    if conn is None:
        return
    try:
        conn.close()
    except Exception as exc:
        log.warning("DB connection close failed %s", exc)


def _cleanup_reserved_run_for_retry(run_id: int) -> None:
    conn = connect_with_retry()
    try:
        delete_run_results(conn, run_id)
    finally:
        _close_connection(conn)


def _worker_db_retry_attempts(reserved_run_id: int | None) -> int:
    if GRID_SEARCH_ENABLED or reserved_run_id is None:
        if BACKTEST_WORKER_DB_RETRY_ATTEMPTS > 1:
            log.info(
                "DB retry disabled for worker grid search %s reserved run id %s",
                GRID_SEARCH_ENABLED,
                str(reserved_run_id) if reserved_run_id is not None else "-",
            )
        return 1
    return BACKTEST_WORKER_DB_RETRY_ATTEMPTS


def run_shared_candidate_timeline_prebuilder() -> None:
    set_log_process_name(f"bt-timeline-{ACCOUNT_PROFILE}")
    conn = connect_with_retry()
    try:
        log.info(
            "Shared candidate timeline prebuild starting account profile %s application name %s",
            ACCOUNT_PROFILE,
            DB["application_name"],
        )
        validate_source_schema(conn)
        timeline_sets, timeline_rows, timeline_identities, timeline_mib = preload_candidate_timelines(
            conn,
            ("LONG", "SHORT"),
            **candidate_policy_kwargs(),
            source_table=SOURCE_FUNDAMENTAL_SCORES_TABLE,
            as_of_date=START_DATE,
            as_of_ts=None,
            pepperstone_table=PS_TRADABLE_SYMBOLS_TABLE,
            required_currency="USD" if REQUIRE_USD_FUNDAMENTALS else None,
            allow_rebuilt_historical_fundamentals=ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS,
            filter_negative_earnings_by_direction={"LONG": False, "SHORT": False},
            ibkr_margin_table=IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
            fundamental_score_mode="peer",
            fundamental_peer_weight=1.0,
            fundamental_abs_weight=0.0,
            long_min_absolute_score=None,
            short_max_absolute_score=None,
        )
        if CANDIDATE_TIMELINE_CACHE_ENABLED and timeline_sets <= 0:
            raise RuntimeError(
                "Shared candidate timeline prebuild produced no cache sets; refusing slow per-worker fallback."
            )
        log.info(
            "Shared candidate timeline prebuild complete account profile %s cache sets %d rows %d identities %d estimated %.0f MiB",
            ACCOUNT_PROFILE,
            timeline_sets,
            timeline_rows,
            timeline_identities,
            timeline_mib,
        )
    finally:
        conn.close()


def log_backtest_context(model_files: list[str]) -> None:
    log.info(
        "Source tables bars %s fundamentals %s earnings calendar %s world regime %s account profile %s PS tradable symbols table %s",
        SOURCE_MARKET_DATA_1H_TABLE,
        SOURCE_FUNDAMENTAL_SCORES_TABLE,
        SOURCE_EARNINGS_CALENDAR_EVENTS_TABLE,
        SOURCE_WORLD_REGIME_TABLE,
        ACCOUNT_PROFILE,
        PS_TRADABLE_SYMBOLS_TABLE,
    )
    if ACCOUNT_PROFILE == "ps_acc":
        log.info(
            "Pepperstone account filter active — candidates must exist in %s with symbol_ps%s and trading enabled",
            PS_TRADABLE_SYMBOLS_TABLE,
            " or symbol_ps24" if PS_24_ENTRY_SL_TP_ACTIVE else "",
        )
        log.info("Pepperstone 24h entry/sl/tp active %s", PS_24_ENTRY_SL_TP_ACTIVE)
    log.info(
        "Backtesting model selection %s count %d files %s dir %s config dir %s config required %s parallelism %d",
        MODEL_SELECTION,
        len(model_files),
        ",".join(model_files),
        MODEL_DIR,
        MODEL_CONFIG_DIR,
        MODEL_CONFIG_REQUIRED,
        MODEL_PARALLELISM,
    )
    log.info("Fundamental relative score only %s", FUNDAMENTAL_RELATIVE_SCORE_ONLY)
    log.info("Account profile %s", ACCOUNT_PROFILE)
    log.info(
        "Portfolio capacity max open positions %d max position opens per day %d max position opens per hour %d",
        MAX_OPEN_POSITIONS,
        MAX_POSITION_OPENS_PER_DAY,
        MAX_POSITION_OPENS_PER_HOUR,
    )
    log.info(
        "Intent score clustering enabled %s delta %.4f min size %d random seed %d",
        INTENT_SCORE_CLUSTERING_ENABLED,
        INTENT_SCORE_CLUSTER_DELTA,
        INTENT_SCORE_CLUSTER_MIN_SIZE,
        INTENT_SCORE_CLUSTER_RANDOM_SEED,
    )
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
            IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
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
        "Common stop loss policy enabled %s lookback bars %d buffer %.4f ATR lookback bars %d ATR multiplier %.2f min stop pct %.2f max stop pct %.2f",
        COMMON_STOP_LOSS_ENABLED,
        COMMON_STOP_LOOKBACK_BARS,
        COMMON_STOP_BUFFER,
        COMMON_STOP_ATR_LOOKBACK_BARS,
        COMMON_STOP_ATR_MULT,
        COMMON_MIN_STOP_PCT,
        COMMON_MAX_STOP_PCT,
    )
    log.info(
        "Common eligibility min market cap %.0f require USD fundamentals %s high leverage filter %s negative earnings long filter %s short filter %s scorer eligibility filter %s require upcoming earnings date %s days %d",
        COMMON_POLICY.min_market_cap_m,
        REQUIRE_USD_FUNDAMENTALS,
        COMMON_POLICY.filter_high_leverage,
        COMMON_POLICY.filter_negative_earnings_long,
        COMMON_POLICY.filter_negative_earnings_short,
        COMMON_POLICY.filter_scorer_eligibility,
        COMMON_POLICY.require_upcoming_earnings_date,
        COMMON_REQUIRE_UPCOMING_EARNINGS_DATE_DAYS,
    )
    log.info(
        "Common earnings blackout enabled %s days %d historical known days before %d source table %s",
        COMMON_EARNINGS_BLACKOUT_ENABLED,
        COMMON_EARNINGS_BLACKOUT_DAYS,
        COMMON_HISTORICAL_EARNINGS_KNOWN_DAYS_BEFORE,
        SOURCE_EARNINGS_CALENDAR_EVENTS_TABLE,
    )
    log.info(
        "Common eligibility long min fundamental %.2f short max fundamental %.2f long blocklist %s short blocklist %s",
        COMMON_POLICY.long_min_fundamental,
        COMMON_POLICY.short_max_fundamental,
        ",".join(COMMON_POLICY.long_label_blocklist) or "-",
        ",".join(COMMON_POLICY.short_label_blocklist) or "-",
    )
    log.info("Sector diversification enabled %s", SECTOR_DIVERSIFICATION_ENABLED)
    log.info("Grid search enabled %s", GRID_SEARCH_ENABLED)
    log.info(
        "Performance caches — trading days on, world regime on, candidate timeline %s max %.0f MiB, bars incremental PIT batches of %d symbols with %d warmup days decision event mode %s flush batch %d",
        CANDIDATE_TIMELINE_CACHE_ENABLED,
        CANDIDATE_TIMELINE_CACHE_MAX_MIB,
        BAR_CACHE_BATCH_SIZE,
        BAR_CACHE_WARMUP_DAYS,
        DECISION_EVENT_MODE,
        DECISION_EVENT_FLUSH_BATCH_SIZE,
    )


def run_single_model_worker() -> None:
    model_file = MODEL_FILE
    reserved_run_id = _reserved_run_id_from_env()
    _validate_model_filename(model_file)
    model_files = [model_file]
    runtime.CURRENT_MODEL_FILE = model_file
    set_log_process_name(f"bt-{Path(model_file).stem}-{ACCOUNT_PROFILE}")
    _install_worker_shutdown_handler()
    runtime.MODEL_MODULE = load_model_module(model_file)
    load_model_config_env(model_file)
    cfg = runtime.MODEL_MODULE.intent_config_from_env()
    require_fundamental_sources = model_requires_fundamental_sources(runtime.MODEL_MODULE)
    max_attempts = _worker_db_retry_attempts(reserved_run_id)

    for attempt in range(1, max_attempts + 1):
        conn = connect_with_retry()
        log.info(
            "Connected starting backtest model %s %s to %s equity %.0f application name %s attempt %d/%d",
            runtime.CURRENT_MODEL_FILE,
            START_DATE,
            END_DATE,
            INITIAL_EQUITY,
            DB["application_name"],
            attempt,
            max_attempts,
        )
        try:
            log.info(
                "Model source requirements fundamentals %s",
                require_fundamental_sources,
            )
            validate_source_schema(
                conn,
                require_fundamental_sources=require_fundamental_sources,
            )
            validate_result_schema(conn)
            log_backtest_context(model_files)
            log.info(
                "Model worker file %s grid search %s min bars %d",
                runtime.CURRENT_MODEL_FILE,
                GRID_SEARCH_ENABLED,
                cfg.min_bars,
            )
            if reserved_run_id is not None:
                log.info("Model worker using reserved run id %d", reserved_run_id)
            log.info(
                "Execution policy model %s take profit mode %s long fixed tp %.2f%% short fixed tp %.2f%% long trailing activation %.2f%% distance %.2f%% short trailing activation %.2f%% distance %.2f%% long max hold %.2f short max hold %.2f stop source common",
                runtime.CURRENT_MODEL_FILE,
                TAKE_PROFIT_MODE,
                EXECUTION_LONG_TAKE_PROFIT_PCT,
                EXECUTION_SHORT_TAKE_PROFIT_PCT,
                EXECUTION_LONG_TRAILING_ACTIVATION_PCT,
                EXECUTION_LONG_TRAILING_DISTANCE_PCT,
                EXECUTION_SHORT_TRAILING_ACTIVATION_PCT,
                EXECUTION_SHORT_TRAILING_DISTANCE_PCT,
                EXECUTION_LONG_MAX_HOLD_DAYS,
                EXECUTION_SHORT_MAX_HOLD_DAYS,
            )
            if GRID_SEARCH_ENABLED:
                if reserved_run_id is not None:
                    raise ValueError("BACKTEST_RUN_ID cannot be used with GRID_SEARCH_ENABLED=true")
                results = run_grid_search(conn, cfg)
                _print_grid_summary(results)
            else:
                try:
                    run_backtest(conn, cfg, reserved_run_id=reserved_run_id)
                finally:
                    clear_market_data_caches("single_run")
            return
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            _close_connection(conn)
            if attempt >= max_attempts or reserved_run_id is None:
                raise
            log.warning(
                "DB connection lost during worker attempt %d/%d model %s account profile %s retry in %.1f seconds error %s",
                attempt,
                max_attempts,
                runtime.CURRENT_MODEL_FILE,
                ACCOUNT_PROFILE,
                BACKTEST_WORKER_DB_RETRY_DELAY_SECONDS,
                exc,
            )
            _time.sleep(BACKTEST_WORKER_DB_RETRY_DELAY_SECONDS)
            _cleanup_reserved_run_for_retry(reserved_run_id)
        finally:
            if not conn.closed:
                _close_connection(conn)
