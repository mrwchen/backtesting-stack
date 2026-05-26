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
from .model_loader import _validate_model_filename, load_model_config_env, load_model_module
from .policy import COMMON_POLICY
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
        "Source tables bars %s fundamentals %s world regime %s account profile %s PS tradable symbols table %s",
        SOURCE_MARKET_DATA_1H_TABLE,
        SOURCE_FUNDAMENTAL_SCORES_TABLE,
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
        "Common eligibility min market cap %.0f require USD fundamentals %s high leverage filter %s negative earnings long filter %s short filter %s",
        COMMON_POLICY.min_market_cap_m,
        REQUIRE_USD_FUNDAMENTALS,
        COMMON_POLICY.filter_high_leverage,
        COMMON_POLICY.filter_negative_earnings_long,
        COMMON_POLICY.filter_negative_earnings_short,
    )
    log.info(
        "Common eligibility long min fundamental %.2f short max fundamental %.2f long blocklist %s short blocklist %s",
        COMMON_POLICY.long_min_fundamental,
        COMMON_POLICY.short_max_fundamental,
        ",".join(COMMON_POLICY.long_label_blocklist) or "-",
        ",".join(COMMON_POLICY.short_label_blocklist) or "-",
    )
    log.info("Sector diversification enabled %s", SECTOR_DIVERSIFICATION_ENABLED)
    log.info(
        "Regime exposure bucket thresholds strong risk-on below %.2f strong risk-off from %.2f",
        COMMON_POLICY.regime_strong_risk_on_max_score,
        COMMON_POLICY.regime_strong_risk_off_min_score,
    )
    log.info(
        "Regime direction thresholds long below %.2f short from %.2f",
        COMMON_POLICY.regime_long_max_score,
        COMMON_POLICY.regime_short_min_score,
    )
    for bucket_name, bucket in REGIME_EXPOSURE_BUCKETS.items():
        log.info(
            "Regime exposure bucket %s long risk multiplier %.2f short risk multiplier %.2f max long positions %d max short positions %d",
            bucket_name,
            bucket["long_risk_multiplier"],
            bucket["short_risk_multiplier"],
            bucket["max_long_positions"],
            bucket["max_short_positions"],
        )
    log.info("Grid search enabled %s", GRID_SEARCH_ENABLED)
    log.info(
        "Performance caches — trading days on, world regime on, candidate timeline %s max %.0f MiB, bars incremental PIT batches of %d symbols with %d warmup days",
        CANDIDATE_TIMELINE_CACHE_ENABLED,
        CANDIDATE_TIMELINE_CACHE_MAX_MIB,
        BAR_CACHE_BATCH_SIZE,
        BAR_CACHE_WARMUP_DAYS,
    )


def run_single_model_worker() -> None:
    model_file = MODEL_FILE
    _validate_model_filename(model_file)
    model_files = [model_file]
    runtime.CURRENT_MODEL_FILE = model_file
    set_log_process_name(f"bt-{Path(model_file).stem}-{ACCOUNT_PROFILE}")
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
        load_model_config_env(model_file)
        cfg = runtime.MODEL_MODULE.intent_config_from_env()
        log.info(
            "Model worker file %s grid search %s min bars %d",
            runtime.CURRENT_MODEL_FILE,
            GRID_SEARCH_ENABLED,
            cfg.min_bars,
        )
        log.info(
            "Execution policy model %s take profit mode %s long fixed tp %.4f short fixed tp %.4f long trailing activation %.4f distance %.4f short trailing activation %.4f distance %.4f long max hold %.2f short max hold %.2f stop source common",
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
            results = run_grid_search(conn, cfg)
            _print_grid_summary(results)
        else:
            try:
                run_backtest(conn, cfg)
            finally:
                clear_market_data_caches("single_run")
    finally:
        conn.close()
