"""Single-model worker entry point and startup context logging."""

import logging
from pathlib import Path

from . import runtime
from .config import *
from .db import connect_with_retry, validate_result_schema, validate_source_schema
from .grid_search import _print_grid_summary, run_grid_search
from .logging_utils import set_log_process_name
from .model_loader import _validate_model_filename, load_model_module
from .simulation import run_backtest

log = logging.getLogger(__name__)

def log_backtest_context(model_files: list[str]) -> None:
    log.info(
        "Source tables — bars=%s  fundamentals=%s  world_regime=%s  account_profile=%s  pepperstone_table=%s",
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
        "Backtesting model selection — selection=%s  count=%d  files=%s  dir=%s  parallelism=%d",
        MODEL_SELECTION,
        len(model_files),
        ",".join(model_files),
        MODEL_DIR,
        MODEL_PARALLELISM,
    )
    log.info("Account profile — profile=%s", ACCOUNT_PROFILE)
    if ACCOUNT_PROFILE == "ps_acc":
        log.info(
            "Pepperstone margin policy — margin_requirement_pct=%.2f  stop_out_margin_level_pct=%.2f  min_entry_margin_level_pct=%.2f",
            MARGIN_REQUIREMENT_PCT,
            PS_MARGIN_STOP_OUT_LEVEL_PCT,
            PS_MIN_ENTRY_MARGIN_LEVEL_PCT,
        )
    else:
        log.info(
            "IBKR margin policy — long_initial=%.2f  long_maintenance=%.2f  short_initial=%.2f  short_maintenance=%.2f",
            IBKR_LONG_INITIAL_MARGIN_PCT,
            IBKR_LONG_MAINTENANCE_MARGIN_PCT,
            IBKR_SHORT_INITIAL_MARGIN_PCT,
            IBKR_SHORT_MAINTENANCE_MARGIN_PCT,
        )
    log.info(
        "Execution model — fractional_shares=%s  spread_bps=%.2f  slippage_bps=%.2f  commission_per_order=%.2f  commission_per_share=%.4f  commission_min=%.2f  commission_max_pct=%.2f  commission_bps=%.2f",
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
            "Pepperstone share CFD overnight model — rollover_tz=America/New_York rollover_time=17:00 arr_pct=%.2f admin_fee_pct=%.2f short_borrow_rate_pct=%.2f day_count=%.0f friday_multiplier=3",
            PS_SHARE_CFD_ARR_PCT,
            PS_SHARE_CFD_ADMIN_FEE_PCT,
            PS_SHARE_CFD_SHORT_BORROW_RATE_PCT,
            PS_SHARE_CFD_OVERNIGHT_DAY_COUNT,
        )
    else:
        log.info(
            "Margin financing model — margin_financing_rate_pct=%.2f",
            MARGIN_FINANCING_RATE_PCT,
        )
    log.info(
        "Entry window — enabled=%s  tz=%s  start=%s  end=%s",
        ENTRY_WINDOW_ENABLED,
        ENTRY_WINDOW_TZ,
        ENTRY_WINDOW_START,
        ENTRY_WINDOW_END,
    )
    log.info(
        "SL/TP window — tz=%s  start=%s  end=%s",
        SL_TP_WINDOW_TZ,
        SL_TP_WINDOW_START,
        SL_TP_WINDOW_END,
    )
    log.info(
        "Holding rule — long_max_hold_days=%.2f  short_max_hold_days=%.2f  sl_tp_active_from=next_1h_bar",
        LONG_MAX_HOLD_DAYS,
        SHORT_MAX_HOLD_DAYS,
    )
    log.info(
        "Candidate filter — min_market_cap_m=%.0f  require_usd_fundamentals=%s  allow_rebuilt_historical_fundamentals=%s",
        MIN_MARKET_CAP_M,
        REQUIRE_USD_FUNDAMENTALS,
        ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS,
    )
    log.info("Sector diversification — enabled=%s", SECTOR_DIVERSIFICATION_ENABLED)
    if ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS:
        log.warning(
            "Point-in-time guard relaxed — rebuilt historical fundamentals are allowed even when data_available_at is after the simulated day. Use these results for research only, not final strategy validation."
        )
    log.info("Grid search — enabled=%s", GRID_SEARCH_ENABLED)
    log.info(
        "Performance caches — trading_days=on  world_regime=on  candidates=on  bars=on",
    )


def run_single_model_worker() -> None:
    model_file = MODEL_FILE
    _validate_model_filename(model_file)
    model_files = [model_file]
    runtime.CURRENT_MODEL_FILE = model_file
    set_log_process_name(f"bt-{Path(model_file).stem}")
    conn = connect_with_retry()
    try:
        log.info(
            "Connected. Starting backtest model=%s %s → %s, equity=%.0f application_name=%s",
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
            "Model worker — file=%s  grid_search=%s",
            runtime.CURRENT_MODEL_FILE,
            GRID_SEARCH_ENABLED,
        )
        if GRID_SEARCH_ENABLED:
            results = run_grid_search(conn, cfg)
            _print_grid_summary(results)
        else:
            run_backtest(conn, cfg, LONG_MAX_HOLD_DAYS, SHORT_MAX_HOLD_DAYS, TP1_CLOSE_RATIO)
    finally:
        conn.close()
