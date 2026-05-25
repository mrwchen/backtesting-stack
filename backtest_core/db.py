"""Database connection and source/result schema validation."""

import logging
import time as _time
from datetime import datetime, timedelta, timezone

import psycopg2
from psycopg2 import sql

from .config import *
from .market_data import _day_close_ts
from .sql_utils import relation_identifier

log = logging.getLogger(__name__)


def _configure_session(conn: psycopg2.extensions.connection) -> None:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (f"{DB_STATEMENT_TIMEOUT_MS}ms",))
        cur.execute("SELECT set_config('lock_timeout', %s, false)", (f"{DB_LOCK_TIMEOUT_MS}ms",))
        cur.execute(
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            (f"{DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS}ms",),
        )
    log.info(
        "DB session configured statement timeout %d ms, lock timeout %d ms, idle transaction timeout %d ms",
        DB_STATEMENT_TIMEOUT_MS,
        DB_LOCK_TIMEOUT_MS,
        DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS,
    )


def connect_with_retry() -> psycopg2.extensions.connection:
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            conn = psycopg2.connect(**DB)
            _configure_session(conn)
            return conn
        except psycopg2.OperationalError as exc:
            if attempt == DB_CONNECT_RETRIES:
                raise
            delay = DB_CONNECT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            log.warning("DB connect failed (%d/%d, retry in %.0fs): %s", attempt, DB_CONNECT_RETRIES, delay, exc)
            _time.sleep(delay)
    raise RuntimeError("unreachable")


# ── Source validation ─────────────────────────────────────────────────────────

def _relation_columns(conn: psycopg2.extensions.connection, relation_name: str) -> set[str]:
    """Return column names for a table/view/materialized view, or fail clearly."""
    relation_identifier(relation_name)  # validates plain/schema-qualified identifier
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass(%s)
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (relation_name,),
        )
        columns = {row[0] for row in cur.fetchall()}
    if not columns:
        raise RuntimeError(f"Required source relation not found or has no columns: {relation_name}")
    return columns


def _require_columns(conn: psycopg2.extensions.connection, relation_name: str, required: set[str]) -> None:
    columns = _relation_columns(conn, relation_name)
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(
            f"Required relation {relation_name} is missing columns: {', '.join(missing)}"
        )
    log.info(
        "Validated relation schema %s required columns %d available columns %d",
        relation_name,
        len(required),
        len(columns),
    )


def validate_source_schema(conn: psycopg2.extensions.connection) -> None:
    """Validate source tables/columns and basic date coverage before the run."""
    fundamental_required = {
        "time",
        "symbol",
        "exchange",
        "cik",
        "data_available_at",
        "fundamental_data_available_at",
        "composite_score",
        "sector",
        "industry",
        "valuation_label",
        "mispricing_score",
        "negative_earnings_flag",
        "high_leverage_flag",
        "market_cap_m",
    }
    if REQUIRE_USD_FUNDAMENTALS:
        fundamental_required.update({
            "current_price_currency",
            "market_cap_currency",
            "currency",
            "financial_currency",
        })

    _require_columns(conn, SOURCE_MARKET_DATA_1H_TABLE, {"symbol", "exchange", "cik", "ts", "open", "high", "low", "close", "volume"})
    _require_columns(conn, SOURCE_FUNDAMENTAL_SCORES_TABLE, fundamental_required)
    _require_columns(conn, SOURCE_WORLD_REGIME_TABLE, {"day", "regime_label", "composite_score"})

    if ACCOUNT_PROFILE == "ps_acc":
        _require_columns(conn, PS_TRADABLE_SYMBOLS_TABLE, {"symbol", "symbol_ps", "is_trading_enabled"})
    elif ACCOUNT_PROFILE == "ibkr_acc":
        _require_columns(conn, IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE, {
            "source_symbol",
            "action",
            "quantity",
            "initial_margin_pct",
            "maintenance_margin_pct",
            "fetched_at",
        })

    _validate_source_coverage(conn)


def validate_result_schema(conn: psycopg2.extensions.connection) -> None:
    """Validate result tables created by the init container before writing."""
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_runs", {
        "run_id",
        "initial_equity",
        "run_duration_seconds",
        "margin_hours_usd",
        "return_per_margin_hour_pct",
        "take_profit_mode",
        "execution_long_take_profit_pct",
        "execution_short_take_profit_pct",
        "execution_long_trailing_activation_pct",
        "execution_short_trailing_activation_pct",
        "execution_long_trailing_distance_pct",
        "execution_short_trailing_distance_pct",
        "common_stop_buffer",
        "ps_share_cfd_arr_pct",
        "ps_share_cfd_admin_fee_pct",
        "ps_share_cfd_short_borrow_rate_pct",
        "ps_share_cfd_overnight_day_count",
    })
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_trades", {
        "run_id", "symbol", "exchange", "cik", "entry_ts", "margin_hours_usd",
        "return_per_margin_hour_pct", "pnl_usd", "equity_after", "intent_score", "intent_reason",
        "take_profit_mode", "take_profit", "trailing_activation_price", "trailing_distance_pct",
        "trailing_activated", "trailing_stop", "trailing_activated_ts"
    })
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_decision_events", {
        "run_id", "symbol", "exchange", "cik", "intent_date", "intent_passed", "intent_score"
    })
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_account_curve", {
        "run_id",
        "ts",
        "trade_date",
        "seq_in_run",
        "balance_usd",
        "open_pnl_usd",
        "equity_usd",
        "initial_margin_usd",
        "maintenance_margin_usd",
        "available_funds_usd",
        "excess_liquidity_usd",
        "open_positions",
        "realized_pnl_usd",
        "closed_trades",
    })


def _validate_source_coverage(conn: psycopg2.extensions.connection) -> None:
    start_ts = datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), datetime.min.time(), tzinfo=timezone.utc)
    end_ts = _day_close_ts(END_DATE)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(ts), MAX(ts) FROM {} "
                "WHERE ts >= %s AND ts <= %s"
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (start_ts, end_ts),
        )
        bar_min_ts, bar_max_ts = cur.fetchone()
    if bar_min_ts is None:
        raise RuntimeError(
            f"No 1h bars in {SOURCE_MARKET_DATA_1H_TABLE} for required window {start_ts} to {end_ts}"
        )
    log.info("Source coverage bars %s min ts %s max ts %s", SOURCE_MARKET_DATA_1H_TABLE, bar_min_ts, bar_max_ts)

    fundamental_where = [
        sql.SQL("time <= %s"),
        sql.SQL("COALESCE(data_available_at, fundamental_data_available_at) <= %s"),
    ]
    fundamental_params: list[object] = [end_ts, end_ts]
    if REQUIRE_USD_FUNDAMENTALS:
        fundamental_where.append(sql.SQL(
            "COALESCE(NULLIF(current_price_currency, ''), "
            "NULLIF(market_cap_currency, ''), "
            "NULLIF(currency, ''), "
            "NULLIF(financial_currency, ''), "
            "%s) = %s"
        ))
        fundamental_params.extend(["USD", "USD"])
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(time), MAX(time), "
                "MAX(COALESCE(data_available_at, fundamental_data_available_at)) "
                "FROM {} WHERE {}"
            ).format(
                relation_identifier(SOURCE_FUNDAMENTAL_SCORES_TABLE),
                sql.SQL(" AND ").join(fundamental_where),
            ),
            fundamental_params,
        )
        fund_min_ts, fund_max_ts, fund_max_available_ts = cur.fetchone()
    if fund_min_ts is None:
        raise RuntimeError(
            f"No point-in-time fundamental rows in {SOURCE_FUNDAMENTAL_SCORES_TABLE} up to {end_ts}"
        )
    log.info(
        "Source coverage fundamentals %s min time %s max time %s max available at %s",
        SOURCE_FUNDAMENTAL_SCORES_TABLE,
        fund_min_ts,
        fund_max_ts,
        fund_max_available_ts,
    )

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(day), MAX(day) FROM {} "
                "WHERE day <= %s AND composite_score IS NOT NULL"
            ).format(relation_identifier(SOURCE_WORLD_REGIME_TABLE)),
            (END_DATE,),
        )
        regime_min_day, regime_max_day = cur.fetchone()
    if regime_min_day is None:
        raise RuntimeError(
            f"No world regime rows in {SOURCE_WORLD_REGIME_TABLE} up to {END_DATE}"
        )
    log.info(
        "Source coverage world regime %s min day %s max day %s",
        SOURCE_WORLD_REGIME_TABLE,
        regime_min_day,
        regime_max_day,
    )

    if ACCOUNT_PROFILE == "ps_acc":
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {} "
                    "WHERE symbol_ps IS NOT NULL AND is_trading_enabled IS NOT FALSE"
                ).format(relation_identifier(PS_TRADABLE_SYMBOLS_TABLE)),
            )
            tradable_symbols = cur.fetchone()[0]
        if tradable_symbols <= 0:
            raise RuntimeError(
                f"Pepperstone account selected, but {PS_TRADABLE_SYMBOLS_TABLE}.symbol_ps has no tradable rows"
            )
        log.info(
            "Source coverage pepperstone %s account profile %s tradable symbols %d",
            PS_TRADABLE_SYMBOLS_TABLE,
            ACCOUNT_PROFILE,
            tradable_symbols,
        )
    elif ACCOUNT_PROFILE == "ibkr_acc":
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        COUNT(*) FILTER (
                            WHERE quantity > 0
                              AND initial_margin_pct > 0
                              AND maintenance_margin_pct > 0
                        ) AS usable_rows,
                        COUNT(DISTINCT source_symbol) FILTER (
                            WHERE quantity > 0
                              AND initial_margin_pct > 0
                              AND maintenance_margin_pct > 0
                        ) AS usable_symbols,
                        COUNT(DISTINCT source_symbol) FILTER (
                            WHERE UPPER(TRIM(action)) = 'BUY'
                              AND quantity > 0
                              AND initial_margin_pct > 0
                              AND maintenance_margin_pct > 0
                        ) AS long_symbols,
                        COUNT(DISTINCT source_symbol) FILTER (
                            WHERE UPPER(TRIM(action)) = 'SELL'
                              AND quantity > 0
                              AND initial_margin_pct > 0
                              AND maintenance_margin_pct > 0
                        ) AS short_symbols
                    FROM {}
                    """
                ).format(relation_identifier(IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE)),
            )
            usable_rows, usable_symbols, long_symbols, short_symbols = cur.fetchone()
        if usable_rows <= 0:
            log.warning(
                "IBKR margin source %s has no usable rows; candidate selection will return no symbols",
                IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
            )
        else:
            log.info(
                "IBKR margin percentage source %s usable rows %d symbols %d long symbols %d short symbols %d",
                IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
                usable_rows,
                usable_symbols,
                long_symbols,
                short_symbols,
            )
