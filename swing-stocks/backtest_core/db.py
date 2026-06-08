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


def _require_not_null_columns(
    conn: psycopg2.extensions.connection,
    relation_name: str,
    required: set[str],
) -> None:
    relation_identifier(relation_name)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass(%s)
              AND a.attnum > 0
              AND NOT a.attisdropped
              AND a.attnotnull
            """,
            (relation_name,),
        )
        not_null_columns = {row[0] for row in cur.fetchall()}
    missing = sorted(required - not_null_columns)
    if missing:
        raise RuntimeError(
            f"Required relation {relation_name} has nullable columns that must be NOT NULL: {', '.join(missing)}"
        )
    log.info("Validated relation not-null columns %s count %d", relation_name, len(required))


def _require_unique_index(
    conn: psycopg2.extensions.connection,
    relation_name: str,
    columns: list[str],
) -> None:
    relation_identifier(relation_name)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index i
                CROSS JOIN LATERAL (
                    SELECT array_agg(a.attname::text ORDER BY k.ordinality) AS column_names
                    FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, ordinality)
                    JOIN pg_attribute a
                      ON a.attrelid = i.indrelid
                     AND a.attnum = k.attnum
                ) cols
                WHERE i.indrelid = to_regclass(%s)
                  AND i.indisunique
                  AND cols.column_names = %s::text[]
            )
            """,
            (relation_name, columns),
        )
        exists = bool(cur.fetchone()[0])
    if not exists:
        raise RuntimeError(
            f"Required relation {relation_name} is missing unique index on: {', '.join(columns)}"
        )
    log.info("Validated relation unique index %s columns %s", relation_name, ",".join(columns))


def validate_source_schema(conn: psycopg2.extensions.connection) -> None:
    """Validate source tables/columns and basic date coverage before the run."""
    world_regime_required = {"day", "regime_label", "composite_score"}
    if WORLD_REGIME_SHOCK_FIELDS_ACTIVE:
        world_regime_required.update({
            "dominant_shock_type",
            "max_shock_type_score",
            "defensive_risk_off_score",
            "energy_commodity_shock_score",
            "rates_inflation_usd_shock_score",
            "credit_banking_stress_score",
            "policy_geopolitical_score",
            "tech_stress_shock_score",
            "precious_metals_score",
            "industrial_metals_score",
            "metals_mining_shock_score",
            "metals_mining_subtype",
        })

    _require_columns(conn, SOURCE_MARKET_DATA_1H_TABLE, {"symbol", "exchange", "cik", "ts", "open", "high", "low", "close", "volume"})
    _require_columns(conn, SOURCE_WORLD_REGIME_TABLE, world_regime_required)

    if ACCOUNT_PROFILE == "ps_acc":
        pepperstone_required = {"symbol", "symbol_ps", "is_trading_enabled"}
        if PS_24_ENTRY_SL_TP_ACTIVE:
            pepperstone_required.add("symbol_ps24")
        if REQUIRE_USD_PRICE_DATA:
            pepperstone_required.add("quote_asset")
        _require_columns(conn, PS_TRADABLE_SYMBOLS_TABLE, pepperstone_required)
    elif ACCOUNT_PROFILE == "ibkr_acc":
        ibkr_required = {
            "source_symbol",
            "action",
            "quantity",
            "initial_margin_pct",
            "maintenance_margin_pct",
            "fetched_at",
        }
        if REQUIRE_USD_PRICE_DATA:
            ibkr_required.add("currency")
        _require_columns(conn, IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE, ibkr_required)

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
        "trailing_activated", "trailing_stop", "trailing_activated_ts",
    })
    _require_not_null_columns(conn, f"{RESULT_SCHEMA}.backtest_trades", {"entry_ts"})
    _require_unique_index(
        conn,
        f"{RESULT_SCHEMA}.backtest_trades",
        ["run_id", "intent_date", "symbol", "exchange", "cik", "direction", "entry_ts"],
    )
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_decision_events", {
        "run_id", "symbol", "exchange", "cik", "intent_date", "intent_passed", "intent_score",
    })
    _require_columns(conn, f"{RESULT_SCHEMA}.backtest_daily_policy_snapshots", {
        "run_id",
        "day",
        "policy_available",
        "daily_policy_phase",
        "world_regime_ma_score",
        "max_long_positions",
        "max_short_positions",
        "max_total_positions",
        "halted",
        "halt_reason_code",
        "prune_enabled",
        "prune_triggered",
        "opens_today",
        "refill_opens_today",
        "policy_block_events",
        "open_positions_end",
        "day_return_pct",
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
        "long_positions",
        "short_positions",
        "gross_notional_usd",
        "long_notional_usd",
        "short_notional_usd",
        "net_notional_usd",
        "gross_exposure_pct",
        "net_exposure_pct",
        "margin_level_pct",
        "position_budget_utilization_pct",
        "total_open_cash_risk_usd",
        "total_open_cash_risk_pct",
        "largest_position_weight_pct",
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
        tradable_where = (
            "(NULLIF(BTRIM(symbol_ps), '') IS NOT NULL OR NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL)"
            if PS_24_ENTRY_SL_TP_ACTIVE
            else "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL"
        )
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {} "
                    "WHERE {} AND is_trading_enabled IS NOT FALSE"
                ).format(
                    relation_identifier(PS_TRADABLE_SYMBOLS_TABLE),
                    sql.SQL(tradable_where),
                ),
            )
            tradable_symbols = cur.fetchone()[0]
        if tradable_symbols <= 0:
            raise RuntimeError(
                f"Pepperstone account selected, but {PS_TRADABLE_SYMBOLS_TABLE} has no tradable rows"
            )
        log.info(
            "Source coverage pepperstone %s account profile %s tradable symbols %d ps24 entry sl tp active %s",
            PS_TRADABLE_SYMBOLS_TABLE,
            ACCOUNT_PROFILE,
            tradable_symbols,
            PS_24_ENTRY_SL_TP_ACTIVE,
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
