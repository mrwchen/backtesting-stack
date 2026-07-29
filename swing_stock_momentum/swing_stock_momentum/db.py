from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
import logging
from typing import Any, Iterable, Iterator, Mapping, Sequence

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from .config import Config
from .contracts import (
    ANALYSER_COLUMNS,
    EQUITY_RESULT_COLUMNS,
    RUN_RESULT_COLUMNS,
    SIGNAL_RESULT_COLUMNS,
    SOURCE_COLUMNS,
    TRADE_RESULT_COLUMNS,
)
from .engine import BacktestResult, Bar


log = logging.getLogger(__name__)
ADVISORY_LOCK_KEY = 8_614_026_001
ANALYSER_STATE_NAME = "stock_analyser"


@dataclass(frozen=True)
class SnapshotMetadata:
    source_watermark_utc: datetime
    analyser_watermark_utc: datetime
    source_end_date: date
    analyser_end_date: date
    lookback_start_date: date
    source_row_count: int
    analyser_row_count: int

    @property
    def end_date(self) -> date:
        return min(self.source_end_date, self.analyser_end_date)


def _qualified_identifier(name: str) -> sql.Composed:
    return sql.SQL(".").join(sql.Identifier(part) for part in name.split("."))


def _schema_and_table(name: str) -> tuple[str, str]:
    parts = name.split(".")
    return (parts[0], parts[1]) if len(parts) == 2 else ("public", parts[0])


@contextmanager
def connect(cfg: Config) -> Iterator[psycopg2.extensions.connection]:
    connection = psycopg2.connect(
        host=cfg.pg_host,
        port=cfg.pg_port,
        dbname=cfg.pg_database,
        user=cfg.pg_user,
        password=cfg.pg_password,
        application_name=cfg.pg_app_name[:63],
        connect_timeout=cfg.db_connect_timeout_seconds,
        options=f"-c statement_timeout={cfg.db_statement_timeout_ms}",
    )
    connection.set_session(isolation_level="REPEATABLE READ", readonly=False)
    try:
        yield connection
    finally:
        connection.close()


def acquire_run_lock(connection: psycopg2.extensions.connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_KEY,))
        if not bool(cursor.fetchone()[0]):
            raise RuntimeError(
                "another swing_stock_momentum backtest is already running"
            )


def release_run_lock(connection: psycopg2.extensions.connection) -> None:
    with connection.cursor() as cursor:
        cursor.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_KEY,))


def _column_names(
    connection: psycopg2.extensions.connection, table_name: str
) -> set[str]:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, relation_name),
        )
        return {str(row[0]) for row in cursor.fetchall()}


def _require_columns(
    connection: psycopg2.extensions.connection,
    table_name: str,
    required: Iterable[str],
) -> None:
    missing = sorted(set(required) - _column_names(connection, table_name))
    if missing:
        raise RuntimeError(f"table {table_name} is missing: {', '.join(missing)}")


def _validate_hypertable(
    connection: psycopg2.extensions.connection, table_name: str
) -> None:
    schema_name, relation_name = _schema_and_table(table_name)
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT compression_enabled
            FROM timescaledb_information.hypertables
            WHERE hypertable_schema = %s AND hypertable_name = %s
            """,
            (schema_name, relation_name),
        )
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"table {table_name} must be a TimescaleDB hypertable")
        if bool(row[0]):
            raise RuntimeError(f"hypertable {table_name} must not use compression")


def validate_schema(
    connection: psycopg2.extensions.connection, cfg: Config
) -> None:
    _require_columns(connection, cfg.source_market_table, SOURCE_COLUMNS)
    _require_columns(connection, cfg.source_analyser_table, ANALYSER_COLUMNS)
    _require_columns(
        connection,
        cfg.source_analyser_state_table,
        ("state_name", "source_last_update_ts"),
    )
    _require_columns(connection, cfg.runs_table, RUN_RESULT_COLUMNS)
    _require_columns(connection, cfg.signals_table, SIGNAL_RESULT_COLUMNS)
    _require_columns(connection, cfg.trades_table, TRADE_RESULT_COLUMNS)
    _require_columns(connection, cfg.equity_table, EQUITY_RESULT_COLUMNS)
    _validate_hypertable(connection, cfg.signals_table)
    _validate_hypertable(connection, cfg.equity_table)


def read_snapshot_metadata(
    connection: psycopg2.extensions.connection, cfg: Config
) -> SnapshotMetadata:
    source = _qualified_identifier(cfg.source_market_table)
    analyser = _qualified_identifier(cfg.source_analyser_table)
    state = _qualified_identifier(cfg.source_analyser_state_table)
    query = sql.SQL(
        """
        SELECT
            (SELECT max(last_update_ts) FROM {source}) AS source_watermark,
            (SELECT source_last_update_ts FROM {state} WHERE state_name = %s)
                AS analyser_watermark,
            (SELECT max(period_end_date) FROM {source}) AS source_end_date,
            (SELECT max(period_end_date) FROM {analyser}) AS analyser_end_date
        """
    ).format(source=source, state=state, analyser=analyser)
    with connection.cursor() as cursor:
        cursor.execute(query, (ANALYSER_STATE_NAME,))
        row = cursor.fetchone()
    source_watermark, analyser_watermark, source_end, analyser_end = row
    if source_watermark is None or source_end is None:
        raise RuntimeError(f"source table {cfg.source_market_table} is empty")
    if analyser_watermark is None or analyser_end is None:
        raise RuntimeError(f"source table {cfg.source_analyser_table} is incomplete")
    if source_watermark != analyser_watermark:
        raise RuntimeError(
            "stock_analyser is not current with stock_core_market_metrics_daily: "
            f"source watermark {source_watermark.isoformat()}, analyser watermark "
            f"{analyser_watermark.isoformat()}"
        )
    if source_end != analyser_end:
        raise RuntimeError(
            "source and analyser latest complete dates differ: "
            f"{source_end} versus {analyser_end}"
        )
    if source_end < cfg.strategy.requested_start_date:
        raise RuntimeError(
            f"no complete source day exists on or after {cfg.strategy.requested_start_date}"
        )

    coverage_query = sql.SQL(
        """
        WITH source_keys AS (
            SELECT period_end_date, symbol, exchange, cik
            FROM {source}
            WHERE period_end_date BETWEEN %s AND %s
        ), analyser_keys AS (
            SELECT period_end_date, symbol, exchange, cik
            FROM {analyser}
            WHERE period_end_date BETWEEN %s AND %s
        )
        SELECT
            (SELECT count(*) FROM source_keys),
            (SELECT count(*) FROM analyser_keys),
            (SELECT count(*) FROM source_keys s JOIN analyser_keys a
               USING (period_end_date, symbol, exchange, cik))
        """
    ).format(source=source, analyser=analyser)
    bounds = (cfg.strategy.requested_start_date, source_end)
    with connection.cursor() as cursor:
        cursor.execute(coverage_query, (*bounds, *bounds))
        source_count, analyser_count, matched_count = map(int, cursor.fetchone())
    if source_count != analyser_count or matched_count != source_count:
        raise RuntimeError(
            "source/analyser identity coverage is incomplete in the backtest range: "
            f"source={source_count}, analyser={analyser_count}, matched={matched_count}"
        )

    required_history = max(
        cfg.strategy.atr_period_sessions,
        cfg.strategy.prior_high_lookback_sessions,
    )
    lookback_query = sql.SQL(
        """
        SELECT min(period_end_date)
        FROM (
            SELECT DISTINCT period_end_date
            FROM {source}
            WHERE period_end_date < %s
            ORDER BY period_end_date DESC
            LIMIT %s
        ) history
        """
    ).format(source=source)
    with connection.cursor() as cursor:
        cursor.execute(
            lookback_query,
            (cfg.strategy.requested_start_date, required_history),
        )
        lookback_start = cursor.fetchone()[0]
    if lookback_start is None:
        raise RuntimeError(
            f"insufficient source history before {cfg.strategy.requested_start_date}"
        )
    return SnapshotMetadata(
        source_watermark_utc=source_watermark,
        analyser_watermark_utc=analyser_watermark,
        source_end_date=source_end,
        analyser_end_date=analyser_end,
        lookback_start_date=lookback_start,
        source_row_count=source_count,
        analyser_row_count=analyser_count,
    )


def _market_query(cfg: Config) -> sql.Composed:
    source = _qualified_identifier(cfg.source_market_table)
    analyser = _qualified_identifier(cfg.source_analyser_table)
    analyser_select = sql.SQL(",\n            ").join(
        sql.SQL("a.{column} AS {alias}").format(
            column=sql.Identifier(column),
            alias=sql.Identifier(f"analyser__{column}"),
        )
        for column in ANALYSER_COLUMNS
    )
    atr_preceding = sql.Literal(cfg.strategy.atr_period_sessions - 1)
    atr_lag = sql.Literal(cfg.strategy.atr_period_sessions)
    high_lookback = sql.Literal(cfg.strategy.prior_high_lookback_sessions)
    return sql.SQL(
        """
        WITH calendar AS (
            SELECT period_end_date,
                   dense_rank() OVER (ORDER BY period_end_date) AS session_number
            FROM (
                SELECT DISTINCT period_end_date
                FROM {source}
                WHERE period_end_date BETWEEN %s AND %s
            ) dates
        ), lagged AS (
            SELECT
                s.period_end_date,
                s.symbol,
                s.exchange,
                s.cik,
                s.adjusted_open,
                s.adjusted_high,
                s.adjusted_low,
                s.adjusted_close,
                s.price_continuity_segment,
                c.session_number,
                lag(s.adjusted_close) OVER identity_window AS previous_close,
                lag(c.session_number, {atr_lag}) OVER identity_window
                    AS atr_boundary_session_number,
                lag(c.session_number, {high_lookback}) OVER identity_window
                    AS high_boundary_session_number,
                count(s.adjusted_high) OVER prior_high_window
                    AS prior_high_observation_count_raw,
                max(s.adjusted_high) OVER prior_high_window
                    AS prior_max_adjusted_high_raw
            FROM {source} s
            JOIN calendar c USING (period_end_date)
            WINDOW identity_window AS (
                PARTITION BY s.symbol, s.exchange, s.cik, s.price_continuity_segment
                ORDER BY s.period_end_date
            ), prior_high_window AS (
                PARTITION BY s.symbol, s.exchange, s.cik, s.price_continuity_segment
                ORDER BY s.period_end_date
                ROWS BETWEEN {high_lookback} PRECEDING AND 1 PRECEDING
            )
        ), true_ranges AS (
            SELECT
                lagged.*,
                CASE
                    WHEN adjusted_high IS NULL OR adjusted_low IS NULL
                      OR previous_close IS NULL OR previous_close <= 0
                    THEN NULL
                    ELSE greatest(
                        adjusted_high - adjusted_low,
                        abs(adjusted_high - previous_close),
                        abs(adjusted_low - previous_close)
                    )
                END AS true_range
            FROM lagged
        ), featured AS (
            SELECT
                true_ranges.*,
                avg(true_range) OVER atr_window AS atr_value,
                count(true_range) OVER atr_window AS atr_observation_count
            FROM true_ranges
            WINDOW atr_window AS (
                PARTITION BY symbol, exchange, cik, price_continuity_segment
                ORDER BY period_end_date
                ROWS BETWEEN {atr_preceding} PRECEDING AND CURRENT ROW
            )
        )
        SELECT
            f.period_end_date,
            f.symbol,
            f.exchange,
            f.cik,
            f.price_continuity_segment,
            f.adjusted_open,
            f.adjusted_high,
            f.adjusted_low,
            f.adjusted_close,
            CASE
                WHEN f.atr_observation_count = {atr_lag}
                 AND f.session_number - f.atr_boundary_session_number = {atr_lag}
                 AND f.adjusted_close > 0
                THEN f.atr_value / f.adjusted_close * 100.0
                ELSE NULL
            END AS atr_pct,
            CASE
                WHEN f.prior_high_observation_count_raw = {high_lookback}
                 AND f.session_number - f.high_boundary_session_number = {high_lookback}
                THEN f.prior_high_observation_count_raw
                ELSE 0
            END AS prior_high_observation_count,
            CASE
                WHEN f.prior_high_observation_count_raw = {high_lookback}
                 AND f.session_number - f.high_boundary_session_number = {high_lookback}
                THEN f.prior_max_adjusted_high_raw
                ELSE NULL
            END AS prior_max_adjusted_high,
            {analyser_select}
        FROM featured f
        LEFT JOIN {analyser} a
          ON a.period_end_date = f.period_end_date
         AND a.symbol = f.symbol
         AND a.exchange = f.exchange
         AND a.cik = f.cik
        WHERE f.period_end_date BETWEEN %s AND %s
        ORDER BY f.period_end_date, f.symbol, f.exchange, f.cik
        """
    ).format(
        source=source,
        analyser=analyser,
        analyser_select=analyser_select,
        atr_preceding=atr_preceding,
        atr_lag=atr_lag,
        high_lookback=high_lookback,
    )


def iter_market_days(
    connection: psycopg2.extensions.connection,
    cfg: Config,
    metadata: SnapshotMetadata,
) -> Iterator[tuple[date, tuple[Bar, ...]]]:
    cursor_name = f"backtest_momentum_source_{datetime.utcnow().strftime('%H%M%S%f')}"
    with connection.cursor(name=cursor_name, cursor_factory=RealDictCursor) as cursor:
        cursor.itersize = cfg.db_fetch_batch_size
        cursor.execute(
            _market_query(cfg),
            (
                metadata.lookback_start_date,
                metadata.end_date,
                cfg.strategy.requested_start_date,
                metadata.end_date,
            ),
        )
        current_date: date | None = None
        bars: list[Bar] = []
        for row in cursor:
            row_date = row["period_end_date"]
            if current_date is not None and row_date != current_date:
                yield current_date, tuple(bars)
                bars = []
            current_date = row_date
            bars.append(Bar.from_mapping(row))
        if current_date is not None:
            yield current_date, tuple(bars)


def _insert_rows(
    cursor: psycopg2.extensions.cursor,
    table_name: str,
    columns: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    page_size: int,
) -> None:
    if not rows:
        return
    query = sql.SQL("INSERT INTO {} ({}) VALUES %s").format(
        _qualified_identifier(table_name),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
    )
    values = [tuple(row.get(column) for column in columns) for row in rows]
    execute_values(cursor, query.as_string(cursor), values, page_size=page_size)


def build_run_row(
    cfg: Config,
    metadata: SnapshotMetadata,
    result: BacktestResult,
    run_id: str,
    started_at_utc: datetime,
    completed_at_utc: datetime,
) -> dict[str, Any]:
    analyser = cfg.analyser
    strategy = cfg.strategy
    return {
        "run_id": run_id,
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "account_type": strategy.account_type,
        "currency": strategy.currency,
        "trading_timezone": strategy.trading_timezone,
        "requested_start_date": strategy.requested_start_date,
        "actual_start_date": result.actual_start_date,
        "end_date": result.end_date,
        "starting_capital_usd": strategy.starting_capital_usd,
        "ending_cash_usd": result.ending_cash_usd,
        "ending_market_value_usd": result.ending_market_value_usd,
        "ending_equity_usd": result.ending_equity_usd,
        "realized_pnl_usd": result.realized_pnl_usd,
        "unrealized_pnl_usd": result.unrealized_pnl_usd,
        "total_return_pct": result.total_return_pct,
        "max_drawdown_pct": result.max_drawdown_pct,
        "total_commission_usd": result.total_commission_usd,
        "signal_count": len(result.signal_decisions),
        "selected_signal_count": result.selected_signal_count,
        "closed_trade_count": result.closed_trade_count,
        "open_trade_count": result.open_trade_count,
        "winning_trade_count": result.winning_trade_count,
        "losing_trade_count": result.losing_trade_count,
        "source_watermark_utc": metadata.source_watermark_utc,
        "analyser_watermark_utc": metadata.analyser_watermark_utc,
        "source_market_table": cfg.source_market_table,
        "source_analyser_table": cfg.source_analyser_table,
        "price_basis": "adjusted_ohlc",
        "entry_execution_model": "signal_day_close",
        "atr_exit_execution_model": "signal_day_close",
        "intraday_conflict_policy": "low_before_high",
        "gap_execution_policy": "open_when_beyond_active_level",
        "end_of_data_policy": "mark_to_market",
        "prior_high_price_field": "adjusted_high",
        "atr_method": "simple_tr14_mean_over_adjusted_close_pct",
        "fractional_shares_allowed": False,
        "risk_equity_basis": "current_account_equity_before_entry",
        "ranking_policy": "volume_ratio_desc_return_desc_symbol_asc",
        "symbol_reentry_policy": "allowed_after_exit",
        "max_positions": strategy.max_positions,
        "max_new_positions_per_day": strategy.max_new_positions_per_day,
        "risk_per_position_pct": strategy.risk_per_position_pct,
        "initial_stop_loss_pct": strategy.initial_stop_loss_pct,
        "stop_step_interval_sessions": strategy.stop_step_interval_sessions,
        "stop_step_pct": strategy.stop_step_pct,
        "initial_take_profit_pct": strategy.initial_take_profit_pct,
        "take_profit_step_interval_sessions": strategy.take_profit_step_interval_sessions,
        "take_profit_step_pct": strategy.take_profit_step_pct,
        "atr_period_sessions": strategy.atr_period_sessions,
        "atr_day1_exit_max_pct": strategy.atr_day1_exit_max_pct,
        "atr_day2_exit_max_pct": strategy.atr_day2_exit_max_pct,
        "prior_high_lookback_sessions": strategy.prior_high_lookback_sessions,
        "prior_high_max_above_signal_close_pct": strategy.prior_high_max_above_signal_close_pct,
        "min_daily_price_change_pct": strategy.min_daily_price_change_pct,
        "max_daily_price_change_pct_exclusive": strategy.max_daily_price_change_pct_exclusive,
        "min_volume_vs_sma21_ratio_exclusive": strategy.min_volume_vs_sma21_ratio_exclusive,
        "commission_bps": strategy.commission_bps,
        "slippage_bps": strategy.slippage_bps,
        "analyser_min_price_usd": analyser.min_price_usd,
        "analyser_min_dollar_volume_usd": analyser.min_dollar_volume_usd,
        "analyser_rs_lookback_1_sessions": analyser.rs_lookbacks[0],
        "analyser_rs_lookback_2_sessions": analyser.rs_lookbacks[1],
        "analyser_rs_lookback_3_sessions": analyser.rs_lookbacks[2],
        "analyser_rs_lookback_4_sessions": analyser.rs_lookbacks[3],
        "analyser_rs_weight_1": analyser.rs_weights[0],
        "analyser_rs_weight_2": analyser.rs_weights[1],
        "analyser_rs_weight_3": analyser.rs_weights[2],
        "analyser_rs_weight_4": analyser.rs_weights[3],
        "analyser_rs_min": analyser.rs_min,
        "analyser_ma200_trend_sessions": analyser.ma200_trend_sessions,
        "analyser_min_above_52w_low_ratio": analyser.min_above_52w_low_ratio,
        "analyser_min_near_52w_high_ratio": analyser.min_near_52w_high_ratio,
    }


def write_result(
    connection: psycopg2.extensions.connection,
    cfg: Config,
    run_row: Mapping[str, Any],
    result: BacktestResult,
) -> None:
    run_id = run_row["run_id"]
    signals = [dict(row, run_id=run_id) for row in result.signal_decisions]
    trades = [dict(row, run_id=run_id) for row in result.trades]
    equity = [dict(row, run_id=run_id) for row in result.equity_daily]
    with connection.cursor() as cursor:
        _insert_rows(cursor, cfg.runs_table, RUN_RESULT_COLUMNS, [run_row], 1)
        _insert_rows(
            cursor,
            cfg.signals_table,
            SIGNAL_RESULT_COLUMNS,
            signals,
            cfg.db_write_page_size,
        )
        _insert_rows(
            cursor,
            cfg.trades_table,
            TRADE_RESULT_COLUMNS,
            trades,
            cfg.db_write_page_size,
        )
        _insert_rows(
            cursor,
            cfg.equity_table,
            EQUITY_RESULT_COLUMNS,
            equity,
            cfg.db_write_page_size,
        )
