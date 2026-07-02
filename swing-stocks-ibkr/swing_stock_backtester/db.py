from __future__ import annotations

from datetime import date
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor, execute_values

from .config import BacktestConfig
from .models import EarningsEvent, StockIdentity, StrategyResult, Trade


def table_identifier(table_name: str) -> sql.Identifier:
    parts = [part.strip('"') for part in table_name.split(".") if part]
    return sql.Identifier(*parts)


def connect(cfg: BacktestConfig):
    return psycopg2.connect(**cfg.connection_kwargs())


def table_exists(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (table_name,))
        return cur.fetchone()[0] is not None


def validate_tables(conn, cfg: BacktestConfig) -> None:
    required = [
        cfg.core_table,
        cfg.market_daily_table,
        cfg.fundamental_daily_table,
        cfg.earnings_table,
        cfg.runs_table,
        cfg.results_table,
        cfg.trades_table,
        cfg.equity_table,
    ]
    if cfg.require_ibkr_symbols:
        required.append(cfg.ibkr_symbols_table)

    missing = [table for table in required if not table_exists(conn, table)]
    if missing:
        raise RuntimeError(f"Missing required tables: {', '.join(missing)}")


def load_universe(conn, cfg: BacktestConfig) -> list[StockIdentity]:
    params: list[Any] = [cfg.start_date, cfg.end_date, cfg.min_price]
    symbol_filter = sql.SQL("")
    if cfg.symbols:
        symbol_filter = sql.SQL("AND UPPER(c.symbol) = ANY(%s)")
        params.append(list(cfg.symbols))

    ibkr_filter = sql.SQL("")
    if cfg.require_ibkr_symbols:
        ibkr_filter = sql.SQL(
            """
            AND EXISTS (
                SELECT 1
                FROM {ibkr} i
                WHERE UPPER(i.ib_symbol) = UPPER(c.symbol)
                   OR UPPER(i.source_symbol) = UPPER(c.symbol)
                   OR UPPER(i.local_symbol) = UPPER(c.symbol)
            )
            """
        ).format(ibkr=table_identifier(cfg.ibkr_symbols_table))

    limit_clause = sql.SQL("")
    if cfg.max_symbols > 0:
        limit_clause = sql.SQL("LIMIT %s")
        params.append(cfg.max_symbols)

    query = sql.SQL(
        """
        SELECT c.symbol, c.exchange, c.cik
        FROM {core} c
        WHERE NULLIF(TRIM(c.symbol), '') IS NOT NULL
          AND NULLIF(TRIM(c.exchange), '') IS NOT NULL
          AND c.cik IS NOT NULL
          AND UPPER(COALESCE(c.quote_type, '')) = 'EQUITY'
          AND COALESCE(c.tradeable, TRUE) = TRUE
          AND UPPER(COALESCE(c.currency, 'USD')) = 'USD'
          AND EXISTS (
              SELECT 1
              FROM {market} m
              WHERE m.symbol = c.symbol
                AND m.exchange = c.exchange
                AND m.cik = c.cik
                AND m.period_end_date BETWEEN %s AND %s
                AND COALESCE(m.adjusted_close, m.raw_close, m.current_price) >= %s
          )
          {symbol_filter}
          {ibkr_filter}
        ORDER BY c.symbol, c.exchange, c.cik
        {limit_clause}
        """
    ).format(
        core=table_identifier(cfg.core_table),
        market=table_identifier(cfg.market_daily_table),
        symbol_filter=symbol_filter,
        ibkr_filter=ibkr_filter,
        limit_clause=limit_clause,
    )

    with conn.cursor() as cur:
        cur.execute(query, params)
        return [StockIdentity(str(row[0]), str(row[1]), int(row[2])) for row in cur.fetchall()]


def load_symbol_market_rows(conn, cfg: BacktestConfig, identity: StockIdentity) -> list[dict[str, Any]]:
    query = sql.SQL(
        """
        SELECT
            m.period_end_date AS day,
            COALESCE(m.adjusted_open, m.raw_open) AS open,
            COALESCE(m.adjusted_high, m.raw_high) AS high,
            COALESCE(m.adjusted_low, m.raw_low) AS low,
            COALESCE(m.adjusted_close, m.raw_close, m.current_price) AS close,
            COALESCE(m.adjusted_volume, m.raw_volume) AS volume,
            m.market_cap,
            m.market_cap_currency,
            m.average_daily_volume_3m,
            m.week_52_change,
            m.fifty_day_average,
            m.two_hundred_day_average,
            m.historical_price_ts,
            c.sector,
            c.industry,
            c.ibkr_category,
            c.ibkr_subcategory,
            f.period_end_date AS fundamental_asof_date,
            f.sec_data_available_at,
            f.sec_latest_filing_date,
            f.sec_latest_period_end_date,
            f.sec_fundamental_currency,
            f.sec_market_currency,
            f.revenue_growth,
            f.earnings_growth,
            f.sec_gross_margin_ttm,
            f.sec_operating_margin_ttm,
            f.sec_net_margin_ttm,
            f.sec_fcf_margin_ttm,
            f.sec_fcf_sbc_adjusted_margin_ttm,
            f.sec_debt_to_capital,
            f.sec_cash_to_assets,
            f.sec_current_ratio,
            f.sec_accruals_ratio
        FROM {market} m
        LEFT JOIN LATERAL (
            SELECT
                f.period_end_date,
                f.sec_data_available_at,
                f.sec_latest_filing_date,
                f.sec_latest_period_end_date,
                f.sec_fundamental_currency,
                f.sec_market_currency,
                f.revenue_growth,
                f.earnings_growth,
                f.sec_gross_margin_ttm,
                f.sec_operating_margin_ttm,
                f.sec_net_margin_ttm,
                f.sec_fcf_margin_ttm,
                f.sec_fcf_sbc_adjusted_margin_ttm,
                f.sec_debt_to_capital,
                f.sec_cash_to_assets,
                f.sec_current_ratio,
                f.sec_accruals_ratio,
                f.last_update_ts
            FROM {fundamental} f
            WHERE f.symbol = m.symbol
              AND f.exchange = m.exchange
              AND f.cik = m.cik
              AND f.period_end_date <= m.period_end_date
            ORDER BY f.period_end_date DESC, f.last_update_ts DESC
            LIMIT 1
        ) f ON TRUE
        LEFT JOIN {core} c
          ON c.symbol = m.symbol
         AND c.exchange = m.exchange
         AND c.cik = m.cik
        WHERE m.symbol = %s
          AND m.exchange = %s
          AND m.cik = %s
          AND m.period_end_date BETWEEN %s AND %s
          AND COALESCE(m.adjusted_close, m.raw_close, m.current_price) IS NOT NULL
          AND COALESCE(m.adjusted_close, m.raw_close, m.current_price) > 0
        ORDER BY m.period_end_date
        """
    ).format(
        market=table_identifier(cfg.market_daily_table),
        fundamental=table_identifier(cfg.fundamental_daily_table),
        core=table_identifier(cfg.core_table),
    )
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            query,
            (
                identity.symbol,
                identity.exchange,
                identity.cik,
                cfg.load_start_date,
                cfg.end_date,
            ),
        )
        return [dict(row) for row in cur.fetchall()]


def load_earnings_events(conn, cfg: BacktestConfig, identity: StockIdentity) -> list[EarningsEvent]:
    query = sql.SQL(
        """
        SELECT
            earnings_date,
            announcement_ts,
            announcement_time_type,
            source,
            known_as_of_ts,
            is_confirmed,
            surprise_pct
        FROM {earnings}
        WHERE symbol = %s
          AND exchange = %s
          AND cik = %s
          AND earnings_date BETWEEN %s AND %s
        ORDER BY COALESCE(known_as_of_ts, announcement_ts), earnings_date, source
        """
    ).format(earnings=table_identifier(cfg.earnings_table))
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(
            query,
            (
                identity.symbol,
                identity.exchange,
                identity.cik,
                cfg.start_date,
                cfg.end_date,
            ),
        )
        rows = cur.fetchall()
    return [
        EarningsEvent(
            earnings_date=row["earnings_date"],
            announcement_ts=row["announcement_ts"],
            announcement_time_type=str(row["announcement_time_type"] or "unknown"),
            source=str(row["source"] or ""),
            known_as_of_ts=row["known_as_of_ts"],
            is_confirmed=bool(row["is_confirmed"]),
            surprise_pct=float(row["surprise_pct"]) if row["surprise_pct"] is not None else None,
        )
        for row in rows
    ]


def create_run(conn, cfg: BacktestConfig, strategy_count: int, symbol_count: int) -> int:
    query = sql.SQL(
        """
        INSERT INTO {runs} (
            strategy_count,
            symbol_count,
            start_date,
            end_date,
            universe_mode,
            max_symbols,
            process_parallelism,
            min_price,
            min_market_cap,
            min_average_daily_volume,
            commission_bps,
            slippage_bps,
            write_equity_daily,
            source_core_table,
            source_market_table,
            source_fundamental_table,
            source_earnings_table
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING run_id
        """
    ).format(runs=table_identifier(cfg.runs_table))
    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                strategy_count,
                symbol_count,
                cfg.start_date,
                cfg.end_date,
                cfg.universe_mode,
                cfg.max_symbols,
                cfg.process_parallelism,
                cfg.min_price,
                cfg.min_market_cap,
                cfg.min_average_daily_volume,
                cfg.commission_bps,
                cfg.slippage_bps,
                cfg.write_equity_daily,
                cfg.core_table,
                cfg.market_daily_table,
                cfg.fundamental_daily_table,
                cfg.earnings_table,
            ),
        )
        run_id = int(cur.fetchone()[0])
    conn.commit()
    return run_id


def update_run_progress(conn, cfg: BacktestConfig, run_id: int, processed: int, failed: int) -> None:
    query = sql.SQL(
        """
        UPDATE {runs}
        SET processed_symbol_count = %s,
            failed_symbol_count = %s
        WHERE run_id = %s
        """
    ).format(runs=table_identifier(cfg.runs_table))
    with conn.cursor() as cur:
        cur.execute(query, (processed, failed, run_id))
    conn.commit()


def finish_run(conn, cfg: BacktestConfig, run_id: int, status: str, error_text: str | None = None) -> None:
    query = sql.SQL(
        """
        UPDATE {runs}
        SET finished_at_utc = now(),
            status = %s,
            error_text = %s
        WHERE run_id = %s
        """
    ).format(runs=table_identifier(cfg.runs_table))
    with conn.cursor() as cur:
        cur.execute(query, (status, error_text, run_id))
    conn.commit()


def result_tuple(run_id: int, result: StrategyResult) -> tuple[Any, ...]:
    identity = result.identity
    return (
        run_id,
        result.strategy_name,
        result.strategy_version,
        identity.symbol,
        identity.exchange,
        identity.cik,
        result.status,
        result.first_trade_date,
        result.last_trade_date,
        result.trade_count,
        result.win_count,
        result.loss_count,
        result.flat_count,
        result.avg_return_pct,
        result.median_return_pct,
        result.best_return_pct,
        result.worst_return_pct,
        result.total_compounded_return_pct,
        result.max_drawdown_pct,
        result.profit_factor,
        result.expectancy_pct,
        result.avg_holding_days,
        result.exposure_days,
        result.signal_count,
        result.skipped_signal_count,
        result.error_text,
    )


def trade_tuple(run_id: int, trade: Trade) -> tuple[Any, ...]:
    identity = trade.identity
    return (
        run_id,
        trade.strategy_name,
        trade.strategy_version,
        identity.symbol,
        identity.exchange,
        identity.cik,
        trade.trade_number,
        trade.signal_date,
        trade.entry_date,
        trade.exit_date,
        trade.entry_price,
        trade.exit_price,
        trade.stop_price,
        trade.gross_return_pct,
        trade.net_return_pct,
        trade.holding_days,
        trade.exit_reason,
        trade.signal_score,
        trade.quality_score,
        trade.momentum_score,
        trade.entry_condition,
        trade.fundamental_asof_date,
        trade.earnings_event_date,
        trade.earnings_known_asof_ts,
    )


def equity_tuple(run_id: int, row: dict[str, Any]) -> tuple[Any, ...]:
    identity = row["identity"]
    return (
        run_id,
        row["strategy_name"],
        row["strategy_version"],
        identity.symbol,
        identity.exchange,
        identity.cik,
        row["day"],
        row["equity"],
        row["drawdown_pct"],
        row["in_position"],
    )


def save_results(conn, cfg: BacktestConfig, run_id: int, results: list[StrategyResult]) -> None:
    if not results:
        return

    result_rows = [result_tuple(run_id, result) for result in results]
    trade_rows = [trade_tuple(run_id, trade) for result in results for trade in result.trades]
    equity_rows = [equity_tuple(run_id, row) for result in results for row in result.equity_rows]

    with conn.cursor() as cur:
        execute_values(
            cur,
            sql.SQL(
                """
                INSERT INTO {results} (
                    run_id, strategy_name, strategy_version, symbol, exchange, cik, status,
                    first_trade_date, last_trade_date, trade_count, win_count, loss_count, flat_count,
                    avg_return_pct, median_return_pct, best_return_pct, worst_return_pct,
                    total_compounded_return_pct, max_drawdown_pct, profit_factor, expectancy_pct,
                    avg_holding_days, exposure_days, signal_count, skipped_signal_count, error_text
                )
                VALUES %s
                """
            ).format(results=table_identifier(cfg.results_table)).as_string(conn),
            result_rows,
        )

        if trade_rows:
            execute_values(
                cur,
                sql.SQL(
                    """
                    INSERT INTO {trades} (
                        run_id, strategy_name, strategy_version, symbol, exchange, cik, trade_number,
                        signal_date, entry_date, exit_date, entry_price, exit_price, stop_price,
                        gross_return_pct, net_return_pct, holding_days, exit_reason,
                        signal_score, quality_score, momentum_score, entry_condition,
                        fundamental_asof_date, earnings_event_date, earnings_known_asof_ts
                    )
                    VALUES %s
                    """
                ).format(trades=table_identifier(cfg.trades_table)).as_string(conn),
                trade_rows,
            )

        if equity_rows:
            execute_values(
                cur,
                sql.SQL(
                    """
                    INSERT INTO {equity} (
                        run_id, strategy_name, strategy_version, symbol, exchange, cik,
                        day, equity, drawdown_pct, in_position
                    )
                    VALUES %s
                    """
                ).format(equity=table_identifier(cfg.equity_table)).as_string(conn),
                equity_rows,
            )
    conn.commit()
