from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from psycopg2 import sql
from psycopg2.extras import execute_values

from .config import BacktestConfig, env_int, load_config
from .db import connect, table_exists, table_identifier
from .logging_utils import configure_logging

log = logging.getLogger(__name__)

DIAGNOSTIC_TABLES: dict[str, str] = {
    "diagnostic_runs": "public.backtest_swing_stock_diagnostic_runs",
    "strategy_edge": "public.backtest_swing_stock_diagnostic_strategy_edge",
    "symbol_breadth": "public.backtest_swing_stock_diagnostic_symbol_breadth",
    "yearly_stability": "public.backtest_swing_stock_diagnostic_yearly_stability",
    "exit_reasons": "public.backtest_swing_stock_diagnostic_exit_reasons",
    "exit_reason_yearly": "public.backtest_swing_stock_diagnostic_exit_reason_yearly",
    "holding_period_buckets": "public.backtest_swing_stock_diagnostic_holding_period_buckets",
    "top_bottom_symbols": "public.backtest_swing_stock_diagnostic_top_bottom_symbols",
    "feature_bucket_strength": "public.backtest_swing_stock_diagnostic_feature_bucket_strength",
    "feature_bucket_stability": "public.backtest_swing_stock_diagnostic_feature_bucket_stability",
}


@dataclass(frozen=True)
class DiagnosticsConfig:
    run_id: int | None
    top_n: int
    min_bucket_trades: int
    min_year_trades: int
    feature_lookback_days: int


def load_diagnostics_config(cfg: BacktestConfig) -> DiagnosticsConfig:
    raw_run_id = os.getenv("DIAGNOSTICS_RUN_ID", "").strip()
    return DiagnosticsConfig(
        run_id=int(raw_run_id) if raw_run_id else None,
        top_n=env_int("DIAGNOSTICS_TOP_N", 20, 1),
        min_bucket_trades=env_int("DIAGNOSTICS_MIN_BUCKET_TRADES", 100, 1),
        min_year_trades=env_int("DIAGNOSTICS_MIN_YEAR_TRADES", 20, 1),
        feature_lookback_days=env_int("DIAGNOSTICS_FEATURE_LOOKBACK_DAYS", cfg.lookback_days, 260),
    )


def fetch_rows(conn: Any, query: sql.Composable, params: tuple[Any, ...] = ()) -> tuple[list[str], list[tuple[Any, ...]]]:
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [desc.name for desc in cur.description]
        rows = cur.fetchall()
    return columns, rows


def latest_run_id(conn: Any, cfg: BacktestConfig) -> int:
    query = sql.SQL("SELECT max(run_id) FROM {runs}").format(runs=table_identifier(cfg.runs_table))
    with conn.cursor() as cur:
        cur.execute(query)
        row = cur.fetchone()
    if not row or row[0] is None:
        raise RuntimeError(f"No backtest runs found in {cfg.runs_table}.")
    return int(row[0])


def log_table_preview(table_name: str, columns: list[str], rows: list[tuple[Any, ...]], max_rows: int = 5) -> None:
    log.info("Diagnostics table %s rows %d columns %s", table_name, len(rows), ",".join(columns))
    for row in rows[:max_rows]:
        values = " | ".join("" if value is None else str(value) for value in row)
        log.info("Diagnostics preview %s %s", table_name, values)


def validate_diagnostic_tables(conn: Any) -> None:
    missing = [name for name in DIAGNOSTIC_TABLES.values() if not table_exists(conn, name)]
    if missing:
        raise RuntimeError(f"Missing diagnostic tables: {', '.join(missing)}")


def create_diagnostic_run(conn: Any, source_run_id: int, diag: DiagnosticsConfig) -> int:
    query = sql.SQL(
        """
        INSERT INTO {diagnostic_runs} (
            source_run_id,
            top_n,
            min_bucket_trades,
            min_year_trades,
            feature_lookback_days
        )
        VALUES (%s, %s, %s, %s, %s)
        RETURNING diagnostic_run_id
        """
    ).format(diagnostic_runs=table_identifier(DIAGNOSTIC_TABLES["diagnostic_runs"]))
    with conn.cursor() as cur:
        cur.execute(
            query,
            (
                source_run_id,
                diag.top_n,
                diag.min_bucket_trades,
                diag.min_year_trades,
                diag.feature_lookback_days,
            ),
        )
        diagnostic_run_id = int(cur.fetchone()[0])
    conn.commit()
    return diagnostic_run_id


def finish_diagnostic_run(
    conn: Any,
    diagnostic_run_id: int,
    status: str,
    error_text: str | None = None,
) -> None:
    query = sql.SQL(
        """
        UPDATE {diagnostic_runs}
        SET finished_at_utc = now(),
            status = %s,
            error_text = %s
        WHERE diagnostic_run_id = %s
        """
    ).format(diagnostic_runs=table_identifier(DIAGNOSTIC_TABLES["diagnostic_runs"]))
    with conn.cursor() as cur:
        cur.execute(query, (status, error_text, diagnostic_run_id))
    conn.commit()


def save_diagnostic_rows(
    conn: Any,
    table_key: str,
    diagnostic_run_id: int,
    columns: list[str],
    rows: list[tuple[Any, ...]],
) -> None:
    if not rows:
        return
    table_name = DIAGNOSTIC_TABLES[table_key]
    insert_columns = ["diagnostic_run_id", *columns]
    values = [(diagnostic_run_id, *row) for row in rows]
    query = sql.SQL("INSERT INTO {table} ({columns}) VALUES %s").format(
        table=table_identifier(table_name),
        columns=sql.SQL(", ").join(sql.Identifier(column) for column in insert_columns),
    )
    with conn.cursor() as cur:
        execute_values(cur, query.as_string(conn), values)
    conn.commit()


def run_overview_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        SELECT
            run_id,
            started_at_utc,
            finished_at_utc,
            status,
            strategy_count,
            symbol_count,
            processed_symbol_count,
            failed_symbol_count,
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
            error_text
        FROM {runs}
        WHERE run_id = %s
        """
    ).format(runs=table_identifier(cfg.runs_table))


def strategy_trade_edge_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        WITH base AS (
            SELECT
                strategy_name,
                symbol,
                exit_date,
                gross_return_pct,
                net_return_pct,
                holding_days,
                ntile(100) OVER (PARTITION BY strategy_name ORDER BY net_return_pct) AS pct_bucket
            FROM {trades}
            WHERE run_id = %s
        ),
        agg AS (
            SELECT
                strategy_name,
                count(*) AS trades,
                count(DISTINCT symbol) AS symbols,
                min(exit_date) AS first_exit,
                max(exit_date) AS last_exit,
                round(avg(gross_return_pct), 4) AS avg_gross_return_pct,
                round(avg(net_return_pct), 4) AS avg_net_return_pct,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                    AS median_net_return_pct,
                round(percentile_cont(0.1) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                    AS p10_net_return_pct,
                round(percentile_cont(0.9) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                    AS p90_net_return_pct,
                round(min(net_return_pct), 4) AS worst_net_return_pct,
                round(max(net_return_pct), 4) AS best_net_return_pct,
                round(100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0), 2)
                    AS win_rate_pct,
                round((sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                    / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0), 4)
                    AS profit_factor,
                round(avg(net_return_pct) FILTER (WHERE pct_bucket BETWEEN 2 AND 99), 4)
                    AS avg_net_return_trim_1pct_each_tail,
                round((sum(net_return_pct) FILTER (WHERE pct_bucket BETWEEN 2 AND 99 AND net_return_pct > 0))
                    / nullif(abs(sum(net_return_pct) FILTER (
                        WHERE pct_bucket BETWEEN 2 AND 99 AND net_return_pct < 0
                    )), 0), 4) AS profit_factor_trim_1pct_each_tail,
                round(avg(holding_days), 2) AS avg_holding_days,
                round(percentile_cont(0.5) WITHIN GROUP (ORDER BY holding_days)::numeric, 2)
                    AS median_holding_days
            FROM base
            GROUP BY strategy_name
        )
        SELECT *
        FROM agg
        ORDER BY strategy_name
        """
    ).format(trades=table_identifier(cfg.trades_table))


def symbol_breadth_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        SELECT
            strategy_name,
            count(*) AS symbol_strategy_rows,
            count(*) FILTER (WHERE status = 'ok') AS ok_rows,
            count(*) FILTER (WHERE status = 'insufficient_history') AS insufficient_history_rows,
            count(*) FILTER (WHERE status = 'error') AS error_rows,
            count(*) FILTER (WHERE trade_count > 0) AS symbols_with_trades,
            count(*) FILTER (WHERE trade_count > 0 AND total_compounded_return_pct > 0)
                AS symbols_positive_compounded,
            round(100.0 * count(*) FILTER (
                WHERE trade_count > 0 AND total_compounded_return_pct > 0
            )::numeric / nullif(count(*) FILTER (WHERE trade_count > 0), 0), 2)
                AS positive_symbol_share_pct,
            round(percentile_cont(0.10) WITHIN GROUP (ORDER BY total_compounded_return_pct)
                FILTER (WHERE trade_count > 0)::numeric, 4) AS p10_symbol_compounded_pct,
            round(percentile_cont(0.50) WITHIN GROUP (ORDER BY total_compounded_return_pct)
                FILTER (WHERE trade_count > 0)::numeric, 4) AS p50_symbol_compounded_pct,
            round(percentile_cont(0.90) WITHIN GROUP (ORDER BY total_compounded_return_pct)
                FILTER (WHERE trade_count > 0)::numeric, 4) AS p90_symbol_compounded_pct,
            round(max(total_compounded_return_pct), 4) AS best_symbol_compounded_pct,
            round(min(total_compounded_return_pct), 4) AS worst_symbol_compounded_pct,
            sum(signal_count) AS signals,
            sum(skipped_signal_count) AS skipped_signals
        FROM {results}
        WHERE run_id = %s
        GROUP BY strategy_name
        ORDER BY strategy_name
        """
    ).format(results=table_identifier(cfg.results_table))


def yearly_stability_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        SELECT
            strategy_name,
            extract(year FROM exit_date)::int AS exit_year,
            count(*) AS trades,
            count(DISTINCT symbol) AS symbols,
            round(avg(net_return_pct), 4) AS avg_net_return_pct,
            round(percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                AS median_net_return_pct,
            round(100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0), 2)
                AS win_rate_pct,
            round((sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0), 4)
                AS profit_factor
        FROM {trades}
        WHERE run_id = %s
        GROUP BY strategy_name, extract(year FROM exit_date)::int
        ORDER BY strategy_name, exit_year
        """
    ).format(trades=table_identifier(cfg.trades_table))


def exit_reason_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        SELECT
            strategy_name,
            exit_reason,
            count(*) AS trades,
            round(100.0 * count(*)::numeric / sum(count(*)) OVER (PARTITION BY strategy_name), 2)
                AS share_pct,
            round(avg(net_return_pct), 4) AS avg_net_return_pct,
            round(percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                AS median_net_return_pct,
            round(100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0), 2)
                AS win_rate_pct
        FROM {trades}
        WHERE run_id = %s
        GROUP BY strategy_name, exit_reason
        ORDER BY strategy_name, trades DESC
        """
    ).format(trades=table_identifier(cfg.trades_table))


def exit_reason_yearly_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        WITH grouped AS (
            SELECT
                strategy_name,
                extract(year FROM exit_date)::int AS exit_year,
                exit_reason,
                count(*) AS trades,
                count(DISTINCT symbol) AS symbols,
                avg(net_return_pct) AS avg_net_return_pct,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric
                    AS median_net_return_pct,
                100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0)
                    AS win_rate_pct,
                (sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                    / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0)
                    AS profit_factor,
                avg(holding_days) AS avg_holding_days
            FROM {trades}
            WHERE run_id = %s
            GROUP BY strategy_name, extract(year FROM exit_date)::int, exit_reason
        )
        SELECT
            strategy_name,
            exit_year,
            exit_reason,
            trades,
            symbols,
            round(100.0 * trades::numeric / nullif(sum(trades) OVER (
                PARTITION BY strategy_name, exit_year
            ), 0), 2) AS share_pct,
            round(avg_net_return_pct, 4) AS avg_net_return_pct,
            round(median_net_return_pct, 4) AS median_net_return_pct,
            round(win_rate_pct, 2) AS win_rate_pct,
            round(profit_factor, 4) AS profit_factor,
            round(avg_holding_days, 2) AS avg_holding_days
        FROM grouped
        ORDER BY strategy_name, exit_year, trades DESC
        """
    ).format(trades=table_identifier(cfg.trades_table))


def holding_period_bucket_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        WITH bucketed AS (
            SELECT
                strategy_name,
                symbol,
                net_return_pct,
                holding_days,
                CASE
                    WHEN holding_days <= 3 THEN '00_0_to_3'
                    WHEN holding_days <= 7 THEN '01_4_to_7'
                    WHEN holding_days <= 14 THEN '02_8_to_14'
                    WHEN holding_days <= 21 THEN '03_15_to_21'
                    WHEN holding_days <= 30 THEN '04_22_to_30'
                    WHEN holding_days <= 45 THEN '05_31_to_45'
                    ELSE '06_over_45'
                END AS holding_days_bucket
            FROM {trades}
            WHERE run_id = %s
        ),
        grouped AS (
            SELECT
                strategy_name,
                holding_days_bucket,
                count(*) AS trades,
                count(DISTINCT symbol) AS symbols,
                avg(net_return_pct) AS avg_net_return_pct,
                percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric
                    AS median_net_return_pct,
                100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0)
                    AS win_rate_pct,
                (sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                    / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0)
                    AS profit_factor,
                avg(holding_days) AS avg_holding_days
            FROM bucketed
            GROUP BY strategy_name, holding_days_bucket
        )
        SELECT
            strategy_name,
            holding_days_bucket,
            trades,
            symbols,
            round(100.0 * trades::numeric / nullif(sum(trades) OVER (
                PARTITION BY strategy_name
            ), 0), 2) AS share_pct,
            round(avg_net_return_pct, 4) AS avg_net_return_pct,
            round(median_net_return_pct, 4) AS median_net_return_pct,
            round(win_rate_pct, 2) AS win_rate_pct,
            round(profit_factor, 4) AS profit_factor,
            round(avg_holding_days, 2) AS avg_holding_days
        FROM grouped
        ORDER BY strategy_name, holding_days_bucket
        """
    ).format(trades=table_identifier(cfg.trades_table))


def top_bottom_symbols_query(cfg: BacktestConfig) -> sql.Composable:
    return sql.SQL(
        """
        WITH ranked AS (
            SELECT
                strategy_name,
                symbol,
                exchange,
                trade_count,
                win_count,
                loss_count,
                round(total_compounded_return_pct, 4) AS total_compounded_return_pct,
                round(max_drawdown_pct, 4) AS max_drawdown_pct,
                round(avg_return_pct, 4) AS avg_return_pct,
                row_number() OVER (
                    PARTITION BY strategy_name
                    ORDER BY total_compounded_return_pct DESC NULLS LAST
                ) AS best_rank,
                row_number() OVER (
                    PARTITION BY strategy_name
                    ORDER BY total_compounded_return_pct ASC NULLS LAST
                ) AS worst_rank
            FROM {results}
            WHERE run_id = %s
              AND trade_count > 0
        )
        SELECT
            'best' AS side,
            strategy_name,
            best_rank AS rank,
            symbol,
            exchange,
            trade_count,
            win_count,
            loss_count,
            total_compounded_return_pct,
            max_drawdown_pct,
            avg_return_pct
        FROM ranked
        WHERE best_rank <= %s
        UNION ALL
        SELECT
            'worst' AS side,
            strategy_name,
            worst_rank AS rank,
            symbol,
            exchange,
            trade_count,
            win_count,
            loss_count,
            total_compounded_return_pct,
            max_drawdown_pct,
            avg_return_pct
        FROM ranked
        WHERE worst_rank <= %s
        ORDER BY strategy_name, side, rank
        """
    ).format(results=table_identifier(cfg.results_table))


def signal_feature_ctes(cfg: BacktestConfig, has_ibkr_symbols: bool) -> sql.Composable:
    if has_ibkr_symbols:
        ibkr_match = sql.SQL(
            """
            CASE WHEN EXISTS (
                SELECT 1
                FROM {ibkr} i
                WHERE UPPER(i.ib_symbol) = UPPER(t.symbol)
                   OR UPPER(i.source_symbol) = UPPER(t.symbol)
                   OR UPPER(i.local_symbol) = UPPER(t.symbol)
            ) THEN 'ibkr_match' ELSE 'no_ibkr_match' END AS ibkr_symbol_match,
            """
        ).format(ibkr=table_identifier(cfg.ibkr_symbols_table))
    else:
        ibkr_match = sql.SQL("'ibkr_table_missing' AS ibkr_symbol_match,")

    return sql.SQL(
        """
        WITH selected_run AS (
            SELECT run_id, start_date, end_date
            FROM {runs}
            WHERE run_id = %s
        ),
        run_bounds AS (
            SELECT
                run_id,
                (start_date - (%s::int * interval '1 day'))::date AS load_start_date,
                end_date
            FROM selected_run
        ),
        run_symbols AS (
            SELECT DISTINCT symbol, exchange, cik
            FROM {trades}
            WHERE run_id = (SELECT run_id FROM selected_run)
        ),
        market_raw AS (
            SELECT
                m.symbol,
                m.exchange,
                m.cik,
                m.period_end_date AS day,
                COALESCE(m.adjusted_open, m.raw_open) AS open,
                COALESCE(m.adjusted_high, m.raw_high) AS high,
                COALESCE(m.adjusted_low, m.raw_low) AS low,
                COALESCE(m.adjusted_close, m.raw_close, m.current_price) AS close,
                COALESCE(m.adjusted_volume, m.raw_volume) AS volume,
                m.market_cap,
                m.average_daily_volume_3m,
                m.beta,
                m.week_52_change,
                m.fifty_day_average,
                m.two_hundred_day_average
            FROM {market} m
            JOIN run_symbols s
              ON s.symbol = m.symbol
             AND s.exchange = m.exchange
             AND s.cik = m.cik
            CROSS JOIN run_bounds b
            WHERE m.period_end_date BETWEEN b.load_start_date AND b.end_date
              AND COALESCE(m.adjusted_close, m.raw_close, m.current_price) IS NOT NULL
              AND COALESCE(m.adjusted_close, m.raw_close, m.current_price) > 0
        ),
        market_lag AS (
            SELECT
                *,
                lag(close) OVER symbol_day AS previous_close,
                lag(close, 5) OVER symbol_day AS close_lag_5,
                lag(close, 20) OVER symbol_day AS close_lag_20,
                lag(close, 63) OVER symbol_day AS close_lag_63,
                lag(close, 126) OVER symbol_day AS close_lag_126,
                lag(close, 252) OVER symbol_day AS close_lag_252
            FROM market_raw
            WINDOW symbol_day AS (PARTITION BY symbol, exchange, cik ORDER BY day)
        ),
        market_true_range AS (
            SELECT
                *,
                CASE
                    WHEN high IS NULL OR low IS NULL OR high <= 0 OR low <= 0 OR high < low THEN NULL
                    WHEN previous_close IS NULL OR previous_close <= 0 THEN high - low
                    ELSE GREATEST(high - low, abs(high - previous_close), abs(low - previous_close))
                END AS true_range
            FROM market_lag
        ),
        market_features AS (
            SELECT
                *,
                avg(close) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                    AS sma_20_calc,
                avg(close) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)
                    AS sma_50_calc,
                avg(close) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 199 PRECEDING AND CURRENT ROW)
                    AS sma_200_calc,
                avg(volume) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                    AS avg_volume_20,
                max(close) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 251 PRECEDING AND CURRENT ROW)
                    AS high_252,
                avg(true_range) OVER (PARTITION BY symbol, exchange, cik ORDER BY day ROWS BETWEEN 13 PRECEDING AND CURRENT ROW)
                    AS atr_14
            FROM market_true_range
        ),
        signal_trades AS (
            SELECT
                t.strategy_name,
                t.symbol,
                t.exchange,
                t.cik,
                t.signal_date,
                t.exit_date,
                extract(year FROM t.exit_date)::int AS exit_year,
                t.net_return_pct,
                t.holding_days,
                t.exit_reason,
                t.signal_score,
                t.quality_score,
                t.momentum_score,
                t.earnings_event_date,
                {ibkr_match}
                c.sector,
                c.industry,
                c.ibkr_category,
                c.ibkr_subcategory,
                mf.close,
                mf.market_cap,
                mf.average_daily_volume_3m,
                mf.avg_volume_20,
                mf.beta,
                CASE WHEN mf.close_lag_5 > 0 THEN mf.close / mf.close_lag_5 - 1.0 END AS ret_5,
                CASE WHEN mf.close_lag_20 > 0 THEN mf.close / mf.close_lag_20 - 1.0 END AS ret_20,
                CASE WHEN mf.close_lag_63 > 0 THEN mf.close / mf.close_lag_63 - 1.0 END AS ret_63,
                CASE WHEN mf.close_lag_126 > 0 THEN mf.close / mf.close_lag_126 - 1.0 END AS ret_126,
                CASE WHEN mf.close_lag_252 > 0 THEN mf.close / mf.close_lag_252 - 1.0 END AS ret_252,
                CASE
                    WHEN COALESCE(mf.fifty_day_average, mf.sma_50_calc) > 0
                    THEN mf.close / COALESCE(mf.fifty_day_average, mf.sma_50_calc) - 1.0
                END AS distance_sma50,
                CASE
                    WHEN COALESCE(mf.two_hundred_day_average, mf.sma_200_calc) > 0
                    THEN mf.close / COALESCE(mf.two_hundred_day_average, mf.sma_200_calc) - 1.0
                END AS distance_sma200,
                CASE WHEN mf.high_252 > 0 THEN mf.close / mf.high_252 - 1.0 END AS distance_252d_high,
                CASE WHEN mf.close > 0 THEN mf.atr_14 / mf.close END AS atr14_pct
            FROM {trades} t
            LEFT JOIN market_features mf
              ON mf.symbol = t.symbol
             AND mf.exchange = t.exchange
             AND mf.cik = t.cik
             AND mf.day = t.signal_date
            LEFT JOIN {core} c
              ON c.symbol = t.symbol
             AND c.exchange = t.exchange
             AND c.cik = t.cik
            WHERE t.run_id = (SELECT run_id FROM selected_run)
        ),
        feature_buckets AS (
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'sector' AS feature_name,
                   COALESCE(NULLIF(sector, ''), 'unknown') AS feature_bucket
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'industry',
                   COALESCE(NULLIF(industry, ''), 'unknown')
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ibkr_category',
                   COALESCE(NULLIF(ibkr_category, ''), 'unknown')
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ibkr_subcategory',
                   COALESCE(NULLIF(ibkr_subcategory, ''), 'unknown')
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ibkr_symbol_match',
                   ibkr_symbol_match
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'market_cap',
                   CASE
                       WHEN market_cap IS NULL THEN '00_unknown'
                       WHEN market_cap < 1000000000 THEN '01_under_1b'
                       WHEN market_cap < 5000000000 THEN '02_1b_to_5b'
                       WHEN market_cap < 20000000000 THEN '03_5b_to_20b'
                       WHEN market_cap < 100000000000 THEN '04_20b_to_100b'
                       ELSE '05_over_100b'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'price',
                   CASE
                       WHEN close IS NULL THEN '00_unknown'
                       WHEN close < 10 THEN '01_under_10'
                       WHEN close < 25 THEN '02_10_to_25'
                       WHEN close < 75 THEN '03_25_to_75'
                       WHEN close < 150 THEN '04_75_to_150'
                       ELSE '05_over_150'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'average_volume_20',
                   CASE
                       WHEN avg_volume_20 IS NULL THEN '00_unknown'
                       WHEN avg_volume_20 < 500000 THEN '01_under_500k'
                       WHEN avg_volume_20 < 1000000 THEN '02_500k_to_1m'
                       WHEN avg_volume_20 < 5000000 THEN '03_1m_to_5m'
                       WHEN avg_volume_20 < 20000000 THEN '04_5m_to_20m'
                       ELSE '05_over_20m'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'beta',
                   CASE
                       WHEN beta IS NULL THEN '00_unknown'
                       WHEN beta < 0.8 THEN '01_under_0_8'
                       WHEN beta < 1.1 THEN '02_0_8_to_1_1'
                       WHEN beta < 1.5 THEN '03_1_1_to_1_5'
                       WHEN beta < 2.0 THEN '04_1_5_to_2_0'
                       ELSE '05_over_2_0'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ret_20',
                   CASE
                       WHEN ret_20 IS NULL THEN '00_unknown'
                       WHEN ret_20 < -0.10 THEN '01_under_minus_10pct'
                       WHEN ret_20 < -0.03 THEN '02_minus_10_to_minus_3pct'
                       WHEN ret_20 < 0.03 THEN '03_minus_3_to_plus_3pct'
                       WHEN ret_20 < 0.10 THEN '04_plus_3_to_plus_10pct'
                       ELSE '05_over_plus_10pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ret_126',
                   CASE
                       WHEN ret_126 IS NULL THEN '00_unknown'
                       WHEN ret_126 < 0 THEN '01_under_0pct'
                       WHEN ret_126 < 0.10 THEN '02_0_to_10pct'
                       WHEN ret_126 < 0.25 THEN '03_10_to_25pct'
                       WHEN ret_126 < 0.50 THEN '04_25_to_50pct'
                       ELSE '05_over_50pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'ret_252',
                   CASE
                       WHEN ret_252 IS NULL THEN '00_unknown'
                       WHEN ret_252 < 0 THEN '01_under_0pct'
                       WHEN ret_252 < 0.20 THEN '02_0_to_20pct'
                       WHEN ret_252 < 0.50 THEN '03_20_to_50pct'
                       WHEN ret_252 < 1.00 THEN '04_50_to_100pct'
                       ELSE '05_over_100pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'atr14_pct',
                   CASE
                       WHEN atr14_pct IS NULL THEN '00_unknown'
                       WHEN atr14_pct < 0.02 THEN '01_under_2pct'
                       WHEN atr14_pct < 0.04 THEN '02_2_to_4pct'
                       WHEN atr14_pct < 0.07 THEN '03_4_to_7pct'
                       WHEN atr14_pct < 0.12 THEN '04_7_to_12pct'
                       ELSE '05_over_12pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'distance_sma50',
                   CASE
                       WHEN distance_sma50 IS NULL THEN '00_unknown'
                       WHEN distance_sma50 < -0.05 THEN '01_below_minus_5pct'
                       WHEN distance_sma50 < 0 THEN '02_minus_5_to_0pct'
                       WHEN distance_sma50 < 0.05 THEN '03_0_to_5pct'
                       WHEN distance_sma50 < 0.15 THEN '04_5_to_15pct'
                       ELSE '05_over_15pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'distance_sma200',
                   CASE
                       WHEN distance_sma200 IS NULL THEN '00_unknown'
                       WHEN distance_sma200 < 0 THEN '01_below_0pct'
                       WHEN distance_sma200 < 0.10 THEN '02_0_to_10pct'
                       WHEN distance_sma200 < 0.25 THEN '03_10_to_25pct'
                       WHEN distance_sma200 < 0.50 THEN '04_25_to_50pct'
                       ELSE '05_over_50pct'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'distance_252d_high',
                   CASE
                       WHEN distance_252d_high IS NULL THEN '00_unknown'
                       WHEN distance_252d_high < -0.40 THEN '01_below_minus_40pct'
                       WHEN distance_252d_high < -0.20 THEN '02_minus_40_to_minus_20pct'
                       WHEN distance_252d_high < -0.10 THEN '03_minus_20_to_minus_10pct'
                       WHEN distance_252d_high < -0.03 THEN '04_minus_10_to_minus_3pct'
                       ELSE '05_near_or_new_high'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'quality_score',
                   CASE
                       WHEN quality_score IS NULL THEN '00_unknown'
                       WHEN quality_score < 40 THEN '01_under_40'
                       WHEN quality_score < 55 THEN '02_40_to_55'
                       WHEN quality_score < 70 THEN '03_55_to_70'
                       WHEN quality_score < 85 THEN '04_70_to_85'
                       ELSE '05_over_85'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'momentum_score',
                   CASE
                       WHEN momentum_score IS NULL THEN '00_unknown'
                       WHEN momentum_score < 40 THEN '01_under_40'
                       WHEN momentum_score < 55 THEN '02_40_to_55'
                       WHEN momentum_score < 70 THEN '03_55_to_70'
                       WHEN momentum_score < 85 THEN '04_70_to_85'
                       ELSE '05_over_85'
                   END
            FROM signal_trades
            UNION ALL
            SELECT strategy_name, symbol, exit_year, net_return_pct, holding_days,
                   'has_earnings_event',
                   CASE WHEN earnings_event_date IS NULL THEN 'no_earnings_event' ELSE 'earnings_event' END
            FROM signal_trades
        )
        """
    ).format(
        runs=table_identifier(cfg.runs_table),
        trades=table_identifier(cfg.trades_table),
        market=table_identifier(cfg.market_daily_table),
        core=table_identifier(cfg.core_table),
        ibkr_match=ibkr_match,
    )


def feature_bucket_strength_query(cfg: BacktestConfig, has_ibkr_symbols: bool) -> sql.Composable:
    return sql.Composed(
        [
            signal_feature_ctes(cfg, has_ibkr_symbols),
            sql.SQL(
                """
                ,
                bucket_stats AS (
                    SELECT
                        strategy_name,
                        feature_name,
                        feature_bucket,
                        count(*) AS trades,
                        count(DISTINCT symbol) AS symbols,
                        count(DISTINCT exit_year) AS years,
                        round(avg(net_return_pct), 4) AS avg_net_return_pct,
                        round(percentile_cont(0.5) WITHIN GROUP (ORDER BY net_return_pct)::numeric, 4)
                            AS median_net_return_pct,
                        round(100.0 * count(*) FILTER (WHERE net_return_pct > 0)::numeric / nullif(count(*), 0), 2)
                            AS win_rate_pct,
                        round((sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                            / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0), 4)
                            AS profit_factor,
                        round(avg(holding_days), 2) AS avg_holding_days
                    FROM feature_buckets
                    GROUP BY strategy_name, feature_name, feature_bucket
                    HAVING count(*) >= %s
                ),
                ranked AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY strategy_name, feature_name
                            ORDER BY avg_net_return_pct DESC NULLS LAST, trades DESC
                        ) AS strongest_rank,
                        row_number() OVER (
                            PARTITION BY strategy_name, feature_name
                            ORDER BY avg_net_return_pct ASC NULLS LAST, trades DESC
                        ) AS weakest_rank
                    FROM bucket_stats
                )
                SELECT
                    'strongest' AS side,
                    strategy_name,
                    feature_name,
                    strongest_rank AS rank,
                    feature_bucket,
                    trades,
                    symbols,
                    years,
                    avg_net_return_pct,
                    median_net_return_pct,
                    win_rate_pct,
                    profit_factor,
                    avg_holding_days
                FROM ranked
                WHERE strongest_rank <= %s
                UNION ALL
                SELECT
                    'weakest' AS side,
                    strategy_name,
                    feature_name,
                    weakest_rank AS rank,
                    feature_bucket,
                    trades,
                    symbols,
                    years,
                    avg_net_return_pct,
                    median_net_return_pct,
                    win_rate_pct,
                    profit_factor,
                    avg_holding_days
                FROM ranked
                WHERE weakest_rank <= %s
                ORDER BY strategy_name, feature_name, side, rank
                """
            ),
        ]
    )


def feature_bucket_stability_query(cfg: BacktestConfig, has_ibkr_symbols: bool) -> sql.Composable:
    return sql.Composed(
        [
            signal_feature_ctes(cfg, has_ibkr_symbols),
            sql.SQL(
                """
                ,
                yearly_bucket_stats AS (
                    SELECT
                        strategy_name,
                        feature_name,
                        feature_bucket,
                        exit_year,
                        count(*) AS trades,
                        count(DISTINCT symbol) AS symbols,
                        avg(net_return_pct) AS avg_net_return_pct,
                        (sum(net_return_pct) FILTER (WHERE net_return_pct > 0))
                            / nullif(abs(sum(net_return_pct) FILTER (WHERE net_return_pct < 0)), 0)
                            AS profit_factor
                    FROM feature_buckets
                    GROUP BY strategy_name, feature_name, feature_bucket, exit_year
                    HAVING count(*) >= %s
                ),
                stable AS (
                    SELECT
                        strategy_name,
                        feature_name,
                        feature_bucket,
                        count(*) AS qualified_years,
                        sum(trades) AS trades,
                        round(avg(avg_net_return_pct)::numeric, 4) AS avg_yearly_return_pct,
                        round(min(avg_net_return_pct)::numeric, 4) AS worst_yearly_return_pct,
                        round(avg(profit_factor)::numeric, 4) AS avg_yearly_profit_factor,
                        count(*) FILTER (WHERE avg_net_return_pct > 0) AS positive_avg_years,
                        count(*) FILTER (WHERE profit_factor > 1.0) AS profit_factor_above_one_years
                    FROM yearly_bucket_stats
                    GROUP BY strategy_name, feature_name, feature_bucket
                    HAVING count(*) >= 3
                       AND sum(trades) >= %s
                ),
                ranked AS (
                    SELECT
                        *,
                        row_number() OVER (
                            PARTITION BY strategy_name, feature_name
                            ORDER BY
                                profit_factor_above_one_years DESC,
                                positive_avg_years DESC,
                                avg_yearly_return_pct DESC,
                                trades DESC
                        ) AS stability_rank
                    FROM stable
                )
                SELECT
                    strategy_name,
                    feature_name,
                    stability_rank AS rank,
                    feature_bucket,
                    qualified_years,
                    trades,
                    avg_yearly_return_pct,
                    worst_yearly_return_pct,
                    avg_yearly_profit_factor,
                    positive_avg_years,
                    profit_factor_above_one_years
                FROM ranked
                WHERE stability_rank <= %s
                ORDER BY strategy_name, feature_name, stability_rank
                """
            ),
        ]
    )


def run_query_set(
    conn: Any,
    run_id: int,
    diagnostic_run_id: int,
    cfg: BacktestConfig,
    diag: DiagnosticsConfig,
) -> None:
    has_ibkr_symbols = table_exists(conn, cfg.ibkr_symbols_table)
    queries: list[tuple[str, sql.Composable, tuple[Any, ...]]] = [
        ("strategy_edge", strategy_trade_edge_query(cfg), (run_id,)),
        ("symbol_breadth", symbol_breadth_query(cfg), (run_id,)),
        ("yearly_stability", yearly_stability_query(cfg), (run_id,)),
        ("exit_reasons", exit_reason_query(cfg), (run_id,)),
        ("exit_reason_yearly", exit_reason_yearly_query(cfg), (run_id,)),
        ("holding_period_buckets", holding_period_bucket_query(cfg), (run_id,)),
        ("top_bottom_symbols", top_bottom_symbols_query(cfg), (run_id, diag.top_n, diag.top_n)),
        (
            "feature_bucket_strength",
            feature_bucket_strength_query(cfg, has_ibkr_symbols),
            (run_id, diag.feature_lookback_days, diag.min_bucket_trades, diag.top_n, diag.top_n),
        ),
        (
            "feature_bucket_stability",
            feature_bucket_stability_query(cfg, has_ibkr_symbols),
            (run_id, diag.feature_lookback_days, diag.min_year_trades, diag.min_bucket_trades, diag.top_n),
        ),
    ]

    for table_name, query, params in queries:
        columns, rows = fetch_rows(conn, query, params)
        save_diagnostic_rows(conn, table_name, diagnostic_run_id, columns, rows)
        log.info("Diagnostics table persisted %s rows %d", table_name, len(rows))
        log_table_preview(table_name, columns, rows)


def run_diagnostics(cfg: BacktestConfig, diag: DiagnosticsConfig) -> int:
    diagnostic_run_id: int | None = None
    with connect(cfg) as conn:
        validate_diagnostic_tables(conn)
        run_id = diag.run_id if diag.run_id is not None else latest_run_id(conn, cfg)
        diagnostic_run_id = create_diagnostic_run(conn, run_id, diag)
        log.info("Swing stock diagnostics starting. Run id: %d. Diagnostic run id: %d.", run_id, diagnostic_run_id)
        try:
            columns, rows = fetch_rows(conn, run_overview_query(cfg), (run_id,))
            log_table_preview("source_run_overview", columns, rows)
            run_query_set(conn, run_id, diagnostic_run_id, cfg, diag)
        except Exception as exc:
            finish_diagnostic_run(conn, diagnostic_run_id, "error", str(exc)[:4000])
            raise
        finish_diagnostic_run(conn, diagnostic_run_id, "ok")
        log.info("Swing stock diagnostics done. Run id: %d. Diagnostic run id: %d.", run_id, diagnostic_run_id)
    return diagnostic_run_id


def main() -> None:
    cfg = load_config()
    configure_logging(cfg.log_level)
    diag = load_diagnostics_config(cfg)
    run_diagnostics(cfg, diag)


if __name__ == "__main__":
    main()
