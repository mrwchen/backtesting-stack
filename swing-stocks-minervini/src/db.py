"""Thin psycopg2 helpers: connection, bulk COPY writes, range deletes."""
from __future__ import annotations

import io
import logging
import os

import numpy as np
import pandas as pd
import psycopg2

log = logging.getLogger(__name__)

RESULT_SCHEMA_COLUMNS = {
    "backtesting_minervini_stage_state": {
        "stage", "model_version", "config_fingerprint", "input_fingerprint",
        "output_fingerprint", "start_date", "end_date",
    },
    "backtesting_minervini_screen_daily": {
        "period_end_date", "symbol", "screen_pass", "fundamental_score",
        "fundamental_coverage", "institutional_manager_count",
        "ibkr_industry_rs_rating",
    },
    "backtesting_minervini_rs_daily": {
        "period_end_date", "symbol", "rs_raw", "rs_rating",
    },
    "backtesting_minervini_market_daily": {
        "period_end_date", "market_status", "entry_exposure_cap",
    },
    "backtesting_minervini_setups": {
        "setup_id", "symbol", "setup_type", "detect_date", "pivot",
        "contraction_depths", "setup_score", "structure_quality_score",
        "prior_advance_pct", "valid_until", "price_continuity_segment",
    },
    "backtesting_minervini_runs": {
        "run_id", "model_version", "input_fingerprint", "params",
    },
    "backtesting_minervini_breakout_events": {
        "run_id", "setup_id", "setup_type", "breakout_date", "decision",
        "snapshot_date", "quality_score", "fill_probability",
        "slate_priority", "setup_age_sessions", "distance_to_pivot_pct",
        "quality_rank",
    },
    "backtesting_minervini_trades": {
        "run_id", "position_id", "setup_type", "entry_date", "exit_date",
    },
    "backtesting_minervini_equity_daily": {
        "run_id", "period_end_date", "equity", "entry_exposure_limit",
    },
}


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "timescaledb"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "market-data-account"),
        password=os.getenv("PGPASSWORD", ""),
        application_name="backtesting_minervini",
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "15")),
    )


def acquire_pipeline_lock(conn) -> None:
    """Serialize writers because functional result tables are not run-scoped."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT pg_try_advisory_lock(hashtext(%s)::bigint)",
            ("backtesting_minervini_pipeline",),
        )
        acquired = bool(cur.fetchone()[0])
    if not acquired:
        raise RuntimeError("another Minervini pipeline writer is already running")


def validate_result_schema(conn) -> None:
    """Fail before data loading when the incompatible result schema was not dropped."""
    tables = tuple(RESULT_SCHEMA_COLUMNS)
    actual = read_df(
        conn,
        """SELECT table_name, column_name
             FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = ANY (%s)""",
        (list(tables),),
    )
    actual_by_table = {
        table: set(actual.loc[actual["table_name"] == table, "column_name"])
        if not actual.empty else set()
        for table in tables
    }
    missing = [
        f"{table}.{column}"
        for table, columns in RESULT_SCHEMA_COLUMNS.items()
        for column in sorted(columns - actual_by_table[table])
    ]
    legacy_columns = {
        "backtesting_minervini_setups": {"vcp_score"},
        "backtesting_minervini_breakout_events": {
            "dynamic_setup_score", "readiness_score", "context_score",
            "candidate_rank",
        },
    }
    legacy = [
        f"{table}.{column}"
        for table, columns in legacy_columns.items()
        for column in sorted(columns & actual_by_table[table])
    ]
    if missing or legacy:
        detail = ",".join((missing + legacy)[:12])
        raise RuntimeError(
            "Minervini result schema is incompatible; run once with "
            f"DROP_ALL_MINERVINI_TABLES_ON_START=true; details {detail}"
        )


def read_df(conn, sql: str, params: dict | tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def copy_df(
    conn,
    df: pd.DataFrame,
    table: str,
    columns: list[str],
    *,
    commit: bool = True,
) -> int:
    """Bulk-insert via COPY; callers may compose several writes atomically."""
    if df.empty:
        return 0
    buf = io.StringIO()
    out = df.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    out.to_csv(buf, index=False, header=False, na_rep="", lineterminator="\n")
    buf.seek(0)
    col_list = ", ".join(columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')", buf
        )
    if commit:
        conn.commit()
    log.info("wrote %d rows into %s", len(df), table)
    return len(df)


def delete_range(conn, table: str, date_col: str, start, end) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {table} WHERE {date_col} BETWEEN %s AND %s", (start, end)
        )
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        log.info("deleted %d existing rows from %s (%s..%s)", deleted, table, start, end)
