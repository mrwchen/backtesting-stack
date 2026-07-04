"""Small psycopg2 helpers for validation, reads and COPY writes."""
from __future__ import annotations

import io
import logging
import os

import numpy as np
import pandas as pd
import psycopg2

log = logging.getLogger(__name__)


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "timescaledb"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "market-data-account"),
        password=os.getenv("PGPASSWORD", ""),
        application_name=os.getenv("PGAPPNAME", "backtest_wei_runner"),
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "15")),
    )


def validate_tables(conn, table_names: tuple[str, ...]) -> None:
    missing: list[str] = []
    with conn.cursor() as cur:
        for table_name in table_names:
            cur.execute("SELECT to_regclass(%s)", (f"public.{table_name}",))
            if cur.fetchone()[0] is None:
                missing.append(table_name)
    if missing:
        raise RuntimeError(f"missing required tables: {', '.join(missing)}")


def read_df(conn, sql: str, params: dict | tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def copy_df(conn, df: pd.DataFrame, table: str, columns: list[str]) -> int:
    if df.empty:
        return 0
    buf = io.StringIO()
    out = df.loc[:, columns].replace([np.inf, -np.inf], np.nan)
    out.to_csv(buf, index=False, header=False, na_rep="", lineterminator="\n")
    buf.seek(0)
    col_list = ", ".join(columns)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY {table} ({col_list}) FROM STDIN WITH (FORMAT csv, NULL '')",
            buf,
        )
    conn.commit()
    log.info("wrote %d rows into %s", len(df), table)
    return len(df)


def delete_date_range(conn, table: str, date_col: str, start, end) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {table} WHERE {date_col} BETWEEN %s AND %s",
            (start, end),
        )
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        log.info("deleted %d rows from %s between %s and %s", deleted, table, start, end)
