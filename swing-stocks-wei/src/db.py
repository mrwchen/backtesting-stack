"""Thin psycopg2 helpers: connection and DataFrame reads."""
from __future__ import annotations

import os

import pandas as pd
import psycopg2
from psycopg2 import sql


def get_conn():
    return psycopg2.connect(
        host=os.getenv("PGHOST", "timescaledb"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE", "postgres"),
        user=os.getenv("PGUSER", "market-data-account"),
        password=os.getenv("PGPASSWORD", ""),
        application_name="backtest_wei_stocks",
        connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT_SECONDS", "15")),
    )


def read_df(conn, query: str | sql.Composable, params: dict | tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, params)
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)
