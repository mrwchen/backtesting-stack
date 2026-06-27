"""Schema validation and range-analysis persistence."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from io import StringIO
import logging
import math
from uuid import UUID

import psycopg2
from psycopg2 import sql

from .config import AnalysisConfig
from .data import BarData, ns_to_datetime
from .profile import RangeProfileArrays

log = logging.getLogger(__name__)

RESULT_COLUMNS = (
    "bar_start_ts",
    "analysis_id",
    "created_at",
    "symbol",
    "source_table",
    "source_start_ts",
    "source_end_ts",
    "data_start_ts",
    "data_end_ts",
    "ticks_loaded",
    "bars_loaded",
    "bar_seconds",
    "price_step",
    "lookback_bars",
    "min_lookback_bars",
    "profile_max_lookback_seconds",
    "bar_open",
    "bar_high",
    "bar_low",
    "bar_close",
    "bar_tick_count",
    "profile_low",
    "profile_high",
    "profile_range_points",
)


def validate_schema(conn: psycopg2.extensions.connection, cfg: AnalysisConfig) -> None:
    source_schema, source_table = split_table_path(cfg.source_table)
    _validate_table_columns(
        conn,
        source_schema,
        source_table,
        {"symbol", "tick_time", "bid", "ask"},
        "source",
    )
    _validate_table_columns(
        conn,
        cfg.result_schema,
        cfg.result_table,
        set(RESULT_COLUMNS),
        "result",
    )
    log.info("Validated source table %s and result table %s.%s", cfg.source_table, cfg.result_schema, cfg.result_table)


def split_table_path(value: str) -> tuple[str, str]:
    parts = value.split(".")
    if len(parts) == 1:
        return "public", parts[0]
    return parts[0], parts[1]


def _validate_table_columns(
    conn: psycopg2.extensions.connection,
    schema_name: str,
    table_name: str,
    required: set[str],
    label: str,
) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"{schema_name}.{table_name}",))
        if cur.fetchone()[0] is None:
            raise RuntimeError(f"Missing {label} table {schema_name}.{table_name}")
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            """,
            (schema_name, table_name),
        )
        found = {row[0] for row in cur.fetchall()}
    missing = sorted(required - found)
    if missing:
        raise RuntimeError(f"Table {schema_name}.{table_name} is missing columns: {', '.join(missing)}")


def copy_profile_rows(
    conn: psycopg2.extensions.connection,
    cfg: AnalysisConfig,
    analysis_id: UUID,
    created_at: datetime,
    data_start_ts: datetime,
    data_end_ts: datetime,
    ticks_loaded: int,
    bars: BarData,
    lookback_bars: int,
    min_lookback_bars: int,
    profile: RangeProfileArrays,
) -> int:
    copy_sql = sql.SQL("COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
        table=sql.Identifier(cfg.result_schema, cfg.result_table),
        columns=sql.SQL(", ").join(sql.Identifier(col) for col in RESULT_COLUMNS),
    )

    total = 0
    for start in range(0, len(bars), cfg.copy_batch_rows):
        end = min(start + cfg.copy_batch_rows, len(bars))
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for idx in range(start, end):
            writer.writerow(
                (
                    _pg_value(ns_to_datetime(int(bars.bar_start_ns[idx]))),
                    str(analysis_id),
                    _pg_value(created_at),
                    cfg.symbol,
                    cfg.source_table,
                    _pg_value(cfg.start_ts_utc),
                    _pg_value(cfg.end_ts_utc),
                    _pg_value(data_start_ts),
                    _pg_value(data_end_ts),
                    int(ticks_loaded),
                    int(len(bars)),
                    int(cfg.bar_seconds),
                    _pg_value(float(cfg.price_step)),
                    int(lookback_bars),
                    int(min_lookback_bars),
                    _pg_value(cfg.profile_max_lookback_seconds),
                    _pg_value(float(bars.open[idx])),
                    _pg_value(float(bars.high[idx])),
                    _pg_value(float(bars.low[idx])),
                    _pg_value(float(bars.close[idx])),
                    int(bars.tick_count[idx]),
                    _pg_value(float(profile.profile_low[idx])),
                    _pg_value(float(profile.profile_high[idx])),
                    _pg_value(float(profile.profile_range_points[idx])),
                )
            )
        buffer.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql.as_string(conn), buffer)
        total += end - start
    return total


def _pg_value(value: object) -> object:
    if value is None:
        return r"\N"
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return r"\N"
        return f"{value:.10f}"
    return value
