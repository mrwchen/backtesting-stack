"""Schema validation and range-analysis persistence."""

from __future__ import annotations

import csv
from datetime import datetime, time, timezone
from io import StringIO
import logging
import math
from uuid import UUID

import psycopg2
from psycopg2 import sql

from .config import AnalysisConfig
from .data import ns_to_datetime
from .daily import DailySessionStat
from .events import CrossingEvents

log = logging.getLogger(__name__)

RESULT_COLUMNS = (
    "cross_ts",
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
    "event_tick_index",
    "bar_start_ts",
    "direction_code",
    "direction",
    "previous_mid",
    "signal_mid",
    "q50_level",
    "profile_low",
    "profile_high",
    "profile_range_points",
    "range_to_price_pct",
    "range_to_price_bps",
)

DAILY_RESULT_COLUMNS = (
    "day_start_ts",
    "analysis_id",
    "created_at",
    "symbol",
    "source_table",
    "source_start_ts",
    "source_end_ts",
    "data_start_ts",
    "data_end_ts",
    "bar_seconds",
    "price_step",
    "lookback_bars",
    "min_lookback_bars",
    "profile_max_lookback_seconds",
    "session_timezone",
    "session_sort_order",
    "session_label",
    "session_start_local_time",
    "session_end_local_time",
    "crossings_total",
    "day_first_cross_ts",
    "day_last_cross_ts",
    "min_range_to_price_bps",
    "avg_range_to_price_bps",
    "median_range_to_price_bps",
    "p75_range_to_price_bps",
    "p95_range_to_price_bps",
    "max_range_to_price_bps",
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
    _validate_table_columns(
        conn,
        cfg.result_schema,
        cfg.daily_result_table,
        set(DAILY_RESULT_COLUMNS),
        "daily result",
    )
    log.info(
        "Validated source table %s result table %s.%s daily table %s.%s",
        cfg.source_table,
        cfg.result_schema,
        cfg.result_table,
        cfg.result_schema,
        cfg.daily_result_table,
    )


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


def copy_crossing_events(
    conn: psycopg2.extensions.connection,
    cfg: AnalysisConfig,
    analysis_id: UUID,
    created_at: datetime,
    data_start_ts: datetime,
    data_end_ts: datetime,
    ticks_loaded: int,
    bars_loaded: int,
    lookback_bars: int,
    min_lookback_bars: int,
    events: CrossingEvents,
) -> int:
    if len(events) == 0:
        return 0

    copy_sql = sql.SQL("COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
        table=sql.Identifier(cfg.result_schema, cfg.result_table),
        columns=sql.SQL(", ").join(sql.Identifier(col) for col in RESULT_COLUMNS),
    )

    total = 0
    for start in range(0, len(events), cfg.copy_batch_rows):
        end = min(start + cfg.copy_batch_rows, len(events))
        buffer = StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        for idx in range(start, end):
            direction_code = int(events.direction_code[idx])
            writer.writerow(
                (
                    _pg_value(ns_to_datetime(int(events.cross_ts_ns[idx]))),
                    str(analysis_id),
                    _pg_value(created_at),
                    cfg.symbol,
                    cfg.source_table,
                    _pg_value(cfg.start_ts_utc),
                    _pg_value(cfg.end_ts_utc),
                    _pg_value(data_start_ts),
                    _pg_value(data_end_ts),
                    int(ticks_loaded),
                    int(bars_loaded),
                    int(cfg.bar_seconds),
                    _pg_value(float(cfg.price_step)),
                    int(lookback_bars),
                    int(min_lookback_bars),
                    _pg_value(cfg.profile_max_lookback_seconds),
                    int(events.tick_index[idx]),
                    _pg_value(ns_to_datetime(int(events.bar_start_ns[idx]))),
                    direction_code,
                    "UP" if direction_code > 0 else "DOWN",
                    _pg_value(float(events.previous_mid[idx])),
                    _pg_value(float(events.signal_mid[idx])),
                    _pg_value(float(events.q50_level[idx])),
                    _pg_value(float(events.profile_low[idx])),
                    _pg_value(float(events.profile_high[idx])),
                    _pg_value(float(events.profile_range_points[idx])),
                    _pg_value(float(events.range_to_price_pct[idx])),
                    _pg_value(float(events.range_to_price_bps[idx])),
                )
            )
        buffer.seek(0)
        with conn.cursor() as cur:
            cur.copy_expert(copy_sql.as_string(conn), buffer)
        total += end - start
    return total


def copy_daily_session_stats(
    conn: psycopg2.extensions.connection,
    cfg: AnalysisConfig,
    analysis_id: UUID,
    created_at: datetime,
    data_start_ts: datetime,
    data_end_ts: datetime,
    lookback_bars: int,
    min_lookback_bars: int,
    stats: list[DailySessionStat],
) -> int:
    if not stats:
        return 0

    copy_sql = sql.SQL("COPY {table} ({columns}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
        table=sql.Identifier(cfg.result_schema, cfg.daily_result_table),
        columns=sql.SQL(", ").join(sql.Identifier(col) for col in DAILY_RESULT_COLUMNS),
    )

    buffer = StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for row in stats:
        writer.writerow(
            (
                _pg_value(row.day_start_ts),
                str(analysis_id),
                _pg_value(created_at),
                cfg.symbol,
                cfg.source_table,
                _pg_value(cfg.start_ts_utc),
                _pg_value(cfg.end_ts_utc),
                _pg_value(data_start_ts),
                _pg_value(data_end_ts),
                int(cfg.bar_seconds),
                _pg_value(float(cfg.price_step)),
                int(lookback_bars),
                int(min_lookback_bars),
                _pg_value(cfg.profile_max_lookback_seconds),
                "America/New_York",
                int(row.session_sort_order),
                row.session_label,
                _pg_value(row.session_start_local_time),
                _pg_value(row.session_end_local_time),
                int(row.crossings_total),
                _pg_value(row.day_first_cross_ts),
                _pg_value(row.day_last_cross_ts),
                _pg_value(row.min_range_to_price_bps),
                _pg_value(row.avg_range_to_price_bps),
                _pg_value(row.median_range_to_price_bps),
                _pg_value(row.p75_range_to_price_bps),
                _pg_value(row.p95_range_to_price_bps),
                _pg_value(row.max_range_to_price_bps),
            )
        )
    buffer.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(copy_sql.as_string(conn), buffer)
    return len(stats)


def _pg_value(value: object) -> object:
    if value is None:
        return r"\N"
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, time):
        return value.isoformat()
    if isinstance(value, float):
        if not math.isfinite(value):
            return r"\N"
        return f"{value:.10f}"
    return value
