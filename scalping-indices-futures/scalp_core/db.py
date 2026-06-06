"""Database connection (port of swing-stocks connect_with_retry pattern)."""

import logging
import time as _time

import psycopg2

from . import config

log = logging.getLogger(__name__)


def _configure_session(conn: psycopg2.extensions.connection) -> None:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('statement_timeout', %s, false)", (f"{config.DB_STATEMENT_TIMEOUT_MS}ms",))
        cur.execute("SELECT set_config('lock_timeout', %s, false)", (f"{config.DB_LOCK_TIMEOUT_MS}ms",))
        cur.execute(
            "SELECT set_config('idle_in_transaction_session_timeout', %s, false)",
            (f"{config.DB_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS}ms",),
        )


def connect_with_retry() -> psycopg2.extensions.connection:
    for attempt in range(1, config.DB_CONNECT_RETRIES + 1):
        try:
            conn = psycopg2.connect(**config.DB)
            _configure_session(conn)
            log.info("DB connected host %s db %s user %s", config.DB["host"], config.DB["dbname"], config.DB["user"])
            return conn
        except psycopg2.OperationalError as exc:
            if attempt == config.DB_CONNECT_RETRIES:
                raise
            delay = config.DB_CONNECT_RETRY_DELAY_SECONDS * (2 ** (attempt - 1))
            log.warning("DB connect failed (%d/%d, retry in %.0fs): %s", attempt, config.DB_CONNECT_RETRIES, delay, exc)
            _time.sleep(delay)
    raise RuntimeError("unreachable")
