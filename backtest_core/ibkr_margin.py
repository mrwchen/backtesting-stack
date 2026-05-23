"""IBKR symbol margin lookup backed by TimescaleDB source data."""

import logging
from dataclasses import dataclass
from datetime import datetime

import psycopg2
from psycopg2 import sql

from .config import IBKR_MARGIN_REQUIREMENTS_TABLE
from .sql_utils import relation_identifier

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IbkrMarginRequirement:
    source_symbol: str
    action: str
    quantity: float
    initial_margin_pct: float
    maintenance_margin_pct: float
    fetched_at: datetime


_IBKR_MARGIN_CACHE: dict[tuple[str, str, str], IbkrMarginRequirement] = {}
_IBKR_MARGIN_SYMBOL_CACHE: dict[tuple[str, str], tuple[str, ...]] = {}


def ibkr_action_for_direction(direction: str) -> str:
    normalized = direction.strip().upper()
    if normalized == "LONG":
        return "BUY"
    if normalized == "SHORT":
        return "SELL"
    raise ValueError(f"Unsupported direction for IBKR margin lookup: {direction!r}")


def get_ibkr_margin_symbols(
    conn: psycopg2.extensions.connection,
    action: str,
    margin_table: str = IBKR_MARGIN_REQUIREMENTS_TABLE,
) -> tuple[str, ...]:
    normalized_action = action.strip().upper()
    cache_key = (margin_table, normalized_action)
    cached = _IBKR_MARGIN_SYMBOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT UPPER(TRIM(source_symbol)) AS symbol_norm
                FROM {}
                WHERE UPPER(TRIM(action)) = %s
                  AND quantity > 0
                  AND initial_margin_pct > 0
                  AND maintenance_margin_pct > 0
                  AND source_symbol IS NOT NULL
                ORDER BY symbol_norm
                """
            ).format(relation_identifier(margin_table)),
            (normalized_action,),
        )
        symbols = tuple(row[0] for row in cur.fetchall() if row[0])

    _IBKR_MARGIN_SYMBOL_CACHE[cache_key] = symbols
    log.info(
        "Loaded IBKR margin eligible symbols table %s action %s count %d",
        margin_table,
        normalized_action,
        len(symbols),
    )
    return symbols


def get_ibkr_margin_requirement(
    conn: psycopg2.extensions.connection,
    symbol: str,
    direction: str,
    margin_table: str = IBKR_MARGIN_REQUIREMENTS_TABLE,
) -> IbkrMarginRequirement:
    action = ibkr_action_for_direction(direction)
    cache_key = (margin_table, symbol.strip().upper(), action)
    cached = _IBKR_MARGIN_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    source_symbol,
                    UPPER(TRIM(action)) AS action,
                    quantity,
                    initial_margin_pct,
                    maintenance_margin_pct,
                    fetched_at
                FROM {}
                WHERE UPPER(TRIM(source_symbol)) = UPPER(TRIM(%s))
                  AND UPPER(TRIM(action)) = %s
                  AND quantity > 0
                  AND initial_margin_pct > 0
                  AND maintenance_margin_pct > 0
                ORDER BY fetched_at DESC, quantity ASC
                LIMIT 1
                """
            ).format(relation_identifier(margin_table)),
            (symbol, action),
        )
        row = cur.fetchone()

    if row is None:
        raise RuntimeError(
            f"Missing usable IBKR margin percentage requirement for {symbol} {action} in {margin_table}"
        )

    quantity = float(row[2])
    requirement = IbkrMarginRequirement(
        source_symbol=row[0],
        action=row[1],
        quantity=quantity,
        initial_margin_pct=float(row[3]),
        maintenance_margin_pct=float(row[4]),
        fetched_at=row[5],
    )
    _IBKR_MARGIN_CACHE[cache_key] = requirement
    return requirement
