"""Load Pepperstone ticks and derive 10-second mid-price bars."""

import logging
from datetime import datetime

import pandas as pd
import psycopg2
from psycopg2 import sql

from . import config

log = logging.getLogger(__name__)


def _source_table() -> sql.Composed:
    return sql.Identifier(*config.SOURCE_TABLE.split("."))


def load_ticks(conn: psycopg2.extensions.connection) -> pd.DataFrame:
    where = [sql.SQL("symbol = %s")]
    params: list[object] = [config.SYMBOL]
    if config.START_TS_UTC is not None:
        where.append(sql.SQL("tick_time >= %s"))
        params.append(config.START_TS_UTC)
    if config.END_TS_UTC is not None:
        where.append(sql.SQL("tick_time < %s"))
        params.append(config.END_TS_UTC)

    query = sql.SQL(
        "SELECT tick_time, bid, ask FROM {tbl} WHERE {where} ORDER BY tick_time"
    ).format(
        tbl=_source_table(),
        where=sql.SQL(" AND ").join(where),
    )

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    if not rows:
        raise RuntimeError(
            f"No ticks found in {config.SOURCE_TABLE} for symbol={config.SYMBOL!r} "
            f"start={config.START_TS_UTC} end={config.END_TS_UTC}"
        )

    df = pd.DataFrame(rows, columns=["tick_time", "bid", "ask"])
    df["tick_time"] = pd.to_datetime(df["tick_time"], utc=True)
    df["bid"] = df["bid"].astype(float)
    df["ask"] = df["ask"].astype(float)
    df = df[df["ask"] >= df["bid"]].copy()
    if df.empty:
        raise RuntimeError("All loaded ticks had ask < bid")
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["bar_start"] = df["tick_time"].dt.floor(f"{config.BAR_SECONDS}s")
    df = df.sort_values("tick_time").reset_index(drop=True)
    log.info(
        "Loaded ticks %d for %s from %s to %s",
        len(df), config.SYMBOL, _fmt_ts(df["tick_time"].iloc[0]), _fmt_ts(df["tick_time"].iloc[-1]),
    )
    return df


def build_mid_bars(ticks: pd.DataFrame) -> pd.DataFrame:
    bars = (
        ticks.groupby("bar_start", sort=True)
        .agg(
            open=("mid", "first"),
            high=("mid", "max"),
            low=("mid", "min"),
            close=("mid", "last"),
            tick_count=("mid", "size"),
        )
        .reset_index()
    )
    bars = bars.sort_values("bar_start").reset_index(drop=True)
    log.info(
        "Built %d mid-price bars of %ds from %d ticks",
        len(bars), config.BAR_SECONDS, len(ticks),
    )
    return bars


def _fmt_ts(value: datetime) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)

