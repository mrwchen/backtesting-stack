"""Load Pepperstone ticks and derive mid-price bars."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
import logging

import numpy as np
import psycopg2
from psycopg2 import sql

from .config import AnalysisConfig

log = logging.getLogger(__name__)

NANOSECONDS_PER_SECOND = 1_000_000_000
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class RawTickData:
    tick_time_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    bar_start_ns: np.ndarray

    def __len__(self) -> int:
        return int(self.tick_time_ns.shape[0])


@dataclass(frozen=True)
class BarData:
    bar_start_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    tick_count: np.ndarray

    def __len__(self) -> int:
        return int(self.bar_start_ns.shape[0])


@dataclass(frozen=True)
class TickData:
    tick_time_ns: np.ndarray
    bid: np.ndarray
    ask: np.ndarray
    mid: np.ndarray
    bar_index: np.ndarray

    def __len__(self) -> int:
        return int(self.tick_time_ns.shape[0])


def datetime_to_ns(value: datetime) -> int:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    delta = value - _EPOCH
    return ((delta.days * 86_400 + delta.seconds) * NANOSECONDS_PER_SECOND) + delta.microseconds * 1000


def ns_to_datetime(value: int) -> datetime:
    seconds, nanos = divmod(int(value), NANOSECONDS_PER_SECOND)
    return datetime.fromtimestamp(seconds, timezone.utc).replace(microsecond=nanos // 1000)


def _source_table(source_table: str) -> sql.Composed:
    return sql.Identifier(*source_table.split("."))


def load_ticks(conn: psycopg2.extensions.connection, cfg: AnalysisConfig) -> RawTickData:
    where = [sql.SQL("symbol = %s")]
    params: list[object] = [cfg.symbol]
    if cfg.start_ts_utc is not None:
        where.append(sql.SQL("tick_time >= %s"))
        params.append(cfg.start_ts_utc)
    if cfg.end_ts_utc is not None:
        where.append(sql.SQL("tick_time < %s"))
        params.append(cfg.end_ts_utc)

    query = sql.SQL(
        "SELECT tick_time, bid, ask FROM {tbl} WHERE {where} ORDER BY tick_time"
    ).format(
        tbl=_source_table(cfg.source_table),
        where=sql.SQL(" AND ").join(where),
    )

    tick_time_buf = array("q")
    bid_buf = array("d")
    ask_buf = array("d")
    raw_rows = 0
    invalid_rows = 0
    fetch_size = 100_000

    old_autocommit = conn.autocommit
    if old_autocommit:
        conn.autocommit = False
    try:
        with conn.cursor(name="hfmed_range_tick_loader") as cur:
            cur.itersize = fetch_size
            cur.execute(query, params)
            while True:
                rows = cur.fetchmany(fetch_size)
                if not rows:
                    break
                raw_rows += len(rows)
                for tick_time, bid_value, ask_value in rows:
                    bid = float(bid_value)
                    ask = float(ask_value)
                    if ask < bid:
                        invalid_rows += 1
                        continue
                    tick_time_buf.append(datetime_to_ns(tick_time))
                    bid_buf.append(bid)
                    ask_buf.append(ask)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        if old_autocommit:
            conn.autocommit = True

    if not tick_time_buf:
        raise RuntimeError(
            f"No ticks found in {cfg.source_table} for symbol={cfg.symbol!r} "
            f"start={cfg.start_ts_utc} end={cfg.end_ts_utc}"
        )

    tick_time_ns = np.array(tick_time_buf, dtype=np.int64)
    bid = np.array(bid_buf, dtype=np.float64)
    ask = np.array(ask_buf, dtype=np.float64)
    if np.any(tick_time_ns[1:] < tick_time_ns[:-1]):
        order = np.argsort(tick_time_ns, kind="mergesort")
        tick_time_ns = tick_time_ns[order]
        bid = bid[order]
        ask = ask[order]

    mid = (bid + ask) / 2.0
    bar_ns = int(cfg.bar_seconds) * NANOSECONDS_PER_SECOND
    bar_start_ns = (tick_time_ns // bar_ns) * bar_ns

    log.info(
        "Loaded ticks %d raw_rows %d invalid_ask_lt_bid %d symbol %s from %s to %s",
        len(tick_time_ns),
        raw_rows,
        invalid_rows,
        cfg.symbol,
        ns_to_datetime(tick_time_ns[0]).isoformat(),
        ns_to_datetime(tick_time_ns[-1]).isoformat(),
    )
    return RawTickData(
        tick_time_ns=tick_time_ns,
        bid=bid,
        ask=ask,
        mid=mid,
        bar_start_ns=bar_start_ns,
    )


def build_mid_bars(raw_ticks: RawTickData, cfg: AnalysisConfig) -> tuple[TickData, BarData]:
    if len(raw_ticks) <= 0:
        raise RuntimeError("Cannot build bars without ticks")

    bar_start_ns, first_idx, counts = np.unique(
        raw_ticks.bar_start_ns,
        return_index=True,
        return_counts=True,
    )
    last_idx = first_idx + counts - 1
    bar_index = np.repeat(np.arange(len(bar_start_ns), dtype=np.int32), counts)
    bars = BarData(
        bar_start_ns=bar_start_ns.astype(np.int64, copy=False),
        open=raw_ticks.mid[first_idx].astype(np.float64, copy=False),
        high=np.maximum.reduceat(raw_ticks.mid, first_idx).astype(np.float64, copy=False),
        low=np.minimum.reduceat(raw_ticks.mid, first_idx).astype(np.float64, copy=False),
        close=raw_ticks.mid[last_idx].astype(np.float64, copy=False),
        tick_count=counts.astype(np.int32, copy=False),
    )
    ticks = TickData(
        tick_time_ns=raw_ticks.tick_time_ns,
        bid=raw_ticks.bid,
        ask=raw_ticks.ask,
        mid=raw_ticks.mid,
        bar_index=bar_index.astype(np.int32, copy=False),
    )
    log.info(
        "Built mid-price bars %d bar_seconds %d ticks %d",
        len(bars),
        cfg.bar_seconds,
        len(raw_ticks),
    )
    return ticks, bars
