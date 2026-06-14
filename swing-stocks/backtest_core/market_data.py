"""Point-in-time price, broker-universe, regime, and bar-cache queries."""

import logging
import time as _time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2 import sql

from backtest_shared import Bar, CandidateRow, InstrumentKey, WorldRegime, instrument_key
from .config import *
from .ibkr_margin import get_ibkr_margin_symbols, ibkr_action_for_direction
from .sql_utils import relation_identifier

log = logging.getLogger(__name__)


@dataclass
class _BarCacheEntry:
    timestamps: list[datetime]
    bars: list[Bar]
    loaded_until_ts: Optional[datetime]


_BAR_CACHE: dict[InstrumentKey, _BarCacheEntry] = {}
_BAR_CACHE_DISABLED = False
_TRADING_DAYS_CACHE: dict[tuple[str, tuple[str, ...], date, date], list[date]] = {}
_WORLD_REGIME_CACHE: dict[tuple[str, Optional[date]], Optional[WorldRegime]] = {}
_CANDIDATE_CACHE: dict[tuple, list[CandidateRow]] = {}
_PEPPERSTONE_SYMBOL_CACHE: dict[tuple[str, bool, Optional[str]], tuple[str, ...]] = {}
_PEPPERSTONE_24_SYMBOL_CACHE: dict[str, frozenset[str]] = {}
_ENTRY_WINDOW_ZONE = ZoneInfo(ENTRY_WINDOW_TZ)
_SL_TP_WINDOW_ZONE = ZoneInfo(SL_TP_WINDOW_TZ)
_BAR_ESTIMATED_BYTES_PER_ROW = 512


def _get_pepperstone_symbols(
    conn: psycopg2.extensions.connection,
    pepperstone_table: str,
    required_currency: Optional[str] = None,
) -> tuple[str, ...]:
    required_currency_norm = required_currency.strip().upper() if required_currency else None
    cache_key = (pepperstone_table, PS_24_ENTRY_SL_TP_ACTIVE, required_currency_norm)
    cached = _PEPPERSTONE_SYMBOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tradable_symbol_filter = (
        "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL OR NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL"
        if PS_24_ENTRY_SL_TP_ACTIVE
        else "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL"
    )
    currency_filter = sql.SQL("")
    params: list[object] = []
    if required_currency_norm:
        currency_filter = sql.SQL("AND UPPER(TRIM(COALESCE(quote_asset, ''))) = %s")
        params.append(required_currency_norm)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT UPPER(TRIM(symbol::text)) AS symbol_norm
                FROM {}
                WHERE ({})
                  AND is_trading_enabled IS NOT FALSE
                  AND symbol IS NOT NULL
                  {}
                ORDER BY symbol_norm
                """
            ).format(
                relation_identifier(pepperstone_table),
                sql.SQL(tradable_symbol_filter),
                currency_filter,
            ),
            params,
        )
        symbols = tuple(row[0] for row in cur.fetchall() if row[0])

    _PEPPERSTONE_SYMBOL_CACHE[cache_key] = symbols
    log.info(
        "Loaded Pepperstone tradable symbols table %s count %d ps24 active %s currency %s",
        pepperstone_table,
        len(symbols),
        PS_24_ENTRY_SL_TP_ACTIVE,
        required_currency_norm or "-",
    )
    return symbols


def _get_pepperstone_24_symbols(
    conn: psycopg2.extensions.connection,
    pepperstone_table: str = PS_TRADABLE_SYMBOLS_TABLE,
) -> frozenset[str]:
    if ACCOUNT_PROFILE != "ps_acc" or not PS_24_ENTRY_SL_TP_ACTIVE:
        return frozenset()

    cached = _PEPPERSTONE_24_SYMBOL_CACHE.get(pepperstone_table)
    if cached is not None:
        return cached

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT UPPER(TRIM(symbol::text)) AS symbol_norm
                FROM {}
                WHERE NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL
                  AND is_trading_enabled IS NOT FALSE
                  AND symbol IS NOT NULL
                ORDER BY symbol_norm
                """
            ).format(relation_identifier(pepperstone_table)),
        )
        symbols = frozenset(row[0] for row in cur.fetchall() if row[0])

    _PEPPERSTONE_24_SYMBOL_CACHE[pepperstone_table] = symbols
    log.info("Loaded Pepperstone 24h symbols table %s count %d", pepperstone_table, len(symbols))
    return symbols


def _direct_candidate_symbols_for_account(
    conn: psycopg2.extensions.connection,
    symbols: tuple[str, ...],
    direction: str,
    pepperstone_table: str,
    required_currency: Optional[str],
    ibkr_margin_table: str,
) -> tuple[str, ...]:
    if not symbols:
        return ()
    required_currency_norm = required_currency.strip().upper() if required_currency else None

    if ACCOUNT_PROFILE == "ps_acc":
        allowed = set(_get_pepperstone_symbols(conn, pepperstone_table, required_currency_norm))
        return tuple(symbol for symbol in symbols if symbol in allowed)

    if ACCOUNT_PROFILE == "ibkr_acc":
        action = ibkr_action_for_direction(direction)
        currency_filter = sql.SQL("")
        params: list[object] = [list(symbols), action]
        if required_currency_norm:
            currency_filter = sql.SQL("AND UPPER(TRIM(COALESCE(currency, ''))) = %s")
            params.append(required_currency_norm)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT DISTINCT UPPER(TRIM(source_symbol)) AS symbol_norm
                    FROM {}
                    WHERE UPPER(TRIM(source_symbol)) = ANY(%s::text[])
                      AND UPPER(TRIM(action)) = %s
                      AND quantity > 0
                      AND initial_margin_pct > 0
                      AND maintenance_margin_pct > 0
                      {}
                    ORDER BY symbol_norm
                    """
                ).format(relation_identifier(ibkr_margin_table), currency_filter),
                params,
            )
            allowed = {row[0] for row in cur.fetchall() if row[0]}
        return tuple(symbol for symbol in symbols if symbol in allowed)

    return symbols


def _candidate_universe_symbols_for_account(
    conn: psycopg2.extensions.connection,
    direction: str,
    *,
    pepperstone_table: str,
    required_currency: Optional[str],
    ibkr_margin_table: str,
) -> tuple[str, ...]:
    if ACCOUNT_PROFILE == "ps_acc":
        return _get_pepperstone_symbols(conn, pepperstone_table, required_currency)
    if ACCOUNT_PROFILE == "ibkr_acc":
        action = ibkr_action_for_direction(direction)
        symbols = get_ibkr_margin_symbols(conn, action, ibkr_margin_table)
        if not required_currency:
            return symbols
        return _direct_candidate_symbols_for_account(
            conn,
            symbols,
            direction,
            pepperstone_table,
            required_currency,
            ibkr_margin_table,
        )
    return ()


def _candidate_cutoff_ts(as_of_date: Optional[date], as_of_ts: Optional[object]) -> Optional[datetime]:
    if as_of_ts is not None:
        if isinstance(as_of_ts, datetime):
            return _ensure_utc_ts(as_of_ts)
        raise TypeError(f"as_of_ts must be datetime or None, got {type(as_of_ts)!r}")
    if as_of_date is not None:
        return _day_close_ts(as_of_date)
    return None


def _latest_price_identities_for_symbols(
    conn: psycopg2.extensions.connection,
    symbols: tuple[str, ...],
    *,
    as_of_date: Optional[date],
    as_of_ts: Optional[object],
    source_table: str,
    broker_eligibility_bypassed: bool = False,
) -> list[CandidateRow]:
    normalized_symbols = tuple(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ))
    if not normalized_symbols:
        return []

    cutoff_ts = _candidate_cutoff_ts(as_of_date, as_of_ts)
    if cutoff_ts is not None:
        cutoff_ts = _last_complete_signal_bar_start_ts(cutoff_ts)
    cutoff_filter = sql.SQL("")
    params: list[object] = [list(normalized_symbols)]
    if cutoff_ts is not None:
        cutoff_filter = sql.SQL("AND b.ts <= %s")
        params.append(cutoff_ts)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH requested AS (
                    SELECT * FROM unnest(%s::text[]) AS u(symbol_norm)
                )
                SELECT
                    r.symbol_norm,
                    b.exchange,
                    b.cik
                FROM requested r
                JOIN LATERAL (
                    SELECT exchange, cik, ts, close
                    FROM {} b
                    WHERE b.symbol = r.symbol_norm
                      AND b.close > 0
                      {}
                    ORDER BY b.ts DESC
                    LIMIT 1
                ) b ON TRUE
                ORDER BY r.symbol_norm
                """
            ).format(relation_identifier(source_table), cutoff_filter),
            params,
        )
        by_symbol = {
            str(symbol).strip().upper(): CandidateRow(
                symbol=str(symbol).strip().upper(),
                exchange=str(exchange).strip().upper(),
                cik=int(cik),
                broker_eligibility_bypassed=broker_eligibility_bypassed,
            )
            for symbol, exchange, cik in cur.fetchall()
            if symbol and exchange and cik is not None
        }

    return [by_symbol[symbol] for symbol in normalized_symbols if symbol in by_symbol]


def get_direct_symbol_candidates(
    conn: psycopg2.extensions.connection,
    symbols: tuple[str, ...] | list[str],
    direction: str,
    *,
    as_of_ts: Optional[object],
    source_table: str = SOURCE_MARKET_DATA_1H_TABLE,
    pepperstone_table: str = PS_TRADABLE_SYMBOLS_TABLE,
    required_currency: Optional[str] = "USD",
    ibkr_margin_table: str = IBKR_SYMBOLS_TABLE,
    require_broker_eligibility: bool = True,
) -> list[CandidateRow]:
    normalized_symbols = tuple(dict.fromkeys(
        str(symbol).strip().upper()
        for symbol in symbols
        if str(symbol).strip()
    ))
    if not normalized_symbols:
        return []

    broker_symbols = (
        _direct_candidate_symbols_for_account(
            conn,
            normalized_symbols,
            direction,
            pepperstone_table,
            required_currency,
            ibkr_margin_table,
        )
        if require_broker_eligibility
        else normalized_symbols
    )
    return _latest_price_identities_for_symbols(
        conn,
        broker_symbols,
        as_of_date=None,
        as_of_ts=as_of_ts,
        source_table=source_table,
        broker_eligibility_bypassed=not require_broker_eligibility,
    )


def get_candidates(
    conn: psycopg2.extensions.connection,
    direction: str,
    *,
    source_table: str = SOURCE_MARKET_DATA_1H_TABLE,
    as_of_date: Optional[date],
    as_of_ts: Optional[object],
    pepperstone_table: str = PS_TRADABLE_SYMBOLS_TABLE,
    required_currency: Optional[str] = "USD",
    ibkr_margin_table: str = IBKR_SYMBOLS_TABLE,
) -> list[CandidateRow]:
    cutoff_ts = _candidate_cutoff_ts(as_of_date, as_of_ts)
    cache_key = (
        ACCOUNT_PROFILE,
        direction,
        source_table,
        cutoff_ts,
        pepperstone_table,
        required_currency,
        ibkr_margin_table,
        PS_24_ENTRY_SL_TP_ACTIVE,
    )
    cached = _CANDIDATE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    symbols = _candidate_universe_symbols_for_account(
        conn,
        direction,
        pepperstone_table=pepperstone_table,
        required_currency=required_currency,
        ibkr_margin_table=ibkr_margin_table,
    )
    candidates = _latest_price_identities_for_symbols(
        conn,
        symbols,
        as_of_date=as_of_date,
        as_of_ts=as_of_ts,
        source_table=source_table,
    )
    _CANDIDATE_CACHE[cache_key] = candidates
    return candidates


def get_world_regime(
    conn: psycopg2.extensions.connection,
    source_table: str = SOURCE_WORLD_REGIME_TABLE,
    as_of_date: Optional[date] = None,
) -> Optional[WorldRegime]:
    cache_key = (source_table, as_of_date)
    cached = _WORLD_REGIME_CACHE.get(cache_key)
    if cache_key in _WORLD_REGIME_CACHE:
        return cached

    day_filter = sql.SQL("")
    params: list[object] = []
    if as_of_date is not None:
        day_filter = sql.SQL("WHERE day <= %s")
        params.append(as_of_date)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    day,
                    regime_label,
                    composite_score,
                    dominant_shock_type,
                    max_shock_type_score,
                    defensive_risk_off_score,
                    energy_commodity_shock_score,
                    rates_inflation_usd_shock_score,
                    credit_banking_stress_score,
                    policy_geopolitical_score,
                    tech_stress_shock_score,
                    precious_metals_score,
                    industrial_metals_score,
                    metals_mining_shock_score,
                    metals_mining_subtype
                FROM {}
                {}
                ORDER BY day DESC
                LIMIT 1
                """
            ).format(relation_identifier(source_table), day_filter),
            params,
        )
        row = cur.fetchone()

    if row is None:
        _WORLD_REGIME_CACHE[cache_key] = None
        return None

    regime = WorldRegime(
        day=row[0],
        label=str(row[1] or "").strip().upper(),
        score=float(row[2]),
        dominant_shock_type=str(row[3] or ""),
        max_shock_type_score=float(row[4]) if row[4] is not None else None,
        defensive_risk_off_score=float(row[5]) if row[5] is not None else None,
        energy_commodity_shock_score=float(row[6]) if row[6] is not None else None,
        rates_inflation_usd_shock_score=float(row[7]) if row[7] is not None else None,
        credit_banking_stress_score=float(row[8]) if row[8] is not None else None,
        policy_geopolitical_score=float(row[9]) if row[9] is not None else None,
        tech_stress_shock_score=float(row[10]) if row[10] is not None else None,
        precious_metals_score=float(row[11]) if row[11] is not None else None,
        industrial_metals_score=float(row[12]) if row[12] is not None else None,
        metals_mining_shock_score=float(row[13]) if row[13] is not None else None,
        metals_mining_subtype=str(row[14] or ""),
    )
    _WORLD_REGIME_CACHE[cache_key] = regime
    return regime


def get_trading_days(conn: psycopg2.extensions.connection, start: date, end: date) -> list[date]:
    cache_key = (SOURCE_MARKET_DATA_1H_TABLE, TRADING_CALENDAR_SYMBOLS, start, end)
    cached = _TRADING_DAYS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if start > end:
        return []

    start_ts = datetime.combine(start - timedelta(days=3), time.min, tzinfo=timezone.utc)
    end_ts = datetime.combine(end + timedelta(days=4), time.min, tzinfo=timezone.utc)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT (ts AT TIME ZONE %s)::date AS trading_day
                FROM {}
                WHERE symbol = ANY(%s::text[])
                  AND ts >= %s
                  AND ts < %s
                ORDER BY trading_day
                """
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (ENTRY_WINDOW_TZ, list(TRADING_CALENDAR_SYMBOLS), start_ts, end_ts),
        )
        days = [row[0] for row in cur.fetchall() if start <= row[0] <= end]

    _TRADING_DAYS_CACHE[cache_key] = days
    log.info(
        "Loaded trading calendar table %s symbols %s days %d from %s to %s",
        SOURCE_MARKET_DATA_1H_TABLE,
        ",".join(TRADING_CALENDAR_SYMBOLS),
        len(days),
        start,
        end,
    )
    return days


def _local_day_start_ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=_ENTRY_WINDOW_ZONE).astimezone(timezone.utc)


def _local_day_end_exclusive_ts(d: date) -> datetime:
    return (datetime(d.year, d.month, d.day, tzinfo=_ENTRY_WINDOW_ZONE) + timedelta(days=1)).astimezone(timezone.utc)


def _day_close_ts(d: date) -> datetime:
    return _local_day_end_exclusive_ts(d)


def _ensure_utc_ts(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _entry_local_date(ts: datetime) -> date:
    return _ensure_utc_ts(ts).astimezone(_ENTRY_WINDOW_ZONE).date()


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = str(value).strip().split(":", 1)
    return int(hour), int(minute)


def _is_local_time_in_window(local: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    start_hour, start_minute = _parse_hhmm(start_hhmm)
    end_hour, end_minute = _parse_hhmm(end_hhmm)
    current = local.time()
    start_t = time(start_hour, start_minute)
    end_t = time(end_hour, end_minute)
    if start_t <= end_t:
        return start_t <= current <= end_t
    return current >= start_t or current <= end_t


def _is_in_entry_window(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    if not ENTRY_WINDOW_ENABLED:
        return True
    if ACCOUNT_PROFILE == "ps_acc" and PS_24_ENTRY_SL_TP_ACTIVE and conn is not None and identity is not None:
        if identity[0] in _get_pepperstone_24_symbols(conn):
            return True
    local = _ensure_utc_ts(ts).astimezone(_ENTRY_WINDOW_ZONE)
    return _is_local_time_in_window(local, ENTRY_WINDOW_START, ENTRY_WINDOW_END)


def _is_in_sl_tp_window(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    if ACCOUNT_PROFILE == "ps_acc" and PS_24_ENTRY_SL_TP_ACTIVE and conn is not None and identity is not None:
        if identity[0] in _get_pepperstone_24_symbols(conn):
            return True
    local = _ensure_utc_ts(ts).astimezone(_SL_TP_WINDOW_ZONE)
    return _is_local_time_in_window(local, SL_TP_WINDOW_START, SL_TP_WINDOW_END)


def _is_stop_loss_active(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    return _is_in_sl_tp_window(ts, conn, identity)


def _last_complete_signal_bar_start_ts(up_to_ts: datetime) -> datetime:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    return up_to_ts.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def signal_bar_close_decisions_for_day(
    conn: psycopg2.extensions.connection,
    day: date,
) -> list[tuple[datetime, datetime]]:
    start_ts = _local_day_start_ts(day)
    end_ts = _local_day_end_exclusive_ts(day)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT ts
                FROM {}
                WHERE symbol = ANY(%s::text[])
                  AND ts >= %s
                  AND ts < %s
                ORDER BY ts
                """
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (list(TRADING_CALENDAR_SYMBOLS), start_ts, end_ts),
        )
        signal_starts = [_ensure_utc_ts(row[0]) for row in cur.fetchall()]
    return [
        (signal_start, signal_start + timedelta(hours=1))
        for signal_start in signal_starts
        if _entry_local_date(signal_start + timedelta(hours=1)) == day
    ]


def _bar_cache_start_ts() -> datetime:
    return datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), time.min, tzinfo=timezone.utc)


def _chunked(values: list[InstrumentKey], size: int) -> list[list[InstrumentKey]]:
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def _bar_from_row(ts: datetime, open_: object, high: object, low: object, close: object, volume: object) -> Bar:
    return Bar(_ensure_utc_ts(ts), float(open_), float(high), float(low), float(close), int(volume))


def _bar_cache_counts() -> tuple[int, int, float]:
    rows = sum(len(entry.bars) for entry in _BAR_CACHE.values())
    estimated_mib = rows * _BAR_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
    return len(_BAR_CACHE), rows, estimated_mib


def _disable_bar_cache(reason: str) -> None:
    global _BAR_CACHE_DISABLED
    symbols, rows, mib = _bar_cache_counts()
    _BAR_CACHE.clear()
    _BAR_CACHE_DISABLED = True
    log.warning("Bar cache disabled %s cleared symbols %d rows %d estimated %.0f MiB", reason, symbols, rows, mib)


def _ensure_identity_bars_loaded(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> bool:
    global _BAR_CACHE_DISABLED
    if _BAR_CACHE_DISABLED:
        return False
    up_to_ts = _ensure_utc_ts(up_to_ts)
    identity = instrument_key(*identity)
    entry = _BAR_CACHE.get(identity)
    lower_bound: Optional[datetime] = None
    if entry is not None and entry.loaded_until_ts is not None and entry.loaded_until_ts >= up_to_ts:
        return True
    if entry is None:
        entry = _BarCacheEntry([], [], None)
        _BAR_CACHE[identity] = entry
        lower_bound = _bar_cache_start_ts()
    elif entry.loaded_until_ts is not None:
        lower_bound = entry.loaded_until_ts

    where = [
        sql.SQL("symbol = %s"),
        sql.SQL("exchange = %s"),
        sql.SQL("cik = %s"),
        sql.SQL("ts <= %s"),
    ]
    params: list[object] = [identity[0], identity[1], identity[2], up_to_ts]
    if lower_bound is not None:
        op = ">=" if not entry.timestamps else ">"
        where.append(sql.SQL(f"ts {op} %s"))
        params.append(lower_bound)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT ts, open, high, low, close, volume
                FROM {}
                WHERE {}
                ORDER BY ts
                """
            ).format(
                relation_identifier(SOURCE_MARKET_DATA_1H_TABLE),
                sql.SQL(" AND ").join(where),
            ),
            params,
        )
        for ts, open_, high, low, close, volume in cur.fetchall():
            bar = _bar_from_row(ts, open_, high, low, close, volume)
            if entry.timestamps and bar.ts <= entry.timestamps[-1]:
                continue
            entry.timestamps.append(bar.ts)
            entry.bars.append(bar)

    entry.loaded_until_ts = up_to_ts
    if _bar_cache_counts()[2] > BAR_CACHE_MAX_MIB:
        _disable_bar_cache(f"memory budget {BAR_CACHE_MAX_MIB} MiB exceeded")
        return False
    return True


def _load_identity_bars_direct(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> tuple[list[datetime], list[Bar]]:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT ts, open, high, low, close, volume
                FROM {}
                WHERE symbol = %s
                  AND exchange = %s
                  AND cik = %s
                  AND ts >= %s
                  AND ts <= %s
                ORDER BY ts
                """
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (identity[0], identity[1], identity[2], _bar_cache_start_ts(), up_to_ts),
        )
        bars = [_bar_from_row(ts, open_, high, low, close, volume) for ts, open_, high, low, close, volume in cur.fetchall()]
    return [bar.ts for bar in bars], bars


def _load_identity_bars_through(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> tuple[list[datetime], list[Bar]]:
    identity = instrument_key(*identity)
    if not _BAR_CACHE_DISABLED and _ensure_identity_bars_loaded(conn, identity, up_to_ts):
        entry = _BAR_CACHE[identity]
        return entry.timestamps, entry.bars
    return _load_identity_bars_direct(conn, identity, up_to_ts)


def preload_identity_bars(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> None:
    unique = sorted({instrument_key(*identity) for identity in identities})
    started = _time.perf_counter()
    batch_size = max(1, int(batch_size))
    batches = _chunked(unique, batch_size)
    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        for identity in batch:
            _ensure_identity_bars_loaded(conn, identity, up_to_ts)
        if log_batches:
            log.info(
                "Preloaded bar batch %d/%d identities %d through %s in %.1f s",
                batch_idx,
                len(batches),
                len(batch),
                up_to_ts,
                _time.perf_counter() - batch_started,
            )
    if unique:
        symbols, rows, mib = _bar_cache_counts()
        log.info(
            "Preloaded bars identities %d through %s in %.1f s cache symbols %d rows %d estimated %.0f MiB",
            len(unique),
            up_to_ts,
            _time.perf_counter() - started,
            symbols,
            rows,
            mib,
        )


def load_recent_bars_for_identities(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> dict[InstrumentKey, list[Bar]]:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    complete_upper_ts = _last_complete_signal_bar_start_ts(up_to_ts)
    unique = sorted({instrument_key(*identity) for identity in identities})
    bars_by_identity: dict[InstrumentKey, list[Bar]] = {identity: [] for identity in unique}
    if not unique or limit <= 0:
        return bars_by_identity

    batches = _chunked(unique, batch_size)
    total_rows = 0
    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        symbols = [identity[0] for identity in batch]
        exchanges = [identity[1] for identity in batch]
        ciks = [identity[2] for identity in batch]
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    WITH requested AS (
                        SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[]) AS u(symbol, exchange, cik)
                    )
                    SELECT r.symbol, r.exchange, r.cik, b.ts, b.open, b.high, b.low, b.close, b.volume
                    FROM requested r
                    JOIN LATERAL (
                        SELECT ts, open, high, low, close, volume
                        FROM {} b
                        WHERE b.symbol = r.symbol
                          AND b.exchange = r.exchange
                          AND b.cik = r.cik
                          AND b.ts <= %s
                        ORDER BY b.ts DESC
                        LIMIT %s
                    ) b ON TRUE
                    ORDER BY r.symbol, r.exchange, r.cik, b.ts
                    """
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (symbols, exchanges, ciks, complete_upper_ts, limit),
            )
            rows = cur.fetchall()

        for symbol, exchange, cik, ts, open_, high, low, close, volume in rows:
            identity = instrument_key(symbol, exchange, cik)
            bars_by_identity[identity].append(_bar_from_row(ts, open_, high, low, close, volume))
        total_rows += len(rows)
        if log_batches:
            log.info(
                "Recent bar batch %d/%d loaded identities %d rows %d in %.1f s",
                batch_idx,
                len(batches),
                len(batch),
                len(rows),
                _time.perf_counter() - batch_started,
            )
    if log_batches:
        log.info(
            "Recent bar load complete identities %d rows %d through %s",
            len(unique),
            total_rows,
            complete_upper_ts,
        )
    return bars_by_identity


def get_cached_bars(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    limit: int,
    up_to_ts: datetime,
) -> list[Bar]:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    timestamps, bars = _load_identity_bars_through(conn, identity, up_to_ts)
    end_idx = bisect_right(timestamps, _last_complete_signal_bar_start_ts(up_to_ts))
    selected: list[Bar] = []
    bar_idx = end_idx - 1
    while bar_idx >= 0 and len(selected) < limit:
        selected.append(bars[bar_idx])
        bar_idx -= 1
    selected.reverse()
    return selected


def get_bars_range(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    after_ts: datetime,
    up_to_date: date,
) -> list:
    return get_bars_range_through(conn, identity, after_ts, _day_close_ts(up_to_date))


def get_bars_range_through(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    after_ts: datetime,
    up_to_ts: datetime,
) -> list:
    after_ts = _ensure_utc_ts(after_ts)
    up_to_ts = _ensure_utc_ts(up_to_ts)
    timestamps, bars = _load_identity_bars_through(conn, identity, up_to_ts)
    start_idx = bisect_right(timestamps, after_ts)
    end_idx = bisect_left(timestamps, up_to_ts)
    return [(bars[i].ts, bars[i].open, bars[i].high, bars[i].low, bars[i].close) for i in range(start_idx, end_idx)]


def load_next_bar_opens(
    conn: psycopg2.extensions.connection,
    requests: list[tuple[InstrumentKey, datetime]],
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
) -> dict[tuple[InstrumentKey, datetime], tuple[datetime, float]]:
    if not requests:
        return {}

    unique_requests = sorted({
        (instrument_key(*identity), _ensure_utc_ts(after_ts))
        for identity, after_ts in requests
    })
    results: dict[tuple[InstrumentKey, datetime], tuple[datetime, float]] = {}

    for batch in [unique_requests[idx:idx + batch_size] for idx in range(0, len(unique_requests), batch_size)]:
        symbols = [identity[0] for identity, _ in batch]
        exchanges = [identity[1] for identity, _ in batch]
        ciks = [identity[2] for identity, _ in batch]
        after_timestamps = [after_ts for _, after_ts in batch]
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    WITH requested AS (
                        SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[], %s::timestamptz[])
                            AS u(symbol, exchange, cik, after_ts)
                    )
                    SELECT r.symbol, r.exchange, r.cik, r.after_ts, b.ts, b.open
                    FROM requested r
                    LEFT JOIN LATERAL (
                        SELECT b.ts, b.open
                        FROM {} b
                        WHERE b.symbol = r.symbol
                          AND b.exchange = r.exchange
                          AND b.cik = r.cik
                          AND b.ts > r.after_ts
                        ORDER BY b.ts
                        LIMIT 1
                    ) b ON TRUE
                    """
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (symbols, exchanges, ciks, after_timestamps),
            )
            for symbol, exchange, cik, after_ts, next_ts, open_ in cur.fetchall():
                if next_ts is None:
                    continue
                key = (instrument_key(symbol, exchange, cik), _ensure_utc_ts(after_ts))
                results[key] = (_ensure_utc_ts(next_ts), float(open_))

    return results


def log_cache_stats(context: str = "current") -> None:
    symbols, rows, mib = _bar_cache_counts()
    log.info(
        "Cache stats context %s bar symbols %d rows %d estimated %.0f MiB trading day sets %d regimes %d candidate sets %d candidate rows %d",
        context,
        symbols,
        rows,
        mib,
        len(_TRADING_DAYS_CACHE),
        len(_WORLD_REGIME_CACHE),
        len(_CANDIDATE_CACHE),
        sum(len(candidates) for candidates in _CANDIDATE_CACHE.values()),
    )


def clear_market_data_caches(context: str = "after_run") -> None:
    global _BAR_CACHE_DISABLED
    symbols, rows, mib = _bar_cache_counts()
    log.info(
        "Cache cleanup context %s clearing bar symbols %d rows %d estimated %.0f MiB trading day sets %d regimes %d candidate sets %d",
        context,
        symbols,
        rows,
        mib,
        len(_TRADING_DAYS_CACHE),
        len(_WORLD_REGIME_CACHE),
        len(_CANDIDATE_CACHE),
    )
    _BAR_CACHE.clear()
    _BAR_CACHE_DISABLED = False
    _TRADING_DAYS_CACHE.clear()
    _WORLD_REGIME_CACHE.clear()
    _CANDIDATE_CACHE.clear()
    _PEPPERSTONE_SYMBOL_CACHE.clear()
    _PEPPERSTONE_24_SYMBOL_CACHE.clear()
