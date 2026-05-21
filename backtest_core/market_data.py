"""Point-in-time source queries, trading calendar, and bar cache."""

import logging
import time as _time
from array import array
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2 import sql

from backtest_shared import Bar, FundamentalRow, InstrumentKey, WorldRegime, instrument_key
from .config import *
from .ibkr_margin import ibkr_action_for_direction
from .sql_utils import relation_identifier

log = logging.getLogger(__name__)


@dataclass
class _BarCacheEntry:
    timestamps: list[datetime]
    bars: list[Bar]
    loaded_until_ts: Optional[datetime]


@dataclass
class _SignalBarCacheEntry:
    ts_epoch_us: array
    opens: array
    highs: array
    lows: array
    closes: array
    volumes: array
    loaded_until_ts: Optional[datetime]


_BAR_CACHE: dict[InstrumentKey, _BarCacheEntry] = {}
_SIGNAL_BAR_CACHE: dict[InstrumentKey, _SignalBarCacheEntry] = {}
_SIGNAL_BAR_CACHE_DISABLED = False
_TRADING_DAYS_CACHE: dict[tuple[str, date, date], list[date]] = {}
_WORLD_REGIME_CACHE: dict[tuple[str, Optional[date]], Optional[WorldRegime]] = {}
_CANDIDATE_CACHE: dict[tuple, list[FundamentalRow]] = {}
_ENTRY_WINDOW_ZONE = ZoneInfo(ENTRY_WINDOW_TZ)
_SL_TP_WINDOW_ZONE = ZoneInfo(SL_TP_WINDOW_TZ)
_STOP_LOSS_RTH_ZONE = ZoneInfo(STOP_LOSS_RTH_TZ)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SIGNAL_BAR_ESTIMATED_BYTES_PER_ROW = 80

def _cache_counts() -> tuple[int, int, int, int, int]:
    cached_bars = sum(len(entry.bars) for entry in _BAR_CACHE.values())
    return (
        len(_BAR_CACHE),
        cached_bars,
        len(_TRADING_DAYS_CACHE),
        len(_WORLD_REGIME_CACHE),
        len(_CANDIDATE_CACHE),
    )


def _signal_cache_counts() -> tuple[int, int, float]:
    signal_bars = sum(len(entry.ts_epoch_us) for entry in _SIGNAL_BAR_CACHE.values())
    estimated_mib = signal_bars * _SIGNAL_BAR_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
    return len(_SIGNAL_BAR_CACHE), signal_bars, estimated_mib


def _default_as_of_ts(as_of_date: date) -> datetime:
    return datetime.combine(as_of_date, time.max, tzinfo=timezone.utc)


def get_world_regime(
    conn: psycopg2.extensions.connection,
    source_table: str = "world_regime_daily_scores_mv",
    as_of_date: Optional[date] = None,
) -> Optional[WorldRegime]:
    cache_key = (source_table, as_of_date)
    if cache_key in _WORLD_REGIME_CACHE:
        return _WORLD_REGIME_CACHE[cache_key]

    if as_of_date:
        query = sql.SQL(
            "SELECT day, regime_label, composite_score FROM {} "
            "WHERE composite_score IS NOT NULL AND day <= %s ORDER BY day DESC LIMIT 1"
        ).format(relation_identifier(source_table))
        params = (as_of_date,)
    else:
        query = sql.SQL(
            "SELECT day, regime_label, composite_score FROM {} "
            "WHERE composite_score IS NOT NULL ORDER BY day DESC LIMIT 1"
        ).format(relation_identifier(source_table))
        params = ()

    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    if not row:
        _WORLD_REGIME_CACHE[cache_key] = None
        return None
    regime = WorldRegime(day=row[0], label=row[1], score=float(row[2]))
    _WORLD_REGIME_CACHE[cache_key] = regime
    return regime


def get_candidates(
    conn: psycopg2.extensions.connection,
    direction: str,
    long_min_fundamental: float,
    short_max_fundamental: float,
    min_market_cap_m: float = 0.0,
    source_table: str = "stocks_analysis_fundamental_scores",
    as_of_date: Optional[date] = None,
    as_of_ts: Optional[object] = None,
    long_label_blocklist: Optional[list] = None,
    short_label_blocklist: Optional[list] = None,
    pepperstone_table: str = "public.pepperstone_data",
    required_currency: Optional[str] = "USD",
    allow_rebuilt_historical_fundamentals: bool = False,
    filter_negative_earnings: bool = False,
    ibkr_margin_table: str = IBKR_MARGIN_REQUIREMENTS_TABLE,
) -> list[FundamentalRow]:
    if allow_rebuilt_historical_fundamentals:
        raise ValueError(
            "allow_rebuilt_historical_fundamentals=True is disabled; candidate queries must stay point-in-time safe."
        )
    cache_key = (
        direction,
        long_min_fundamental,
        short_max_fundamental,
        min_market_cap_m,
        source_table,
        as_of_date,
        as_of_ts,
        tuple(long_label_blocklist or ()),
        tuple(short_label_blocklist or ()),
        ACCOUNT_PROFILE,
        pepperstone_table,
        required_currency,
        allow_rebuilt_historical_fundamentals,
        filter_negative_earnings,
        ibkr_margin_table,
    )
    if cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key]

    if direction == "LONG":
        score_filter = sql.SQL("composite_score >= %(score_val)s")
        score_val = long_min_fundamental
    else:
        score_filter = sql.SQL("composite_score <= %(score_val)s")
        score_val = short_max_fundamental

    params: dict = {"score_val": score_val, "min_market_cap_m": min_market_cap_m}
    where_parts = [
        score_filter,
        sql.SQL("symbol IS NOT NULL"),
        sql.SQL("exchange IS NOT NULL"),
        sql.SQL("cik IS NOT NULL"),
        sql.SQL("composite_score IS NOT NULL"),
        sql.SQL("COALESCE(market_cap_m, 0) >= %(min_market_cap_m)s"),
        sql.SQL("high_leverage_flag IS NOT TRUE"),
    ]
    if filter_negative_earnings:
        where_parts.append(sql.SQL("negative_earnings_flag IS NOT TRUE"))

    if as_of_ts is None and as_of_date:
        as_of_ts = _default_as_of_ts(as_of_date)
    if as_of_ts is not None:
        params["as_of_ts"] = as_of_ts
        where_parts.append(sql.SQL("time <= %(as_of_ts)s"))
        where_parts.append(sql.SQL("COALESCE(data_available_at, fundamental_data_available_at) <= %(as_of_ts)s"))

    if direction == "LONG" and long_label_blocklist:
        where_parts.append(sql.SQL("(valuation_label IS NULL OR valuation_label != ALL(%(label_list)s))"))
        params["label_list"] = long_label_blocklist
    elif direction == "SHORT" and short_label_blocklist:
        where_parts.append(sql.SQL("(valuation_label IS NULL OR valuation_label != ALL(%(label_list)s))"))
        params["label_list"] = short_label_blocklist

    if required_currency:
        params["required_currency"] = required_currency.upper()
        where_parts.append(sql.SQL(
            "COALESCE(NULLIF(current_price_currency, ''), "
            "NULLIF(market_cap_currency, ''), "
            "NULLIF(currency, ''), "
            "NULLIF(financial_currency, ''), "
            "%(required_currency)s) = %(required_currency)s"
        ))

    if ACCOUNT_PROFILE == "ps_acc":
        where_parts.append(sql.SQL(
            "symbol IN (SELECT symbol FROM {} "
            "WHERE symbol_ps IS NOT NULL AND is_trading_enabled IS NOT FALSE)"
        ).format(relation_identifier(pepperstone_table)))
    elif ACCOUNT_PROFILE == "ibkr_acc":
        params["ibkr_margin_action"] = ibkr_action_for_direction(direction)
        where_parts.append(sql.SQL(
            "UPPER(TRIM(symbol)) IN (SELECT UPPER(TRIM(source_symbol)) FROM {} "
            "WHERE UPPER(TRIM(action)) = %(ibkr_margin_action)s "
            "AND quantity > 0 AND initial_margin_pct > 0 AND maintenance_margin_pct > 0)"
        ).format(relation_identifier(ibkr_margin_table)))

    recency_order = sql.SQL("COALESCE(data_available_at, fundamental_data_available_at) DESC NULLS LAST, time DESC")

    query = sql.SQL("""
        SELECT DISTINCT ON (symbol, exchange, cik)
            symbol,
            exchange,
            cik,
            composite_score,
            COALESCE(sector, ''),
            COALESCE(industry, ''),
            COALESCE(valuation_label, ''),
            mispricing_score,
            COALESCE(negative_earnings_flag, false),
            COALESCE(high_leverage_flag, false),
            market_cap_m
        FROM {}
        WHERE {}
        ORDER BY
            symbol,
            exchange,
            cik,
            {}
    """).format(
        relation_identifier(source_table),
        sql.SQL("\n          AND ").join(where_parts),
        recency_order,
    )
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    candidates = [
        FundamentalRow(
            symbol=r[0],
            exchange=r[1],
            cik=int(r[2]),
            composite_score=float(r[3]),
            sector=r[4],
            industry=r[5],
            valuation_label=r[6],
            mispricing_score=float(r[7]) if r[7] is not None else None,
            negative_earnings_flag=bool(r[8]),
            high_leverage_flag=bool(r[9]),
            market_cap_m=float(r[10]) if r[10] is not None else None,
        )
        for r in rows
    ]
    _CANDIDATE_CACHE[cache_key] = candidates
    return candidates

# ── Trading day calendar ──────────────────────────────────────────────────────

def get_trading_days(conn: psycopg2.extensions.connection, start: date, end: date) -> list[date]:
    """Return distinct NY trading dates present in the configured 1h source."""
    cache_key = (SOURCE_1H, start, end)
    if cache_key in _TRADING_DAYS_CACHE:
        return _TRADING_DAYS_CACHE[cache_key]

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT DISTINCT (ts AT TIME ZONE 'America/New_York')::date AS d "
                "FROM {} "
                "WHERE ts >= %s AND ts < %s "
                "ORDER BY d"
            ).format(relation_identifier(SOURCE_1H)),
            (start, end + timedelta(days=1)),
        )
        days = [row[0] for row in cur.fetchall()]
    _TRADING_DAYS_CACHE[cache_key] = days
    return days


# ── Outcome simulation ────────────────────────────────────────────────────────

def _day_close_ts(d: date) -> datetime:
    """23:59:59 UTC on the given date — used to cap bar queries to end of day."""
    return datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=timezone.utc)


def _ensure_utc_ts(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _ts_to_epoch_us(ts: datetime) -> int:
    delta = _ensure_utc_ts(ts) - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _epoch_us_to_ts(epoch_us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(epoch_us))


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.strip().split(":", 1)
    return int(hour), int(minute)


def _session_ts(d: date, hhmm: str) -> datetime:
    hour, minute = _parse_hhmm(hhmm)
    local_ts = datetime(d.year, d.month, d.day, hour, minute, tzinfo=_ENTRY_WINDOW_ZONE)
    return local_ts.astimezone(timezone.utc)


def _session_start_ts(d: date) -> datetime:
    return _session_ts(d, ENTRY_WINDOW_START)


def _session_end_ts(d: date) -> datetime:
    return _session_ts(d, ENTRY_WINDOW_END)


def _is_in_entry_window(ts: datetime) -> bool:
    if not ENTRY_WINDOW_ENABLED:
        return True
    local = ts.astimezone(_ENTRY_WINDOW_ZONE)
    return _is_local_time_in_window(local, ENTRY_WINDOW_START, ENTRY_WINDOW_END)


def _is_local_time_in_window(local: datetime, start_hhmm: str, end_hhmm: str) -> bool:
    start_h, start_m = _parse_hhmm(start_hhmm)
    end_h, end_m = _parse_hhmm(end_hhmm)
    current = local.hour * 60 + local.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def _is_in_sl_tp_window(ts: datetime) -> bool:
    local = ts.astimezone(_SL_TP_WINDOW_ZONE)
    return _is_local_time_in_window(local, SL_TP_WINDOW_START, SL_TP_WINDOW_END)


def _is_stop_loss_active(ts: datetime) -> bool:
    if not _is_in_sl_tp_window(ts):
        return False
    if not STOP_LOSS_RTH_ONLY:
        return True
    local = ts.astimezone(_STOP_LOSS_RTH_ZONE)
    return _is_local_time_in_window(local, STOP_LOSS_RTH_START, STOP_LOSS_RTH_END)


def _day_signal_cutoff_ts(d: date) -> datetime:
    return _session_end_ts(d) if ENTRY_WINDOW_ENABLED else _day_close_ts(d)


def _bar_cache_start_ts() -> datetime:
    return datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), datetime.min.time(), tzinfo=timezone.utc)


def _chunked(values: list[InstrumentKey], size: int) -> list[list[InstrumentKey]]:
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def _bar_from_row(ts: datetime, open_: object, high: object, low: object, close: object, volume: object) -> Bar:
    return Bar(ts=ts, open=float(open_), high=float(high), low=float(low), close=float(close), volume=int(volume))


def _new_signal_bar_cache_entry() -> _SignalBarCacheEntry:
    return _SignalBarCacheEntry(
        ts_epoch_us=array("q"),
        opens=array("d"),
        highs=array("d"),
        lows=array("d"),
        closes=array("d"),
        volumes=array("q"),
        loaded_until_ts=None,
    )


def _append_signal_bar(
    entry: _SignalBarCacheEntry,
    ts: datetime,
    open_: object,
    high: object,
    low: object,
    close: object,
    volume: object,
) -> None:
    entry.ts_epoch_us.append(_ts_to_epoch_us(ts))
    entry.opens.append(float(open_))
    entry.highs.append(float(high))
    entry.lows.append(float(low))
    entry.closes.append(float(close))
    entry.volumes.append(int(volume))


def _disable_signal_bar_cache(reason: str) -> None:
    global _SIGNAL_BAR_CACHE_DISABLED
    signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
    _SIGNAL_BAR_CACHE.clear()
    _SIGNAL_BAR_CACHE_DISABLED = True
    log.warning(
        "Signal bar cache disabled %s after clearing %d instruments %d rows estimated %.0f MiB",
        reason,
        signal_symbols,
        signal_bars,
        signal_mib,
    )


def _ensure_identity_bars_loaded(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> int:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    if not unique_identities:
        return 0

    to_load = [
        identity
        for identity in unique_identities
        if identity not in _BAR_CACHE
        or _BAR_CACHE[identity].loaded_until_ts is None
        or _BAR_CACHE[identity].loaded_until_ts < up_to_ts
    ]
    if not to_load:
        if log_batches:
            log.info("Bar preload skipped %d instruments already cached through %s", len(unique_identities), up_to_ts)
        return 0

    total_rows = 0
    batches = _chunked(to_load, batch_size)
    if log_batches:
        log.info(
            "Bar preload starting %d instruments in %d batches of %d through %s",
            len(to_load),
            len(batches),
            batch_size,
            up_to_ts,
        )

    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        for identity in batch:
            _BAR_CACHE.setdefault(identity, _BarCacheEntry(timestamps=[], bars=[], loaded_until_ts=None))

        lower_bound = min(
            _BAR_CACHE[identity].loaded_until_ts or _bar_cache_start_ts()
            for identity in batch
        )
        symbols = [identity[0] for identity in batch]
        exchanges = [identity[1] for identity in batch]
        ciks = [identity[2] for identity in batch]
        rows_loaded = 0
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "WITH requested AS ("
                    "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[]) AS u(symbol, exchange, cik)"
                    ") "
                    "SELECT b.symbol, b.exchange, b.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                    "FROM {} b "
                    "JOIN requested r "
                    "  ON r.symbol = b.symbol AND r.exchange = b.exchange AND r.cik = b.cik "
                    "WHERE b.ts >= %s AND b.ts <= %s "
                    "ORDER BY b.symbol, b.exchange, b.cik, b.ts"
                ).format(relation_identifier(SOURCE_1H)),
                (symbols, exchanges, ciks, lower_bound, up_to_ts),
            )
            for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
                identity = instrument_key(symbol, exchange, cik)
                entry = _BAR_CACHE[identity]
                if entry.loaded_until_ts is not None and _ensure_utc_ts(ts) <= entry.loaded_until_ts:
                    continue
                bar = _bar_from_row(ts, open_, high, low, close, volume)
                entry.timestamps.append(bar.ts)
                entry.bars.append(bar)
                rows_loaded += 1

        for identity in batch:
            entry = _BAR_CACHE[identity]
            if entry.loaded_until_ts is None or entry.loaded_until_ts < up_to_ts:
                entry.loaded_until_ts = up_to_ts

        total_rows += rows_loaded
        if log_batches:
            elapsed = _time.perf_counter() - batch_started
            log.info(
                "Bar preload batch %d/%d loaded %d instruments and %d rows in %.1f s",
                batch_idx,
                len(batches),
                len(batch),
                rows_loaded,
                elapsed,
            )

    if log_batches:
        log.info(
            "Bar preload complete %d instruments and %d new rows through %s",
            len(to_load),
            total_rows,
            up_to_ts,
        )
    return total_rows


def _load_identity_bars_through(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> tuple[list[datetime], list[Bar]]:
    """Load and cache one instrument only through the requested point-in-time timestamp."""
    identity = instrument_key(*identity)
    up_to_ts = _ensure_utc_ts(up_to_ts)
    cached = _BAR_CACHE.get(identity)
    if cached is not None and cached.loaded_until_ts is not None and cached.loaded_until_ts >= up_to_ts:
        return cached.timestamps, cached.bars

    _ensure_identity_bars_loaded(conn, [identity], up_to_ts, batch_size=1, log_batches=False)
    cached = _BAR_CACHE.get(identity)
    if cached is None:
        cached = _BarCacheEntry(timestamps=[], bars=[], loaded_until_ts=up_to_ts)
        _BAR_CACHE[identity] = cached
    return cached.timestamps, cached.bars


def preload_identity_bars(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> int:
    """Batch-load bars for candidate instruments only through the current simulated time."""
    return _ensure_identity_bars_loaded(
        conn,
        identities,
        up_to_ts,
        batch_size=batch_size,
        log_batches=log_batches,
    )


def _entry_window_sql_filter() -> tuple[sql.SQL, list[object]]:
    if not ENTRY_WINDOW_ENABLED:
        return sql.SQL(""), []

    start_h, start_m = _parse_hhmm(ENTRY_WINDOW_START)
    end_h, end_m = _parse_hhmm(ENTRY_WINDOW_END)
    start_time = time(start_h, start_m)
    end_time = time(end_h, end_m)
    local_time = sql.SQL("(b.ts AT TIME ZONE %s)::time")

    if start_time <= end_time:
        return (
            sql.SQL("AND {} BETWEEN %s AND %s").format(local_time),
            [ENTRY_WINDOW_TZ, start_time, end_time],
        )
    return (
        sql.SQL("AND ({} >= %s OR {} <= %s)").format(local_time, local_time),
        [ENTRY_WINDOW_TZ, start_time, ENTRY_WINDOW_TZ, end_time],
    )


def _ensure_signal_bars_loaded(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> bool:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    if not unique_identities:
        return True

    to_load = [
        identity
        for identity in unique_identities
        if identity not in _SIGNAL_BAR_CACHE
        or _SIGNAL_BAR_CACHE[identity].loaded_until_ts is None
        or _SIGNAL_BAR_CACHE[identity].loaded_until_ts < up_to_ts
    ]
    if not to_load:
        if log_batches:
            signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
            log.info(
                "Signal bar cache hit %d instruments through %s cache %d instruments %d rows estimated %.0f MiB",
                len(unique_identities),
                up_to_ts,
                signal_symbols,
                signal_bars,
                signal_mib,
            )
        return True

    total_rows = 0
    batches = _chunked(to_load, batch_size)
    if log_batches:
        log.info(
            "Signal bar cache load starting %d instruments in %d batches of %d through %s",
            len(to_load),
            len(batches),
            batch_size,
            up_to_ts,
        )

    entry_filter, entry_params = _entry_window_sql_filter()
    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        for identity in batch:
            _SIGNAL_BAR_CACHE.setdefault(identity, _new_signal_bar_cache_entry())

        fresh_batch = [
            identity
            for identity in batch
            if _SIGNAL_BAR_CACHE[identity].loaded_until_ts is None
        ]
        incremental_batch = [
            identity
            for identity in batch
            if _SIGNAL_BAR_CACHE[identity].loaded_until_ts is not None
            and _SIGNAL_BAR_CACHE[identity].loaded_until_ts < up_to_ts
        ]
        rows_loaded = 0
        seed_rows_loaded = 0
        incremental_rows_loaded = 0

        if fresh_batch:
            symbols = [identity[0] for identity in fresh_batch]
            exchanges = [identity[1] for identity in fresh_batch]
            ciks = [identity[2] for identity in fresh_batch]
            params: list[object] = [symbols, exchanges, ciks, up_to_ts, *entry_params, limit]
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "WITH requested AS ("
                        "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[]) AS u(symbol, exchange, cik)"
                        ") "
                        "SELECT r.symbol, r.exchange, r.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                        "FROM requested r "
                        "JOIN LATERAL ("
                        "  SELECT b.ts, b.open, b.high, b.low, b.close, b.volume "
                        "  FROM {} b "
                        "  WHERE b.symbol = r.symbol "
                        "    AND b.exchange = r.exchange "
                        "    AND b.cik = r.cik "
                        "    AND b.ts <= %s "
                        "    {} "
                        "  ORDER BY b.ts DESC "
                        "  LIMIT %s"
                        ") b ON TRUE "
                        "ORDER BY r.symbol, r.exchange, r.cik, b.ts"
                    ).format(relation_identifier(SOURCE_1H), entry_filter),
                    params,
                )
                for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
                    identity = instrument_key(symbol, exchange, cik)
                    entry = _SIGNAL_BAR_CACHE[identity]
                    _append_signal_bar(entry, _ensure_utc_ts(ts), open_, high, low, close, volume)
                    rows_loaded += 1
                    seed_rows_loaded += 1

            for identity in fresh_batch:
                _SIGNAL_BAR_CACHE[identity].loaded_until_ts = up_to_ts

        if incremental_batch:
            lower_bound = min(_SIGNAL_BAR_CACHE[identity].loaded_until_ts for identity in incremental_batch)
            symbols = [identity[0] for identity in incremental_batch]
            exchanges = [identity[1] for identity in incremental_batch]
            ciks = [identity[2] for identity in incremental_batch]
            params = [symbols, exchanges, ciks, lower_bound, up_to_ts, *entry_params]
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "WITH requested AS ("
                        "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[]) AS u(symbol, exchange, cik)"
                        ") "
                        "SELECT b.symbol, b.exchange, b.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                        "FROM {} b "
                        "JOIN requested r "
                        "  ON r.symbol = b.symbol AND r.exchange = b.exchange AND r.cik = b.cik "
                        "WHERE b.ts > %s "
                        "  AND b.ts <= %s "
                        "  {} "
                        "ORDER BY b.symbol, b.exchange, b.cik, b.ts"
                    ).format(relation_identifier(SOURCE_1H), entry_filter),
                    params,
                )
                for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
                    identity = instrument_key(symbol, exchange, cik)
                    entry = _SIGNAL_BAR_CACHE[identity]
                    ts_utc = _ensure_utc_ts(ts)
                    if entry.loaded_until_ts is not None and ts_utc <= entry.loaded_until_ts:
                        continue
                    _append_signal_bar(entry, ts_utc, open_, high, low, close, volume)
                    rows_loaded += 1
                    incremental_rows_loaded += 1

            for identity in incremental_batch:
                _SIGNAL_BAR_CACHE[identity].loaded_until_ts = up_to_ts

        total_rows += rows_loaded
        if log_batches:
            elapsed = _time.perf_counter() - batch_started
            signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
            log.info(
                "Signal bar cache batch %d/%d loaded %d instruments seeded %d rows incremental %d rows total %d rows in %.1f s cache %d instruments %d rows estimated %.0f MiB",
                batch_idx,
                len(batches),
                len(batch),
                seed_rows_loaded,
                incremental_rows_loaded,
                rows_loaded,
                elapsed,
                signal_symbols,
                signal_bars,
                signal_mib,
            )

        if _signal_cache_counts()[2] > SIGNAL_BAR_CACHE_MAX_MIB:
            _disable_signal_bar_cache(f"memory budget {SIGNAL_BAR_CACHE_MAX_MIB} MiB exceeded")
            return False

    if log_batches:
        signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
        log.info(
            "Signal bar cache load complete %d instruments and %d new rows through %s cache %d instruments %d rows estimated %.0f MiB",
            len(to_load),
            total_rows,
            up_to_ts,
            signal_symbols,
            signal_bars,
            signal_mib,
        )
    return True


def _recent_bars_from_signal_cache(
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
) -> dict[InstrumentKey, list[Bar]]:
    up_to_epoch_us = _ts_to_epoch_us(up_to_ts)
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    bars_by_identity: dict[InstrumentKey, list[Bar]] = {identity: [] for identity in unique_identities}
    for identity in unique_identities:
        entry = _SIGNAL_BAR_CACHE.get(identity)
        if entry is None:
            continue
        end_idx = bisect_right(entry.ts_epoch_us, up_to_epoch_us)
        start_idx = max(0, end_idx - limit)
        ts_epoch_us = entry.ts_epoch_us
        opens = entry.opens
        highs = entry.highs
        lows = entry.lows
        closes = entry.closes
        volumes = entry.volumes
        bars_by_identity[identity] = [
            Bar(
                ts=_epoch_us_to_ts(ts_epoch_us[idx]),
                open=opens[idx],
                high=highs[idx],
                low=lows[idx],
                close=closes[idx],
                volume=volumes[idx],
            )
            for idx in range(start_idx, end_idx)
        ]
    return bars_by_identity


def _load_recent_bars_for_identities_direct(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> dict[InstrumentKey, list[Bar]]:
    up_to_ts = _ensure_utc_ts(up_to_ts)
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    bars_by_identity: dict[InstrumentKey, list[Bar]] = {identity: [] for identity in unique_identities}
    if not unique_identities or limit <= 0:
        return bars_by_identity

    total_rows = 0
    batches = _chunked(unique_identities, batch_size)
    if log_batches:
        log.info(
            "Recent bar direct load starting %d instruments in %d batches of %d limit %d through %s",
            len(unique_identities),
            len(batches),
            batch_size,
            limit,
            up_to_ts,
        )

    entry_filter, entry_params = _entry_window_sql_filter()
    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        symbols = [identity[0] for identity in batch]
        exchanges = [identity[1] for identity in batch]
        ciks = [identity[2] for identity in batch]
        params: list[object] = [symbols, exchanges, ciks, up_to_ts, *entry_params, limit]
        rows_loaded = 0

        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "WITH requested AS ("
                    "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[]) AS u(symbol, exchange, cik)"
                    ") "
                    "SELECT r.symbol, r.exchange, r.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                    "FROM requested r "
                    "JOIN LATERAL ("
                    "  SELECT b.ts, b.open, b.high, b.low, b.close, b.volume "
                    "  FROM {} b "
                    "  WHERE b.symbol = r.symbol "
                    "    AND b.exchange = r.exchange "
                    "    AND b.cik = r.cik "
                    "    AND b.ts <= %s "
                    "    {} "
                    "  ORDER BY b.ts DESC "
                    "  LIMIT %s"
                    ") b ON TRUE "
                    "ORDER BY r.symbol, r.exchange, r.cik, b.ts"
                ).format(relation_identifier(SOURCE_1H), entry_filter),
                params,
            )
            for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
                identity = instrument_key(symbol, exchange, cik)
                bars_by_identity[identity].append(_bar_from_row(ts, open_, high, low, close, volume))
                rows_loaded += 1

        total_rows += rows_loaded
        if log_batches:
            elapsed = _time.perf_counter() - batch_started
            log.info(
                "Recent bar direct load batch %d/%d loaded %d instruments and %d rows in %.1f s",
                batch_idx,
                len(batches),
                len(batch),
                rows_loaded,
                elapsed,
            )

    if log_batches:
        log.info(
            "Recent bar direct load complete %d instruments and %d rows through %s",
            len(unique_identities),
            total_rows,
            up_to_ts,
        )
    return bars_by_identity


def load_recent_bars_for_identities(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> dict[InstrumentKey, list[Bar]]:
    """Load bounded recent entry-window bars for one signal-evaluation day.

    Signal evaluation reuses a compact, incremental cache. Open-position outcome
    simulation keeps using the full-bar cache via get_bars_range().
    """
    up_to_ts = _ensure_utc_ts(up_to_ts)
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    if not unique_identities or limit <= 0:
        return {identity: [] for identity in unique_identities}

    if SIGNAL_BAR_CACHE_ENABLED and not _SIGNAL_BAR_CACHE_DISABLED:
        loaded = _ensure_signal_bars_loaded(
            conn,
            unique_identities,
            limit,
            up_to_ts,
            batch_size=batch_size,
            log_batches=log_batches,
        )
        if loaded:
            bars_by_identity = _recent_bars_from_signal_cache(unique_identities, limit, up_to_ts)
            if log_batches:
                signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
                returned_rows = sum(len(bars) for bars in bars_by_identity.values())
                log.info(
                    "Recent bar load complete from signal cache %d instruments and %d rows through %s cache %d instruments %d rows estimated %.0f MiB",
                    len(unique_identities),
                    returned_rows,
                    up_to_ts,
                    signal_symbols,
                    signal_bars,
                    signal_mib,
                )
            return bars_by_identity

    return _load_recent_bars_for_identities_direct(
        conn,
        unique_identities,
        limit,
        up_to_ts,
        batch_size=batch_size,
        log_batches=log_batches,
    )


def get_cached_bars(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    limit: int,
    up_to_ts: datetime,
) -> list[Bar]:
    """Return up to `limit` bars using the per-run instrument cache."""
    up_to_ts = _ensure_utc_ts(up_to_ts)
    timestamps, bars = _load_identity_bars_through(conn, identity, up_to_ts)
    end_idx = bisect_right(timestamps, up_to_ts)
    selected: list[Bar] = []
    bar_idx = end_idx - 1
    while bar_idx >= 0 and len(selected) < limit:
        bar = bars[bar_idx]
        if not _is_in_entry_window(bar.ts):
            bar_idx -= 1
            continue
        selected.append(bar)
        bar_idx -= 1
    selected.reverse()
    return selected


def get_bars_range(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    after_ts: datetime,
    up_to_date: date,
) -> list:
    """Return cached 1h bars strictly after after_ts and up to end of up_to_date.

    SL/TP simulation intentionally uses all available bars, not only entry-window bars.
    """
    after_ts = _ensure_utc_ts(after_ts)
    up_to_ts = _day_close_ts(up_to_date)
    timestamps, bars = _load_identity_bars_through(conn, identity, up_to_ts)
    start_idx = bisect_right(timestamps, after_ts)
    end_idx = bisect_right(timestamps, up_to_ts)
    return [(bars[i].ts, bars[i].open, bars[i].high, bars[i].low, bars[i].close) for i in range(start_idx, end_idx)]


def log_cache_stats(context: str = "current") -> None:
    symbols, cached_bars, trading_day_sets, regimes, candidate_sets = _cache_counts()
    signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
    log.info(
        "Cache stats context %s symbols %d bars %d signal symbols %d signal bars %d signal estimated %.0f MiB trading day sets %d regimes %d candidate sets %d",
        context,
        symbols,
        cached_bars,
        signal_symbols,
        signal_bars,
        signal_mib,
        trading_day_sets,
        regimes,
        candidate_sets,
    )


def clear_market_data_caches(context: str = "after_run") -> None:
    global _SIGNAL_BAR_CACHE_DISABLED
    symbols, cached_bars, trading_day_sets, regimes, candidate_sets = _cache_counts()
    signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
    log.info(
        "Cache cleanup context %s clearing symbols %d bars %d signal symbols %d signal bars %d signal estimated %.0f MiB trading day sets %d regimes %d candidate sets %d",
        context,
        symbols,
        cached_bars,
        signal_symbols,
        signal_bars,
        signal_mib,
        trading_day_sets,
        regimes,
        candidate_sets,
    )
    _BAR_CACHE.clear()
    _SIGNAL_BAR_CACHE.clear()
    _SIGNAL_BAR_CACHE_DISABLED = False
    _TRADING_DAYS_CACHE.clear()
    _WORLD_REGIME_CACHE.clear()
    _CANDIDATE_CACHE.clear()
    log.info(
        "Cache cleanup context %s complete symbols %d bars %d signal symbols %d signal bars %d signal estimated %d MiB trading day sets %d regimes %d candidate sets %d",
        context,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
