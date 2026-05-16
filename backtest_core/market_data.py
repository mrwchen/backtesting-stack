"""Point-in-time source queries, trading calendar, and bar cache."""

import logging
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2 import sql

from backtest_shared import Bar, FundamentalRow, WorldRegime
from .config import *
from .sql_utils import relation_identifier

log = logging.getLogger(__name__)

_BAR_CACHE: dict[str, tuple[list[datetime], list[Bar]]] = {}
_TRADING_DAYS_CACHE: dict[tuple[str, date, date], list[date]] = {}
_WORLD_REGIME_CACHE: dict[tuple[str, Optional[date]], Optional[WorldRegime]] = {}
_CANDIDATE_CACHE: dict[tuple, list[FundamentalRow]] = {}
_ENTRY_WINDOW_ZONE = ZoneInfo(ENTRY_WINDOW_TZ)
_SL_TP_WINDOW_ZONE = ZoneInfo(SL_TP_WINDOW_TZ)

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
) -> list[FundamentalRow]:
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
        sql.SQL("composite_score IS NOT NULL"),
        sql.SQL("COALESCE(market_cap_m, 0) >= %(min_market_cap_m)s"),
        sql.SQL("negative_earnings_flag IS NOT TRUE"),
        sql.SQL("high_leverage_flag IS NOT TRUE"),
    ]

    if as_of_ts is None and as_of_date:
        as_of_ts = _default_as_of_ts(as_of_date)
    if as_of_ts is not None:
        params["as_of_ts"] = as_of_ts
        where_parts.append(sql.SQL("time <= %(as_of_ts)s"))
        if not allow_rebuilt_historical_fundamentals:
            where_parts.append(sql.SQL("COALESCE(data_available_at, fundamental_data_available_at, time) <= %(as_of_ts)s"))

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

    if allow_rebuilt_historical_fundamentals:
        recency_order = sql.SQL("time DESC")
    else:
        recency_order = sql.SQL("COALESCE(data_available_at, fundamental_data_available_at, time) DESC NULLS LAST, time DESC")

    query = sql.SQL("""
        SELECT DISTINCT ON (symbol)
            symbol,
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
            composite_score=float(r[1]),
            sector=r[2],
            industry=r[3],
            valuation_label=r[4],
            mispricing_score=float(r[5]) if r[5] is not None else None,
            negative_earnings_flag=bool(r[6]),
            high_leverage_flag=bool(r[7]),
            market_cap_m=float(r[8]) if r[8] is not None else None,
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


def _day_signal_cutoff_ts(d: date) -> datetime:
    return _session_end_ts(d) if ENTRY_WINDOW_ENABLED else _day_close_ts(d)


def _load_symbol_bars(
    conn: psycopg2.extensions.connection,
    symbol: str,
) -> tuple[list[datetime], list[Bar]]:
    """Load and cache all bars needed for this backtest run for one symbol."""
    cached = _BAR_CACHE.get(symbol)
    if cached is not None:
        return cached

    start_ts = datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), datetime.min.time(), tzinfo=timezone.utc)
    end_ts = _day_close_ts(END_DATE)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT ts, open, high, low, close, volume FROM {} "
                "WHERE symbol = %s AND ts >= %s AND ts <= %s ORDER BY ts"
            ).format(relation_identifier(SOURCE_1H)),
            (symbol, start_ts, end_ts),
        )
        rows = cur.fetchall()

    bars = [
        Bar(ts=r[0], open=float(r[1]), high=float(r[2]), low=float(r[3]), close=float(r[4]), volume=int(r[5]))
        for r in rows
    ]
    timestamps = [b.ts for b in bars]
    cached = (timestamps, bars)
    _BAR_CACHE[symbol] = cached
    return cached


def preload_symbol_bars(
    conn: psycopg2.extensions.connection,
    symbols: list[str],
) -> None:
    """Batch-load bars for candidate symbols that are not already cached."""
    missing = sorted({s for s in symbols if s not in _BAR_CACHE})
    if not missing:
        return

    start_ts = datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), datetime.min.time(), tzinfo=timezone.utc)
    end_ts = _day_close_ts(END_DATE)
    grouped: dict[str, list[Bar]] = {symbol: [] for symbol in missing}

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT symbol, ts, open, high, low, close, volume FROM {} "
                "WHERE symbol = ANY(%s) AND ts >= %s AND ts <= %s "
                "ORDER BY symbol, ts"
            ).format(relation_identifier(SOURCE_1H)),
            (missing, start_ts, end_ts),
        )
        for symbol, ts, open_, high, low, close, volume in cur.fetchall():
            grouped.setdefault(symbol, []).append(
                Bar(ts=ts, open=float(open_), high=float(high), low=float(low), close=float(close), volume=int(volume))
            )

    for symbol in missing:
        bars = grouped.get(symbol, [])
        _BAR_CACHE[symbol] = ([b.ts for b in bars], bars)

    log.info("Preloaded 1h bars — symbols=%d  bars=%d", len(missing), sum(len(v) for v in grouped.values()))


def get_cached_bars(
    conn: psycopg2.extensions.connection,
    symbol: str,
    limit: int,
    up_to_ts: datetime,
) -> list[Bar]:
    """Return up to `limit` bars using the per-run symbol cache."""
    timestamps, bars = _load_symbol_bars(conn, symbol)
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
    symbol: str,
    after_ts: datetime,
    up_to_date: date,
) -> list:
    """Return cached 1h bars strictly after after_ts and up to end of up_to_date.

    SL/TP simulation intentionally uses all available bars, not only entry-window bars.
    """
    timestamps, bars = _load_symbol_bars(conn, symbol)
    start_idx = bisect_right(timestamps, after_ts)
    end_idx = bisect_right(timestamps, _day_close_ts(up_to_date))
    return [(bars[i].ts, bars[i].open, bars[i].high, bars[i].low, bars[i].close) for i in range(start_idx, end_idx)]


def log_cache_stats() -> None:
    cached_bars = sum(len(bars) for _, bars in _BAR_CACHE.values())
    log.info(
        "Cache stats — symbols_with_bars=%d  bars=%d  trading_day_sets=%d  regimes=%d  candidate_sets=%d",
        len(_BAR_CACHE),
        cached_bars,
        len(_TRADING_DAYS_CACHE),
        len(_WORLD_REGIME_CACHE),
        len(_CANDIDATE_CACHE),
    )
