"""Point-in-time source queries, trading calendar, and bar cache."""

import hashlib
import json
import logging
import os
import shutil
import sys as _sys
import time as _time
from array import array
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
from psycopg2 import sql

from backtest_shared import (
    Bar,
    FundamentalRow,
    InstrumentKey,
    WorldRegime,
    combine_peer_absolute_scores,
    instrument_key,
    normalize_fundamental_score_mode,
)
from .config import *
from .ibkr_margin import get_ibkr_margin_symbols, get_ibkr_margin_universe, ibkr_action_for_direction
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


@dataclass(frozen=True)
class EarningsBlackoutEvent:
    symbol: str
    exchange: str
    cik: int
    earnings_date: date
    announcement_ts: Optional[datetime]
    announcement_time_type: str
    source: str
    source_priority: int
    known_as_of_ts: datetime
    is_confirmed: bool


@dataclass
class _SharedCandidateTimeline:
    cache_dir: Path
    strings: list[str]
    loaded_through_ts: datetime
    loaded_through_epoch_us: int
    rows: int
    identity_count: int
    estimated_mib: float
    identity_start: np.ndarray
    identity_end: np.ndarray
    identity_symbol_code: np.ndarray
    identity_exchange_code: np.ndarray
    identity_cik: np.ndarray
    available_epoch_us: np.ndarray
    source_epoch_us: np.ndarray
    composite_score: np.ndarray
    composite_score_abs: np.ndarray
    mispricing_score: np.ndarray
    market_cap_m: np.ndarray
    flags: np.ndarray
    sector_code: np.ndarray
    industry_code: np.ndarray
    valuation_label_code: np.ndarray
    relative_absolute_divergence_code: np.ndarray
    long_block_reason_code: np.ndarray
    short_block_reason_code: np.ndarray
    current_price_currency_code: np.ndarray
    market_cap_currency_code: np.ndarray
    currency_code: np.ndarray
    financial_currency_code: np.ndarray


_BAR_CACHE: dict[InstrumentKey, _BarCacheEntry] = {}
_BAR_CACHE_DISABLED = False
_SIGNAL_BAR_CACHE: dict[InstrumentKey, _SignalBarCacheEntry] = {}
_SIGNAL_BAR_CACHE_DISABLED = False
_TRADING_DAYS_CACHE: dict[tuple[str, str, tuple[str, ...], date, date], list[date]] = {}
_WORLD_REGIME_CACHE: dict[tuple[str, Optional[date]], Optional[WorldRegime]] = {}
_CANDIDATE_CACHE: dict[tuple, list[FundamentalRow]] = {}
_CANDIDATE_TIMELINE_CACHE: dict[tuple, _SharedCandidateTimeline] = {}
_CANDIDATE_TIMELINE_CACHE_DISABLED = False
_EARNINGS_BLACKOUT_EVENT_CACHE: dict[tuple[str, InstrumentKey, date, int], list[EarningsBlackoutEvent]] = {}
_PEPPERSTONE_SYMBOL_CACHE: dict[tuple[str, bool], tuple[str, ...]] = {}
_PEPPERSTONE_24_SYMBOL_CACHE: dict[str, frozenset[str]] = {}
_ENTRY_WINDOW_ZONE = ZoneInfo(ENTRY_WINDOW_TZ)
_SL_TP_WINDOW_ZONE = ZoneInfo(SL_TP_WINDOW_TZ)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_SIGNAL_BAR_DURATION = timedelta(hours=1)
_BAR_ESTIMATED_BYTES_PER_ROW = 512
_SIGNAL_BAR_ESTIMATED_BYTES_PER_ROW = 80
_CANDIDATE_TIMELINE_ESTIMATED_BYTES_PER_ROW = 640
_SHARED_CANDIDATE_TIMELINE_VERSION = 1
_SIGNAL_DECISIONS_CACHE: dict[tuple[str, str, tuple[str, ...], date], list[tuple[datetime, datetime]]] = {}
_TIMELINE_ROW_BUFFER_SPECS = {
    "available_epoch_us": ("q", np.int64),
    "source_epoch_us": ("q", np.int64),
    "composite_score": ("d", np.float64),
    "composite_score_abs": ("d", np.float64),
    "mispricing_score": ("d", np.float64),
    "market_cap_m": ("d", np.float64),
    "flags": ("B", np.uint8),
    "sector_code": ("i", np.int32),
    "industry_code": ("i", np.int32),
    "valuation_label_code": ("i", np.int32),
    "relative_absolute_divergence_code": ("i", np.int32),
    "long_block_reason_code": ("i", np.int32),
    "short_block_reason_code": ("i", np.int32),
    "current_price_currency_code": ("i", np.int32),
    "market_cap_currency_code": ("i", np.int32),
    "currency_code": ("i", np.int32),
    "financial_currency_code": ("i", np.int32),
}
_SHARED_CANDIDATE_TIMELINE_ARRAYS = (
    "identity_start",
    "identity_end",
    "identity_symbol_code",
    "identity_exchange_code",
    "identity_cik",
    "available_epoch_us",
    "source_epoch_us",
    "composite_score",
    "composite_score_abs",
    "mispricing_score",
    "market_cap_m",
    "flags",
    "sector_code",
    "industry_code",
    "valuation_label_code",
    "relative_absolute_divergence_code",
    "long_block_reason_code",
    "short_block_reason_code",
    "current_price_currency_code",
    "market_cap_currency_code",
    "currency_code",
    "financial_currency_code",
)

def _cache_counts() -> tuple[int, int, float, int, int, int]:
    cached_bars = sum(len(entry.bars) for entry in _BAR_CACHE.values())
    estimated_mib = cached_bars * _BAR_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
    return (
        len(_BAR_CACHE),
        cached_bars,
        estimated_mib,
        len(_TRADING_DAYS_CACHE),
        len(_WORLD_REGIME_CACHE),
        len(_CANDIDATE_CACHE),
    )


def _signal_cache_counts() -> tuple[int, int, float]:
    signal_bars = sum(len(entry.ts_epoch_us) for entry in _SIGNAL_BAR_CACHE.values())
    estimated_mib = signal_bars * _SIGNAL_BAR_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
    return len(_SIGNAL_BAR_CACHE), signal_bars, estimated_mib


def _candidate_cache_counts() -> tuple[int, int]:
    return len(_CANDIDATE_CACHE), sum(len(candidates) for candidates in _CANDIDATE_CACHE.values())


def _candidate_timeline_cache_counts() -> tuple[int, int, int, float]:
    rows = sum(timeline.rows for timeline in _CANDIDATE_TIMELINE_CACHE.values())
    identities = sum(timeline.identity_count for timeline in _CANDIDATE_TIMELINE_CACHE.values())
    estimated_mib = sum(timeline.estimated_mib for timeline in _CANDIDATE_TIMELINE_CACHE.values())
    return len(_CANDIDATE_TIMELINE_CACHE), rows, identities, estimated_mib


def _disable_candidate_timeline_cache(reason: str) -> None:
    global _CANDIDATE_TIMELINE_CACHE_DISABLED
    timeline_sets, timeline_rows, timeline_identities, timeline_mib = _candidate_timeline_cache_counts()
    _CANDIDATE_TIMELINE_CACHE.clear()
    _CANDIDATE_TIMELINE_CACHE_DISABLED = True
    log.warning(
        "Candidate timeline cache disabled %s cleared sets %d rows %d identities %d estimated %.0f MiB",
        reason,
        timeline_sets,
        timeline_rows,
        timeline_identities,
        timeline_mib,
    )


def reset_shared_candidate_timeline_cache() -> None:
    root = _shared_candidate_timeline_cache_root()
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    log.info("Shared candidate timeline cache reset path %s", root)


def _shared_candidate_timeline_cache_root() -> Path:
    return Path(CANDIDATE_TIMELINE_SHARED_CACHE_DIR)


def _shared_candidate_timeline_key_repr(timeline_key: tuple) -> str:
    return repr((
        "candidate_timeline_shared",
        _SHARED_CANDIDATE_TIMELINE_VERSION,
        timeline_key,
    ))


def _shared_candidate_timeline_cache_dir(timeline_key: tuple) -> Path:
    key_repr = _shared_candidate_timeline_key_repr(timeline_key)
    digest = hashlib.sha256(key_repr.encode("utf-8")).hexdigest()[:24]
    profile = str(timeline_key[1]) if len(timeline_key) > 1 else ACCOUNT_PROFILE
    safe_profile = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in profile)
    return _shared_candidate_timeline_cache_root() / f"v{_SHARED_CANDIDATE_TIMELINE_VERSION}_{safe_profile}_{digest}"


def _load_shared_candidate_timeline(timeline_key: tuple) -> Optional[_SharedCandidateTimeline]:
    cache_dir = _shared_candidate_timeline_cache_dir(timeline_key)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("version") != _SHARED_CANDIDATE_TIMELINE_VERSION:
            return None
        if manifest.get("key_repr") != _shared_candidate_timeline_key_repr(timeline_key):
            return None
        arrays = {
            name: np.load(cache_dir / f"{name}.npy", mmap_mode="r")
            for name in _SHARED_CANDIDATE_TIMELINE_ARRAYS
        }
        loaded_through_ts = datetime.fromisoformat(manifest["loaded_through_ts"])
        timeline = _SharedCandidateTimeline(
            cache_dir=cache_dir,
            strings=list(manifest["strings"]),
            loaded_through_ts=loaded_through_ts,
            loaded_through_epoch_us=int(manifest["loaded_through_epoch_us"]),
            rows=int(manifest["rows"]),
            identity_count=int(manifest["identities"]),
            estimated_mib=float(manifest["estimated_mib"]),
            identity_start=arrays["identity_start"],
            identity_end=arrays["identity_end"],
            identity_symbol_code=arrays["identity_symbol_code"],
            identity_exchange_code=arrays["identity_exchange_code"],
            identity_cik=arrays["identity_cik"],
            available_epoch_us=arrays["available_epoch_us"],
            source_epoch_us=arrays["source_epoch_us"],
            composite_score=arrays["composite_score"],
            composite_score_abs=arrays["composite_score_abs"],
            mispricing_score=arrays["mispricing_score"],
            market_cap_m=arrays["market_cap_m"],
            flags=arrays["flags"],
            sector_code=arrays["sector_code"],
            industry_code=arrays["industry_code"],
            valuation_label_code=arrays["valuation_label_code"],
            relative_absolute_divergence_code=arrays["relative_absolute_divergence_code"],
            long_block_reason_code=arrays["long_block_reason_code"],
            short_block_reason_code=arrays["short_block_reason_code"],
            current_price_currency_code=arrays["current_price_currency_code"],
            market_cap_currency_code=arrays["market_cap_currency_code"],
            currency_code=arrays["currency_code"],
            financial_currency_code=arrays["financial_currency_code"],
        )
    except Exception as exc:
        log.warning("Shared candidate timeline cache load failed path %s error %s", cache_dir, exc)
        return None

    log.info(
        "Shared candidate timeline cache loaded path %s rows %d identities %d estimated %.0f MiB",
        cache_dir,
        timeline.rows,
        timeline.identity_count,
        timeline.estimated_mib,
    )
    return timeline


def _open_timeline_row_buffers() -> dict[str, array]:
    return {
        name: array(typecode)
        for name, (typecode, _dtype) in _TIMELINE_ROW_BUFFER_SPECS.items()
    }


def _save_timeline_row_buffers(cache_dir: Path, row_buffers: dict[str, array]) -> None:
    for name, values in row_buffers.items():
        _typecode, dtype = _TIMELINE_ROW_BUFFER_SPECS[name]
        np.save(cache_dir / f"{name}.npy", np.asarray(values, dtype=dtype))


def _float_or_nan(value: object) -> float:
    return float(value) if value is not None else np.nan


def _shared_float_or_none(values: np.ndarray, idx: int) -> Optional[float]:
    value = float(values[idx])
    return value if value == value else None


def _shared_text(timeline: _SharedCandidateTimeline, code: object) -> str:
    return timeline.strings[int(code)]


def _get_pepperstone_symbols(
    conn: psycopg2.extensions.connection,
    pepperstone_table: str,
) -> tuple[str, ...]:
    cache_key = (pepperstone_table, PS_24_ENTRY_SL_TP_ACTIVE)
    cached = _PEPPERSTONE_SYMBOL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    tradable_symbol_filter = (
        "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL OR NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL"
        if PS_24_ENTRY_SL_TP_ACTIVE
        else "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL"
    )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT DISTINCT symbol::text AS symbol
                FROM {}
                WHERE ({})
                  AND is_trading_enabled IS NOT FALSE
                  AND symbol IS NOT NULL
                ORDER BY symbol
                """
            ).format(
                relation_identifier(pepperstone_table),
                sql.SQL(tradable_symbol_filter),
            ),
        )
        symbols = tuple(row[0] for row in cur.fetchall() if row[0])

    _PEPPERSTONE_SYMBOL_CACHE[cache_key] = symbols
    log.info(
        "Loaded Pepperstone tradable symbols table %s count %d ps24 entry sl tp active %s",
        pepperstone_table,
        len(symbols),
        PS_24_ENTRY_SL_TP_ACTIVE,
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
                SELECT DISTINCT symbol::text AS symbol
                FROM {}
                WHERE NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL
                  AND is_trading_enabled IS NOT FALSE
                  AND symbol IS NOT NULL
                ORDER BY symbol
                """
            ).format(relation_identifier(pepperstone_table)),
        )
        symbols = frozenset(row[0] for row in cur.fetchall() if row[0])

    _PEPPERSTONE_24_SYMBOL_CACHE[pepperstone_table] = symbols
    log.info(
        "Loaded Pepperstone 24h symbols table %s count %d",
        pepperstone_table,
        len(symbols),
    )
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
        tradable_symbol_filter = (
            "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL OR NULLIF(BTRIM(symbol_ps24), '') IS NOT NULL"
            if PS_24_ENTRY_SL_TP_ACTIVE
            else "NULLIF(BTRIM(symbol_ps), '') IS NOT NULL"
        )
        currency_filter = sql.SQL("")
        params: list[object] = [list(symbols)]
        if required_currency_norm:
            currency_filter = sql.SQL("AND UPPER(TRIM(COALESCE(quote_asset, ''))) = %s")
            params.append(required_currency_norm)
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT DISTINCT UPPER(TRIM(symbol::text)) AS symbol_norm
                    FROM {}
                    WHERE UPPER(TRIM(symbol::text)) = ANY(%s::text[])
                      AND ({})
                      AND is_trading_enabled IS NOT FALSE
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
            allowed = {row[0] for row in cur.fetchall() if row[0]}
        return tuple(symbol for symbol in symbols if symbol in allowed)

    if ACCOUNT_PROFILE == "ibkr_acc":
        action = ibkr_action_for_direction(direction)
        currency_filter = sql.SQL("")
        params = [list(symbols), action]
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
                ).format(
                    relation_identifier(ibkr_margin_table),
                    currency_filter,
                ),
                params,
            )
            allowed = {row[0] for row in cur.fetchall() if row[0]}
        return tuple(symbol for symbol in symbols if symbol in allowed)

    return symbols


def get_direct_symbol_candidates(
    conn: psycopg2.extensions.connection,
    symbols: tuple[str, ...] | list[str],
    direction: str,
    *,
    as_of_ts: Optional[object],
    source_table: str = SOURCE_MARKET_DATA_1H_TABLE,
    pepperstone_table: str = PS_TRADABLE_SYMBOLS_TABLE,
    required_currency: Optional[str] = "USD",
    ibkr_margin_table: str = IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
    require_broker_eligibility: bool = True,
    require_upcoming_earnings_date: bool = COMMON_REQUIRE_UPCOMING_EARNINGS_DATE,
) -> list[FundamentalRow]:
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
    if not broker_symbols:
        return []

    cutoff_ts = _candidate_as_of_ts(None, as_of_ts)
    if cutoff_ts is not None:
        cutoff_ts = _last_complete_signal_bar_start_ts(cutoff_ts)

    cutoff_filter = sql.SQL("")
    upcoming_earnings_filter = sql.SQL("")
    params: dict[str, object] = {"symbols": list(broker_symbols)}
    if cutoff_ts is not None:
        cutoff_filter = sql.SQL("AND b.ts <= %(cutoff_ts)s")
        params["cutoff_ts"] = cutoff_ts
    if require_upcoming_earnings_date:
        _add_upcoming_earnings_date_params(params, cutoff_ts)
        upcoming_earnings_filter = sql.SQL("AND {}").format(_upcoming_earnings_date_exists_sql("b"))

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                WITH requested AS (
                    SELECT symbol, ord
                    FROM unnest(%(symbols)s::text[]) WITH ORDINALITY AS u(symbol, ord)
                )
                SELECT r.symbol, b.exchange, b.cik
                FROM requested r
                JOIN LATERAL (
                    SELECT b.exchange, b.cik, b.ts
                    FROM {} b
                    WHERE b.symbol = r.symbol
                      {}
                      {}
                    ORDER BY b.ts DESC
                    LIMIT 1
                ) b ON TRUE
                ORDER BY r.ord
                """
            ).format(
                relation_identifier(source_table),
                cutoff_filter,
                upcoming_earnings_filter,
            ),
            params,
        )
        rows = cur.fetchall()

    return [
        FundamentalRow(
            symbol=row[0],
            exchange=row[1],
            cik=int(row[2]),
            composite_score=100.0,
            sector="Benchmark",
            industry="ETF",
            composite_score_abs=100.0,
            valuation_label="benchmark",
            mispricing_score=None,
            negative_earnings_flag=False,
            high_leverage_flag=False,
            market_cap_m=None,
            long_eligible=True,
            short_eligible=True,
            relative_absolute_divergence="",
            long_block_reason="",
            short_block_reason="",
            broker_eligibility_bypassed=not require_broker_eligibility,
        )
        for row in rows
    ]


def _is_pepperstone_24_symbol(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    pepperstone_table: str = PS_TRADABLE_SYMBOLS_TABLE,
) -> bool:
    if ACCOUNT_PROFILE != "ps_acc" or not PS_24_ENTRY_SL_TP_ACTIVE:
        return False
    return instrument_key(*identity)[0] in _get_pepperstone_24_symbols(conn, pepperstone_table)


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

    selected_columns = (
        "day, regime_label, composite_score, dominant_shock_type, max_shock_type_score, "
        "defensive_risk_off_score, energy_commodity_shock_score, rates_inflation_usd_shock_score, "
        "credit_banking_stress_score, policy_geopolitical_score, tech_stress_shock_score, precious_metals_score, "
        "industrial_metals_score, metals_mining_shock_score, metals_mining_subtype"
        if WORLD_REGIME_SHOCK_FIELDS_ACTIVE
        else "day, regime_label, composite_score"
    )
    if as_of_date:
        query = sql.SQL(
            "SELECT {} FROM {} "
            "WHERE composite_score IS NOT NULL AND day <= %s ORDER BY day DESC LIMIT 1"
        ).format(sql.SQL(selected_columns), relation_identifier(source_table))
        params = (as_of_date,)
    else:
        query = sql.SQL(
            "SELECT {} FROM {} "
            "WHERE composite_score IS NOT NULL ORDER BY day DESC LIMIT 1"
        ).format(sql.SQL(selected_columns), relation_identifier(source_table))
        params = ()

    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
    if not row:
        _WORLD_REGIME_CACHE[cache_key] = None
        return None
    if WORLD_REGIME_SHOCK_FIELDS_ACTIVE:
        regime = WorldRegime(
            day=row[0],
            label=row[1],
            score=float(row[2]),
            dominant_shock_type=row[3] or "",
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
            metals_mining_subtype=row[14] or "",
        )
    else:
        regime = WorldRegime(day=row[0], label=row[1], score=float(row[2]))
    _WORLD_REGIME_CACHE[cache_key] = regime
    return regime


def _candidate_text(value: object, default: str = "") -> str:
    if value is None:
        return default
    return _sys.intern(str(value))


def _candidate_effective_currency(
    current_price_currency: object,
    market_cap_currency: object,
    currency: object,
    financial_currency: object,
    required_currency: Optional[str],
) -> str:
    for value in (current_price_currency, market_cap_currency, currency, financial_currency):
        text = str(value).strip().upper() if value is not None else ""
        if text:
            return text
    return required_currency.upper() if required_currency else ""


def _candidate_score_sql(score_mode: str) -> sql.SQL:
    mode = normalize_fundamental_score_mode(score_mode)
    if mode == "peer":
        return sql.SQL("candidates.composite_score")
    if mode == "absolute":
        return sql.SQL("COALESCE(candidates.composite_score_abs, candidates.composite_score)")
    return sql.SQL(
        "((candidates.composite_score * %(fundamental_peer_weight)s) + "
        "(COALESCE(candidates.composite_score_abs, candidates.composite_score) * %(fundamental_abs_weight)s)) "
        "/ %(fundamental_score_weight_sum)s"
    )


def _candidate_as_of_ts(as_of_date: Optional[date], as_of_ts: Optional[object]) -> Optional[datetime]:
    if as_of_ts is None:
        return _default_as_of_ts(as_of_date) if as_of_date else None
    if isinstance(as_of_ts, datetime):
        return _ensure_utc_ts(as_of_ts)
    if isinstance(as_of_ts, date):
        return datetime.combine(as_of_ts, time.max, tzinfo=timezone.utc)
    return None


def _add_upcoming_earnings_date_params(params: dict, as_of_ts: Optional[datetime]) -> None:
    if as_of_ts is None:
        raise ValueError("COMMON_REQUIRE_UPCOMING_EARNINGS_DATE requires as_of_date or as_of_ts")
    entry_date = _entry_local_date(as_of_ts)
    params["required_earnings_min_date"] = entry_date
    params["required_earnings_max_date"] = entry_date + timedelta(days=COMMON_REQUIRE_UPCOMING_EARNINGS_DATE_DAYS)


def _upcoming_earnings_date_exists_sql(candidate_alias: str) -> sql.Composed:
    alias = sql.Identifier(candidate_alias)
    return sql.SQL(
        """
        EXISTS (
            SELECT 1
            FROM {} e
            WHERE e.symbol = {}.symbol
              AND e.exchange = {}.exchange
              AND e.cik = {}.cik
              AND e.earnings_date IS NOT NULL
              AND e.earnings_date >= %(required_earnings_min_date)s::date
              AND e.earnings_date <= %(required_earnings_max_date)s::date
        )
        """
    ).format(
        relation_identifier(SOURCE_EARNINGS_CALENDAR_EVENTS_TABLE),
        alias,
        alias,
        alias,
    )


def _candidate_timeline_key(
    direction: str,
    long_min_fundamental: float,
    short_max_fundamental: float,
    min_market_cap_m: float,
    source_table: str,
    long_label_blocklist: Optional[list],
    short_label_blocklist: Optional[list],
    pepperstone_table: str,
    required_currency: Optional[str],
    allow_rebuilt_historical_fundamentals: bool,
    filter_high_leverage: bool,
    filter_negative_earnings: bool,
    ibkr_margin_table: str,
    broker_universe_key_override: Optional[tuple] = None,
) -> tuple:
    if ACCOUNT_PROFILE == "ps_acc":
        broker_universe_key = ("ps_acc", pepperstone_table, PS_24_ENTRY_SL_TP_ACTIVE)
    elif ACCOUNT_PROFILE == "ibkr_acc":
        broker_universe_key = broker_universe_key_override or (
            "ibkr_acc",
            ibkr_margin_table,
            ibkr_action_for_direction(direction),
        )
    else:
        broker_universe_key = (ACCOUNT_PROFILE,)
    return (
        source_table,
        ACCOUNT_PROFILE,
        broker_universe_key,
        allow_rebuilt_historical_fundamentals,
        END_DATE,
        ENTRY_WINDOW_ENABLED,
        ENTRY_WINDOW_TZ,
        ENTRY_WINDOW_START,
        ENTRY_WINDOW_END,
    )


def _timeline_query(
    account_profile: str,
    source_relation: sql.Composed,
    select_columns: sql.SQL,
    where_parts: list[sql.SQL],
    pepperstone_table: str,
    ibkr_margin_table: str,
) -> sql.Composed:
    timeline_where = list(where_parts)
    timeline_where.append(sql.SQL("f.time <= %(timeline_end_ts)s"))
    timeline_where.append(sql.SQL("COALESCE(f.data_available_at, f.fundamental_data_available_at) <= %(timeline_end_ts)s"))
    timeline_select_columns = sql.SQL("""
        COALESCE(f.data_available_at, f.fundamental_data_available_at) AS available_at,
        f.time AS source_time,
        {}
    """).format(select_columns)
    recency_order = sql.SQL("COALESCE(f.data_available_at, f.fundamental_data_available_at) DESC NULLS LAST, f.time DESC")

    if account_profile == "ps_acc":
        return sql.SQL("""
            WITH broker_symbols AS (
                SELECT broker.symbol
                FROM unnest(%(pepperstone_symbols)s::text[]) AS broker(symbol)
                WHERE broker.symbol IS NOT NULL
                GROUP BY broker.symbol
                ORDER BY broker.symbol
            )
            SELECT {}
            FROM broker_symbols broker
            CROSS JOIN LATERAL (
                SELECT *
                FROM {} f
                WHERE {}
                  AND f.symbol = broker.symbol
                ORDER BY
                    f.exchange,
                    f.cik,
                    {}
            ) f
        """).format(
            timeline_select_columns,
            source_relation,
            sql.SQL("\n              AND ").join(timeline_where),
            recency_order,
        )
    if account_profile == "ibkr_acc":
        return sql.SQL("""
            WITH broker_symbols AS (
                SELECT broker.symbol
                FROM unnest(%(ibkr_margin_symbols)s::text[]) AS broker(symbol)
                WHERE broker.symbol IS NOT NULL
                GROUP BY broker.symbol
                ORDER BY broker.symbol
            )
            SELECT {}
            FROM broker_symbols broker
            CROSS JOIN LATERAL (
                SELECT *
                FROM {} f
                WHERE {}
                  AND f.symbol = broker.symbol
                ORDER BY
                    f.exchange,
                    f.cik,
                    {}
            ) f
        """).format(
            timeline_select_columns,
            source_relation,
            sql.SQL("\n              AND ").join(timeline_where),
            recency_order,
        )
    return sql.SQL("""
        SELECT {}
        FROM {} f
        WHERE {}
        ORDER BY
            f.symbol,
            f.exchange,
            f.cik,
            {}
    """).format(
        timeline_select_columns,
        source_relation,
        sql.SQL("\n          AND ").join(timeline_where),
        recency_order,
    )


def _build_shared_candidate_timeline(
    conn: psycopg2.extensions.connection,
    timeline_key: tuple,
    direction: str,
    query: sql.Composed,
    query_params: dict,
    loaded_through_ts: datetime,
    loaded_through_epoch_us: int,
) -> Optional[_SharedCandidateTimeline]:
    cache_dir = _shared_candidate_timeline_cache_dir(timeline_key)
    loaded = _load_shared_candidate_timeline(timeline_key)
    if loaded is not None:
        return loaded

    root = _shared_candidate_timeline_cache_root()
    root.mkdir(parents=True, exist_ok=True)
    tmp_dir = root / f".building_{cache_dir.name}_{os.getpid()}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    strings: list[str] = [""]
    string_codes: dict[str, int] = {"": 0}

    def code_for(value: object) -> int:
        text = _candidate_text(value)
        code = string_codes.get(text)
        if code is None:
            code = len(strings)
            strings.append(text)
            string_codes[text] = code
        return code

    row_buffers = _open_timeline_row_buffers()
    identity_start: list[int] = []
    identity_end: list[int] = []
    identity_symbol_code: list[int] = []
    identity_exchange_code: list[int] = []
    identity_cik: list[int] = []
    rows_loaded = 0
    current_identity: Optional[InstrumentKey] = None
    current_rows: list[tuple] = []
    flushed_identities: set[InstrumentKey] = set()
    last_flushed_identity: Optional[InstrumentKey] = None
    previous_autocommit = conn.autocommit
    transaction_started = False
    cursor_name = f"shared_candidate_timeline_{abs(hash(timeline_key)) % 10_000_000_000}"
    started = _time.perf_counter()
    existing_timeline_mib = _candidate_timeline_cache_counts()[3]

    def assert_cache_budget() -> None:
        estimated_mib = rows_loaded * _CANDIDATE_TIMELINE_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
        estimated_total_mib = existing_timeline_mib + estimated_mib
        if estimated_total_mib > CANDIDATE_TIMELINE_CACHE_MAX_MIB:
            raise RuntimeError(
                "streamed estimated %.0f MiB total %.0f MiB exceeds max %.0f MiB"
                % (estimated_mib, estimated_total_mib, CANDIDATE_TIMELINE_CACHE_MAX_MIB)
            )

    def flush_identity() -> None:
        nonlocal rows_loaded, current_identity, current_rows, last_flushed_identity
        if current_identity is None:
            return
        start_idx = rows_loaded
        identity_start.append(start_idx)
        identity_symbol_code.append(code_for(current_identity[0]))
        identity_exchange_code.append(code_for(current_identity[1]))
        identity_cik.append(current_identity[2])
        for item in reversed(current_rows):
            (
                available_epoch_us,
                source_epoch_us,
                composite_score,
                composite_score_abs,
                mispricing_score,
                market_cap_m,
                flags,
                sector_code,
                industry_code,
                valuation_label_code,
                relative_absolute_divergence_code,
                long_block_reason_code,
                short_block_reason_code,
                current_price_currency_code,
                market_cap_currency_code,
                currency_code,
                financial_currency_code,
            ) = item
            row_buffers["available_epoch_us"].append(available_epoch_us)
            row_buffers["source_epoch_us"].append(source_epoch_us)
            row_buffers["composite_score"].append(composite_score)
            row_buffers["composite_score_abs"].append(composite_score_abs)
            row_buffers["mispricing_score"].append(mispricing_score)
            row_buffers["market_cap_m"].append(market_cap_m)
            row_buffers["flags"].append(flags)
            row_buffers["sector_code"].append(sector_code)
            row_buffers["industry_code"].append(industry_code)
            row_buffers["valuation_label_code"].append(valuation_label_code)
            row_buffers["relative_absolute_divergence_code"].append(relative_absolute_divergence_code)
            row_buffers["long_block_reason_code"].append(long_block_reason_code)
            row_buffers["short_block_reason_code"].append(short_block_reason_code)
            row_buffers["current_price_currency_code"].append(current_price_currency_code)
            row_buffers["market_cap_currency_code"].append(market_cap_currency_code)
            row_buffers["currency_code"].append(currency_code)
            row_buffers["financial_currency_code"].append(financial_currency_code)
            rows_loaded += 1
            if rows_loaded % CANDIDATE_TIMELINE_CURSOR_ITERSIZE == 0:
                assert_cache_budget()
        identity_end.append(rows_loaded)
        flushed_identities.add(current_identity)
        last_flushed_identity = current_identity
        current_rows = []
        assert_cache_budget()

    try:
        log.info(
            "Shared candidate timeline cache build starting direction %s path %s through %s existing %.0f MiB max %.0f MiB",
            direction,
            cache_dir,
            loaded_through_ts,
            existing_timeline_mib,
            CANDIDATE_TIMELINE_CACHE_MAX_MIB,
        )
        if previous_autocommit:
            conn.autocommit = False
            transaction_started = True
        with conn.cursor(name=cursor_name) as cur:
            cur.itersize = CANDIDATE_TIMELINE_CURSOR_ITERSIZE
            cur.execute(query, query_params)
            for row in cur:
                available_at, source_time = row[0], row[1]
                symbol = _candidate_text(row[2])
                exchange = _candidate_text(row[3])
                cik = int(row[4])
                identity = instrument_key(symbol, exchange, cik)
                if current_identity is None:
                    current_identity = identity
                elif identity != current_identity:
                    flush_identity()
                    if identity in flushed_identities:
                        raise RuntimeError(f"Timeline query returned non-contiguous rows for identity {identity!r}")
                    if last_flushed_identity is not None and identity <= last_flushed_identity:
                        raise RuntimeError(
                            "Timeline query returned non-sorted identity %r after %r"
                            % (identity, last_flushed_identity)
                        )
                    current_identity = identity

                flags = (
                    (1 if bool(row[11]) else 0)
                    | (2 if bool(row[12]) else 0)
                    | (4 if bool(row[14]) else 0)
                    | (8 if bool(row[15]) else 0)
                )
                current_rows.append((
                    _ts_to_epoch_us(available_at),
                    _ts_to_epoch_us(source_time),
                    _float_or_nan(row[5]),
                    _float_or_nan(row[8]),
                    _float_or_nan(row[10]),
                    _float_or_nan(row[13]),
                    flags,
                    code_for(row[6]),
                    code_for(row[7]),
                    code_for(row[9]),
                    code_for(row[16]),
                    code_for(row[17]),
                    code_for(row[18]),
                    code_for(row[19]),
                    code_for(row[20]),
                    code_for(row[21]),
                    code_for(row[22]),
                ))
        flush_identity()
        if transaction_started:
            conn.commit()

        _save_timeline_row_buffers(tmp_dir, row_buffers)
        del row_buffers

        np.save(tmp_dir / "identity_start.npy", np.asarray(identity_start, dtype=np.int64))
        np.save(tmp_dir / "identity_end.npy", np.asarray(identity_end, dtype=np.int64))
        np.save(tmp_dir / "identity_symbol_code.npy", np.asarray(identity_symbol_code, dtype=np.int32))
        np.save(tmp_dir / "identity_exchange_code.npy", np.asarray(identity_exchange_code, dtype=np.int32))
        np.save(tmp_dir / "identity_cik.npy", np.asarray(identity_cik, dtype=np.int64))

        estimated_mib = rows_loaded * _CANDIDATE_TIMELINE_ESTIMATED_BYTES_PER_ROW / 1024 / 1024
        manifest = {
            "version": _SHARED_CANDIDATE_TIMELINE_VERSION,
            "key_repr": _shared_candidate_timeline_key_repr(timeline_key),
            "loaded_through_ts": loaded_through_ts.isoformat(),
            "loaded_through_epoch_us": loaded_through_epoch_us,
            "rows": rows_loaded,
            "identities": len(identity_start),
            "estimated_mib": estimated_mib,
            "strings": strings,
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        tmp_dir.rename(cache_dir)
        elapsed = _time.perf_counter() - started
        log.info(
            "Shared candidate timeline cache build complete direction %s path %s rows %d identities %d strings %d estimated %.0f MiB in %.1f s",
            direction,
            cache_dir,
            rows_loaded,
            len(identity_start),
            len(strings),
            estimated_mib,
            elapsed,
        )
        return _load_shared_candidate_timeline(timeline_key)
    except Exception as exc:
        if transaction_started:
            conn.rollback()
        log.error(
            "Shared candidate timeline cache build failed direction %s path %s after %d rows error %s",
            direction,
            cache_dir,
            rows_loaded,
            exc,
        )
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        return None
    finally:
        if previous_autocommit and not conn.autocommit:
            conn.autocommit = True


def _build_candidate_timeline(
    conn: psycopg2.extensions.connection,
    timeline_key: tuple,
    direction: str,
    source_relation: sql.Composed,
    select_columns: sql.SQL,
    where_parts: list[sql.SQL],
    params: dict,
    pepperstone_table: str,
    ibkr_margin_table: str,
) -> Optional[_SharedCandidateTimeline]:
    if _CANDIDATE_TIMELINE_CACHE_DISABLED or not CANDIDATE_TIMELINE_CACHE_ENABLED:
        return None
    if timeline_key in _CANDIDATE_TIMELINE_CACHE:
        return _CANDIDATE_TIMELINE_CACHE[timeline_key]

    loaded_through_ts = _day_close_ts(END_DATE)
    loaded_through_epoch_us = _ts_to_epoch_us(loaded_through_ts)
    query_params = dict(params)
    query_params["timeline_end_ts"] = loaded_through_ts
    query = _timeline_query(
        ACCOUNT_PROFILE,
        source_relation,
        select_columns,
        where_parts,
        pepperstone_table,
        ibkr_margin_table,
    )
    if ACCOUNT_PROFILE == "ps_acc" and "pepperstone_symbols" not in query_params:
        query_params["pepperstone_symbols"] = list(_get_pepperstone_symbols(conn, pepperstone_table))
    if ACCOUNT_PROFILE == "ibkr_acc" and "ibkr_margin_action" not in query_params:
        query_params["ibkr_margin_action"] = ibkr_action_for_direction(direction)
    if ACCOUNT_PROFILE == "ibkr_acc" and "ibkr_margin_symbols" not in query_params:
        query_params["ibkr_margin_symbols"] = list(
            get_ibkr_margin_symbols(conn, query_params["ibkr_margin_action"], ibkr_margin_table)
        )

    shared_timeline = _build_shared_candidate_timeline(
        conn,
        timeline_key,
        direction,
        query,
        query_params,
        loaded_through_ts,
        loaded_through_epoch_us,
    )
    if shared_timeline is None:
        _disable_candidate_timeline_cache("shared file cache build failed")
        return None
    _CANDIDATE_TIMELINE_CACHE[timeline_key] = shared_timeline
    return shared_timeline


def _get_candidates_from_shared_timeline(
    timeline: _SharedCandidateTimeline,
    direction: str,
    as_of_epoch_us: int,
    params: dict,
    long_label_blocklist: Optional[list],
    short_label_blocklist: Optional[list],
    filter_high_leverage: bool,
    filter_negative_earnings: bool,
    fundamental_score_mode: str,
    fundamental_peer_weight: float,
    fundamental_abs_weight: float,
    long_min_absolute_score: Optional[float],
    short_max_absolute_score: Optional[float],
) -> list[FundamentalRow]:
    candidates: list[FundamentalRow] = []
    score_val = params["score_val"]
    label_blocklist = long_label_blocklist if direction == "LONG" else short_label_blocklist
    required_currency = params.get("required_currency")

    for identity_idx in range(timeline.identity_count):
        start = int(timeline.identity_start[identity_idx])
        end = int(timeline.identity_end[identity_idx])
        row_idx = int(np.searchsorted(
            timeline.available_epoch_us[start:end],
            as_of_epoch_us,
            side="right",
        )) - 1
        while row_idx >= 0:
            absolute_idx = start + row_idx
            if int(timeline.source_epoch_us[absolute_idx]) <= as_of_epoch_us:
                composite_score = _shared_float_or_none(timeline.composite_score, absolute_idx)
                if composite_score is None:
                    break
                composite_score_abs = _shared_float_or_none(timeline.composite_score_abs, absolute_idx)
                score = combine_peer_absolute_scores(
                    composite_score,
                    composite_score_abs,
                    fundamental_score_mode,
                    fundamental_peer_weight,
                    fundamental_abs_weight,
                )
                absolute_score = composite_score_abs if composite_score_abs is not None else composite_score
                if direction == "LONG" and score < score_val:
                    break
                if direction != "LONG" and score > score_val:
                    break
                if direction == "LONG" and long_min_absolute_score is not None and absolute_score < long_min_absolute_score:
                    break
                if direction != "LONG" and short_max_absolute_score is not None and absolute_score > short_max_absolute_score:
                    break

                market_cap_m = _shared_float_or_none(timeline.market_cap_m, absolute_idx)
                if (market_cap_m or 0.0) < params["min_market_cap_m"]:
                    break

                flags = int(timeline.flags[absolute_idx])
                negative_earnings_flag = bool(flags & 1)
                high_leverage_flag = bool(flags & 2)
                if filter_high_leverage and high_leverage_flag:
                    break
                if filter_negative_earnings and negative_earnings_flag:
                    break

                valuation_label = _shared_text(timeline, timeline.valuation_label_code[absolute_idx])
                if label_blocklist and valuation_label in label_blocklist:
                    break
                if required_currency:
                    effective_currency = _candidate_effective_currency(
                        _shared_text(timeline, timeline.current_price_currency_code[absolute_idx]),
                        _shared_text(timeline, timeline.market_cap_currency_code[absolute_idx]),
                        _shared_text(timeline, timeline.currency_code[absolute_idx]),
                        _shared_text(timeline, timeline.financial_currency_code[absolute_idx]),
                        required_currency,
                    )
                    if effective_currency != required_currency.upper():
                        break

                symbol = _shared_text(timeline, timeline.identity_symbol_code[identity_idx])
                exchange = _shared_text(timeline, timeline.identity_exchange_code[identity_idx])
                candidates.append(FundamentalRow(
                    symbol=symbol,
                    exchange=exchange,
                    cik=int(timeline.identity_cik[identity_idx]),
                    composite_score=composite_score,
                    sector=_shared_text(timeline, timeline.sector_code[absolute_idx]),
                    industry=_shared_text(timeline, timeline.industry_code[absolute_idx]),
                    composite_score_abs=composite_score_abs,
                    valuation_label=valuation_label,
                    mispricing_score=_shared_float_or_none(timeline.mispricing_score, absolute_idx),
                    negative_earnings_flag=negative_earnings_flag,
                    high_leverage_flag=high_leverage_flag,
                    market_cap_m=market_cap_m,
                    long_eligible=bool(flags & 4),
                    short_eligible=bool(flags & 8),
                    relative_absolute_divergence=_shared_text(
                        timeline,
                        timeline.relative_absolute_divergence_code[absolute_idx],
                    ),
                    long_block_reason=_shared_text(timeline, timeline.long_block_reason_code[absolute_idx]),
                    short_block_reason=_shared_text(timeline, timeline.short_block_reason_code[absolute_idx]),
                ))
                break
            row_idx -= 1
    return candidates


def _get_candidates_from_timeline(
    conn: psycopg2.extensions.connection,
    timeline_key: tuple,
    direction: str,
    as_of_ts: datetime,
    source_relation: sql.Composed,
    select_columns: sql.SQL,
    where_parts: list[sql.SQL],
    params: dict,
    long_label_blocklist: Optional[list],
    short_label_blocklist: Optional[list],
    filter_high_leverage: bool,
    filter_negative_earnings: bool,
    pepperstone_table: str,
    ibkr_margin_table: str,
    fundamental_score_mode: str,
    fundamental_peer_weight: float,
    fundamental_abs_weight: float,
    long_min_absolute_score: Optional[float],
    short_max_absolute_score: Optional[float],
) -> Optional[list[FundamentalRow]]:
    timeline = _build_candidate_timeline(
        conn,
        timeline_key,
        direction,
        source_relation,
        select_columns,
        where_parts,
        params,
        pepperstone_table,
        ibkr_margin_table,
    )
    if timeline is None:
        return None

    as_of_epoch_us = _ts_to_epoch_us(as_of_ts)
    if as_of_epoch_us > timeline.loaded_through_epoch_us:
        log.warning(
            "Candidate timeline cache skipped direction %s requested %s beyond loaded through %s",
            direction,
            as_of_ts,
            timeline.loaded_through_ts,
        )
        return None

    return _get_candidates_from_shared_timeline(
        timeline,
        direction,
        as_of_epoch_us,
        params,
        long_label_blocklist,
        short_label_blocklist,
        filter_high_leverage,
        filter_negative_earnings,
        fundamental_score_mode,
        fundamental_peer_weight,
        fundamental_abs_weight,
        long_min_absolute_score,
        short_max_absolute_score,
    )


def get_candidates(
    conn: psycopg2.extensions.connection,
    direction: str,
    long_min_fundamental: float,
    short_max_fundamental: float,
    min_market_cap_m: float = 0.0,
    source_table: str = "stock_scorer_fundamental_scores",
    as_of_date: Optional[date] = None,
    as_of_ts: Optional[object] = None,
    long_label_blocklist: Optional[list] = None,
    short_label_blocklist: Optional[list] = None,
    pepperstone_table: str = "public.pepperstone_data",
    required_currency: Optional[str] = "USD",
    allow_rebuilt_historical_fundamentals: bool = False,
    filter_high_leverage: bool = False,
    filter_negative_earnings: bool = False,
    require_upcoming_earnings_date: bool = COMMON_REQUIRE_UPCOMING_EARNINGS_DATE,
    ibkr_margin_table: str = IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
    fundamental_score_mode: str = "peer",
    fundamental_peer_weight: float = 1.0,
    fundamental_abs_weight: float = 0.0,
    long_min_absolute_score: Optional[float] = None,
    short_max_absolute_score: Optional[float] = None,
) -> list[FundamentalRow]:
    if allow_rebuilt_historical_fundamentals:
        raise ValueError(
            "allow_rebuilt_historical_fundamentals=True is disabled; candidate queries must stay point-in-time safe."
        )
    resolved_as_of_ts = _candidate_as_of_ts(as_of_date, as_of_ts)
    fundamental_score_mode = normalize_fundamental_score_mode(fundamental_score_mode)
    fundamental_peer_weight = float(fundamental_peer_weight)
    fundamental_abs_weight = float(fundamental_abs_weight)
    if fundamental_score_mode == "blend" and fundamental_peer_weight + fundamental_abs_weight <= 0.0:
        raise ValueError("FUNDAMENTAL_PEER_WEIGHT + FUNDAMENTAL_ABS_WEIGHT must be > 0 for blend mode")
    if resolved_as_of_ts is not None:
        as_of_ts = resolved_as_of_ts
    cacheable_result = resolved_as_of_ts is None and as_of_date is None and as_of_ts is None
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
        PS_24_ENTRY_SL_TP_ACTIVE,
        required_currency,
        allow_rebuilt_historical_fundamentals,
        filter_high_leverage,
        filter_negative_earnings,
        require_upcoming_earnings_date,
        ibkr_margin_table,
        fundamental_score_mode,
        fundamental_peer_weight,
        fundamental_abs_weight,
        long_min_absolute_score,
        short_max_absolute_score,
    )
    if cacheable_result and cache_key in _CANDIDATE_CACHE:
        return _CANDIDATE_CACHE[cache_key]

    candidate_score_expr = _candidate_score_sql(fundamental_score_mode)
    if direction == "LONG":
        score_filter = sql.SQL("{} >= %(score_val)s").format(candidate_score_expr)
        score_val = long_min_fundamental
    else:
        score_filter = sql.SQL("{} <= %(score_val)s").format(candidate_score_expr)
        score_val = short_max_fundamental

    params: dict = {
        "score_val": score_val,
        "min_market_cap_m": min_market_cap_m,
        "fundamental_peer_weight": fundamental_peer_weight,
        "fundamental_abs_weight": fundamental_abs_weight,
        "fundamental_score_weight_sum": fundamental_peer_weight + fundamental_abs_weight,
    }
    broker_universe_key_override: Optional[tuple] = None
    base_where_parts = [
        sql.SQL("f.symbol IS NOT NULL"),
        sql.SQL("f.exchange IS NOT NULL"),
        sql.SQL("f.cik IS NOT NULL"),
    ]
    eligibility_where_parts = [
        score_filter,
        sql.SQL("candidates.composite_score IS NOT NULL"),
        sql.SQL("COALESCE(candidates.market_cap_m, 0) >= %(min_market_cap_m)s"),
    ]
    absolute_score_expr = sql.SQL("COALESCE(candidates.composite_score_abs, candidates.composite_score)")
    if direction == "LONG" and long_min_absolute_score is not None:
        params["long_min_absolute_score"] = long_min_absolute_score
        eligibility_where_parts.append(sql.SQL("{} >= %(long_min_absolute_score)s").format(absolute_score_expr))
    elif direction != "LONG" and short_max_absolute_score is not None:
        params["short_max_absolute_score"] = short_max_absolute_score
        eligibility_where_parts.append(sql.SQL("{} <= %(short_max_absolute_score)s").format(absolute_score_expr))
    if filter_high_leverage:
        eligibility_where_parts.append(sql.SQL("candidates.high_leverage_flag IS NOT TRUE"))
    if filter_negative_earnings:
        eligibility_where_parts.append(sql.SQL("candidates.negative_earnings_flag IS NOT TRUE"))
    if require_upcoming_earnings_date:
        _add_upcoming_earnings_date_params(params, resolved_as_of_ts)
        base_where_parts.append(_upcoming_earnings_date_exists_sql("f"))

    if direction == "LONG" and long_label_blocklist:
        eligibility_where_parts.append(sql.SQL("(candidates.valuation_label IS NULL OR candidates.valuation_label != ALL(%(label_list)s))"))
        params["label_list"] = long_label_blocklist
    elif direction == "SHORT" and short_label_blocklist:
        eligibility_where_parts.append(sql.SQL("(candidates.valuation_label IS NULL OR candidates.valuation_label != ALL(%(label_list)s))"))
        params["label_list"] = short_label_blocklist

    if required_currency:
        params["required_currency"] = required_currency.upper()
        eligibility_where_parts.append(sql.SQL(
            "COALESCE(NULLIF(candidates.current_price_currency, ''), "
            "NULLIF(candidates.market_cap_currency, ''), "
            "NULLIF(candidates.currency, ''), "
            "NULLIF(candidates.financial_currency, ''), "
            "%(required_currency)s) = %(required_currency)s"
        ))

    if ACCOUNT_PROFILE == "ibkr_acc":
        params["ibkr_margin_action"] = ibkr_action_for_direction(direction)
        broker_universe_key_override, ibkr_margin_symbols = get_ibkr_margin_universe(
            conn,
            params["ibkr_margin_action"],
            ibkr_margin_table,
        )
        params["ibkr_margin_symbols"] = list(ibkr_margin_symbols)
        if not ibkr_margin_symbols:
            if cacheable_result:
                _CANDIDATE_CACHE[cache_key] = []
            return []

    recency_order = sql.SQL("COALESCE(f.data_available_at, f.fundamental_data_available_at) DESC NULLS LAST, f.time DESC")

    select_columns = sql.SQL("""
        f.symbol,
        f.exchange,
        f.cik,
        f.composite_score,
        COALESCE(f.sector, '') AS sector,
        COALESCE(f.industry, '') AS industry,
        f.composite_score_abs,
        COALESCE(f.valuation_label, '') AS valuation_label,
        f.mispricing_score,
        COALESCE(f.negative_earnings_flag, false) AS negative_earnings_flag,
        COALESCE(f.high_leverage_flag, false) AS high_leverage_flag,
        f.market_cap_m,
        COALESCE(f.long_eligible, false) AS long_eligible,
        COALESCE(f.short_eligible, false) AS short_eligible,
        COALESCE(f.relative_absolute_divergence, '') AS relative_absolute_divergence,
        COALESCE(f.long_block_reason, '') AS long_block_reason,
        COALESCE(f.short_block_reason, '') AS short_block_reason,
        f.current_price_currency,
        f.market_cap_currency,
        f.currency,
        f.financial_currency
    """)
    outer_select_columns = sql.SQL("""
        candidates.symbol,
        candidates.exchange,
        candidates.cik,
        candidates.composite_score,
        candidates.sector,
        candidates.industry,
        candidates.composite_score_abs,
        candidates.valuation_label,
        candidates.mispricing_score,
        candidates.negative_earnings_flag,
        candidates.high_leverage_flag,
        candidates.market_cap_m,
        candidates.long_eligible,
        candidates.short_eligible,
        candidates.relative_absolute_divergence,
        candidates.long_block_reason,
        candidates.short_block_reason
    """)
    source_relation = relation_identifier(source_table)

    if resolved_as_of_ts is not None and not require_upcoming_earnings_date:
        timeline_key = _candidate_timeline_key(
            direction,
            long_min_fundamental,
            short_max_fundamental,
            min_market_cap_m,
            source_table,
            long_label_blocklist,
            short_label_blocklist,
            pepperstone_table,
            required_currency,
            allow_rebuilt_historical_fundamentals,
            filter_high_leverage,
            filter_negative_earnings,
            ibkr_margin_table,
            broker_universe_key_override,
        )
        timeline_candidates = _get_candidates_from_timeline(
            conn,
            timeline_key,
            direction,
            resolved_as_of_ts,
            source_relation,
            select_columns,
            base_where_parts,
            params,
            long_label_blocklist,
            short_label_blocklist,
            filter_high_leverage,
            filter_negative_earnings,
            pepperstone_table,
            ibkr_margin_table,
            fundamental_score_mode,
            fundamental_peer_weight,
            fundamental_abs_weight,
            long_min_absolute_score,
            short_max_absolute_score,
        )
        if timeline_candidates is not None:
            return timeline_candidates

    where_parts = list(base_where_parts)
    if as_of_ts is not None:
        params["as_of_ts"] = as_of_ts
        where_parts.append(sql.SQL("f.time <= %(as_of_ts)s"))
        where_parts.append(sql.SQL("COALESCE(f.data_available_at, f.fundamental_data_available_at) <= %(as_of_ts)s"))

    if ACCOUNT_PROFILE == "ps_acc":
        params["pepperstone_symbols"] = list(_get_pepperstone_symbols(conn, pepperstone_table))
        if not params["pepperstone_symbols"]:
            if cacheable_result:
                _CANDIDATE_CACHE[cache_key] = []
            return []
        query = sql.SQL("""
            SELECT {}
            FROM (
                SELECT DISTINCT ON (f.symbol, f.exchange, f.cik)
                    {}
                FROM {} f
                WHERE {}
                  AND f.symbol = ANY(%(pepperstone_symbols)s::text[])
                ORDER BY
                    f.symbol,
                    f.exchange,
                    f.cik,
                    {}
            ) candidates
            WHERE {}
            ORDER BY candidates.symbol, candidates.exchange, candidates.cik
        """).format(
            outer_select_columns,
            select_columns,
            source_relation,
            sql.SQL("\n                  AND ").join(where_parts),
            recency_order,
            sql.SQL("\n              AND ").join(eligibility_where_parts),
        )
    elif ACCOUNT_PROFILE == "ibkr_acc":
        if not params["ibkr_margin_symbols"]:
            if cacheable_result:
                _CANDIDATE_CACHE[cache_key] = []
            return []
        query = sql.SQL("""
            SELECT {}
            FROM (
                SELECT DISTINCT ON (f.symbol, f.exchange, f.cik)
                    {}
                FROM {} f
                WHERE {}
                  AND f.symbol = ANY(%(ibkr_margin_symbols)s::text[])
                ORDER BY
                    f.symbol,
                    f.exchange,
                    f.cik,
                    {}
            ) candidates
            WHERE {}
            ORDER BY candidates.symbol, candidates.exchange, candidates.cik
        """).format(
            outer_select_columns,
            select_columns,
            source_relation,
            sql.SQL("\n                  AND ").join(where_parts),
            recency_order,
            sql.SQL("\n              AND ").join(eligibility_where_parts),
        )
    else:
        query = sql.SQL("""
            SELECT {}
            FROM (
                SELECT DISTINCT ON (f.symbol, f.exchange, f.cik)
                    {}
                FROM {} f
                WHERE {}
                ORDER BY
                    f.symbol,
                    f.exchange,
                    f.cik,
                    {}
            ) candidates
            WHERE {}
            ORDER BY candidates.symbol, candidates.exchange, candidates.cik
        """).format(
            outer_select_columns,
            select_columns,
            source_relation,
            sql.SQL("\n          AND ").join(where_parts),
            recency_order,
            sql.SQL("\n          AND ").join(eligibility_where_parts),
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
            composite_score_abs=float(r[6]) if r[6] is not None else None,
            valuation_label=r[7],
            mispricing_score=float(r[8]) if r[8] is not None else None,
            negative_earnings_flag=bool(r[9]),
            high_leverage_flag=bool(r[10]),
            market_cap_m=float(r[11]) if r[11] is not None else None,
            long_eligible=bool(r[12]),
            short_eligible=bool(r[13]),
            relative_absolute_divergence=r[14],
            long_block_reason=r[15],
            short_block_reason=r[16],
        )
        for r in rows
    ]
    if cacheable_result:
        _CANDIDATE_CACHE[cache_key] = candidates
    return candidates


def preload_candidate_timelines(
    conn: psycopg2.extensions.connection,
    directions: tuple[str, ...],
    *,
    long_min_fundamental: float,
    short_max_fundamental: float,
    min_market_cap_m: float = 0.0,
    source_table: str = "stock_scorer_fundamental_scores",
    as_of_date: Optional[date] = None,
    as_of_ts: Optional[object] = None,
    long_label_blocklist: Optional[list] = None,
    short_label_blocklist: Optional[list] = None,
    pepperstone_table: str = "public.pepperstone_data",
    required_currency: Optional[str] = "USD",
    allow_rebuilt_historical_fundamentals: bool = False,
    filter_high_leverage: bool = False,
    filter_negative_earnings_by_direction: Optional[dict[str, bool]] = None,
    require_upcoming_earnings_date: bool = COMMON_REQUIRE_UPCOMING_EARNINGS_DATE,
    ibkr_margin_table: str = IBKR_SYMBOL_MARGIN_REQUIREMENTS_TABLE,
    fundamental_score_mode: str = "peer",
    fundamental_peer_weight: float = 1.0,
    fundamental_abs_weight: float = 0.0,
    long_min_absolute_score: Optional[float] = None,
    short_max_absolute_score: Optional[float] = None,
) -> tuple[int, int, int, float]:
    if not CANDIDATE_TIMELINE_CACHE_ENABLED or not directions:
        return _candidate_timeline_cache_counts()
    if require_upcoming_earnings_date:
        log.info(
            "Skipping candidate timeline preload because upcoming earnings date eligibility requires per-day filtering"
        )
        return _candidate_timeline_cache_counts()

    resolved_as_of_ts = _candidate_as_of_ts(as_of_date, as_of_ts)
    if resolved_as_of_ts is None:
        return _candidate_timeline_cache_counts()

    started = _time.perf_counter()
    log.info(
        "Candidate timeline preload starting directions %s as of %s",
        ",".join(directions),
        resolved_as_of_ts,
    )
    for direction in directions:
        if _CANDIDATE_TIMELINE_CACHE_DISABLED:
            break
        direction_started = _time.perf_counter()
        candidates = get_candidates(
            conn,
            direction,
            long_min_fundamental=long_min_fundamental,
            short_max_fundamental=short_max_fundamental,
            min_market_cap_m=min_market_cap_m,
            source_table=source_table,
            as_of_date=as_of_date,
            as_of_ts=resolved_as_of_ts,
            long_label_blocklist=long_label_blocklist,
            short_label_blocklist=short_label_blocklist,
            pepperstone_table=pepperstone_table,
            required_currency=required_currency,
            allow_rebuilt_historical_fundamentals=allow_rebuilt_historical_fundamentals,
            filter_high_leverage=filter_high_leverage,
            filter_negative_earnings=(filter_negative_earnings_by_direction or {}).get(direction, False),
            ibkr_margin_table=ibkr_margin_table,
            fundamental_score_mode=fundamental_score_mode,
            fundamental_peer_weight=fundamental_peer_weight,
            fundamental_abs_weight=fundamental_abs_weight,
            long_min_absolute_score=long_min_absolute_score,
            short_max_absolute_score=short_max_absolute_score,
        )
        timeline_sets, timeline_rows, timeline_identities, timeline_mib = _candidate_timeline_cache_counts()
        log.info(
            "Candidate timeline preload direction %s first candidates %d cache sets %d rows %d identities %d estimated %.0f MiB in %.1f s",
            direction,
            len(candidates),
            timeline_sets,
            timeline_rows,
            timeline_identities,
            timeline_mib,
            _time.perf_counter() - direction_started,
        )

    timeline_sets, timeline_rows, timeline_identities, timeline_mib = _candidate_timeline_cache_counts()
    log.info(
        "Candidate timeline preload complete cache sets %d rows %d identities %d estimated %.0f MiB in %.1f s",
        timeline_sets,
        timeline_rows,
        timeline_identities,
        timeline_mib,
        _time.perf_counter() - started,
    )
    return timeline_sets, timeline_rows, timeline_identities, timeline_mib

# ── Trading day calendar ──────────────────────────────────────────────────────

def get_trading_days(conn: psycopg2.extensions.connection, start: date, end: date) -> list[date]:
    """Return distinct local trading dates present in the configured 1h source."""
    cache_key = (SOURCE_MARKET_DATA_1H_TABLE, ENTRY_WINDOW_TZ, TRADING_CALENDAR_SYMBOLS, start, end)
    if cache_key in _TRADING_DAYS_CACHE:
        return _TRADING_DAYS_CACHE[cache_key]

    with conn.cursor() as cur:
        days: list[date] = []
        if TRADING_CALENDAR_SYMBOLS:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT (ts AT TIME ZONE %s)::date AS d "
                    "FROM {} "
                    "WHERE symbol = ANY(%s::text[]) "
                    "  AND ts >= %s AND ts < %s "
                    "ORDER BY d"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (
                    ENTRY_WINDOW_TZ,
                    list(TRADING_CALENDAR_SYMBOLS),
                    _local_day_start_ts(start),
                    _local_day_end_exclusive_ts(end),
                ),
            )
            days = [row[0] for row in cur.fetchall()]
            if not days:
                log.warning(
                    "Trading calendar symbols %s produced no days; falling back to all symbols",
                    ",".join(TRADING_CALENDAR_SYMBOLS),
                )

        if not days:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT (ts AT TIME ZONE %s)::date AS d "
                    "FROM {} "
                    "WHERE ts >= %s AND ts < %s "
                    "ORDER BY d"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (ENTRY_WINDOW_TZ, _local_day_start_ts(start), _local_day_end_exclusive_ts(end)),
            )
            days = [row[0] for row in cur.fetchall()]
    _TRADING_DAYS_CACHE[cache_key] = days
    return days


# ── Outcome simulation ────────────────────────────────────────────────────────

def _local_day_start_ts(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=_ENTRY_WINDOW_ZONE).astimezone(timezone.utc)


def _local_day_end_exclusive_ts(d: date) -> datetime:
    local_next_day = datetime(d.year, d.month, d.day, tzinfo=_ENTRY_WINDOW_ZONE) + timedelta(days=1)
    return local_next_day.astimezone(timezone.utc)


def _day_close_ts(d: date) -> datetime:
    """End of the configured local trading date, used to cap bar queries."""
    return _local_day_end_exclusive_ts(d) - timedelta(microseconds=1)


def _ensure_utc_ts(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def _entry_local_date(ts: datetime) -> date:
    return _ensure_utc_ts(ts).astimezone(_ENTRY_WINDOW_ZONE).date()


def _earnings_event_known_from_date(event: EarningsBlackoutEvent, historical_known_days_before: int) -> date:
    if event.source == "sec_8k_item_2_02":
        return event.earnings_date - timedelta(days=historical_known_days_before)
    return _entry_local_date(event.known_as_of_ts)


def _earnings_event_is_publicly_known(
    event: EarningsBlackoutEvent,
    knowledge_ts: datetime,
    entry_date: date,
    historical_known_days_before: int,
) -> bool:
    if event.source == "sec_8k_item_2_02":
        return entry_date >= _earnings_event_known_from_date(event, historical_known_days_before)
    return _ensure_utc_ts(event.known_as_of_ts) <= _ensure_utc_ts(knowledge_ts)


def _load_earnings_blackout_events(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    entry_date: date,
    blackout_days: int,
    source_table: str,
) -> list[EarningsBlackoutEvent]:
    cache_key = (source_table, identity, entry_date, blackout_days)
    cached = _EARNINGS_BLACKOUT_EVENT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    symbol, exchange, cik = instrument_key(*identity)
    max_earnings_date = entry_date + timedelta(days=blackout_days)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                """
                SELECT
                    symbol,
                    exchange,
                    cik,
                    earnings_date,
                    announcement_ts,
                    COALESCE(announcement_time_type, 'unknown') AS announcement_time_type,
                    source,
                    source_priority,
                    known_as_of_ts,
                    is_confirmed
                FROM {}
                WHERE symbol = %s
                  AND exchange = %s
                  AND cik = %s
                  AND earnings_date >= %s
                  AND earnings_date <= %s
                ORDER BY earnings_date ASC, source_priority DESC, known_as_of_ts DESC
                """
            ).format(relation_identifier(source_table)),
            (symbol, exchange, int(cik), entry_date, max_earnings_date),
        )
        rows = cur.fetchall()

    events = [
        EarningsBlackoutEvent(
            symbol=row[0],
            exchange=row[1],
            cik=int(row[2]),
            earnings_date=row[3],
            announcement_ts=_ensure_utc_ts(row[4]) if row[4] is not None else None,
            announcement_time_type=row[5] or "unknown",
            source=row[6],
            source_priority=int(row[7]),
            known_as_of_ts=_ensure_utc_ts(row[8]),
            is_confirmed=bool(row[9]),
        )
        for row in rows
    ]
    _EARNINGS_BLACKOUT_EVENT_CACHE[cache_key] = events
    return events


def get_upcoming_earnings_blackout_event(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    entry_ts: datetime,
    *,
    knowledge_ts: Optional[datetime] = None,
    source_table: str = SOURCE_EARNINGS_CALENDAR_EVENTS_TABLE,
    blackout_days: int = COMMON_EARNINGS_BLACKOUT_DAYS,
    historical_known_days_before: int = COMMON_HISTORICAL_EARNINGS_KNOWN_DAYS_BEFORE,
) -> Optional[EarningsBlackoutEvent]:
    if blackout_days < 0:
        return None

    entry_ts = _ensure_utc_ts(entry_ts)
    knowledge_ts = _ensure_utc_ts(knowledge_ts) if knowledge_ts is not None else entry_ts
    entry_date = _entry_local_date(entry_ts)
    events = _load_earnings_blackout_events(
        conn,
        identity,
        entry_date,
        blackout_days,
        source_table,
    )
    for event in events:
        if _earnings_event_is_publicly_known(
            event,
            knowledge_ts,
            entry_date,
            historical_known_days_before,
        ):
            return event
    return None


def earnings_blackout_reason_text(
    event: EarningsBlackoutEvent,
    entry_ts: datetime,
    *,
    blackout_days: int = COMMON_EARNINGS_BLACKOUT_DAYS,
    historical_known_days_before: int = COMMON_HISTORICAL_EARNINGS_KNOWN_DAYS_BEFORE,
) -> str:
    entry_date = _entry_local_date(entry_ts)
    known_from_date = _earnings_event_known_from_date(event, historical_known_days_before)
    if event.source == "sec_8k_item_2_02":
        known_from_text = (
            f"{known_from_date} modeled from historical SEC event "
            f"and {historical_known_days_before} known-before days"
        )
    else:
        known_from_text = event.known_as_of_ts.isoformat()
    return (
        f"Upcoming earnings date {event.earnings_date} source {event.source} "
        f"time type {event.announcement_time_type} confirmed {event.is_confirmed} "
        f"is within {blackout_days} calendar days of entry date {entry_date}; "
        f"public known from {known_from_text}."
    )


def _ts_to_epoch_us(ts: datetime) -> int:
    delta = _ensure_utc_ts(ts) - _EPOCH
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


def _epoch_us_to_ts(epoch_us: int) -> datetime:
    return _EPOCH + timedelta(microseconds=int(epoch_us))


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = value.strip().split(":", 1)
    return int(hour), int(minute)


def _is_in_entry_window(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    if conn is not None and identity is not None and _is_pepperstone_24_symbol(conn, identity):
        return True
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


def _is_in_sl_tp_window(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    if conn is not None and identity is not None and _is_pepperstone_24_symbol(conn, identity):
        return True
    local = ts.astimezone(_SL_TP_WINDOW_ZONE)
    return _is_local_time_in_window(local, SL_TP_WINDOW_START, SL_TP_WINDOW_END)


def _is_stop_loss_active(
    ts: datetime,
    conn: Optional[psycopg2.extensions.connection] = None,
    identity: Optional[InstrumentKey] = None,
) -> bool:
    return _is_in_sl_tp_window(ts, conn, identity)


def _last_complete_signal_bar_start_ts(up_to_ts: datetime) -> datetime:
    return _ensure_utc_ts(up_to_ts) - _SIGNAL_BAR_DURATION


def signal_bar_close_decisions_for_day(
    conn: psycopg2.extensions.connection,
    d: date,
) -> list[tuple[datetime, datetime]]:
    cache_key = (SOURCE_MARKET_DATA_1H_TABLE, ENTRY_WINDOW_TZ, TRADING_CALENDAR_SYMBOLS, d)
    if cache_key in _SIGNAL_DECISIONS_CACHE:
        return _SIGNAL_DECISIONS_CACHE[cache_key]

    with conn.cursor() as cur:
        signal_starts: list[datetime] = []
        if TRADING_CALENDAR_SYMBOLS:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT ts "
                    "FROM {} "
                    "WHERE symbol = ANY(%s::text[]) "
                    "  AND ts >= %s AND ts < %s "
                    "ORDER BY ts"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (list(TRADING_CALENDAR_SYMBOLS), _local_day_start_ts(d), _local_day_end_exclusive_ts(d)),
            )
            signal_starts = [_ensure_utc_ts(row[0]) for row in cur.fetchall()]
            if not signal_starts:
                log.warning(
                    "Signal calendar symbols %s produced no bars for %s; falling back to all symbols",
                    ",".join(TRADING_CALENDAR_SYMBOLS),
                    d,
                )

        if not signal_starts:
            cur.execute(
                sql.SQL(
                    "SELECT DISTINCT ts "
                    "FROM {} "
                    "WHERE ts >= %s AND ts < %s "
                    "ORDER BY ts"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (_local_day_start_ts(d), _local_day_end_exclusive_ts(d)),
            )
            signal_starts = [_ensure_utc_ts(row[0]) for row in cur.fetchall()]

    decisions = [(ts, ts + _SIGNAL_BAR_DURATION) for ts in signal_starts]
    _SIGNAL_DECISIONS_CACHE[cache_key] = decisions
    return decisions


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


def _disable_bar_cache(reason: str) -> None:
    global _BAR_CACHE_DISABLED
    bar_symbols, bar_rows, bar_mib, *_ = _cache_counts()
    _BAR_CACHE.clear()
    _BAR_CACHE_DISABLED = True
    log.warning(
        "Bar cache disabled %s after clearing %d instruments %d rows estimated %.0f MiB",
        reason,
        bar_symbols,
        bar_rows,
        bar_mib,
    )


def _ensure_identity_bars_loaded(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> int:
    if _BAR_CACHE_DISABLED:
        return 0
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

        symbols = [identity[0] for identity in batch]
        exchanges = [identity[1] for identity in batch]
        ciks = [identity[2] for identity in batch]
        lower_bounds = [
            _BAR_CACHE[identity].loaded_until_ts or _bar_cache_start_ts()
            for identity in batch
        ]
        rows_loaded = 0
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "WITH requested AS ("
                    "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[], %s::timestamptz[]) AS u(symbol, exchange, cik, lower_bound)"
                    ") "
                    "SELECT b.symbol, b.exchange, b.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                    "FROM {} b "
                    "JOIN requested r "
                    "  ON r.symbol = b.symbol AND r.exchange = b.exchange AND r.cik = b.cik "
                    "WHERE b.ts >= r.lower_bound AND b.ts <= %s "
                    "ORDER BY b.symbol, b.exchange, b.cik, b.ts"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (symbols, exchanges, ciks, lower_bounds, up_to_ts),
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
        if _cache_counts()[2] > BAR_CACHE_MAX_MIB:
            _disable_bar_cache(f"memory budget {BAR_CACHE_MAX_MIB} MiB exceeded")
            return total_rows

    if log_batches:
        bar_symbols, bar_rows, bar_mib, *_ = _cache_counts()
        log.info(
            "Bar preload complete %d instruments and %d new rows through %s cache %d instruments %d rows estimated %.0f MiB",
            len(to_load),
            total_rows,
            up_to_ts,
            bar_symbols,
            bar_rows,
            bar_mib,
        )
    return total_rows


def _load_identity_bars_direct(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> tuple[list[datetime], list[Bar]]:
    identity = instrument_key(*identity)
    up_to_ts = _ensure_utc_ts(up_to_ts)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT ts, open, high, low, close, volume "
                "FROM {} "
                "WHERE symbol = %s AND exchange = %s AND cik = %s "
                "  AND ts >= %s AND ts <= %s "
                "ORDER BY ts"
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
            (identity[0], identity[1], identity[2], _bar_cache_start_ts(), up_to_ts),
        )
        rows = cur.fetchall()
    bars = [_bar_from_row(ts, open_, high, low, close, volume) for ts, open_, high, low, close, volume in rows]
    return [bar.ts for bar in bars], bars


def _load_identity_bars_through(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    up_to_ts: datetime,
) -> tuple[list[datetime], list[Bar]]:
    """Load and cache one instrument only through the requested point-in-time timestamp."""
    identity = instrument_key(*identity)
    up_to_ts = _ensure_utc_ts(up_to_ts)
    if _BAR_CACHE_DISABLED:
        return _load_identity_bars_direct(conn, identity, up_to_ts)
    cached = _BAR_CACHE.get(identity)
    if cached is not None and cached.loaded_until_ts is not None and cached.loaded_until_ts >= up_to_ts:
        return cached.timestamps, cached.bars

    _ensure_identity_bars_loaded(conn, [identity], up_to_ts, batch_size=1, log_batches=False)
    if _BAR_CACHE_DISABLED:
        return _load_identity_bars_direct(conn, identity, up_to_ts)
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
    complete_upper_ts = _last_complete_signal_bar_start_ts(up_to_ts)
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
            params: list[object] = [symbols, exchanges, ciks, complete_upper_ts, limit]
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
                        "  ORDER BY b.ts DESC "
                        "  LIMIT %s"
                        ") b ON TRUE "
                        "ORDER BY r.symbol, r.exchange, r.cik, b.ts"
                    ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
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
            symbols = [identity[0] for identity in incremental_batch]
            exchanges = [identity[1] for identity in incremental_batch]
            ciks = [identity[2] for identity in incremental_batch]
            lower_bounds = [
                _last_complete_signal_bar_start_ts(_SIGNAL_BAR_CACHE[identity].loaded_until_ts)
                for identity in incremental_batch
                if _SIGNAL_BAR_CACHE[identity].loaded_until_ts is not None
            ]
            params = [symbols, exchanges, ciks, lower_bounds, complete_upper_ts]
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "WITH requested AS ("
                        "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[], %s::timestamptz[]) AS u(symbol, exchange, cik, lower_bound)"
                        ") "
                        "SELECT b.symbol, b.exchange, b.cik, b.ts, b.open, b.high, b.low, b.close, b.volume "
                        "FROM {} b "
                        "JOIN requested r "
                        "  ON r.symbol = b.symbol AND r.exchange = b.exchange AND r.cik = b.cik "
                        "WHERE b.ts > r.lower_bound "
                        "  AND b.ts <= %s "
                        "ORDER BY b.symbol, b.exchange, b.cik, b.ts"
                    ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                    params,
                )
                for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
                    identity = instrument_key(symbol, exchange, cik)
                    entry = _SIGNAL_BAR_CACHE[identity]
                    ts_utc = _ensure_utc_ts(ts)
                    if (
                        entry.loaded_until_ts is not None
                        and ts_utc <= _last_complete_signal_bar_start_ts(entry.loaded_until_ts)
                    ):
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
    complete_upper_epoch_us = _ts_to_epoch_us(_last_complete_signal_bar_start_ts(up_to_ts))
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    bars_by_identity: dict[InstrumentKey, list[Bar]] = {identity: [] for identity in unique_identities}
    for identity in unique_identities:
        entry = _SIGNAL_BAR_CACHE.get(identity)
        if entry is None:
            continue
        end_idx = bisect_right(entry.ts_epoch_us, complete_upper_epoch_us)
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
    complete_upper_ts = _last_complete_signal_bar_start_ts(up_to_ts)
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

    for batch_idx, batch in enumerate(batches, start=1):
        batch_started = _time.perf_counter()
        symbols = [identity[0] for identity in batch]
        exchanges = [identity[1] for identity in batch]
        ciks = [identity[2] for identity in batch]
        params: list[object] = [symbols, exchanges, ciks, complete_upper_ts, limit]
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
                    "  ORDER BY b.ts DESC "
                    "  LIMIT %s"
                    ") b ON TRUE "
                    "ORDER BY r.symbol, r.exchange, r.cik, b.ts"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
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


def _load_recent_bars_for_identity_group(
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


def load_recent_bars_for_identities(
    conn: psycopg2.extensions.connection,
    identities: list[InstrumentKey],
    limit: int,
    up_to_ts: datetime,
    *,
    batch_size: int = BAR_CACHE_BATCH_SIZE,
    log_batches: bool = False,
) -> dict[InstrumentKey, list[Bar]]:
    """Load bounded recent signal bars for one signal-evaluation timestamp.

    Open-position outcome simulation keeps using the full-bar cache via
    get_bars_range().
    """
    unique_identities = sorted({instrument_key(symbol, exchange, cik) for symbol, exchange, cik in identities})
    if not unique_identities or limit <= 0:
        return {identity: [] for identity in unique_identities}

    bars_by_identity = _load_recent_bars_for_identity_group(
        conn,
        unique_identities,
        limit,
        up_to_ts,
        batch_size=batch_size,
        log_batches=log_batches,
    )
    for identity in unique_identities:
        bars_by_identity.setdefault(identity, [])
    return bars_by_identity


def get_cached_bars(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    limit: int,
    up_to_ts: datetime,
) -> list[Bar]:
    """Return up to `limit` bars using the per-run instrument cache."""
    up_to_ts = _ensure_utc_ts(up_to_ts)
    timestamps, bars = _load_identity_bars_through(conn, identity, up_to_ts)
    end_idx = bisect_right(timestamps, _last_complete_signal_bar_start_ts(up_to_ts))
    selected: list[Bar] = []
    bar_idx = end_idx - 1
    while bar_idx >= 0 and len(selected) < limit:
        bar = bars[bar_idx]
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
    return get_bars_range_through(conn, identity, after_ts, _day_close_ts(up_to_date))


def get_bars_range_through(
    conn: psycopg2.extensions.connection,
    identity: InstrumentKey,
    after_ts: datetime,
    up_to_ts: datetime,
) -> list:
    """Return cached 1h bars strictly after after_ts and up to up_to_ts."""
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
    """Return next available 1h bar opens for many identity/after_ts pairs."""
    if not requests:
        return {}

    unique_requests = sorted({
        (instrument_key(*identity), _ensure_utc_ts(after_ts))
        for identity, after_ts in requests
    })
    results: dict[tuple[InstrumentKey, datetime], tuple[datetime, float]] = {}

    for batch in _chunked_next_bar_requests(unique_requests, batch_size):
        symbols = [identity[0] for identity, _ in batch]
        exchanges = [identity[1] for identity, _ in batch]
        ciks = [identity[2] for identity, _ in batch]
        after_timestamps = [after_ts for _, after_ts in batch]
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "WITH requested AS ("
                    "  SELECT * FROM unnest(%s::text[], %s::text[], %s::bigint[], %s::timestamptz[]) "
                    "    AS u(symbol, exchange, cik, after_ts)"
                    ") "
                    "SELECT r.symbol, r.exchange, r.cik, r.after_ts, b.ts, b.open "
                    "FROM requested r "
                    "LEFT JOIN LATERAL ("
                    "  SELECT b.ts, b.open "
                    "  FROM {} b "
                    "  WHERE b.symbol = r.symbol "
                    "    AND b.exchange = r.exchange "
                    "    AND b.cik = r.cik "
                    "    AND b.ts > r.after_ts "
                    "  ORDER BY b.ts "
                    "  LIMIT 1"
                    ") b ON TRUE"
                ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE)),
                (symbols, exchanges, ciks, after_timestamps),
            )
            for symbol, exchange, cik, after_ts, next_ts, open_ in cur.fetchall():
                if next_ts is None:
                    continue
                key = (instrument_key(symbol, exchange, cik), _ensure_utc_ts(after_ts))
                results[key] = (_ensure_utc_ts(next_ts), float(open_))

    return results


def _chunked_next_bar_requests(
    values: list[tuple[InstrumentKey, datetime]],
    size: int,
) -> list[list[tuple[InstrumentKey, datetime]]]:
    return [values[idx:idx + size] for idx in range(0, len(values), size)]


def log_cache_stats(context: str = "current") -> None:
    symbols, cached_bars, bar_mib, trading_day_sets, regimes, candidate_sets = _cache_counts()
    candidate_sets, candidate_rows = _candidate_cache_counts()
    signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
    timeline_sets, timeline_rows, timeline_identities, timeline_mib = _candidate_timeline_cache_counts()
    log.info(
        "Cache stats context %s symbols %d bars %d estimated %.0f MiB signal symbols %d signal bars %d signal estimated %.0f MiB candidate timeline sets %d rows %d identities %d estimated %.0f MiB trading day sets %d regimes %d candidate sets %d candidate rows %d",
        context,
        symbols,
        cached_bars,
        bar_mib,
        signal_symbols,
        signal_bars,
        signal_mib,
        timeline_sets,
        timeline_rows,
        timeline_identities,
        timeline_mib,
        trading_day_sets,
        regimes,
        candidate_sets,
        candidate_rows,
    )


def clear_market_data_caches(context: str = "after_run") -> None:
    global _BAR_CACHE_DISABLED, _SIGNAL_BAR_CACHE_DISABLED, _CANDIDATE_TIMELINE_CACHE_DISABLED
    symbols, cached_bars, bar_mib, trading_day_sets, regimes, candidate_sets = _cache_counts()
    candidate_sets, candidate_rows = _candidate_cache_counts()
    signal_symbols, signal_bars, signal_mib = _signal_cache_counts()
    timeline_sets, timeline_rows, timeline_identities, timeline_mib = _candidate_timeline_cache_counts()
    log.info(
        "Cache cleanup context %s clearing symbols %d bars %d estimated %.0f MiB signal symbols %d signal bars %d signal estimated %.0f MiB candidate timeline sets %d rows %d identities %d estimated %.0f MiB trading day sets %d regimes %d candidate sets %d candidate rows %d",
        context,
        symbols,
        cached_bars,
        bar_mib,
        signal_symbols,
        signal_bars,
        signal_mib,
        timeline_sets,
        timeline_rows,
        timeline_identities,
        timeline_mib,
        trading_day_sets,
        regimes,
        candidate_sets,
        candidate_rows,
    )
    _BAR_CACHE.clear()
    _BAR_CACHE_DISABLED = False
    _SIGNAL_BAR_CACHE.clear()
    _SIGNAL_BAR_CACHE_DISABLED = False
    _CANDIDATE_TIMELINE_CACHE.clear()
    _CANDIDATE_TIMELINE_CACHE_DISABLED = False
    _PEPPERSTONE_SYMBOL_CACHE.clear()
    _PEPPERSTONE_24_SYMBOL_CACHE.clear()
    _EARNINGS_BLACKOUT_EVENT_CACHE.clear()
    _SIGNAL_DECISIONS_CACHE.clear()
    _TRADING_DAYS_CACHE.clear()
    _WORLD_REGIME_CACHE.clear()
    _CANDIDATE_CACHE.clear()
    log.info(
        "Cache cleanup context %s complete symbols %d bars %d signal symbols %d signal bars %d signal estimated %d MiB candidate timeline sets %d rows %d identities %d estimated %d MiB trading day sets %d regimes %d candidate sets %d candidate rows %d",
        context,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
