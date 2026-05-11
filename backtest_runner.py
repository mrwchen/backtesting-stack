"""
Swing trade backtester.

Generates swing signals day-by-day on historical data, simulates a margin
account, and writes results to backtest_runs / backtest_trades.

Point-in-time data used:
  - world_regime_daily_scores_mv  : as_of each simulated day  (true PIT)
  - stocks_analysis_fundamental_scores : available at each simulated entry cutoff (true PIT)
  - alpaca_market_data_1h         : up_to each simulated entry cutoff (true PIT)

Run once, write results, exit.
"""

import logging
import os
import re
import importlib.util
import sys
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Shared model API: types, env helpers, indicators, source queries ──────────

@dataclass(frozen=True)
class WorldRegime:
    day: object
    label: str
    score: float


@dataclass(frozen=True)
class FundamentalRow:
    symbol: str
    composite_score: float
    sector: str
    industry: str
    valuation_label: str = ""
    mispricing_score: float | None = None
    negative_earnings_flag: bool = False
    high_leverage_flag: bool = False


@dataclass(frozen=True)
class Bar:
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Signal:
    symbol: str
    direction: str
    fundamental_score: float
    entry_score: float
    combined_score: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    pullback_pct: float
    rsi_1h: float
    volume_ratio: float
    entry_reason: str
    signal_valid_until: datetime
    valuation_label: str = ""
    sector: str = ""
    industry: str = ""
    entry_ts: Optional[datetime] = None


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_list(name: str, default: Iterable[str]) -> list[str]:
    raw = os.getenv(name, ",".join(default))
    return [x.strip() for x in raw.split(",") if x.strip()]


def compute_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) < period + 2:
        return 50.0
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_relation_name(relation_name: str) -> tuple[str, ...]:
    value = (relation_name or "").strip()
    parts = tuple(part.strip() for part in value.split(".") if part.strip())
    if len(parts) not in (1, 2):
        raise ValueError(f"Invalid relation name: {relation_name!r}")
    for part in parts:
        if not _IDENTIFIER_RE.fullmatch(part):
            raise ValueError(f"Invalid relation identifier: {relation_name!r}")
    return parts


def relation_identifier(relation_name: str) -> sql.Identifier:
    return sql.Identifier(*parse_relation_name(relation_name))


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
    symbol_universe: str = "all",
    pepperstone_table: str = "pepperstone_data",
    required_currency: Optional[str] = "USD",
    fundamental_market_universe: Optional[str] = None,
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
        symbol_universe,
        pepperstone_table,
        required_currency,
        fundamental_market_universe,
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
        where_parts.extend([
            sql.SQL("time <= %(as_of_ts)s"),
            sql.SQL("COALESCE(data_available_at, fundamental_data_available_at, time) <= %(as_of_ts)s"),
        ])

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

    if fundamental_market_universe:
        params["fundamental_market_universe"] = fundamental_market_universe
        where_parts.append(sql.SQL("market_universe = %(fundamental_market_universe)s"))

    universe = (symbol_universe or "all").strip().lower()
    if universe == "pepperstone":
        where_parts.append(sql.SQL(
            "symbol IN (SELECT symbol FROM {} "
            "WHERE symbol_ps IS NOT NULL AND is_trading_enabled IS NOT FALSE)"
        ).format(relation_identifier(pepperstone_table)))
    elif universe == "pepperstone24":
        where_parts.append(sql.SQL(
            "symbol IN (SELECT symbol FROM {} "
            "WHERE symbol_ps24 IS NOT NULL AND is_trading_enabled IS NOT FALSE)"
        ).format(relation_identifier(pepperstone_table)))

    query = sql.SQL("""
        SELECT DISTINCT ON (symbol)
            symbol,
            composite_score,
            COALESCE(sector, ''),
            COALESCE(industry, ''),
            COALESCE(valuation_label, ''),
            mispricing_score,
            COALESCE(negative_earnings_flag, false),
            COALESCE(high_leverage_flag, false)
        FROM {}
        WHERE {}
        ORDER BY
            symbol,
            COALESCE(data_available_at, fundamental_data_available_at, time) DESC NULLS LAST,
            time DESC
    """).format(
        relation_identifier(source_table),
        sql.SQL("\n          AND ").join(where_parts),
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
        )
        for r in rows
    ]
    _CANDIDATE_CACHE[cache_key] = candidates
    return candidates


sys.modules.setdefault("backtest_shared", sys.modules[__name__])

# ── Backtest-specific config ──────────────────────────────────────────────────

START_DATE             = date.fromisoformat(os.getenv("START_DATE", "2023-01-01"))
END_DATE               = date.fromisoformat(os.getenv("END_DATE", str(date.today())))
ACCOUNT_PROFILE        = os.getenv("ACCOUNT_PROFILE", "ps_acc").strip().lower()
ACCOUNT_CURRENCY       = os.getenv("ACCOUNT_CURRENCY", "USD").strip().upper()
INITIAL_EQUITY         = float(os.getenv("INITIAL_EQUITY", "100000.0"))
RISK_PER_TRADE_PCT     = float(os.getenv("RISK_PER_TRADE_PCT", "2.0"))
MAX_OPEN_POSITIONS     = int(os.getenv("MAX_OPEN_POSITIONS", "5"))
MIN_FREE_MARGIN_PCT    = float(os.getenv("MIN_FREE_MARGIN_PCT", "20.0"))
ALLOW_FRACTIONAL_SHARES = os.getenv("ALLOW_FRACTIONAL_SHARES", "true").strip().lower() in {"1", "true", "yes", "y", "on"}

ACCOUNT_PROFILE_DEFAULTS = {
    # Pepperstone EU retail, individual shares CFDs: 5:1 leverage => 20% margin.
    # US Share/ETF CFDs: 0.02 USD/share per side, 0.02 USD minimum.
    "ps_acc": {
        "margin_requirement_pct": 20.0,
        "maintenance_margin_pct": 20.0,
        "commission_per_order_usd": 0.0,
        "commission_per_share_usd": 0.02,
        "commission_min_per_order_usd": 0.02,
        "commission_max_pct": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 2.0,
        "slippage_bps": 1.0,
        "margin_financing_rate_pct": 5.0,
        "allow_fractional_shares": False,
    },
    # IBKR margin account approximation for US stocks, Germany retail user:
    # Reg-T-like overnight initial margin 50%; IBKR Pro tiered first tier.
    "ibkr_acc": {
        "margin_requirement_pct": 50.0,
        "maintenance_margin_pct": 25.0,
        "commission_per_order_usd": 0.0,
        "commission_per_share_usd": 0.0035,
        "commission_min_per_order_usd": 0.35,
        "commission_max_pct": 1.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 1.0,
        "margin_financing_rate_pct": 6.5,
        "allow_fractional_shares": True,
    },
}
if ACCOUNT_PROFILE not in ACCOUNT_PROFILE_DEFAULTS:
    raise ValueError("ACCOUNT_PROFILE must be one of: ps_acc, ibkr_acc")
_ACC = ACCOUNT_PROFILE_DEFAULTS[ACCOUNT_PROFILE]

def _account_float(env_key: str, default_key: str) -> float:
    return float(os.getenv(env_key, str(_ACC[default_key])))

def _account_bool(env_key: str, default_key: str) -> bool:
    raw = os.getenv(env_key)
    if raw is None:
        return bool(_ACC[default_key])
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

MARGIN_REQUIREMENT_PCT = _account_float("MARGIN_REQUIREMENT_PCT", "margin_requirement_pct")
MAINTENANCE_MARGIN_PCT = _account_float("MAINTENANCE_MARGIN_PCT", "maintenance_margin_pct")
COMMISSION_PER_ORDER_USD = _account_float("COMMISSION_PER_ORDER_USD", "commission_per_order_usd")
COMMISSION_PER_SHARE_USD = _account_float("COMMISSION_PER_SHARE_USD", "commission_per_share_usd")
COMMISSION_MIN_PER_ORDER_USD = _account_float("COMMISSION_MIN_PER_ORDER_USD", "commission_min_per_order_usd")
COMMISSION_MAX_PCT = _account_float("COMMISSION_MAX_PCT", "commission_max_pct")
COMMISSION_BPS        = _account_float("COMMISSION_BPS", "commission_bps")
SPREAD_BPS            = _account_float("SPREAD_BPS", "spread_bps")
SLIPPAGE_BPS          = _account_float("SLIPPAGE_BPS", "slippage_bps")
MARGIN_FINANCING_RATE_PCT = _account_float("MARGIN_FINANCING_RATE_PCT", "margin_financing_rate_pct")
ALLOW_FRACTIONAL_SHARES = _account_bool("ALLOW_FRACTIONAL_SHARES", "allow_fractional_shares")
RUN_NOTES              = os.getenv("RUN_NOTES", "")
RUN_LABEL_TZ           = os.getenv("RUN_LABEL_TZ", "Europe/Berlin")
PROGRESS_LOG_EVERY_DAYS = max(1, int(os.getenv("PROGRESS_LOG_EVERY_DAYS", "25")))
BAR_CACHE_WARMUP_DAYS = max(1, int(os.getenv("BAR_CACHE_WARMUP_DAYS", "120")))
ENSURE_SOURCE_INDEXES = os.getenv("ENSURE_SOURCE_INDEXES", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
MONTE_CARLO_ENABLED       = os.getenv("MONTE_CARLO_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
N_MONTE_CARLO_SIMULATIONS = max(0, int(os.getenv("N_MONTE_CARLO_SIMULATIONS", "2000")))
MIN_MARKET_CAP_M = float(os.getenv("MIN_MARKET_CAP_M", "1000.0"))
LONG_MAX_HOLD_DAYS = max(0.0, float(os.getenv("LONG_MAX_HOLD_DAYS", "5.0")))
SHORT_MAX_HOLD_DAYS = max(0.0, float(os.getenv("SHORT_MAX_HOLD_DAYS", "5.0")))
TP1_CLOSE_RATIO = max(0.0, min(1.0, float(os.getenv("TP1_CLOSE_RATIO", "0.5"))))

GRID_SEARCH_ENABLED = os.getenv("GRID_SEARCH_ENABLED", "false").strip().lower() in {"1", "true", "yes", "y", "on"}
MODEL_FILE = os.getenv("MODEL_FILE", "pullback_bounce_fundamental_v1.py").strip()
MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "backtest_models")).strip()
RESULT_SCHEMA = os.getenv("RESULT_SCHEMA", "public").strip() or "public"
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", RESULT_SCHEMA):
    raise ValueError(f"Invalid RESULT_SCHEMA: {RESULT_SCHEMA!r}")


def _qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _result_table(name: str) -> str:
    return f"{_qident(RESULT_SCHEMA)}.{_qident(name)}"


def _parse_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})


def _parse_hold_grid_vals(env_key: str, default_val: float) -> list[float]:
    raw = os.getenv(env_key, str(default_val))
    return sorted({float(x.strip()) for x in raw.split(",") if x.strip()})
ENTRY_WINDOW_ENABLED = os.getenv("ENTRY_WINDOW_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
ENTRY_WINDOW_TZ = os.getenv("ENTRY_WINDOW_TZ", "America/New_York")
ENTRY_WINDOW_START = os.getenv("ENTRY_WINDOW_START", "06:30")
ENTRY_WINDOW_END = os.getenv("ENTRY_WINDOW_END", "19:00")

SOURCE_1H           = os.getenv("SOURCE_1H", "alpaca_market_data_1h")
SOURCE_FUNDAMENTAL  = os.getenv("SOURCE_FUNDAMENTAL", "stocks_analysis_fundamental_scores")
SOURCE_WORLD_REGIME = os.getenv("SOURCE_WORLD_REGIME", "world_regime_daily_scores_mv")
SYMBOL_UNIVERSE     = os.getenv("SYMBOL_UNIVERSE", "all")  # all | pepperstone | pepperstone24
PEPPERSTONE_TABLE   = os.getenv("PEPPERSTONE_TABLE", "pepperstone_data")
REQUIRE_USD_FUNDAMENTALS = os.getenv("REQUIRE_USD_FUNDAMENTALS", "true").strip().lower() in {"1", "true", "yes", "y", "on"}
FUNDAMENTAL_MARKET_UNIVERSE = os.getenv("FUNDAMENTAL_MARKET_UNIVERSE", "").strip()

DB_CONNECT_RETRIES       = int(os.getenv("DB_CONNECT_RETRIES", "5"))
DB_CONNECT_RETRY_DELAY_S = float(os.getenv("DB_CONNECT_RETRY_DELAY_S", "5.0"))

import time as _time

DB = {
    "host":            os.getenv("PGHOST", "timescaledb"),
    "port":            int(os.getenv("PGPORT", "5432")),
    "dbname":          os.getenv("PGDATABASE", "postgres"),
    "user":            os.getenv("PGUSER", "market-data-account"),
    "password":        os.getenv("PGPASSWORD", "market-data-account-pw"),
    "connect_timeout": int(os.getenv("CONNECT_TIMEOUT_SECONDS", "10")),
    "application_name": "backtest_runner",
    "options": os.getenv("PGOPTIONS", f"-c search_path={RESULT_SCHEMA}"),
}

SOURCE_INDEX_DB = {
    "host":            os.getenv("SOURCE_INDEX_PGHOST", DB["host"]),
    "port":            int(os.getenv("SOURCE_INDEX_PGPORT", str(DB["port"]))),
    "dbname":          os.getenv("SOURCE_INDEX_PGDATABASE", DB["dbname"]),
    "user":            os.getenv("SOURCE_INDEX_PGUSER", DB["user"]),
    "password":        os.getenv("SOURCE_INDEX_PGPASSWORD", DB["password"]),
    "connect_timeout": int(os.getenv("CONNECT_TIMEOUT_SECONDS", "10")),
    "application_name": "backtest_runner_source_index_builder",
    "options": os.getenv("SOURCE_INDEX_PGOPTIONS", DB["options"]),
}

_BAR_CACHE: dict[str, tuple[list[datetime], list[Bar]]] = {}
_TRADING_DAYS_CACHE: dict[tuple[str, date, date], list[date]] = {}
_WORLD_REGIME_CACHE: dict[tuple[str, Optional[date]], Optional[WorldRegime]] = {}
_CANDIDATE_CACHE: dict[tuple, list[FundamentalRow]] = {}
_ENTRY_WINDOW_ZONE = ZoneInfo(ENTRY_WINDOW_TZ)
_MODEL_MODULE: Optional[ModuleType] = None


# ── Data structures ───────────────────────────────────────────────────────────

def load_model_module() -> ModuleType:
    """Load the configured backtesting model from backtest_models/<MODEL_FILE>."""
    model_file = MODEL_FILE
    model_path = Path(model_file)
    if not model_file or model_path.name != model_file or model_path.suffix != ".py":
        raise ValueError(
            f"Invalid MODEL_FILE {MODEL_FILE!r}. Use a plain Python filename like pullback_bounce_fundamental_v1.py"
        )

    full_path = Path(MODEL_DIR) / model_file
    if not full_path.is_file():
        raise FileNotFoundError(f"Backtesting model file not found: {full_path}")

    module_name = f"backtest_model_{model_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load backtesting model spec from {full_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    required_attrs = [
        "SignalConfig",
        "signal_config_from_env",
        "compute_long_signal",
        "compute_short_signal",
    ]
    missing = [name for name in required_attrs if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"Backtesting model {model_file} is missing required symbols: {', '.join(missing)}"
        )

    log.info("Loaded backtesting model — file=%s path=%s", model_file, full_path)
    return module


def get_model_module() -> ModuleType:
    if _MODEL_MODULE is None:
        raise RuntimeError("Backtesting model has not been loaded yet")
    return _MODEL_MODULE

@dataclass
class OpenPosition:
    symbol: str
    direction: str
    entry_date: date
    entry_ts: datetime
    entry_price: float
    stop_loss: float
    effective_sl: float        # moves to entry after TP1 hit
    take_profit_1: float
    take_profit_2: float
    valid_until: datetime
    shares: float
    position_size_usd: float
    margin_used: float
    equity_before: float
    signal: Signal
    world_regime_label: str = ""
    world_regime_score: float = 0.0
    valuation_label: str = ""
    tp1_close_ratio: float = 0.5
    # incremental simulate_outcome state — updated in-place on each day's pass
    tp1_hit: bool = False
    tp1_price: Optional[float] = None
    tp1_exit_ts: Optional[datetime] = None
    last_bar_ts: Optional[datetime] = None
    bars_processed: int = 0


@dataclass
class ClosedTrade:
    position: OpenPosition
    outcome_status: str        # HIT_TP2 | HIT_TP1_THEN_BE | HIT_SL | MAX_HOLD | FORCE_CLOSED
    outcome_price: float
    outcome_date: date
    outcome_bars: int
    tp1_hit: bool
    return_pct: float
    pnl_usd: float
    equity_after: float
    exit_ts: datetime = None
    tp1_exit_ts: Optional[datetime] = None


# ── DB connect ────────────────────────────────────────────────────────────────

def connect_with_retry() -> psycopg2.extensions.connection:
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            return psycopg2.connect(**DB)
        except psycopg2.OperationalError as exc:
            if attempt == DB_CONNECT_RETRIES:
                raise
            delay = DB_CONNECT_RETRY_DELAY_S * (2 ** (attempt - 1))
            log.warning("DB connect failed (%d/%d, retry in %.0fs): %s", attempt, DB_CONNECT_RETRIES, delay, exc)
            _time.sleep(delay)
    raise RuntimeError("unreachable")


def connect_source_index_with_retry() -> psycopg2.extensions.connection:
    """Connect with credentials allowed to CREATE INDEX on source hypertables."""
    for attempt in range(1, DB_CONNECT_RETRIES + 1):
        try:
            return psycopg2.connect(**SOURCE_INDEX_DB)
        except psycopg2.OperationalError as exc:
            if attempt == DB_CONNECT_RETRIES:
                raise
            delay = DB_CONNECT_RETRY_DELAY_S * (2 ** (attempt - 1))
            log.warning(
                "Source-index DB connect failed (%d/%d, retry in %.0fs): %s",
                attempt,
                DB_CONNECT_RETRIES,
                delay,
                exc,
            )
            _time.sleep(delay)
    raise RuntimeError("unreachable")


# ── Source validation ─────────────────────────────────────────────────────────

def _relation_columns(conn: psycopg2.extensions.connection, relation_name: str) -> set[str]:
    """Return column names for a table/view/materialized view, or fail clearly."""
    relation_identifier(relation_name)  # validates plain/schema-qualified identifier
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_attribute a
            WHERE a.attrelid = to_regclass(%s)
              AND a.attnum > 0
              AND NOT a.attisdropped
            """,
            (relation_name,),
        )
        columns = {row[0] for row in cur.fetchall()}
    if not columns:
        raise RuntimeError(f"Required source relation not found or has no columns: {relation_name}")
    return columns


def _require_columns(conn: psycopg2.extensions.connection, relation_name: str, required: set[str]) -> None:
    columns = _relation_columns(conn, relation_name)
    missing = sorted(required - columns)
    if missing:
        raise RuntimeError(
            f"Required source relation {relation_name} is missing columns: {', '.join(missing)}"
        )
    log.info(
        "Validated source schema — relation=%s required_columns=%d available_columns=%d",
        relation_name,
        len(required),
        len(columns),
    )


# IMPORTANT PROJECT EXCEPTION:
# These source-table performance indexes are intentionally created here, not in
# init/schema.sql. Reason: on large TimescaleDB hypertables, CREATE INDEX can
# take minutes. Running it here makes the container logs show exactly which
# index is currently being created/checked and how long it takes.
#
# Do not move these source-table index statements back to init/schema.sql unless
# the visible startup logging requirement is explicitly removed.
SOURCE_INDEXES_ARE_INTENTIONALLY_CREATED_IN_PYTHON_FOR_VISIBLE_STARTUP_LOGS = True


def ensure_source_indexes(conn: psycopg2.extensions.connection) -> None:
    """Create read-performance indexes on source tables, with explicit logs.

    Project exception: these are source-table performance helpers for the
    backtest runner, intentionally created here so the container logs show
    exactly which potentially long-running index operation is in progress.
    """
    index_statements = [
        (
            "idx_backtest_alpaca_market_data_1h_symbol_ts_cover",
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_backtest_alpaca_market_data_1h_symbol_ts_cover
                ON {} (symbol, ts DESC)
                INCLUDE (open, high, low, close, volume)
            """).format(relation_identifier(SOURCE_1H)),
        ),
        (
            "idx_backtest_safs_symbol_available_time_cover",
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_backtest_safs_symbol_available_time_cover
                ON {} (
                    symbol,
                    (COALESCE(data_available_at, fundamental_data_available_at, time)) DESC,
                    time DESC
                )
                INCLUDE (
                    composite_score, sector, industry, valuation_label, mispricing_score,
                    negative_earnings_flag, high_leverage_flag, market_cap_m,
                    current_price_currency, market_cap_currency, currency, financial_currency,
                    market_universe
                )
            """).format(relation_identifier(SOURCE_FUNDAMENTAL)),
        ),
        (
            "idx_backtest_safs_available_time_symbol_cover",
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_backtest_safs_available_time_symbol_cover
                ON {} (
                    (COALESCE(data_available_at, fundamental_data_available_at, time)) DESC,
                    time DESC,
                    symbol
                )
                INCLUDE (
                    composite_score, sector, industry, valuation_label, mispricing_score,
                    negative_earnings_flag, high_leverage_flag, market_cap_m,
                    current_price_currency, market_cap_currency, currency, financial_currency,
                    market_universe
                )
            """).format(relation_identifier(SOURCE_FUNDAMENTAL)),
        ),
        (
            "idx_backtest_pepperstone_symbol_ps",
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_backtest_pepperstone_symbol_ps
                ON {} (symbol)
                WHERE symbol_ps IS NOT NULL AND is_trading_enabled IS NOT FALSE
            """).format(relation_identifier(PEPPERSTONE_TABLE)),
        ),
        (
            "idx_backtest_pepperstone_symbol_ps24",
            sql.SQL("""
                CREATE INDEX IF NOT EXISTS idx_backtest_pepperstone_symbol_ps24
                ON {} (symbol)
                WHERE symbol_ps24 IS NOT NULL AND is_trading_enabled IS NOT FALSE
            """).format(relation_identifier(PEPPERSTONE_TABLE)),
        ),
    ]

    with conn.cursor() as cur:
        for index_name, statement in index_statements:
            log.info(
                "Ensuring source index %s — first run can take a while on large TimescaleDB tables; index_user=%s",
                index_name,
                SOURCE_INDEX_DB["user"],
            )
            started_at = _time.monotonic()
            try:
                cur.execute(statement)
                conn.commit()
            except psycopg2.errors.InsufficientPrivilege as exc:
                conn.rollback()
                raise RuntimeError(
                    "Cannot create source indexes with current SOURCE_INDEX_PGUSER="
                    f"{SOURCE_INDEX_DB['user']!r}. Use the owner/superuser of the source hypertables "
                    "(for example postgres-ts), or set ENSURE_SOURCE_INDEXES=false if the indexes already exist."
                ) from exc
            log.info(
                "Source index ready %s — elapsed=%.1fs",
                index_name,
                _time.monotonic() - started_at,
            )


def validate_source_schema(conn: psycopg2.extensions.connection) -> None:
    """Validate source tables/columns and basic date coverage before the run."""
    fundamental_required = {
        "time",
        "symbol",
        "data_available_at",
        "fundamental_data_available_at",
        "composite_score",
        "sector",
        "industry",
        "valuation_label",
        "mispricing_score",
        "negative_earnings_flag",
        "high_leverage_flag",
        "market_cap_m",
    }
    if REQUIRE_USD_FUNDAMENTALS:
        fundamental_required.update({
            "current_price_currency",
            "market_cap_currency",
            "currency",
            "financial_currency",
        })
    if FUNDAMENTAL_MARKET_UNIVERSE:
        fundamental_required.add("market_universe")

    _require_columns(conn, SOURCE_1H, {"symbol", "ts", "open", "high", "low", "close", "volume"})
    _require_columns(conn, SOURCE_FUNDAMENTAL, fundamental_required)
    _require_columns(conn, SOURCE_WORLD_REGIME, {"day", "regime_label", "composite_score"})

    universe = (SYMBOL_UNIVERSE or "all").strip().lower()
    if universe in {"pepperstone", "pepperstone24"}:
        symbol_column = "symbol_ps24" if universe == "pepperstone24" else "symbol_ps"
        _require_columns(conn, PEPPERSTONE_TABLE, {"symbol", symbol_column, "is_trading_enabled"})

    _validate_source_coverage(conn)


def _validate_source_coverage(conn: psycopg2.extensions.connection) -> None:
    start_ts = datetime.combine(START_DATE - timedelta(days=BAR_CACHE_WARMUP_DAYS), datetime.min.time(), tzinfo=timezone.utc)
    end_ts = _day_close_ts(END_DATE)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(ts), MAX(ts) FROM {} "
                "WHERE ts >= %s AND ts <= %s"
            ).format(relation_identifier(SOURCE_1H)),
            (start_ts, end_ts),
        )
        bar_min_ts, bar_max_ts = cur.fetchone()
    if bar_min_ts is None:
        raise RuntimeError(
            f"No 1h bars in {SOURCE_1H} for required window {start_ts} to {end_ts}"
        )
    log.info("Source coverage — bars relation=%s min_ts=%s max_ts=%s", SOURCE_1H, bar_min_ts, bar_max_ts)

    fundamental_where = [
        sql.SQL("time <= %s"),
        sql.SQL("COALESCE(data_available_at, fundamental_data_available_at, time) <= %s"),
    ]
    fundamental_params: list[object] = [end_ts, end_ts]
    if REQUIRE_USD_FUNDAMENTALS:
        fundamental_where.append(sql.SQL(
            "COALESCE(NULLIF(current_price_currency, ''), "
            "NULLIF(market_cap_currency, ''), "
            "NULLIF(currency, ''), "
            "NULLIF(financial_currency, ''), "
            "%s) = %s"
        ))
        fundamental_params.extend(["USD", "USD"])
    if FUNDAMENTAL_MARKET_UNIVERSE:
        fundamental_where.append(sql.SQL("market_universe = %s"))
        fundamental_params.append(FUNDAMENTAL_MARKET_UNIVERSE)

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(time), MAX(time), "
                "MAX(COALESCE(data_available_at, fundamental_data_available_at, time)) "
                "FROM {} WHERE {}"
            ).format(
                relation_identifier(SOURCE_FUNDAMENTAL),
                sql.SQL(" AND ").join(fundamental_where),
            ),
            fundamental_params,
        )
        fund_min_ts, fund_max_ts, fund_max_available_ts = cur.fetchone()
    if fund_min_ts is None:
        raise RuntimeError(
            f"No point-in-time fundamental rows in {SOURCE_FUNDAMENTAL} up to {end_ts}"
        )
    log.info(
        "Source coverage — fundamentals relation=%s min_time=%s max_time=%s max_available_at=%s",
        SOURCE_FUNDAMENTAL,
        fund_min_ts,
        fund_max_ts,
        fund_max_available_ts,
    )

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT MIN(day), MAX(day) FROM {} "
                "WHERE day <= %s AND composite_score IS NOT NULL"
            ).format(relation_identifier(SOURCE_WORLD_REGIME)),
            (END_DATE,),
        )
        regime_min_day, regime_max_day = cur.fetchone()
    if regime_min_day is None:
        raise RuntimeError(
            f"No world regime rows in {SOURCE_WORLD_REGIME} up to {END_DATE}"
        )
    log.info(
        "Source coverage — world_regime relation=%s min_day=%s max_day=%s",
        SOURCE_WORLD_REGIME,
        regime_min_day,
        regime_max_day,
    )

    universe = (SYMBOL_UNIVERSE or "all").strip().lower()
    if universe in {"pepperstone", "pepperstone24"}:
        symbol_column = "symbol_ps24" if universe == "pepperstone24" else "symbol_ps"
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT COUNT(*) FROM {} "
                    "WHERE {} IS NOT NULL AND is_trading_enabled IS NOT FALSE"
                ).format(relation_identifier(PEPPERSTONE_TABLE), sql.Identifier(symbol_column)),
            )
            tradable_symbols = cur.fetchone()[0]
        if tradable_symbols <= 0:
            raise RuntimeError(
                f"Symbol universe {SYMBOL_UNIVERSE} selected, but {PEPPERSTONE_TABLE}.{symbol_column} has no tradable rows"
            )
        log.info(
            "Source coverage — pepperstone relation=%s universe=%s tradable_symbols=%d",
            PEPPERSTONE_TABLE,
            universe,
            tradable_symbols,
        )


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
    start_h, start_m = _parse_hhmm(ENTRY_WINDOW_START)
    end_h, end_m = _parse_hhmm(ENTRY_WINDOW_END)
    current = local.hour * 60 + local.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


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


def simulate_outcome(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_date: date,
    equity: float,
) -> Optional[ClosedTrade]:
    """
    Check whether pos has closed by as_of_date.
    Returns ClosedTrade if closed, None if still open.

    TP logic: position is split 50/50 between TP1 and TP2.
    After TP1 hit, SL moves to entry (breakeven).

    Incremental: each call only scans bars newer than pos.last_bar_ts and
    resumes from the TP1/SL state stored on pos, making the loop O(N total)
    across all daily calls rather than O(N²).
    """
    after_ts = pos.last_bar_ts if pos.last_bar_ts is not None else pos.entry_ts
    bars = get_bars_range(conn, pos.symbol, after_ts, as_of_date)
    if not bars:
        return None

    tp1_hit = pos.tp1_hit
    tp1_price = pos.tp1_price
    tp1_exit_ts = pos.tp1_exit_ts
    effective_sl = pos.effective_sl
    is_long = pos.direction == "LONG"

    for bar_idx, (ts, _, high, low, close) in enumerate(bars):
        bar_date = ts.date() if hasattr(ts, "date") else ts
        total_bars = pos.bars_processed + bar_idx + 1

        if is_long:
            # SL check first (conservative — if same bar hits both, SL wins)
            if low <= effective_sl:
                price = effective_sl
                if tp1_hit:
                    pnl = _pnl_long(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
                    status = "HIT_TP1_THEN_BE"
                else:
                    pnl = _pnl_long(pos, price, price, split_exits=False)
                    status = "HIT_SL"
                return _make_trade(pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

            if not tp1_hit and high >= pos.take_profit_1:
                tp1_hit = True
                tp1_price = pos.take_profit_1
                tp1_exit_ts = ts
                effective_sl = pos.entry_price  # move SL to breakeven

            if tp1_hit and high >= pos.take_profit_2:
                price = pos.take_profit_2
                pnl = _pnl_long(pos, tp1_price, price)
                return _make_trade(pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)

        else:  # SHORT
            if high >= effective_sl:
                price = effective_sl
                if tp1_hit:
                    pnl = _pnl_short(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
                    status = "HIT_TP1_THEN_BE"
                else:
                    pnl = _pnl_short(pos, price, price, split_exits=False)
                    status = "HIT_SL"
                return _make_trade(pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

            if not tp1_hit and low <= pos.take_profit_1:
                tp1_hit = True
                tp1_price = pos.take_profit_1
                tp1_exit_ts = ts
                effective_sl = pos.entry_price

            if tp1_hit and low <= pos.take_profit_2:
                price = pos.take_profit_2
                pnl = _pnl_short(pos, tp1_price, price)
                return _make_trade(pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)

        if ts >= pos.valid_until:
            price = float(close)
            if is_long:
                pnl = _pnl_long(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            else:
                pnl = _pnl_short(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            status = "MAX_HOLD_TP1" if tp1_hit else "MAX_HOLD"
            return _make_trade(pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

    # Still open — persist incremental state for the next day's call
    pos.tp1_hit = tp1_hit
    pos.tp1_price = tp1_price
    pos.tp1_exit_ts = tp1_exit_ts
    pos.effective_sl = effective_sl
    pos.last_bar_ts = bars[-1][0]
    pos.bars_processed += len(bars)
    return None


def _pnl_long(pos: OpenPosition, tp1_price: float, tp2_price: float, split_exits: bool = True) -> float:
    tp1_shares = pos.shares * pos.tp1_close_ratio
    tp2_shares = pos.shares * (1.0 - pos.tp1_close_ratio)
    entry = _buy_fill(pos.entry_price)
    exit_1 = _sell_fill(tp1_price)
    exit_2 = _sell_fill(tp2_price)
    gross = tp1_shares * (exit_1 - entry) + tp2_shares * (exit_2 - entry)
    if split_exits:
        costs = _entry_cost(pos.shares, entry) + _exit_cost(tp1_shares, exit_1) + _exit_cost(tp2_shares, exit_2)
    else:
        costs = _entry_cost(pos.shares, entry) + _exit_cost(pos.shares, exit_2)
    return gross - costs


def _pnl_short(pos: OpenPosition, tp1_price: float, tp2_price: float, split_exits: bool = True) -> float:
    tp1_shares = pos.shares * pos.tp1_close_ratio
    tp2_shares = pos.shares * (1.0 - pos.tp1_close_ratio)
    entry = _sell_fill(pos.entry_price)
    exit_1 = _buy_fill(tp1_price)
    exit_2 = _buy_fill(tp2_price)
    gross = tp1_shares * (entry - exit_1) + tp2_shares * (entry - exit_2)
    if split_exits:
        costs = _entry_cost(pos.shares, entry) + _exit_cost(tp1_shares, exit_1) + _exit_cost(tp2_shares, exit_2)
    else:
        costs = _entry_cost(pos.shares, entry) + _exit_cost(pos.shares, exit_2)
    return gross - costs


def _buy_fill(mid_price: float) -> float:
    return mid_price * (1.0 + _execution_bps() / 10000.0)


def _sell_fill(mid_price: float) -> float:
    return mid_price * (1.0 - _execution_bps() / 10000.0)


def _execution_bps() -> float:
    return SPREAD_BPS * 0.5 + SLIPPAGE_BPS


def _entry_cost(shares: float, fill_price: float) -> float:
    return _order_cost(shares, fill_price)


def _exit_cost(shares: float, fill_price: float) -> float:
    return _order_cost(shares, fill_price)


def _order_cost(shares: float, fill_price: float) -> float:
    notional = abs(shares * fill_price)
    cost = (
        COMMISSION_PER_ORDER_USD
        + abs(shares) * COMMISSION_PER_SHARE_USD
        + notional * COMMISSION_BPS / 10000.0
    )
    if COMMISSION_MIN_PER_ORDER_USD > 0:
        cost = max(cost, COMMISSION_MIN_PER_ORDER_USD)
    if COMMISSION_MAX_PCT > 0 and notional > 0:
        cost = min(cost, notional * COMMISSION_MAX_PCT / 100.0)
    return cost


def _active_position_ratio(pos: OpenPosition, tp1_hit: Optional[bool] = None) -> float:
    if (tp1_hit if tp1_hit is not None else pos.tp1_hit):
        return max(0.0, 1.0 - pos.tp1_close_ratio)
    return 1.0


def _active_margin_used(pos: OpenPosition) -> float:
    return pos.margin_used * _active_position_ratio(pos)


def _latest_close_price(
    conn: psycopg2.extensions.connection,
    pos: OpenPosition,
    as_of_date: date,
) -> float:
    timestamps, bars = _load_symbol_bars(conn, pos.symbol)
    idx = bisect_right(timestamps, _day_close_ts(as_of_date)) - 1
    if idx < 0:
        return pos.entry_price
    return float(bars[idx].close)


def _open_position_mark_to_market_pnl(pos: OpenPosition, mark_price: float, as_of_date: date) -> float:
    if pos.direction == "LONG":
        if pos.tp1_hit:
            pnl = _pnl_long(pos, pos.tp1_price or pos.take_profit_1, mark_price, split_exits=True)
        else:
            pnl = _pnl_long(pos, mark_price, mark_price, split_exits=False)
    else:
        if pos.tp1_hit:
            pnl = _pnl_short(pos, pos.tp1_price or pos.take_profit_1, mark_price, split_exits=True)
        else:
            pnl = _pnl_short(pos, mark_price, mark_price, split_exits=False)
    return pnl - _financing_cost(pos, as_of_date, _day_close_ts(as_of_date), pos.tp1_exit_ts, pos.tp1_hit)


def _account_equity(
    conn: psycopg2.extensions.connection,
    open_positions: list[OpenPosition],
    realized_equity: float,
    as_of_date: date,
) -> float:
    open_pnl = 0.0
    for pos in open_positions:
        open_pnl += _open_position_mark_to_market_pnl(pos, _latest_close_price(conn, pos, as_of_date), as_of_date)
    return realized_equity + open_pnl


def _financing_days(start_ts: datetime, end_ts: datetime) -> float:
    return max(0.0, (end_ts - start_ts).total_seconds() / 86400.0)


def _financing_cost(
    pos: OpenPosition,
    outcome_date: date,
    exit_ts: Optional[datetime] = None,
    tp1_exit_ts: Optional[datetime] = None,
    tp1_hit: bool = False,
) -> float:
    borrowed_notional = max(0.0, pos.position_size_usd - pos.margin_used)
    if borrowed_notional <= 0.0:
        return 0.0

    end_ts = exit_ts or _day_close_ts(outcome_date)
    total_days = max(1.0, _financing_days(pos.entry_ts, end_ts))
    tp1_ts = tp1_exit_ts or pos.tp1_exit_ts
    if not tp1_hit or tp1_ts is None or tp1_ts >= end_ts:
        return borrowed_notional * MARGIN_FINANCING_RATE_PCT / 100.0 * total_days / 365.0

    before_days = _financing_days(pos.entry_ts, tp1_ts)
    after_days = _financing_days(tp1_ts, end_ts)
    actual_days = before_days + after_days
    if actual_days <= 0.0:
        before_days = total_days
        after_days = 0.0
    else:
        scale = total_days / actual_days
        before_days *= scale
        after_days *= scale

    remaining_borrowed = borrowed_notional * _active_position_ratio(pos, tp1_hit=True)
    financed_notional_days = borrowed_notional * before_days + remaining_borrowed * after_days
    return financed_notional_days * MARGIN_FINANCING_RATE_PCT / 100.0 / 365.0


def _make_trade(
    pos: OpenPosition,
    status: str,
    outcome_price: float,
    outcome_date: date,
    outcome_bars: int,
    tp1_hit: bool,
    pnl: float,
    equity: float,
    exit_ts: datetime = None,
    tp1_exit_ts: Optional[datetime] = None,
) -> ClosedTrade:
    pnl -= _financing_cost(pos, outcome_date, exit_ts, tp1_exit_ts, tp1_hit)
    return_pct = pnl / pos.position_size_usd * 100.0 if pos.position_size_usd else 0.0
    return ClosedTrade(
        position=pos,
        outcome_status=status,
        outcome_price=outcome_price,
        outcome_date=outcome_date,
        outcome_bars=outcome_bars,
        tp1_hit=tp1_hit,
        return_pct=round(return_pct, 4),
        pnl_usd=round(pnl, 2),
        equity_after=round(equity + pnl, 2),
        exit_ts=exit_ts,
        tp1_exit_ts=tp1_exit_ts,
    )


# ── Position sizing ───────────────────────────────────────────────────────────

def calc_position(
    signal: Signal,
    equity: float,
) -> tuple[float, float, float]:
    """Return (margin_used, shares, position_size_usd)."""
    risk_usd = equity * RISK_PER_TRADE_PCT / 100.0
    if signal.direction == "LONG":
        entry_fill = _buy_fill(signal.entry_price)
        stop_fill = _sell_fill(signal.stop_loss)
        loss_per_share_before_commission = entry_fill - stop_fill
    else:
        entry_fill = _sell_fill(signal.entry_price)
        stop_fill = _buy_fill(signal.stop_loss)
        loss_per_share_before_commission = stop_fill - entry_fill

    commission_per_share = (abs(entry_fill) + abs(stop_fill)) * COMMISSION_BPS / 10000.0
    loss_per_share = loss_per_share_before_commission + commission_per_share
    fixed_round_trip_cost = COMMISSION_PER_ORDER_USD * 2.0
    if loss_per_share <= 0 or risk_usd <= fixed_round_trip_cost:
        return 0.0, 0.0, 0.0

    shares = (risk_usd - fixed_round_trip_cost) / loss_per_share
    if not ALLOW_FRACTIONAL_SHARES:
        shares = float(int(shares))
        if shares < 1:
            return 0.0, 0.0, 0.0
    position_size_usd = abs(shares * entry_fill)
    margin_used = position_size_usd * MARGIN_REQUIREMENT_PCT / 100.0
    return margin_used, shares, position_size_usd


# ── DB writes ─────────────────────────────────────────────────────────────────

def create_run(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    long_max_hold_days: float,
    short_max_hold_days: float,
    tp1_close_ratio: float,
    notes: Optional[str] = None,
) -> int:
    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {_result_table("backtest_runs")} (
                start_date, end_date, notes, run_label, model_file,
                account_profile, account_currency,
                initial_equity, risk_per_trade_pct, max_open_positions,
                margin_requirement_pct, maintenance_margin_pct, min_free_margin_pct,
                allow_fractional_shares, spread_bps, slippage_bps,
                commission_per_order_usd, commission_per_share_usd,
                commission_min_per_order_usd, commission_max_pct,
                commission_bps, margin_financing_rate_pct,
                entry_window_enabled, entry_window_tz, entry_window_start, entry_window_end,
                long_max_score, short_min_score,
                long_min_fundamental, short_max_fundamental, min_market_cap_m,
                long_min_pullback, long_max_pullback, long_ideal_pullback, long_max_rsi,
                short_min_bounce, short_max_bounce, short_ideal_bounce, short_min_rsi, short_max_rsi,
                long_sl_buffer, short_sl_buffer,
                long_tp1_pct, long_tp2_pct, short_tp1_pct, short_tp2_pct,
                long_valid_days, short_valid_days, long_max_hold_days, short_max_hold_days,
                tp1_close_ratio
            ) VALUES (
                %s, %s, %s, to_char(NOW() AT TIME ZONE %s, 'YYYY-MM-DD HH24:MI'), %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s
            ) RETURNING run_id
            """,
            (
                START_DATE, END_DATE, notes if notes is not None else (RUN_NOTES or None), RUN_LABEL_TZ, MODEL_FILE,
                ACCOUNT_PROFILE, ACCOUNT_CURRENCY,
                INITIAL_EQUITY, RISK_PER_TRADE_PCT, MAX_OPEN_POSITIONS,
                MARGIN_REQUIREMENT_PCT, MAINTENANCE_MARGIN_PCT, MIN_FREE_MARGIN_PCT,
                ALLOW_FRACTIONAL_SHARES, SPREAD_BPS, SLIPPAGE_BPS,
                COMMISSION_PER_ORDER_USD, COMMISSION_PER_SHARE_USD,
                COMMISSION_MIN_PER_ORDER_USD, COMMISSION_MAX_PCT,
                COMMISSION_BPS, MARGIN_FINANCING_RATE_PCT,
                ENTRY_WINDOW_ENABLED, ENTRY_WINDOW_TZ, ENTRY_WINDOW_START, ENTRY_WINDOW_END,
                cfg.long_max_score, cfg.short_min_score,
                cfg.long_min_fundamental, cfg.short_max_fundamental, MIN_MARKET_CAP_M,
                cfg.long_min_pullback, cfg.long_max_pullback, cfg.long_ideal_pullback, cfg.long_max_rsi,
                cfg.short_min_bounce, cfg.short_max_bounce, cfg.short_ideal_bounce, cfg.short_min_rsi, cfg.short_max_rsi,
                cfg.long_sl_buffer, cfg.short_sl_buffer,
                cfg.long_tp1_pct, cfg.long_tp2_pct, cfg.short_tp1_pct, cfg.short_tp2_pct,
                cfg.long_valid_days, cfg.short_valid_days, long_max_hold_days, short_max_hold_days,
                tp1_close_ratio,
            ),
        )
        run_id = cur.fetchone()[0]
    conn.commit()
    log.info("Created run_id=%d model_file=%s account_profile=%s", run_id, MODEL_FILE, ACCOUNT_PROFILE)
    return run_id


def write_trades(
    conn: psycopg2.extensions.connection,
    run_id: int,
    trades: list[ClosedTrade],
) -> None:
    if not trades:
        return
    rows = []
    for t in trades:
        p = t.position
        s = p.signal
        rows.append((
            run_id,
            p.entry_date,
            p.symbol,
            p.direction,
            p.world_regime_label or None,
            Decimal(str(round(p.world_regime_score, 2))) if p.world_regime_score else None,
            p.valuation_label or None,
            Decimal(str(round(s.fundamental_score, 2))),
            Decimal(str(round(s.entry_score, 4))),
            Decimal(str(round(s.combined_score, 4))),
            Decimal(str(round(p.entry_price, 4))),
            Decimal(str(round(p.stop_loss, 4))),
            Decimal(str(round(p.take_profit_1, 4))),
            Decimal(str(round(p.take_profit_2, 4))),
            Decimal(str(round(s.pullback_pct, 2))),
            Decimal(str(round(s.rsi_1h, 2))),
            Decimal(str(round(s.volume_ratio, 3))),
            s.entry_reason,
            Decimal(str(round(p.position_size_usd, 2))),
            Decimal(str(round(p.shares, 6))),
            Decimal(str(round(p.margin_used, 2))),
            Decimal(str(round(p.equity_before, 2))),
            t.outcome_status,
            Decimal(str(round(t.outcome_price, 4))),
            t.outcome_date,
            t.outcome_bars,
            t.tp1_hit,
            Decimal(str(round(t.return_pct, 4))),
            Decimal(str(round(t.pnl_usd, 2))),
            Decimal(str(round(t.equity_after, 2))),
            p.entry_ts,
            t.tp1_exit_ts,
            t.exit_ts,
        ))

    query = """
        INSERT INTO {table} (
            run_id, signal_date, symbol, direction,
            world_regime_label, world_regime_score, valuation_label,
            fundamental_score, entry_score, combined_score,
            entry_price, stop_loss, take_profit_1, take_profit_2,
            pullback_pct, rsi_1h, volume_ratio, entry_reason,
            position_size_usd, shares, margin_used, equity_before,
            outcome_status, outcome_price, outcome_date, outcome_bars,
            tp1_hit, return_pct, pnl_usd, equity_after,
            entry_ts, tp1_exit_ts, exit_ts
        ) VALUES %s
        ON CONFLICT (run_id, signal_date, symbol) DO UPDATE SET
            world_regime_score = EXCLUDED.world_regime_score,
            outcome_status = EXCLUDED.outcome_status,
            outcome_price  = EXCLUDED.outcome_price,
            pnl_usd        = EXCLUDED.pnl_usd,
            equity_after   = EXCLUDED.equity_after,
            entry_ts       = EXCLUDED.entry_ts,
            tp1_exit_ts    = EXCLUDED.tp1_exit_ts,
            exit_ts        = EXCLUDED.exit_ts
    """.format(table=_result_table("backtest_trades"))
    with conn.cursor() as cur:
        execute_values(cur, query, rows, page_size=200)
    conn.commit()


def update_run_summary(
    conn: psycopg2.extensions.connection,
    run_id: int,
    trades: list[ClosedTrade],
    final_equity: float,
) -> None:
    if not trades:
        return

    wins      = [t for t in trades if t.pnl_usd > 0]
    losses    = [t for t in trades if t.pnl_usd < 0]
    breakevens= [t for t in trades if t.pnl_usd == 0]
    expired   = [t for t in trades if "MAX_HOLD" in t.outcome_status]

    win_rate  = len(wins) / len(trades) * 100.0 if trades else 0.0
    total_ret = (final_equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0
    avg_ret   = sum(t.return_pct for t in trades) / len(trades) if trades else 0.0
    avg_win   = sum(t.return_pct for t in wins)   / len(wins)   if wins   else 0.0
    avg_loss  = sum(t.return_pct for t in losses) / len(losses) if losses else 0.0

    gross_profit = sum(t.pnl_usd for t in wins)
    gross_loss   = abs(sum(t.pnl_usd for t in losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None

    # Max drawdown (peak-to-trough on equity_after series)
    equity_series = [INITIAL_EQUITY] + [t.equity_after for t in trades]
    peak = equity_series[0]
    max_dd = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    with conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE {_result_table("backtest_runs")} SET
                final_equity     = %s,
                total_trades     = %s,
                winning_trades   = %s,
                losing_trades    = %s,
                breakeven_trades = %s,
                expired_trades   = %s,
                win_rate_pct     = %s,
                total_return_pct = %s,
                max_drawdown_pct = %s,
                avg_return_pct   = %s,
                avg_win_pct      = %s,
                avg_loss_pct     = %s,
                profit_factor    = %s
            WHERE run_id = %s
            """,
            (
                round(final_equity, 2),
                len(trades), len(wins), len(losses), len(breakevens), len(expired),
                round(win_rate, 2), round(total_ret, 2), round(max_dd, 2),
                round(avg_ret, 4), round(avg_win, 4), round(avg_loss, 4),
                round(profit_factor, 4) if profit_factor else None,
                run_id,
            ),
        )
    conn.commit()


# ── Monte Carlo ───────────────────────────────────────────────────────────────

def run_monte_carlo(
    conn: psycopg2.extensions.connection,
    run_id: int,
    closed_trades: list[ClosedTrade],
    initial_equity: float,
    n_simulations: int = 2000,
) -> None:
    if n_simulations <= 0 or len(closed_trades) < 2:
        return

    # Use equity fraction (pnl / equity_before) so that compounding is preserved
    # when trades are reshuffled — correct for a fixed-%-risk strategy.
    fractions = np.array([t.pnl_usd / t.position.equity_before for t in closed_trades], dtype=np.float64)

    rng = np.random.default_rng()
    shuffled = np.tile(fractions, (n_simulations, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)

    equity_curves = np.empty((n_simulations, len(fractions) + 1), dtype=np.float64)
    equity_curves[:, 0] = initial_equity
    equity_curves[:, 1:] = initial_equity * np.cumprod(1.0 + shuffled, axis=1)

    final_equities = equity_curves[:, -1]
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdown_pct = (equity_curves - running_max) / running_max * 100.0
    max_drawdowns = drawdown_pct.min(axis=1)
    total_returns = (final_equities - initial_equity) / initial_equity * 100.0

    def p(arr: np.ndarray, pct: int) -> float:
        return round(float(np.percentile(arr, pct)), 4)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {_result_table("backtest_monte_carlo")} (
                run_id, n_simulations,
                final_equity_p05, final_equity_p25, final_equity_p50, final_equity_p75, final_equity_p95,
                max_drawdown_p05, max_drawdown_p25, max_drawdown_p50, max_drawdown_p75, max_drawdown_p95,
                total_return_p05, total_return_p25, total_return_p50, total_return_p75, total_return_p95,
                prob_of_ruin_pct, prob_profitable_pct,
                worst_final_equity, worst_max_drawdown_pct, best_final_equity
            ) VALUES (
                %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (run_id) DO UPDATE SET
                n_simulations          = EXCLUDED.n_simulations,
                final_equity_p05       = EXCLUDED.final_equity_p05,
                final_equity_p25       = EXCLUDED.final_equity_p25,
                final_equity_p50       = EXCLUDED.final_equity_p50,
                final_equity_p75       = EXCLUDED.final_equity_p75,
                final_equity_p95       = EXCLUDED.final_equity_p95,
                max_drawdown_p05       = EXCLUDED.max_drawdown_p05,
                max_drawdown_p25       = EXCLUDED.max_drawdown_p25,
                max_drawdown_p50       = EXCLUDED.max_drawdown_p50,
                max_drawdown_p75       = EXCLUDED.max_drawdown_p75,
                max_drawdown_p95       = EXCLUDED.max_drawdown_p95,
                total_return_p05       = EXCLUDED.total_return_p05,
                total_return_p25       = EXCLUDED.total_return_p25,
                total_return_p50       = EXCLUDED.total_return_p50,
                total_return_p75       = EXCLUDED.total_return_p75,
                total_return_p95       = EXCLUDED.total_return_p95,
                prob_of_ruin_pct       = EXCLUDED.prob_of_ruin_pct,
                prob_profitable_pct    = EXCLUDED.prob_profitable_pct,
                worst_final_equity     = EXCLUDED.worst_final_equity,
                worst_max_drawdown_pct = EXCLUDED.worst_max_drawdown_pct,
                best_final_equity      = EXCLUDED.best_final_equity,
                created_at             = NOW()
            """,
            (
                run_id, n_simulations,
                p(final_equities, 5),  p(final_equities, 25), p(final_equities, 50),
                p(final_equities, 75), p(final_equities, 95),
                p(max_drawdowns, 5),   p(max_drawdowns, 25),  p(max_drawdowns, 50),
                p(max_drawdowns, 75),  p(max_drawdowns, 95),
                p(total_returns, 5),   p(total_returns, 25),  p(total_returns, 50),
                p(total_returns, 75),  p(total_returns, 95),
                round(float(np.mean(final_equities < initial_equity * 0.5) * 100), 2),
                round(float(np.mean(final_equities > initial_equity) * 100), 2),
                round(float(final_equities.min()), 2),
                round(float(max_drawdowns.min()), 4),
                round(float(final_equities.max()), 2),
            ),
        )
    conn.commit()
    log.info(
        "Monte Carlo — run_id=%d  n=%d  return_p50=%.1f%%  return_p05=%.1f%%  dd_p05=%.1f%%  ruin=%.1f%%  profitable=%.1f%%",
        run_id, n_simulations,
        p(total_returns, 50), p(total_returns, 5), p(max_drawdowns, 5),
        float(np.mean(final_equities < initial_equity * 0.5) * 100),
        float(np.mean(final_equities > initial_equity) * 100),
    )


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    conn: psycopg2.extensions.connection,
    cfg: Any,
    long_max_hold_days: float = LONG_MAX_HOLD_DAYS,
    short_max_hold_days: float = SHORT_MAX_HOLD_DAYS,
    tp1_close_ratio: float = TP1_CLOSE_RATIO,
    notes: Optional[str] = None,
) -> tuple[int, dict]:
    run_id = create_run(conn, cfg, long_max_hold_days, short_max_hold_days, tp1_close_ratio, notes)

    equity: float = INITIAL_EQUITY
    open_positions: list[OpenPosition] = []
    closed_trades: list[ClosedTrade] = []

    trading_days = get_trading_days(conn, START_DATE, END_DATE)
    log.info("Trading days to simulate: %d (%s → %s)", len(trading_days), START_DATE, END_DATE)

    # Diagnostic counters
    days_no_regime = 0
    days_neutral   = 0
    days_no_candidates = 0
    days_no_signals    = 0
    days_with_signals  = 0

    for day_idx, day in enumerate(trading_days, start=1):
        log_progress_today = day_idx == 1 or day_idx == len(trading_days) or day_idx % PROGRESS_LOG_EVERY_DAYS == 0

        # ── 1. Close positions that resolved today ──────────────────────────
        still_open = []
        closed_today = 0
        day_pnl = 0.0
        for pos in open_positions:
            trade = simulate_outcome(conn, pos, day, equity)
            if trade is not None:
                equity = trade.equity_after
                closed_trades.append(trade)
                closed_today += 1
                day_pnl += trade.pnl_usd
                log.debug("Closed %-6s %s %s  pnl=%.0f  equity=%.0f",
                          pos.symbol, pos.direction, trade.outcome_status,
                          trade.pnl_usd, equity)
            else:
                still_open.append(pos)
        open_positions = still_open

        # ── 2. Generate signals for today ───────────────────────────────────
        regime = get_world_regime(conn, source_table=SOURCE_WORLD_REGIME, as_of_date=day)
        if not regime:
            days_no_regime += 1
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s — no regime  day_pnl=%.0f  equity=%.0f  open=%d  closed_today=%d  closed_total=%d",
                    day_idx, len(trading_days), day, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        if regime.score < cfg.long_max_score:
            direction = "LONG"
        elif regime.score >= cfg.short_min_score:
            direction = "SHORT"
        else:
            days_neutral += 1
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s — neutral regime %.1f  day_pnl=%.0f  equity=%.0f  open=%d  closed_today=%d  closed_total=%d",
                    day_idx, len(trading_days), day, regime.score, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        day_end_ts = _day_signal_cutoff_ts(day)
        candidates = get_candidates(
            conn, direction,
            long_min_fundamental=cfg.long_min_fundamental,
            short_max_fundamental=cfg.short_max_fundamental,
            min_market_cap_m=MIN_MARKET_CAP_M,
            source_table=SOURCE_FUNDAMENTAL,
            as_of_date=day,
            as_of_ts=day_end_ts,
            long_label_blocklist=cfg.long_label_blocklist or None,
            short_label_blocklist=cfg.short_label_blocklist or None,
            symbol_universe=SYMBOL_UNIVERSE,
            pepperstone_table=PEPPERSTONE_TABLE,
            required_currency="USD" if REQUIRE_USD_FUNDAMENTALS else None,
            fundamental_market_universe=FUNDAMENTAL_MARKET_UNIVERSE or None,
        )

        if not candidates:
            days_no_candidates += 1
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s — %s regime %.1f  no candidates  day_pnl=%.0f  equity=%.0f  open=%d  closed_today=%d  closed_total=%d",
                    day_idx, len(trading_days), day, direction, regime.score, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        candidate_symbols = [fundamental.symbol for fundamental in candidates]
        preload_symbol_bars(conn, candidate_symbols)

        model = get_model_module()
        compute_fn = model.compute_long_signal if direction == "LONG" else model.compute_short_signal
        signals: list[Signal] = []
        skipped_no_bars = 0

        for fundamental in candidates:
            bars = get_cached_bars(
                conn, fundamental.symbol,
                cfg.min_bars + cfg.price_lookback_bars,
                up_to_ts=day_end_ts,
            )
            if len(bars) < cfg.min_bars:
                skipped_no_bars += 1
                continue
            signal = compute_fn(bars, fundamental, datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc), cfg)
            if signal:
                signal.entry_ts = bars[-1].ts
                signals.append(signal)

        signals.sort(key=lambda s: s.combined_score, reverse=True)

        if signals:
            days_with_signals += 1
        else:
            days_no_signals += 1

        # ── 3. Open new positions ────────────────────────────────────────────
        open_symbols = {p.symbol for p in open_positions}
        open_sectors: set[str] = {p.signal.sector for p in open_positions if p.signal.sector}
        open_sector_industries: set[tuple[str, str]] = {
            (p.signal.sector, p.signal.industry)
            for p in open_positions
            if p.signal.sector
        }

        def _sector_tier(s: Signal) -> int:
            if not s.sector or s.sector not in open_sectors:
                return 0  # new sector — preferred
            if (s.sector, s.industry) not in open_sector_industries:
                return 1  # same sector, different industry — acceptable
            return 2      # same sector and industry — last resort

        signals.sort(key=lambda s: (_sector_tier(s), -s.combined_score))
        opened_today = 0
        account_equity_today = _account_equity(conn, open_positions, equity, day)
        used_margin = sum(_active_margin_used(p) for p in open_positions)

        for signal in signals:
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                break
            if signal.symbol in open_symbols:
                continue

            if account_equity_today <= 0:
                break

            margin_used, shares, position_size_usd = calc_position(signal, account_equity_today)
            if position_size_usd <= 0:
                continue

            free_margin = account_equity_today - used_margin
            free_margin_after = free_margin - margin_used

            if free_margin_after < 0:
                continue
            if free_margin_after < account_equity_today * MIN_FREE_MARGIN_PCT / 100.0:
                continue

            open_positions.append(OpenPosition(
                symbol=signal.symbol,
                direction=signal.direction,
                entry_date=day,
                entry_ts=signal.entry_ts or day_end_ts,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                effective_sl=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                valid_until=(signal.entry_ts or day_end_ts) + timedelta(
                    days=long_max_hold_days if signal.direction == "LONG" else short_max_hold_days
                ),
                tp1_close_ratio=tp1_close_ratio,
                shares=shares,
                position_size_usd=position_size_usd,
                margin_used=margin_used,
                equity_before=account_equity_today,
                signal=signal,
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                valuation_label=signal.valuation_label,
            ))
            open_symbols.add(signal.symbol)
            used_margin += margin_used
            opened_today += 1
            log.debug("Opened  %-6s %s  entry=%.2f  sl=%.2f  margin=%.0f  equity=%.0f",
                      signal.symbol, signal.direction, signal.entry_price,
                      signal.stop_loss, margin_used, equity)

        if log_progress_today or opened_today > 0:
            log.info(
                "Progress %d/%d %s — %s regime %.1f  candidates=%d  signals=%d  skipped_no_bars=%d  opened=%d  closed_today=%d  day_pnl=%.0f  open=%d  equity=%.0f  closed_total=%d",
                day_idx,
                len(trading_days),
                day,
                direction,
                regime.score,
                len(candidates),
                len(signals),
                skipped_no_bars,
                opened_today,
                closed_today,
                day_pnl,
                len(open_positions),
                equity,
                len(closed_trades),
            )

    log.info(
        "Day breakdown — no_regime=%d  neutral=%d  no_candidates=%d  no_signals=%d  with_signals=%d",
        days_no_regime, days_neutral, days_no_candidates, days_no_signals, days_with_signals,
    )
    log_cache_stats()

    # ── 4. Force-close remaining open positions at last available price ──────
    last_day = trading_days[-1] if trading_days else END_DATE
    for pos in open_positions:
        bars = get_bars_range(conn, pos.symbol, pos.entry_ts, last_day)
        last_price = float(bars[-1][4]) if bars else pos.entry_price
        if pos.direction == "LONG":
            if pos.tp1_hit:
                pnl = _pnl_long(pos, pos.tp1_price or pos.take_profit_1, last_price, split_exits=True)
            else:
                pnl = _pnl_long(pos, last_price, last_price, split_exits=False)
        else:
            if pos.tp1_hit:
                pnl = _pnl_short(pos, pos.tp1_price or pos.take_profit_1, last_price, split_exits=True)
            else:
                pnl = _pnl_short(pos, last_price, last_price, split_exits=False)
        trade = _make_trade(
            pos,
            "FORCE_CLOSED",
            last_price,
            last_day,
            len(bars) if bars else 0,
            pos.tp1_hit,
            pnl,
            equity,
            _day_close_ts(last_day),
            pos.tp1_exit_ts,
        )
        equity = trade.equity_after
        closed_trades.append(trade)

    # ── 5. Persist results ───────────────────────────────────────────────────
    log.info("Writing %d trades for run_id=%d", len(closed_trades), run_id)

    # Patch world_regime_label into rows (stored on signal, pass through)
    # (already embedded in entry_reason; trade write accesses signal directly)
    write_trades(conn, run_id, closed_trades)
    update_run_summary(conn, run_id, closed_trades, equity)
    if MONTE_CARLO_ENABLED:
        run_monte_carlo(conn, run_id, closed_trades, INITIAL_EQUITY, N_MONTE_CARLO_SIMULATIONS)

    n_trades = len(closed_trades)
    n_wins = sum(1 for t in closed_trades if t.pnl_usd > 0)
    n_losses = sum(1 for t in closed_trades if t.pnl_usd < 0)
    gross_profit = sum(t.pnl_usd for t in closed_trades if t.pnl_usd > 0)
    gross_loss = abs(sum(t.pnl_usd for t in closed_trades if t.pnl_usd < 0))
    total_return = (equity - INITIAL_EQUITY) / INITIAL_EQUITY * 100.0

    equity_series = [INITIAL_EQUITY] + [t.equity_after for t in closed_trades]
    peak = equity_series[0]
    max_dd = 0.0
    for eq in equity_series:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    summary = {
        "run_id": run_id,
        "total_trades": n_trades,
        "win_rate_pct": n_wins / n_trades * 100.0 if n_trades else 0.0,
        "total_return_pct": total_return,
        "max_drawdown_pct": max_dd,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else None,
    }

    log.info(
        "Run %d complete — trades=%d  wins=%d  final_equity=%.0f  return=%.1f%%",
        run_id, n_trades, n_wins, equity, total_return,
    )
    return run_id, summary


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(conn: psycopg2.extensions.connection, base_cfg: Any) -> list[dict]:
    model = get_model_module()
    if not hasattr(model, "iter_grid_search_configs"):
        raise RuntimeError(
            f"Backtesting model {MODEL_FILE} does not define iter_grid_search_configs(). "
            "Either disable GRID_SEARCH_ENABLED or add the grid hook to the model file."
        )

    grid_items = list(model.iter_grid_search_configs(
        base_cfg=base_cfg,
        parse_grid_vals=_parse_grid_vals,
        parse_hold_grid_vals=_parse_hold_grid_vals,
        long_max_hold_days=LONG_MAX_HOLD_DAYS,
        short_max_hold_days=SHORT_MAX_HOLD_DAYS,
        tp1_close_ratio=TP1_CLOSE_RATIO,
    ))

    total = len(grid_items)
    log.info("Grid search — model=%s combinations=%d", MODEL_FILE, total)

    results: list[dict] = []
    for i, item in enumerate(grid_items, 1):
        cfg = item["config"]
        lmhd = item.get("long_max_hold_days", LONG_MAX_HOLD_DAYS)
        smhd = item.get("short_max_hold_days", SHORT_MAX_HOLD_DAYS)
        tcr = item.get("tp1_close_ratio", TP1_CLOSE_RATIO)
        run_notes = item.get("notes", f"grid model={MODEL_FILE} idx={i}")
        log.info("Grid %d/%d — %s", i, total, run_notes)
        _, summary = run_backtest(conn, cfg, lmhd, smhd, tcr, notes=run_notes)
        summary.update(item.get("summary", {}))
        results.append(summary)

    return results


def _print_grid_summary(results: list[dict]) -> None:
    if not results:
        log.info("Grid search produced no results.")
        return

    model = get_model_module()
    if hasattr(model, "log_grid_summary"):
        model.log_grid_summary(log, results)
        return

    required_keys = {
        "long_tp1_pct",
        "long_tp2_pct",
        "short_tp1_pct",
        "short_tp2_pct",
        "long_max_hold_days",
        "short_max_hold_days",
        "tp1_close_ratio",
    }
    if any(required_keys - set(result.keys()) for result in results):
        ranked_generic = sorted(
            results,
            key=lambda r: (r["profit_factor"] or 0.0, r["total_return_pct"]),
            reverse=True,
        )
        log.info("Grid search results (generic summary, sorted by profit factor):")
        for r in ranked_generic:
            log.info(
                "run_id=%d trades=%d win_rate=%.1f%% return=%.2f%% dd=%.2f%% profit_factor=%s",
                r["run_id"],
                r["total_trades"],
                r["win_rate_pct"],
                r["total_return_pct"],
                r["max_drawdown_pct"],
                f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "N/A",
            )
        return

    ranked = sorted(
        results,
        key=lambda r: (r["profit_factor"] or 0.0, r["total_return_pct"]),
        reverse=True,
    )

    header = (
        f"{'run_id':>7}  {'ltp1':>5}  {'ltp2':>5}  {'stp1':>5}  {'stp2':>5}  "
        f"{'lmhd':>4}  {'smhd':>4}  {'tcr':>4}  {'trades':>6}  {'wr%':>5}  "
        f"{'ret%':>7}  {'dd%':>6}  {'PF':>5}"
    )
    sep = "-" * len(header)
    log.info("Grid search results (sorted by profit factor):\n%s\n%s", header, sep)
    for r in ranked:
        pf = f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "  N/A"
        log.info(
            "%7d  %5.3f  %5.3f  %5.3f  %5.3f  %4.1f  %4.1f  %4.2f  %6d  %5.1f  %7.2f  %6.2f  %5s",
            r["run_id"],
            r["long_tp1_pct"], r["long_tp2_pct"],
            r["short_tp1_pct"], r["short_tp2_pct"],
            r["long_max_hold_days"], r["short_max_hold_days"], r["tp1_close_ratio"],
            r["total_trades"],
            r["win_rate_pct"],
            r["total_return_pct"],
            r["max_drawdown_pct"],
            pf,
        )
    best = ranked[0]
    log.info(
        "Best combination — run_id=%d  PF=%s  return=%.2f%%  dd=%.2f%%  "
        "ltp1=%.3f  ltp2=%.3f  stp1=%.3f  stp2=%.3f  lmhd=%.1f  smhd=%.1f  tcr=%.2f",
        best["run_id"],
        f"{best['profit_factor']:.3f}" if best["profit_factor"] else "N/A",
        best["total_return_pct"], best["max_drawdown_pct"],
        best["long_tp1_pct"], best["long_tp2_pct"],
        best["short_tp1_pct"], best["short_tp2_pct"],
        best["long_max_hold_days"], best["short_max_hold_days"], best["tp1_close_ratio"],
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    global _MODEL_MODULE
    _MODEL_MODULE = load_model_module()
    cfg = _MODEL_MODULE.signal_config_from_env()
    conn = connect_with_retry()
    try:
        log.info("Connected. Starting backtest %s → %s, equity=%.0f", START_DATE, END_DATE, INITIAL_EQUITY)
        validate_source_schema(conn)
        if ENSURE_SOURCE_INDEXES:
            index_conn = connect_source_index_with_retry()
            try:
                ensure_source_indexes(index_conn)
            finally:
                index_conn.close()
        else:
            log.info("Source index check skipped — ENSURE_SOURCE_INDEXES=false")
        log.info(
            "Source tables — bars=%s  fundamentals=%s  world_regime=%s  symbol_universe=%s  pepperstone_table=%s",
            SOURCE_1H,
            SOURCE_FUNDAMENTAL,
            SOURCE_WORLD_REGIME,
            SYMBOL_UNIVERSE,
            PEPPERSTONE_TABLE,
        )
        log.info("Backtesting model — file=%s  dir=%s", MODEL_FILE, MODEL_DIR)
        log.info(
            "Account profile — profile=%s  currency=%s  margin_requirement_pct=%.2f  maintenance_margin_pct=%.2f",
            ACCOUNT_PROFILE,
            ACCOUNT_CURRENCY,
            MARGIN_REQUIREMENT_PCT,
            MAINTENANCE_MARGIN_PCT,
        )
        log.info(
            "Execution model — fractional_shares=%s  spread_bps=%.2f  slippage_bps=%.2f  commission_per_order=%.2f  commission_per_share=%.4f  commission_min=%.2f  commission_max_pct=%.2f  commission_bps=%.2f  margin_financing_rate_pct=%.2f",
            ALLOW_FRACTIONAL_SHARES,
            SPREAD_BPS,
            SLIPPAGE_BPS,
            COMMISSION_PER_ORDER_USD,
            COMMISSION_PER_SHARE_USD,
            COMMISSION_MIN_PER_ORDER_USD,
            COMMISSION_MAX_PCT,
            COMMISSION_BPS,
            MARGIN_FINANCING_RATE_PCT,
        )
        log.info(
            "Entry window — enabled=%s  tz=%s  start=%s  end=%s  sl_tp_window=all_bars",
            ENTRY_WINDOW_ENABLED,
            ENTRY_WINDOW_TZ,
            ENTRY_WINDOW_START,
            ENTRY_WINDOW_END,
        )
        log.info(
            "Holding rule — long_max_hold_days=%.2f  short_max_hold_days=%.2f  sl_tp_active_from=next_1h_bar",
            LONG_MAX_HOLD_DAYS,
            SHORT_MAX_HOLD_DAYS,
        )
        log.info(
            "Candidate filter — min_market_cap_m=%.0f  require_usd_fundamentals=%s  fundamental_market_universe=%s",
            MIN_MARKET_CAP_M,
            REQUIRE_USD_FUNDAMENTALS,
            FUNDAMENTAL_MARKET_UNIVERSE or "all",
        )
        log.info("Grid search — enabled=%s", GRID_SEARCH_ENABLED)
        log.info(
            "Performance caches — trading_days=on  world_regime=on  candidates=on  bars=on  ensure_source_indexes=%s",
            ENSURE_SOURCE_INDEXES,
        )
        if GRID_SEARCH_ENABLED:
            results = run_grid_search(conn, cfg)
            _print_grid_summary(results)
        else:
            run_backtest(conn, cfg, LONG_MAX_HOLD_DAYS, SHORT_MAX_HOLD_DAYS, TP1_CLOSE_RATIO)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

