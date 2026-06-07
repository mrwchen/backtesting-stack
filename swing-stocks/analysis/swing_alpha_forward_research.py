"""Point-in-time scorer research for swing-stock candidates.

This script is deliberately outside the backtest model loop. It asks the same
PIT-safe candidate selector that the backtest uses, then checks whether score
deciles explain later 5/10/20/60 day returns, MAE, and MFE.

Example:
    python analysis/swing_alpha_forward_research.py --frequency daily --write-candidates
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time as time_module
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Iterable

os.environ.setdefault("PGAPPNAME", "swing_alpha_forward_research")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_models.swing_alpha_momentum_v1 import (
    IntentConfig,
    _long_price_alpha,
    _long_scorer_alpha,
    _short_price_alpha,
    _short_scorer_alpha,
    required_bar_lookback,
)
from backtest_shared import Bar, FundamentalRow, InstrumentKey, instrument_key

LOG = logging.getLogger("swing_alpha_forward_research")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else int(value)


def _env_optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None or value.strip() == "" else float(value)


def _env_list(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return list(default)
    return [part.strip() for part in value.split(",") if part.strip()]


DEFAULT_START_DATE = date.fromisoformat(os.getenv("START_DATE", "2023-01-01"))
DEFAULT_END_DATE = date.fromisoformat(os.getenv("END_DATE", date.today().isoformat()))
DEFAULT_LONG_MIN_FUNDAMENTAL = _env_float("COMMON_LONG_MIN_FUNDAMENTAL", 62.0)
DEFAULT_SHORT_MAX_FUNDAMENTAL = _env_float("COMMON_SHORT_MAX_FUNDAMENTAL", 42.0)
DEFAULT_LONG_LABEL_BLOCKLIST = _env_list("COMMON_LONG_LABEL_BLOCKLIST", ["value_trap", "overvalued", "overvalued_weak"])
DEFAULT_SHORT_LABEL_BLOCKLIST = _env_list("COMMON_SHORT_LABEL_BLOCKLIST", ["deep_value", "quality_value", "compounder"])
DEFAULT_MIN_MARKET_CAP_M = _env_float("COMMON_MIN_MARKET_CAP_USD_M", 1000.0)
DEFAULT_FILTER_HIGH_LEVERAGE = _env_bool("COMMON_FILTER_FUNDAMENTAL_HIGH_LEVERAGE", True)
DEFAULT_FILTER_NEGATIVE_EARNINGS_LONG = _env_bool("COMMON_FILTER_NEGATIVE_EARNINGS_LONG", False)
DEFAULT_FILTER_NEGATIVE_EARNINGS_SHORT = _env_bool("COMMON_FILTER_NEGATIVE_EARNINGS_SHORT", False)
DEFAULT_FILTER_SCORER_ELIGIBILITY = _env_bool("COMMON_FILTER_SCORER_ELIGIBILITY", False)
DEFAULT_REQUIRE_UPCOMING_EARNINGS_DATE = _env_bool("COMMON_REQUIRE_UPCOMING_EARNINGS_DATE", False)
DEFAULT_REQUIRE_USD_FUNDAMENTALS = _env_bool("REQUIRE_USD_FUNDAMENTALS", True)
DEFAULT_MARKET_TABLE = os.getenv("SOURCE_MARKET_DATA_1H_TABLE", "alpaca_market_data_1h")
DEFAULT_FUNDAMENTAL_TABLE = os.getenv("SOURCE_FUNDAMENTAL_SCORES_TABLE", "stock_scorer_fundamental_scores")
DEFAULT_PEPPERSTONE_TABLE = os.getenv("PS_TRADABLE_SYMBOLS_TABLE", "public.pepperstone_data")
DEFAULT_WORLD_REGIME_TABLE = os.getenv("SOURCE_WORLD_REGIME_TABLE", "world_regime_daily_scores_mv")
DEFAULT_MARKET_REGIME_TABLE = os.getenv("SOURCE_MARKET_REGIME_TABLE", "alpaca_market_data_1h_daily_regime_scores")
DEFAULT_MARKET_REGIME_LOOKBACK_DAYS = _env_int("MARKET_REGIME_LOOKBACK_DAYS", 60)
DEFAULT_FUNDAMENTAL_SCORE_MODE = os.getenv("FUNDAMENTAL_SCORE_MODE", "peer").strip().lower()
DEFAULT_FUNDAMENTAL_PEER_WEIGHT = _env_float("FUNDAMENTAL_PEER_WEIGHT", 1.0)
DEFAULT_FUNDAMENTAL_ABS_WEIGHT = _env_float("FUNDAMENTAL_ABS_WEIGHT", 0.0)
DEFAULT_LONG_MIN_ABSOLUTE_SCORE = _env_optional_float("LONG_MIN_ABSOLUTE_SCORE")
DEFAULT_SHORT_MAX_ABSOLUTE_SCORE = _env_optional_float("SHORT_MAX_ABSOLUTE_SCORE")


@dataclass(frozen=True)
class RegimeContext:
    world_regime_label: str = "UNKNOWN"
    world_regime_score: float | None = None
    market_regime_label: str = "UNKNOWN"
    market_regime_lookback_days: int | None = None
    market_regime_trend_up_score: float | None = None
    market_regime_trend_down_score: float | None = None
    market_regime_range_score: float | None = None
    market_regime_unclear_score: float | None = None
    market_regime_atr_pct: float | None = None
    market_regime_close_change_pct: float | None = None


@dataclass(frozen=True)
class CandidateMetrics:
    day: date
    direction: str
    as_of_ts: datetime
    fundamental: FundamentalRow
    regime: RegimeContext
    history_close: float
    scorer_alpha: float
    price_alpha: float
    swing_alpha: float
    price_metrics: dict[str, float]
    score_values: dict[str, float]
    score_deciles: dict[str, int]


@dataclass(frozen=True)
class ForwardMetrics:
    horizon_days: int
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    return_pct: float
    mae_pct: float
    mfe_pct: float
    benchmark_return_pct: float | None
    excess_benchmark_pct: float | None


@dataclass
class StatBucket:
    count: int = 0
    score_sum: float = 0.0
    return_sum: float = 0.0
    excess_sum: float = 0.0
    mae_sum: float = 0.0
    mfe_sum: float = 0.0
    win_count: int = 0
    excess_win_count: int = 0
    returns: list[float] | None = None
    excess_returns: list[float] | None = None

    def add(self, score: float, forward: ForwardMetrics) -> None:
        self.count += 1
        self.score_sum += score
        self.return_sum += forward.return_pct
        self.mae_sum += forward.mae_pct
        self.mfe_sum += forward.mfe_pct
        if forward.return_pct > 0.0:
            self.win_count += 1
        if self.returns is None:
            self.returns = []
        self.returns.append(forward.return_pct)
        if forward.excess_benchmark_pct is not None:
            self.excess_sum += forward.excess_benchmark_pct
            if forward.excess_benchmark_pct > 0.0:
                self.excess_win_count += 1
            if self.excess_returns is None:
                self.excess_returns = []
            self.excess_returns.append(forward.excess_benchmark_pct)

    def as_row(self, score_name: str, direction: str, horizon_days: int, decile: int) -> dict[str, object]:
        avg_mae = self.mae_sum / self.count if self.count else 0.0
        avg_mfe = self.mfe_sum / self.count if self.count else 0.0
        excess_count = len(self.excess_returns or [])
        return {
            "direction": direction,
            "score_name": score_name,
            "horizon_days": horizon_days,
            "decile": decile,
            "count": self.count,
            "avg_score": self.score_sum / self.count if self.count else None,
            "avg_return_pct": self.return_sum / self.count if self.count else None,
            "median_return_pct": median(self.returns or []) if self.returns else None,
            "win_rate_pct": self.win_count / self.count * 100.0 if self.count else None,
            "avg_excess_benchmark_pct": self.excess_sum / excess_count if excess_count else None,
            "median_excess_benchmark_pct": median(self.excess_returns or []) if self.excess_returns else None,
            "positive_excess_rate_pct": self.excess_win_count / excess_count * 100.0 if excess_count else None,
            "avg_mae_pct": avg_mae if self.count else None,
            "avg_mfe_pct": avg_mfe if self.count else None,
            "avg_mfe_to_abs_mae": avg_mfe / abs(avg_mae) if avg_mae < 0.0 else None,
            "avg_mfe_minus_abs_mae_pct": avg_mfe - abs(avg_mae) if self.count else None,
        }


@dataclass
class SliceBucket:
    count: int = 0
    price_momentum_sum: float = 0.0
    entry_pullback_sum: float = 0.0
    price_alpha_sum: float = 0.0
    scorer_alpha_sum: float = 0.0
    swing_alpha_sum: float = 0.0
    return_sum: float = 0.0
    excess_sum: float = 0.0
    mae_sum: float = 0.0
    mfe_sum: float = 0.0
    win_count: int = 0
    excess_win_count: int = 0
    returns: list[float] | None = None
    excess_returns: list[float] | None = None

    def add(self, metric: CandidateMetrics, forward: ForwardMetrics, entry_pullback_pct: float) -> None:
        self.count += 1
        self.price_momentum_sum += metric.score_values.get("directional_price_momentum", 0.0)
        self.entry_pullback_sum += entry_pullback_pct
        self.price_alpha_sum += metric.price_alpha
        self.scorer_alpha_sum += metric.scorer_alpha
        self.swing_alpha_sum += metric.swing_alpha
        self.return_sum += forward.return_pct
        self.mae_sum += forward.mae_pct
        self.mfe_sum += forward.mfe_pct
        if forward.return_pct > 0.0:
            self.win_count += 1
        if self.returns is None:
            self.returns = []
        self.returns.append(forward.return_pct)
        if forward.excess_benchmark_pct is not None:
            self.excess_sum += forward.excess_benchmark_pct
            if forward.excess_benchmark_pct > 0.0:
                self.excess_win_count += 1
            if self.excess_returns is None:
                self.excess_returns = []
            self.excess_returns.append(forward.excess_benchmark_pct)

    def as_row(
        self,
        direction: str,
        horizon_days: int,
        regime_source: str,
        regime_label: str,
        price_momentum_decile: int,
        entry_pullback_bucket: str,
    ) -> dict[str, object]:
        excess_count = len(self.excess_returns or [])
        avg_mae = self.mae_sum / self.count if self.count else 0.0
        avg_mfe = self.mfe_sum / self.count if self.count else 0.0
        return {
            "direction": direction,
            "horizon_days": horizon_days,
            "regime_source": regime_source,
            "regime_label": regime_label,
            "price_momentum_decile": price_momentum_decile,
            "entry_pullback_bucket": entry_pullback_bucket,
            "count": self.count,
            "avg_directional_price_momentum": self.price_momentum_sum / self.count if self.count else None,
            "avg_entry_pullback_pct": self.entry_pullback_sum / self.count if self.count else None,
            "avg_price_alpha": self.price_alpha_sum / self.count if self.count else None,
            "avg_scorer_alpha": self.scorer_alpha_sum / self.count if self.count else None,
            "avg_swing_alpha": self.swing_alpha_sum / self.count if self.count else None,
            "avg_return_pct": self.return_sum / self.count if self.count else None,
            "median_return_pct": median(self.returns or []) if self.returns else None,
            "win_rate_pct": self.win_count / self.count * 100.0 if self.count else None,
            "avg_excess_benchmark_pct": self.excess_sum / excess_count if excess_count else None,
            "median_excess_benchmark_pct": median(self.excess_returns or []) if self.excess_returns else None,
            "positive_excess_rate_pct": self.excess_win_count / excess_count * 100.0 if excess_count else None,
            "avg_mae_pct": avg_mae if self.count else None,
            "avg_mfe_pct": avg_mfe if self.count else None,
            "avg_mfe_to_abs_mae": avg_mfe / abs(avg_mae) if avg_mae < 0.0 else None,
            "avg_mfe_minus_abs_mae_pct": avg_mfe - abs(avg_mae) if self.count else None,
        }


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    formatter.converter = time_module.gmtime
    handler.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level.upper()), handlers=[handler], force=True)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def parse_horizons(value: str) -> list[int]:
    horizons = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not horizons or any(item <= 0 for item in horizons):
        raise argparse.ArgumentTypeError("--horizons must contain positive integers")
    return horizons


def parse_directions(value: str) -> list[str]:
    directions = [part.strip().upper() for part in value.split(",") if part.strip()]
    invalid = [item for item in directions if item not in {"LONG", "SHORT"}]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unsupported directions: {', '.join(invalid)}")
    return list(dict.fromkeys(directions))


def utc_as_of(day: date, hour: int, minute: int) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=minute, tzinfo=timezone.utc))


def to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


def fetch_trading_days(conn, market_table: str, calendar_symbol: str, start_date: date, end_date: date) -> list[date]:
    from psycopg2 import sql

    from backtest_core.sql_utils import relation_identifier

    query = sql.SQL(
        """
        SELECT DISTINCT ts::date AS day
        FROM {}
        WHERE symbol = %s
          AND ts::date >= %s
          AND ts::date <= %s
        ORDER BY day
        """
    ).format(relation_identifier(market_table))
    with conn.cursor() as cur:
        cur.execute(query, (calendar_symbol.upper(), start_date, end_date))
        return [row[0] for row in cur.fetchall()]


def relation_exists(conn, relation_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (relation_name,))
        return cur.fetchone()[0] is not None


def fetch_world_regime(conn, world_regime_table: str, day: date, available: bool) -> tuple[str, float | None]:
    if not available:
        return "UNKNOWN", None
    from psycopg2 import sql

    from backtest_core.sql_utils import relation_identifier

    query = sql.SQL(
        """
        SELECT regime_label, composite_score
        FROM {}
        WHERE day <= %s
        ORDER BY day DESC
        LIMIT 1
        """
    ).format(relation_identifier(world_regime_table))
    with conn.cursor() as cur:
        cur.execute(query, (day,))
        row = cur.fetchone()
    if not row:
        return "UNKNOWN", None
    label = str(row[0] or "").strip().upper() or "UNKNOWN"
    score = float(row[1]) if row[1] is not None else None
    return label, score


def fetch_market_regime(
    conn,
    market_regime_table: str,
    benchmark_symbol: str,
    as_of_ts: datetime,
    preferred_lookback_days: int,
    available: bool,
) -> dict[str, object]:
    if not available:
        return {}
    from psycopg2 import sql

    from backtest_core.sql_utils import relation_identifier

    query = sql.SQL(
        """
        SELECT
            market_state,
            lookback_days,
            trend_up_score,
            trend_down_score,
            range_score,
            unclear_score,
            atr_pct,
            close_change_pct
        FROM {}
        WHERE symbol = %s
          AND end_ts <= %s
        ORDER BY
            end_ts DESC,
            CASE WHEN lookback_days = %s THEN 0 ELSE 1 END,
            lookback_days DESC
        LIMIT 1
        """
    ).format(relation_identifier(market_regime_table))
    with conn.cursor() as cur:
        cur.execute(query, (benchmark_symbol.upper(), as_of_ts, preferred_lookback_days))
        row = cur.fetchone()
    if not row:
        return {}
    return {
        "market_regime_label": str(row[0] or "").strip().upper() or "UNKNOWN",
        "market_regime_lookback_days": int(row[1]) if row[1] is not None else None,
        "market_regime_trend_up_score": float(row[2]) if row[2] is not None else None,
        "market_regime_trend_down_score": float(row[3]) if row[3] is not None else None,
        "market_regime_range_score": float(row[4]) if row[4] is not None else None,
        "market_regime_unclear_score": float(row[5]) if row[5] is not None else None,
        "market_regime_atr_pct": float(row[6]) if row[6] is not None else None,
        "market_regime_close_change_pct": float(row[7]) if row[7] is not None else None,
    }


def fetch_regime_context(
    conn,
    args: argparse.Namespace,
    day: date,
    as_of_ts: datetime,
    world_regime_available: bool,
    market_regime_available: bool,
) -> RegimeContext:
    world_label, world_score = fetch_world_regime(conn, args.world_regime_table, day, world_regime_available)
    market = fetch_market_regime(
        conn,
        args.market_regime_table,
        args.benchmark_symbol,
        as_of_ts,
        args.market_regime_lookback_days,
        market_regime_available,
    )
    return RegimeContext(
        world_regime_label=world_label,
        world_regime_score=world_score,
        market_regime_label=str(market.get("market_regime_label") or "UNKNOWN"),
        market_regime_lookback_days=market.get("market_regime_lookback_days"),
        market_regime_trend_up_score=market.get("market_regime_trend_up_score"),
        market_regime_trend_down_score=market.get("market_regime_trend_down_score"),
        market_regime_range_score=market.get("market_regime_range_score"),
        market_regime_unclear_score=market.get("market_regime_unclear_score"),
        market_regime_atr_pct=market.get("market_regime_atr_pct"),
        market_regime_close_change_pct=market.get("market_regime_close_change_pct"),
    )


def apply_frequency(days: list[date], frequency: str) -> list[date]:
    if frequency == "daily":
        return days
    selected: list[date] = []
    seen: set[tuple[int, int]] = set()
    for day in days:
        if frequency == "weekly":
            key = day.isocalendar()[:2]
        elif frequency == "monthly":
            key = (day.year, day.month)
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")
        if key not in seen:
            selected.append(day)
            seen.add(key)
    return selected


def candidate_filter_kwargs(args: argparse.Namespace, direction: str) -> dict[str, object]:
    return {
        "long_min_fundamental": args.long_min_fundamental,
        "short_max_fundamental": args.short_max_fundamental,
        "min_market_cap_m": args.min_market_cap_m,
        "source_table": args.fundamental_table,
        "long_label_blocklist": args.long_label_blocklist,
        "short_label_blocklist": args.short_label_blocklist,
        "pepperstone_table": args.pepperstone_table,
        "required_currency": None if args.any_currency else "USD",
        "filter_high_leverage": args.filter_high_leverage,
        "filter_negative_earnings": args.filter_negative_earnings_short
        if direction == "SHORT"
        else args.filter_negative_earnings_long,
        "filter_scorer_eligibility": args.filter_scorer_eligibility,
        "require_upcoming_earnings_date": args.require_upcoming_earnings_date,
        "fundamental_score_mode": args.fundamental_score_mode,
        "fundamental_peer_weight": args.fundamental_peer_weight,
        "fundamental_abs_weight": args.fundamental_abs_weight,
        "long_min_absolute_score": args.long_min_absolute_score,
        "short_max_absolute_score": args.short_max_absolute_score,
    }


def fetch_bars(
    conn,
    market_table: str,
    symbols: Iterable[str],
    start_ts: datetime,
    end_ts: datetime,
) -> dict[InstrumentKey, list[Bar]]:
    from psycopg2 import sql

    from backtest_core.sql_utils import relation_identifier

    symbol_list = sorted({symbol.strip().upper() for symbol in symbols if symbol.strip()})
    if not symbol_list:
        return {}
    query = sql.SQL(
        """
        SELECT symbol, exchange, cik, ts, open, high, low, close, COALESCE(volume, 0)
        FROM {}
        WHERE symbol = ANY(%s)
          AND ts >= %s
          AND ts <= %s
        ORDER BY symbol, exchange, cik, ts
        """
    ).format(relation_identifier(market_table))
    bars: dict[InstrumentKey, list[Bar]] = defaultdict(list)
    with conn.cursor() as cur:
        cur.execute(query, (symbol_list, start_ts, end_ts))
        for symbol, exchange, cik, ts, open_, high, low, close, volume in cur.fetchall():
            bars[instrument_key(symbol, exchange, int(cik))].append(
                Bar(
                    ts=to_utc(ts),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=int(volume or 0),
                )
            )
    return dict(bars)


def select_benchmark_bars(
    bars_by_identity: dict[InstrumentKey, list[Bar]],
    benchmark_symbol: str,
) -> list[Bar]:
    benchmark_symbol = benchmark_symbol.strip().upper()
    candidates = [
        bars
        for identity, bars in bars_by_identity.items()
        if identity[0] == benchmark_symbol and bars
    ]
    if not candidates:
        return []
    return max(candidates, key=len)


def first_bar_after(bars: list[Bar], ts: datetime) -> Bar | None:
    for bar in bars:
        if bar.ts > ts:
            return bar
    return None


def first_bar_at_or_after(bars: list[Bar], ts: datetime) -> Bar | None:
    for bar in bars:
        if bar.ts >= ts:
            return bar
    return None


def forward_metrics(
    direction: str,
    bars: list[Bar],
    benchmark_bars: list[Bar],
    as_of_ts: datetime,
    horizon_days: int,
) -> ForwardMetrics | None:
    entry = first_bar_after(bars, as_of_ts)
    if entry is None:
        return None
    target_ts = entry.ts + timedelta(days=horizon_days)
    exit_bar = first_bar_at_or_after(bars, target_ts)
    if exit_bar is None:
        return None
    path = [bar for bar in bars if entry.ts <= bar.ts <= exit_bar.ts]
    if not path or entry.close <= 0.0 or exit_bar.close <= 0.0:
        return None

    if direction == "SHORT":
        ret = (entry.close / exit_bar.close - 1.0) * 100.0
        mae = (entry.close / max(bar.high for bar in path) - 1.0) * 100.0
        mfe = (entry.close / min(bar.low for bar in path) - 1.0) * 100.0
    else:
        ret = (exit_bar.close / entry.close - 1.0) * 100.0
        mae = (min(bar.low for bar in path) / entry.close - 1.0) * 100.0
        mfe = (max(bar.high for bar in path) / entry.close - 1.0) * 100.0

    benchmark_return = None
    excess = None
    benchmark_entry = first_bar_after(benchmark_bars, as_of_ts) if benchmark_bars else None
    if benchmark_entry is not None and benchmark_entry.close > 0.0:
        benchmark_exit = first_bar_at_or_after(benchmark_bars, benchmark_entry.ts + timedelta(days=horizon_days))
        if benchmark_exit is not None and benchmark_exit.close > 0.0:
            benchmark_return = (benchmark_exit.close / benchmark_entry.close - 1.0) * 100.0
            excess = ret - benchmark_return

    return ForwardMetrics(
        horizon_days=horizon_days,
        entry_ts=entry.ts,
        entry_price=entry.close,
        exit_ts=exit_bar.ts,
        exit_price=exit_bar.close,
        return_pct=ret,
        mae_pct=mae,
        mfe_pct=mfe,
        benchmark_return_pct=benchmark_return,
        excess_benchmark_pct=excess,
    )


def finite_score(value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def directional_score(direction: str, value: float | None) -> float | None:
    value = finite_score(value)
    if value is None:
        return None
    if direction == "SHORT":
        return 100.0 - value
    return value


def build_candidate_metrics(
    day: date,
    direction: str,
    as_of_ts: datetime,
    fundamental: FundamentalRow,
    bars: list[Bar],
    cfg: IntentConfig,
    lookback_bars: int,
    regime: RegimeContext,
) -> CandidateMetrics | None:
    history = [bar for bar in bars if bar.ts <= as_of_ts]
    if len(history) < lookback_bars:
        return None
    history = history[-lookback_bars:]
    if history[-1].close <= 0.0:
        return None

    if direction == "SHORT":
        scorer_alpha = _short_scorer_alpha(fundamental)
        price_alpha, price_metrics = _short_price_alpha(history, cfg)
    else:
        scorer_alpha = _long_scorer_alpha(fundamental)
        price_alpha, price_metrics = _long_price_alpha(history, cfg)
    swing_alpha = scorer_alpha * 0.45 + price_alpha * 0.55

    score_values = {
        "directional_composite": directional_score(direction, fundamental.composite_score),
        "directional_composite_abs": directional_score(direction, fundamental.composite_score_abs),
        "directional_momentum": directional_score(direction, fundamental.momentum_score),
        "directional_price_momentum": directional_score(direction, fundamental.price_momentum_score),
        "directional_leadership": directional_score(direction, fundamental.leadership_score),
        "directional_quality": directional_score(direction, fundamental.quality_score),
        "directional_valuation": directional_score(direction, fundamental.valuation_score),
        "scorer_alpha": scorer_alpha,
        "price_alpha": price_alpha,
        "swing_alpha": swing_alpha,
    }
    score_values = {key: value for key, value in score_values.items() if value is not None}
    return CandidateMetrics(
        day=day,
        direction=direction,
        as_of_ts=as_of_ts,
        fundamental=fundamental,
        regime=regime,
        history_close=history[-1].close,
        scorer_alpha=scorer_alpha,
        price_alpha=price_alpha,
        swing_alpha=swing_alpha,
        price_metrics=price_metrics,
        score_values=score_values,
        score_deciles={},
    )


def assign_deciles(metrics: list[CandidateMetrics]) -> None:
    score_names = sorted({name for metric in metrics for name in metric.score_values})
    for score_name in score_names:
        valid = [metric for metric in metrics if score_name in metric.score_values]
        valid.sort(key=lambda item: item.score_values[score_name])
        count = len(valid)
        if count == 0:
            continue
        for idx, metric in enumerate(valid, start=1):
            decile = max(1, min(10, math.ceil(idx * 10 / count)))
            metric.score_deciles[score_name] = decile


def entry_pullback_value(metric: CandidateMetrics) -> float | None:
    if metric.direction == "SHORT":
        return finite_score(metric.price_metrics.get("bounce"))
    return finite_score(metric.price_metrics.get("drawdown"))


def entry_pullback_bucket(metric: CandidateMetrics) -> str:
    value = entry_pullback_value(metric)
    if value is None:
        return "unknown"
    if metric.direction == "SHORT":
        prefix = "short_bounce"
    else:
        prefix = "long_drawdown"
    if value <= 2.0:
        return f"{prefix}_00_02"
    if value <= 5.0:
        return f"{prefix}_02_05"
    if value <= 10.0:
        return f"{prefix}_05_10"
    if value <= 15.0:
        return f"{prefix}_10_15"
    return f"{prefix}_15_plus"


def slice_regime_labels(metric: CandidateMetrics) -> list[tuple[str, str]]:
    return [
        ("market", metric.regime.market_regime_label or "UNKNOWN"),
        ("world", metric.regime.world_regime_label or "UNKNOWN"),
    ]


def add_slice_buckets(
    slice_buckets: dict[tuple[str, int, str, str, int, str], SliceBucket],
    metric: CandidateMetrics,
    forward: ForwardMetrics,
) -> None:
    price_momentum_decile = metric.score_deciles.get("directional_price_momentum")
    pullback_value = entry_pullback_value(metric)
    if price_momentum_decile is None or pullback_value is None:
        return
    pullback_bucket = entry_pullback_bucket(metric)
    for regime_source, regime_label in slice_regime_labels(metric):
        key = (
            metric.direction,
            forward.horizon_days,
            regime_source,
            regime_label,
            price_momentum_decile,
            pullback_bucket,
        )
        slice_buckets[key].add(metric, forward, pullback_value)


def write_run_metadata(path: Path, args: argparse.Namespace, sample_days: list[date]) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "start_date": args.start_date.isoformat(),
        "end_date": args.end_date.isoformat(),
        "frequency": args.frequency,
        "directions": args.directions,
        "horizons": args.horizons,
        "market_table": args.market_table,
        "fundamental_table": args.fundamental_table,
        "pepperstone_table": args.pepperstone_table,
        "world_regime_table": args.world_regime_table,
        "market_regime_table": args.market_regime_table,
        "market_regime_lookback_days": args.market_regime_lookback_days,
        "benchmark_symbol": args.benchmark_symbol,
        "calendar_symbol": args.calendar_symbol,
        "sample_days": len(sample_days),
        "as_of_hour_utc": args.as_of_hour_utc,
        "as_of_minute_utc": args.as_of_minute_utc,
        "lookback_days": args.lookback_days,
        "long_min_fundamental": args.long_min_fundamental,
        "short_max_fundamental": args.short_max_fundamental,
        "min_market_cap_m": args.min_market_cap_m,
        "filter_high_leverage": args.filter_high_leverage,
        "filter_negative_earnings_long": args.filter_negative_earnings_long,
        "filter_negative_earnings_short": args.filter_negative_earnings_short,
        "filter_scorer_eligibility": args.filter_scorer_eligibility,
        "require_upcoming_earnings_date": args.require_upcoming_earnings_date,
        "fundamental_score_mode": args.fundamental_score_mode,
        "fundamental_peer_weight": args.fundamental_peer_weight,
        "fundamental_abs_weight": args.fundamental_abs_weight,
        "long_min_absolute_score": args.long_min_absolute_score,
        "short_max_absolute_score": args.short_max_absolute_score,
        "min_slice_count": args.min_slice_count,
        "any_currency": args.any_currency,
    }
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def candidate_fieldnames(horizons: list[int]) -> list[str]:
    return [
        "day",
        "direction",
        "as_of_ts",
        "symbol",
        "exchange",
        "cik",
        "sector",
        "industry",
        "world_regime_label",
        "world_regime_score",
        "market_regime_label",
        "market_regime_lookback_days",
        "market_regime_trend_up_score",
        "market_regime_trend_down_score",
        "market_regime_range_score",
        "market_regime_unclear_score",
        "market_regime_atr_pct",
        "market_regime_close_change_pct",
        "market_cap_m",
        "composite_score",
        "composite_score_abs",
        "momentum_score",
        "price_momentum_score",
        "leadership_score",
        "quality_score",
        "valuation_score",
        "fundamental_momentum_score",
        "mispricing_score",
        "valuation_label",
        "long_eligible",
        "short_eligible",
        "history_close",
        "scorer_alpha",
        "price_alpha",
        "swing_alpha",
        "trend",
        "intermediate",
        "confirmation",
        "drawdown",
        "breakout_gap",
        "bounce",
        "rsi",
        "atr",
        "fast_ma",
        "slow_ma",
        "entry_pullback_bucket",
        "entry_pullback_pct",
        "horizon_days",
        "entry_ts",
        "entry_price",
        "exit_ts",
        "exit_price",
        "return_pct",
        "mae_pct",
        "mfe_pct",
        "benchmark_return_pct",
        "excess_benchmark_pct",
    ] + [f"{score_name}_decile" for score_name in [
        "directional_composite",
        "directional_composite_abs",
        "directional_momentum",
        "directional_price_momentum",
        "directional_leadership",
        "directional_quality",
        "directional_valuation",
        "scorer_alpha",
        "price_alpha",
        "swing_alpha",
    ]]


def candidate_row(metric: CandidateMetrics, forward: ForwardMetrics) -> dict[str, object]:
    f = metric.fundamental
    row = {
        "day": metric.day.isoformat(),
        "direction": metric.direction,
        "as_of_ts": metric.as_of_ts.isoformat(),
        "symbol": f.symbol,
        "exchange": f.exchange,
        "cik": f.cik,
        "sector": f.sector,
        "industry": f.industry,
        "world_regime_label": metric.regime.world_regime_label,
        "world_regime_score": metric.regime.world_regime_score,
        "market_regime_label": metric.regime.market_regime_label,
        "market_regime_lookback_days": metric.regime.market_regime_lookback_days,
        "market_regime_trend_up_score": metric.regime.market_regime_trend_up_score,
        "market_regime_trend_down_score": metric.regime.market_regime_trend_down_score,
        "market_regime_range_score": metric.regime.market_regime_range_score,
        "market_regime_unclear_score": metric.regime.market_regime_unclear_score,
        "market_regime_atr_pct": metric.regime.market_regime_atr_pct,
        "market_regime_close_change_pct": metric.regime.market_regime_close_change_pct,
        "market_cap_m": f.market_cap_m,
        "composite_score": f.composite_score,
        "composite_score_abs": f.composite_score_abs,
        "momentum_score": f.momentum_score,
        "price_momentum_score": f.price_momentum_score,
        "leadership_score": f.leadership_score,
        "quality_score": f.quality_score,
        "valuation_score": f.valuation_score,
        "fundamental_momentum_score": f.fundamental_momentum_score,
        "mispricing_score": f.mispricing_score,
        "valuation_label": f.valuation_label,
        "long_eligible": f.long_eligible,
        "short_eligible": f.short_eligible,
        "history_close": metric.history_close,
        "scorer_alpha": metric.scorer_alpha,
        "price_alpha": metric.price_alpha,
        "swing_alpha": metric.swing_alpha,
        "horizon_days": forward.horizon_days,
        "entry_ts": forward.entry_ts.isoformat(),
        "entry_price": forward.entry_price,
        "exit_ts": forward.exit_ts.isoformat(),
        "exit_price": forward.exit_price,
        "return_pct": forward.return_pct,
        "mae_pct": forward.mae_pct,
        "mfe_pct": forward.mfe_pct,
        "benchmark_return_pct": forward.benchmark_return_pct,
        "excess_benchmark_pct": forward.excess_benchmark_pct,
        "entry_pullback_bucket": entry_pullback_bucket(metric),
        "entry_pullback_pct": entry_pullback_value(metric),
    }
    for key in ("trend", "intermediate", "confirmation", "drawdown", "breakout_gap", "bounce", "rsi", "atr", "fast_ma", "slow_ma"):
        row[key] = metric.price_metrics.get(key)
    for score_name, decile in metric.score_deciles.items():
        row[f"{score_name}_decile"] = decile
    return row


def write_summary(path: Path, buckets: dict[tuple[str, str, int, int], StatBucket]) -> None:
    fieldnames = [
        "direction",
        "score_name",
        "horizon_days",
        "decile",
        "count",
        "avg_score",
        "avg_return_pct",
        "median_return_pct",
        "win_rate_pct",
        "avg_excess_benchmark_pct",
        "median_excess_benchmark_pct",
        "positive_excess_rate_pct",
        "avg_mae_pct",
        "avg_mfe_pct",
        "avg_mfe_to_abs_mae",
        "avg_mfe_minus_abs_mae_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for direction, score_name, horizon_days, decile in sorted(buckets):
            writer.writerow(buckets[(direction, score_name, horizon_days, decile)].as_row(score_name, direction, horizon_days, decile))


def write_spreads(path: Path, buckets: dict[tuple[str, str, int, int], StatBucket]) -> None:
    fieldnames = [
        "direction",
        "score_name",
        "horizon_days",
        "bottom_decile_count",
        "top_decile_count",
        "bottom_avg_return_pct",
        "top_avg_return_pct",
        "top_minus_bottom_return_pct",
        "bottom_avg_excess_benchmark_pct",
        "top_avg_excess_benchmark_pct",
        "top_minus_bottom_excess_pct",
        "bottom_avg_mae_pct",
        "top_avg_mae_pct",
        "bottom_avg_mfe_pct",
        "top_avg_mfe_pct",
    ]
    groups = sorted({key[:3] for key in buckets})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for direction, score_name, horizon_days in groups:
            bottom = buckets.get((direction, score_name, horizon_days, 1))
            top = buckets.get((direction, score_name, horizon_days, 10))
            if not bottom or not top or bottom.count == 0 or top.count == 0:
                continue
            bottom_avg_return = bottom.return_sum / bottom.count
            top_avg_return = top.return_sum / top.count
            bottom_excess_count = len(bottom.excess_returns or [])
            top_excess_count = len(top.excess_returns or [])
            bottom_avg_excess = bottom.excess_sum / bottom_excess_count if bottom_excess_count else None
            top_avg_excess = top.excess_sum / top_excess_count if top_excess_count else None
            writer.writerow({
                "direction": direction,
                "score_name": score_name,
                "horizon_days": horizon_days,
                "bottom_decile_count": bottom.count,
                "top_decile_count": top.count,
                "bottom_avg_return_pct": bottom_avg_return,
                "top_avg_return_pct": top_avg_return,
                "top_minus_bottom_return_pct": top_avg_return - bottom_avg_return,
                "bottom_avg_excess_benchmark_pct": bottom_avg_excess,
                "top_avg_excess_benchmark_pct": top_avg_excess,
                "top_minus_bottom_excess_pct": (
                    top_avg_excess - bottom_avg_excess
                    if top_avg_excess is not None and bottom_avg_excess is not None
                    else None
                ),
                "bottom_avg_mae_pct": bottom.mae_sum / bottom.count,
                "top_avg_mae_pct": top.mae_sum / top.count,
                "bottom_avg_mfe_pct": bottom.mfe_sum / bottom.count,
                "top_avg_mfe_pct": top.mfe_sum / top.count,
            })


SLICE_FIELDNAMES = [
    "direction",
    "horizon_days",
    "regime_source",
    "regime_label",
    "price_momentum_decile",
    "entry_pullback_bucket",
    "count",
    "avg_directional_price_momentum",
    "avg_entry_pullback_pct",
    "avg_price_alpha",
    "avg_scorer_alpha",
    "avg_swing_alpha",
    "avg_return_pct",
    "median_return_pct",
    "win_rate_pct",
    "avg_excess_benchmark_pct",
    "median_excess_benchmark_pct",
    "positive_excess_rate_pct",
    "avg_mae_pct",
    "avg_mfe_pct",
    "avg_mfe_to_abs_mae",
    "avg_mfe_minus_abs_mae_pct",
]


def slice_row(
    key: tuple[str, int, str, str, int, str],
    bucket: SliceBucket,
) -> dict[str, object]:
    direction, horizon_days, regime_source, regime_label, price_momentum_decile, pullback_bucket = key
    return bucket.as_row(
        direction,
        horizon_days,
        regime_source,
        regime_label,
        price_momentum_decile,
        pullback_bucket,
    )


def write_slices(path: Path, slice_buckets: dict[tuple[str, int, str, str, int, str], SliceBucket]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SLICE_FIELDNAMES)
        writer.writeheader()
        for key in sorted(slice_buckets):
            writer.writerow(slice_row(key, slice_buckets[key]))


def _leader_sort_value(row: dict[str, object]) -> tuple[float, float, int]:
    excess = row.get("avg_excess_benchmark_pct")
    avg_return = row.get("avg_return_pct")
    count = row.get("count") or 0
    excess_value = float(excess) if excess is not None else -999.0
    return_value = float(avg_return) if avg_return is not None else -999.0
    return (excess_value, return_value, int(count))


def write_slice_leaders(
    path: Path,
    slice_buckets: dict[tuple[str, int, str, str, int, str], SliceBucket],
    min_slice_count: int,
) -> None:
    rows = [
        slice_row(key, bucket)
        for key, bucket in slice_buckets.items()
        if bucket.count >= min_slice_count
    ]
    rows.sort(key=_leader_sort_value, reverse=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SLICE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=parse_date, default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", type=parse_date, default=DEFAULT_END_DATE)
    parser.add_argument("--frequency", choices=("daily", "weekly", "monthly"), default="daily")
    parser.add_argument("--directions", type=parse_directions, default=parse_directions("LONG"))
    parser.add_argument("--horizons", type=parse_horizons, default=parse_horizons("5,10,20,60"))
    parser.add_argument("--calendar-symbol", default="QQQ")
    parser.add_argument("--benchmark-symbol", default="QQQ")
    parser.add_argument("--as-of-hour-utc", type=int, default=23)
    parser.add_argument("--as-of-minute-utc", type=int, default=59)
    parser.add_argument("--lookback-days", type=int, default=280)
    parser.add_argument("--market-table", default=DEFAULT_MARKET_TABLE)
    parser.add_argument("--fundamental-table", default=DEFAULT_FUNDAMENTAL_TABLE)
    parser.add_argument("--pepperstone-table", default=DEFAULT_PEPPERSTONE_TABLE)
    parser.add_argument("--world-regime-table", default=DEFAULT_WORLD_REGIME_TABLE)
    parser.add_argument("--market-regime-table", default=DEFAULT_MARKET_REGIME_TABLE)
    parser.add_argument("--market-regime-lookback-days", type=int, default=DEFAULT_MARKET_REGIME_LOOKBACK_DAYS)
    parser.add_argument("--long-min-fundamental", type=float, default=DEFAULT_LONG_MIN_FUNDAMENTAL)
    parser.add_argument("--short-max-fundamental", type=float, default=DEFAULT_SHORT_MAX_FUNDAMENTAL)
    parser.add_argument("--min-market-cap-m", type=float, default=DEFAULT_MIN_MARKET_CAP_M)
    parser.add_argument("--max-candidates-per-day", type=int, default=0)
    parser.add_argument("--fundamental-score-mode", choices=("peer", "absolute", "blend"), default=DEFAULT_FUNDAMENTAL_SCORE_MODE)
    parser.add_argument("--fundamental-peer-weight", type=float, default=DEFAULT_FUNDAMENTAL_PEER_WEIGHT)
    parser.add_argument("--fundamental-abs-weight", type=float, default=DEFAULT_FUNDAMENTAL_ABS_WEIGHT)
    parser.add_argument("--long-min-absolute-score", type=float, default=DEFAULT_LONG_MIN_ABSOLUTE_SCORE)
    parser.add_argument("--short-max-absolute-score", type=float, default=DEFAULT_SHORT_MAX_ABSOLUTE_SCORE)
    parser.add_argument("--any-currency", action="store_true", default=not DEFAULT_REQUIRE_USD_FUNDAMENTALS)
    parser.add_argument("--filter-high-leverage", action=argparse.BooleanOptionalAction, default=DEFAULT_FILTER_HIGH_LEVERAGE)
    parser.add_argument("--filter-negative-earnings-long", action=argparse.BooleanOptionalAction, default=DEFAULT_FILTER_NEGATIVE_EARNINGS_LONG)
    parser.add_argument("--filter-negative-earnings-short", action=argparse.BooleanOptionalAction, default=DEFAULT_FILTER_NEGATIVE_EARNINGS_SHORT)
    parser.add_argument("--filter-scorer-eligibility", action=argparse.BooleanOptionalAction, default=DEFAULT_FILTER_SCORER_ELIGIBILITY)
    parser.add_argument("--require-upcoming-earnings-date", action=argparse.BooleanOptionalAction, default=DEFAULT_REQUIRE_UPCOMING_EARNINGS_DATE)
    parser.add_argument("--long-label-blocklist", nargs="*", default=list(DEFAULT_LONG_LABEL_BLOCKLIST))
    parser.add_argument("--short-label-blocklist", nargs="*", default=list(DEFAULT_SHORT_LABEL_BLOCKLIST))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "analysis" / "output" / "swing_alpha_forward_research")
    parser.add_argument("--write-candidates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-slice-count", type=int, default=30)
    parser.add_argument("--progress-every-days", type=int, default=10)
    parser.add_argument("--log-level", default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.start_date > args.end_date:
        raise ValueError("--start-date must be <= --end-date")
    if not (0 <= args.as_of_hour_utc <= 23):
        raise ValueError("--as-of-hour-utc must be between 0 and 23")
    if not (0 <= args.as_of_minute_utc <= 59):
        raise ValueError("--as-of-minute-utc must be between 0 and 59")
    if args.lookback_days <= 0:
        raise ValueError("--lookback-days must be positive")
    if args.market_regime_lookback_days <= 0:
        raise ValueError("--market-regime-lookback-days must be positive")
    if args.max_candidates_per_day < 0:
        raise ValueError("--max-candidates-per-day must be >= 0")
    if args.min_slice_count < 1:
        raise ValueError("--min-slice-count must be >= 1")
    if args.fundamental_peer_weight < 0.0 or args.fundamental_abs_weight < 0.0:
        raise ValueError("--fundamental-peer-weight and --fundamental-abs-weight must be >= 0")
    if args.fundamental_score_mode == "blend" and args.fundamental_peer_weight + args.fundamental_abs_weight <= 0.0:
        raise ValueError("blend mode requires positive --fundamental-peer-weight or --fundamental-abs-weight")


def run() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    configure_logging(args.log_level)

    os.environ["START_DATE"] = args.start_date.isoformat()
    os.environ["END_DATE"] = args.end_date.isoformat()

    from backtest_core.db import connect_with_retry
    from backtest_core.market_data import get_candidates

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "swing_alpha_forward_summary.csv"
    spreads_path = args.output_dir / "swing_alpha_forward_spreads.csv"
    slices_path = args.output_dir / "swing_alpha_forward_slices.csv"
    slice_leaders_path = args.output_dir / "swing_alpha_forward_slice_leaders.csv"
    metadata_path = args.output_dir / "swing_alpha_forward_run.json"
    candidates_path = args.output_dir / "swing_alpha_forward_candidates.csv"

    cfg = IntentConfig()
    lookback_bars = required_bar_lookback(cfg)
    max_horizon = max(args.horizons)
    buckets: dict[tuple[str, str, int, int], StatBucket] = defaultdict(StatBucket)
    slice_buckets: dict[tuple[str, int, str, str, int, str], SliceBucket] = defaultdict(SliceBucket)
    total_candidate_rows = 0
    total_forward_rows = 0
    skipped_no_history = 0
    skipped_no_forward = 0

    with connect_with_retry() as conn:
        trading_days = fetch_trading_days(
            conn,
            args.market_table,
            args.calendar_symbol,
            args.start_date,
            args.end_date,
        )
        sample_days = apply_frequency(trading_days, args.frequency)
        world_regime_available = relation_exists(conn, args.world_regime_table)
        market_regime_available = relation_exists(conn, args.market_regime_table)
        write_run_metadata(metadata_path, args, sample_days)
        LOG.info(
            "Research starting days %d frequency %s directions %s horizons %s output %s world regime available %s market regime available %s",
            len(sample_days),
            args.frequency,
            ",".join(args.directions),
            ",".join(str(item) for item in args.horizons),
            args.output_dir,
            world_regime_available,
            market_regime_available,
        )

        candidate_handle = None
        candidate_writer = None
        if args.write_candidates:
            candidate_handle = candidates_path.open("w", newline="", encoding="utf-8")
            candidate_writer = csv.DictWriter(candidate_handle, fieldnames=candidate_fieldnames(args.horizons), extrasaction="ignore")
            candidate_writer.writeheader()

        try:
            for day_index, day in enumerate(sample_days, start=1):
                as_of_ts = utc_as_of(day, args.as_of_hour_utc, args.as_of_minute_utc)
                load_start = as_of_ts - timedelta(days=args.lookback_days)
                load_end = as_of_ts + timedelta(days=max_horizon + 14)
                regime_context = fetch_regime_context(
                    conn,
                    args,
                    day,
                    as_of_ts,
                    world_regime_available,
                    market_regime_available,
                )
                day_metrics: list[CandidateMetrics] = []
                day_bars_by_identity: dict[InstrumentKey, list[Bar]] = {}
                raw_candidate_count = 0

                for direction in args.directions:
                    candidates = get_candidates(
                        conn,
                        direction,
                        as_of_ts=as_of_ts,
                        **candidate_filter_kwargs(args, direction),
                    )
                    raw_candidate_count += len(candidates)
                    if args.max_candidates_per_day > 0:
                        reverse = direction == "LONG"
                        candidates = sorted(
                            candidates,
                            key=lambda item: item.composite_score if item.composite_score is not None else -999.0,
                            reverse=reverse,
                        )[:args.max_candidates_per_day]

                    symbols = [candidate.symbol for candidate in candidates]
                    symbols.append(args.benchmark_symbol)
                    bars_by_identity = fetch_bars(conn, args.market_table, symbols, load_start, load_end)
                    day_bars_by_identity.update(bars_by_identity)

                    for fundamental in candidates:
                        bars = bars_by_identity.get(fundamental.identity_key, [])
                        metric = build_candidate_metrics(
                            day,
                            direction,
                            as_of_ts,
                            fundamental,
                            bars,
                            cfg,
                            lookback_bars,
                            regime_context,
                        )
                        if metric is None:
                            skipped_no_history += 1
                            continue
                        day_metrics.append(metric)

                assign_deciles(day_metrics)
                total_candidate_rows += len(day_metrics)
                benchmark_bars = select_benchmark_bars(day_bars_by_identity, args.benchmark_symbol)

                for metric in day_metrics:
                    instrument_bars = day_bars_by_identity.get(metric.fundamental.identity_key, [])
                    for horizon in args.horizons:
                        forward = forward_metrics(metric.direction, instrument_bars, benchmark_bars, metric.as_of_ts, horizon)
                        if forward is None:
                            skipped_no_forward += 1
                            continue
                        total_forward_rows += 1
                        for score_name, decile in metric.score_deciles.items():
                            score_value = metric.score_values.get(score_name)
                            if score_value is None:
                                continue
                            buckets[(metric.direction, score_name, horizon, decile)].add(score_value, forward)
                        add_slice_buckets(slice_buckets, metric, forward)
                        if candidate_writer is not None:
                            candidate_writer.writerow(candidate_row(metric, forward))

                if args.progress_every_days > 0 and (
                    day_index == 1 or day_index % args.progress_every_days == 0 or day_index == len(sample_days)
                ):
                    LOG.info(
                        "Progress %d/%d day %s raw candidates %d usable candidates %d forward rows %d skipped history %d skipped forward %d",
                        day_index,
                        len(sample_days),
                        day.isoformat(),
                        raw_candidate_count,
                        len(day_metrics),
                        total_forward_rows,
                        skipped_no_history,
                        skipped_no_forward,
                    )
        finally:
            if candidate_handle is not None:
                candidate_handle.close()

    write_summary(summary_path, buckets)
    write_spreads(spreads_path, buckets)
    write_slices(slices_path, slice_buckets)
    write_slice_leaders(slice_leaders_path, slice_buckets, args.min_slice_count)
    LOG.info(
        "Research complete candidates %d forward rows %d skipped history %d skipped forward %d summary %s spreads %s slices %s slice leaders %s",
        total_candidate_rows,
        total_forward_rows,
        skipped_no_history,
        skipped_no_forward,
        summary_path,
        spreads_path,
        slices_path,
        slice_leaders_path,
    )


if __name__ == "__main__":
    run()
