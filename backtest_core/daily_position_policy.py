"""Daily portfolio policy for entries, stress phases, and intraday halts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

import psycopg2

from backtest_shared import WorldRegime

from .config import (
    DAILY_POLICY_BLOCK_LONG_SECTOR,
    DAILY_POLICY_ENERGY_SHOCK_LOOKBACK_DAYS,
    DAILY_POLICY_ENERGY_SHOCK_THRESHOLD,
    DAILY_POLICY_HIGH_STRESS_MAX_LONG_POSITIONS,
    DAILY_POLICY_HIGH_STRESS_MAX_SHORT_POSITIONS,
    DAILY_POLICY_INITIAL_PREVIOUS_MAX_LONG_POSITIONS,
    DAILY_POLICY_LOOKBACK_CALENDAR_DAYS,
    DAILY_POLICY_LOW_STRESS_MAX_LONG_POSITIONS,
    DAILY_POLICY_LOW_STRESS_MAX_SHORT_POSITIONS,
    DAILY_POLICY_MARKET_DROP_OPEN_BAR_MODE,
    DAILY_POLICY_MARKET_DROP_PRICE_FIELD,
    DAILY_POLICY_MARKET_DROP_SESSION_END,
    DAILY_POLICY_MARKET_DROP_SESSION_START,
    DAILY_POLICY_MARKET_DROP_SYMBOL,
    DAILY_POLICY_MARKET_DROP_THRESHOLD_PCT,
    DAILY_POLICY_MARKET_DROP_TZ,
    DAILY_POLICY_MIN_HOURS_BETWEEN_OPENS,
    DAILY_POLICY_PREFERRED_INDUSTRY,
    DAILY_POLICY_PORTFOLIO_DROP_LOOKBACK_DAYS,
    DAILY_POLICY_PORTFOLIO_DROP_THRESHOLD_PCT,
    DAILY_POLICY_PREVIOUS_MARKET_DROP_PRICE_FIELD,
    DAILY_POLICY_PREVIOUS_MARKET_DROP_SYMBOL,
    DAILY_POLICY_PREVIOUS_MARKET_DROP_THRESHOLD_PCT,
    DAILY_POLICY_PRUNE_TIME,
    DAILY_POLICY_PRUNE_TIME_TZ,
    DAILY_POLICY_SL_HALT_COUNT,
    DAILY_POLICY_SL_HALT_STATUSES,
    DAILY_POLICY_SL_HALT_WINDOW_HOURS,
    DAILY_POLICY_STRESS_BUILD_MAX_LONG_DELTA,
    DAILY_POLICY_STRESS_RECOVERY_MAX_LONG_DELTA,
    DAILY_POLICY_STRESS_RISK_REDUCTION_PCT,
    DAILY_POLICY_STRESS_SIDEWAYS_MAX_LONG_DELTA,
    DAILY_POLICY_TECH_STRESS_LOOKBACK_DAYS,
    DAILY_POLICY_TECH_STRESS_THRESHOLD,
    DAILY_POLICY_TZ,
    DAILY_POLICY_WORLD_REGIME_HIGH_STRESS_THRESHOLD,
    DAILY_POLICY_WORLD_REGIME_MA_DAYS,
    DAILY_POLICY_WORLD_REGIME_STRESS_BUILDING_TREND_DAYS,
    DAILY_POLICY_WORLD_REGIME_STRESS_RECEDING_TREND_DAYS,
    DAILY_POLICY_WORLD_REGIME_STRESS_THRESHOLD,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_OPENS_PER_DAY,
    RISK_PER_TRADE_PCT,
    SOURCE_MARKET_DATA_1H_TABLE,
    SOURCE_WORLD_REGIME_TABLE,
)
from .market_data import _ensure_utc_ts, get_trading_days, get_world_regime
from .sql_utils import relation_identifier


@dataclass(frozen=True)
class DailyPositionPolicyContext:
    day: date
    regime: WorldRegime
    regime_label: str
    phase: str
    world_regime_ma_score: float
    world_regime_ma_days: tuple[date, ...]
    world_regime_stress_building_trend_ma_scores: tuple[float, ...]
    world_regime_stress_receding_trend_ma_scores: tuple[float, ...]
    exposure: dict
    calculated_max_long_positions: int
    blocked_long_sectors: tuple[str, ...]
    preferred_industry: str
    tech_stress_active: bool
    energy_preferred_active: bool
    previous_trading_days: tuple[date, ...]


@dataclass(frozen=True)
class DailyPolicyEntryCheck:
    accepted: bool
    reason_code: str = ""
    reason_text: str = ""


@dataclass(frozen=True)
class PortfolioDailyReturn:
    day: date
    start_equity: float
    end_equity: float
    return_pct: float


@dataclass(frozen=True)
class MarketDropSnapshot:
    available: bool
    open_ts: Optional[datetime] = None
    open_price: Optional[float] = None
    min_price: Optional[float] = None
    drop_pct: float = 0.0


@dataclass(frozen=True)
class PreviousMarketDropSnapshot:
    available: bool
    day: Optional[date] = None
    open_ts: Optional[datetime] = None
    open_price: Optional[float] = None
    comparison_price: Optional[float] = None
    drop_pct: float = 0.0


@dataclass
class DailyPolicyRuntimeState:
    context: DailyPositionPolicyContext
    opened_timestamps: list[datetime]
    sl_close_timestamps: list[datetime]
    halted: bool = False
    halt_reason_code: str = ""
    halt_reason_text: str = ""
    market_drop_snapshot: Optional[MarketDropSnapshot] = None
    previous_market_drop_snapshot: Optional[PreviousMarketDropSnapshot] = None
    prune_done: bool = False

    def record_open(self, ts: datetime) -> None:
        self.opened_timestamps.append(_ensure_utc_ts(ts))

    def record_closed_trades(self, closed_trades: list) -> None:
        if self.halted or not closed_trades:
            return
        local_zone = ZoneInfo(DAILY_POLICY_TZ)
        halt_statuses = {status.strip().upper() for status in DAILY_POLICY_SL_HALT_STATUSES}
        for trade in closed_trades:
            status = str(getattr(trade, "outcome_status", "") or "").strip().upper()
            if status not in halt_statuses:
                continue
            exit_ts = getattr(trade, "exit_ts", None)
            if exit_ts is None:
                continue
            exit_ts = _ensure_utc_ts(exit_ts)
            if exit_ts.astimezone(local_zone).date() != self.context.day:
                continue
            self.sl_close_timestamps.append(exit_ts)
        self._refresh_sl_halt()

    def refresh_market_drop_halt(self, conn: psycopg2.extensions.connection, as_of_ts: datetime) -> None:
        if self.halted or DAILY_POLICY_MARKET_DROP_THRESHOLD_PCT <= 0.0:
            return
        snapshot = market_drop_snapshot(conn, self.context.day, as_of_ts)
        self.market_drop_snapshot = snapshot
        if not snapshot.available or snapshot.open_price is None:
            return
        if snapshot.drop_pct > DAILY_POLICY_MARKET_DROP_THRESHOLD_PCT:
            self.halted = True
            self.halt_reason_code = "daily_policy_market_drop_halt"
            self.halt_reason_text = (
                f"{DAILY_POLICY_MARKET_DROP_SYMBOL} fell {snapshot.drop_pct:.2f}% from the daily "
                f"open {snapshot.open_price:.4f}; threshold is > {DAILY_POLICY_MARKET_DROP_THRESHOLD_PCT:.2f}%."
            )

    def refresh_portfolio_drop_halt(self, previous_daily_returns: Sequence[PortfolioDailyReturn]) -> None:
        if (
            self.halted
            or DAILY_POLICY_PORTFOLIO_DROP_LOOKBACK_DAYS <= 0
            or DAILY_POLICY_PORTFOLIO_DROP_THRESHOLD_PCT <= 0.0
        ):
            return
        recent_returns = list(previous_daily_returns)[-DAILY_POLICY_PORTFOLIO_DROP_LOOKBACK_DAYS:]
        breaches = [
            daily_return
            for daily_return in recent_returns
            if daily_return.return_pct <= -DAILY_POLICY_PORTFOLIO_DROP_THRESHOLD_PCT
        ]
        if not breaches:
            return
        worst = min(breaches, key=lambda daily_return: daily_return.return_pct)
        self.halted = True
        self.halt_reason_code = "daily_policy_portfolio_drop_halt"
        self.halt_reason_text = (
            f"Portfolio fell {abs(worst.return_pct):.2f}% on {worst.day}; threshold is >= "
            f"{DAILY_POLICY_PORTFOLIO_DROP_THRESHOLD_PCT:.2f}% within the last "
            f"{DAILY_POLICY_PORTFOLIO_DROP_LOOKBACK_DAYS} trading days."
        )

    def refresh_previous_market_drop_halt(self, conn: psycopg2.extensions.connection) -> None:
        if self.halted or DAILY_POLICY_PREVIOUS_MARKET_DROP_THRESHOLD_PCT <= 0.0:
            return
        if not self.context.previous_trading_days:
            return
        previous_day = self.context.previous_trading_days[-1]
        snapshot = previous_market_drop_snapshot(conn, previous_day)
        self.previous_market_drop_snapshot = snapshot
        if not snapshot.available or snapshot.open_price is None:
            return
        if snapshot.drop_pct > DAILY_POLICY_PREVIOUS_MARKET_DROP_THRESHOLD_PCT:
            self.halted = True
            self.halt_reason_code = "daily_policy_previous_market_drop_halt"
            self.halt_reason_text = (
                f"{DAILY_POLICY_PREVIOUS_MARKET_DROP_SYMBOL} fell {snapshot.drop_pct:.2f}% on "
                f"previous trading day {previous_day} from the regular-session open "
                f"{snapshot.open_price:.4f}; threshold is > "
                f"{DAILY_POLICY_PREVIOUS_MARKET_DROP_THRESHOLD_PCT:.2f}%."
            )

    def entry_check(self, entry_ts: datetime) -> DailyPolicyEntryCheck:
        entry_ts = _ensure_utc_ts(entry_ts)
        if self.halted:
            return DailyPolicyEntryCheck(False, self.halt_reason_code, self.halt_reason_text)
        if len(self.opened_timestamps) >= MAX_POSITION_OPENS_PER_DAY:
            return DailyPolicyEntryCheck(
                False,
                "daily_position_open_limit_reached",
                (
                    f"Daily position open limit {MAX_POSITION_OPENS_PER_DAY} was already reached; "
                    "this limit includes initial opens and refill opens."
                ),
            )
        if self.opened_timestamps and DAILY_POLICY_MIN_HOURS_BETWEEN_OPENS > 0.0:
            last_open_ts = max(self.opened_timestamps)
            elapsed_hours = (entry_ts - last_open_ts).total_seconds() / 3600.0
            if elapsed_hours < DAILY_POLICY_MIN_HOURS_BETWEEN_OPENS:
                return DailyPolicyEntryCheck(
                    False,
                    "daily_policy_min_hours_between_opens",
                    (
                        f"Only {elapsed_hours:.2f} hours elapsed since the previous open at {last_open_ts}; "
                        f"minimum is {DAILY_POLICY_MIN_HOURS_BETWEEN_OPENS:.2f} hours."
                    ),
                )
        return DailyPolicyEntryCheck(True)

    def _refresh_sl_halt(self) -> None:
        if len(self.sl_close_timestamps) < DAILY_POLICY_SL_HALT_COUNT:
            return
        ordered = sorted(self.sl_close_timestamps)
        window = timedelta(hours=DAILY_POLICY_SL_HALT_WINDOW_HOURS)
        for idx, close_ts in enumerate(ordered):
            window_start = close_ts - window
            count = sum(1 for ts in ordered[:idx + 1] if window_start <= ts <= close_ts)
            if count >= DAILY_POLICY_SL_HALT_COUNT:
                self.halted = True
                self.halt_reason_code = "daily_policy_sl_cluster_halt"
                self.halt_reason_text = (
                    f"{count} stop-loss closes occurred within {DAILY_POLICY_SL_HALT_WINDOW_HOURS:.2f} "
                    "hours; no more entries are allowed today."
                )
                return


def initial_previous_max_long_positions() -> int:
    return _clamp_position_count(DAILY_POLICY_INITIAL_PREVIOUS_MAX_LONG_POSITIONS)


def build_daily_position_policy_context(
    conn: psycopg2.extensions.connection,
    day: date,
    previous_max_long_positions: int,
) -> Optional[DailyPositionPolicyContext]:
    previous_days = _previous_trading_days(conn, day)
    if not previous_days:
        return None

    regimes_by_day = {
        previous_day: get_world_regime(conn, source_table=SOURCE_WORLD_REGIME_TABLE, as_of_date=previous_day)
        for previous_day in previous_days
    }
    latest_previous_day = previous_days[-1]
    latest_regime = regimes_by_day.get(latest_previous_day)
    if latest_regime is None:
        return None

    latest_idx = len(previous_days) - 1
    ma_score = _moving_average_score(previous_days, regimes_by_day, latest_idx)
    if ma_score is None:
        return None

    building_trend_scores = _world_regime_trend_scores(
        previous_days,
        regimes_by_day,
        latest_idx,
        DAILY_POLICY_WORLD_REGIME_STRESS_BUILDING_TREND_DAYS,
    )
    receding_trend_scores = _world_regime_trend_scores(
        previous_days,
        regimes_by_day,
        latest_idx,
        DAILY_POLICY_WORLD_REGIME_STRESS_RECEDING_TREND_DAYS,
    )
    rising = (
        len(building_trend_scores) == DAILY_POLICY_WORLD_REGIME_STRESS_BUILDING_TREND_DAYS + 1
        and all(
            building_trend_scores[idx] > building_trend_scores[idx - 1]
            for idx in range(1, len(building_trend_scores))
        )
    )
    falling = (
        len(receding_trend_scores) == DAILY_POLICY_WORLD_REGIME_STRESS_RECEDING_TREND_DAYS + 1
        and all(
            receding_trend_scores[idx] < receding_trend_scores[idx - 1]
            for idx in range(1, len(receding_trend_scores))
        )
    )

    phase, max_long_positions, max_short_positions, max_total_positions, risk_multiplier = _daily_exposure_numbers(
        ma_score,
        previous_max_long_positions,
        rising=rising,
        falling=falling,
    )
    exposure = {
        "long_risk_multiplier": risk_multiplier if max_long_positions > 0 else 0.0,
        "short_risk_multiplier": risk_multiplier if max_short_positions > 0 else 0.0,
        "max_long_positions": max_long_positions,
        "max_short_positions": max_short_positions,
        "max_total_positions": max_total_positions,
    }

    tech_stress_active = _tech_stress_active(previous_days, regimes_by_day)
    energy_preferred_active = _energy_preferred_active(previous_days, regimes_by_day)
    blocked_long_sectors = (DAILY_POLICY_BLOCK_LONG_SECTOR,) if tech_stress_active and DAILY_POLICY_BLOCK_LONG_SECTOR else ()
    preferred_industry = DAILY_POLICY_PREFERRED_INDUSTRY if energy_preferred_active else ""
    ma_days = tuple(previous_days[max(0, latest_idx - DAILY_POLICY_WORLD_REGIME_MA_DAYS + 1):latest_idx + 1])

    return DailyPositionPolicyContext(
        day=day,
        regime=replace(latest_regime, score=ma_score),
        regime_label=str(latest_regime.label or "").strip().upper(),
        phase=phase,
        world_regime_ma_score=ma_score,
        world_regime_ma_days=ma_days,
        world_regime_stress_building_trend_ma_scores=tuple(building_trend_scores),
        world_regime_stress_receding_trend_ma_scores=tuple(receding_trend_scores),
        exposure=exposure,
        calculated_max_long_positions=max_long_positions,
        blocked_long_sectors=blocked_long_sectors,
        preferred_industry=preferred_industry,
        tech_stress_active=tech_stress_active,
        energy_preferred_active=energy_preferred_active,
        previous_trading_days=tuple(previous_days),
    )


def apply_daily_policy_context_to_config(cfg: object, context: DailyPositionPolicyContext) -> None:
    setattr(cfg, "daily_policy_phase", context.phase)
    setattr(cfg, "daily_policy_world_regime_ma_score", context.world_regime_ma_score)
    setattr(cfg, "daily_policy_blocked_long_sectors", context.blocked_long_sectors)
    setattr(cfg, "daily_policy_preferred_industry", context.preferred_industry)


def long_sector_blocked(context: DailyPositionPolicyContext, sector: str) -> bool:
    blocked = {value.strip().lower() for value in context.blocked_long_sectors if value.strip()}
    return bool(str(sector or "").strip().lower() in blocked)


def preferred_industry_tier(context: DailyPositionPolicyContext, industry: str) -> int:
    preferred = context.preferred_industry.strip().lower()
    if not preferred:
        return 0
    return 0 if str(industry or "").strip().lower() == preferred else 1


def daily_prune_ts(day: date) -> datetime:
    hour, minute = _parse_hhmm(DAILY_POLICY_PRUNE_TIME)
    zone = ZoneInfo(DAILY_POLICY_PRUNE_TIME_TZ)
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone).astimezone(timezone.utc)


def market_drop_snapshot(
    conn: psycopg2.extensions.connection,
    day: date,
    as_of_ts: datetime,
) -> MarketDropSnapshot:
    symbol = DAILY_POLICY_MARKET_DROP_SYMBOL.strip().upper()
    if not symbol:
        return MarketDropSnapshot(False)
    as_of_ts = _ensure_utc_ts(as_of_ts)
    zone = ZoneInfo(DAILY_POLICY_MARKET_DROP_TZ)
    local_start = datetime(day.year, day.month, day.day, tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            (
                "SELECT ts, open, low, close "
                "FROM {} "
                "WHERE symbol = %s AND ts >= %s AND ts < %s "
                "ORDER BY ts"
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE).as_string(conn)),
            (symbol, local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc)),
        )
        rows = cur.fetchall()
    if not rows:
        return MarketDropSnapshot(False)

    session_start = _parse_time(DAILY_POLICY_MARKET_DROP_SESSION_START)
    session_end = _parse_time(DAILY_POLICY_MARKET_DROP_SESSION_END)
    bars = [
        (_ensure_utc_ts(ts), float(open_), float(low), float(close))
        for ts, open_, low, close in rows
        if _ensure_utc_ts(ts) <= as_of_ts
    ]
    open_bar = _first_session_open_bar(bars, zone, day, session_start)
    if open_bar is None:
        return MarketDropSnapshot(False)

    open_ts, open_price, _open_low, _open_close = open_bar
    if open_price <= 0.0:
        return MarketDropSnapshot(False)
    completed_bars = [
        bar
        for bar in bars
        if bar[0] >= open_ts and bar[0] < as_of_ts and _bar_local_time_in_session(bar[0], zone, session_start, session_end)
    ]
    if not completed_bars:
        min_price = open_price
    elif DAILY_POLICY_MARKET_DROP_PRICE_FIELD == "close":
        min_price = min(bar[3] for bar in completed_bars)
    else:
        min_price = min(bar[2] for bar in completed_bars)
    drop_pct = max(0.0, (open_price - min_price) / open_price * 100.0)
    return MarketDropSnapshot(True, open_ts=open_ts, open_price=open_price, min_price=min_price, drop_pct=drop_pct)


def previous_market_drop_snapshot(
    conn: psycopg2.extensions.connection,
    day: date,
) -> PreviousMarketDropSnapshot:
    symbol = DAILY_POLICY_PREVIOUS_MARKET_DROP_SYMBOL.strip().upper()
    if not symbol:
        return PreviousMarketDropSnapshot(False)
    zone = ZoneInfo(DAILY_POLICY_MARKET_DROP_TZ)
    local_start = datetime(day.year, day.month, day.day, tzinfo=zone)
    local_end = local_start + timedelta(days=1)
    with conn.cursor() as cur:
        cur.execute(
            (
                "SELECT ts, open, low, close "
                "FROM {} "
                "WHERE symbol = %s AND ts >= %s AND ts < %s "
                "ORDER BY ts"
            ).format(relation_identifier(SOURCE_MARKET_DATA_1H_TABLE).as_string(conn)),
            (symbol, local_start.astimezone(timezone.utc), local_end.astimezone(timezone.utc)),
        )
        rows = cur.fetchall()
    if not rows:
        return PreviousMarketDropSnapshot(False, day=day)

    session_start = _parse_time(DAILY_POLICY_MARKET_DROP_SESSION_START)
    session_end = _parse_time(DAILY_POLICY_MARKET_DROP_SESSION_END)
    bars = [(_ensure_utc_ts(ts), float(open_), float(low), float(close)) for ts, open_, low, close in rows]
    open_bar = _first_session_open_bar(bars, zone, day, session_start)
    if open_bar is None:
        return PreviousMarketDropSnapshot(False, day=day)

    open_ts, open_price, _open_low, _open_close = open_bar
    if open_price <= 0.0:
        return PreviousMarketDropSnapshot(False, day=day)
    session_bars = [
        bar
        for bar in bars
        if bar[0] >= open_ts and _bar_local_time_in_session(bar[0], zone, session_start, session_end)
    ]
    if not session_bars:
        return PreviousMarketDropSnapshot(False, day=day)
    if DAILY_POLICY_PREVIOUS_MARKET_DROP_PRICE_FIELD == "low":
        comparison_price = min(bar[2] for bar in session_bars)
    else:
        comparison_price = session_bars[-1][3]
    drop_pct = max(0.0, (open_price - comparison_price) / open_price * 100.0)
    return PreviousMarketDropSnapshot(
        True,
        day=day,
        open_ts=open_ts,
        open_price=open_price,
        comparison_price=comparison_price,
        drop_pct=drop_pct,
    )


def _daily_exposure_numbers(
    ma_score: float,
    previous_max_long_positions: int,
    *,
    rising: bool,
    falling: bool,
) -> tuple[str, int, int, int, float]:
    if ma_score <= DAILY_POLICY_WORLD_REGIME_STRESS_THRESHOLD:
        phase = "LOW_STRESS"
        max_long = DAILY_POLICY_LOW_STRESS_MAX_LONG_POSITIONS
        max_short = DAILY_POLICY_LOW_STRESS_MAX_SHORT_POSITIONS
        risk_multiplier = 1.0
    else:
        risk_multiplier = _stress_risk_multiplier()
        if ma_score > DAILY_POLICY_WORLD_REGIME_HIGH_STRESS_THRESHOLD:
            phase = "STRESS_HIGH"
            max_long = DAILY_POLICY_HIGH_STRESS_MAX_LONG_POSITIONS
            max_short = DAILY_POLICY_HIGH_STRESS_MAX_SHORT_POSITIONS
        elif rising:
            phase = "STRESS_BUILDING"
            max_long = previous_max_long_positions + DAILY_POLICY_STRESS_BUILD_MAX_LONG_DELTA
            max_short = 0
        elif falling:
            phase = "STRESS_RECEDING"
            max_long = previous_max_long_positions + DAILY_POLICY_STRESS_RECOVERY_MAX_LONG_DELTA
            max_short = 0
        else:
            phase = "STRESS_SIDEWAYS"
            max_long = previous_max_long_positions + DAILY_POLICY_STRESS_SIDEWAYS_MAX_LONG_DELTA
            max_short = 0

    max_long = _clamp_position_count(max_long)
    max_short = _clamp_position_count(max_short)
    max_total = max_long + max_short
    return phase, max_long, max_short, max_total, risk_multiplier


def _stress_risk_multiplier() -> float:
    if RISK_PER_TRADE_PCT <= 0.0:
        return 0.0
    adjusted_risk = max(0.0, RISK_PER_TRADE_PCT - DAILY_POLICY_STRESS_RISK_REDUCTION_PCT)
    return adjusted_risk / RISK_PER_TRADE_PCT


def _previous_trading_days(conn: psycopg2.extensions.connection, day: date) -> list[date]:
    lookback_start = day - timedelta(days=DAILY_POLICY_LOOKBACK_CALENDAR_DAYS)
    days = get_trading_days(conn, lookback_start, day - timedelta(days=1))
    return [trading_day for trading_day in days if trading_day < day]


def _moving_average_score(
    previous_days: list[date],
    regimes_by_day: dict[date, Optional[WorldRegime]],
    idx: int,
) -> Optional[float]:
    if idx < 0:
        return None
    start_idx = idx - DAILY_POLICY_WORLD_REGIME_MA_DAYS + 1
    if start_idx < 0:
        return None
    scores: list[float] = []
    for previous_day in previous_days[start_idx:idx + 1]:
        regime = regimes_by_day.get(previous_day)
        if regime is None:
            return None
        scores.append(float(regime.score))
    if len(scores) != DAILY_POLICY_WORLD_REGIME_MA_DAYS:
        return None
    return sum(scores) / len(scores)


def _world_regime_trend_scores(
    previous_days: list[date],
    regimes_by_day: dict[date, Optional[WorldRegime]],
    latest_idx: int,
    trend_days: int,
) -> list[float]:
    scores = []
    for idx in range(latest_idx - trend_days, latest_idx + 1):
        score = _moving_average_score(previous_days, regimes_by_day, idx)
        if score is None:
            return []
        scores.append(score)
    return scores


def _tech_stress_active(
    previous_days: list[date],
    regimes_by_day: dict[date, Optional[WorldRegime]],
) -> bool:
    days = previous_days[-DAILY_POLICY_TECH_STRESS_LOOKBACK_DAYS:]
    if len(days) < DAILY_POLICY_TECH_STRESS_LOOKBACK_DAYS:
        return False
    for previous_day in days:
        regime = regimes_by_day.get(previous_day)
        score = getattr(regime, "tech_stress_shock_score", None) if regime is not None else None
        if score is not None and float(score) >= DAILY_POLICY_TECH_STRESS_THRESHOLD:
            return True
    return False


def _energy_preferred_active(
    previous_days: list[date],
    regimes_by_day: dict[date, Optional[WorldRegime]],
) -> bool:
    days = previous_days[-DAILY_POLICY_ENERGY_SHOCK_LOOKBACK_DAYS:]
    if len(days) < DAILY_POLICY_ENERGY_SHOCK_LOOKBACK_DAYS:
        return False
    for previous_day in days:
        regime = regimes_by_day.get(previous_day)
        score = getattr(regime, "energy_commodity_shock_score", None) if regime is not None else None
        if score is None or float(score) < DAILY_POLICY_ENERGY_SHOCK_THRESHOLD:
            return False
    return True


def _clamp_position_count(value: int) -> int:
    return max(0, min(MAX_OPEN_POSITIONS, int(value)))


def _parse_hhmm(value: str) -> tuple[int, int]:
    hour, minute = str(value).strip().split(":", 1)
    return int(hour), int(minute)


def _parse_time(value: str) -> time:
    hour, minute = _parse_hhmm(value)
    return time(hour, minute)


def _first_session_open_bar(
    bars: list[tuple[datetime, float, float, float]],
    zone: ZoneInfo,
    day: date,
    session_start: time,
) -> Optional[tuple[datetime, float, float, float]]:
    for bar in bars:
        ts = bar[0]
        local = ts.astimezone(zone)
        if local.date() != day:
            continue
        if DAILY_POLICY_MARKET_DROP_OPEN_BAR_MODE == "containing":
            local_end = local + timedelta(hours=1)
            if local.time() <= session_start < local_end.time() or local.time() >= session_start:
                return bar
        elif local.time() >= session_start:
            return bar
    return None


def _bar_local_time_in_session(ts: datetime, zone: ZoneInfo, session_start: time, session_end: time) -> bool:
    local_time = ts.astimezone(zone).time()
    return session_start <= local_time <= session_end
