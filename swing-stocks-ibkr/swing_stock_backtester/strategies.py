from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from statistics import mean, median
from typing import Any, Callable

from .config import BacktestConfig
from .indicators import enrich_price_rows, first_index_after, safe_float
from .models import EarningsEvent, StockIdentity, StrategyResult, Trade


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: str
    max_hold_days: int
    initial_stop_atr: float
    trailing_stop_atr: float
    fallback_stop_pct: float
    entry_checker: Callable[[list[dict[str, Any]], int, BacktestConfig], tuple[bool, str, float | None]]
    exit_checker: Callable[[list[dict[str, Any]], int], bool]


def price_is_tradeable(row: dict[str, Any], cfg: BacktestConfig) -> bool:
    close = safe_float(row.get("close"))
    market_cap = safe_float(row.get("market_cap"))
    avg_volume = safe_float(row.get("avg_volume_20")) or safe_float(row.get("average_daily_volume_3m"))
    if close is None or close < cfg.min_price:
        return False
    if market_cap is None or market_cap < cfg.min_market_cap:
        return False
    if avg_volume is None or avg_volume < cfg.min_average_daily_volume:
        return False
    return True


def above_primary_trend(row: dict[str, Any]) -> bool:
    close = safe_float(row.get("close"))
    sma50 = safe_float(row.get("sma_50"))
    sma200 = safe_float(row.get("sma_200"))
    if close is None or sma50 is None or sma200 is None:
        return False
    return close > sma50 > sma200


def quality_momentum_entry(
    rows: list[dict[str, Any]],
    index: int,
    cfg: BacktestConfig,
) -> tuple[bool, str, float | None]:
    row = rows[index]
    if not price_is_tradeable(row, cfg) or not above_primary_trend(row):
        return False, "", None

    quality = safe_float(row.get("quality_score"))
    momentum = safe_float(row.get("momentum_score"))
    ret126 = safe_float(row.get("ret_126"))
    week52 = safe_float(row.get("week_52_change")) or safe_float(row.get("ret_252"))
    ret20 = safe_float(row.get("ret_20"))
    if quality is None or momentum is None or ret126 is None:
        return False, "", None
    if quality < 60.0 or momentum < 55.0:
        return False, "", None
    if ret126 < 0.08:
        return False, "", None
    if week52 is not None and week52 < 0.10:
        return False, "", None
    if ret20 is not None and ret20 < -0.08:
        return False, "", None

    signal_score = 0.55 * momentum + 0.45 * quality
    return True, "quality>=60 momentum>=55 trend_above_50_200", signal_score


def quality_momentum_exit(rows: list[dict[str, Any]], index: int) -> bool:
    row = rows[index]
    close = safe_float(row.get("close"))
    sma50 = safe_float(row.get("sma_50"))
    momentum = safe_float(row.get("momentum_score"))
    if close is not None and sma50 is not None and close < sma50:
        return True
    return momentum is not None and momentum < 45.0


def trend_pullback_entry(
    rows: list[dict[str, Any]],
    index: int,
    cfg: BacktestConfig,
) -> tuple[bool, str, float | None]:
    if index == 0:
        return False, "", None
    row = rows[index]
    prev = rows[index - 1]
    if not price_is_tradeable(row, cfg):
        return False, "", None

    close = safe_float(row.get("close"))
    open_price = safe_float(row.get("open"))
    low = safe_float(row.get("low"))
    prev_close = safe_float(prev.get("close"))
    sma20 = safe_float(row.get("sma_20"))
    sma50 = safe_float(row.get("sma_50"))
    sma200 = safe_float(row.get("sma_200"))
    ret5 = safe_float(row.get("ret_5"))
    ret126 = safe_float(row.get("ret_126"))
    quality = safe_float(row.get("quality_score"))
    momentum = safe_float(row.get("momentum_score"))
    if None in {close, open_price, low, prev_close, sma20, sma50, sma200, ret126}:
        return False, "", None
    if not (close > sma50 > sma200 and ret126 >= 0.05):
        return False, "", None
    if quality is not None and quality < 40.0:
        return False, "", None
    touched_pullback_zone = low <= sma20 * 1.015 or close <= sma20 * 1.01
    reclaimed = close >= open_price and close >= prev_close * 0.995
    not_freefall = ret5 is None or ret5 > -0.12
    if not (touched_pullback_zone and reclaimed and not_freefall):
        return False, "", None

    q = quality if quality is not None else 50.0
    m = momentum if momentum is not None else 50.0
    signal_score = 0.45 * m + 0.35 * q + 20.0
    return True, "uptrend_pullback_to_20dma_reclaim", min(signal_score, 100.0)


def trend_pullback_exit(rows: list[dict[str, Any]], index: int) -> bool:
    row = rows[index]
    close = safe_float(row.get("close"))
    sma20 = safe_float(row.get("sma_20"))
    sma50 = safe_float(row.get("sma_50"))
    if close is None:
        return False
    if sma20 is not None and close < sma20 * 0.96:
        return True
    return sma50 is not None and close < sma50 * 0.985


def attach_earnings_reaction_features(rows: list[dict[str, Any]], events: list[EarningsEvent]) -> None:
    days = [row["day"] for row in rows]
    for event in events:
        known_ts = event.known_as_of_ts or event.announcement_ts
        if known_ts is None:
            continue
        known_date = known_ts.date()
        reaction_index = first_index_after(days, known_date)
        if reaction_index is None or reaction_index <= 0:
            continue
        row = rows[reaction_index]
        previous = rows[reaction_index - 1]
        close = safe_float(row.get("close"))
        open_price = safe_float(row.get("open"))
        previous_close = safe_float(previous.get("close"))
        if close is None or previous_close is None or previous_close <= 0:
            continue
        row["earnings_event_date"] = event.earnings_date
        row["earnings_known_asof_ts"] = known_ts
        row["earnings_source"] = event.source
        row["earnings_surprise_pct"] = event.surprise_pct
        row["earnings_is_confirmed"] = event.is_confirmed
        row["earnings_reaction_return"] = close / previous_close - 1.0
        row["earnings_intraday_reaction_return"] = close / open_price - 1.0 if open_price and open_price > 0 else None


def earnings_reaction_entry(
    rows: list[dict[str, Any]],
    index: int,
    cfg: BacktestConfig,
) -> tuple[bool, str, float | None]:
    row = rows[index]
    if row.get("earnings_known_asof_ts") is None:
        return False, "", None
    if not price_is_tradeable(row, cfg):
        return False, "", None

    close = safe_float(row.get("close"))
    sma20 = safe_float(row.get("sma_20"))
    sma50 = safe_float(row.get("sma_50"))
    reaction_return = safe_float(row.get("earnings_reaction_return"))
    intraday_reaction = safe_float(row.get("earnings_intraday_reaction_return"))
    volume = safe_float(row.get("volume"))
    avg_volume = safe_float(row.get("avg_volume_20"))
    surprise_pct = safe_float(row.get("earnings_surprise_pct"))
    if close is None or sma20 is None or reaction_return is None:
        return False, "", None
    if close < sma20:
        return False, "", None
    if sma50 is not None and close < sma50 * 0.97:
        return False, "", None
    if reaction_return < 0.025 and (intraday_reaction is None or intraday_reaction < 0.015):
        return False, "", None
    if avg_volume is not None and volume is not None and volume < avg_volume * 1.10:
        return False, "", None
    if surprise_pct is not None and surprise_pct < -2.0:
        return False, "", None

    signal_score = min(100.0, 50.0 + reaction_return * 900.0 + max(surprise_pct or 0.0, 0.0))
    return True, "positive_post_earnings_reaction_next_day_entry", signal_score


def earnings_reaction_exit(rows: list[dict[str, Any]], index: int) -> bool:
    row = rows[index]
    close = safe_float(row.get("close"))
    sma10 = safe_float(row.get("sma_10"))
    if close is None or sma10 is None:
        return False
    return close < sma10 * 0.985


STRATEGIES: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="quality_momentum_swing",
        version="1.0",
        max_hold_days=45,
        initial_stop_atr=2.5,
        trailing_stop_atr=3.0,
        fallback_stop_pct=0.10,
        entry_checker=quality_momentum_entry,
        exit_checker=quality_momentum_exit,
    ),
    StrategySpec(
        name="trend_pullback",
        version="1.0",
        max_hold_days=30,
        initial_stop_atr=2.2,
        trailing_stop_atr=2.6,
        fallback_stop_pct=0.08,
        entry_checker=trend_pullback_entry,
        exit_checker=trend_pullback_exit,
    ),
    StrategySpec(
        name="earnings_reaction_drift",
        version="1.0",
        max_hold_days=20,
        initial_stop_atr=2.5,
        trailing_stop_atr=2.8,
        fallback_stop_pct=0.09,
        entry_checker=earnings_reaction_entry,
        exit_checker=earnings_reaction_exit,
    ),
)


def execution_price(row: dict[str, Any], side: str, cfg: BacktestConfig) -> float | None:
    price = safe_float(row.get("open")) or safe_float(row.get("close"))
    if price is None or price <= 0:
        return None
    slip = cfg.slippage_bps / 10_000.0
    if side == "buy":
        return price * (1.0 + slip)
    return price * (1.0 - slip)


def stop_from_signal(entry_price: float, signal_row: dict[str, Any], spec: StrategySpec) -> float:
    atr = safe_float(signal_row.get("atr_14"))
    if atr is not None and atr > 0:
        return max(0.01, entry_price - spec.initial_stop_atr * atr)
    return max(0.01, entry_price * (1.0 - spec.fallback_stop_pct))


def stop_exit_price(row: dict[str, Any], stop_price: float, cfg: BacktestConfig) -> float | None:
    low = safe_float(row.get("low"))
    open_price = safe_float(row.get("open"))
    if low is None or low > stop_price:
        return None
    raw_exit = min(open_price, stop_price) if open_price and open_price > 0 else stop_price
    return raw_exit * (1.0 - cfg.slippage_bps / 10_000.0)


def summarize_result(
    identity: StockIdentity,
    spec: StrategySpec,
    status: str,
    trades: list[Trade],
    signal_count: int,
    skipped_signal_count: int,
    error_text: str | None = None,
) -> StrategyResult:
    if not trades:
        return StrategyResult(
            strategy_name=spec.name,
            strategy_version=spec.version,
            identity=identity,
            status=status,
            signal_count=signal_count,
            skipped_signal_count=skipped_signal_count,
            error_text=error_text,
        )

    returns = [trade.net_return_pct for trade in trades]
    gains = [value for value in returns if value > 0]
    losses = [value for value in returns if value < 0]
    compounded = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        compounded *= 1.0 + value / 100.0
        peak = max(peak, compounded)
        if peak > 0:
            max_drawdown = min(max_drawdown, compounded / peak - 1.0)

    return StrategyResult(
        strategy_name=spec.name,
        strategy_version=spec.version,
        identity=identity,
        status=status,
        first_trade_date=min(trade.entry_date for trade in trades),
        last_trade_date=max(trade.exit_date for trade in trades),
        trade_count=len(trades),
        win_count=len(gains),
        loss_count=len(losses),
        flat_count=len([value for value in returns if value == 0]),
        avg_return_pct=mean(returns),
        median_return_pct=median(returns),
        best_return_pct=max(returns),
        worst_return_pct=min(returns),
        total_compounded_return_pct=(compounded - 1.0) * 100.0,
        max_drawdown_pct=max_drawdown * 100.0,
        profit_factor=sum(gains) / abs(sum(losses)) if losses and sum(losses) != 0 else None,
        expectancy_pct=mean(returns),
        avg_holding_days=mean([trade.holding_days for trade in trades]),
        exposure_days=sum(trade.holding_days for trade in trades),
        signal_count=signal_count,
        skipped_signal_count=skipped_signal_count,
        trades=trades,
    )


def maybe_make_equity_rows(
    identity: StockIdentity,
    spec: StrategySpec,
    rows: list[dict[str, Any]],
    trades: list[Trade],
    cfg: BacktestConfig,
) -> list[dict[str, Any]]:
    if not cfg.write_equity_daily:
        return []

    trades_by_exit: dict[date, list[Trade]] = {}
    position_dates: set[date] = set()
    for trade in trades:
        trades_by_exit.setdefault(trade.exit_date, []).append(trade)
        for row in rows:
            day = row["day"]
            if trade.entry_date <= day <= trade.exit_date:
                position_dates.add(day)

    equity = 1.0
    peak = 1.0
    out: list[dict[str, Any]] = []
    for row in rows:
        day = row["day"]
        if day < cfg.start_date or day > cfg.end_date:
            continue
        for trade in trades_by_exit.get(day, []):
            equity *= 1.0 + trade.net_return_pct / 100.0
        peak = max(peak, equity)
        drawdown = equity / peak - 1.0 if peak > 0 else 0.0
        out.append(
            {
                "strategy_name": spec.name,
                "strategy_version": spec.version,
                "identity": identity,
                "day": day,
                "equity": equity,
                "drawdown_pct": drawdown * 100.0,
                "in_position": day in position_dates,
            }
        )
    return out


def simulate_strategy(
    identity: StockIdentity,
    rows: list[dict[str, Any]],
    spec: StrategySpec,
    cfg: BacktestConfig,
) -> StrategyResult:
    trades: list[Trade] = []
    signal_count = 0
    skipped_signal_count = 0
    index = 0

    while index < len(rows) - 1:
        row = rows[index]
        day = row["day"]
        if day < cfg.start_date:
            index += 1
            continue
        if day > cfg.end_date:
            break

        is_entry, condition, signal_score = spec.entry_checker(rows, index, cfg)
        if not is_entry:
            index += 1
            continue
        signal_count += 1

        entry_index = index + 1
        entry_row = rows[entry_index]
        entry_price = execution_price(entry_row, "buy", cfg)
        if entry_price is None:
            skipped_signal_count += 1
            index += 1
            continue

        signal_row = rows[index]
        stop_price = stop_from_signal(entry_price, signal_row, spec)
        highest_close = entry_price
        exit_index = entry_index
        exit_price: float | None = None
        exit_reason = "end_of_data"
        final_stop_price = stop_price

        for current_index in range(entry_index, len(rows)):
            current = rows[current_index]
            current_day = current["day"]
            if current_day > cfg.end_date:
                break

            current_close = safe_float(current.get("close"))
            if current_close is not None:
                highest_close = max(highest_close, current_close)
            atr = safe_float(current.get("atr_14"))
            if atr is not None and atr > 0:
                stop_price = max(stop_price, highest_close - spec.trailing_stop_atr * atr)
            final_stop_price = stop_price

            stopped = stop_exit_price(current, stop_price, cfg)
            if stopped is not None:
                exit_index = current_index
                exit_price = stopped
                exit_reason = "stop"
                break

            holding_days = (current_day - entry_row["day"]).days
            close_exit = spec.exit_checker(rows, current_index)
            max_hold_exit = holding_days >= spec.max_hold_days
            if close_exit or max_hold_exit:
                next_index = min(current_index + 1, len(rows) - 1)
                next_row = rows[next_index]
                exit_index = next_index
                exit_price = execution_price(next_row, "sell", cfg)
                exit_reason = "close_signal" if close_exit else "max_hold"
                break

        if exit_price is None:
            exit_row = rows[exit_index]
            raw_exit = safe_float(exit_row.get("close"))
            if raw_exit is None or raw_exit <= 0:
                skipped_signal_count += 1
                index = max(index + 1, exit_index + 1)
                continue
            exit_price = raw_exit * (1.0 - cfg.slippage_bps / 10_000.0)

        exit_row = rows[exit_index]
        gross_return = exit_price / entry_price - 1.0
        net_return = gross_return - (2.0 * cfg.commission_bps / 10_000.0)
        trade = Trade(
            strategy_name=spec.name,
            strategy_version=spec.version,
            identity=identity,
            trade_number=len(trades) + 1,
            signal_date=signal_row["day"],
            entry_date=entry_row["day"],
            exit_date=exit_row["day"],
            entry_price=entry_price,
            exit_price=exit_price,
            stop_price=final_stop_price,
            gross_return_pct=gross_return * 100.0,
            net_return_pct=net_return * 100.0,
            holding_days=max(0, (exit_row["day"] - entry_row["day"]).days),
            exit_reason=exit_reason,
            signal_score=signal_score,
            quality_score=safe_float(signal_row.get("quality_score")),
            momentum_score=safe_float(signal_row.get("momentum_score")),
            entry_condition=condition,
            fundamental_asof_date=signal_row.get("fundamental_asof_date"),
            earnings_event_date=signal_row.get("earnings_event_date"),
            earnings_known_asof_ts=signal_row.get("earnings_known_asof_ts"),
        )
        trades.append(trade)
        index = max(exit_index + 1, index + 1)

    status = "ok"
    result = summarize_result(identity, spec, status, trades, signal_count, skipped_signal_count)
    result.equity_rows = maybe_make_equity_rows(identity, spec, rows, trades, cfg)
    return result


def empty_strategy_results(identity: StockIdentity, status: str, error_text: str | None = None) -> list[StrategyResult]:
    return [
        StrategyResult(
            strategy_name=spec.name,
            strategy_version=spec.version,
            identity=identity,
            status=status,
            error_text=error_text,
        )
        for spec in STRATEGIES
    ]


def run_strategies_for_symbol(
    identity: StockIdentity,
    rows: list[dict[str, Any]],
    earnings_events: list[EarningsEvent],
    cfg: BacktestConfig,
) -> list[StrategyResult]:
    if not rows:
        return empty_strategy_results(identity, "no_data")
    if len(rows) < cfg.min_history_days:
        return empty_strategy_results(identity, "insufficient_history")

    enrich_price_rows(rows)
    attach_earnings_reaction_features(rows, earnings_events)
    return [simulate_strategy(identity, rows, spec, cfg) for spec in STRATEGIES]
