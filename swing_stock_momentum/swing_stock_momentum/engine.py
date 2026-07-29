from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, ROUND_FLOOR
from typing import Any, Callable, Iterable, Mapping, Sequence

from .config import StrategyParameters
from .contracts import ANALYSER_CRITERION_COLUMNS, ANALYSER_PAYLOAD_COLUMNS


ZERO = Decimal("0")
ONE_HUNDRED = Decimal("100")
TEN_THOUSAND = Decimal("10000")


def _decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _positive(value: Decimal | None) -> bool:
    return value is not None and value > ZERO


def _whole_shares(value: Decimal) -> int:
    if value <= ZERO:
        return 0
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


@dataclass(frozen=True)
class Bar:
    period_end_date: date
    symbol: str
    exchange: str
    cik: int
    price_continuity_segment: int
    adjusted_open: Decimal | None
    adjusted_high: Decimal | None
    adjusted_low: Decimal | None
    adjusted_close: Decimal | None
    atr_pct: Decimal | None
    prior_high_observation_count: int
    prior_max_adjusted_high: Decimal | None
    analyser: Mapping[str, Any] | None = None
    earnings_horizon_complete: bool = True
    next_earnings_date: date | None = None
    next_earnings_sessions_ahead: int | None = None

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.symbol, self.exchange, self.cik

    @classmethod
    def from_market_row(
        cls,
        row: Sequence[Any],
        analyser: Mapping[str, Any] | None,
        earnings_horizon_complete: bool = True,
        next_earnings_date: date | None = None,
        next_earnings_sessions_ahead: int | None = None,
    ) -> "Bar":
        if len(row) != 12:
            raise ValueError(f"market row must contain 12 columns, got {len(row)}")
        return cls(
            period_end_date=row[0],
            symbol=str(row[1]),
            exchange=str(row[2]),
            cik=int(row[3]),
            price_continuity_segment=int(row[4]),
            adjusted_open=_decimal(row[5]),
            adjusted_high=_decimal(row[6]),
            adjusted_low=_decimal(row[7]),
            adjusted_close=_decimal(row[8]),
            atr_pct=_decimal(row[9]),
            prior_high_observation_count=int(row[10] or 0),
            prior_max_adjusted_high=_decimal(row[11]),
            analyser=analyser,
            earnings_horizon_complete=earnings_horizon_complete,
            next_earnings_date=next_earnings_date,
            next_earnings_sessions_ahead=next_earnings_sessions_ahead,
        )


@dataclass
class Position:
    trade_id: int
    symbol: str
    exchange: str
    cik: int
    entry_date: date
    shares: int
    entry_reference_price: Decimal
    entry_fill_price: Decimal
    entry_commission_usd: Decimal
    risk_budget_usd: Decimal
    planned_initial_stop_loss_usd: Decimal
    holding_sessions: int = 0
    last_valuation_date: date | None = None
    last_mark_price: Decimal | None = None
    exit_date: date | None = None
    exit_reason: str | None = None
    active_stop_price_at_exit: Decimal | None = None
    active_take_profit_price_at_exit: Decimal | None = None
    exit_reference_price: Decimal | None = None
    exit_fill_price: Decimal | None = None
    exit_commission_usd: Decimal | None = None

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.symbol, self.exchange, self.cik

    @property
    def entry_notional_usd(self) -> Decimal:
        return self.entry_fill_price * self.shares

    @property
    def entry_cash_cost_usd(self) -> Decimal:
        return self.entry_notional_usd + self.entry_commission_usd


@dataclass(frozen=True)
class BacktestResult:
    actual_start_date: date
    end_date: date
    ending_cash_usd: Decimal
    ending_market_value_usd: Decimal
    ending_equity_usd: Decimal
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal
    total_return_pct: Decimal
    max_drawdown_pct: Decimal
    total_commission_usd: Decimal
    signal_decisions: tuple[dict[str, Any], ...]
    trades: tuple[dict[str, Any], ...]
    equity_daily: tuple[dict[str, Any], ...]

    @property
    def selected_signal_count(self) -> int:
        return sum(bool(row["selected"]) for row in self.signal_decisions)

    @property
    def closed_trade_count(self) -> int:
        return sum(row["status"] == "closed" for row in self.trades)

    @property
    def open_trade_count(self) -> int:
        return sum(row["status"] == "open" for row in self.trades)

    @property
    def winning_trade_count(self) -> int:
        return sum(
            row["status"] == "closed" and row["net_pnl_usd"] > ZERO
            for row in self.trades
        )

    @property
    def losing_trade_count(self) -> int:
        return sum(
            row["status"] == "closed" and row["net_pnl_usd"] < ZERO
            for row in self.trades
        )


@dataclass(frozen=True)
class BacktestProgress:
    sessions_processed: int
    valuation_date: date
    signal_count: int
    selected_signal_count: int
    open_position_count: int
    closed_trade_count: int
    total_equity_usd: Decimal


@dataclass
class _Portfolio:
    cash_usd: Decimal
    open_positions: dict[str, Position] = field(default_factory=dict)
    all_positions: list[Position] = field(default_factory=list)
    realized_pnl_usd: Decimal = ZERO
    total_commission_usd: Decimal = ZERO
    closed_trade_count: int = 0
    next_trade_id: int = 1


def active_exit_levels(
    position: Position, strategy: StrategyParameters
) -> tuple[Decimal, Decimal]:
    completed_before_session = max(0, position.holding_sessions - 1)
    stop_block = completed_before_session // strategy.stop_step_interval_sessions
    take_block = (
        completed_before_session // strategy.take_profit_step_interval_sessions
    )
    stop_pct = strategy.initial_stop_loss_pct + strategy.stop_step_pct * stop_block
    take_pct = (
        strategy.initial_take_profit_pct + strategy.take_profit_step_pct * take_block
    )
    return (
        position.entry_fill_price * (ONE_HUNDRED + stop_pct) / ONE_HUNDRED,
        position.entry_fill_price * (ONE_HUNDRED + take_pct) / ONE_HUNDRED,
    )


def _sell_fill(reference_price: Decimal, strategy: StrategyParameters) -> Decimal:
    return reference_price * (TEN_THOUSAND - strategy.slippage_bps) / TEN_THOUSAND


def _buy_fill(reference_price: Decimal, strategy: StrategyParameters) -> Decimal:
    return reference_price * (TEN_THOUSAND + strategy.slippage_bps) / TEN_THOUSAND


def _commission(notional: Decimal, strategy: StrategyParameters) -> Decimal:
    return notional * strategy.commission_bps / TEN_THOUSAND


def _exit_trigger(
    position: Position,
    bar: Bar,
    strategy: StrategyParameters,
) -> tuple[str, Decimal, Decimal, Decimal] | None:
    stop_price, take_profit_price = active_exit_levels(position, strategy)
    open_price = bar.adjusted_open
    if _positive(open_price):
        assert open_price is not None
        if open_price <= stop_price:
            return "stop_loss_gap", open_price, stop_price, take_profit_price
        if open_price >= take_profit_price:
            return "take_profit_gap", open_price, stop_price, take_profit_price

    low_price = bar.adjusted_low
    high_price = bar.adjusted_high
    if _positive(low_price) and low_price <= stop_price:
        return "stop_loss_intraday", stop_price, stop_price, take_profit_price
    if _positive(high_price) and high_price >= take_profit_price:
        return (
            "take_profit_intraday",
            take_profit_price,
            stop_price,
            take_profit_price,
        )

    if _positive(bar.adjusted_close):
        if (
            position.holding_sessions == 1
            and bar.atr_pct is not None
            and bar.atr_pct <= strategy.atr_day1_exit_max_pct
        ):
            return "atr_day1", bar.adjusted_close, stop_price, take_profit_price
        if (
            position.holding_sessions == 2
            and bar.atr_pct is not None
            and bar.atr_pct <= strategy.atr_day2_exit_max_pct
        ):
            return "atr_day2", bar.adjusted_close, stop_price, take_profit_price
    return None


def _passes_analyser_entry_filter(bar: Bar, strategy: StrategyParameters) -> bool:
    analyser = bar.analyser
    if analyser is None:
        return False
    if int(analyser.get("price_continuity_segment") or 0) != bar.price_continuity_segment:
        return False
    if analyser.get("currency") != strategy.currency:
        return False
    if not bool(analyser.get("trend_template_pass")):
        return False
    if not all(bool(analyser.get(column)) for column in ANALYSER_CRITERION_COLUMNS):
        return False
    daily_change = _decimal(analyser.get("daily_price_change_pct"))
    volume_ratio = _decimal(analyser.get("adjusted_volume_vs_sma21_prior_ratio"))
    return bool(
        daily_change is not None
        and strategy.min_daily_price_change_pct
        <= daily_change
        < strategy.max_daily_price_change_pct_exclusive
        and volume_ratio is not None
        and volume_ratio > strategy.min_volume_vs_sma21_ratio_exclusive
    )


def _signal_row(bar: Bar) -> dict[str, Any]:
    assert bar.analyser is not None
    row: dict[str, Any] = {
        "signal_date": bar.period_end_date,
        "symbol": bar.symbol,
        "exchange": bar.exchange,
        "cik": bar.cik,
        "selection_rank": None,
        "decision": None,
        "selected": False,
        "prior_high_observation_count": bar.prior_high_observation_count,
        "prior_max_adjusted_high": bar.prior_max_adjusted_high,
        "prior_high_limit_adjusted_price": None,
        "earnings_horizon_complete": bar.earnings_horizon_complete,
        "next_earnings_date": bar.next_earnings_date,
        "next_earnings_sessions_ahead": bar.next_earnings_sessions_ahead,
        "account_equity_before_entry_usd": None,
        "available_cash_before_entry_usd": None,
        "risk_budget_usd": None,
        "risk_per_share_usd": None,
        "risk_sized_shares": None,
        "cash_limited_shares": None,
        "selected_shares": None,
        "entry_reference_price": None,
        "entry_fill_price": None,
        "entry_commission_usd": None,
    }
    row.update(
        {column: bar.analyser.get(column) for column in ANALYSER_PAYLOAD_COLUMNS}
    )
    return row


def _mark_price(position: Position) -> Decimal:
    return position.last_mark_price or position.entry_reference_price


def _portfolio_equity(portfolio: _Portfolio) -> Decimal:
    return portfolio.cash_usd + sum(
        _mark_price(position) * position.shares
        for position in portfolio.open_positions.values()
    )


def _unrealized_pnl(portfolio: _Portfolio) -> Decimal:
    return sum(
        _mark_price(position) * position.shares - position.entry_cash_cost_usd
        for position in portfolio.open_positions.values()
    )


def _close_position(
    portfolio: _Portfolio,
    position: Position,
    exit_date: date,
    exit_reason: str,
    reference_price: Decimal,
    stop_price: Decimal,
    take_profit_price: Decimal,
    strategy: StrategyParameters,
) -> None:
    fill_price = _sell_fill(reference_price, strategy)
    exit_notional = fill_price * position.shares
    exit_commission = _commission(exit_notional, strategy)
    portfolio.cash_usd += exit_notional - exit_commission
    portfolio.realized_pnl_usd += (
        exit_notional
        - exit_commission
        - position.entry_notional_usd
        - position.entry_commission_usd
    )
    portfolio.total_commission_usd += exit_commission
    portfolio.closed_trade_count += 1
    position.exit_date = exit_date
    position.exit_reason = exit_reason
    position.active_stop_price_at_exit = stop_price
    position.active_take_profit_price_at_exit = take_profit_price
    position.exit_reference_price = reference_price
    position.exit_fill_price = fill_price
    position.exit_commission_usd = exit_commission
    position.last_valuation_date = exit_date
    position.last_mark_price = reference_price
    del portfolio.open_positions[position.symbol]


def _process_existing_positions(
    portfolio: _Portfolio,
    bars: Sequence[Bar],
    strategy: StrategyParameters,
    valuation_date: date,
) -> None:
    bars_by_identity = {bar.identity: bar for bar in bars}
    for position in tuple(portfolio.open_positions.values()):
        bar = bars_by_identity.get(position.identity)
        if bar is None or not _positive(bar.adjusted_close):
            continue
        position.holding_sessions += 1
        position.last_valuation_date = valuation_date
        position.last_mark_price = bar.adjusted_close
        trigger = _exit_trigger(position, bar, strategy)
        if trigger is not None:
            reason, reference_price, stop_price, take_profit_price = trigger
            _close_position(
                portfolio,
                position,
                valuation_date,
                reason,
                reference_price,
                stop_price,
                take_profit_price,
                strategy,
            )


def _size_position(
    reference_price: Decimal,
    equity_usd: Decimal,
    cash_usd: Decimal,
    strategy: StrategyParameters,
) -> tuple[Decimal, Decimal, int, int, int, Decimal, Decimal]:
    entry_fill = _buy_fill(reference_price, strategy)
    initial_stop = (
        entry_fill
        * (ONE_HUNDRED + strategy.initial_stop_loss_pct)
        / ONE_HUNDRED
    )
    stop_fill = _sell_fill(initial_stop, strategy)
    entry_commission_per_share = _commission(entry_fill, strategy)
    exit_commission_per_share = _commission(stop_fill, strategy)
    risk_per_share = (
        entry_fill
        + entry_commission_per_share
        - stop_fill
        + exit_commission_per_share
    )
    risk_budget = equity_usd * strategy.risk_per_position_pct / ONE_HUNDRED
    risk_sized_shares = _whole_shares(risk_budget / risk_per_share)
    cash_cost_per_share = entry_fill + entry_commission_per_share
    cash_limited_shares = _whole_shares(cash_usd / cash_cost_per_share)
    shares = min(risk_sized_shares, cash_limited_shares)
    return (
        entry_fill,
        risk_per_share,
        risk_sized_shares,
        cash_limited_shares,
        shares,
        risk_budget,
        initial_stop,
    )


def _process_new_entries(
    portfolio: _Portfolio,
    bars: Sequence[Bar],
    strategy: StrategyParameters,
    signal_decisions: list[dict[str, Any]],
) -> None:
    eligible: list[tuple[Bar, dict[str, Any]]] = []
    opened_today = 0
    for bar in bars:
        if not _passes_analyser_entry_filter(bar, strategy):
            continue
        decision = _signal_row(bar)
        close_price = bar.adjusted_close
        if not _positive(close_price):
            decision["decision"] = "invalid_execution_price"
            signal_decisions.append(decision)
            continue
        assert close_price is not None
        limit_price = (
            close_price
            * (ONE_HUNDRED + strategy.prior_high_max_above_signal_close_pct)
            / ONE_HUNDRED
        )
        decision["prior_high_limit_adjusted_price"] = limit_price
        if bar.prior_high_observation_count < strategy.prior_high_lookback_sessions:
            decision["decision"] = "prior_high_history_incomplete"
            signal_decisions.append(decision)
            continue
        if (
            bar.prior_max_adjusted_high is None
            or bar.prior_max_adjusted_high > limit_price
        ):
            decision["decision"] = "prior_high_limit_exceeded"
            signal_decisions.append(decision)
            continue
        if bar.next_earnings_date is not None:
            if (
                bar.next_earnings_sessions_ahead is None
                or not 1
                <= bar.next_earnings_sessions_ahead
                <= strategy.earnings_blackout_sessions
            ):
                raise RuntimeError(
                    f"invalid earnings-session distance for {bar.identity} on "
                    f"{bar.period_end_date}"
                )
            decision["decision"] = "earnings_blackout"
            signal_decisions.append(decision)
            continue
        if not bar.earnings_horizon_complete:
            decision["decision"] = "earnings_horizon_incomplete"
            signal_decisions.append(decision)
            continue
        eligible.append((bar, decision))

    eligible.sort(
        key=lambda item: (
            -(_decimal(item[0].analyser["adjusted_volume_vs_sma21_prior_ratio"]) or ZERO),
            -(_decimal(item[0].analyser["daily_price_change_pct"]) or ZERO),
            item[0].symbol,
            item[0].exchange,
            item[0].cik,
        )
    )

    for rank, (bar, decision) in enumerate(eligible, start=1):
        decision["selection_rank"] = rank
        decision["account_equity_before_entry_usd"] = _portfolio_equity(portfolio)
        decision["available_cash_before_entry_usd"] = portfolio.cash_usd
        if bar.symbol in portfolio.open_positions:
            decision["decision"] = "symbol_already_open"
            signal_decisions.append(decision)
            continue
        if len(portfolio.open_positions) >= strategy.max_positions:
            decision["decision"] = "position_limit_reached"
            signal_decisions.append(decision)
            continue
        if opened_today >= strategy.max_new_positions_per_day:
            decision["decision"] = "daily_entry_limit_reached"
            signal_decisions.append(decision)
            continue

        assert bar.adjusted_close is not None
        (
            entry_fill,
            risk_per_share,
            risk_sized_shares,
            cash_limited_shares,
            shares,
            risk_budget,
            initial_stop,
        ) = _size_position(
            bar.adjusted_close,
            decision["account_equity_before_entry_usd"],
            portfolio.cash_usd,
            strategy,
        )
        decision.update(
            {
                "risk_budget_usd": risk_budget,
                "risk_per_share_usd": risk_per_share,
                "risk_sized_shares": risk_sized_shares,
                "cash_limited_shares": cash_limited_shares,
                "entry_reference_price": bar.adjusted_close,
                "entry_fill_price": entry_fill,
            }
        )
        if shares < 1:
            decision["decision"] = (
                "risk_budget_below_one_share"
                if risk_sized_shares < 1
                else "insufficient_cash"
            )
            signal_decisions.append(decision)
            continue

        entry_notional = entry_fill * shares
        entry_commission = _commission(entry_notional, strategy)
        planned_stop_loss = risk_per_share * shares
        position = Position(
            trade_id=portfolio.next_trade_id,
            symbol=bar.symbol,
            exchange=bar.exchange,
            cik=bar.cik,
            entry_date=bar.period_end_date,
            shares=shares,
            entry_reference_price=bar.adjusted_close,
            entry_fill_price=entry_fill,
            entry_commission_usd=entry_commission,
            risk_budget_usd=risk_budget,
            planned_initial_stop_loss_usd=planned_stop_loss,
            last_valuation_date=bar.period_end_date,
            last_mark_price=bar.adjusted_close,
        )
        portfolio.next_trade_id += 1
        portfolio.cash_usd -= entry_notional + entry_commission
        portfolio.total_commission_usd += entry_commission
        portfolio.open_positions[bar.symbol] = position
        portfolio.all_positions.append(position)
        opened_today += 1
        decision.update(
            {
                "decision": "selected",
                "selected": True,
                "selected_shares": shares,
                "entry_commission_usd": entry_commission,
            }
        )
        signal_decisions.append(decision)


def _trade_row(position: Position, strategy: StrategyParameters) -> dict[str, Any]:
    initial_stop = (
        position.entry_fill_price
        * (ONE_HUNDRED + strategy.initial_stop_loss_pct)
        / ONE_HUNDRED
    )
    initial_take_profit = (
        position.entry_fill_price
        * (ONE_HUNDRED + strategy.initial_take_profit_pct)
        / ONE_HUNDRED
    )
    is_closed = position.exit_date is not None
    exit_notional = (
        position.exit_fill_price * position.shares
        if position.exit_fill_price is not None
        else None
    )
    mark_price = position.last_mark_price or position.entry_reference_price
    market_value = ZERO if is_closed else mark_price * position.shares
    gross_pnl = (
        (position.exit_fill_price - position.entry_fill_price) * position.shares
        if position.exit_fill_price is not None
        else (mark_price - position.entry_fill_price) * position.shares
    )
    net_pnl = gross_pnl - position.entry_commission_usd
    if position.exit_commission_usd is not None:
        net_pnl -= position.exit_commission_usd
    return_pct = net_pnl / position.entry_cash_cost_usd * ONE_HUNDRED
    return {
        "trade_id": position.trade_id,
        "symbol": position.symbol,
        "exchange": position.exchange,
        "cik": position.cik,
        "entry_date": position.entry_date,
        "exit_date": position.exit_date,
        "status": "closed" if is_closed else "open",
        "exit_reason": position.exit_reason,
        "shares": position.shares,
        "holding_sessions": position.holding_sessions,
        "entry_reference_price": position.entry_reference_price,
        "entry_fill_price": position.entry_fill_price,
        "entry_notional_usd": position.entry_notional_usd,
        "entry_commission_usd": position.entry_commission_usd,
        "risk_budget_usd": position.risk_budget_usd,
        "planned_initial_stop_loss_usd": position.planned_initial_stop_loss_usd,
        "initial_stop_price": initial_stop,
        "initial_take_profit_price": initial_take_profit,
        "active_stop_price_at_exit": position.active_stop_price_at_exit,
        "active_take_profit_price_at_exit": position.active_take_profit_price_at_exit,
        "exit_reference_price": position.exit_reference_price,
        "exit_fill_price": position.exit_fill_price,
        "exit_notional_usd": exit_notional,
        "exit_commission_usd": position.exit_commission_usd,
        "last_valuation_date": position.last_valuation_date,
        "last_mark_price": mark_price,
        "market_value_usd": market_value,
        "gross_pnl_usd": gross_pnl,
        "net_pnl_usd": net_pnl,
        "return_pct": return_pct,
    }


def run_backtest(
    market_days: Iterable[tuple[date, Sequence[Bar]]],
    strategy: StrategyParameters,
    progress_callback: Callable[[BacktestProgress], None] | None = None,
) -> BacktestResult:
    strategy.validate()
    portfolio = _Portfolio(cash_usd=strategy.starting_capital_usd)
    signal_decisions: list[dict[str, Any]] = []
    equity_daily: list[dict[str, Any]] = []
    previous_equity = strategy.starting_capital_usd
    peak_equity = strategy.starting_capital_usd
    max_drawdown = ZERO
    actual_start_date: date | None = None
    end_date: date | None = None

    for valuation_date, bars in market_days:
        if valuation_date < strategy.requested_start_date:
            continue
        if actual_start_date is None:
            actual_start_date = valuation_date
        end_date = valuation_date
        _process_existing_positions(portfolio, bars, strategy, valuation_date)
        _process_new_entries(portfolio, bars, strategy, signal_decisions)

        market_value = sum(
            _mark_price(position) * position.shares
            for position in portfolio.open_positions.values()
        )
        total_equity = portfolio.cash_usd + market_value
        unrealized = _unrealized_pnl(portfolio)
        peak_equity = max(peak_equity, total_equity)
        drawdown = (
            (total_equity / peak_equity - Decimal("1")) * ONE_HUNDRED
            if peak_equity > ZERO
            else ZERO
        )
        max_drawdown = min(max_drawdown, drawdown)
        daily_return = (
            (total_equity / previous_equity - Decimal("1")) * ONE_HUNDRED
            if previous_equity > ZERO
            else ZERO
        )
        equity_daily.append(
            {
                "valuation_date": valuation_date,
                "cash_usd": portfolio.cash_usd,
                "positions_market_value_usd": market_value,
                "total_equity_usd": total_equity,
                "realized_pnl_usd": portfolio.realized_pnl_usd,
                "unrealized_pnl_usd": unrealized,
                "cumulative_commission_usd": portfolio.total_commission_usd,
                "open_position_count": len(portfolio.open_positions),
                "closed_trade_count": portfolio.closed_trade_count,
                "daily_return_pct": daily_return,
                "total_return_pct": (
                    total_equity / strategy.starting_capital_usd - Decimal("1")
                )
                * ONE_HUNDRED,
                "drawdown_pct": drawdown,
            }
        )
        if progress_callback is not None:
            progress_callback(
                BacktestProgress(
                    sessions_processed=len(equity_daily),
                    valuation_date=valuation_date,
                    signal_count=len(signal_decisions),
                    selected_signal_count=len(portfolio.all_positions),
                    open_position_count=len(portfolio.open_positions),
                    closed_trade_count=portfolio.closed_trade_count,
                    total_equity_usd=total_equity,
                )
            )
        previous_equity = total_equity

    if actual_start_date is None or end_date is None or not equity_daily:
        raise RuntimeError("no market sessions are available in the requested range")

    trades = tuple(_trade_row(position, strategy) for position in portfolio.all_positions)
    ending = equity_daily[-1]
    return BacktestResult(
        actual_start_date=actual_start_date,
        end_date=end_date,
        ending_cash_usd=ending["cash_usd"],
        ending_market_value_usd=ending["positions_market_value_usd"],
        ending_equity_usd=ending["total_equity_usd"],
        realized_pnl_usd=ending["realized_pnl_usd"],
        unrealized_pnl_usd=ending["unrealized_pnl_usd"],
        total_return_pct=ending["total_return_pct"],
        max_drawdown_pct=max_drawdown,
        total_commission_usd=portfolio.total_commission_usd,
        signal_decisions=tuple(signal_decisions),
        trades=trades,
        equity_daily=tuple(equity_daily),
    )
