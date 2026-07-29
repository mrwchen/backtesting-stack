from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal

import pytest

from swing_stock_momentum.config import StrategyParameters
from swing_stock_momentum.contracts import ANALYSER_CRITERION_COLUMNS
from swing_stock_momentum.engine import Bar, Position, active_exit_levels, run_backtest


D = Decimal


def strategy(**changes: object) -> StrategyParameters:
    base = StrategyParameters(
        account_type="unlevered",
        currency="USD",
        trading_timezone="America/New_York",
        requested_start_date=date(2026, 1, 1),
        starting_capital_usd=D("30000"),
        max_positions=2,
        max_new_positions_per_day=2,
        risk_per_position_pct=D("1"),
        initial_stop_loss_pct=D("-5"),
        stop_step_interval_sessions=5,
        stop_step_pct=D("5"),
        initial_take_profit_pct=D("10"),
        take_profit_step_interval_sessions=5,
        take_profit_step_pct=D("7.5"),
        atr_period_sessions=14,
        atr_day1_exit_max_pct=D("1.5"),
        atr_day2_exit_max_pct=D("2"),
        prior_high_lookback_sessions=10,
        prior_high_max_above_signal_close_pct=D("10"),
        min_daily_price_change_pct=D("1"),
        max_daily_price_change_pct_exclusive=D("5"),
        min_volume_vs_sma21_ratio_exclusive=D("1.2"),
        commission_bps=D("0"),
        slippage_bps=D("0"),
    )
    return replace(base, **changes)


def analyser(
    *, volume_ratio: str = "2", daily_change: str = "2", forward_gain: str = "99"
) -> dict[str, object]:
    result: dict[str, object] = {
        "period_end_date": date(2026, 1, 2),
        "symbol": "A",
        "exchange": "NASDAQ",
        "cik": 1,
        "price_continuity_segment": 1,
        "currency": "USD",
        "adjusted_close": D("100"),
        "adjusted_high": D("102"),
        "adjusted_low": D("99"),
        "daily_price_change_pct": D(daily_change),
        "adjusted_volume_vs_sma21_prior_ratio": D(volume_ratio),
        "rs_universe_size": 100,
        "forward_5d_max_gain_pct": D(forward_gain),
        "trend_template_pass": True,
    }
    result.update({column: True for column in ANALYSER_CRITERION_COLUMNS})
    return result


def bar(
    day: date,
    symbol: str,
    *,
    open_: str = "100",
    high: str = "104",
    low: str = "96",
    close: str = "100",
    atr: str | None = "3",
    signal: dict[str, object] | None = None,
    prior_high: str = "105",
    prior_count: int = 10,
    cik: int = 1,
) -> Bar:
    if signal is not None:
        signal = dict(signal)
        signal.update(
            {
                "period_end_date": day,
                "symbol": symbol,
                "exchange": "NASDAQ",
                "cik": cik,
            }
        )
    return Bar(
        period_end_date=day,
        symbol=symbol,
        exchange="NASDAQ",
        cik=cik,
        price_continuity_segment=1,
        adjusted_open=D(open_),
        adjusted_high=D(high),
        adjusted_low=D(low),
        adjusted_close=D(close),
        atr_pct=D(atr) if atr is not None else None,
        prior_high_observation_count=prior_count,
        prior_max_adjusted_high=D(prior_high) if prior_high else None,
        analyser=signal,
    )


def test_ranking_max_positions_and_one_percent_risk_sizing() -> None:
    day = date(2026, 1, 2)
    bars = (
        bar(day, "C", signal=analyser(volume_ratio="3", daily_change="2"), cik=3),
        bar(day, "A", signal=analyser(volume_ratio="2", daily_change="4"), cik=1),
        bar(day, "B", signal=analyser(volume_ratio="3", daily_change="3"), cik=2),
    )

    result = run_backtest([(day, bars)], strategy())

    selected = [row for row in result.signal_decisions if row["selected"]]
    assert [row["symbol"] for row in selected] == ["B", "C"]
    assert [row["selection_rank"] for row in selected] == [1, 2]
    assert all(row["selected_shares"] == 60 for row in selected)
    rejected = next(row for row in result.signal_decisions if row["symbol"] == "A")
    assert rejected["decision"] == "position_limit_reached"
    assert result.ending_cash_usd == D("18000")
    assert result.ending_equity_usd == D("30000")


def test_progress_callback_reports_each_completed_session() -> None:
    day1 = date(2026, 1, 2)
    day2 = date(2026, 1, 5)
    progress = []

    run_backtest(
        [
            (day1, (bar(day1, "A", signal=analyser()),)),
            (day2, (bar(day2, "A", signal=None),)),
        ],
        strategy(),
        progress_callback=progress.append,
    )

    assert [item.sessions_processed for item in progress] == [1, 2]
    assert [item.valuation_date for item in progress] == [day1, day2]
    assert progress[-1].signal_count == 1
    assert progress[-1].selected_signal_count == 1
    assert progress[-1].open_position_count == 1
    assert progress[-1].closed_trade_count == 0
    assert progress[-1].total_equity_usd == D("30000")


def test_daily_entry_limit_is_separate_from_total_position_limit() -> None:
    day1 = date(2026, 1, 2)
    day2 = date(2026, 1, 5)
    day3 = date(2026, 1, 6)

    def signal_bar(day: date, symbol: str, cik: int) -> Bar:
        return bar(day, symbol, signal=analyser(), cik=cik)

    def holding_bar(day: date, symbol: str, cik: int) -> Bar:
        return bar(day, symbol, signal=None, cik=cik)

    result = run_backtest(
        [
            (day1, tuple(signal_bar(day1, symbol, index) for index, symbol in enumerate("ABCDE", 1))),
            (
                day2,
                (
                    holding_bar(day2, "A", 1),
                    holding_bar(day2, "B", 2),
                    signal_bar(day2, "C", 3),
                    signal_bar(day2, "D", 4),
                    signal_bar(day2, "E", 5),
                ),
            ),
            (
                day3,
                (
                    holding_bar(day3, "A", 1),
                    holding_bar(day3, "B", 2),
                    holding_bar(day3, "C", 3),
                    holding_bar(day3, "D", 4),
                    signal_bar(day3, "E", 5),
                    signal_bar(day3, "F", 6),
                ),
            ),
        ],
        strategy(max_positions=5, max_new_positions_per_day=2),
    )

    selected_by_day = {
        day: sum(
            row["selected"]
            for row in result.signal_decisions
            if row["signal_date"] == day
        )
        for day in (day1, day2, day3)
    }
    assert selected_by_day == {day1: 2, day2: 2, day3: 1}
    assert result.open_trade_count == 5
    assert sum(
        row["decision"] == "daily_entry_limit_reached"
        for row in result.signal_decisions
    ) == 4
    final_rejection = next(
        row for row in result.signal_decisions
        if row["signal_date"] == day3 and row["symbol"] == "F"
    )
    assert final_rejection["decision"] == "position_limit_reached"


def test_prior_ten_session_high_filter_is_strictly_greater_than_limit() -> None:
    day = date(2026, 1, 2)
    accepted = bar(day, "A", signal=analyser(), prior_high="110", cik=1)
    rejected = bar(day, "B", signal=analyser(), prior_high="110.0001", cik=2)

    result = run_backtest([(day, (accepted, rejected))], strategy())

    decisions = {row["symbol"]: row["decision"] for row in result.signal_decisions}
    assert decisions == {"B": "prior_high_limit_exceeded", "A": "selected"}


def test_entry_filter_boundaries_are_inclusive_one_exclusive_five_and_volume_strict() -> None:
    day = date(2026, 1, 2)
    bars = (
        bar(day, "A", signal=analyser(daily_change="1", volume_ratio="1.2001"), cik=1),
        bar(day, "B", signal=analyser(daily_change="5", volume_ratio="2"), cik=2),
        bar(day, "C", signal=analyser(daily_change="2", volume_ratio="1.2"), cik=3),
    )

    result = run_backtest([(day, bars)], strategy())

    assert [row["symbol"] for row in result.signal_decisions] == ["A"]
    assert result.signal_decisions[0]["decision"] == "selected"


def test_low_wins_when_stop_and_take_profit_are_touched_same_day() -> None:
    entry_day = date(2026, 1, 2)
    next_day = date(2026, 1, 5)
    result = run_backtest(
        [
            (entry_day, (bar(entry_day, "A", signal=analyser()),)),
            (
                next_day,
                (bar(next_day, "A", open_="100", low="94", high="111", atr="3"),),
            ),
        ],
        strategy(),
    )

    trade = result.trades[0]
    assert trade["exit_reason"] == "stop_loss_intraday"
    assert trade["exit_fill_price"] == D("95")
    assert trade["net_pnl_usd"] == D("-300")


def test_gap_beyond_stop_executes_at_open() -> None:
    entry_day = date(2026, 1, 2)
    result = run_backtest(
        [
            (entry_day, (bar(entry_day, "A", signal=analyser()),)),
            (
                date(2026, 1, 5),
                (bar(date(2026, 1, 5), "A", open_="90", low="89", high="101"),),
            ),
        ],
        strategy(),
    )

    assert result.trades[0]["exit_reason"] == "stop_loss_gap"
    assert result.trades[0]["exit_fill_price"] == D("90")


@pytest.mark.parametrize(
    ("day1_atr", "day2_atr", "expected_reason", "expected_price"),
    [
        ("1.5", None, "atr_day1", D("101")),
        ("2.5", "2", "atr_day2", D("102")),
    ],
)
def test_atr_exits_at_close_after_price_levels(
    day1_atr: str, day2_atr: str | None, expected_reason: str, expected_price: Decimal
) -> None:
    entry_day = date(2026, 1, 2)
    days: list[tuple[date, tuple[Bar, ...]]] = [
        (entry_day, (bar(entry_day, "A", signal=analyser()),)),
        (
            date(2026, 1, 5),
            (
                bar(
                    date(2026, 1, 5),
                    "A",
                    open_="100",
                    low="96",
                    high="105",
                    close="101",
                    atr=day1_atr,
                ),
            ),
        ),
    ]
    if day2_atr is not None:
        days.append(
            (
                date(2026, 1, 6),
                (
                    bar(
                        date(2026, 1, 6),
                        "A",
                        open_="101",
                        low="97",
                        high="106",
                        close="102",
                        atr=day2_atr,
                    ),
                ),
            )
        )

    trade = run_backtest(days, strategy()).trades[0]
    assert trade["exit_reason"] == expected_reason
    assert trade["exit_fill_price"] == expected_price


def test_levels_step_again_after_each_five_holding_sessions() -> None:
    position = Position(
        trade_id=1,
        symbol="A",
        exchange="NASDAQ",
        cik=1,
        entry_date=date(2026, 1, 2),
        shares=1,
        entry_reference_price=D("100"),
        entry_fill_price=D("100"),
        entry_commission_usd=D("0"),
        risk_budget_usd=D("5"),
        planned_initial_stop_loss_usd=D("5"),
    )
    params = strategy()
    expected = {
        1: (D("95"), D("110")),
        5: (D("95"), D("110")),
        6: (D("100"), D("117.5")),
        10: (D("100"), D("117.5")),
        11: (D("105"), D("125")),
    }
    for holding_sessions, levels in expected.items():
        position.holding_sessions = holding_sessions
        assert active_exit_levels(position, params) == levels


def test_forward_outcome_labels_cannot_change_entry_decision() -> None:
    day = date(2026, 1, 2)
    low_future = run_backtest(
        [(day, (bar(day, "A", signal=analyser(forward_gain="0")),))], strategy()
    )
    high_future = run_backtest(
        [(day, (bar(day, "A", signal=analyser(forward_gain="999")),))], strategy()
    )

    assert low_future.signal_decisions[0]["decision"] == "selected"
    assert high_future.signal_decisions[0]["decision"] == "selected"
    assert low_future.trades[0]["shares"] == high_future.trades[0]["shares"]


def test_cost_aware_sizing_never_exceeds_risk_budget_and_uses_whole_shares() -> None:
    day = date(2026, 1, 2)
    result = run_backtest(
        [(day, (bar(day, "A", signal=analyser()),))],
        strategy(commission_bps=D("10"), slippage_bps=D("10")),
    )

    trade = result.trades[0]
    assert isinstance(trade["shares"], int)
    assert trade["planned_initial_stop_loss_usd"] <= trade["risk_budget_usd"]


def test_open_position_is_marked_not_force_closed_at_data_end() -> None:
    entry_day = date(2026, 1, 2)
    end_day = entry_day + timedelta(days=3)
    result = run_backtest(
        [
            (entry_day, (bar(entry_day, "A", signal=analyser()),)),
            (end_day, (bar(end_day, "A", close="104", high="105", low="96"),)),
        ],
        strategy(),
    )

    trade = result.trades[0]
    assert trade["status"] == "open"
    assert trade["exit_date"] is None
    assert trade["last_mark_price"] == D("104")
    assert result.open_trade_count == 1
