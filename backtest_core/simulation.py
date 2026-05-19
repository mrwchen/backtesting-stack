"""Core point-in-time portfolio simulation loop."""

import logging
import time as _time
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

import psycopg2

from backtest_shared import Signal, SignalEvaluation
from . import runtime
from .broker import (
    _account_equity,
    _account_snapshot_values,
    _active_maintenance_margin_used,
    _active_margin_used,
    _enforce_account_margin_liquidation,
    _make_trade,
    _margin_level_pct,
    _pnl_long,
    _pnl_short,
    _remove_position_by_identity,
    calc_position,
)
from .config import *
from .entities import AccountCurvePoint, ClosedTrade, DecisionEvent, OpenPosition, PortfolioEvent
from .market_data import (
    _day_close_ts,
    _day_signal_cutoff_ts,
    _ensure_utc_ts,
    get_bars_range,
    get_cached_bars,
    get_candidates,
    get_trading_days,
    get_world_regime,
    _is_stop_loss_active,
    _is_in_sl_tp_window,
    log_cache_stats,
    preload_identity_bars,
)
from .model_loader import get_model_module
from .monte_carlo import run_monte_carlo
from .persistence import (
    create_run,
    update_run_summary,
    write_account_curve,
    write_decision_events,
    write_trades,
)

log = logging.getLogger(__name__)

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
    bars = get_bars_range(conn, pos.identity_key, after_ts, as_of_date)
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
        sl_tp_active = _is_in_sl_tp_window(ts)
        stop_loss_active = _is_stop_loss_active(ts)

        if is_long:
            # SL check first (conservative — if same bar hits both, SL wins)
            if stop_loss_active and low <= effective_sl:
                price = effective_sl
                if tp1_hit:
                    pnl = _pnl_long(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
                    status = "HIT_TP1_THEN_BE"
                else:
                    pnl = _pnl_long(pos, price, price, split_exits=False)
                    status = "HIT_SL"
                return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

            if sl_tp_active and not tp1_hit and high >= pos.take_profit_1:
                tp1_hit = True
                tp1_price = pos.take_profit_1
                tp1_exit_ts = ts
                effective_sl = pos.entry_price  # move SL to breakeven

            if sl_tp_active and tp1_hit and high >= pos.take_profit_2:
                price = pos.take_profit_2
                pnl = _pnl_long(pos, tp1_price, price)
                return _make_trade(conn, pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)

        else:  # SHORT
            if stop_loss_active and high >= effective_sl:
                price = effective_sl
                if tp1_hit:
                    pnl = _pnl_short(pos, tp1_price if tp1_price is not None else pos.take_profit_1, price)
                    status = "HIT_TP1_THEN_BE"
                else:
                    pnl = _pnl_short(pos, price, price, split_exits=False)
                    status = "HIT_SL"
                return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

            if sl_tp_active and not tp1_hit and low <= pos.take_profit_1:
                tp1_hit = True
                tp1_price = pos.take_profit_1
                tp1_exit_ts = ts
                effective_sl = pos.entry_price

            if sl_tp_active and tp1_hit and low <= pos.take_profit_2:
                price = pos.take_profit_2
                pnl = _pnl_short(pos, tp1_price, price)
                return _make_trade(conn, pos, "HIT_TP2", price, bar_date, total_bars, True, pnl, equity, ts, tp1_exit_ts)

        if ts >= pos.valid_until:
            price = float(close)
            if is_long:
                pnl = _pnl_long(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            else:
                pnl = _pnl_short(pos, tp1_price if tp1_hit else price, price, split_exits=tp1_hit)
            status = "MAX_HOLD_TP1" if tp1_hit else "MAX_HOLD"
            return _make_trade(conn, pos, status, price, bar_date, total_bars, tp1_hit, pnl, equity, ts, tp1_exit_ts)

    # Still open — persist incremental state for the next day's call
    pos.tp1_hit = tp1_hit
    pos.tp1_price = tp1_price
    pos.tp1_exit_ts = tp1_exit_ts
    pos.effective_sl = effective_sl
    pos.last_bar_ts = bars[-1][0]
    pos.bars_processed += len(bars)
    return None

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
    account_curve: list[AccountCurvePoint] = []
    account_curve_seq = 0

    def record_account_curve(as_of_ts: datetime, active_positions: list[OpenPosition]) -> None:
        nonlocal account_curve_seq
        account_curve_seq += 1
        as_of_ts = _ensure_utc_ts(as_of_ts)
        snapshot = _account_snapshot_values(
            conn,
            active_positions,
            equity,
            as_of_ts,
        )
        account_curve.append(AccountCurvePoint(
            run_id=run_id,
            ts=as_of_ts,
            trade_date=as_of_ts.date(),
            seq_in_run=account_curve_seq,
            balance_usd=round(equity, 2),
            open_pnl_usd=round(snapshot.open_pnl, 2),
            equity_usd=round(snapshot.equity_with_loan_value, 2),
            initial_margin_usd=round(snapshot.initial_margin, 2),
            maintenance_margin_usd=round(snapshot.maintenance_margin, 2),
            available_funds_usd=round(snapshot.available_funds, 2),
            excess_liquidity_usd=round(snapshot.excess_liquidity, 2),
            open_positions=len(active_positions),
            realized_pnl_usd=round(equity - INITIAL_EQUITY, 2),
            closed_trades=len(closed_trades),
        ))

    trading_days = get_trading_days(conn, START_DATE, END_DATE)
    log.info("Trading days to simulate: %d (%s → %s)", len(trading_days), START_DATE, END_DATE)
    record_account_curve(datetime.combine(START_DATE, datetime.min.time(), tzinfo=timezone.utc), open_positions)

    # Diagnostic counters
    days_no_regime = 0
    days_neutral   = 0
    days_no_candidates = 0
    days_no_signals    = 0
    days_with_signals  = 0

    for day_idx, day in enumerate(trading_days, start=1):
        log_progress_today = day_idx == 1 or day_idx == len(trading_days) or day_idx % PROGRESS_LOG_EVERY_DAYS == 0
        if log_progress_today:
            log.info(
                "Day %d/%d %s starting model %s open positions %d closed trades %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                len(open_positions),
                len(closed_trades),
            )
            log_cache_stats(f"day_start {day_idx}/{len(trading_days)} {day}")

        # ── 1. Apply open-position state changes for today ──────────────────
        portfolio_events: list[PortfolioEvent] = []
        closed_today = 0
        day_pnl = 0.0
        for pos in open_positions:
            before_tp1_hit = pos.tp1_hit
            before_tp1_price = pos.tp1_price
            before_tp1_exit_ts = pos.tp1_exit_ts
            before_effective_sl = pos.effective_sl
            trade = simulate_outcome(conn, pos, day, equity)
            if trade is not None:
                close_ts = trade.exit_ts or _day_close_ts(trade.outcome_date)
                if not before_tp1_hit and trade.tp1_hit and trade.tp1_exit_ts and trade.tp1_exit_ts < close_ts:
                    portfolio_events.append(PortfolioEvent(
                        ts=trade.tp1_exit_ts,
                        priority=0,
                        kind="tp1",
                        position=pos,
                    ))
                portfolio_events.append(PortfolioEvent(
                    ts=close_ts,
                    priority=1,
                    kind="close",
                    position=pos,
                    trade=trade,
                ))
                continue

            if not before_tp1_hit and pos.tp1_hit:
                tp1_event_ts = pos.tp1_exit_ts or _day_close_ts(day)
                pos.tp1_hit = before_tp1_hit
                pos.tp1_price = before_tp1_price
                pos.tp1_exit_ts = before_tp1_exit_ts
                pos.effective_sl = before_effective_sl
                portfolio_events.append(PortfolioEvent(
                    ts=tp1_event_ts,
                    priority=0,
                    kind="tp1",
                    position=pos,
                ))

        active_positions = list(open_positions)
        for event in sorted(portfolio_events, key=lambda e: (_ensure_utc_ts(e.ts), e.priority)):
            if event.kind == "tp1":
                event.position.tp1_hit = True
                event.position.tp1_price = event.position.take_profit_1
                event.position.tp1_exit_ts = _ensure_utc_ts(event.ts)
                event.position.effective_sl = event.position.entry_price
                record_account_curve(event.ts, active_positions)
                continue

            if event.kind == "close" and event.trade is not None:
                _remove_position_by_identity(active_positions, event.position)
                event.trade.equity_after = round(equity + event.trade.pnl_usd, 2)
                equity = event.trade.equity_after
                closed_trades.append(event.trade)
                closed_today += 1
                day_pnl += event.trade.pnl_usd
                log.debug("Closed %-6s %s %s pnl %.0f balance %.0f",
                          event.position.symbol, event.position.direction, event.trade.outcome_status,
                          event.trade.pnl_usd, equity)
                record_account_curve(event.ts, active_positions)
        open_positions = active_positions
        stop_out_ts = _day_close_ts(day)
        liquidation_trades, equity = _enforce_account_margin_liquidation(
            conn,
            open_positions,
            equity,
            stop_out_ts,
        )
        if liquidation_trades:
            closed_trades.extend(liquidation_trades)
            closed_today += len(liquidation_trades)
            day_pnl += sum(t.pnl_usd for t in liquidation_trades)
            record_account_curve(stop_out_ts, open_positions)

        # ── 2. Generate signals for today ───────────────────────────────────
        day_end_ts = _day_signal_cutoff_ts(day)
        regime = get_world_regime(conn, source_table=SOURCE_WORLD_REGIME, as_of_date=day)
        if not regime:
            days_no_regime += 1
            write_decision_events(conn, [DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="no_regime",
                reason_text="No world-regime row was available for this trading day.",
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s no regime, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        if regime.score < cfg.long_max_score:
            direction = "LONG"
        elif regime.score >= cfg.short_min_score:
            direction = "SHORT"
        else:
            days_neutral += 1
            write_decision_events(conn, [DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=None,
                decision_stage="regime_filter",
                decision="skipped_day",
                reason_code="neutral_regime",
                reason_text=(
                    f"World-regime score {regime.score:.2f} is between long threshold "
                    f"{cfg.long_max_score:.2f} and short threshold {cfg.short_min_score:.2f}."
                ),
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s neutral regime %.1f, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, regime.score, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        if log_progress_today:
            log.info(
                "Candidate query starting day %d/%d %s model %s direction %s cutoff %s",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                direction,
                day_end_ts,
            )
        candidate_started = _time.perf_counter()
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
            pepperstone_table=PEPPERSTONE_TABLE,
            required_currency="USD" if REQUIRE_USD_FUNDAMENTALS else None,
            allow_rebuilt_historical_fundamentals=ALLOW_REBUILT_HISTORICAL_FUNDAMENTALS,
            filter_negative_earnings=(
                FILTER_NEGATIVE_EARNINGS_LONG if direction == "LONG" else FILTER_NEGATIVE_EARNINGS_SHORT
            ),
            ibkr_margin_table=IBKR_MARGIN_REQUIREMENTS_TABLE,
        )
        candidate_elapsed = _time.perf_counter() - candidate_started
        if log_progress_today or candidate_elapsed >= 5.0:
            log.info(
                "Candidate query complete day %d/%d %s model %s direction %s found %d candidates in %.1f s",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                direction,
                len(candidates),
                candidate_elapsed,
            )

        if not candidates:
            days_no_candidates += 1
            write_decision_events(conn, [DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=None,
                exchange=None,
                cik=None,
                direction=direction,
                decision_stage="candidate_filter",
                decision="no_candidates",
                reason_code="no_candidates_after_fundamental_filters",
                reason_text="No symbols passed the point-in-time fundamental, currency, market-cap and broker filters.",
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )])
            if log_progress_today:
                log.info(
                    "Progress %d/%d %s model %s %s regime %.1f, no candidates, day pnl %.0f, equity %.0f, open %d, closed today %d, closed total %d",
                    day_idx, len(trading_days), day, runtime.CURRENT_MODEL_FILE, direction, regime.score, day_pnl, equity, len(open_positions), closed_today, len(closed_trades),
                )
            continue

        candidate_identities = [fundamental.identity_key for fundamental in candidates]
        preload_started = _time.perf_counter()
        loaded_bar_rows = preload_identity_bars(
            conn,
            candidate_identities,
            day_end_ts,
            batch_size=BAR_CACHE_BATCH_SIZE,
            log_batches=log_progress_today,
        )
        preload_elapsed = _time.perf_counter() - preload_started
        if log_progress_today or preload_elapsed >= 5.0:
            log.info(
                "Bar preload complete day %d/%d %s model %s loaded %d new rows for %d candidates through %s in %.1f s",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
                loaded_bar_rows,
                len(candidate_identities),
                day_end_ts,
                preload_elapsed,
            )

        model = get_model_module()
        compute_fn = model.compute_long_signal if direction == "LONG" else model.compute_short_signal
        evaluate_fn = getattr(
            model,
            "evaluate_long_signal" if direction == "LONG" else "evaluate_short_signal",
            None,
        )
        signals: list[Signal] = []
        decision_events: list[DecisionEvent] = []
        signal_events: dict[tuple[str, str, int], DecisionEvent] = {}
        skipped_no_bars = 0

        for candidate_rank, fundamental in enumerate(candidates, start=1):
            bars = get_cached_bars(
                conn, fundamental.identity_key,
                cfg.min_bars + cfg.price_lookback_bars,
                up_to_ts=day_end_ts,
            )
            if len(bars) < cfg.min_bars:
                skipped_no_bars += 1
                decision_events.append(DecisionEvent(
                    run_id=run_id,
                    signal_date=day,
                    as_of_ts=day_end_ts,
                    symbol=fundamental.symbol,
                    exchange=fundamental.exchange,
                    cik=fundamental.cik,
                    direction=direction,
                    decision_stage="bar_load",
                    decision="rejected",
                    reason_code="insufficient_bars",
                    reason_text=f"Only {len(bars)} cached 1h bars were available; model requires at least {cfg.min_bars}.",
                    candidate_rank=candidate_rank,
                    world_regime_label=regime.label,
                    world_regime_score=regime.score,
                    valuation_label=fundamental.valuation_label,
                    sector=fundamental.sector,
                    industry=fundamental.industry,
                    fundamental_score=fundamental.composite_score,
                    mispricing_score=fundamental.mispricing_score,
                    market_cap_m=fundamental.market_cap_m,
                    bar_count=len(bars),
                    min_bars=cfg.min_bars,
                    open_positions=len(open_positions),
                    max_open_positions=MAX_OPEN_POSITIONS,
                    account_equity=equity,
                ))
                continue
            if evaluate_fn is not None:
                evaluation = evaluate_fn(
                    bars,
                    fundamental,
                    datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                    cfg,
                )
                signal = evaluation.signal
            else:
                signal = compute_fn(
                    bars,
                    fundamental,
                    datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
                    cfg,
                )
                evaluation = SignalEvaluation(
                    signal=signal,
                    decision="signal" if signal else "rejected",
                    reason_code="signal_passed" if signal else "no_signal",
                    reason_text=signal.entry_reason if signal else "Model returned no signal without a detailed reason.",
                )
            if signal:
                signal.exchange = fundamental.exchange
                signal.cik = fundamental.cik
                signal.entry_ts = bars[-1].ts
                signals.append(signal)
            event = DecisionEvent(
                run_id=run_id,
                signal_date=day,
                as_of_ts=day_end_ts,
                symbol=fundamental.symbol,
                exchange=fundamental.exchange,
                cik=fundamental.cik,
                direction=direction,
                decision_stage="signal_eval",
                decision="signal" if signal else "rejected",
                reason_code=evaluation.reason_code,
                reason_text=evaluation.reason_text,
                signal_passed=bool(signal),
                candidate_rank=candidate_rank,
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                valuation_label=fundamental.valuation_label,
                sector=fundamental.sector,
                industry=fundamental.industry,
                fundamental_score=fundamental.composite_score,
                mispricing_score=fundamental.mispricing_score,
                market_cap_m=fundamental.market_cap_m,
                bar_count=len(bars),
                min_bars=cfg.min_bars,
                entry_ts=signal.entry_ts if signal else None,
                entry_price=evaluation.entry_price if evaluation.entry_price is not None else (signal.entry_price if signal else None),
                stop_loss=evaluation.stop_loss if evaluation.stop_loss is not None else (signal.stop_loss if signal else None),
                take_profit_1=evaluation.take_profit_1 if evaluation.take_profit_1 is not None else (signal.take_profit_1 if signal else None),
                take_profit_2=evaluation.take_profit_2 if evaluation.take_profit_2 is not None else (signal.take_profit_2 if signal else None),
                pullback_pct=evaluation.pullback_pct if evaluation.pullback_pct is not None else (signal.pullback_pct if signal else None),
                rsi_1h=evaluation.rsi_1h if evaluation.rsi_1h is not None else (signal.rsi_1h if signal else None),
                volume_ratio=evaluation.volume_ratio if evaluation.volume_ratio is not None else (signal.volume_ratio if signal else None),
                entry_score=evaluation.entry_score if evaluation.entry_score is not None else (signal.entry_score if signal else None),
                combined_score=evaluation.combined_score if evaluation.combined_score is not None else (signal.combined_score if signal else None),
                open_positions=len(open_positions),
                max_open_positions=MAX_OPEN_POSITIONS,
                account_equity=equity,
            )
            decision_events.append(event)
            if signal:
                signal_events[signal.identity_key] = event

        signals.sort(key=lambda s: s.combined_score, reverse=True)
        for signal_rank, signal in enumerate(signals, start=1):
            event = signal_events.get(signal.identity_key)
            if event:
                event.signal_rank = signal_rank

        if signals:
            days_with_signals += 1
        else:
            days_no_signals += 1

        # ── 3. Open new positions ────────────────────────────────────────────
        open_identities = {p.identity_key for p in open_positions}
        if SECTOR_DIVERSIFICATION_ENABLED:
            open_sectors: set[str] = {p.signal.sector for p in open_positions if p.signal.sector}
            open_sector_industries: set[tuple[str, str]] = {
                (p.signal.sector, p.signal.industry)
                for p in open_positions
                if p.signal.sector
            }

            def _sector_tier(s: Signal) -> int:
                if not s.sector or s.sector not in open_sectors:
                    return 0  # new sector preferred
                if (s.sector, s.industry) not in open_sector_industries:
                    return 1  # same sector, different industry
                return 2      # same sector and industry

            signals.sort(key=lambda s: (_sector_tier(s), -s.combined_score))
            for signal_rank, signal in enumerate(signals, start=1):
                event = signal_events.get(signal.identity_key)
                if event:
                    event.signal_rank = signal_rank
        opened_today = 0
        account_equity_today = _account_equity(conn, open_positions, equity, day)
        initial_margin = sum(_active_margin_used(p) for p in open_positions)
        maintenance_margin = sum(_active_maintenance_margin_used(p) for p in open_positions)

        for signal in signals:
            event = signal_events.get(signal.identity_key)
            available_funds = account_equity_today - initial_margin
            excess_liquidity = account_equity_today - maintenance_margin
            if event:
                event.open_positions = len(open_positions)
                event.max_open_positions = MAX_OPEN_POSITIONS
                event.account_equity = account_equity_today
                event.initial_margin = initial_margin
                event.maintenance_margin = maintenance_margin
                event.available_funds = available_funds
                event.excess_liquidity = excess_liquidity

            if len(open_positions) >= MAX_OPEN_POSITIONS:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "max_open_positions_reached"
                    event.reason_text = f"Maximum open positions {MAX_OPEN_POSITIONS} was already reached."
                continue
            if signal.identity_key in open_identities:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "instrument_already_open"
                    event.reason_text = "Instrument already had an open position."
                continue

            if account_equity_today <= 0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "account_equity_non_positive"
                    event.reason_text = "Account equity was not positive at decision time."
                continue

            initial_margin_used, maintenance_margin_used, shares, position_size_usd = calc_position(conn, signal, account_equity_today)
            if event:
                event.required_initial_margin = initial_margin_used
                event.required_maintenance_margin = maintenance_margin_used
                event.position_size_usd = position_size_usd
                event.shares = shares
            if position_size_usd <= 0:
                if event:
                    event.decision_stage = "portfolio_filter"
                    event.decision = "blocked"
                    event.reason_code = "position_size_non_positive"
                    event.reason_text = "Position sizing produced a non-positive position size."
                continue

            initial_margin_after = initial_margin + initial_margin_used
            maintenance_margin_after = maintenance_margin + maintenance_margin_used
            available_funds_after = account_equity_today - initial_margin_after
            excess_liquidity_after = account_equity_today - maintenance_margin_after
            if event:
                event.available_funds_after = available_funds_after
                event.excess_liquidity_after = excess_liquidity_after

            if ACCOUNT_PROFILE == "ps_acc":
                margin_level_after = _margin_level_pct(account_equity_today, initial_margin_after)
                if margin_level_after <= PS_MARGIN_STOP_OUT_LEVEL_PCT:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "margin_level_stop_out_guard"
                        event.reason_text = (
                            f"Margin level after entry would be {margin_level_after:.2f}%, "
                            f"at or below Pepperstone stop-out level {PS_MARGIN_STOP_OUT_LEVEL_PCT:.2f}%."
                        )
                    continue
                if margin_level_after < PS_MIN_ENTRY_MARGIN_LEVEL_PCT:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "margin_level_entry_guard"
                        event.reason_text = (
                            f"Margin level after entry would be {margin_level_after:.2f}%, "
                            f"below configured Pepperstone backtest minimum {PS_MIN_ENTRY_MARGIN_LEVEL_PCT:.2f}%."
                        )
                    continue
            elif ACCOUNT_PROFILE == "ibkr_acc":
                if available_funds_after < 0:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "available_funds_insufficient"
                        event.reason_text = (
                            f"Available Funds after entry would be {available_funds_after:.2f}, "
                            "below zero."
                        )
                    continue
                if excess_liquidity_after <= 0:
                    if event:
                        event.decision_stage = "portfolio_filter"
                        event.decision = "blocked"
                        event.reason_code = "excess_liquidity_non_positive_guard"
                        event.reason_text = (
                            f"Excess Liquidity after entry would be {excess_liquidity_after:.2f}, "
                            "at or below zero."
                        )
                    continue

            open_positions.append(OpenPosition(
                symbol=signal.symbol,
                exchange=signal.exchange,
                cik=signal.cik,
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
                margin_used=initial_margin_used,
                maintenance_margin_used=maintenance_margin_used,
                equity_before=account_equity_today,
                signal=signal,
                world_regime_label=regime.label,
                world_regime_score=regime.score,
                valuation_label=signal.valuation_label,
            ))
            open_identities.add(signal.identity_key)
            initial_margin += initial_margin_used
            maintenance_margin += maintenance_margin_used
            opened_today += 1
            record_account_curve(signal.entry_ts or day_end_ts, open_positions)
            if event:
                event.decision_stage = "order_open"
                event.decision = "opened"
                event.reason_code = "opened"
                event.reason_text = "Signal passed portfolio checks and a simulated position was opened."
                event.opened = True
            log.debug("Opened %-6s %s entry %.2f stop %.2f margin %.0f equity %.0f",
                      signal.symbol, signal.direction, signal.entry_price,
                      signal.stop_loss, initial_margin_used, equity)

        write_decision_events(conn, decision_events)

        if log_progress_today or opened_today > 0:
            log.info(
                "Progress %d/%d %s model %s %s regime %.1f, candidates %d, signals %d, skipped no bars %d, opened %d, closed today %d, day pnl %.0f, open %d, equity %.0f, closed total %d",
                day_idx,
                len(trading_days),
                day,
                runtime.CURRENT_MODEL_FILE,
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
        "Day breakdown no regime %d, neutral %d, no candidates %d, no signals %d, with signals %d",
        days_no_regime, days_neutral, days_no_candidates, days_no_signals, days_with_signals,
    )
    log_cache_stats("before_force_close")

    # ── 4. Force-close remaining open positions at last available price ──────
    last_day = trading_days[-1] if trading_days else END_DATE
    for pos in list(open_positions):
        bars = get_bars_range(conn, pos.identity_key, pos.entry_ts, last_day)
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
            conn,
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
        _remove_position_by_identity(open_positions, pos)
        trade.equity_after = round(equity + trade.pnl_usd, 2)
        equity = trade.equity_after
        closed_trades.append(trade)
        record_account_curve(trade.exit_ts or _day_close_ts(last_day), open_positions)

    # ── 5. Persist results ───────────────────────────────────────────────────
    log.info("Writing %d trades and %d account snapshots for run %d", len(closed_trades), len(account_curve), run_id)

    # Patch world_regime_label into rows (stored on signal, pass through)
    # (already embedded in entry_reason; trade write accesses signal directly)
    write_account_curve(conn, run_id, account_curve)
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
        "Run %d complete trades %d wins %d final equity %.0f return %.1f%%",
        run_id, n_trades, n_wins, equity, total_return,
    )
    return run_id, summary
