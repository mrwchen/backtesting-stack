"""Event-driven daily simulation for independent and portfolio research modes.

In independent mode there is deliberately NO portfolio logic — no cash
constraint, no position limit, no exposure cap and no compounding. Every stock
whose setup triggers is taken, so the results measure the strategy itself
across the whole universe. In portfolio mode entries compete for cash, gross
exposure and a maximum number of simultaneous positions. Both modes keep the
same per-symbol rule: while a trade in a symbol is open, no second trade is
opened in the same symbol.

Entry:  before each session a deterministic, fully funded stop-buy slate is
        built from information known at the previous close. Quantity and
        portfolio capacity are reserved at the worst permitted buy-zone fill. Only
        orders on that slate may fill; unused reservations are not reassigned
        after observing the day's highs. The market-breadth state produced at
        close t becomes effective for entries in session t+1.
Exits:  initial stop (gap-aware, also checked on the entry bar itself —
        breakout and stop-out on the same day is assumed to resolve against
        us), failed-breakout and time-stop exits, delayed profit protection,
        trailing exit on a close below the TRAIL_MA_DAYS MA (both executed next
        open), forced close at the end of the simulation only when that symbol
        has an executable final-session bar.
Sizing: independent mode uses fixed INITIAL_EQUITY as sizing base. Portfolio
        mode uses previous-close equity, cash, gross exposure and position
        count. Same-session exits never finance or make room for new entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Position:
    position_id: int
    setup_id: int | None
    symbol: str
    col: int
    entry_idx: int
    entry_price: float
    stop: float
    initial_stop: float
    shares: int
    initial_shares: int
    pivot: float
    max_high: float
    next_stop: float | None = None
    realized_position_pnl: float = 0.0
    partial_done: bool = False
    exit_next_open_reason: str | None = None


@dataclass(frozen=True)
class PlannedOrder:
    setup_key: int
    setup: object
    shares: int
    worst_fill: float


@dataclass
class SimResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    metrics: dict = field(default_factory=dict)


def simulate(
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    open_m: pd.DataFrame,
    high_m: pd.DataFrame,
    low_m: pd.DataFrame,
    close_m: pd.DataFrame,
    setups: pd.DataFrame,
    cfg: Config,
    sim_start_idx: int = 0,
    market_on: np.ndarray | None = None,
    regime_entry_allowed: np.ndarray | None = None,
) -> SimResult:
    portfolio_mode = cfg.simulation_mode == "portfolio"
    o = open_m.to_numpy()
    h = high_m.to_numpy()
    lo = low_m.to_numpy()
    c = close_m.to_numpy()
    c_ffill = close_m.ffill().to_numpy()
    ma_trail = close_m.rolling(cfg.trail_ma_days, min_periods=cfg.trail_ma_days).mean().to_numpy()
    col_index = {s: i for i, s in enumerate(symbols)}

    setups = setups[setups["symbol"].isin(col_index)].copy()
    if cfg.bad_fundamentals_filter_enable and {"eps_yoy", "revenue_yoy"}.issubset(setups.columns):
        eps_yoy = pd.to_numeric(setups["eps_yoy"], errors="coerce")
        revenue_yoy = pd.to_numeric(setups["revenue_yoy"], errors="coerce")
        bad_known_growth = (
            eps_yoy.notna()
            & revenue_yoy.notna()
            & (eps_yoy < cfg.eps_yoy_min)
            & (revenue_yoy < cfg.revenue_yoy_min)
        )
        setups = setups.loc[~bad_known_growth].copy()
    if "dryup_ratio" in setups.columns:
        dryup = pd.to_numeric(setups["dryup_ratio"], errors="coerce")
        dryup_ok = dryup.isna() | dryup.between(cfg.dryup_ratio_min, cfg.dryup_ratio_max)
        setups = setups.loc[dryup_ok].copy()
    setups["start_idx"] = dates.searchsorted(pd.to_datetime(setups["detect_date"])) + 1
    setups["end_idx"] = dates.searchsorted(
        pd.to_datetime(setups["valid_until"]), side="right"
    ) - 1
    setups["sim_setup_key"] = np.arange(len(setups), dtype=int)
    setup_sort = ["start_idx", "symbol", "detect_date"]
    if "setup_id" in setups.columns:
        setup_sort.append("setup_id")
    setups = setups.sort_values(setup_sort, na_position="first").reset_index(drop=True)
    setup_rows = list(setups.itertuples(index=False))

    sizing_base = cfg.initial_equity
    cash = cfg.initial_equity
    realized_pnl = 0.0
    positions: dict[str, Position] = {}
    # At most one setup per symbol is active. A newly detected setup replaces
    # an older base before the next session's slate is built.
    active_setups: dict[str, object] = {}
    trades: list[dict] = []
    equity_rows: list[dict] = []
    next_setup = 0
    next_position_id = 1
    exposure_index = 0 if portfolio_mode else len(cfg.exposure_levels) - 1
    consecutive_winners = 0
    consecutive_losses = 0
    peak_equity = cfg.initial_equity
    closed_outcomes: list[float] = []

    def _num_attr(setup, name: str, default: float) -> float:
        value = getattr(setup, name, default)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return default if np.isnan(out) else out

    def _known_bad_growth(setup) -> bool:
        eps_yoy = _num_attr(setup, "eps_yoy", np.nan)
        revenue_yoy = _num_attr(setup, "revenue_yoy", np.nan)
        return (
            np.isfinite(eps_yoy)
            and np.isfinite(revenue_yoy)
            and eps_yoy < cfg.eps_yoy_min
            and revenue_yoy < cfg.revenue_yoy_min
        )

    def _setup_stop_pct(setup) -> float:
        pivot = _num_attr(setup, "pivot", np.nan)
        stop = _num_attr(setup, "stop_level", np.nan)
        if not np.isfinite(pivot) or pivot <= 0 or not np.isfinite(stop):
            return cfg.stop_max_pct
        return max(0.0, (pivot - stop) / pivot)

    def setup_priority(setup) -> tuple:
        dryup_value = _num_attr(setup, "dryup_ratio", cfg.dryup_ratio_preferred)
        eps_yoy = min(_num_attr(setup, "eps_yoy", 0.0), 5.0)
        revenue_yoy = min(_num_attr(setup, "revenue_yoy", 0.0), 5.0)
        return (
            1 if cfg.bad_fundamentals_filter_enable and _known_bad_growth(setup) else 0,
            -_num_attr(setup, "vcp_score", 0.0),
            _setup_stop_pct(setup),
            abs(dryup_value - cfg.dryup_ratio_preferred),
            -_num_attr(setup, "rs_rating", 0.0),
            -_num_attr(setup, "stock_industry_rs_rating", 0.0),
            -_num_attr(setup, "stock_category_rs_rating", 0.0),
            -_num_attr(setup, "ibkr_industry_rs_rating", 0.0),
            -_num_attr(setup, "ibkr_category_rs_rating", 0.0),
            -_num_attr(setup, "n_contractions", 0.0),
            -eps_yoy,
            -revenue_yoy,
            int(getattr(setup, "start_idx", 0)),
            str(getattr(setup, "symbol", "")),
            int(_num_attr(setup, "setup_id", 0.0)),
        )

    def open_notional(t: int) -> float:
        total = 0.0
        for p in positions.values():
            price = c_ffill[t, p.col]
            if np.isnan(price):
                price = p.entry_price
            total += p.shares * price
        return total

    def account_equity(t: int) -> float:
        if not portfolio_mode:
            unrealized = 0.0
            for p in positions.values():
                price = c_ffill[t, p.col]
                if np.isnan(price):
                    continue
                # Independent mode has no cash ledger, so accrue the entry
                # commission on still-open shares explicitly. Closed legs
                # already include their proportional entry cost in realized_pnl.
                unrealized += p.shares * (
                    price - p.entry_price * (1 + cfg.commission_pct)
                )
            return sizing_base + realized_pnl + unrealized
        return cash + open_notional(t)

    def gate_value(values: np.ndarray | None, idx: int) -> bool:
        if values is None:
            return True
        if idx < 0 or idx >= len(values):
            return False
        value = values[idx]
        return bool(value) if not pd.isna(value) else False

    def record_trade(pos: Position, t: int, shares: int, price: float, leg: str, reason: str):
        nonlocal cash, realized_pnl
        proceeds = shares * price * (1 - cfg.commission_pct)
        cost = shares * pos.entry_price * (1 + cfg.commission_pct)
        pnl = proceeds - cost
        if portfolio_mode:
            cash += proceeds
        realized_pnl += pnl
        pos.realized_position_pnl += pnl
        r_unit = pos.entry_price - pos.initial_stop
        trades.append(
            {
                "position_id": pos.position_id,
                "setup_id": pos.setup_id,
                "symbol": pos.symbol,
                "leg": leg,
                "exit_reason": reason,
                "entry_date": dates[pos.entry_idx].date(),
                "entry_price": round(pos.entry_price, 4),
                "stop_price": round(pos.initial_stop, 4),
                "pivot": round(pos.pivot, 4),
                "shares": shares,
                "exit_date": dates[t].date(),
                "exit_price": round(price, 4),
                "pnl": round(pnl, 2),
                "r_multiple": round((price - pos.entry_price) / r_unit, 4) if r_unit > 0 else None,
                "holding_days": int(t - pos.entry_idx),
            }
        )
        if leg == "final":
            closed_outcomes.append(pos.realized_position_pnl)

    def take_partial(pos: Position, t: int, day_open: float, target: float) -> bool:
        """Execute a real partial fill; a rounded zero-share leg changes nothing."""
        part = min(pos.shares, int(pos.shares * cfg.partial_fraction))
        if part < 1:
            return False
        fill = max(day_open, target) * (1 - cfg.slippage_pct)
        record_trade(pos, t, part, fill, "partial", "partial_target")
        pos.shares -= part
        pos.partial_done = True
        if cfg.breakeven_after_partial:
            pos.stop = max(pos.stop, pos.entry_price)
        return True

    for t in range(sim_start_idx, len(dates)):
        # ---- make newly known setups active; newest setup wins -------------
        while next_setup < len(setup_rows) and setup_rows[next_setup].start_idx <= t:
            setup = setup_rows[next_setup]
            active_setups[setup.symbol] = setup
            next_setup += 1
        active_setups = {
            symbol: setup
            for symbol, setup in active_setups.items()
            if setup.end_idx >= t
        }

        # ---- pre-session order slate ---------------------------------------
        # Capture budgets before processing any event from session t. Exits at
        # the open or intraday therefore cannot be recycled into today's slate.
        start_symbols = set(positions)
        if portfolio_mode:
            if t > 0:
                sizing_equity = account_equity(t - 1)
                start_gross = open_notional(t - 1)
            else:
                sizing_equity = sizing_base
                start_gross = 0.0
            remaining_cash = max(0.0, cash)
            remaining_gross = max(
                0.0,
                sizing_equity * cfg.portfolio_max_gross_exposure_pct
                * cfg.exposure_levels[exposure_index] - start_gross,
            )
            scaled_slots = max(
                1,
                int(np.floor(
                    cfg.portfolio_max_open_positions * cfg.exposure_levels[exposure_index]
                )),
            )
            remaining_slots = max(
                0, scaled_slots - len(start_symbols)
            )
        else:
            sizing_equity = sizing_base
            remaining_cash = float("inf")
            remaining_gross = float("inf")
            remaining_slots = len(active_setups)

        # market_on[t] is calculated with close[t], hence only t-1 may govern
        # an intraday stop order in session t. regime_entry_allowed is already
        # defined by its caller as a per-session availability gate.
        entries_allowed = gate_value(market_on, t - 1) and gate_value(
            regime_entry_allowed, t
        )
        slate: list[PlannedOrder] = []
        selected_symbols: set[str] = set()
        unusable_setup_keys: set[int] = set()
        if entries_allowed:
            for setup in sorted(active_setups.values(), key=setup_priority):
                if setup.symbol in start_symbols or setup.symbol in selected_symbols:
                    continue
                if portfolio_mode and remaining_slots <= 0:
                    break

                pivot = _num_attr(setup, "pivot", np.nan)
                stop = _num_attr(setup, "stop_level", np.nan)
                trigger = pivot * (1 + cfg.pivot_buffer_pct)
                worst_fill = trigger * (1 + cfg.max_buy_zone_pct) * (1 + cfg.slippage_pct)
                risk_per_share = worst_fill - stop
                setup_key = int(setup.sim_setup_key)
                if (
                    not np.isfinite(pivot)
                    or pivot <= 0
                    or not np.isfinite(stop)
                    or not np.isfinite(worst_fill)
                    or risk_per_share <= 0
                ):
                    unusable_setup_keys.add(setup_key)
                    continue

                limits = [
                    sizing_equity * cfg.risk_pct
                    * (cfg.exposure_levels[exposure_index] if portfolio_mode else 1.0)
                    / risk_per_share,
                    sizing_equity * cfg.max_position_pct
                    * (cfg.exposure_levels[exposure_index] if portfolio_mode else 1.0)
                    / worst_fill,
                ]
                if portfolio_mode:
                    limits.extend(
                        [
                            remaining_gross / worst_fill,
                            remaining_cash
                            / (worst_fill * (1 + cfg.commission_pct)),
                        ]
                    )
                shares = int(min(limits))
                if shares < 1:
                    continue

                slate.append(
                    PlannedOrder(
                        setup_key=setup_key,
                        setup=setup,
                        shares=shares,
                        worst_fill=worst_fill,
                    )
                )
                selected_symbols.add(setup.symbol)
                if portfolio_mode:
                    remaining_slots -= 1
                    remaining_gross -= shares * worst_fill
                    remaining_cash -= (
                        shares * worst_fill * (1 + cfg.commission_pct)
                    )

        # ---- exits ---------------------------------------------------------
        for sym in list(positions):
            pos = positions[sym]
            col = pos.col
            day_open, day_high, day_low = o[t, col], h[t, col], lo[t, col]
            if np.isnan(day_open):
                continue  # no bar today, hold

            if pos.exit_next_open_reason is not None:
                record_trade(
                    pos, t, pos.shares, day_open * (1 - cfg.slippage_pct),
                    "final", pos.exit_next_open_reason,
                )
                del positions[sym]
                continue

            if pos.next_stop is not None:
                pos.stop = max(pos.stop, pos.next_stop)
                pos.next_stop = None

            if day_open <= pos.stop:
                record_trade(pos, t, pos.shares, day_open * (1 - cfg.slippage_pct), "final", "stop_gap")
                del positions[sym]
                continue

            r_unit = pos.entry_price - pos.initial_stop
            target = pos.entry_price + cfg.partial_at_r * r_unit
            partial_at_open = (
                not pos.partial_done
                and r_unit > 0
                and day_open >= target
                and take_partial(pos, t, day_open, target)
            )
            if partial_at_open:
                if pos.shares <= 0:
                    del positions[sym]
                    continue
                # The opening print is known to precede the day's low. The
                # partial therefore fills at the open before any later stop.
                if day_low <= pos.stop:
                    record_trade(
                        pos,
                        t,
                        pos.shares,
                        pos.stop * (1 - cfg.slippage_pct),
                        "final",
                        "stop_after_partial",
                    )
                    del positions[sym]
                    continue
            else:
                # When the target was not already crossed at the open, daily
                # OHLC does not reveal whether the low or high came first.
                # Resolve the ambiguity adversely: the existing stop wins.
                if day_low <= pos.stop:
                    record_trade(
                        pos,
                        t,
                        pos.shares,
                        pos.stop * (1 - cfg.slippage_pct),
                        "final",
                        "stop",
                    )
                    del positions[sym]
                    continue

                if (
                    not pos.partial_done
                    and r_unit > 0
                    and day_high >= target
                    and take_partial(pos, t, day_open, target)
                ):
                    if pos.shares <= 0:
                        del positions[sym]
                        continue
                    # Target and the newly raised break-even stop both lie in
                    # the bar: assume the adverse target-then-stop path.
                    if day_low <= pos.stop:
                        record_trade(
                            pos,
                            t,
                            pos.shares,
                            pos.stop * (1 - cfg.slippage_pct),
                            "final",
                            "stop_after_partial",
                        )
                        del positions[sym]
                        continue

            day_close = c[t, col]
            if np.isfinite(day_high):
                pos.max_high = max(pos.max_high, float(day_high))
            if (
                r_unit > 0
                and pos.max_high >= pos.entry_price + cfg.profit_protection_trigger_r * r_unit
            ):
                pos.next_stop = pos.entry_price + cfg.profit_protection_lock_r * r_unit
            if (
                r_unit > 0
                and t - pos.entry_idx >= cfg.time_stop_sessions
                and pos.max_high < pos.entry_price + cfg.time_stop_min_r * r_unit
            ):
                pos.exit_next_open_reason = "time_stop"
                continue
            failed_breakout_level = max(
                pos.pivot,
                pos.entry_price + cfg.failed_breakout_min_r * r_unit,
            )
            if (
                cfg.failed_breakout_exit_enable
                and not pos.partial_done
                and r_unit > 0
                and t - pos.entry_idx >= cfg.failed_breakout_days
                and not np.isnan(day_close)
                and day_close <= failed_breakout_level
            ):
                pos.exit_next_open_reason = "failed_breakout"
                continue

            trail = ma_trail[t, col]
            if not np.isnan(trail) and c[t, col] < trail:
                pos.exit_next_open_reason = "trail_ma"

        # ---- fill only orders selected before the session ------------------
        consumed_setup_keys = set(unusable_setup_keys)
        for order in slate:
            setup = order.setup
            col = col_index[setup.symbol]
            day_open, day_high = o[t, col], h[t, col]
            if not np.isfinite([day_open, day_high, lo[t, col], c[t, col]]).all():
                # Stop-buy execution cannot be simulated safely without the
                # complete session range and close. The setup continuity is
                # broken and the order is consumed.
                consumed_setup_keys.add(order.setup_key)
                continue
            pivot = float(setup.pivot)
            trigger = pivot * (1 + cfg.pivot_buffer_pct)
            invalidation_level = max(
                _num_attr(setup, "last_low", -np.inf),
                _num_attr(setup, "stop_level", -np.inf),
            )
            # An opening print at/below the structural invalidation level is
            # known before any later intraday pivot trade and cancels the order.
            if np.isfinite(invalidation_level) and day_open <= invalidation_level:
                consumed_setup_keys.add(order.setup_key)
                continue
            if day_high <= trigger:
                continue
            if day_open > trigger * (1 + cfg.max_buy_zone_pct):
                continue

            fill = max(day_open, trigger) * (1 + cfg.slippage_pct)
            stop = float(setup.stop_level)
            shares = order.shares

            new_position = Position(
                position_id=next_position_id,
                setup_id=getattr(setup, "setup_id", None),
                symbol=setup.symbol,
                col=col,
                entry_idx=t,
                entry_price=fill,
                stop=stop,
                initial_stop=stop,
                shares=shares,
                initial_shares=shares,
                pivot=pivot,
                max_high=fill,
            )
            next_position_id += 1
            consumed_setup_keys.add(order.setup_key)
            if portfolio_mode:
                cash -= shares * fill * (1 + cfg.commission_pct)
            # entry-day stop: if the bar's low also breaches the stop, assume
            # the breakout came first and we were stopped out the same day
            if lo[t, col] <= stop:
                record_trade(
                    new_position, t, shares, stop * (1 - cfg.slippage_pct), "final", "stop_entry_day"
                )
            else:
                # Apply the profit target on the entry bar too. If the same
                # bar also contains the newly raised break-even stop, select
                # the adverse target-then-stop path.
                r_unit = new_position.entry_price - new_position.initial_stop
                target = new_position.entry_price + cfg.partial_at_r * r_unit
                if (
                    r_unit > 0
                    and day_high >= target
                    and take_partial(new_position, t, day_open, target)
                ):
                    if (
                        new_position.shares > 0
                        and lo[t, col] <= new_position.stop
                    ):
                        record_trade(
                            new_position,
                            t,
                            new_position.shares,
                            new_position.stop * (1 - cfg.slippage_pct),
                            "final",
                            "stop_after_partial",
                        )
                        new_position.shares = 0
                if new_position.shares > 0:
                    positions[setup.symbol] = new_position

        # ---- setup lifecycle ----------------------------------------------
        # A breakout without a fill is missed, not a future invitation to
        # chase. A stop-level breach destroys the base. Filled and statically
        # unusable setups are consumed as well.
        for symbol, setup in list(active_setups.items()):
            setup_key = int(setup.sim_setup_key)
            col = col_index[symbol]
            incomplete_bar = not np.isfinite(
                [o[t, col], h[t, col], lo[t, col], c[t, col]]
            ).all()
            day_high = h[t, col]
            day_low = lo[t, col]
            trigger = float(setup.pivot) * (1 + cfg.pivot_buffer_pct)
            missed_breakout = np.isfinite(day_high) and day_high > trigger
            invalidation_level = max(
                _num_attr(setup, "last_low", -np.inf),
                _num_attr(setup, "stop_level", -np.inf),
            )
            damaged_base = (
                np.isfinite(day_low)
                and np.isfinite(invalidation_level)
                and day_low <= invalidation_level
            )
            if (
                setup_key in consumed_setup_keys
                or incomplete_bar
                or missed_breakout
                or damaged_base
            ):
                del active_setups[symbol]

        # ---- daily aggregate (research curve, not a cash-constrained account) --
        day_open_notional = open_notional(t)
        day_equity = account_equity(t)
        exposure_base = day_equity if portfolio_mode else sizing_base
        equity_rows.append(
            {
                "period_end_date": dates[t].date(),
                "equity": round(day_equity, 2),
                "open_positions": len(positions),
                "exposure_pct": round(day_open_notional / exposure_base, 6) if exposure_base > 0 else 0.0,
                "exposure_level": cfg.exposure_levels[exposure_index],
            }
        )

        if portfolio_mode:
            for outcome in closed_outcomes:
                if outcome > 0:
                    consecutive_winners += 1
                    consecutive_losses = 0
                    if consecutive_winners >= cfg.exposure_winners_to_step_up:
                        exposure_index = min(exposure_index + 1, len(cfg.exposure_levels) - 1)
                        consecutive_winners = 0
                else:
                    consecutive_losses += 1
                    consecutive_winners = 0
                    exposure_index = max(0, exposure_index - 1)
                    if consecutive_losses >= cfg.exposure_losses_to_reset:
                        exposure_index = 0
                        consecutive_losses = 0
            closed_outcomes.clear()
            peak_equity = max(peak_equity, day_equity)
            if peak_equity > 0 and day_equity / peak_equity - 1.0 <= -cfg.exposure_drawdown_reset_pct:
                exposure_index = 0
                consecutive_winners = 0
                consecutive_losses = 0

    # ---- close remaining positions only when the final session is tradable ----
    last = len(dates) - 1
    for sym in list(positions):
        pos = positions[sym]
        final_close = c[last, pos.col]
        if np.isnan(final_close):
            continue
        price = final_close * (1 - cfg.slippage_pct)
        record_trade(pos, last, pos.shares, price, "final", "eod")
        del positions[sym]

    if positions:
        log.warning(
            "%d positions remain open at end-of-data because their symbols have no final-session bar",
            len(positions),
        )

    # The last daily row was marked before executable forced liquidations.
    # Replace it with the post-close account state. Positions lacking a final
    # bar remain open and are marked at their last known close; no fill, exit
    # commission or fictional end-date trade is created.
    if equity_rows:
        final_equity = account_equity(last)
        final_notional = open_notional(last)
        exposure_base = final_equity if portfolio_mode else sizing_base
        equity_rows[-1].update(
            {
                "equity": round(final_equity, 2),
                "open_positions": len(positions),
                "exposure_pct": round(final_notional / exposure_base, 6)
                if exposure_base > 0
                else 0.0,
                "exposure_level": cfg.exposure_levels[exposure_index],
            }
        )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    open_position_ids = {position.position_id for position in positions.values()}
    metrics = _metrics(
        trades_df,
        equity_df,
        cfg,
        open_position_ids=open_position_ids,
    )
    # Opened positions without an executable final bar have no exit leg, but
    # they still count as strategy positions in the run summary.
    metrics["num_positions"] = next_position_id - 1
    return SimResult(trades=trades_df, equity=equity_df, metrics=metrics)


def _metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    cfg: Config,
    *,
    open_position_ids: set[int] | None = None,
) -> dict:
    initial = cfg.initial_equity
    if equity.empty:
        return {"initial_equity": initial, "final_equity": initial}
    final = float(equity["equity"].iloc[-1])
    eq = equity["equity"].astype(float)
    years = len(eq) / 252.0
    drawdown = 1.0 - eq / eq.cummax()

    metrics = {
        "initial_equity": initial,
        "final_equity": round(final, 2),
        "total_return": round(final / initial - 1.0, 6),
        "cagr": round((final / initial) ** (1.0 / years) - 1.0, 6) if years > 0 and final > 0 else None,
        "max_drawdown": round(float(drawdown.max()), 6),
        "avg_exposure": round(float(equity["exposure_pct"].astype(float).mean()), 6),
        "num_trade_legs": int(len(trades)),
        "num_positions": 0,
        "win_rate": None,
        "profit_factor": None,
        "avg_r_multiple": None,
    }
    closed_trades = trades
    if open_position_ids and not trades.empty:
        closed_trades = trades.loc[~trades["position_id"].isin(open_position_ids)]
    if closed_trades.empty:
        return metrics

    per_position = closed_trades.groupby("position_id")["pnl"].sum()
    wins = (per_position > 0).sum()
    gross_profit = per_position.loc[per_position > 0].sum()
    gross_loss = per_position.loc[per_position < 0].sum()

    r_rows = closed_trades.dropna(subset=["r_multiple"]).copy()
    position_r = pd.Series(dtype=float)
    if not r_rows.empty:
        r_rows["_weighted_r"] = (
            r_rows["r_multiple"].astype(float) * r_rows["shares"].astype(float)
        )
        all_shares = closed_trades.groupby("position_id")["shares"].sum()
        covered_shares = r_rows.groupby("position_id")["shares"].sum()
        fully_covered = covered_shares.index[
            covered_shares.eq(all_shares.reindex(covered_shares.index))
        ]
        position_r = (
            r_rows.groupby("position_id")["_weighted_r"].sum()
            / covered_shares
        ).reindex(fully_covered)
    metrics.update(
        {
            "num_positions": int(len(per_position)),
            "win_rate": round(float(wins / len(per_position)), 6),
            "profit_factor": round(float(gross_profit / abs(gross_loss)), 6)
            if gross_loss < 0
            else None,
            "avg_r_multiple": round(float(position_r.mean()), 6)
            if not position_r.empty
            else None,
        }
    )
    return metrics
