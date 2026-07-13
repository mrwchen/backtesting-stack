"""Event-driven daily simulation for first-touch and portfolio research modes.

In independent mode there is deliberately NO interaction between setups: no
cash constraint, position limit, exposure cap, compounding or same-symbol
blocking. Every causal first touch is its own research position. In portfolio
mode entries compete for cash, gross exposure and a maximum number of
simultaneous positions, with at most one open position per symbol.

Entry:  before each session a deterministic, fully funded stop-buy slate is
        built from information known at the previous close. Portfolio orders
        are scaled together from their standalone target sizes and must retain
        a configured minimum share of that target. Quantity and capacity are
        reserved at the worst permitted buy-zone fill. Only orders on that
        slate may fill; unused reservations are not reassigned after observing
        the day's highs. The market exposure cap produced at close t becomes
        effective for entries in session t+1.
Exits:  initial stop (gap-aware, also checked on the entry bar itself —
        breakout and stop-out on the same day is assumed to resolve against
        us), failed-breakout and time-stop exits, delayed profit protection,
        trailing exit on a close below the TRAIL_MA_DAYS MA (both executed next
        open), forced close at the end of the simulation only when that symbol
        has an executable final-session bar.
Sizing: independent mode uses fixed INITIAL_EQUITY as sizing base. Portfolio
        mode uses previous-close equity, cash, gross exposure and position
        count. Same-session exits never finance or make room for new entries.

Add-ons are intentionally outside this engine. ``Position`` represents one
entry price, one initial risk unit and one setup id; pyramiding would require a
lot-aware position/trade schema rather than silently averaging unlike entries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .candidate_ranking import CandidateRanker, CandidateSnapshot
from .config import Config

log = logging.getLogger(__name__)


@dataclass
class Position:
    position_id: int
    setup_id: int | None
    setup_type: str
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
    feedback_risk_units: float
    next_stop: float | None = None
    realized_position_pnl: float = 0.0
    partial_done: bool = False
    exit_next_open_reason: str | None = None


@dataclass(frozen=True)
class PlannedOrder:
    setup_key: int
    setup: object
    shares: int
    standalone_shares: int
    worst_fill: float
    snapshot: CandidateSnapshot

    @property
    def feedback_risk_units(self) -> float:
        return self.shares / self.standalone_shares


@dataclass(frozen=True)
class SlateCandidate:
    setup_key: int
    setup: object
    standalone_shares: int
    worst_fill: float
    snapshot: CandidateSnapshot


@dataclass
class SimResult:
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity: pd.DataFrame = field(default_factory=pd.DataFrame)
    breakout_events: pd.DataFrame = field(default_factory=pd.DataFrame)
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
    state_start_idx: int | None = None,
    market_exposure_cap: np.ndarray | None = None,
    regime_entry_allowed: np.ndarray | None = None,
    volume_m: pd.DataFrame | None = None,
    candidate_context: dict[str, pd.DataFrame] | None = None,
) -> SimResult:
    state_start_idx = sim_start_idx if state_start_idx is None else state_start_idx
    if not 0 <= state_start_idx <= sim_start_idx < len(dates):
        raise ValueError("state_start_idx and sim_start_idx are outside the date range")
    if cfg.simulation_mode not in ("independent", "portfolio"):
        raise ValueError("simulate requires independent or portfolio mode")
    portfolio_mode = cfg.simulation_mode == "portfolio"
    o = open_m.to_numpy()
    h = high_m.to_numpy()
    lo = low_m.to_numpy()
    c = close_m.to_numpy()
    c_ffill = close_m.ffill().to_numpy()
    ma_trail = close_m.rolling(cfg.trail_ma_days, min_periods=cfg.trail_ma_days).mean().to_numpy()
    col_index = {s: i for i, s in enumerate(symbols)}
    context = dict(candidate_context or {})
    rs_matrix = context.pop("rs_rating", None)
    ranker = CandidateRanker(
        dates,
        symbols,
        close_m,
        volume=volume_m,
        rs_rating=rs_matrix,
        context=context,
        dryup_zero_ratio=cfg.dryup_score_zero_ratio,
    )

    setups = setups[setups["symbol"].isin(col_index)].copy()
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
    positions: dict[object, Position] = {}
    # Multiple causal structures may coexist for one symbol. Portfolio mode
    # nominates at most one per symbol; independent mode keeps every causal
    # first-touch path separate.
    active_setups: dict[int, object] = {}
    trades: list[dict] = []
    breakout_events: list[dict] = []
    equity_rows: list[dict] = []
    entry_decisions: dict[int, dict] = {}
    next_setup = 0
    next_position_id = 1
    exposure_index = 0 if portfolio_mode else len(cfg.exposure_levels) - 1
    consecutive_winners = 0.0
    consecutive_losses = 0.0
    peak_equity = cfg.initial_equity
    drawdown_reset_armed = True
    closed_outcomes: list[tuple[int, float, float]] = []
    feedback_winner_positions: set[int] = set()
    current_snapshots: dict[int, CandidateSnapshot] = {}
    current_candidate_ranks: dict[int, int] = {}

    def _num_attr(setup, name: str, default: float) -> float:
        value = getattr(setup, name, default)
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
        return default if np.isnan(out) else out

    def _known_bad_growth(snapshot: CandidateSnapshot) -> bool:
        eps_yoy = snapshot.eps_yoy
        revenue_yoy = snapshot.revenue_yoy
        return (
            eps_yoy is not None
            and revenue_yoy is not None
            and eps_yoy < cfg.eps_yoy_min
            and revenue_yoy < cfg.revenue_yoy_min
        )

    def allocate_portfolio_slate(
        candidates: list[SlateCandidate],
        gross_budget: float,
        cash_budget: float,
    ) -> list[PlannedOrder] | None:
        """Scale and integer-round a causal pre-session slate as one unit.

        Every candidate first has a standalone target that uses only prior-close
        equity and its own risk/position limits. The integer minimum allocation
        is funded first, then one common scale distributes the remaining gross
        and cash budgets across all residual targets. Integer remainders are
        assigned deterministically without exceeding either budget. ``None``
        means the tentative lower-ranked candidate cannot retain the configured
        minimum target utilization.
        """
        if not candidates:
            return []

        gross_tolerance = 1e-9 * max(1.0, gross_budget)
        cash_tolerance = 1e-9 * max(1.0, cash_budget)
        minimum_shares = [
            int(
                np.ceil(
                    cfg.min_slate_risk_utilization
                    * candidate.standalone_shares
                    - 1e-12
                )
            )
            for candidate in candidates
        ]
        minimum_gross = sum(
            shares * candidate.worst_fill
            for shares, candidate in zip(minimum_shares, candidates)
        )
        minimum_cash = sum(
            shares * candidate.worst_fill * (1 + cfg.commission_pct)
            for shares, candidate in zip(minimum_shares, candidates)
        )
        if (
            minimum_gross > gross_budget + gross_tolerance
            or minimum_cash > cash_budget + cash_tolerance
        ):
            return None

        residual_targets = [
            candidate.standalone_shares - minimum
            for candidate, minimum in zip(candidates, minimum_shares)
        ]
        residual_target_gross = sum(
            shares * candidate.worst_fill
            for shares, candidate in zip(residual_targets, candidates)
        )
        residual_target_cash = sum(
            shares * candidate.worst_fill * (1 + cfg.commission_pct)
            for shares, candidate in zip(residual_targets, candidates)
        )
        scale_limits = [1.0]
        if residual_target_gross > 0:
            scale_limits.append(
                max(0.0, gross_budget - minimum_gross)
                / residual_target_gross
            )
        if residual_target_cash > 0:
            scale_limits.append(
                max(0.0, cash_budget - minimum_cash)
                / residual_target_cash
            )
        residual_scale = min(scale_limits)
        exact_shares = [
            minimum + residual * residual_scale
            for minimum, residual in zip(minimum_shares, residual_targets)
        ]
        allocated = [int(np.floor(value + 1e-12)) for value in exact_shares]
        used_gross = sum(
            shares * candidate.worst_fill
            for shares, candidate in zip(allocated, candidates)
        )
        used_cash = sum(
            shares * candidate.worst_fill * (1 + cfg.commission_pct)
            for shares, candidate in zip(allocated, candidates)
        )

        def add_one_share(index: int) -> bool:
            nonlocal used_gross, used_cash
            candidate = candidates[index]
            gross_cost = candidate.worst_fill
            cash_cost = gross_cost * (1 + cfg.commission_pct)
            if (
                used_gross + gross_cost > gross_budget + gross_tolerance
                or used_cash + cash_cost > cash_budget + cash_tolerance
            ):
                return False
            allocated[index] += 1
            used_gross += gross_cost
            used_cash += cash_cost
            return True

        rounding_caps = [
            min(
                candidate.standalone_shares,
                int(np.ceil(exact - 1e-12)),
            )
            for candidate, exact in zip(candidates, exact_shares)
        ]

        # Largest-remainder rounding keeps the final slate close to the common
        # residual scale. Stable candidate order breaks exact ties.
        remainder_order = sorted(
            range(len(candidates)),
            key=lambda index: (
                -(exact_shares[index] - np.floor(exact_shares[index])),
                index,
            ),
        )
        for index in remainder_order:
            if allocated[index] >= rounding_caps[index]:
                continue
            add_one_share(index)

        if any(
            shares < 1
            or shares / candidate.standalone_shares + 1e-12
            < cfg.min_slate_risk_utilization
            for shares, candidate in zip(allocated, candidates)
        ):
            return None

        return [
            PlannedOrder(
                setup_key=candidate.setup_key,
                setup=candidate.setup,
                shares=shares,
                standalone_shares=candidate.standalone_shares,
                worst_fill=candidate.worst_fill,
                snapshot=candidate.snapshot,
            )
            for shares, candidate in zip(allocated, candidates)
        ]

    def set_entry_decision(
        setup,
        decision: str,
        *,
        entry_date=None,
        entry_price: float | None = None,
    ) -> None:
        setup_id = getattr(setup, "setup_id", None)
        if setup_id is None or pd.isna(setup_id):
            return
        setup_key = int(getattr(setup, "sim_setup_key"))
        snapshot = current_snapshots.get(setup_key)
        if snapshot is None:
            raise RuntimeError("entry decision has no causal candidate snapshot")
        entry_decisions[int(setup_id)] = {
            "setup_id": int(setup_id),
            "entry_decision": decision,
            "entry_date": entry_date,
            "entry_price": entry_price,
            "snapshot_date": snapshot.information_date.date(),
            "dynamic_setup_score": round(snapshot.ranking_score, 4),
            "readiness_score": (
                round(snapshot.readiness_score, 4)
                if snapshot.readiness_score is not None
                else None
            ),
            "context_score": round(snapshot.context_score, 4),
            "setup_age_sessions": snapshot.setup_age_sessions,
            "distance_to_pivot_pct": (
                round(snapshot.distance_to_pivot_pct, 6)
                if snapshot.distance_to_pivot_pct is not None
                else None
            ),
            "candidate_rank": current_candidate_ranks.get(setup_key),
        }

    def record_breakout_event(setup, t: int, trigger: float) -> None:
        setup_id = getattr(setup, "setup_id", None)
        if setup_id is None or pd.isna(setup_id):
            raise RuntimeError("breakout setup has no setup_id")
        decision_row = entry_decisions.get(int(setup_id))
        if decision_row is None:
            raise RuntimeError("breakout setup has no pre-session entry decision")
        decision = decision_row["entry_decision"]
        filled = decision == "filled"
        breakout_events.append(
            {
                "setup_id": setup_id,
                "setup_type": getattr(setup, "setup_type", "unknown"),
                "symbol": setup.symbol,
                "setup_detect_date": pd.Timestamp(setup.detect_date).date(),
                "snapshot_date": decision_row["snapshot_date"],
                "dynamic_setup_score": decision_row["dynamic_setup_score"],
                "readiness_score": decision_row["readiness_score"],
                "context_score": decision_row["context_score"],
                "setup_age_sessions": decision_row["setup_age_sessions"],
                "distance_to_pivot_pct": decision_row["distance_to_pivot_pct"],
                "candidate_rank": decision_row["candidate_rank"],
                "breakout_date": dates[t].date(),
                "pivot": round(float(setup.pivot), 4),
                "trigger_price": round(trigger, 4),
                "entry_filled": filled,
                "entry_date": decision_row["entry_date"] if filled else None,
                "entry_price": decision_row["entry_price"] if filled else None,
                "decision": decision,
            }
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

    def exposure_cap_value(idx: int) -> float:
        if market_exposure_cap is None:
            return 1.0
        if idx < 0 or idx >= len(market_exposure_cap):
            return 0.0
        value = market_exposure_cap[idx]
        if pd.isna(value):
            return 0.0
        return min(1.0, max(0.0, float(value)))

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
                "setup_type": pos.setup_type,
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
        if portfolio_mode and leg == "final":
            closed_outcomes.append(
                (
                    pos.position_id,
                    pos.realized_position_pnl,
                    pos.feedback_risk_units,
                )
            )

    def ratchet_stop_from_known_high(pos: Position) -> float | None:
        """Return tomorrow's causal stop from today's completed high.

        The fixed ladder is deliberately simple and monotone: a completed
        +2R high protects break-even, +3R protects +1R, +4R protects +2R and
        each further completed whole R raises the floor by another R. The
        caller stores this as ``next_stop`` so the new floor can never affect
        the bar whose high first qualified it.
        """
        r_unit = pos.entry_price - pos.initial_stop
        if r_unit <= 0 or not np.isfinite(pos.max_high):
            return None
        achieved_r = (pos.max_high - pos.entry_price) / r_unit
        completed_r = int(np.floor(achieved_r + 1e-12))
        if completed_r < 2:
            return None
        return pos.entry_price + (completed_r - 2) * r_unit

    def arm_next_ratchet_stop(pos: Position) -> None:
        candidate = ratchet_stop_from_known_high(pos)
        if candidate is None:
            return
        pending_floor = pos.stop if pos.next_stop is None else pos.next_stop
        pos.next_stop = max(pending_floor, candidate)

    def register_feedback_winner(risk_units: float) -> None:
        nonlocal exposure_index, consecutive_winners, consecutive_losses
        if risk_units <= 0:
            return
        consecutive_winners += risk_units
        consecutive_losses = 0.0
        threshold = float(cfg.exposure_winners_to_step_up)
        while consecutive_winners + 1e-12 >= threshold:
            exposure_index = min(exposure_index + 1, len(cfg.exposure_levels) - 1)
            consecutive_winners = max(0.0, consecutive_winners - threshold)

    def register_feedback_loss(risk_units: float) -> None:
        nonlocal exposure_index, consecutive_winners, consecutive_losses
        if risk_units <= 0:
            return
        previous_losses = consecutive_losses
        consecutive_losses += risk_units
        consecutive_winners = 0.0
        completed_before = int(np.floor(previous_losses + 1e-12))
        completed_after = int(np.floor(consecutive_losses + 1e-12))
        exposure_index = max(
            0,
            exposure_index - (completed_after - completed_before),
        )
        reset_threshold = float(cfg.exposure_losses_to_reset)
        completed_resets = int(
            np.floor((consecutive_losses + 1e-12) / reset_threshold)
        )
        if completed_resets > 0:
            exposure_index = 0
            consecutive_losses = max(
                0.0,
                consecutive_losses - completed_resets * reset_threshold,
            )
            if (
                consecutive_losses < 1e-12
                or reset_threshold - consecutive_losses < 1e-12
            ):
                consecutive_losses = 0.0

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

    for t in range(state_start_idx, len(dates)):
        # ---- make newly known causal structures active --------------------
        while next_setup < len(setup_rows) and setup_rows[next_setup].start_idx <= t:
            setup = setup_rows[next_setup]
            active_setups[int(setup.sim_setup_key)] = setup
            next_setup += 1
        active_setups = {
            setup_key: setup
            for setup_key, setup in active_setups.items()
            if setup.end_idx >= t
        }

        # Replay setup lifecycle before the measured period without opening
        # positions. This prevents an already broken or invalidated setup from
        # being resurrected at an OOS boundary.
        if t < sim_start_idx:
            for setup_key, setup in list(active_setups.items()):
                symbol = setup.symbol
                col = col_index[symbol]
                day_high, day_low = h[t, col], lo[t, col]
                trigger = float(setup.pivot) * (1 + cfg.pivot_buffer_pct)
                invalidation_level = max(
                    _num_attr(setup, "last_low", -np.inf),
                    _num_attr(setup, "stop_level", -np.inf),
                )
                incomplete_range = not np.isfinite([day_high, day_low]).all()
                broke_out = np.isfinite(day_high) and day_high >= trigger
                damaged_base = (
                    np.isfinite(day_low)
                    and np.isfinite(invalidation_level)
                    and day_low <= invalidation_level
                )
                if incomplete_range or broke_out or damaged_base:
                    del active_setups[setup_key]
            continue

        # Re-score every active structure from information available at the
        # previous close. Portfolio eligibility is a daily decision, not a
        # mutation of the underlying chart structure. A blocked first touch is
        # therefore still recorded and a later requalification remains possible
        # if price has not touched the pivot or invalidated the base.
        ranked_snapshots = list(ranker.rank(active_setups, t))
        current_snapshots = {
            snapshot.setup_key: snapshot for snapshot in ranked_snapshots
        }
        current_candidate_ranks = {
            snapshot.setup_key: rank
            for rank, snapshot in enumerate(ranked_snapshots, start=1)
        }
        ineligible: dict[int, str] = {}
        if portfolio_mode:
            trend_context_enabled = "trend_template_pass" in ranker.context
            for snapshot in ranked_snapshots:
                trend_value = snapshot.context_value("trend_template_pass")
                if trend_context_enabled and (
                    trend_value is None or trend_value < 0.5
                ):
                    ineligible[snapshot.setup_key] = "trend_template_not_passed"
                elif (
                    cfg.bad_fundamentals_filter_enable
                    and _known_bad_growth(snapshot)
                ):
                    ineligible[snapshot.setup_key] = "bad_fundamentals"
        for setup_key, decision in ineligible.items():
            set_entry_decision(active_setups[setup_key], decision)
        eligible_snapshots = [
            snapshot
            for snapshot in ranked_snapshots
            if snapshot.setup_key not in ineligible
        ]
        prioritized_setups = [
            active_setups[snapshot.setup_key] for snapshot in eligible_snapshots
        ]

        # ---- pre-session order slate ---------------------------------------
        # Capture budgets before processing any event from session t. Exits at
        # the open or intraday therefore cannot be recycled into today's slate.
        start_symbols = {position.symbol for position in positions.values()}
        # A next-open exit was decided at the previous close. A fresh setup in
        # that symbol may therefore be nominated for a later intraday re-entry
        # without ever holding two same-symbol positions. Its outgoing lot
        # still consumes today's opening cash, gross budget and slot.
        same_symbol_entry_blocks = (
            {
                pos.symbol
                for pos in positions.values()
                if pos.exit_next_open_reason is None
            }
            if portfolio_mode
            else set()
        )
        market_cap = exposure_cap_value(t - 1)
        if portfolio_mode and market_exposure_cap is not None and market_cap <= 0:
            exposure_index = 0
            consecutive_winners = 0.0
            consecutive_losses = 0.0
        # An open position may confirm portfolio feedback at yesterday's
        # close. Count each position only once and in stable position-id order;
        # no current-session high, close or exit can influence today's cap.
        if portfolio_mode and market_cap > 0 and t > 0:
            for pos in sorted(positions.values(), key=lambda item: item.position_id):
                if pos.position_id in feedback_winner_positions:
                    continue
                r_unit = pos.entry_price - pos.initial_stop
                previous_close = c[t - 1, pos.col]
                if (
                    r_unit > 0
                    and np.isfinite(previous_close)
                    and previous_close >= pos.entry_price + r_unit
                ):
                    feedback_winner_positions.add(pos.position_id)
                    register_feedback_winner(pos.feedback_risk_units)
        feedback_exposure_level = (
            cfg.exposure_levels[exposure_index] if portfolio_mode else 1.0
        )
        entry_exposure_limit = (
            min(feedback_exposure_level, market_cap)
            if portfolio_mode
            else (1.0 if market_cap > 0 else 0.0)
        )
        if portfolio_mode:
            if t > 0:
                sizing_equity = account_equity(t - 1)
                start_gross = open_notional(t - 1)
            else:
                sizing_equity = sizing_base
                start_gross = 0.0
            remaining_cash = max(0.0, cash)
            aggregate_gross_cap = (
                cfg.portfolio_max_gross_exposure_pct * entry_exposure_limit
            )
            gross_headroom = max(
                0.0,
                sizing_equity * aggregate_gross_cap - start_gross,
            )
            # Entry commission immediately lowers portfolio equity. Reserve
            # gross N against the post-commission denominator as well:
            # start_gross + N <= cap * (equity - commission * N).
            remaining_gross = gross_headroom / (
                1 + aggregate_gross_cap * cfg.commission_pct
            )
            # Feedback/market exposure is one aggregate gross-notional cap.
            # Per-trade risk, max-position size and slot count remain the
            # strategy's configured values and are not scaled a second time.
            remaining_slots = max(
                0, cfg.portfolio_max_open_positions - len(start_symbols)
            )
        else:
            sizing_equity = sizing_base
            remaining_cash = float("inf")
            remaining_gross = float("inf")
            remaining_slots = len(active_setups)

        # The market cap for row t is calculated with close[t], hence only t-1
        # may govern an intraday stop order in session t. regime_entry_allowed
        # is already defined by its caller as a per-session availability gate.
        entries_allowed = entry_exposure_limit > 0 and gate_value(
            regime_entry_allowed, t
        )
        slate: list[PlannedOrder] = []
        slate_candidates: list[SlateCandidate] = []
        selected_symbols: set[str] = set()
        unusable_setup_keys: set[int] = set()
        if not entries_allowed:
            gate_decision = (
                "market_gate_blocked"
                if entry_exposure_limit <= 0
                else "regime_gate_blocked"
            )
            for setup in prioritized_setups:
                set_entry_decision(
                    setup,
                    "existing_position"
                    if setup.symbol in same_symbol_entry_blocks
                    else gate_decision,
                )
        else:
            for setup in prioritized_setups:
                if portfolio_mode and (
                    setup.symbol in same_symbol_entry_blocks
                    or setup.symbol in selected_symbols
                ):
                    set_entry_decision(setup, "existing_position")
                    continue

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
                    set_entry_decision(setup, "invalid_order_parameters")
                    continue

                standalone_limits = [
                    sizing_equity * cfg.risk_pct / risk_per_share,
                    sizing_equity * cfg.max_position_pct / worst_fill,
                ]
                standalone_shares = int(min(standalone_limits))
                if standalone_shares < 1:
                    set_entry_decision(setup, "size_below_one_share")
                    continue

                candidate = SlateCandidate(
                    setup_key=setup_key,
                    setup=setup,
                    standalone_shares=standalone_shares,
                    worst_fill=worst_fill,
                    snapshot=current_snapshots[setup_key],
                )
                if portfolio_mode:
                    if len(slate_candidates) >= remaining_slots:
                        set_entry_decision(setup, "portfolio_capacity")
                        continue
                    tentative_candidates = [*slate_candidates, candidate]
                    tentative_slate = allocate_portfolio_slate(
                        tentative_candidates,
                        remaining_gross,
                        remaining_cash,
                    )
                    if tentative_slate is None:
                        set_entry_decision(setup, "portfolio_capacity")
                        continue
                    slate_candidates = tentative_candidates
                    slate = tentative_slate
                    selected_symbols.add(setup.symbol)
                else:
                    slate.append(
                        PlannedOrder(
                            setup_key=setup_key,
                            setup=setup,
                            shares=standalone_shares,
                            standalone_shares=standalone_shares,
                            worst_fill=worst_fill,
                            snapshot=current_snapshots[setup_key],
                        )
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
            arm_next_ratchet_stop(pos)
            if (
                r_unit > 0
                and t - pos.entry_idx >= cfg.time_stop_sessions
                and pos.max_high < pos.entry_price + cfg.time_stop_min_r * r_unit
                and np.isfinite(day_close)
                and day_close <= pos.pivot
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
            if not np.isfinite([day_open, day_high, lo[t, col]]).all():
                # Stop-buy execution cannot be simulated safely without the
                # complete session range. The setup continuity is
                # broken and the order is consumed.
                consumed_setup_keys.add(order.setup_key)
                set_entry_decision(setup, "incomplete_entry_bar")
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
                set_entry_decision(setup, "opened_below_invalidation")
                continue
            if day_high < trigger:
                set_entry_decision(setup, "not_triggered")
                continue
            if day_open > trigger * (1 + cfg.max_buy_zone_pct):
                set_entry_decision(setup, "excessive_gap")
                continue

            fill = max(day_open, trigger) * (1 + cfg.slippage_pct)
            stop = float(setup.stop_level)
            shares = order.shares

            new_position = Position(
                position_id=next_position_id,
                setup_id=getattr(setup, "setup_id", None),
                setup_type=str(getattr(setup, "setup_type", "unknown")),
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
                feedback_risk_units=order.feedback_risk_units,
            )
            next_position_id += 1
            consumed_setup_keys.add(order.setup_key)
            set_entry_decision(
                setup,
                "filled",
                entry_date=dates[t].date(),
                entry_price=fill,
            )
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
                    new_position.max_high = max(fill, float(day_high))
                    arm_next_ratchet_stop(new_position)
                    position_key: object = (
                        setup.symbol if portfolio_mode else order.setup_key
                    )
                    positions[position_key] = new_position

        # ---- setup lifecycle ----------------------------------------------
        # A breakout without a fill is missed, not a future invitation to
        # chase. A stop-level breach destroys the base. Filled and statically
        # unusable setups are consumed as well.
        for setup_key, setup in list(active_setups.items()):
            symbol = setup.symbol
            col = col_index[symbol]
            incomplete_bar = not np.isfinite([o[t, col], h[t, col], lo[t, col]]).all()
            day_high = h[t, col]
            day_low = lo[t, col]
            trigger = float(setup.pivot) * (1 + cfg.pivot_buffer_pct)
            missed_breakout = np.isfinite(day_high) and day_high >= trigger
            invalidation_level = max(
                _num_attr(setup, "last_low", -np.inf),
                _num_attr(setup, "stop_level", -np.inf),
            )
            damaged_base = (
                np.isfinite(day_low)
                and np.isfinite(invalidation_level)
                and day_low <= invalidation_level
            )
            if missed_breakout:
                record_breakout_event(setup, t, trigger)
            if (
                setup_key in consumed_setup_keys
                or incomplete_bar
                or missed_breakout
                or damaged_base
            ):
                del active_setups[setup_key]

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
                "feedback_exposure_level": feedback_exposure_level,
                "market_exposure_cap": market_cap,
                "entry_exposure_limit": entry_exposure_limit,
            }
        )

        if portfolio_mode:
            winner_risk_units = 0.0
            loser_risk_units = 0.0
            for position_id, outcome, risk_units in sorted(closed_outcomes):
                if outcome > 0:
                    if position_id not in feedback_winner_positions:
                        feedback_winner_positions.add(position_id)
                        winner_risk_units += risk_units
                else:
                    loser_risk_units += risk_units
            # Daily OHLC cannot establish the cross-symbol order of exits.
            # Process all known units but resolve a mixed session adversely:
            # winners first, losses last, so losses determine the next streak.
            if winner_risk_units > 0:
                register_feedback_winner(winner_risk_units)
            if loser_risk_units > 0:
                register_feedback_loss(loser_risk_units)
            closed_outcomes.clear()
            peak_equity = max(peak_equity, day_equity)
            drawdown = day_equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
            if (
                drawdown_reset_armed
                and drawdown <= -cfg.exposure_drawdown_reset_pct
            ):
                exposure_index = 0
                consecutive_winners = 0.0
                consecutive_losses = 0.0
                drawdown_reset_armed = False
            elif drawdown > -0.5 * cfg.exposure_drawdown_reset_pct:
                drawdown_reset_armed = True

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
    breakout_events_df = pd.DataFrame(
        breakout_events,
        columns=[
            "setup_id", "setup_type", "symbol", "setup_detect_date",
            "snapshot_date", "dynamic_setup_score", "readiness_score",
            "context_score", "setup_age_sessions", "distance_to_pivot_pct",
            "candidate_rank",
            "breakout_date", "pivot", "trigger_price", "entry_filled",
            "entry_date", "entry_price", "decision",
        ],
    )
    return SimResult(
        trades=trades_df,
        equity=equity_df,
        breakout_events=breakout_events_df,
        metrics=metrics,
    )


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
