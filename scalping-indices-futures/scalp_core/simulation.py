"""Walk-forward simulation tying the five layers together.

At every refit checkpoint all statistical layers are fit on past-only data. The
decision layer is trained on historical trade-outcome labels whose exits are
already known at the refit bar, then scores the current bar's long/short trade
edge. A signal at the close of bar t is still filled at the open of bar t+1.
"""

import logging
from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from . import broker, config
from .data import session_entry_end, session_entry_start, session_flat_cutoff
from .entities import ClosedTrade, DecisionTrace
from .layer_decision import DecisionScores, make_decision_model
from .layer_price import make_price_filter
from .layer_regime import RegimeModel
from .layer_trade_outcome import (
    OutcomeCache,
    TradePlan,
    build_atr_outcome_cache,
    make_trade_plan,
    simulate_trade_outcome,
)
from .layer_volatility import make_vol_model

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    decisions: list[DecisionTrace] = field(default_factory=list)
    initial_equity: float = 0.0
    final_equity: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    bars_total: int = 0
    bars_simulated: int = 0
    ruined: bool = False


def _fmt_ts(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _minutes(value: time) -> int:
    return value.hour * 60 + value.minute


def _session_progress(local_time: np.ndarray, entry_start: time, entry_end: time) -> np.ndarray:
    start = _minutes(entry_start)
    end = _minutes(entry_end)
    denom = max(1, end - start)
    out = np.empty(local_time.shape[0], dtype=np.float64)
    for idx, value in enumerate(local_time):
        out[idx] = (_minutes(value) - start) / denom
    return np.clip(out, 0.0, 1.0)


def _empty_x(n_features: int) -> np.ndarray:
    return np.empty((0, n_features), dtype=np.float64)


def run_simulation(features: pd.DataFrame) -> SimulationResult:
    n = len(features)
    ts = features["ts"].to_numpy()
    o = features["open"].to_numpy(dtype=np.float64)
    h = features["high"].to_numpy(dtype=np.float64)
    low = features["low"].to_numpy(dtype=np.float64)
    c = features["close"].to_numpy(dtype=np.float64)
    log_ret = features["log_ret"].to_numpy(dtype=np.float64)
    abs_ret = features["abs_ret"].to_numpy(dtype=np.float64)
    momentum = features["momentum"].to_numpy(dtype=np.float64)
    rsi = features["rsi"].to_numpy(dtype=np.float64)
    atr = features["atr"].to_numpy(dtype=np.float64)
    session_date = features["session_date"].to_numpy()
    local_time = features["local_time"].to_numpy()
    entry_start: time = session_entry_start()
    entry_end: time = session_entry_end()
    cutoff: time = session_flat_cutoff()

    regime_feats = np.column_stack([log_ret, abs_ret])
    progress = _session_progress(local_time, entry_start, entry_end)
    entry_window = np.array([(entry_start <= value < entry_end) for value in local_time], dtype=bool)

    regime = RegimeModel(config.REGIME_STATES)
    price = make_price_filter(config.PRICE_MODEL)
    vol = make_vol_model(config.VOL_MODEL)
    decision = make_decision_model(config.DECISION_MODEL)

    start = config.WARMUP_BARS
    result = SimulationResult(initial_equity=config.INITIAL_EQUITY, bars_total=n)
    if start >= n - 1:
        log.warning("Warmup bars %d exceed available bars %d; nothing to simulate", start, n)
        result.final_equity = config.INITIAL_EQUITY
        result.equity_curve = [config.INITIAL_EQUITY]
        return result

    outcome_cache: OutcomeCache | None = None
    if config.STOP_MODE == "atr":
        outcome_cache = build_atr_outcome_cache(o, h, low, c, atr, session_date, local_time, entry_start, entry_end, cutoff)
        log.info("Built ATR trade-outcome label cache for %d bars", n)

    log.info(
        "Simulation start at bar %d utc %s session %s local %s total bars %d entry window %s-%s %s refit every %d bars",
        start, _fmt_ts(ts[start]), session_date[start], local_time[start], n,
        config.ENTRY_START_TIME, config.ENTRY_END_TIME, config.SESSION_TZ, config.REFIT_EVERY_BARS,
    )

    equity = config.INITIAL_EQUITY
    equity_curve = [equity]

    states = np.zeros(n, dtype=np.int64)
    high_vol = np.zeros(n, dtype=bool)
    levels = c.copy()
    sigma_ret = np.full(n, np.nan)
    X_full = np.zeros((n, 8), dtype=np.float64)
    dynamic_label_cache: dict[tuple[int, str, float, float], object] = {}

    def _feature_matrix(slopes: np.ndarray) -> np.ndarray:
        sigma_pts = np.where(np.isfinite(sigma_ret) & (sigma_ret > 0), sigma_ret * c, 0.0)
        basis = np.maximum.reduce([
            np.where(np.isfinite(atr) & (atr > 0), atr, 0.0),
            sigma_pts,
            np.full(n, 1e-6, dtype=np.float64),
        ])
        price_dev_basis = (c - levels) / basis
        slope_basis = slopes / basis
        rsi_centered = (rsi - 50.0) / 50.0
        X = np.column_stack([
            states.astype(np.float64),
            high_vol.astype(np.float64),
            price_dev_basis,
            slope_basis,
            sigma_ret,
            momentum,
            rsi_centered,
            progress,
        ])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    def _dynamic_outcome(idx: int, direction: str, plan: TradePlan):
        key = (idx, direction, round(plan.stop_pct, 8), round(plan.tp_pct, 8))
        if key not in dynamic_label_cache:
            dynamic_label_cache[key] = simulate_trade_outcome(idx, direction, plan, o, h, low, c, session_date, local_time, cutoff)
        return dynamic_label_cache[key]

    def _training_data(upto: int, train_start: int):
        base_mask = np.zeros(n, dtype=bool)
        base_mask[train_start:min(upto, n - 1)] = True
        base_mask &= entry_window
        if config.REGIME_BLOCK_HIGH_VOL_STATE:
            base_mask &= ~high_vol

        if outcome_cache is not None:
            long_mask = base_mask & outcome_cache.long_valid & (outcome_cache.long_exit_idx >= 0) & (outcome_cache.long_exit_idx < upto)
            short_mask = base_mask & outcome_cache.short_valid & (outcome_cache.short_exit_idx >= 0) & (outcome_cache.short_exit_idx < upto)
            return (
                X_full[long_mask], outcome_cache.long_win[long_mask], outcome_cache.long_net_r[long_mask],
                X_full[short_mask], outcome_cache.short_win[short_mask], outcome_cache.short_net_r[short_mask],
            )

        long_x, long_y, long_r = [], [], []
        short_x, short_y, short_r = [], [], []
        for idx in np.flatnonzero(base_mask):
            plan = make_trade_plan(float(c[idx]), float(sigma_ret[idx]), float(atr[idx]))
            if plan is None:
                continue
            long_out = _dynamic_outcome(idx, "LONG", plan)
            if long_out is not None and long_out.exit_idx < upto:
                long_x.append(X_full[idx])
                long_y.append(1.0 if long_out.net_r > 0.0 else 0.0)
                long_r.append(long_out.net_r)
            if config.ALLOW_SHORT:
                short_out = _dynamic_outcome(idx, "SHORT", plan)
                if short_out is not None and short_out.exit_idx < upto:
                    short_x.append(X_full[idx])
                    short_y.append(1.0 if short_out.net_r > 0.0 else 0.0)
                    short_r.append(short_out.net_r)

        n_features = X_full.shape[1]
        return (
            np.asarray(long_x, dtype=np.float64) if long_x else _empty_x(n_features),
            np.asarray(long_y, dtype=np.float64),
            np.asarray(long_r, dtype=np.float64),
            np.asarray(short_x, dtype=np.float64) if short_x else _empty_x(n_features),
            np.asarray(short_y, dtype=np.float64),
            np.asarray(short_r, dtype=np.float64),
        )

    def refit(upto: int) -> None:
        nonlocal states, high_vol, levels, sigma_ret, X_full
        train_start = 0 if config.TRAIN_WINDOW_BARS <= 0 else max(0, upto - config.TRAIN_WINDOW_BARS)
        regime.fit(regime_feats[train_start:upto])
        price.update_params(c[train_start:upto])
        vol.update_params(log_ret[train_start:upto])

        states, high_vol = regime.filtered_states(regime_feats)
        levels, slopes = price.filtered(c)
        sigma_ret = vol.conditional_vol(log_ret)
        X_full = _feature_matrix(slopes)

        X_long, y_long, r_long, X_short, y_short, r_short = _training_data(upto, train_start)
        decision.fit(X_long, y_long, r_long, X_short, y_short, r_short)
        log.info(
            "Refit at bar %d utc %s session %s local %s train bars %d to %d train utc %s to %s long labels %d short labels %d decision fitted %s",
            upto, _fmt_ts(ts[upto]), session_date[upto], local_time[upto],
            train_start, upto, _fmt_ts(ts[train_start]), _fmt_ts(ts[upto - 1]),
            len(y_long), len(y_short), decision.fitted,
        )

    position = None
    pending = None
    last_exit_t = -10**9

    def append_decision(
        idx: int,
        action: str,
        reason: str,
        direction: str | None = None,
        scores: DecisionScores | None = None,
        plan: TradePlan | None = None,
    ) -> None:
        if scores is None:
            prob_long = prob_short = selected_prob = expected = expected_long = expected_short = None
        else:
            prob_long = scores.prob_long_win
            prob_short = scores.prob_short_win
            expected_long = scores.expected_long_r
            expected_short = scores.expected_short_r
            if direction == "LONG":
                selected_prob = scores.prob_long_win
                expected = scores.expected_long_r
            elif direction == "SHORT":
                selected_prob = scores.prob_short_win
                expected = scores.expected_short_r
            else:
                selected_prob = max(scores.prob_long_win, scores.prob_short_win)
                expected = max(scores.expected_long_r, scores.expected_short_r)

        result.decisions.append(DecisionTrace(
            ts=ts[idx],
            session_date=session_date[idx],
            local_time=local_time[idx],
            close_price=float(c[idx]),
            in_entry_window=bool(entry_window[idx]),
            decision_action=action,
            decision_reason=reason,
            direction=direction,
            prob_long_win=prob_long,
            prob_short_win=prob_short,
            selected_trade_prob=selected_prob,
            expected_net_r=expected,
            expected_long_r=expected_long,
            expected_short_r=expected_short,
            regime_state=int(states[idx]) if idx < len(states) else None,
            high_vol_state=bool(high_vol[idx]) if idx < len(high_vol) else None,
            sigma_pts=plan.sigma_pts if plan is not None else None,
            atr_pts=float(atr[idx]) if np.isfinite(atr[idx]) else None,
            stop_pct=plan.stop_pct if plan is not None else None,
            tp_pct=plan.tp_pct if plan is not None else None,
        ))

    def close_position(exit_ts, exit_price, status, bars_held, exit_idx) -> None:
        nonlocal equity, position, last_exit_t
        p = position
        gross = broker.gross_pnl(p["units"], p["entry_price"], exit_price, p["direction"])
        costs = broker.round_trip_costs(p["units"], p["entry_price"], exit_price)
        pnl = gross - costs
        equity_before = p["equity_before"]
        equity_after = equity + pnl
        ret_pct = (pnl / equity_before * 100.0) if equity_before > 0 else 0.0
        result.trades.append(ClosedTrade(
            intent_ts=p["intent_ts"], entry_ts=p["entry_ts"], entry_price=p["entry_price"],
            direction=p["direction"], units=p["units"], notional=p["notional"],
            margin_used=p["margin_used"], regime_state=p["regime_state"],
            prob_long_win=p["prob_long_win"], prob_short_win=p["prob_short_win"],
            selected_trade_prob=p["selected_trade_prob"], expected_net_r=p["expected_net_r"],
            decision_reason=p["decision_reason"],
            sigma_pts=p["sigma_pts"], stop_price=p["stop_price"], take_profit_price=p["tp_price"],
            outcome_status=status, exit_ts=exit_ts, exit_price=exit_price, bars_held=bars_held,
            return_pct=ret_pct, pnl=pnl, costs=costs,
            equity_before=equity_before, equity_after=equity_after,
        ))
        equity = equity_after
        equity_curve.append(equity)
        last_exit_t = exit_idx
        position = None

    def choose_direction(scores: DecisionScores) -> tuple[str | None, str]:
        long_ok = (
            scores.long_fitted
            and scores.prob_long_win >= config.PROB_THRESHOLD
            and scores.expected_long_r >= config.MIN_EXPECTED_NET_R
        )
        short_ok = (
            config.ALLOW_SHORT
            and scores.short_fitted
            and scores.prob_short_win >= config.PROB_THRESHOLD
            and scores.expected_short_r >= config.MIN_EXPECTED_NET_R
        )
        if long_ok and short_ok:
            if scores.expected_long_r >= scores.expected_short_r:
                return "LONG", "LONG_EXPECTED_EDGE"
            return "SHORT", "SHORT_EXPECTED_EDGE"
        if long_ok:
            return "LONG", "LONG_EXPECTED_EDGE"
        if short_ok:
            return "SHORT", "SHORT_EXPECTED_EDGE"

        max_prob = max(scores.prob_long_win if scores.long_fitted else 0.0, scores.prob_short_win if scores.short_fitted else 0.0)
        max_expected = max(
            scores.expected_long_r if scores.long_fitted else -np.inf,
            scores.expected_short_r if scores.short_fitted else -np.inf,
        )
        if not scores.long_fitted and not scores.short_fitted:
            return None, "MODEL_NOT_FITTED"
        if max_prob < config.PROB_THRESHOLD:
            return None, "PROBABILITY_BELOW_THRESHOLD"
        if max_expected < config.MIN_EXPECTED_NET_R:
            return None, "EXPECTED_NET_R_BELOW_MIN"
        return None, "SIDE_DISABLED"

    last_simulated = start
    for t in range(start, n):
        last_simulated = t
        if t == start or (t - start) % config.REFIT_EVERY_BARS == 0:
            refit(t)

        if position is None and pending is not None:
            entry_price = o[t]
            sizing = broker.size_position(equity, entry_price, pending["stop_pct"])
            if sizing.units > 0:
                if pending["direction"] == "LONG":
                    stop_price = entry_price * (1.0 - pending["stop_pct"] / 100.0)
                    tp_price = entry_price * (1.0 + pending["tp_pct"] / 100.0)
                    trail_activation_price = entry_price * (1.0 + pending["trail_activation_pct"] / 100.0)
                else:
                    stop_price = entry_price * (1.0 + pending["stop_pct"] / 100.0)
                    tp_price = entry_price * (1.0 - pending["tp_pct"] / 100.0)
                    trail_activation_price = entry_price * (1.0 - pending["trail_activation_pct"] / 100.0)
                position = {
                    "intent_ts": pending["intent_ts"], "entry_ts": ts[t], "entry_price": entry_price,
                    "direction": pending["direction"], "units": sizing.units,
                    "notional": sizing.notional_eur, "margin_used": sizing.margin_used_eur,
                    "regime_state": pending["regime_state"],
                    "prob_long_win": pending["prob_long_win"], "prob_short_win": pending["prob_short_win"],
                    "selected_trade_prob": pending["selected_trade_prob"], "expected_net_r": pending["expected_net_r"],
                    "decision_reason": pending["decision_reason"],
                    "sigma_pts": pending["sigma_pts"], "stop_price": stop_price,
                    "tp_price": (tp_price if config.TP_MODE == "fixed" else None),
                    "equity_before": equity, "entry_session": session_date[t], "bars_held": 0,
                    "trail_active": False, "trail_distance_pct": pending["trail_distance_pct"],
                    "trail_activation_price": trail_activation_price,
                    "trailing_stop": None, "extreme": entry_price,
                }
            pending = None

        if position is not None:
            position["bars_held"] += 1
            bars_held = position["bars_held"]
            d = position["direction"]
            sp = position["stop_price"]
            exited = False

            if config.TP_MODE == "fixed":
                tp = position["tp_price"]
                stop_first = config.INTRABAR_FILL_PRIORITY == "stop"
                if d == "LONG":
                    hit_stop, hit_tp = low[t] <= sp, h[t] >= tp
                else:
                    hit_stop, hit_tp = h[t] >= sp, low[t] <= tp
                order = [("stop", hit_stop, sp, "HIT_SL"), ("tp", hit_tp, tp, "HIT_TP")]
                if not stop_first:
                    order.reverse()
                for _kind, hit, px, status in order:
                    if hit:
                        close_position(ts[t], px, status, bars_held, t)
                        exited = True
                        break
            else:
                eff_stop = position["trailing_stop"] if (position["trail_active"] and position["trailing_stop"] is not None) else sp
                if d == "LONG":
                    if low[t] <= eff_stop:
                        status = "HIT_TRAILING_STOP" if position["trail_active"] else "HIT_SL"
                        close_position(ts[t], eff_stop, status, bars_held, t)
                        exited = True
                    else:
                        position["extreme"] = max(position["extreme"], h[t])
                        if not position["trail_active"] and h[t] >= position["trail_activation_price"]:
                            position["trail_active"] = True
                        if position["trail_active"]:
                            new_stop = position["extreme"] * (1.0 - position["trail_distance_pct"] / 100.0)
                            prev = position["trailing_stop"]
                            position["trailing_stop"] = new_stop if prev is None else max(prev, new_stop)
                else:
                    if h[t] >= eff_stop:
                        status = "HIT_TRAILING_STOP" if position["trail_active"] else "HIT_SL"
                        close_position(ts[t], eff_stop, status, bars_held, t)
                        exited = True
                    else:
                        position["extreme"] = min(position["extreme"], low[t])
                        if not position["trail_active"] and low[t] <= position["trail_activation_price"]:
                            position["trail_active"] = True
                        if position["trail_active"]:
                            new_stop = position["extreme"] * (1.0 + position["trail_distance_pct"] / 100.0)
                            prev = position["trailing_stop"]
                            position["trailing_stop"] = new_stop if prev is None else min(prev, new_stop)

            if not exited:
                new_session = session_date[t] != position["entry_session"]
                if new_session or local_time[t] >= cutoff:
                    close_position(ts[t], c[t], "SESSION_FLAT", bars_held, t)
                    exited = True
                elif bars_held >= config.MAX_HOLD_BARS:
                    close_position(ts[t], c[t], "MAX_HOLD", bars_held, t)
                    exited = True

            if equity <= 0:
                result.ruined = True
                break

        if t >= n - 1:
            continue

        if position is not None:
            append_decision(t, "POSITION_OPEN", "position open between entry and exit")
            continue
        if pending is not None:
            append_decision(t, "PENDING_SIGNAL", "pending signal waits for next bar open")
            continue
        if not entry_window[t]:
            append_decision(t, "NO_TRADE", "OUT_OF_ENTRY_WINDOW")
            continue
        if t - last_exit_t < config.REENTRY_COOLDOWN_BARS:
            append_decision(t, "NO_TRADE", "REENTRY_COOLDOWN")
            continue
        if config.REGIME_BLOCK_HIGH_VOL_STATE and high_vol[t]:
            append_decision(t, "NO_TRADE", "HIGH_VOL_REGIME_BLOCK")
            continue

        plan = make_trade_plan(float(c[t]), float(sigma_ret[t]), float(atr[t]))
        if plan is None:
            append_decision(t, "NO_TRADE", "INVALID_STOP_BASIS")
            continue

        scores = decision.score(X_full[t:t + 1])
        direction, reason = choose_direction(scores)
        if direction is None:
            append_decision(t, "NO_TRADE", reason, scores=scores, plan=plan)
            continue

        selected_prob = scores.prob_long_win if direction == "LONG" else scores.prob_short_win
        expected_net_r = scores.expected_long_r if direction == "LONG" else scores.expected_short_r
        pending = {
            "direction": direction,
            "stop_pct": plan.stop_pct,
            "tp_pct": plan.tp_pct,
            "trail_activation_pct": plan.trail_activation_pct,
            "trail_distance_pct": plan.trail_distance_pct,
            "sigma_pts": plan.sigma_pts,
            "regime_state": int(states[t]),
            "prob_long_win": scores.prob_long_win,
            "prob_short_win": scores.prob_short_win,
            "selected_trade_prob": selected_prob,
            "expected_net_r": expected_net_r,
            "decision_reason": reason,
            "intent_ts": ts[t],
        }
        append_decision(t, f"{direction}_SIGNAL", reason, direction=direction, scores=scores, plan=plan)

    if position is not None:
        close_position(ts[last_simulated], c[last_simulated], "SESSION_FLAT", position["bars_held"], last_simulated)

    result.final_equity = equity
    result.equity_curve = equity_curve
    result.bars_simulated = max(0, last_simulated - start + 1)
    log.info(
        "Simulation done at bar %d utc %s session %s local %s simulated bars %d trades %d decisions %d final equity %.2f ruined %s",
        last_simulated, _fmt_ts(ts[last_simulated]), session_date[last_simulated], local_time[last_simulated],
        result.bars_simulated, len(result.trades), len(result.decisions), result.final_equity, result.ruined,
    )
    return result
