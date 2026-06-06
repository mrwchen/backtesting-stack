"""Walk-forward simulation tying the five layers together.

At every refit checkpoint (each REFIT_EVERY_BARS, beginning at WARMUP_BARS) all
layers are re-fit on a past-only training window. Layer outputs for every bar are
then recomputed *causally* with those fresh parameters and used both to train the
decision classifier and to generate signals for the upcoming block. No per-bar
inference ever uses information beyond that bar.

Execution model: a signal at the close of bar t is filled at the open of bar t+1
(no look-ahead). Positions are intraday-only and forced flat at the session cutoff.
"""

import logging
from dataclasses import dataclass, field
from datetime import time

import numpy as np
import pandas as pd

from . import broker, config
from .data import session_flat_cutoff
from .entities import ClosedTrade
from .layer_decision import make_decision_model
from .layer_price import make_price_filter
from .layer_regime import RegimeModel
from .layer_volatility import make_vol_model

log = logging.getLogger(__name__)


@dataclass
class SimulationResult:
    trades: list[ClosedTrade] = field(default_factory=list)
    initial_equity: float = 0.0
    final_equity: float = 0.0
    equity_curve: list[float] = field(default_factory=list)
    bars_total: int = 0
    bars_simulated: int = 0
    ruined: bool = False


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


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
    target_up = features["target_up"].to_numpy(dtype=np.float64)
    session_date = features["session_date"].to_numpy()
    local_time = features["local_time"].to_numpy()
    cutoff: time = session_flat_cutoff()

    regime_feats = np.column_stack([log_ret, abs_ret])

    regime = RegimeModel(config.REGIME_STATES)
    price = make_price_filter(config.PRICE_MODEL)
    vol = make_vol_model(config.VOL_MODEL)
    decision = make_decision_model(config.DECISION_MODEL)

    start = config.WARMUP_BARS
    result = SimulationResult(initial_equity=config.INITIAL_EQUITY, bars_total=n)
    if start >= n - 1:
        log.warning("WARMUP_BARS=%d >= available bars %d; nothing to simulate", start, n)
        result.final_equity = config.INITIAL_EQUITY
        result.equity_curve = [config.INITIAL_EQUITY]
        return result

    equity = config.INITIAL_EQUITY
    equity_curve = [equity]

    # Per-refit caches (full-length, causal arrays recomputed each checkpoint).
    states = np.zeros(n, dtype=np.int64)
    high_vol = np.zeros(n, dtype=bool)
    levels = c.copy()
    sigma_ret = np.full(n, np.nan)
    X_full = np.zeros((n, 6), dtype=np.float64)

    def refit(upto: int) -> None:
        nonlocal states, high_vol, levels, sigma_ret, X_full
        train_start = 0 if config.TRAIN_WINDOW_BARS <= 0 else max(0, upto - config.TRAIN_WINDOW_BARS)
        regime.fit(regime_feats[train_start:upto])
        price.update_params(c[train_start:upto])
        vol.update_params(log_ret[train_start:upto])

        states, high_vol = regime.filtered_states(regime_feats)
        levels, slopes = price.filtered(c)
        sigma_ret = vol.conditional_vol(log_ret)

        price_dev = c - levels
        X_full = np.column_stack([
            states.astype(np.float64),
            price_dev,
            slopes,
            sigma_ret,
            momentum,
            rsi,
        ])
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

        # Train decision model on past-only labelled rows.
        lo = train_start
        hi = upto  # rows [lo, hi) all have known next-bar labels (hi <= n-1 typically)
        hi = min(hi, n - 1)
        if hi - lo >= 50:
            Xtr = X_full[lo:hi]
            ytr = target_up[lo:hi]
            mask = np.isfinite(ytr)
            decision.fit(Xtr[mask], ytr[mask])
        else:
            decision.fit(np.empty((0, 6)), np.empty(0))
        log.info(
            "Refit @bar %d train[%d:%d] regime_fitted=%s decision_fitted=%s",
            upto, train_start, upto, regime.fitted, decision.fitted,
        )

    # ── position state machine ──────────────────────────────────────────────
    position = None          # dict | None
    pending = None           # signal generated previous bar, to fill at this bar's open

    def close_position(exit_ts, exit_price, status, bars_held) -> None:
        nonlocal equity, position
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
            margin_used=p["margin_used"], regime_state=p["regime_state"], prob_up=p["prob_up"],
            sigma_pts=p["sigma_pts"], stop_price=p["stop_price"], take_profit_price=p["tp_price"],
            outcome_status=status, exit_ts=exit_ts, exit_price=exit_price, bars_held=bars_held,
            return_pct=ret_pct, pnl=pnl, costs=costs,
            equity_before=equity_before, equity_after=equity_after,
        ))
        equity = equity_after
        equity_curve.append(equity)
        position = None

    last_simulated = start
    for t in range(start, n):
        last_simulated = t
        # refit at checkpoints
        if t == start or (t - start) % config.REFIT_EVERY_BARS == 0:
            refit(t)

        # 1. Fill a pending signal at this bar's open.
        if position is None and pending is not None:
            entry_price = o[t]
            sizing = broker.size_position(equity, entry_price, pending["stop_pct"])
            if sizing.units > 0:
                if pending["direction"] == "LONG":
                    stop_price = entry_price * (1.0 - pending["stop_pct"] / 100.0)
                    tp_price = entry_price * (1.0 + pending["tp_pct"] / 100.0)
                else:
                    stop_price = entry_price * (1.0 + pending["stop_pct"] / 100.0)
                    tp_price = entry_price * (1.0 - pending["tp_pct"] / 100.0)
                position = {
                    "intent_ts": pending["intent_ts"], "entry_ts": ts[t], "entry_price": entry_price,
                    "direction": pending["direction"], "units": sizing.units,
                    "notional": sizing.notional_eur, "margin_used": sizing.margin_used_eur,
                    "regime_state": pending["regime_state"], "prob_up": pending["prob_up"],
                    "sigma_pts": pending["sigma_pts"], "stop_price": stop_price, "tp_price": tp_price,
                    "equity_before": equity, "entry_session": session_date[t], "bars_held": 0,
                }
            pending = None

        # 2. Manage an open position on this bar.
        if position is not None:
            position["bars_held"] += 1
            bars_held = position["bars_held"]
            d = position["direction"]
            sp, tp = position["stop_price"], position["tp_price"]
            exited = False
            # Intrabar SL/TP (conservative: stop checked before target).
            if d == "LONG":
                if low[t] <= sp:
                    close_position(ts[t], sp, "HIT_SL", bars_held); exited = True
                elif h[t] >= tp:
                    close_position(ts[t], tp, "HIT_TP", bars_held); exited = True
            else:
                if h[t] >= sp:
                    close_position(ts[t], sp, "HIT_SL", bars_held); exited = True
                elif low[t] <= tp:
                    close_position(ts[t], tp, "HIT_TP", bars_held); exited = True
            # Session flat / new session / max hold.
            if not exited:
                new_session = session_date[t] != position["entry_session"]
                if new_session or local_time[t] >= cutoff:
                    close_position(ts[t], c[t], "SESSION_FLAT", bars_held); exited = True
                elif bars_held >= config.MAX_HOLD_BARS:
                    close_position(ts[t], c[t], "MAX_HOLD", bars_held); exited = True

            if equity <= 0:
                result.ruined = True
                break

        # 3. Generate a new signal (fills next bar). Only when flat and no pending.
        if position is None and pending is None and t < n - 1:
            # Don't open into the session-close window.
            if local_time[t] >= cutoff:
                continue
            p_up = float(decision.proba_up(X_full[t : t + 1])[0])
            if config.REGIME_BLOCK_HIGH_VOL_STATE and high_vol[t]:
                continue
            sr = sigma_ret[t]
            if not np.isfinite(sr) or sr <= 0:
                continue
            stop_pct = _clamp(config.STOP_VOL_MULT * sr * 100.0, config.MIN_STOP_PCT, config.MAX_STOP_PCT)
            tp_pct = max(config.TP_VOL_MULT * sr * 100.0, stop_pct)
            sigma_pts = sr * c[t]

            direction = None
            if p_up >= config.PROB_THRESHOLD:
                direction = "LONG"
            elif config.ALLOW_SHORT and (1.0 - p_up) >= config.PROB_THRESHOLD:
                direction = "SHORT"
            if direction is not None:
                pending = {
                    "direction": direction, "stop_pct": stop_pct, "tp_pct": tp_pct,
                    "sigma_pts": sigma_pts, "regime_state": int(states[t]), "prob_up": p_up,
                    "intent_ts": ts[t],
                }

    # Force-close any residual position at the last bar.
    if position is not None:
        close_position(ts[last_simulated], c[last_simulated], "SESSION_FLAT", position["bars_held"])

    result.final_equity = equity
    result.equity_curve = equity_curve
    result.bars_simulated = max(0, last_simulated - start + 1)
    log.info(
        "Simulation done trades %d final_equity %.2f ruined %s",
        len(result.trades), result.final_equity, result.ruined,
    )
    return result
