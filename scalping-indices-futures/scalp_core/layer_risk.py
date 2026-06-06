"""Layer 5 — Risk: Monte-Carlo simulation for drawdown, slippage and sequence risk.

Built on the equity-fraction resampling pattern (pnl / equity_before preserves the
fixed-fractional compounding of the strategy). Three views are produced:

  base      trade order permuted (classic reshuffle) -> drawdown / ruin distribution
  slippage  same, after deducting MC_EXTRA_SLIPPAGE_POINTS per trade -> slippage stress
  sequence  block bootstrap (MC_BLOCK_SIZE) -> preserves win/loss streak structure
"""

import logging

import numpy as np

from . import config
from .entities import ClosedTrade

log = logging.getLogger(__name__)


def summarize_trades(trades: list[ClosedTrade], initial_equity: float, final_equity: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0, "breakeven_trades": 0,
            "win_rate_pct": 0.0, "profit_factor": 0.0, "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0, "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "final_equity": final_equity,
        }
    pnls = np.array([t.pnl for t in trades], dtype=np.float64)
    rets = np.array([t.return_pct for t in trades], dtype=np.float64)
    wins = pnls > 0
    losses = pnls < 0
    gross_win = pnls[wins].sum()
    gross_loss = -pnls[losses].sum()
    profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0

    equity = np.concatenate([[initial_equity], np.array([t.equity_after for t in trades])])
    running_max = np.maximum.accumulate(equity)
    dd = (equity - running_max) / running_max * 100.0
    max_dd = float(dd.min())

    return {
        "total_trades": int(n),
        "winning_trades": int(wins.sum()),
        "losing_trades": int(losses.sum()),
        "breakeven_trades": int(n - wins.sum() - losses.sum()),
        "win_rate_pct": round(float(wins.mean() * 100.0), 2),
        "profit_factor": round(profit_factor, 4) if np.isfinite(profit_factor) else None,
        "total_return_pct": round((final_equity - initial_equity) / initial_equity * 100.0, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "avg_win_pct": round(float(rets[wins].mean()), 4) if wins.any() else 0.0,
        "avg_loss_pct": round(float(rets[losses].mean()), 4) if losses.any() else 0.0,
        "final_equity": round(float(final_equity), 2),
    }


def _permute(fractions: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.tile(fractions, (n_sims, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)
    return shuffled


def _block_bootstrap(fractions: np.ndarray, n_sims: int, block: int, rng: np.random.Generator) -> np.ndarray:
    m = fractions.shape[0]
    n_blocks = int(np.ceil(m / block))
    starts = rng.integers(0, m, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % m  # wrap-around blocks
    sampled = fractions[idx].reshape(n_sims, n_blocks * block)[:, :m]
    return sampled


def _curve_stats(fractions: np.ndarray, n_sims: int, initial_equity: float, rng: np.random.Generator,
                 method: str) -> dict:
    if method == "block":
        sampled = _block_bootstrap(fractions, n_sims, config.MC_BLOCK_SIZE, rng)
    else:
        sampled = _permute(fractions, n_sims, rng)

    curves = np.empty((n_sims, fractions.shape[0] + 1), dtype=np.float64)
    curves[:, 0] = initial_equity
    curves[:, 1:] = initial_equity * np.cumprod(1.0 + sampled, axis=1)

    final_eq = curves[:, -1]
    running_max = np.maximum.accumulate(curves, axis=1)
    dd = (curves - running_max) / running_max * 100.0
    max_dd = dd.min(axis=1)
    total_ret = (final_eq - initial_equity) / initial_equity * 100.0

    def p(arr, q):
        return round(float(np.percentile(arr, q)), 4)

    ruin_threshold = initial_equity * (1.0 - config.MC_RUIN_DRAWDOWN_PCT / 100.0)
    return {
        "final_equity_p05": p(final_eq, 5), "final_equity_p25": p(final_eq, 25),
        "final_equity_p50": p(final_eq, 50), "final_equity_p75": p(final_eq, 75),
        "final_equity_p95": p(final_eq, 95),
        "max_drawdown_p05": p(max_dd, 5), "max_drawdown_p25": p(max_dd, 25),
        "max_drawdown_p50": p(max_dd, 50), "max_drawdown_p75": p(max_dd, 75),
        "max_drawdown_p95": p(max_dd, 95),
        "total_return_p05": p(total_ret, 5), "total_return_p25": p(total_ret, 25),
        "total_return_p50": p(total_ret, 50), "total_return_p75": p(total_ret, 75),
        "total_return_p95": p(total_ret, 95),
        "prob_of_ruin_pct": round(float(np.mean(final_eq < ruin_threshold) * 100.0), 2),
        "prob_profitable_pct": round(float(np.mean(final_eq > initial_equity) * 100.0), 2),
        "worst_final_equity": round(float(final_eq.min()), 2),
        "best_final_equity": round(float(final_eq.max()), 2),
        "worst_max_drawdown_pct": round(float(max_dd.min()), 4),
    }


def run_monte_carlo(trades: list[ClosedTrade], initial_equity: float) -> dict | None:
    n_sims = config.MONTE_CARLO_SIMULATIONS
    if not config.MONTE_CARLO_ENABLED or n_sims <= 0 or len(trades) < 2:
        return None

    eq_before = np.array([t.equity_before for t in trades], dtype=np.float64)
    pnl = np.array([t.pnl for t in trades], dtype=np.float64)
    base_frac = pnl / eq_before

    # Slippage stress: deduct extra round-trip slippage in index points.
    units = np.array([t.units for t in trades], dtype=np.float64)
    extra_cost = units * config.MC_EXTRA_SLIPPAGE_POINTS * config.CONTRACT_MULTIPLIER
    if config.EURUSD_RATE > 0:
        extra_cost = extra_cost / config.EURUSD_RATE
    if config.MC_EXTRA_SLIPPAGE_BPS:
        notional = np.array([t.notional for t in trades], dtype=np.float64)
        extra_cost = extra_cost + notional * (config.MC_EXTRA_SLIPPAGE_BPS / 10000.0) * 2.0
    slip_frac = (pnl - extra_cost) / eq_before

    rng = np.random.default_rng(config.MC_RANDOM_SEED)
    base = _curve_stats(base_frac, n_sims, initial_equity, rng, "permute")
    slippage = _curve_stats(slip_frac, n_sims, initial_equity, rng, "permute")
    sequence = _curve_stats(base_frac, n_sims, initial_equity, rng, "block")

    out = {"n_simulations": n_sims}
    out.update({f"base_{k}": v for k, v in base.items()})
    out.update({f"slip_{k}": v for k, v in slippage.items()})
    out.update({f"seq_{k}": v for k, v in sequence.items()})
    log.info(
        "Monte-Carlo done sims %d base_prob_ruin %.2f%% seq_prob_ruin %.2f%%",
        n_sims, base["prob_of_ruin_pct"], sequence["prob_of_ruin_pct"],
    )
    return out
