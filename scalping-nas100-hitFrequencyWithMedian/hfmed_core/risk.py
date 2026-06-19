"""Trade summary and Monte-Carlo risk views."""

import logging

import numpy as np

from . import broker
from .config import RunConfig
from .entities import ClosedTrade
from .sessions import SESSION_TYPES

log = logging.getLogger(__name__)


def summarize_trades(trades: list[ClosedTrade], initial_equity: float, final_equity: float) -> dict:
    n = len(trades)
    if n == 0:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "net_profit_eur": 0.0,
            "avg_trade_pnl_eur": 0.0,
            "final_equity": round(float(final_equity), 2),
        }

    pnls = np.array([t.pnl_eur for t in trades], dtype=np.float64)
    rets = np.array([t.return_pct for t in trades], dtype=np.float64)
    wins = pnls > 0
    losses = pnls < 0
    gross_win = float(pnls[wins].sum())
    gross_loss = float(-pnls[losses].sum())
    if gross_loss > 0:
        profit_factor = gross_win / gross_loss
    elif gross_win > 0:
        profit_factor = None
    else:
        profit_factor = 0.0

    equity = np.concatenate([[initial_equity], np.array([t.equity_after for t in trades])])
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100.0

    return {
        "total_trades": int(n),
        "winning_trades": int(wins.sum()),
        "losing_trades": int(losses.sum()),
        "breakeven_trades": int(n - wins.sum() - losses.sum()),
        "win_rate_pct": round(float(wins.mean() * 100.0), 2),
        "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
        "total_return_pct": round((final_equity - initial_equity) / initial_equity * 100.0, 4),
        "max_drawdown_pct": round(float(drawdown.min()), 4),
        "avg_win_pct": round(float(rets[wins].mean()), 4) if wins.any() else 0.0,
        "avg_loss_pct": round(float(rets[losses].mean()), 4) if losses.any() else 0.0,
        "gross_profit_eur": round(float(gross_win), 2),
        "gross_loss_eur": round(float(gross_loss), 2),
        "net_profit_eur": round(float(pnls.sum()), 2),
        "avg_trade_pnl_eur": round(float(pnls.mean()), 4),
        "final_equity": round(float(final_equity), 2),
    }


def summarize_trades_by_session(trades: list[ClosedTrade]) -> dict[str, dict]:
    stats = {
        session_type: {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "net_profit_eur": 0.0,
            "avg_trade_pnl_eur": 0.0,
        }
        for session_type, _label, _sort_order in SESSION_TYPES
    }
    for trade in trades:
        row = stats.get(trade.entry_session)
        if row is None:
            continue
        pnl = float(trade.pnl_eur)
        row["total_trades"] += 1
        row["net_profit_eur"] += pnl
        if pnl > 0.0:
            row["winning_trades"] += 1
            row["gross_profit_eur"] += pnl
        elif pnl < 0.0:
            row["losing_trades"] += 1
            row["gross_loss_eur"] += -pnl
        else:
            row["breakeven_trades"] += 1

    for row in stats.values():
        total = int(row["total_trades"])
        row["win_rate_pct"] = round(float(row["winning_trades"]) / total * 100.0, 2) if total > 0 else 0.0
        row["gross_profit_eur"] = round(float(row["gross_profit_eur"]), 2)
        row["gross_loss_eur"] = round(float(row["gross_loss_eur"]), 2)
        row["net_profit_eur"] = round(float(row["net_profit_eur"]), 2)
        row["avg_trade_pnl_eur"] = round(float(row["net_profit_eur"]) / total, 4) if total > 0 else 0.0
    return stats


def _permute(fractions: np.ndarray, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    shuffled = np.tile(fractions, (n_sims, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)
    return shuffled


def _block_bootstrap(fractions: np.ndarray, n_sims: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n = fractions.shape[0]
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return fractions[idx].reshape(n_sims, n_blocks * block)[:, :n]


def _curve_stats(fractions: np.ndarray, n_sims: int, initial_equity: float, rng: np.random.Generator, method: str, cfg: RunConfig) -> dict:
    sampled = _block_bootstrap(fractions, n_sims, cfg.mc_block_size, rng) if method == "block" else _permute(fractions, n_sims, rng)
    curves = np.empty((n_sims, fractions.shape[0] + 1), dtype=np.float64)
    curves[:, 0] = initial_equity
    curves[:, 1:] = initial_equity * np.cumprod(1.0 + sampled, axis=1)

    final_eq = curves[:, -1]
    running_max = np.maximum.accumulate(curves, axis=1)
    drawdown = (curves - running_max) / running_max * 100.0
    max_drawdown = drawdown.min(axis=1)
    total_return = (final_eq - initial_equity) / initial_equity * 100.0
    ruin_threshold = initial_equity * (1.0 - cfg.mc_ruin_drawdown_pct / 100.0)

    def p(values: np.ndarray, q: int) -> float:
        return round(float(np.percentile(values, q)), 4)

    return {
        "final_equity_p05": p(final_eq, 5),
        "final_equity_p25": p(final_eq, 25),
        "final_equity_p50": p(final_eq, 50),
        "final_equity_p75": p(final_eq, 75),
        "final_equity_p95": p(final_eq, 95),
        "max_drawdown_p05": p(max_drawdown, 5),
        "max_drawdown_p25": p(max_drawdown, 25),
        "max_drawdown_p50": p(max_drawdown, 50),
        "max_drawdown_p75": p(max_drawdown, 75),
        "max_drawdown_p95": p(max_drawdown, 95),
        "total_return_p05": p(total_return, 5),
        "total_return_p25": p(total_return, 25),
        "total_return_p50": p(total_return, 50),
        "total_return_p75": p(total_return, 75),
        "total_return_p95": p(total_return, 95),
        "prob_of_ruin_pct": round(float(np.mean(final_eq < ruin_threshold) * 100.0), 2),
        "prob_profitable_pct": round(float(np.mean(final_eq > initial_equity) * 100.0), 2),
        "worst_final_equity": round(float(final_eq.min()), 2),
        "best_final_equity": round(float(final_eq.max()), 2),
        "worst_max_drawdown_pct": round(float(max_drawdown.min()), 4),
    }


def run_monte_carlo(trades: list[ClosedTrade], initial_equity: float, cfg: RunConfig, seed_offset: int = 0) -> dict | None:
    n_sims = cfg.monte_carlo_simulations
    if not cfg.monte_carlo_enabled or n_sims <= 0 or len(trades) < 2:
        return None

    equity_before = np.array([t.equity_before for t in trades], dtype=np.float64)
    pnl = np.array([t.pnl_eur for t in trades], dtype=np.float64)
    base_frac = pnl / equity_before

    extra_cost = np.array([broker.usd_to_eur(t.units * cfg.mc_extra_slippage_points * cfg.contract_multiplier, cfg) for t in trades])
    slip_frac = (pnl - extra_cost) / equity_before

    rng = np.random.default_rng(cfg.mc_random_seed + seed_offset)
    base = _curve_stats(base_frac, n_sims, initial_equity, rng, "permute", cfg)
    slippage = _curve_stats(slip_frac, n_sims, initial_equity, rng, "permute", cfg)
    sequence = _curve_stats(base_frac, n_sims, initial_equity, rng, "block", cfg)

    out = {"n_simulations": n_sims}
    out.update({f"base_{key}": value for key, value in base.items()})
    out.update({f"slip_{key}": value for key, value in slippage.items()})
    out.update({f"seq_{key}": value for key, value in sequence.items()})
    log.info(
        "Monte-Carlo done sims %d base_prob_ruin %.2f%% seq_prob_ruin %.2f%%",
        n_sims, base["prob_of_ruin_pct"], sequence["prob_of_ruin_pct"],
    )
    return out
