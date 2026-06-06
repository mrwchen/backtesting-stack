"""Monte Carlo analytics for completed backtest trade sequences."""

import logging

import numpy as np
import psycopg2

from .config import _result_table
from .entities import ClosedTrade

log = logging.getLogger(__name__)

def run_monte_carlo(
    conn: psycopg2.extensions.connection,
    run_id: int,
    closed_trades: list[ClosedTrade],
    initial_equity: float,
    n_simulations: int = 2000,
) -> None:
    if n_simulations <= 0 or len(closed_trades) < 2:
        return

    # Use equity fraction (pnl / equity_before) so that compounding is preserved
    # when trades are reshuffled — correct for a fixed-%-risk strategy.
    fractions = np.array([t.pnl_usd / t.position.equity_before for t in closed_trades], dtype=np.float64)

    rng = np.random.default_rng()
    shuffled = np.tile(fractions, (n_simulations, 1))
    rng.permuted(shuffled, axis=1, out=shuffled)

    equity_curves = np.empty((n_simulations, len(fractions) + 1), dtype=np.float64)
    equity_curves[:, 0] = initial_equity
    equity_curves[:, 1:] = initial_equity * np.cumprod(1.0 + shuffled, axis=1)

    final_equities = equity_curves[:, -1]
    running_max = np.maximum.accumulate(equity_curves, axis=1)
    drawdown_pct = (equity_curves - running_max) / running_max * 100.0
    max_drawdowns = drawdown_pct.min(axis=1)
    total_returns = (final_equities - initial_equity) / initial_equity * 100.0

    def p(arr: np.ndarray, pct: int) -> float:
        return round(float(np.percentile(arr, pct)), 4)

    with conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {_result_table("backtest_monte_carlo")} (
                run_id, n_simulations,
                final_equity_p05, final_equity_p25, final_equity_p50, final_equity_p75, final_equity_p95,
                max_drawdown_p05, max_drawdown_p25, max_drawdown_p50, max_drawdown_p75, max_drawdown_p95,
                total_return_p05, total_return_p25, total_return_p50, total_return_p75, total_return_p95,
                prob_of_ruin_pct, prob_profitable_pct,
                worst_final_equity, worst_max_drawdown_pct, best_final_equity
            ) VALUES (
                %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (run_id) DO UPDATE SET
                n_simulations          = EXCLUDED.n_simulations,
                final_equity_p05       = EXCLUDED.final_equity_p05,
                final_equity_p25       = EXCLUDED.final_equity_p25,
                final_equity_p50       = EXCLUDED.final_equity_p50,
                final_equity_p75       = EXCLUDED.final_equity_p75,
                final_equity_p95       = EXCLUDED.final_equity_p95,
                max_drawdown_p05       = EXCLUDED.max_drawdown_p05,
                max_drawdown_p25       = EXCLUDED.max_drawdown_p25,
                max_drawdown_p50       = EXCLUDED.max_drawdown_p50,
                max_drawdown_p75       = EXCLUDED.max_drawdown_p75,
                max_drawdown_p95       = EXCLUDED.max_drawdown_p95,
                total_return_p05       = EXCLUDED.total_return_p05,
                total_return_p25       = EXCLUDED.total_return_p25,
                total_return_p50       = EXCLUDED.total_return_p50,
                total_return_p75       = EXCLUDED.total_return_p75,
                total_return_p95       = EXCLUDED.total_return_p95,
                prob_of_ruin_pct       = EXCLUDED.prob_of_ruin_pct,
                prob_profitable_pct    = EXCLUDED.prob_profitable_pct,
                worst_final_equity     = EXCLUDED.worst_final_equity,
                worst_max_drawdown_pct = EXCLUDED.worst_max_drawdown_pct,
                best_final_equity      = EXCLUDED.best_final_equity,
                created_at             = NOW()
            """,
            (
                run_id, n_simulations,
                p(final_equities, 5),  p(final_equities, 25), p(final_equities, 50),
                p(final_equities, 75), p(final_equities, 95),
                p(max_drawdowns, 5),   p(max_drawdowns, 25),  p(max_drawdowns, 50),
                p(max_drawdowns, 75),  p(max_drawdowns, 95),
                p(total_returns, 5),   p(total_returns, 25),  p(total_returns, 50),
                p(total_returns, 75),  p(total_returns, 95),
                round(float(np.mean(final_equities < initial_equity * 0.5) * 100), 2),
                round(float(np.mean(final_equities > initial_equity) * 100), 2),
                round(float(final_equities.min()), 2),
                round(float(max_drawdowns.min()), 4),
                round(float(final_equities.max()), 2),
            ),
        )
    conn.commit()
    log.info(
        "Monte Carlo run %d simulations %d return p50 %.1f%% return p05 %.1f%% drawdown p05 %.1f%% ruin %.1f%% profitable %.1f%%",
        run_id, n_simulations,
        p(total_returns, 50), p(total_returns, 5), p(max_drawdowns, 5),
        float(np.mean(final_equities < initial_equity * 0.5) * 100),
        float(np.mean(final_equities > initial_equity) * 100),
    )
