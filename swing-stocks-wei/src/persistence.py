"""Write run summary, trades and daily equity to the backtest_wei_stocks_* tables."""
from __future__ import annotations

import logging

import numpy as np
from psycopg2 import sql

from .config import Config
from .portfolio import PortfolioResult

log = logging.getLogger(__name__)


def persist_run(conn, cfg: Config, result: PortfolioResult, universe_size: int,
                lagged_scores: np.ndarray, stress_on: np.ndarray) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("""
            INSERT INTO {} (
                run_label, start_date, end_date, universe_size,
                top_n_per_category, min_market_cap_usd,
                ema_fast, ema_slow, stress_enter, stress_exit,
                cat_mom_window, cat_mom_deep_threshold,
                weight_deep_pct, weight_mild_pct, weight_pos_pct,
                max_positions, max_per_category, cost_bps_per_side,
                total_return_pct, bh_return_pct, max_drawdown_pct,
                bh_max_drawdown_pct, cagr_pct, n_trades, n_winning_trades,
                avg_gross_exposure_pct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING run_id
            """).format(sql.Identifier(cfg.runs_table)),
            (
                cfg.run_label, result.days[0], result.days[-1], universe_size,
                cfg.top_n_per_category, cfg.min_market_cap_usd,
                cfg.ema_fast, cfg.ema_slow, cfg.stress_enter, cfg.stress_exit,
                cfg.cat_mom_window, cfg.cat_mom_deep_threshold,
                cfg.weight_deep_pct, cfg.weight_mild_pct, cfg.weight_pos_pct,
                cfg.max_positions, cfg.max_per_category, cfg.cost_bps_per_side,
                round(result.total_return_pct, 2), round(result.bh_return_pct, 2),
                round(result.max_drawdown_pct, 2), round(result.bh_max_drawdown_pct, 2),
                round(result.cagr_pct, 2), len(result.trades),
                sum(1 for t in result.trades if (t.gross_return_pct or 0) > 0),
                round(result.avg_gross_exposure_pct, 1),
            ),
        )
        run_id = cur.fetchone()[0]

        cur.executemany(
            sql.SQL("""
            INSERT INTO {} (
                run_id, trade_no, symbol, ibkr_category, entry_date, exit_date,
                entry_price, exit_price, gross_return_pct, holding_days,
                weight_pct, sizing_tier, cat_mom_at_entry_pct, is_open
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """).format(sql.Identifier(cfg.trades_table)),
            [
                (run_id, t.trade_no, t.symbol, t.category, t.entry_date, t.exit_date,
                 round(t.entry_price, 6),
                 round(t.exit_price, 6) if t.exit_price is not None else None,
                 round(t.gross_return_pct, 2) if t.gross_return_pct is not None else None,
                 t.holding_days, round(t.weight_pct, 2), t.tier,
                 round(t.cat_mom_at_entry * 100, 2) if t.cat_mom_at_entry is not None else None,
                 t.is_open)
                for t in result.trades
            ],
        )

        cur.executemany(
            sql.SQL("""
            INSERT INTO {} (
                day, run_id, equity, bh_equity, n_positions,
                gross_exposure_pct, composite_score, stress_on
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """).format(sql.Identifier(cfg.equity_table)),
            [
                (result.days[i], run_id,
                 round(float(result.equity[i]), 8), round(float(result.bh_equity[i]), 8),
                 int(result.n_positions[i]), round(float(result.gross_exposure_pct[i]), 1),
                 None if np.isnan(lagged_scores[i]) else round(float(lagged_scores[i]), 1),
                 bool(stress_on[i]))
                for i in range(len(result.days))
            ],
        )
    conn.commit()
    log.info("persisted run_id=%d (%d trades, %d daily rows)",
             run_id, len(result.trades), len(result.days))
    return run_id
