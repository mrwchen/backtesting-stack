"""Write run summary, trades and daily equity to the backtest_wei_* tables."""
from __future__ import annotations

import logging

import numpy as np
from psycopg2 import sql

from .backtest import BacktestResult
from .config import Config

log = logging.getLogger(__name__)


def persist_run(conn, cfg: Config, result: BacktestResult, signals: dict, start_idx: int) -> int:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("""
            INSERT INTO {} (
                run_label, symbol, start_date, end_date,
                ema_fast, ema_slow, stress_enter, stress_exit, cost_bps_per_side,
                total_return_pct, bh_return_pct, max_drawdown_pct, bh_max_drawdown_pct,
                cagr_pct, n_trades, n_winning_trades, days_invested_pct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING run_id
            """).format(sql.Identifier(cfg.runs_table)),
            (
                cfg.run_label, cfg.symbol, result.days[0], result.days[-1],
                cfg.ema_fast, cfg.ema_slow, cfg.stress_enter, cfg.stress_exit,
                cfg.cost_bps_per_side,
                round(result.total_return_pct, 2), round(result.bh_return_pct, 2),
                round(result.max_drawdown_pct, 2), round(result.bh_max_drawdown_pct, 2),
                round(result.cagr_pct, 2), len(result.trades),
                sum(1 for t in result.trades if (t.gross_return_pct or 0) > 0),
                round(result.days_invested_pct, 1),
            ),
        )
        run_id = cur.fetchone()[0]

        cur.executemany(
            sql.SQL("""
            INSERT INTO {} (
                run_id, trade_no, entry_date, exit_date, entry_price, exit_price,
                gross_return_pct, holding_days, is_open
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """).format(sql.Identifier(cfg.trades_table)),
            [
                (run_id, t.trade_no, t.entry_date, t.exit_date,
                 round(t.entry_price, 6), round(t.exit_price, 6) if t.exit_price else None,
                 round(t.gross_return_pct, 2) if t.gross_return_pct is not None else None,
                 t.holding_days, t.is_open)
                for t in result.trades
            ],
        )

        ema_f = signals["ema_fast"][start_idx:]
        ema_s = signals["ema_slow"][start_idx:]
        stress = signals["stress_on"][start_idx:]
        pos = signals["position"][start_idx:]
        scores = signals["lagged_score"][start_idx:]
        closes = signals["close"][start_idx:]
        cur.executemany(
            sql.SQL("""
            INSERT INTO {} (
                day, run_id, close, ema_fast_value, ema_slow_value,
                composite_score, stress_on, position, equity, bh_equity
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """).format(sql.Identifier(cfg.equity_table)),
            [
                (result.days[i], run_id, round(float(closes[i]), 6),
                 round(float(ema_f[i]), 6), round(float(ema_s[i]), 6),
                 None if np.isnan(scores[i]) else round(float(scores[i]), 1),
                 bool(stress[i]), int(pos[i]),
                 round(float(result.equity[i]), 8), round(float(result.bh_equity[i]), 8))
                for i in range(len(result.days))
            ],
        )
    conn.commit()
    log.info("persisted run_id=%d (%d trades, %d daily rows)",
             run_id, len(result.trades), len(result.days))
    return run_id
