"""Write run summary, trades and optional daily equity to result tables."""
from __future__ import annotations

import logging

import numpy as np
from psycopg2 import sql

from .config import Config
from .independent import IndependentResult
from .portfolio import PortfolioResult, StockTrade

log = logging.getLogger(__name__)


def _rounded(value: float | None, ndigits: int) -> float | None:
    if value is None:
        return None
    return round(float(value), ndigits)


def _trade_stats(trades: list[StockTrade]) -> dict[str, float | int | None]:
    returns = np.array(
        [t.gross_return_pct for t in trades if t.gross_return_pct is not None],
        dtype=float,
    )
    holding_days = np.array(
        [t.holding_days for t in trades if t.holding_days is not None],
        dtype=float,
    )
    if len(returns) == 0:
        return {
            "n_trades": len(trades),
            "n_winning_trades": 0,
            "n_open_trades": sum(1 for t in trades if t.is_open),
            "win_rate_pct": None,
            "avg_trade_return_pct": None,
            "median_trade_return_pct": None,
            "avg_holding_days": None,
        }
    return {
        "n_trades": len(trades),
        "n_winning_trades": int(np.sum(returns > 0)),
        "n_open_trades": sum(1 for t in trades if t.is_open),
        "win_rate_pct": float(np.mean(returns > 0) * 100.0),
        "avg_trade_return_pct": float(np.mean(returns)),
        "median_trade_return_pct": float(np.median(returns)),
        "avg_holding_days": float(np.mean(holding_days)) if len(holding_days) else None,
    }


def _insert_run(conn, cfg: Config, universe_size: int, days,
                trades: list[StockTrade],
                portfolio_result: PortfolioResult | None = None) -> int:
    stats = _trade_stats(trades)
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("""
            INSERT INTO {} (
                simulation_mode, run_label, start_date, end_date, universe_size,
                top_n_per_category, min_market_cap_usd,
                ema_fast, ema_slow, stress_enter, stress_exit,
                cat_mom_window, cat_mom_deep_threshold,
                weight_deep_pct, weight_mild_pct, weight_pos_pct,
                max_positions, max_per_category, entry_confirm_days,
                trim_above_pct, trim_target_pct,
                sl_pct, time_stop_days, time_stop_min_ret_pct,
                reentry_cooldown_days, cost_bps_per_side,
                total_return_pct, bh_return_pct, max_drawdown_pct,
                bh_max_drawdown_pct, cagr_pct,
                n_trades, n_winning_trades, n_open_trades, win_rate_pct,
                avg_trade_return_pct, median_trade_return_pct, avg_holding_days,
                avg_gross_exposure_pct
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                      %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING run_id
            """).format(sql.Identifier(cfg.runs_table)),
            (
                cfg.simulation_mode, cfg.run_label, days[0], days[-1], universe_size,
                cfg.top_n_per_category, cfg.min_market_cap_usd,
                cfg.ema_fast, cfg.ema_slow, cfg.stress_enter, cfg.stress_exit,
                cfg.cat_mom_window, cfg.cat_mom_deep_threshold,
                cfg.weight_deep_pct, cfg.weight_mild_pct, cfg.weight_pos_pct,
                cfg.max_positions, cfg.max_per_category, cfg.entry_confirm_days,
                cfg.trim_above_pct, cfg.trim_target_pct,
                cfg.sl_pct, cfg.time_stop_days, cfg.time_stop_min_ret_pct,
                cfg.reentry_cooldown_days, cfg.cost_bps_per_side,
                _rounded(portfolio_result.total_return_pct, 2)
                if portfolio_result else None,
                _rounded(portfolio_result.bh_return_pct, 2)
                if portfolio_result else None,
                _rounded(portfolio_result.max_drawdown_pct, 2)
                if portfolio_result else None,
                _rounded(portfolio_result.bh_max_drawdown_pct, 2)
                if portfolio_result else None,
                _rounded(portfolio_result.cagr_pct, 2)
                if portfolio_result else None,
                stats["n_trades"],
                stats["n_winning_trades"],
                stats["n_open_trades"],
                _rounded(stats["win_rate_pct"], 1),
                _rounded(stats["avg_trade_return_pct"], 2),
                _rounded(stats["median_trade_return_pct"], 2),
                _rounded(stats["avg_holding_days"], 1),
                _rounded(portfolio_result.avg_gross_exposure_pct, 1)
                if portfolio_result else None,
            ),
        )
        return cur.fetchone()[0]


def _insert_trades(conn, cfg: Config, run_id: int,
                   trades: list[StockTrade]) -> None:
    if not trades:
        return
    with conn.cursor() as cur:
        cur.executemany(
            sql.SQL("""
            INSERT INTO {} (
                run_id, trade_no, symbol, ibkr_category, entry_date, exit_date,
                entry_price, exit_price, gross_return_pct, holding_days,
                target_weight_pct, effective_weight_pct, sizing_tier,
                cat_mom_at_entry_pct, is_open
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """).format(sql.Identifier(cfg.trades_table)),
            [
                (run_id, t.trade_no, t.symbol, t.category, t.entry_date, t.exit_date,
                 round(t.entry_price, 6),
                 round(t.exit_price, 6) if t.exit_price is not None else None,
                 round(t.gross_return_pct, 2) if t.gross_return_pct is not None else None,
                 t.holding_days, round(t.target_weight_pct, 2),
                 round(t.effective_weight_pct, 2), t.tier,
                 round(t.cat_mom_at_entry * 100, 2) if t.cat_mom_at_entry is not None else None,
                 t.is_open)
                for t in trades
            ],
        )


def persist_portfolio_run(conn, cfg: Config, result: PortfolioResult,
                          universe_size: int, lagged_scores: np.ndarray,
                          stress_on: np.ndarray) -> int:
    run_id = _insert_run(conn, cfg, universe_size, result.days, result.trades,
                         portfolio_result=result)
    _insert_trades(conn, cfg, run_id, result.trades)

    with conn.cursor() as cur:
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
    log.info("Persisted portfolio run %d with %d trades and %d daily rows",
             run_id, len(result.trades), len(result.days))
    return run_id


def persist_independent_run(conn, cfg: Config, result: IndependentResult,
                            universe_size: int) -> int:
    run_id = _insert_run(conn, cfg, universe_size, result.days, result.trades)
    _insert_trades(conn, cfg, run_id, result.trades)
    conn.commit()
    log.info("Persisted independent run %d with %d trades",
             run_id, len(result.trades))
    return run_id
