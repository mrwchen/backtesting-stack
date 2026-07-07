"""Entry point: load data, compute signals, run backtest, persist results."""
from __future__ import annotations

import logging

import numpy as np

from .backtest import run_backtest
from .config import Config
from .data_loader import lagged_scores_for_trading_days, load_composite_scores, load_prices
from .db import get_conn
from .persistence import persist_run
from .strategy import positions


def main() -> None:
    cfg = Config.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("runner")
    log.info("run_label=%s symbol=%s window=%s..%s EMA%d/%d stress %s/%s cost %sbps",
             cfg.run_label, cfg.symbol, cfg.start_date, cfg.end_date,
             cfg.ema_fast, cfg.ema_slow, cfg.stress_enter, cfg.stress_exit,
             cfg.cost_bps_per_side)

    conn = get_conn()
    try:
        prices = load_prices(conn, cfg)
        scores = load_composite_scores(conn, cfg)
        lagged = lagged_scores_for_trading_days(prices, scores).to_numpy(dtype=float)

        closes = prices["close"].to_numpy(dtype=float)
        days = list(prices["day"])
        signals = positions(closes, lagged, cfg.ema_fast, cfg.ema_slow,
                            cfg.stress_enter, cfg.stress_exit)
        signals["lagged_score"] = lagged
        signals["close"] = closes

        result = run_backtest(days, closes, signals["position"],
                              cfg.start_date, cfg.cost_bps_per_side)

        start_idx = next(i for i, d in enumerate(days) if d >= cfg.start_date)
        run_id = persist_run(conn, cfg, result, signals, start_idx)

        log.info("=== run_id=%d %s %s..%s ===", run_id, cfg.symbol,
                 result.days[0], result.days[-1])
        log.info("strategy: %+.1f%% (CAGR %+.1f%%), MaxDD %.1f%%, invested %.0f%% of days",
                 result.total_return_pct, result.cagr_pct,
                 result.max_drawdown_pct, result.days_invested_pct)
        log.info("buy&hold: %+.1f%%, MaxDD %.1f%%",
                 result.bh_return_pct, result.bh_max_drawdown_pct)
        log.info("trades: %d (%d winners)", len(result.trades),
                 sum(1 for t in result.trades if (t.gross_return_pct or 0) > 0))
        for t in result.trades:
            log.info("  #%d %s -> %s  %+6.2f%%  (%d days%s)",
                     t.trade_no, t.entry_date, t.exit_date, t.gross_return_pct,
                     t.holding_days, ", open" if t.is_open else "")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
