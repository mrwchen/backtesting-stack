"""Entry point: load data, compute signals, simulate mode, persist results."""
from __future__ import annotations

import logging
import time

import numpy as np

from .config import Config
from .data_loader import (lagged_scores_for_trading_days, load_category_momentum,
                          load_composite_scores, load_prices, load_universe)
from .db import get_conn
from .independent import run_independent_trades
from .persistence import persist_independent_run, persist_portfolio_run
from .portfolio import run_portfolio
from .strategy import stock_positions, stress_gate


def main() -> None:
    cfg = Config.from_env()
    logging.Formatter.converter = time.gmtime
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    log = logging.getLogger("runner")
    log.info("%s run %s window %s..%s EMA%d/%d stress %s/%s "
             "weights deep/mild/pos %s/%s/%s%% max %d pos (%d/cat) cost %sbps",
             cfg.simulation_mode, cfg.run_label, cfg.start_date, cfg.end_date,
             cfg.ema_fast, cfg.ema_slow, cfg.stress_enter, cfg.stress_exit,
             cfg.weight_deep_pct, cfg.weight_mild_pct, cfg.weight_pos_pct,
             cfg.max_positions, cfg.max_per_category, cfg.cost_bps_per_side)

    conn = get_conn()
    try:
        universe = load_universe(conn, cfg)
        symbols = universe["symbol"].tolist()
        categories = dict(zip(universe["symbol"], universe["ibkr_category"]))

        prices = load_prices(conn, cfg, symbols)
        symbols = [s for s in symbols if s in prices.columns]
        prices = prices[symbols]

        scores = load_composite_scores(conn, cfg)
        lagged = lagged_scores_for_trading_days(prices.index, scores)
        stress = stress_gate(lagged, cfg.stress_enter, cfg.stress_exit)

        cat_mom = load_category_momentum(conn, cfg).reindex(prices.index).ffill()

        fresh = prices.notna().to_numpy()
        closes = prices.ffill().to_numpy(dtype=float)
        positions = np.column_stack([
            stock_positions(closes[:, i], stress, cfg.ema_fast, cfg.ema_slow)
            for i in range(len(symbols))
        ])

        # evaluation window: warmup rows only feed the EMAs / momentum
        days_all = list(prices.index)
        start = next(i for i, d in enumerate(days_all) if d >= cfg.start_date)
        days = days_all[start:]
        cat_mom_eval = {c: cat_mom[c].to_numpy(dtype=float)[start:]
                        for c in cat_mom.columns}
        for cat in set(categories.values()) - set(cat_mom_eval):
            cat_mom_eval[cat] = np.full(len(days), np.nan)

        weights = {"deep": cfg.weight_deep_pct,
                   "mild": cfg.weight_mild_pct,
                   "pos": cfg.weight_pos_pct}
        if cfg.simulation_mode == "portfolio":
            result = run_portfolio(
                days=days, symbols=symbols, categories=categories,
                closes=closes[start:], fresh=fresh[start:],
                positions=positions[start:], stress_on=stress[start:],
                cat_momentum=cat_mom_eval, weight_pct_by_tier=weights,
                deep_threshold=cfg.cat_mom_deep_threshold,
                max_positions=cfg.max_positions,
                max_per_category=cfg.max_per_category,
                cost_bps_per_side=cfg.cost_bps_per_side,
            )
            run_id = persist_portfolio_run(conn, cfg, result, len(symbols),
                                           lagged[start:], stress[start:])
            log.info("Run %d portfolio window %s..%s universe %d",
                     run_id, result.days[0], result.days[-1], len(symbols))
            log.info("strategy %+.1f%% CAGR %+.1f%% MaxDD %.1f%% avg exposure %.0f%%",
                     result.total_return_pct, result.cagr_pct,
                     result.max_drawdown_pct, result.avg_gross_exposure_pct)
            log.info("benchmark EW universe %+.1f%% MaxDD %.1f%%",
                     result.bh_return_pct, result.bh_max_drawdown_pct)
        else:
            result = run_independent_trades(
                days=days, symbols=symbols, categories=categories,
                closes=closes[start:], fresh=fresh[start:],
                positions=positions[start:], stress_on=stress[start:],
                cat_momentum=cat_mom_eval, weight_pct_by_tier=weights,
                deep_threshold=cfg.cat_mom_deep_threshold,
            )
            run_id = persist_independent_run(conn, cfg, result, len(symbols))
            log.info("Run %d independent window %s..%s universe %d",
                     run_id, result.days[0], result.days[-1], len(symbols))
            log.info("independent trades avg %+.2f%% median %+.2f%% win %.1f%% avg hold %.1f days",
                     result.avg_trade_return_pct or 0.0,
                     result.median_trade_return_pct or 0.0,
                     result.win_rate_pct or 0.0,
                     result.avg_holding_days or 0.0)

        winners = sum(1 for t in result.trades if (t.gross_return_pct or 0) > 0)
        log.info("Trades %d winners %d still open %d", len(result.trades),
                 winners, sum(1 for t in result.trades if t.is_open))
        by_tier: dict[str, list[float]] = {}
        for t in result.trades:
            by_tier.setdefault(t.tier, []).append(t.gross_return_pct or 0.0)
        for tier, rets in sorted(by_tier.items()):
            log.info("Tier %-4s count %3d avg %+6.2f%% win %4.1f%%", tier, len(rets),
                     float(np.mean(rets)),
                     100.0 * sum(1 for r in rets if r > 0) / len(rets))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
