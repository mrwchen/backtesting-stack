"""Orchestration: load -> build features -> simulate -> summarise -> Monte-Carlo -> persist."""

import logging
import time

from . import config, persistence
from .data import build_features, load_bars
from .db import connect_with_retry
from .layer_risk import run_monte_carlo, summarize_trades
from .simulation import run_simulation

log = logging.getLogger(__name__)


def main() -> None:
    started = time.time()
    cfg = config.active_run_config()
    log.info(
        "Scalping backtest start symbol=%s bar_size=%s price=%s vol=%s decision=%s account=%s",
        cfg.symbol, cfg.bar_size, cfg.price_model, cfg.vol_model, cfg.decision_model, cfg.account_profile,
    )

    conn = connect_with_retry()
    try:
        bars = load_bars(conn)
        features = build_features(bars)
        data_start_ts = bars["ts"].iloc[0].to_pydatetime()
        data_end_ts = bars["ts"].iloc[-1].to_pydatetime()

        run_id = persistence.create_run(conn, cfg, data_start_ts, data_end_ts, len(bars))

        result = run_simulation(features)
        summary = summarize_trades(result.trades, result.initial_equity, result.final_equity)
        mc = run_monte_carlo(result.trades, result.initial_equity)

        persistence.write_trades(conn, run_id, result.trades)
        persistence.write_monte_carlo(conn, run_id, mc)
        persistence.update_run_summary(
            conn, run_id, summary,
            run_duration_seconds=time.time() - started,
            bars_simulated=result.bars_simulated,
            ruined=result.ruined,
        )

        log.info(
            "Run %d complete: trades=%d final_equity=%.2f return=%.2f%% max_dd=%.2f%% win_rate=%.2f%%",
            run_id, summary["total_trades"], result.final_equity,
            summary["total_return_pct"], summary["max_drawdown_pct"], summary["win_rate_pct"],
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
