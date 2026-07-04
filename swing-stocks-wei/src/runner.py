"""Entry point for the Wei 52-week-high pullback backtester."""
from __future__ import annotations

import logging

import pandas as pd

from . import data_loader, db, persistence, strategy
from .config import Config
from .logging_utils import configure_logging
from .simulator import simulate

log = logging.getLogger("runner")


def main() -> None:
    cfg = Config.from_env()
    configure_logging(cfg.log_level)
    log.info("start run_label %s start %s end %s", cfg.run_label, cfg.start_date, cfg.end_date)

    conn = db.get_conn()
    try:
        db.validate_tables(conn, data_loader.SOURCE_TABLES + persistence.RESULT_TABLES)
        universe = data_loader.load_universe(conn, cfg)
        prices = data_loader.load_prices(conn, cfg)
        fundamentals = data_loader.load_fundamentals(conn, cfg)
        prices = prices[prices["symbol"].isin(set(universe["symbol"]))].copy()
        prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)
        if prices.empty:
            raise SystemExit("no source price rows after universe filter")

        window_dates = pd.to_datetime(prices["date"])
        start_mask = window_dates.dt.date >= pd.Timestamp(cfg.start_date).date()
        if not start_mask.any():
            raise SystemExit(f"no price data on or after START_DATE={cfg.start_date}")
        start = window_dates[start_mask].min().date()
        requested_end = data_loader.effective_end(cfg)
        available_end = window_dates.dt.date.max()
        end = min(requested_end, available_end)
        if end < start:
            raise SystemExit(f"END_DATE={end} is before first available backtest date {start}")

        taxonomy_ok = universe["ibkr_industry"].notna() & universe["ibkr_category"].notna()
        log.info(
            "loaded %d daily bars for %d IBKR-backed USD equity symbols; taxonomy for %d of %d symbols",
            len(prices),
            prices["symbol"].nunique(),
            int(taxonomy_ok.sum()),
            len(universe),
        )

        signals = strategy.compute_signals(prices, universe, fundamentals, cfg, start, end)
        run_id = persistence.create_run(conn, cfg, start, end, signal_count=len(signals))
        persistence.write_signals(conn, run_id, signals)
        result = simulate(prices, signals, cfg, start, end)
        persistence.update_run_result(conn, run_id, result.metrics, trade_count=len(result.trades))
        persistence.write_trades(conn, run_id, result.trades)
        persistence.write_equity(conn, run_id, result.equity)
        log.info(
            "run %d done signals %d trades %d final_equity %.2f total_return %.4f",
            run_id,
            len(signals),
            len(result.trades),
            result.metrics["final_equity"],
            result.metrics["total_return"],
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
