"""Grid-search orchestration for model-specific parameter sweeps."""

import logging
from typing import Any

import psycopg2

from . import runtime
from .config import *
from .market_data import clear_market_data_caches
from .model_loader import get_model_module
from .simulation import run_backtest

log = logging.getLogger(__name__)

def run_grid_search(conn: psycopg2.extensions.connection, base_cfg: Any) -> list[dict]:
    model = get_model_module()
    if not hasattr(model, "iter_grid_search_configs"):
        raise RuntimeError(
            f"Backtesting model {runtime.CURRENT_MODEL_FILE} does not define iter_grid_search_configs(). "
            "Either disable GRID_SEARCH_ENABLED or add the grid hook to the model file."
        )

    grid_items = list(model.iter_grid_search_configs(
        base_cfg=base_cfg,
        parse_grid_vals=_parse_grid_vals,
        parse_hold_grid_vals=_parse_hold_grid_vals,
    ))

    total = len(grid_items)
    log.info("Grid search model %s combinations %d", runtime.CURRENT_MODEL_FILE, total)

    results: list[dict] = []
    for i, item in enumerate(grid_items, 1):
        cfg = item["config"]
        run_notes = item.get("notes", f"grid model={runtime.CURRENT_MODEL_FILE} idx={i}")
        log.info("Grid %d/%d — %s", i, total, run_notes)
        try:
            _, summary = run_backtest(conn, cfg, notes=run_notes)
        finally:
            clear_market_data_caches(f"grid {i}/{total}")
        summary.update(item.get("summary", {}))
        results.append(summary)

    return results


def _print_grid_summary(results: list[dict]) -> None:
    if not results:
        log.info("Grid search produced no results.")
        return

    model = get_model_module()
    if hasattr(model, "log_grid_summary"):
        model.log_grid_summary(log, results)
        return

    ranked = sorted(
        results,
        key=lambda r: (r["profit_factor"] or 0.0, r["total_return_pct"]),
        reverse=True,
    )
    log.info("Grid search results sorted by profit factor")
    for r in ranked:
        log.info(
            "Run %d trades %d win rate %.1f%% return %.2f%% drawdown %.2f%% profit factor %s",
            r["run_id"],
            r["total_trades"],
            r["win_rate_pct"],
            r["total_return_pct"],
            r["max_drawdown_pct"],
            f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "N/A",
        )
    best = ranked[0]
    log.info(
        "Best combination run %d PF %s return %.2f%% drawdown %.2f%%",
        best["run_id"],
        f"{best['profit_factor']:.3f}" if best["profit_factor"] else "N/A",
        best["total_return_pct"], best["max_drawdown_pct"],
    )
