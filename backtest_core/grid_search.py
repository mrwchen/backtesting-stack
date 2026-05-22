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

    required_keys = {
        "long_tp1_pct",
        "long_tp2_pct",
        "short_tp1_pct",
        "short_tp2_pct",
        "long_max_hold_days",
        "short_max_hold_days",
        "tp1_close_ratio",
    }
    if any(required_keys - set(result.keys()) for result in results):
        ranked_generic = sorted(
            results,
            key=lambda r: (r["profit_factor"] or 0.0, r["total_return_pct"]),
            reverse=True,
        )
        log.info("Grid search results (generic summary, sorted by profit factor):")
        for r in ranked_generic:
            log.info(
                "Run %d trades %d win rate %.1f%% return %.2f%% drawdown %.2f%% profit factor %s",
                r["run_id"],
                r["total_trades"],
                r["win_rate_pct"],
                r["total_return_pct"],
                r["max_drawdown_pct"],
                f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "N/A",
            )
        return

    ranked = sorted(
        results,
        key=lambda r: (r["profit_factor"] or 0.0, r["total_return_pct"]),
        reverse=True,
    )

    header = (
        f"{'run_id':>7}  {'ltp1':>5}  {'ltp2':>5}  {'stp1':>5}  {'stp2':>5}  "
        f"{'lmhd':>4}  {'smhd':>4}  {'tcr':>4}  {'trades':>6}  {'wr%':>5}  "
        f"{'ret%':>7}  {'dd%':>6}  {'PF':>5}"
    )
    sep = "-" * len(header)
    log.info("Grid search results (sorted by profit factor):\n%s\n%s", header, sep)
    for r in ranked:
        pf = f"{r['profit_factor']:.3f}" if r["profit_factor"] is not None else "  N/A"
        log.info(
            "%7d  %5.3f  %5.3f  %5.3f  %5.3f  %4.1f  %4.1f  %4.2f  %6d  %5.1f  %7.2f  %6.2f  %5s",
            r["run_id"],
            r["long_tp1_pct"], r["long_tp2_pct"],
            r["short_tp1_pct"], r["short_tp2_pct"],
            r["long_max_hold_days"], r["short_max_hold_days"], r["tp1_close_ratio"],
            r["total_trades"],
            r["win_rate_pct"],
            r["total_return_pct"],
            r["max_drawdown_pct"],
            pf,
        )
    best = ranked[0]
    log.info(
        "Best combination run %d PF %s return %.2f%% drawdown %.2f%% "
        "long TP1 %.3f long TP2 %.3f short TP1 %.3f short TP2 %.3f long max hold %.1f short max hold %.1f TP1 close ratio %.2f",
        best["run_id"],
        f"{best['profit_factor']:.3f}" if best["profit_factor"] else "N/A",
        best["total_return_pct"], best["max_drawdown_pct"],
        best["long_tp1_pct"], best["long_tp2_pct"],
        best["short_tp1_pct"], best["short_tp2_pct"],
        best["long_max_hold_days"], best["short_max_hold_days"], best["tp1_close_ratio"],
    )
