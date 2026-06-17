"""Two-stage walk-forward optimizer for the NAS100 hit-frequency median model."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import parameters, persistence
from .config import OptimizerConfig, RunConfig, apply_parameter_values
from .entities import ClosedTrade, SimulationResult
from .profile import rolling_profile_levels
from .risk import run_monte_carlo, summarize_trades
from .simulation import attach_profile_to_ticks, run_simulation

log = logging.getLogger(__name__)

_WORKER_BASE_CFG: RunConfig | None = None
_WORKER_WINDOW: WindowData | None = None


@dataclass(frozen=True)
class FoldSpec:
    fold_index: int
    train_start: object
    train_end: object
    test_start: object
    test_end: object


@dataclass
class WindowData:
    ticks: pd.DataFrame
    bars: pd.DataFrame
    trade_start: object
    trade_end: object
    profile_cache_size: int
    profile_cache: OrderedDict[tuple, pd.DataFrame] = field(default_factory=OrderedDict)


@dataclass
class Evaluation:
    stage: str
    fold_index: int
    window_role: str
    values: dict[str, int | float]
    parameter_hash: str
    parameter_label: str
    window_start: object
    window_end: object
    ticks_simulated: int
    bars_total: int
    signals_total: int
    long_signals: int
    short_signals: int
    rejected_missing_band: int
    rejected_band_too_narrow: int
    rejected_stop_too_small: int
    rejected_stop_too_large: int
    skipped_no_size: int
    ruined: bool
    summary: dict
    score: float
    trades: list[ClosedTrade] = field(default_factory=list)


@dataclass
class StageResult:
    stage: str
    candidates: list[dict[str, int | float]]
    aggregates: list[dict]
    oos_evals: list[Evaluation]
    mc_by_hash: dict[str, dict]
    parameter_ids: dict[str, int] = field(default_factory=dict)


def run_walk_forward_optimizer(
    conn,
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    started: float,
) -> None:
    folds = build_folds(ticks, opt_cfg)
    data_start_ts = ticks["tick_time"].iloc[0].to_pydatetime()
    data_end_ts = ticks["tick_time"].iloc[-1].to_pydatetime()
    run_id = persistence.create_run(
        conn,
        base_cfg,
        opt_cfg,
        mode="walk_forward",
        data_start_ts=data_start_ts,
        data_end_ts=data_end_ts,
        ticks_loaded=len(ticks),
        bars_built=len(bars),
        folds_built=len(folds),
    )

    grid = parameters.load_grid(opt_cfg.parameter_grid_path)
    stage1_candidates = parameters.build_stage1_candidates(grid, opt_cfg.stage1_max_parameter_sets)
    stage1_candidates = screen_stage1_candidates(stage1_candidates, folds, base_cfg, opt_cfg, ticks, bars)
    log.info("Stage 1 candidates %d folds %d", len(stage1_candidates), len(folds))
    stage1 = run_stage(conn, run_id, "stage1", stage1_candidates, folds, base_cfg, opt_cfg, ticks, bars)
    persistence.insert_monte_carlo(conn, stage1.mc_by_hash, stage1.parameter_ids)

    final_stage = stage1
    stage2_count = 0
    if opt_cfg.stage2_enabled:
        seed_values = [
            item["values"]
            for item in sorted(stage1.aggregates, key=lambda row: row["score"], reverse=True)
            if item["oos_folds"] > 0
        ][: opt_cfg.stage2_seed_top_n]
        previous_hashes = {parameters.parameter_hash(candidate) for candidate in stage1_candidates}
        stage2_candidates = parameters.build_stage2_candidates(
            seed_values,
            grid,
            previous_hashes,
            opt_cfg.stage2_max_parameter_sets,
        )
        stage2_count = len(stage2_candidates)
        if stage2_candidates:
            log.info("Stage 2 candidates %d seeds %d folds %d", len(stage2_candidates), len(seed_values), len(folds))
            stage2 = run_stage(conn, run_id, "stage2", stage2_candidates, folds, base_cfg, opt_cfg, ticks, bars)
            persistence.insert_monte_carlo(conn, stage2.mc_by_hash, stage2.parameter_ids)
            final_stage = stage2
        else:
            log.info("Stage 2 skipped no new candidates")

    persist_top_trades(conn, run_id, final_stage, opt_cfg)
    best = max((row for row in final_stage.aggregates if row["oos_folds"] > 0), key=lambda row: row["score"], default=None)
    persistence.update_run_complete(
        conn,
        run_id,
        status="complete",
        run_duration_seconds=time.time() - started,
        stage1_parameter_sets=len(stage1_candidates),
        stage2_parameter_sets=stage2_count,
        best_parameter_set_id=final_stage.parameter_ids.get(best["parameter_hash"]) if best else None,
        best_score=best["score"] if best else None,
    )
    if best:
        log.info(
            "Walk-forward complete best rank %d score %.4f oos_trades %d oos_return %.4f profit_factor %.4f max_drawdown %.4f",
            best["stage_rank"],
            best["score"],
            best["oos_total_trades"],
            best["oos_total_return_pct"],
            best["oos_profit_factor"] if best["oos_profit_factor"] is not None else 0.0,
            best["oos_max_drawdown_pct"],
        )


def run_single_backtest(
    conn,
    cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    started: float,
) -> None:
    data_start_ts = ticks["tick_time"].iloc[0].to_pydatetime()
    data_end_ts = ticks["tick_time"].iloc[-1].to_pydatetime()
    run_id = persistence.create_run(
        conn,
        cfg,
        opt_cfg,
        mode="single",
        data_start_ts=data_start_ts,
        data_end_ts=data_end_ts,
        ticks_loaded=len(ticks),
        bars_built=len(bars),
        folds_built=1,
    )
    result = run_simulation(ticks, bars, cfg, log_result=True)
    summary = summarize_trades(result.trades, cfg.initial_equity, result.final_equity)
    values = parameters.values_from_config(cfg)
    evaluation = Evaluation(
        stage="single",
        fold_index=1,
        window_role="full",
        values=values,
        parameter_hash=parameters.parameter_hash(values),
        parameter_label=parameters.parameter_label(values),
        window_start=data_start_ts,
        window_end=data_end_ts,
        ticks_simulated=result.ticks_simulated,
        bars_total=result.bars_total,
        signals_total=result.signals_total,
        long_signals=result.long_signals,
        short_signals=result.short_signals,
        rejected_missing_band=result.rejected_signals_missing_band,
        rejected_band_too_narrow=result.rejected_signals_band_too_narrow,
        rejected_stop_too_small=result.rejected_signals_stop_too_small,
        rejected_stop_too_large=result.rejected_signals_stop_too_large,
        skipped_no_size=result.skipped_signals_no_size,
        ruined=result.ruined,
        summary=summary,
        score=score_evaluation(summary, result),
        trades=result.trades,
    )
    train_by_hash = {evaluation.parameter_hash: []}
    oos_by_hash = {evaluation.parameter_hash: [evaluation]}
    aggregates = build_aggregates("single", [values], train_by_hash, oos_by_hash, 1, opt_cfg)
    mc_by_hash = score_monte_carlo(aggregates, oos_by_hash, cfg, opt_cfg)
    apply_final_scores(aggregates, mc_by_hash, opt_cfg)
    aggregates[0]["stage_rank"] = 1
    parameter_ids = persistence.insert_parameter_sets(conn, run_id, aggregates)
    persistence.insert_fold_results(conn, run_id, [evaluation], parameter_ids)
    persistence.insert_monte_carlo(conn, mc_by_hash, parameter_ids)
    parameter_set_id = parameter_ids[evaluation.parameter_hash]
    persistence.insert_trade_rows(
        conn,
        persistence.trade_rows(parameter_set_id, "single", 1, "full", evaluation.trades),
    )
    persistence.mark_top_trade_sets(conn, run_id, [parameter_set_id])
    persistence.update_run_complete(
        conn,
        run_id,
        status="complete",
        run_duration_seconds=time.time() - started,
        stage1_parameter_sets=1,
        stage2_parameter_sets=0,
        best_parameter_set_id=parameter_set_id,
        best_score=aggregates[0]["score"],
    )
    log.info(
        "Single run complete score %.4f trades %d return %.4f profit_factor %.4f max_drawdown %.4f",
        aggregates[0]["score"],
        aggregates[0]["oos_total_trades"],
        aggregates[0]["oos_total_return_pct"],
        aggregates[0]["oos_profit_factor"] if aggregates[0]["oos_profit_factor"] is not None else 0.0,
        aggregates[0]["oos_max_drawdown_pct"],
    )


def screen_stage1_candidates(
    candidates: list[dict[str, int | float]],
    folds: list[FoldSpec],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
) -> list[dict[str, int | float]]:
    if not opt_cfg.stage1_screening_enabled:
        return candidates
    if len(candidates) <= opt_cfg.stage1_screening_top_n:
        log.info("Stage 1 screening skipped candidates %d top_n %d", len(candidates), opt_cfg.stage1_screening_top_n)
        return candidates

    active = candidates
    initial_count = len(active)
    rounds = max(1, opt_cfg.stage1_screening_rounds)
    for round_no in range(1, rounds + 1):
        if len(active) <= opt_cfg.stage1_screening_top_n:
            break
        keep_n = _screening_keep_count(len(active), opt_cfg.stage1_screening_top_n, rounds - round_no + 1)
        train_days = min(opt_cfg.train_days, opt_cfg.stage1_screening_train_days * round_no)
        fold = _shortened_train_fold(folds[(round_no - 1) % len(folds)], train_days)
        log.info(
            "Stage 1 screening round %d candidates %d keep %d train_days %d fold %d",
            round_no,
            len(active),
            keep_n,
            train_days,
            fold.fold_index,
        )
        evaluations = evaluate_many(
            f"screen{round_no}",
            active,
            fold,
            "train",
            base_cfg,
            opt_cfg,
            ticks,
            bars,
            keep_trades=False,
        )
        active = [item.values for item in sorted(evaluations, key=lambda evaluation: evaluation.score, reverse=True)[:keep_n]]

    log.info("Stage 1 screening complete initial %d selected %d", initial_count, len(active))
    return active


def _screening_keep_count(active_count: int, final_top_n: int, remaining_rounds: int) -> int:
    if remaining_rounds <= 1:
        return min(active_count, final_top_n)
    ratio = final_top_n / max(1, active_count)
    keep = int(math.ceil(active_count * (ratio ** (1.0 / remaining_rounds))))
    return min(active_count, max(final_top_n, keep))


def _shortened_train_fold(fold: FoldSpec, train_days: int) -> FoldSpec:
    train_start = pd.Timestamp(fold.train_start)
    train_end = min(pd.Timestamp(fold.train_end), train_start + pd.Timedelta(days=train_days))
    return FoldSpec(
        fold_index=fold.fold_index,
        train_start=fold.train_start,
        train_end=train_end.to_pydatetime(),
        test_start=fold.test_start,
        test_end=fold.test_end,
    )


def run_stage(
    conn,
    run_id: int,
    stage: str,
    candidates: list[dict[str, int | float]],
    folds: list[FoldSpec],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
) -> StageResult:
    parameter_ids = persistence.insert_parameter_stubs(conn, run_id, stage, candidates)
    train_by_hash: dict[str, list[Evaluation]] = {parameters.parameter_hash(candidate): [] for candidate in candidates}
    oos_by_hash: dict[str, list[Evaluation]] = {}
    oos_evals: list[Evaluation] = []

    for fold in folds:
        log.info("Stage %s fold %d train candidates %d", stage, fold.fold_index, len(candidates))
        train_evals = evaluate_many(stage, candidates, fold, "train", base_cfg, opt_cfg, ticks, bars, keep_trades=False)
        for evaluation in train_evals:
            train_by_hash[evaluation.parameter_hash].append(evaluation)

        selected = sorted(train_evals, key=lambda item: item.score, reverse=True)[: opt_cfg.train_top_n_per_fold]
        persistence.insert_fold_results(conn, run_id, selected, parameter_ids)
        aggregates = build_aggregates(stage, candidates, train_by_hash, oos_by_hash, len(folds), opt_cfg)
        persistence.update_parameter_set_results(conn, run_id, aggregates)
        selected_candidates = [evaluation.values for evaluation in selected]
        log.info("Stage %s fold %d oos candidates %d", stage, fold.fold_index, len(selected_candidates))
        fold_oos = evaluate_many(stage, selected_candidates, fold, "oos", base_cfg, opt_cfg, ticks, bars, keep_trades=True)
        for evaluation in fold_oos:
            oos_by_hash.setdefault(evaluation.parameter_hash, []).append(evaluation)
        persistence.insert_fold_results(conn, run_id, fold_oos, parameter_ids)
        oos_evals.extend(fold_oos)
        aggregates = build_aggregates(stage, candidates, train_by_hash, oos_by_hash, len(folds), opt_cfg)
        persistence.update_parameter_set_results(conn, run_id, aggregates)

    aggregates = build_aggregates(stage, candidates, train_by_hash, oos_by_hash, len(folds), opt_cfg)
    mc_by_hash = score_monte_carlo(aggregates, oos_by_hash, base_cfg, opt_cfg)
    apply_final_scores(aggregates, mc_by_hash, opt_cfg)
    aggregates.sort(key=lambda row: row["score"], reverse=True)
    for rank, row in enumerate(aggregates, start=1):
        row["stage_rank"] = rank
    persistence.update_parameter_set_results(conn, run_id, aggregates)
    return StageResult(stage, candidates, aggregates, oos_evals, mc_by_hash, parameter_ids)


def build_folds(ticks: pd.DataFrame, opt_cfg: OptimizerConfig) -> list[FoldSpec]:
    data_start = ticks["tick_time"].iloc[0]
    data_end = ticks["tick_time"].iloc[-1]
    train_delta = pd.Timedelta(days=opt_cfg.train_days)
    test_delta = pd.Timedelta(days=opt_cfg.test_days)
    step_delta = pd.Timedelta(days=opt_cfg.step_days)

    folds: list[FoldSpec] = []
    train_start = data_start
    fold_index = 1
    while True:
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta
        if test_end > data_end:
            break
        folds.append(
            FoldSpec(
                fold_index=fold_index,
                train_start=train_start.to_pydatetime(),
                train_end=train_end.to_pydatetime(),
                test_start=test_start.to_pydatetime(),
                test_end=test_end.to_pydatetime(),
            )
        )
        fold_index += 1
        train_start = train_start + step_delta

    if not folds:
        needed_days = opt_cfg.train_days + opt_cfg.test_days
        available_days = (data_end - data_start) / pd.Timedelta(days=1)
        raise RuntimeError(
            f"Not enough data for walk-forward: available_days={available_days:.2f} needed_days={needed_days}"
        )
    log.info("Built walk-forward folds %d train_days %d test_days %d step_days %d", len(folds), opt_cfg.train_days, opt_cfg.test_days, opt_cfg.step_days)
    return folds


def evaluate_many(
    stage: str,
    candidates: list[dict[str, int | float]],
    fold: FoldSpec,
    role: str,
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    keep_trades: bool,
) -> list[Evaluation]:
    tasks = _group_evaluation_tasks(stage, candidates, fold, role, keep_trades, base_cfg)
    total = len(candidates)
    if total <= 0:
        return []
    total_groups = len(tasks)

    started = time.monotonic()
    log.info(
        "Stage %s fold %d %s evaluate start candidates %d groups %d processes %d chunk_size %d progress_every %d progress_seconds %d",
        stage,
        fold.fold_index,
        role,
        total,
        total_groups,
        opt_cfg.processes if total > 1 else 1,
        1 if opt_cfg.processes > 1 and total_groups > 1 else 1,
        opt_cfg.progress_log_every,
        opt_cfg.progress_log_seconds,
    )

    if opt_cfg.processes <= 1 or total_groups <= 1:
        results: list[Evaluation] = []
        window = _prepare_window_data(ticks, bars, fold, role, opt_cfg.profile_cache_size)
        progress = _ProgressLogState(started_at=started, last_logged_at=started)
        completed = 0
        for task in tasks:
            evaluations = _evaluate_group_with_window(task, window, base_cfg)
            results.extend(evaluations)
            completed += len(evaluations)
            _log_evaluation_progress(stage, fold.fold_index, role, completed, total, progress, opt_cfg)
        _log_evaluation_complete(stage, fold.fold_index, role, total, started)
        return results

    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    chunk_size = 1
    results: list[Evaluation] = []
    progress = _ProgressLogState(started_at=started, last_logged_at=started)
    with ctx.Pool(
        processes=opt_cfg.processes,
        initializer=_init_worker,
        initargs=(ticks, bars, base_cfg, fold, role, opt_cfg.profile_cache_size),
    ) as pool:
        completed = 0
        for evaluations in pool.imap_unordered(_evaluate_group_task, tasks, chunksize=chunk_size):
            results.extend(evaluations)
            completed += len(evaluations)
            _log_evaluation_progress(stage, fold.fold_index, role, completed, total, progress, opt_cfg)
    _log_evaluation_complete(stage, fold.fold_index, role, total, started)
    return results


@dataclass
class _ProgressLogState:
    started_at: float
    last_logged_at: float
    last_logged_completed: int = 0


def _log_evaluation_progress(
    stage: str,
    fold_index: int,
    role: str,
    completed: int,
    total: int,
    progress: _ProgressLogState,
    opt_cfg: OptimizerConfig,
) -> None:
    if completed >= total:
        return

    now = time.monotonic()
    completed_delta = completed - progress.last_logged_completed
    seconds_delta = now - progress.last_logged_at
    if completed_delta < opt_cfg.progress_log_every and seconds_delta < opt_cfg.progress_log_seconds:
        return

    elapsed = now - progress.started_at
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = total - completed
    eta = remaining / rate if rate > 0 else 0.0
    log.info(
        "Stage %s fold %d %s progress %d/%d elapsed %.1fs rate %.2f/s eta %.1fs",
        stage,
        fold_index,
        role,
        completed,
        total,
        elapsed,
        rate,
        eta,
    )
    progress.last_logged_at = now
    progress.last_logged_completed = completed


def _log_evaluation_complete(stage: str, fold_index: int, role: str, total: int, started: float) -> None:
    elapsed = time.monotonic() - started
    rate = total / elapsed if elapsed > 0 else 0.0
    log.info(
        "Stage %s fold %d %s complete candidates %d elapsed %.1fs rate %.2f/s",
        stage,
        fold_index,
        role,
        total,
        elapsed,
        rate,
    )


def _group_evaluation_tasks(
    stage: str,
    candidates: list[dict[str, int | float]],
    fold: FoldSpec,
    role: str,
    keep_trades: bool,
    base_cfg: RunConfig,
) -> list[tuple]:
    groups: OrderedDict[tuple, list[dict[str, int | float]]] = OrderedDict()
    for candidate in sorted(candidates, key=lambda values: _candidate_profile_key(values, base_cfg)):
        groups.setdefault(_candidate_profile_key(candidate, base_cfg), []).append(candidate)
    return [(stage, group, fold, role, keep_trades) for group in groups.values()]


def _candidate_profile_key(values: dict[str, int | float], base_cfg: RunConfig) -> tuple:
    return _profile_cache_key(apply_parameter_values(base_cfg, values))


def _init_worker(
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    base_cfg: RunConfig,
    fold: FoldSpec,
    role: str,
    profile_cache_size: int,
) -> None:
    global _WORKER_BASE_CFG, _WORKER_WINDOW
    _WORKER_BASE_CFG = base_cfg
    _WORKER_WINDOW = _prepare_window_data(ticks, bars, fold, role, profile_cache_size)


def _evaluate_group_task(task) -> list[Evaluation]:
    if _WORKER_BASE_CFG is None or _WORKER_WINDOW is None:
        raise RuntimeError("Worker data was not initialized")
    return _evaluate_group_with_window(task, _WORKER_WINDOW, _WORKER_BASE_CFG)


def _prepare_window_data(
    ticks: pd.DataFrame,
    bars: pd.DataFrame,
    fold: FoldSpec,
    role: str,
    profile_cache_size: int,
) -> WindowData:
    slice_start, slice_end, trade_start, trade_end = _window_bounds(fold, role)
    tick_slice = ticks[(ticks["tick_time"] >= slice_start) & (ticks["tick_time"] < slice_end)].copy()
    bar_slice = bars[(bars["bar_start"] >= slice_start) & (bars["bar_start"] < slice_end)].copy()
    return WindowData(
        ticks=tick_slice,
        bars=bar_slice,
        trade_start=trade_start,
        trade_end=trade_end,
        profile_cache_size=profile_cache_size,
    )


def _window_bounds(fold: FoldSpec, role: str) -> tuple[object, object, object, object]:
    if role == "train":
        return fold.train_start, fold.train_end, fold.train_start, fold.train_end
    return fold.train_start, fold.test_end, fold.test_start, fold.test_end


def _evaluate_group_with_window(task, window: WindowData, base_cfg: RunConfig) -> list[Evaluation]:
    stage, group, fold, role, keep_trades = task
    if not group:
        return []
    profile_cfg = apply_parameter_values(base_cfg, group[0])
    profile = rolling_profile_levels(window.bars, profile_cfg)
    profiled_ticks = attach_profile_to_ticks(window.ticks, window.bars, profile)
    return [
        _evaluate_candidate_with_profiled_ticks(stage, values, fold, role, keep_trades, window, base_cfg, profiled_ticks)
        for values in group
    ]


def _evaluate_candidate_with_profiled_ticks(
    stage: str,
    values: dict[str, int | float],
    fold: FoldSpec,
    role: str,
    keep_trades: bool,
    window: WindowData,
    base_cfg: RunConfig,
    profiled_ticks: pd.DataFrame,
) -> Evaluation:
    cfg = apply_parameter_values(base_cfg, values)
    result = run_simulation(
        window.ticks,
        window.bars,
        cfg,
        trade_start_ts=window.trade_start,
        trade_end_ts=window.trade_end,
        log_result=False,
        profiled_ticks=profiled_ticks,
    )
    summary = summarize_trades(result.trades, cfg.initial_equity, result.final_equity)
    score = score_evaluation(summary, result)
    if not keep_trades:
        result.trades = []
    return Evaluation(
        stage=stage,
        fold_index=fold.fold_index,
        window_role=role,
        values=dict(values),
        parameter_hash=parameters.parameter_hash(values),
        parameter_label=parameters.parameter_label(values),
        window_start=window.trade_start,
        window_end=window.trade_end,
        ticks_simulated=result.ticks_simulated,
        bars_total=result.bars_total,
        signals_total=result.signals_total,
        long_signals=result.long_signals,
        short_signals=result.short_signals,
        rejected_missing_band=result.rejected_signals_missing_band,
        rejected_band_too_narrow=result.rejected_signals_band_too_narrow,
        rejected_stop_too_small=result.rejected_signals_stop_too_small,
        rejected_stop_too_large=result.rejected_signals_stop_too_large,
        skipped_no_size=result.skipped_signals_no_size,
        ruined=result.ruined,
        summary=summary,
        score=score,
        trades=result.trades,
    )


def _profile_for_config(window: WindowData, cfg: RunConfig) -> pd.DataFrame:
    if window.profile_cache_size <= 0:
        return rolling_profile_levels(window.bars, cfg)

    key = _profile_cache_key(cfg)
    cached = window.profile_cache.get(key)
    if cached is not None:
        window.profile_cache.move_to_end(key)
        return cached

    profile = rolling_profile_levels(window.bars, cfg)
    window.profile_cache[key] = profile
    while len(window.profile_cache) > window.profile_cache_size:
        window.profile_cache.popitem(last=False)
    return profile


def _profile_cache_key(cfg: RunConfig) -> tuple:
    return (
        cfg.lookback_bars,
        cfg.min_lookback_bars,
        round(float(cfg.price_step), 8),
        round(float(cfg.median_quantile), 8),
        round(float(cfg.band_lower_quantile), 8),
        round(float(cfg.band_upper_quantile), 8),
        round(float(cfg.stop_profile_lower_quantile), 8),
        round(float(cfg.stop_profile_upper_quantile), 8),
    )


def score_evaluation(summary: dict, result: SimulationResult) -> float:
    trades = int(summary.get("total_trades") or 0)
    if trades <= 0:
        return -10000.0
    total_return = float(summary.get("total_return_pct") or 0.0)
    profit_factor = _finite_profit_factor(summary.get("profit_factor"), summary.get("gross_profit_eur"), summary.get("gross_loss_eur"))
    drawdown = abs(min(0.0, float(summary.get("max_drawdown_pct") or 0.0)))
    win_rate = float(summary.get("win_rate_pct") or 0.0)
    score = total_return
    score += min(profit_factor, 3.0) * 12.0
    score += min(trades, 300) / 300.0 * 8.0
    score += (win_rate - 50.0) * 0.05
    score -= drawdown * 1.2
    if result.ruined:
        score -= 1000.0
    return round(score, 4)


def build_aggregates(
    stage: str,
    candidates: list[dict[str, int | float]],
    train_by_hash: dict[str, list[Evaluation]],
    oos_by_hash: dict[str, list[Evaluation]],
    expected_folds: int,
    opt_cfg: OptimizerConfig,
) -> list[dict]:
    aggregates = []
    for values in candidates:
        digest = parameters.parameter_hash(values)
        train_metrics = aggregate_evaluations(train_by_hash.get(digest, []), expected_folds)
        oos_metrics = aggregate_evaluations(oos_by_hash.get(digest, []), expected_folds)
        pre_mc_score = score_aggregate(oos_metrics, train_metrics, opt_cfg)
        row = {
            "stage": stage,
            "stage_rank": None,
            "values": values,
            "parameter_hash": digest,
            "parameter_label": parameters.parameter_label(values),
            "parameter_signature": parameters.parameter_signature(values),
            "pre_mc_score": pre_mc_score,
            "score": pre_mc_score,
            "mc_scored": False,
            "passed_pre_mc_filters": passed_pre_mc_filters(oos_metrics, opt_cfg),
            "passed_filters": False,
            **_prefix_metrics("train", train_metrics),
            **_prefix_metrics("oos", oos_metrics),
        }
        aggregates.append(row)
    return aggregates


def aggregate_evaluations(evaluations: list[Evaluation], expected_folds: int) -> dict:
    if not evaluations:
        return {
            "folds": 0,
            "expected_folds": expected_folds,
            "total_trades": 0,
            "total_return_pct": 0.0,
            "mean_return_pct": 0.0,
            "median_return_pct": 0.0,
            "std_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "profit_factor": 0.0,
            "win_rate_pct": 0.0,
            "profitable_folds_pct": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "net_profit_eur": 0.0,
            "avg_trade_pnl_eur": 0.0,
            "signals_total": 0,
            "ruined_folds": 0,
        }

    returns = np.array([float(item.summary.get("total_return_pct") or 0.0) for item in evaluations], dtype=np.float64)
    trade_counts = np.array([int(item.summary.get("total_trades") or 0) for item in evaluations], dtype=np.int64)
    gross_profit = float(sum(float(item.summary.get("gross_profit_eur") or 0.0) for item in evaluations))
    gross_loss = float(sum(float(item.summary.get("gross_loss_eur") or 0.0) for item in evaluations))
    total_trades = int(trade_counts.sum())
    net_profit = float(sum(float(item.summary.get("net_profit_eur") or 0.0) for item in evaluations))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = None
    else:
        profit_factor = 0.0
    weighted_wins = sum(float(item.summary.get("win_rate_pct") or 0.0) * int(item.summary.get("total_trades") or 0) for item in evaluations)
    win_rate = weighted_wins / total_trades if total_trades > 0 else 0.0
    factors = np.maximum(0.0, 1.0 + returns / 100.0)
    total_return = (float(np.prod(factors)) - 1.0) * 100.0
    drawdowns = [float(item.summary.get("max_drawdown_pct") or 0.0) for item in evaluations]
    return {
        "folds": len(evaluations),
        "expected_folds": expected_folds,
        "total_trades": total_trades,
        "total_return_pct": round(total_return, 4),
        "mean_return_pct": round(float(returns.mean()), 4),
        "median_return_pct": round(float(np.median(returns)), 4),
        "std_return_pct": round(float(returns.std(ddof=0)), 4),
        "max_drawdown_pct": round(float(min(drawdowns)), 4),
        "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
        "win_rate_pct": round(float(win_rate), 4),
        "profitable_folds_pct": round(float(np.mean(returns > 0.0) * 100.0), 4),
        "gross_profit_eur": round(gross_profit, 2),
        "gross_loss_eur": round(gross_loss, 2),
        "net_profit_eur": round(net_profit, 2),
        "avg_trade_pnl_eur": round(net_profit / total_trades, 4) if total_trades > 0 else 0.0,
        "signals_total": int(sum(item.signals_total for item in evaluations)),
        "ruined_folds": int(sum(1 for item in evaluations if item.ruined)),
    }


def score_aggregate(oos: dict, train: dict, opt_cfg: OptimizerConfig) -> float:
    if oos["folds"] <= 0:
        return round(-100000.0 + train["mean_return_pct"], 4)
    profit_factor = _finite_profit_factor(oos["profit_factor"], oos["gross_profit_eur"], oos["gross_loss_eur"])
    drawdown = abs(min(0.0, float(oos["max_drawdown_pct"])))
    missing_folds = max(0, int(oos["expected_folds"]) - int(oos["folds"]))
    score = float(oos["total_return_pct"])
    score += min(profit_factor, 3.0) * 16.0
    score += min(float(oos["total_trades"]) / max(1.0, float(opt_cfg.min_oos_trades)), 2.0) * 10.0
    score += float(oos["profitable_folds_pct"]) * 0.18
    score -= float(oos["std_return_pct"]) * 0.55
    score -= drawdown * 1.4
    score -= missing_folds * 12.0
    score -= int(oos["ruined_folds"]) * 250.0
    if oos["total_trades"] < opt_cfg.min_oos_trades:
        score -= (opt_cfg.min_oos_trades - oos["total_trades"]) / max(1, opt_cfg.min_oos_trades) * 60.0
    if profit_factor < opt_cfg.min_oos_profit_factor:
        score -= (opt_cfg.min_oos_profit_factor - profit_factor) * 70.0
    if drawdown > opt_cfg.max_oos_drawdown_pct:
        score -= (drawdown - opt_cfg.max_oos_drawdown_pct) * 5.0
    return round(score, 4)


def score_monte_carlo(
    aggregates: list[dict],
    oos_by_hash: dict[str, list[Evaluation]],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
) -> dict[str, dict]:
    if opt_cfg.mc_score_top_n <= 0 or not base_cfg.monte_carlo_enabled:
        return {}
    ranked = [
        row for row in sorted(aggregates, key=lambda item: item["pre_mc_score"], reverse=True)
        if row["oos_folds"] > 0 and row["oos_total_trades"] > 1
    ][: opt_cfg.mc_score_top_n]
    out: dict[str, dict] = {}
    for rank, row in enumerate(ranked, start=1):
        trades = _oos_trades_for_hash(oos_by_hash, row["parameter_hash"])
        if len(trades) < 2:
            continue
        cfg = apply_parameter_values(base_cfg, row["values"])
        seed_offset = int(row["parameter_hash"][:8], 16) % 1_000_000
        mc = run_monte_carlo(trades, cfg.initial_equity, cfg, seed_offset=seed_offset)
        if mc:
            mc["mc_score_rank"] = rank
            out[row["parameter_hash"]] = mc
    return out


def apply_final_scores(aggregates: list[dict], mc_by_hash: dict[str, dict], opt_cfg: OptimizerConfig) -> None:
    for row in aggregates:
        mc = mc_by_hash.get(row["parameter_hash"])
        row["mc_scored"] = bool(mc)
        row["mc_prob_of_ruin_pct"] = _mc_ruin(mc) if mc else None
        row["score"] = row["pre_mc_score"]
        if mc:
            ruin = _mc_ruin(mc)
            row["score"] = round(row["score"] - ruin * 2.0 - max(0.0, ruin - opt_cfg.max_mc_ruin_pct) * 20.0, 4)
        row["passed_filters"] = row["passed_pre_mc_filters"] and bool(mc) and (row["mc_prob_of_ruin_pct"] is not None) and row["mc_prob_of_ruin_pct"] <= opt_cfg.max_mc_ruin_pct


def persist_top_trades(conn, run_id: int, stage: StageResult, opt_cfg: OptimizerConfig) -> None:
    if opt_cfg.persist_top_trades_n <= 0:
        return
    top = [
        row for row in sorted(stage.aggregates, key=lambda item: item["score"], reverse=True)
        if row["oos_folds"] > 0 and row["parameter_hash"] in stage.parameter_ids
    ][: opt_cfg.persist_top_trades_n]
    hashes = {row["parameter_hash"] for row in top}
    if not hashes:
        return
    rows = []
    for evaluation in stage.oos_evals:
        if evaluation.parameter_hash not in hashes:
            continue
        parameter_set_id = stage.parameter_ids[evaluation.parameter_hash]
        rows.extend(
            persistence.trade_rows(
                parameter_set_id=parameter_set_id,
                stage=evaluation.stage,
                fold_index=evaluation.fold_index,
                window_role=evaluation.window_role,
                trades=evaluation.trades,
            )
        )
    persistence.insert_trade_rows(conn, rows)
    persistence.mark_top_trade_sets(conn, run_id, [stage.parameter_ids[row["parameter_hash"]] for row in top])


def passed_pre_mc_filters(metrics: dict, opt_cfg: OptimizerConfig) -> bool:
    if metrics["folds"] <= 0:
        return False
    profit_factor = _finite_profit_factor(metrics["profit_factor"], metrics["gross_profit_eur"], metrics["gross_loss_eur"])
    drawdown = abs(min(0.0, float(metrics["max_drawdown_pct"])))
    return (
        int(metrics["total_trades"]) >= opt_cfg.min_oos_trades
        and profit_factor >= opt_cfg.min_oos_profit_factor
        and drawdown <= opt_cfg.max_oos_drawdown_pct
        and int(metrics["ruined_folds"]) == 0
    )


def _prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _finite_profit_factor(value, gross_profit, gross_loss) -> float:
    if value is None:
        return 4.0 if float(gross_profit or 0.0) > 0 and float(gross_loss or 0.0) == 0 else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _mc_ruin(mc: dict | None) -> float:
    if not mc:
        return 0.0
    values = [
        float(mc.get("base_prob_of_ruin_pct") or 0.0),
        float(mc.get("slip_prob_of_ruin_pct") or 0.0),
        float(mc.get("seq_prob_of_ruin_pct") or 0.0),
    ]
    return max(values)


def _oos_trades_for_hash(oos_by_hash: dict[str, list[Evaluation]], digest: str) -> list[ClosedTrade]:
    trades: list[ClosedTrade] = []
    for evaluation in sorted(oos_by_hash.get(digest, []), key=lambda item: item.fold_index):
        trades.extend(evaluation.trades)
    return trades
