"""Two-stage walk-forward optimizer for the NAS100 hit-frequency median model."""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import time
import gc
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from . import parameters, persistence
from .config import OptimizerConfig, RunConfig, apply_parameter_values, effective_session_selector_min_trades, enabled_session_keys
from .data import BarData, TickData, NANOSECONDS_PER_DAY, ns_to_datetime
from .entities import ClosedTrade, SimulationResult
from .profile import ProfileArrays, rolling_profile_arrays, warmup as warmup_profile
from .risk import run_monte_carlo, summarize_trades, summarize_trades_by_session
from .sessions import SESSION_LABELS, SESSION_SORT_ORDERS, SESSION_TYPES
from .sim_core import warmup as warmup_sim_core
from .simulation import (
    precompute_events,
    run_session_portfolio_simulation,
    run_simulation,
    run_simulation_summary,
)

log = logging.getLogger(__name__)

_WORKER_BASE_CFG: RunConfig | None = None
_WORKER_WINDOW: WindowData | None = None
STREAM_CANDIDATE_BATCH_SIZE = 65_536


@dataclass(frozen=True, slots=True)
class FoldSpec:
    fold_index: int
    train_start_ns: int
    train_end_ns: int
    test_start_ns: int
    test_end_ns: int


@dataclass(slots=True)
class WindowData:
    ticks: TickData
    bars: BarData
    tick_bar_index: np.ndarray
    trade_start_ns: int
    trade_end_ns: int
    profile_cache_size: int
    profile_cache: OrderedDict[tuple, ProfileArrays] = field(default_factory=OrderedDict)


@dataclass(slots=True)
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
    rejected_price_range_position: int
    rejected_stop_too_small: int
    rejected_stop_too_large: int
    skipped_no_size: int
    ruined: bool
    summary: dict
    score: float
    session_stats: dict[str, dict] = field(default_factory=dict)
    trades: list[ClosedTrade] = field(default_factory=list)


@dataclass(slots=True)
class SessionSelectionCandidate:
    evaluation: Evaluation
    stats: dict
    base_score: float
    robust_score: float
    neighbor_count: int
    candidate_index: int = -1


@dataclass(frozen=True, slots=True)
class CandidateSource:
    count: int
    iter_batches: Callable[[], Iterable[parameters.CandidateBatch]]


@dataclass(frozen=True, slots=True)
class EvaluationProgressContext:
    batch_index: int
    batch_count: int
    fold_completed_offset: int
    fold_total: int


@dataclass(slots=True)
class StageResult:
    stage: str
    candidates: list[dict[str, int | float]]
    aggregates: list[dict]
    oos_evals: list[Evaluation]
    mc_by_hash: dict[str, dict]
    parameter_ids: dict[str, int] = field(default_factory=dict)
    portfolio_id: int | None = None
    portfolio_aggregate: dict | None = None
    selected_values: list[dict[str, int | float]] = field(default_factory=list)


@dataclass(slots=True)
class MetricsAccumulator:
    folds: int = 0
    return_pcts: list[float] = field(default_factory=list)
    total_trades: int = 0
    winning_trades: int = 0
    gross_profit_eur: float = 0.0
    gross_loss_eur: float = 0.0
    net_profit_eur: float = 0.0
    max_drawdown_pct: float = 0.0
    signals_total: int = 0
    ruined_folds: int = 0

    def add(self, evaluation: Evaluation) -> None:
        summary = evaluation.summary
        self.folds += 1
        self.return_pcts.append(float(summary.get("total_return_pct") or 0.0))
        self.total_trades += int(summary.get("total_trades") or 0)
        self.winning_trades += int(summary.get("winning_trades") or 0)
        self.gross_profit_eur += float(summary.get("gross_profit_eur") or 0.0)
        self.gross_loss_eur += float(summary.get("gross_loss_eur") or 0.0)
        self.net_profit_eur += float(summary.get("net_profit_eur") or 0.0)
        self.max_drawdown_pct = min(self.max_drawdown_pct, float(summary.get("max_drawdown_pct") or 0.0))
        self.signals_total += int(evaluation.signals_total)
        self.ruined_folds += 1 if evaluation.ruined else 0

    def metrics(self, expected_folds: int) -> dict:
        if self.folds <= 0:
            return _empty_aggregate_metrics(expected_folds)
        returns = np.array(self.return_pcts, dtype=np.float64)
        if self.gross_loss_eur > 0.0:
            profit_factor = self.gross_profit_eur / self.gross_loss_eur
        elif self.gross_profit_eur > 0.0:
            profit_factor = None
        else:
            profit_factor = 0.0
        factors = np.maximum(0.0, 1.0 + returns / 100.0)
        total_return = (float(np.prod(factors)) - 1.0) * 100.0
        win_rate = (float(self.winning_trades) / self.total_trades * 100.0) if self.total_trades > 0 else 0.0
        return {
            "folds": self.folds,
            "expected_folds": expected_folds,
            "total_trades": self.total_trades,
            "total_return_pct": round(total_return, 4),
            "mean_return_pct": round(float(returns.mean()), 4),
            "median_return_pct": round(float(np.median(returns)), 4),
            "std_return_pct": round(float(returns.std(ddof=0)), 4),
            "max_drawdown_pct": round(float(self.max_drawdown_pct), 4),
            "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
            "win_rate_pct": round(win_rate, 4),
            "profitable_folds_pct": round(float(np.mean(returns > 0.0) * 100.0), 4),
            "gross_profit_eur": round(self.gross_profit_eur, 2),
            "gross_loss_eur": round(self.gross_loss_eur, 2),
            "net_profit_eur": round(self.net_profit_eur, 2),
            "avg_trade_pnl_eur": round(self.net_profit_eur / self.total_trades, 4) if self.total_trades > 0 else 0.0,
            "signals_total": self.signals_total,
            "ruined_folds": self.ruined_folds,
        }


class MetricsArrayAccumulator:
    def __init__(self, count: int, expected_folds: int):
        self.count = int(count)
        self.expected_folds = int(expected_folds)
        shape = (self.count,)
        self.folds = np.zeros(shape, dtype=np.uint16)
        self.return_pcts = np.zeros((self.count, max(1, self.expected_folds)), dtype=np.float32)
        self.total_trades = np.zeros(shape, dtype=np.int32)
        self.winning_trades = np.zeros(shape, dtype=np.int32)
        self.gross_profit_eur = np.zeros(shape, dtype=np.float64)
        self.gross_loss_eur = np.zeros(shape, dtype=np.float64)
        self.net_profit_eur = np.zeros(shape, dtype=np.float64)
        self.max_drawdown_pct = np.zeros(shape, dtype=np.float32)
        self.signals_total = np.zeros(shape, dtype=np.int32)
        self.ruined_folds = np.zeros(shape, dtype=np.uint16)

    def add(self, index: int, evaluation: Evaluation) -> None:
        idx = int(index)
        summary = evaluation.summary
        slot = int(self.folds[idx])
        if slot < self.return_pcts.shape[1]:
            self.return_pcts[idx, slot] = float(summary.get("total_return_pct") or 0.0)
        self.folds[idx] += 1
        self.total_trades[idx] += int(summary.get("total_trades") or 0)
        self.winning_trades[idx] += int(summary.get("winning_trades") or 0)
        self.gross_profit_eur[idx] += float(summary.get("gross_profit_eur") or 0.0)
        self.gross_loss_eur[idx] += float(summary.get("gross_loss_eur") or 0.0)
        self.net_profit_eur[idx] += float(summary.get("net_profit_eur") or 0.0)
        self.max_drawdown_pct[idx] = min(
            float(self.max_drawdown_pct[idx]),
            float(summary.get("max_drawdown_pct") or 0.0),
        )
        self.signals_total[idx] += int(evaluation.signals_total)
        if evaluation.ruined:
            self.ruined_folds[idx] += 1

    def metrics(self, index: int) -> dict:
        idx = int(index)
        folds = int(self.folds[idx])
        if folds <= 0:
            return _empty_aggregate_metrics(self.expected_folds)
        returns = self.return_pcts[idx, : min(folds, self.return_pcts.shape[1])].astype(np.float64, copy=False)
        gross_profit = float(self.gross_profit_eur[idx])
        gross_loss = float(self.gross_loss_eur[idx])
        if gross_loss > 0.0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0.0:
            profit_factor = None
        else:
            profit_factor = 0.0
        factors = np.maximum(0.0, 1.0 + returns / 100.0)
        total_return = (float(np.prod(factors)) - 1.0) * 100.0
        trades = int(self.total_trades[idx])
        winning = int(self.winning_trades[idx])
        win_rate = (float(winning) / trades * 100.0) if trades > 0 else 0.0
        return {
            "folds": folds,
            "expected_folds": self.expected_folds,
            "total_trades": trades,
            "total_return_pct": round(total_return, 4),
            "mean_return_pct": round(float(returns.mean()), 4),
            "median_return_pct": round(float(np.median(returns)), 4),
            "std_return_pct": round(float(returns.std(ddof=0)), 4),
            "max_drawdown_pct": round(float(self.max_drawdown_pct[idx]), 4),
            "profit_factor": round(float(profit_factor), 4) if profit_factor is not None else None,
            "win_rate_pct": round(win_rate, 4),
            "profitable_folds_pct": round(float(np.mean(returns > 0.0) * 100.0), 4),
            "gross_profit_eur": round(gross_profit, 2),
            "gross_loss_eur": round(gross_loss, 2),
            "net_profit_eur": round(float(self.net_profit_eur[idx]), 2),
            "avg_trade_pnl_eur": round(float(self.net_profit_eur[idx]) / trades, 4) if trades > 0 else 0.0,
            "signals_total": int(self.signals_total[idx]),
            "ruined_folds": int(self.ruined_folds[idx]),
        }


@dataclass(slots=True)
class SessionStatsAccumulator:
    folds: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    gross_profit_eur: float = 0.0
    gross_loss_eur: float = 0.0
    net_profit_eur: float = 0.0

    def add(self, stats: dict) -> None:
        self.folds += 1
        self.total_trades += int(stats.get("total_trades") or 0)
        self.winning_trades += int(stats.get("winning_trades") or 0)
        self.losing_trades += int(stats.get("losing_trades") or 0)
        self.breakeven_trades += int(stats.get("breakeven_trades") or 0)
        self.gross_profit_eur += float(stats.get("gross_profit_eur") or 0.0)
        self.gross_loss_eur += float(stats.get("gross_loss_eur") or 0.0)
        self.net_profit_eur += float(stats.get("net_profit_eur") or 0.0)

    def metrics(self, expected_folds: int) -> dict:
        total = int(self.total_trades)
        return {
            "folds": self.folds,
            "expected_folds": expected_folds,
            "total_trades": total,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "breakeven_trades": self.breakeven_trades,
            "win_rate_pct": round(float(self.winning_trades) / total * 100.0, 4) if total > 0 else 0.0,
            "gross_profit_eur": round(self.gross_profit_eur, 2),
            "gross_loss_eur": round(self.gross_loss_eur, 2),
            "net_profit_eur": round(self.net_profit_eur, 2),
            "avg_trade_pnl_eur": round(self.net_profit_eur / total, 4) if total > 0 else 0.0,
        }


def _stage1_candidate_source(grid: dict[str, list[int | float]], opt_cfg: OptimizerConfig) -> CandidateSource:
    count = parameters.stage1_candidate_count(grid, opt_cfg.stage1_max_parameter_sets)

    def _iter_batches() -> Iterable[parameters.CandidateBatch]:
        return parameters.iter_stage1_candidate_batches(
            grid,
            opt_cfg.stage1_max_parameter_sets,
            opt_cfg.sampling_seed,
            STREAM_CANDIDATE_BATCH_SIZE,
        )

    return CandidateSource(count=count, iter_batches=_iter_batches)


def _list_candidate_source(candidates: list[dict[str, int | float]]) -> CandidateSource:
    count = len(candidates)

    def _iter_batches() -> Iterable[parameters.CandidateBatch]:
        for start in range(0, count, STREAM_CANDIDATE_BATCH_SIZE):
            yield parameters.CandidateBatch(start, candidates[start : start + STREAM_CANDIDATE_BATCH_SIZE])

    return CandidateSource(count=count, iter_batches=_iter_batches)


def run_walk_forward_optimizer(
    conn,
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: TickData,
    bars: BarData,
    started: float,
) -> None:
    warmup_sim_core()
    warmup_profile()
    folds = build_folds(ticks, opt_cfg)
    data_start_ts = ns_to_datetime(int(ticks.tick_time_ns[0]))
    data_end_ts = ns_to_datetime(int(ticks.tick_time_ns[-1]))
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
    stage1_source = _stage1_candidate_source(grid, opt_cfg)
    log.info("Stage 1 candidates %d folds %d", stage1_source.count, len(folds))
    stage1 = run_stage(conn, run_id, "stage1", stage1_source, folds, base_cfg, opt_cfg, ticks, bars)

    final_stage = stage1
    stage2_count = 0
    if opt_cfg.stage2_enabled:
        seed_values = _dedupe_values(stage1.selected_values)[: opt_cfg.stage2_seed_top_n]
        stage2_candidates = parameters.build_stage2_candidates(
            seed_values,
            grid,
            set(),
            opt_cfg.stage2_max_parameter_sets,
            opt_cfg.sampling_seed,
        )
        existing_stage1_hashes = persistence.fetch_existing_parameter_hashes(
            conn,
            run_id,
            "stage1",
            (parameters.parameter_hash(candidate) for candidate in stage2_candidates),
        )
        stage2_candidates = [
            candidate
            for candidate in stage2_candidates
            if parameters.parameter_hash(candidate) not in existing_stage1_hashes
        ]
        stage2_count = len(stage2_candidates)
        if stage2_candidates:
            log.info("Stage 2 candidates %d seeds %d folds %d", len(stage2_candidates), len(seed_values), len(folds))
            stage2 = run_stage(conn, run_id, "stage2", _list_candidate_source(stage2_candidates), folds, base_cfg, opt_cfg, ticks, bars)
            final_stage = stage2
        else:
            log.info("Stage 2 skipped no new candidates")

    best = final_stage.portfolio_aggregate
    persistence.update_run_complete(
        conn,
        run_id,
        status="complete",
        run_duration_seconds=time.time() - started,
        stage1_parameter_sets=stage1_source.count,
        stage2_parameter_sets=stage2_count,
        best_parameter_set_id=None,
        best_portfolio_id=final_stage.portfolio_id,
        best_score=best["score"] if best else None,
    )
    if best:
        log.info(
            "Walk-forward complete session_portfolio_id %s score %.4f oos_trades %d oos_return %.4f profit_factor %.4f max_drawdown %.4f",
            final_stage.portfolio_id,
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
    ticks: TickData,
    bars: BarData,
    started: float,
) -> None:
    warmup_sim_core()
    warmup_profile()
    data_start_ts = ns_to_datetime(int(ticks.tick_time_ns[0]))
    data_end_ts = ns_to_datetime(int(ticks.tick_time_ns[-1]))
    session_values = parameters.load_single_session_parameters(opt_cfg.single_parameter_path)
    active_sessions = enabled_session_keys(cfg)
    if not active_sessions:
        raise RuntimeError("RUN_MODE=single has no enabled sessions")
    run_metadata_cfg = apply_parameter_values(cfg, session_values[active_sessions[0]])
    run_id = persistence.create_run(
        conn,
        run_metadata_cfg,
        opt_cfg,
        mode="single",
        data_start_ts=data_start_ts,
        data_end_ts=data_end_ts,
        ticks_loaded=len(ticks),
        bars_built=len(bars),
        folds_built=1,
    )
    _run_single_session_parameter_backtest(
        conn,
        run_id,
        cfg,
        opt_cfg,
        ticks,
        bars,
        started,
        data_start_ts,
        data_end_ts,
        session_values,
        active_sessions,
    )


def _run_single_session_parameter_backtest(
    conn,
    run_id: int,
    cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: TickData,
    bars: BarData,
    started: float,
    data_start_ts,
    data_end_ts,
    session_values: dict[str, dict[str, int | float]],
    active_sessions: list[str],
) -> None:
    stage = "single"
    trade_start_ns = int(ticks.tick_time_ns[0])
    trade_end_ns = int(ticks.tick_time_ns[-1]) + 1
    fold = FoldSpec(
        fold_index=1,
        train_start_ns=trade_start_ns,
        train_end_ns=trade_start_ns,
        test_start_ns=trade_start_ns,
        test_end_ns=trade_end_ns,
    )
    selected_by_session = {
        session_type: _manual_selection_evaluation(
            stage,
            session_type,
            session_values[session_type],
            data_start_ts,
            data_end_ts,
        )
        for session_type in active_sessions
    }
    unique_values = _dedupe_values([evaluation.values for evaluation in selected_by_session.values()])

    persistence.insert_parameter_stubs(conn, run_id, stage, [(0, unique_values)])
    parameter_ids = persistence.fetch_parameter_ids(
        conn,
        run_id,
        stage,
        (parameters.parameter_hash(values) for values in unique_values),
    )
    portfolio_id = persistence.insert_portfolio_stub(conn, run_id, stage)
    selection_ids = persistence.insert_session_selections(
        conn,
        run_id,
        portfolio_id,
        stage,
        fold,
        selected_by_session,
        parameter_ids,
        cfg,
        opt_cfg,
    )

    portfolio_eval = evaluate_session_portfolio(
        stage,
        fold,
        selected_by_session,
        cfg,
        opt_cfg,
        ticks,
        bars,
        keep_trades=True,
    )
    portfolio_aggregate, mc = build_portfolio_aggregate(stage, [portfolio_eval], 1, cfg, opt_cfg)
    persistence.insert_portfolio_fold_result(conn, portfolio_id, portfolio_eval)
    persistence.update_portfolio_results(conn, portfolio_id, portfolio_aggregate, mc)
    persistence.update_session_selection_oos_stats(conn, selection_ids, portfolio_eval.session_stats)
    persistence.upsert_parameter_session_stats(
        conn,
        run_id,
        _single_parameter_session_aggregates(stage, selected_by_session, portfolio_eval.session_stats),
        parameter_ids,
    )
    persistence.insert_portfolio_trade_rows(
        conn,
        run_id,
        portfolio_id,
        portfolio_eval,
        selection_ids,
        {session: evaluation.parameter_hash for session, evaluation in selected_by_session.items()},
        parameter_ids,
    )
    persistence.mark_top_trade_sets(conn, run_id, list(parameter_ids.values()))
    persistence.update_run_complete(
        conn,
        run_id,
        status="complete",
        run_duration_seconds=time.time() - started,
        stage1_parameter_sets=len(unique_values),
        stage2_parameter_sets=0,
        best_parameter_set_id=None,
        best_portfolio_id=portfolio_id,
        best_score=portfolio_aggregate["score"],
    )
    log.info(
        "Single session-parameter run complete portfolio_id %s sessions %d unique_parameter_sets %d score %.4f trades %d return %.4f profit_factor %.4f max_drawdown %.4f",
        portfolio_id,
        len(selected_by_session),
        len(unique_values),
        portfolio_aggregate["score"],
        portfolio_aggregate["oos_total_trades"],
        portfolio_aggregate["oos_total_return_pct"],
        portfolio_aggregate["oos_profit_factor"] if portfolio_aggregate["oos_profit_factor"] is not None else 0.0,
        portfolio_aggregate["oos_max_drawdown_pct"],
    )


def _manual_selection_evaluation(
    stage: str,
    session_type: str,
    values: dict[str, int | float],
    data_start_ts,
    data_end_ts,
) -> Evaluation:
    digest = parameters.parameter_hash(values)
    return Evaluation(
        stage=stage,
        fold_index=1,
        window_role="manual",
        values=values,
        parameter_hash=digest,
        parameter_label=parameters.parameter_label(values),
        window_start=data_start_ts,
        window_end=data_end_ts,
        ticks_simulated=0,
        bars_total=0,
        signals_total=0,
        long_signals=0,
        short_signals=0,
        rejected_missing_band=0,
        rejected_band_too_narrow=0,
        rejected_price_range_position=0,
        rejected_stop_too_small=0,
        rejected_stop_too_large=0,
        skipped_no_size=0,
        ruined=False,
        summary={},
        score=0.0,
        session_stats={session_type: _empty_manual_session_stats()},
        trades=[],
    )


def _empty_manual_session_stats() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "breakeven_trades": 0,
        "win_rate_pct": 0.0,
        "gross_profit_eur": 0.0,
        "gross_loss_eur": 0.0,
        "net_profit_eur": 0.0,
        "avg_trade_pnl_eur": 0.0,
    }


def _single_parameter_session_aggregates(
    stage: str,
    selected_by_session: dict[str, Evaluation],
    session_stats: dict[str, dict],
) -> list[dict]:
    rows = []
    for session_type, evaluation in selected_by_session.items():
        stats = session_stats.get(session_type, _empty_manual_session_stats())
        rows.append(
            {
                "stage": stage,
                "parameter_hash": evaluation.parameter_hash,
                "window_role": "oos",
                "session_type": session_type,
                "session_label": SESSION_LABELS[session_type],
                "session_sort_order": SESSION_SORT_ORDERS[session_type],
                "folds": 1,
                "expected_folds": 1,
                "total_trades": int(stats.get("total_trades") or 0),
                "winning_trades": int(stats.get("winning_trades") or 0),
                "losing_trades": int(stats.get("losing_trades") or 0),
                "breakeven_trades": int(stats.get("breakeven_trades") or 0),
                "win_rate_pct": float(stats.get("win_rate_pct") or 0.0),
                "gross_profit_eur": float(stats.get("gross_profit_eur") or 0.0),
                "gross_loss_eur": float(stats.get("gross_loss_eur") or 0.0),
                "net_profit_eur": float(stats.get("net_profit_eur") or 0.0),
                "avg_trade_pnl_eur": float(stats.get("avg_trade_pnl_eur") or 0.0),
            }
        )
    return rows


def _empty_aggregate_metrics(expected_folds: int) -> dict:
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


def _accumulate_evaluation(
    metrics_by_hash: dict[str, MetricsAccumulator],
    sessions_by_hash: dict[str, dict[str, SessionStatsAccumulator]],
    evaluation: Evaluation,
) -> None:
    digest = evaluation.parameter_hash
    metrics = metrics_by_hash.get(digest)
    if metrics is None:
        metrics = MetricsAccumulator()
        metrics_by_hash[digest] = metrics
    metrics.add(evaluation)

    session_accs = sessions_by_hash.get(digest)
    if session_accs is None:
        session_accs = {
            session_type: SessionStatsAccumulator()
            for session_type, _label, _sort_order in SESSION_TYPES
        }
        sessions_by_hash[digest] = session_accs
    for session_type, _label, _sort_order in SESSION_TYPES:
        session_accs[session_type].add(evaluation.session_stats.get(session_type, {}))


def _accumulate_session_evaluation(
    sessions_by_hash: dict[str, dict[str, SessionStatsAccumulator]],
    evaluation: Evaluation,
) -> None:
    digest = evaluation.parameter_hash
    session_accs = sessions_by_hash.get(digest)
    if session_accs is None:
        session_accs = {
            session_type: SessionStatsAccumulator()
            for session_type, _label, _sort_order in SESSION_TYPES
        }
        sessions_by_hash[digest] = session_accs
    for session_type, _label, _sort_order in SESSION_TYPES:
        session_accs[session_type].add(evaluation.session_stats.get(session_type, {}))


def _aggregate_row_from_metrics(
    stage: str,
    values: dict[str, int | float],
    train_metrics: dict,
    oos_metrics: dict,
    opt_cfg: OptimizerConfig,
    stage_rank: int | None = None,
) -> dict:
    pre_mc_score = score_aggregate(oos_metrics, train_metrics, opt_cfg)
    return {
        "stage": stage,
        "stage_rank": stage_rank,
        "values": values,
        "parameter_hash": parameters.parameter_hash(values),
        "parameter_label": parameters.parameter_label(values),
        "parameter_signature": parameters.parameter_signature(values),
        "pre_mc_score": pre_mc_score,
        "score": pre_mc_score,
        "mc_scored": False,
        "mc_prob_of_ruin_pct": None,
        "passed_pre_mc_filters": passed_pre_mc_filters(oos_metrics, opt_cfg),
        "passed_filters": False,
        "oos_full_coverage": _oos_full_coverage(oos_metrics, train_metrics["expected_folds"]),
        **_prefix_metrics("train", train_metrics),
        **_prefix_metrics("oos", oos_metrics),
    }


def build_aggregates_from_accumulators(
    stage: str,
    candidates: list[dict[str, int | float]],
    train_metrics_by_hash: dict[str, MetricsAccumulator],
    oos_metrics_by_hash: dict[str, MetricsAccumulator],
    expected_folds: int,
    opt_cfg: OptimizerConfig,
) -> list[dict]:
    aggregates = []
    for values in candidates:
        digest = parameters.parameter_hash(values)
        train_accumulator = train_metrics_by_hash.get(digest)
        oos_accumulator = oos_metrics_by_hash.get(digest)
        train_metrics = (
            train_accumulator.metrics(expected_folds)
            if train_accumulator
            else _empty_aggregate_metrics(expected_folds)
        )
        oos_metrics = (
            oos_accumulator.metrics(expected_folds)
            if oos_accumulator
            else _empty_aggregate_metrics(expected_folds)
        )
        aggregates.append(_aggregate_row_from_metrics(stage, values, train_metrics, oos_metrics, opt_cfg))
    return aggregates


def _oos_full_coverage(oos_metrics: dict, expected_folds: int) -> bool:
    folds = int(oos_metrics.get("folds") or 0)
    return folds > 0 and folds >= int(expected_folds)


def build_session_aggregates_from_accumulators(
    stage: str,
    candidates: list[dict[str, int | float]],
    sessions_by_hash: dict[str, dict[str, SessionStatsAccumulator]],
    expected_folds: int,
    window_role: str,
) -> list[dict]:
    rows = []
    for values in candidates:
        digest = parameters.parameter_hash(values)
        session_accs = sessions_by_hash.get(digest, {})
        for session_type, _label, _sort_order in SESSION_TYPES:
            accumulator = session_accs.get(session_type)
            metrics = accumulator.metrics(expected_folds) if accumulator else {
                "folds": 0,
                "expected_folds": expected_folds,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0,
                "win_rate_pct": 0.0,
                "gross_profit_eur": 0.0,
                "gross_loss_eur": 0.0,
                "net_profit_eur": 0.0,
                "avg_trade_pnl_eur": 0.0,
            }
            rows.append(
                {
                    "stage": stage,
                    "parameter_hash": digest,
                    "window_role": window_role,
                    "session_type": session_type,
                    "session_label": SESSION_LABELS[session_type],
                    "session_sort_order": SESSION_SORT_ORDERS[session_type],
                    **metrics,
                }
            )
    return rows


def _score_candidate_source(
    candidate_source: CandidateSource,
    stage: str,
    train_metrics: MetricsArrayAccumulator,
    oos_metrics_by_hash: dict[str, MetricsAccumulator],
    expected_folds: int,
    opt_cfg: OptimizerConfig,
) -> np.ndarray:
    scores = np.empty(candidate_source.count, dtype=np.float64)
    for batch in candidate_source.iter_batches():
        for offset, values in enumerate(batch.candidates):
            index = batch.start_index + offset
            digest = parameters.parameter_hash(values)
            train = train_metrics.metrics(index)
            oos_accumulator = oos_metrics_by_hash.get(digest)
            oos = oos_accumulator.metrics(expected_folds) if oos_accumulator else _empty_aggregate_metrics(expected_folds)
            scores[index] = score_aggregate(oos, train, opt_cfg)
    log.info("Stage %s scored candidates %d", stage, candidate_source.count)
    return scores


def _stage_ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(scores.shape[0], dtype=np.int32)
    ranks[order] = np.arange(1, scores.shape[0] + 1, dtype=np.int32)
    return ranks


def _persist_parameter_aggregates_stream(
    conn,
    run_id: int,
    stage: str,
    candidate_source: CandidateSource,
    train_metrics: MetricsArrayAccumulator,
    oos_metrics_by_hash: dict[str, MetricsAccumulator],
    stage_ranks: np.ndarray,
    expected_folds: int,
    opt_cfg: OptimizerConfig,
) -> None:
    updated = 0
    for batch in candidate_source.iter_batches():
        aggregates = []
        for offset, values in enumerate(batch.candidates):
            index = batch.start_index + offset
            digest = parameters.parameter_hash(values)
            train = train_metrics.metrics(index)
            oos_accumulator = oos_metrics_by_hash.get(digest)
            oos = oos_accumulator.metrics(expected_folds) if oos_accumulator else _empty_aggregate_metrics(expected_folds)
            aggregates.append(
                _aggregate_row_from_metrics(
                    stage,
                    values,
                    train,
                    oos,
                    opt_cfg,
                    stage_rank=int(stage_ranks[index]),
                )
            )
        persistence.update_parameter_set_results(conn, run_id, aggregates)
        updated += len(aggregates)
        if updated % 250_000 < len(aggregates):
            log.info("Stage %s updated parameter aggregates progress %d", stage, updated)
        del aggregates
    log.info("Stage %s updated parameter aggregates %d", stage, updated)


def run_stage(
    conn,
    run_id: int,
    stage: str,
    candidate_source: CandidateSource,
    folds: list[FoldSpec],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: TickData,
    bars: BarData,
) -> StageResult:
    inserted_count = persistence.insert_parameter_stubs(conn, run_id, stage, candidate_source.iter_batches())
    if inserted_count != candidate_source.count:
        log.warning("Stage %s candidate count mismatch source %d inserted %d", stage, candidate_source.count, inserted_count)
    portfolio_id = persistence.insert_portfolio_stub(conn, run_id, stage)
    train_metrics = MetricsArrayAccumulator(candidate_source.count, len(folds))
    selected_train_sessions_by_hash: dict[str, dict[str, SessionStatsAccumulator]] = {}
    portfolio_oos_evals: list[Evaluation] = []
    selected_values: list[dict[str, int | float]] = []
    previous_selected_by_session: dict[str, Evaluation] = {}
    finalist_union: OrderedDict[str, tuple[dict[str, int | float], int]] = OrderedDict()

    # Phase 1: per fold, evaluate every candidate on train, select one parameter
    # set per session and evaluate the resulting session portfolio OOS. Finalist
    # candidates are collected here but validated OOS later (phase 2).
    for fold in folds:
        log.info(
            "Stage %s fold %d train %s candidates %d",
            stage,
            fold.fold_index,
            _fold_role_period(fold, "train"),
            candidate_source.count,
        )
        session_pools = _empty_session_candidate_pools()
        global_finalist_pool: list[tuple[tuple, Evaluation, int]] = []
        session_pool_limit = max(1, int(opt_cfg.session_selector_top_n), int(opt_cfg.finalist_top_n_per_session))
        batch_count = max(1, math.ceil(candidate_source.count / STREAM_CANDIDATE_BATCH_SIZE))

        for batch_index, batch in enumerate(candidate_source.iter_batches(), start=1):
            train_evals = evaluate_many(
                stage,
                batch.candidates,
                fold,
                "train",
                base_cfg,
                opt_cfg,
                ticks,
                bars,
                keep_trades=False,
                progress_context=EvaluationProgressContext(
                    batch_index=batch_index,
                    batch_count=batch_count,
                    fold_completed_offset=batch.start_index,
                    fold_total=candidate_source.count,
                ),
            )
            index_by_hash = {
                parameters.parameter_hash(values): batch.start_index + offset
                for offset, values in enumerate(batch.candidates)
            }
            for evaluation in train_evals:
                candidate_index = index_by_hash[evaluation.parameter_hash]
                train_metrics.add(candidate_index, evaluation)
                _add_session_pool_candidates(session_pools, evaluation, candidate_index, base_cfg, opt_cfg, session_pool_limit)
                _add_global_finalist_candidate(global_finalist_pool, evaluation, candidate_index, opt_cfg)
            del train_evals
            del index_by_hash

        selected_by_session = select_session_parameters_from_pools(
            session_pools,
            base_cfg,
            opt_cfg,
            previous_selected_by_session=previous_selected_by_session,
        )
        previous_selected_by_session = selected_by_session
        selected = _unique_evaluations(selected_by_session.values())
        selected_values.extend(evaluation.values for evaluation in selected)
        selected_parameter_ids = persistence.fetch_parameter_ids(
            conn,
            run_id,
            stage,
            (evaluation.parameter_hash for evaluation in selected),
        )
        persistence.insert_fold_results(conn, run_id, selected, selected_parameter_ids)
        selection_ids = persistence.insert_session_selections(
            conn,
            run_id,
            portfolio_id,
            stage,
            fold,
            selected_by_session,
            selected_parameter_ids,
            base_cfg,
            opt_cfg,
        )

        for values, candidate_index in select_finalist_candidates_from_pools(session_pools, global_finalist_pool, opt_cfg):
            finalist_union.setdefault(parameters.parameter_hash(values), (values, candidate_index))

        log.info(
            "Stage %s fold %d session_portfolio oos %s sessions %d",
            stage,
            fold.fold_index,
            _fold_role_period(fold, "oos"),
            len(selected_by_session),
        )
        portfolio_eval = evaluate_session_portfolio(
            stage,
            fold,
            selected_by_session,
            base_cfg,
            opt_cfg,
            ticks,
            bars,
            keep_trades=True,
        )
        portfolio_oos_evals.append(portfolio_eval)
        persistence.insert_portfolio_fold_result(conn, portfolio_id, portfolio_eval)
        persistence.update_session_selection_oos_stats(conn, selection_ids, portfolio_eval.session_stats)
        if opt_cfg.persist_top_trades_n > 0:
            persistence.insert_portfolio_trade_rows(
                conn,
                run_id,
                portfolio_id,
                portfolio_eval,
                selection_ids,
                {session: evaluation.parameter_hash for session, evaluation in selected_by_session.items()},
                selected_parameter_ids,
            )
        for evaluation in selected:
            _accumulate_session_evaluation(selected_train_sessions_by_hash, evaluation)
        persistence.upsert_parameter_session_stats(
            conn,
            run_id,
            build_session_aggregates_from_accumulators(
                stage,
                [evaluation.values for evaluation in selected],
                selected_train_sessions_by_hash,
                len(folds),
                "train",
            ),
            selected_parameter_ids,
        )
        gc.collect()

    # Finalist set: cap the cross-fold union by cumulative train score so every
    # finalist can be validated OOS in every fold (consistent full coverage).
    finalist_items = _cap_finalists_streaming(list(finalist_union.values()), train_metrics, opt_cfg)
    finalist_values = [values for values, _candidate_index in finalist_items]
    del finalist_union

    # Phase 2: evaluate the fixed finalist set OOS in *every* fold.
    oos_metrics_by_hash: dict[str, MetricsAccumulator] = {}
    oos_sessions_by_hash: dict[str, dict[str, SessionStatsAccumulator]] = {}
    finalist_parameter_ids = persistence.fetch_parameter_ids(
        conn,
        run_id,
        stage,
        (parameters.parameter_hash(values) for values in finalist_values),
    )
    if finalist_values:
        log.info(
            "Stage %s finalist OOS validation candidates %d folds %d keep_trades %s persist %s",
            stage,
            len(finalist_values),
            len(folds),
            opt_cfg.finalist_keep_trades,
            opt_cfg.finalist_persist_fold_results,
        )
        for fold in folds:
            oos_evals = evaluate_many(
                stage, finalist_values, fold, "oos", base_cfg, opt_cfg, ticks, bars,
                keep_trades=opt_cfg.finalist_keep_trades,
            )
            for evaluation in oos_evals:
                _accumulate_evaluation(oos_metrics_by_hash, oos_sessions_by_hash, evaluation)
            if opt_cfg.finalist_persist_fold_results:
                persistence.insert_fold_results(conn, run_id, oos_evals, finalist_parameter_ids)
                if opt_cfg.finalist_keep_trades:
                    persist_finalist_trades(conn, run_id, oos_evals, finalist_parameter_ids)
            del oos_evals
            gc.collect()

    # Final aggregates: train metrics for all candidates, OOS metrics for the
    # finalists (full coverage). oos_full_coverage flags the comparable subset.
    scores = _score_candidate_source(candidate_source, stage, train_metrics, oos_metrics_by_hash, len(folds), opt_cfg)
    stage_ranks = _stage_ranks_from_scores(scores)
    _persist_parameter_aggregates_stream(
        conn,
        run_id,
        stage,
        candidate_source,
        train_metrics,
        oos_metrics_by_hash,
        stage_ranks,
        len(folds),
        opt_cfg,
    )
    persistence.upsert_parameter_session_stats(
        conn,
        run_id,
        build_session_aggregates_from_accumulators(
            stage,
            _dedupe_values(selected_values),
            selected_train_sessions_by_hash,
            len(folds),
            "train",
        ),
        persistence.fetch_parameter_ids(
            conn,
            run_id,
            stage,
            (parameters.parameter_hash(values) for values in _dedupe_values(selected_values)),
        ),
    )
    if finalist_values:
        persistence.upsert_parameter_session_stats(
            conn,
            run_id,
            build_session_aggregates_from_accumulators(
                stage,
                finalist_values,
                oos_sessions_by_hash,
                len(folds),
                "oos",
            ),
            finalist_parameter_ids,
        )
    portfolio_aggregate, portfolio_mc = build_portfolio_aggregate(stage, portfolio_oos_evals, len(folds), base_cfg, opt_cfg)
    persistence.update_portfolio_results(conn, portfolio_id, portfolio_aggregate, portfolio_mc)
    ranked_oos_values = [
        values
        for values, candidate_index in sorted(finalist_items, key=lambda item: float(scores[item[1]]), reverse=True)
        if int(oos_metrics_by_hash.get(parameters.parameter_hash(values), MetricsAccumulator()).folds) > 0
    ]
    selected_values = _dedupe_values([*ranked_oos_values, *selected_values])
    del oos_metrics_by_hash
    del oos_sessions_by_hash
    return StageResult(
        stage=stage,
        candidates=[],
        aggregates=[],
        oos_evals=portfolio_oos_evals,
        mc_by_hash={},
        parameter_ids={},
        portfolio_id=portfolio_id,
        portfolio_aggregate=portfolio_aggregate,
        selected_values=selected_values,
    )


def select_finalist_candidates(
    train_evals: list[Evaluation],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
) -> list[dict[str, int | float]]:
    """Per-fold finalist candidates: union of per-session train top-N and global
    train top-N. The cross-fold union of these is validated OOS in every fold."""
    if not opt_cfg.finalist_enabled:
        return []
    per_session_top_n = int(opt_cfg.finalist_top_n_per_session)
    global_top_n = int(opt_cfg.finalist_global_top_n)
    if per_session_top_n <= 0 and global_top_n <= 0:
        return []

    selected: OrderedDict[str, Evaluation] = OrderedDict()
    if per_session_top_n > 0:
        for session_type, _label, _sort_order in SESSION_TYPES:
            ranked = _rank_session_candidates(
                session_type,
                train_evals,
                base_cfg,
                opt_cfg,
                limit=per_session_top_n,
            )
            for candidate in ranked[:per_session_top_n]:
                selected.setdefault(candidate.evaluation.parameter_hash, candidate.evaluation)

    if global_top_n > 0:
        ranked_global = [
            evaluation
            for evaluation in sorted(train_evals, key=_global_finalist_sort_key, reverse=True)
            if int(evaluation.summary.get("total_trades") or 0) > 0
        ][:global_top_n]
        for evaluation in ranked_global:
            selected.setdefault(evaluation.parameter_hash, evaluation)

    return [evaluation.values for evaluation in selected.values()]


def _empty_session_candidate_pools() -> dict[str, list[SessionSelectionCandidate]]:
    return {session_type: [] for session_type, _label, _sort_order in SESSION_TYPES}


def _add_session_pool_candidates(
    pools: dict[str, list[SessionSelectionCandidate]],
    evaluation: Evaluation,
    candidate_index: int,
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    limit: int,
) -> None:
    if limit <= 0:
        return
    prune_at = max(limit * 2, limit + 128)
    for session_type, _label, _sort_order in SESSION_TYPES:
        stats = evaluation.session_stats.get(session_type)
        score = score_session_stats(stats, base_cfg, opt_cfg)
        if score <= -9999.0:
            continue
        pool = pools[session_type]
        pool.append(
            SessionSelectionCandidate(
                evaluation=evaluation,
                stats=stats or {},
                base_score=score,
                robust_score=score,
                neighbor_count=1,
                candidate_index=candidate_index,
            )
        )
        if len(pool) > prune_at:
            pool.sort(key=_session_candidate_sort_key, reverse=True)
            del pool[limit:]


def _add_global_finalist_candidate(
    pool: list[tuple[tuple, Evaluation, int]],
    evaluation: Evaluation,
    candidate_index: int,
    opt_cfg: OptimizerConfig,
) -> None:
    limit = int(opt_cfg.finalist_global_top_n)
    if not opt_cfg.finalist_enabled or limit <= 0:
        return
    if int(evaluation.summary.get("total_trades") or 0) <= 0:
        return
    pool.append((_global_finalist_sort_key(evaluation), evaluation, candidate_index))
    prune_at = max(limit * 2, limit + 128)
    if len(pool) > prune_at:
        pool.sort(key=lambda item: item[0], reverse=True)
        del pool[limit:]


def select_finalist_candidates_from_pools(
    session_pools: dict[str, list[SessionSelectionCandidate]],
    global_pool: list[tuple[tuple, Evaluation, int]],
    opt_cfg: OptimizerConfig,
) -> list[tuple[dict[str, int | float], int]]:
    if not opt_cfg.finalist_enabled:
        return []
    per_session_top_n = int(opt_cfg.finalist_top_n_per_session)
    global_top_n = int(opt_cfg.finalist_global_top_n)
    if per_session_top_n <= 0 and global_top_n <= 0:
        return []

    selected: OrderedDict[str, tuple[dict[str, int | float], int]] = OrderedDict()
    if per_session_top_n > 0:
        for session_type, _label, _sort_order in SESSION_TYPES:
            ranked = _rank_session_candidate_pool(session_pools.get(session_type, []), opt_cfg, limit=per_session_top_n)
            for candidate in ranked[:per_session_top_n]:
                selected.setdefault(
                    candidate.evaluation.parameter_hash,
                    (candidate.evaluation.values, candidate.candidate_index),
                )

    if global_top_n > 0:
        global_pool.sort(key=lambda item: item[0], reverse=True)
        for _key, evaluation, candidate_index in global_pool[:global_top_n]:
            selected.setdefault(evaluation.parameter_hash, (evaluation.values, candidate_index))

    return list(selected.values())


def _cap_finalists(
    finalist_values: list[dict[str, int | float]],
    train_metrics_by_hash: dict[str, MetricsAccumulator],
    expected_folds: int,
    opt_cfg: OptimizerConfig,
) -> list[dict[str, int | float]]:
    """Cap the cross-fold finalist union at finalist_max_sets, keeping the sets
    with the best cumulative *train* score (no OOS leakage into the selection)."""
    max_sets = int(opt_cfg.finalist_max_sets)
    if max_sets <= 0 or len(finalist_values) <= max_sets:
        return finalist_values

    def _train_score(values: dict[str, int | float]) -> float:
        accumulator = train_metrics_by_hash.get(parameters.parameter_hash(values))
        metrics = accumulator.metrics(expected_folds) if accumulator else _empty_aggregate_metrics(expected_folds)
        return score_aggregate(metrics, metrics, opt_cfg)

    ranked = sorted(finalist_values, key=_train_score, reverse=True)
    return ranked[:max_sets]


def _cap_finalists_streaming(
    finalist_items: list[tuple[dict[str, int | float], int]],
    train_metrics: MetricsArrayAccumulator,
    opt_cfg: OptimizerConfig,
) -> list[tuple[dict[str, int | float], int]]:
    max_sets = int(opt_cfg.finalist_max_sets)
    if max_sets <= 0 or len(finalist_items) <= max_sets:
        return finalist_items

    def _train_score(item: tuple[dict[str, int | float], int]) -> float:
        metrics = train_metrics.metrics(item[1])
        return score_aggregate(metrics, metrics, opt_cfg)

    return sorted(finalist_items, key=_train_score, reverse=True)[:max_sets]


def persist_finalist_trades(
    conn,
    run_id: int,
    evaluations: list[Evaluation],
    parameter_ids: dict[str, int],
) -> None:
    rows = []
    for evaluation in evaluations:
        parameter_set_id = parameter_ids.get(evaluation.parameter_hash)
        if parameter_set_id is None:
            continue
        rows.extend(
            persistence.trade_rows(
                parameter_set_id=parameter_set_id,
                stage=evaluation.stage,
                fold_index=evaluation.fold_index,
                window_role=evaluation.window_role,
                trades=evaluation.trades,
            )
        )
    persistence.insert_trade_rows(conn, rows, run_id=run_id)


def select_session_parameters(
    train_evals: list[Evaluation],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    previous_selected_by_session: dict[str, Evaluation] | None = None,
) -> dict[str, Evaluation]:
    selected: dict[str, Evaluation] = {}
    for session_type, _label, _sort_order in SESSION_TYPES:
        ranked = _rank_session_candidates(session_type, train_evals, base_cfg, opt_cfg)
        if not ranked:
            continue
        chosen = ranked[0]
        previous_hash = None
        if previous_selected_by_session and session_type in previous_selected_by_session:
            previous_hash = previous_selected_by_session[session_type].parameter_hash
            previous_current = next((item for item in ranked if item.evaluation.parameter_hash == previous_hash), None)
            tolerance = float(opt_cfg.session_selector_previous_keep_score_tolerance)
            if previous_current and previous_current.robust_score >= chosen.robust_score - tolerance:
                chosen = previous_current
        best_eval = chosen.evaluation
        selected[session_type] = best_eval
        stats = chosen.stats
        log.info(
            "Selected session parameter session %s hash %s robust_score %.4f base_score %.4f neighbor_count %d trades %d net_profit %.2f previous_hash %s",
            session_type,
            best_eval.parameter_hash[:10],
            chosen.robust_score,
            chosen.base_score,
            chosen.neighbor_count,
            int(stats.get("total_trades") or 0),
            float(stats.get("net_profit_eur") or 0.0),
            previous_hash[:10] if previous_hash else "-",
        )
    return selected


def select_session_parameters_from_pools(
    session_pools: dict[str, list[SessionSelectionCandidate]],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    previous_selected_by_session: dict[str, Evaluation] | None = None,
) -> dict[str, Evaluation]:
    selected: dict[str, Evaluation] = {}
    for session_type, _label, _sort_order in SESSION_TYPES:
        ranked = _rank_session_candidate_pool(session_pools.get(session_type, []), opt_cfg)
        if not ranked:
            continue
        chosen = ranked[0]
        previous_hash = None
        if previous_selected_by_session and session_type in previous_selected_by_session:
            previous_hash = previous_selected_by_session[session_type].parameter_hash
            previous_current = next((item for item in ranked if item.evaluation.parameter_hash == previous_hash), None)
            tolerance = float(opt_cfg.session_selector_previous_keep_score_tolerance)
            if previous_current and previous_current.robust_score >= chosen.robust_score - tolerance:
                chosen = previous_current
        best_eval = chosen.evaluation
        selected[session_type] = best_eval
        stats = chosen.stats
        log.info(
            "Selected session parameter session %s hash %s robust_score %.4f base_score %.4f neighbor_count %d trades %d net_profit %.2f previous_hash %s",
            session_type,
            best_eval.parameter_hash[:10],
            chosen.robust_score,
            chosen.base_score,
            chosen.neighbor_count,
            int(stats.get("total_trades") or 0),
            float(stats.get("net_profit_eur") or 0.0),
            previous_hash[:10] if previous_hash else "-",
        )
    return selected


def score_session_stats(stats: dict | None, base_cfg: RunConfig, opt_cfg: OptimizerConfig) -> float:
    if not stats:
        return -10000.0
    trades = int(stats.get("total_trades") or 0)
    if trades <= 0:
        return -10000.0
    gross_profit = float(stats.get("gross_profit_eur") or 0.0)
    gross_loss = float(stats.get("gross_loss_eur") or 0.0)
    profit_factor = _finite_profit_factor(
        (gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit > 0 else 0.0),
        gross_profit,
        gross_loss,
    )
    win_rate = float(stats.get("win_rate_pct") or 0.0)
    net_profit = float(stats.get("net_profit_eur") or 0.0)
    std_trade_pnl = float(stats.get("std_trade_pnl_eur") or 0.0)
    uncertainty_eur = float(opt_cfg.session_selector_lcb_z) * std_trade_pnl * math.sqrt(float(trades))
    conservative_net_profit = net_profit - uncertainty_eur
    total_return = conservative_net_profit / max(1.0, float(base_cfg.initial_equity)) * 100.0
    min_session_trades = _session_selector_min_trades(opt_cfg)
    score = total_return
    score += min(profit_factor, 3.0) * 12.0
    score += min(trades / max(1.0, float(min_session_trades)), 2.0) * 8.0
    score += (win_rate - 50.0) * 0.05
    if net_profit <= 0.0:
        score -= 20.0
    if trades < min_session_trades:
        score -= (min_session_trades - trades) / max(1, min_session_trades) * 20.0
    if profit_factor < opt_cfg.min_oos_profit_factor:
        score -= (opt_cfg.min_oos_profit_factor - profit_factor) * 30.0
    return round(score, 4)


def _rank_session_candidates(
    session_type: str,
    train_evals: list[Evaluation],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    limit: int | None = None,
) -> list[SessionSelectionCandidate]:
    scored: list[SessionSelectionCandidate] = []
    for evaluation in train_evals:
        stats = evaluation.session_stats.get(session_type)
        score = score_session_stats(stats, base_cfg, opt_cfg)
        if score <= -9999.0:
            continue
        scored.append(
            SessionSelectionCandidate(
                evaluation=evaluation,
                stats=stats or {},
                base_score=score,
                robust_score=score,
                neighbor_count=1,
            )
        )
    if not scored:
        return []

    return _rank_session_candidate_pool(scored, opt_cfg, limit=limit)


def _rank_session_candidate_pool(
    scored: list[SessionSelectionCandidate],
    opt_cfg: OptimizerConfig,
    limit: int | None = None,
) -> list[SessionSelectionCandidate]:
    if not scored:
        return []
    scored = [
        SessionSelectionCandidate(
            evaluation=item.evaluation,
            stats=item.stats,
            base_score=item.base_score,
            robust_score=item.base_score,
            neighbor_count=1,
            candidate_index=item.candidate_index,
        )
        for item in scored
    ]
    scored.sort(key=_session_candidate_sort_key, reverse=True)
    pool_size = max(1, int(limit if limit is not None else opt_cfg.session_selector_top_n))
    pool = scored[:pool_size]
    weight = float(opt_cfg.session_selector_plateau_weight)
    if weight > 0.0 and len(pool) > 1:
        for candidate in pool:
            neighbor_scores = [
                item.base_score
                for item in pool
                if _parameter_distance(candidate.evaluation.values, item.evaluation.values)
                <= float(opt_cfg.session_selector_neighbor_distance)
            ]
            if not neighbor_scores:
                neighbor_scores = [candidate.base_score]
            plateau_score = float(np.median(np.array(neighbor_scores, dtype=np.float64)))
            isolation_penalty = max(0, 2 - len(neighbor_scores)) * weight * 5.0
            candidate.neighbor_count = len(neighbor_scores)
            candidate.robust_score = round(
                candidate.base_score * (1.0 - weight) + plateau_score * weight - isolation_penalty,
                4,
            )
    pool.sort(key=_session_candidate_sort_key, reverse=True)
    return pool


def _session_candidate_sort_key(candidate: SessionSelectionCandidate) -> tuple:
    stats = candidate.stats
    return (
        candidate.robust_score,
        candidate.base_score,
        int(stats.get("total_trades") or 0),
        float(stats.get("net_profit_eur") or 0.0),
        -_parameter_complexity(candidate.evaluation.values),
    )


def _global_finalist_sort_key(evaluation: Evaluation) -> tuple:
    summary = evaluation.summary
    return (
        float(evaluation.score),
        int(summary.get("total_trades") or 0),
        float(summary.get("net_profit_eur") or 0.0),
        -_parameter_complexity(evaluation.values),
    )


def _session_selector_min_trades(opt_cfg: OptimizerConfig) -> int:
    return effective_session_selector_min_trades(opt_cfg)


def _parameter_distance(left: dict[str, int | float], right: dict[str, int | float]) -> float:
    scales = {
        "LOOKBACK_BARS": 30.0,
        "LONG_CROSS_QUANTILE": 0.05,
        "SHORT_CROSS_QUANTILE": 0.05,
        "ENTRY_PRICE_RANGE_POSITION_MAX_DEVIATION_PCT": 5.0,
        "ALL_STOP_MODES_TAKE_PROFIT_POINTS": 3.0,
        "BAND_STOP_MIN_PROFILE_RANGE_POINTS": 10.0,
        "BAND_STOP_PROFILE_LOWER_QUANTILE": 0.05,
        "BAND_STOP_PROFILE_UPPER_QUANTILE": 0.05,
        "BAND_STOP_PROFILE_BUFFER_POINTS": 1.0,
        "BAND_STOP_MIN_DISTANCE_POINTS": 3.0,
        "BAND_STOP_MAX_DISTANCE_POINTS": 5.0,
    }
    total = 0.0
    count = 0
    for name, scale in scales.items():
        if name not in left or name not in right:
            continue
        delta = abs(float(left[name]) - float(right[name])) / scale
        total += min(delta, 3.0) ** 2
        count += 1
    return math.sqrt(total / max(1, count))


def _parameter_complexity(values: dict[str, int | float]) -> float:
    if not values:
        return 0.0
    return (
        abs(float(values.get("LONG_CROSS_QUANTILE", 0.5)) - 0.5)
        + abs(float(values.get("SHORT_CROSS_QUANTILE", 0.5)) - 0.5)
        + abs(float(values.get("BAND_STOP_PROFILE_LOWER_QUANTILE", 0.0)) - 0.0)
        + abs(float(values.get("BAND_STOP_PROFILE_UPPER_QUANTILE", 1.0)) - 1.0)
        + abs(float(values.get("BAND_STOP_PROFILE_BUFFER_POINTS", 0.0))) * 0.05
    )


def evaluate_session_portfolio(
    stage: str,
    fold: FoldSpec,
    selected_by_session: dict[str, Evaluation],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: TickData,
    bars: BarData,
    keep_trades: bool,
) -> Evaluation:
    window = _prepare_window_data(ticks, bars, fold, "oos", opt_cfg.profile_cache_size)
    session_cfgs = {
        session_type: apply_parameter_values(base_cfg, evaluation.values)
        for session_type, evaluation in selected_by_session.items()
    }
    profiles = {
        session_type: _profile_for_config(window, session_cfg)
        for session_type, session_cfg in session_cfgs.items()
    }
    result = run_session_portfolio_simulation(
        window.ticks,
        window.bars,
        window.tick_bar_index,
        base_cfg,
        session_cfgs,
        trade_start_ns=window.trade_start_ns,
        trade_end_ns=window.trade_end_ns,
        log_result=False,
        profiles=profiles,
    )
    summary = summarize_trades(result.trades, base_cfg.initial_equity, result.final_equity)
    session_stats = summarize_trades_by_session(result.trades)
    score = score_evaluation(summary, result)
    trades = result.trades
    if not keep_trades:
        trades = []
    return Evaluation(
        stage=stage,
        fold_index=fold.fold_index,
        window_role="oos",
        values={},
        parameter_hash=_portfolio_hash(selected_by_session),
        parameter_label="session_portfolio",
        window_start=ns_to_datetime(window.trade_start_ns),
        window_end=ns_to_datetime(window.trade_end_ns),
        ticks_simulated=result.ticks_simulated,
        bars_total=result.bars_total,
        signals_total=result.signals_total,
        long_signals=result.long_signals,
        short_signals=result.short_signals,
        rejected_missing_band=result.rejected_signals_missing_band,
        rejected_band_too_narrow=result.rejected_signals_band_too_narrow,
        rejected_price_range_position=result.rejected_signals_price_range_position,
        rejected_stop_too_small=result.rejected_signals_stop_too_small,
        rejected_stop_too_large=result.rejected_signals_stop_too_large,
        skipped_no_size=result.skipped_signals_no_size,
        ruined=result.ruined,
        summary=summary,
        score=score,
        session_stats=session_stats,
        trades=trades,
    )


def build_portfolio_aggregate(
    stage: str,
    evaluations: list[Evaluation],
    expected_folds: int,
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
) -> tuple[dict, dict | None]:
    oos_metrics = aggregate_evaluations(evaluations, expected_folds)
    pre_mc_score = score_aggregate(oos_metrics, oos_metrics, opt_cfg)
    trades = _trades_for_evaluations(evaluations)
    mc = run_monte_carlo(trades, base_cfg.initial_equity, base_cfg, seed_offset=0) if len(trades) > 1 else None
    if mc:
        mc["mc_score_rank"] = 1
    ruin = _mc_ruin(mc)
    score = pre_mc_score
    if mc:
        score = round(score - ruin * 2.0 - max(0.0, ruin - opt_cfg.max_mc_ruin_pct) * 20.0, 4)
    aggregate = {
        "stage": stage,
        "stage_rank": 1,
        "pre_mc_score": pre_mc_score,
        "score": score,
        "mc_scored": bool(mc),
        "mc_prob_of_ruin_pct": ruin if mc else None,
        "passed_pre_mc_filters": passed_pre_mc_filters(oos_metrics, opt_cfg),
        "passed_filters": passed_pre_mc_filters(oos_metrics, opt_cfg)
        and bool(mc)
        and ruin <= opt_cfg.max_mc_ruin_pct,
        **_prefix_metrics("oos", oos_metrics),
    }
    return aggregate, mc


def _unique_evaluations(evaluations) -> list[Evaluation]:
    out: list[Evaluation] = []
    seen: set[str] = set()
    for evaluation in evaluations:
        if evaluation.parameter_hash in seen:
            continue
        seen.add(evaluation.parameter_hash)
        out.append(evaluation)
    return out


def _dedupe_values(values: list[dict[str, int | float]]) -> list[dict[str, int | float]]:
    out: list[dict[str, int | float]] = []
    seen: set[str] = set()
    for item in values:
        if not item:
            continue
        digest = parameters.parameter_hash(item)
        if digest in seen:
            continue
        seen.add(digest)
        out.append(item)
    return out


def _portfolio_hash(selected_by_session: dict[str, Evaluation]) -> str:
    parts = [
        f"{session_type}:{selected_by_session[session_type].parameter_hash}"
        for session_type, _label, _sort_order in SESSION_TYPES
        if session_type in selected_by_session
    ]
    return "portfolio|" + "|".join(parts)


def _trades_for_evaluations(evaluations: list[Evaluation]) -> list[ClosedTrade]:
    trades: list[ClosedTrade] = []
    for evaluation in sorted(evaluations, key=lambda item: item.fold_index):
        trades.extend(evaluation.trades)
    return trades


def build_folds(ticks: TickData, opt_cfg: OptimizerConfig) -> list[FoldSpec]:
    data_start = int(ticks.tick_time_ns[0])
    data_end = int(ticks.tick_time_ns[-1])
    train_delta = opt_cfg.train_days * NANOSECONDS_PER_DAY
    test_delta = opt_cfg.test_days * NANOSECONDS_PER_DAY
    step_delta = opt_cfg.step_days * NANOSECONDS_PER_DAY

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
                train_start_ns=train_start,
                train_end_ns=train_end,
                test_start_ns=test_start,
                test_end_ns=test_end,
            )
        )
        fold_index += 1
        train_start = train_start + step_delta

    if not folds:
        needed_days = opt_cfg.train_days + opt_cfg.test_days
        available_days = (data_end - data_start) / NANOSECONDS_PER_DAY
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
    ticks: TickData,
    bars: BarData,
    keep_trades: bool,
    progress_context: EvaluationProgressContext | None = None,
) -> list[Evaluation]:
    tasks = _group_evaluation_tasks(stage, candidates, fold, role, keep_trades, base_cfg)
    total = len(candidates)
    if total <= 0:
        return []
    total_groups = len(tasks)
    chunk_size = _evaluation_group_chunk_size(total_groups, opt_cfg)

    started = time.monotonic()
    if progress_context:
        log.info(
            "Stage %s fold %d %s %s evaluate start batch %d/%d batch_candidates %d fold_candidates %d fold_progress %d/%d groups %d processes %d chunk_size %d progress_every %d progress_seconds %d",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
            progress_context.batch_index,
            progress_context.batch_count,
            total,
            progress_context.fold_total,
            progress_context.fold_completed_offset,
            progress_context.fold_total,
            total_groups,
            opt_cfg.processes if total > 1 else 1,
            chunk_size if opt_cfg.processes > 1 and total_groups > 1 else 1,
            opt_cfg.progress_log_every,
            opt_cfg.progress_log_seconds,
        )
    else:
        log.info(
            "Stage %s fold %d %s %s evaluate start candidates %d groups %d processes %d chunk_size %d progress_every %d progress_seconds %d",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
            total,
            total_groups,
            opt_cfg.processes if total > 1 else 1,
            chunk_size if opt_cfg.processes > 1 and total_groups > 1 else 1,
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
            _log_evaluation_progress(stage, fold, role, completed, total, progress, opt_cfg, progress_context)
        _log_evaluation_complete(stage, fold, role, total, started, progress_context)
        return results

    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(ctx_name)
    results: list[Evaluation] = []
    progress = _ProgressLogState(started_at=started, last_logged_at=started)
    with ctx.Pool(
        processes=opt_cfg.processes,
        initializer=_init_worker,
        initargs=(ticks, bars, base_cfg, fold, role, opt_cfg.profile_cache_size),
    ) as pool:
        completed = 0
        task_chunks = _chunk_evaluation_tasks(tasks, chunk_size)
        iterator = pool.imap_unordered(_evaluate_group_chunk_task, task_chunks, chunksize=1)
        while completed < total:
            try:
                evaluations = iterator.next(timeout=opt_cfg.progress_log_seconds)
            except mp.TimeoutError:
                elapsed = time.monotonic() - started
                if progress_context:
                    fold_completed = min(progress_context.fold_total, progress_context.fold_completed_offset + completed)
                    log.info(
                        "Stage %s fold %d %s %s waiting batch %d/%d batch_progress %d/%d fold_progress %d/%d elapsed %.1fs groups %d no_result_for %ds",
                        stage,
                        fold.fold_index,
                        role,
                        _fold_role_period(fold, role),
                        progress_context.batch_index,
                        progress_context.batch_count,
                        completed,
                        total,
                        fold_completed,
                        progress_context.fold_total,
                        elapsed,
                        total_groups,
                        opt_cfg.progress_log_seconds,
                    )
                else:
                    log.info(
                        "Stage %s fold %d %s %s waiting progress %d/%d elapsed %.1fs groups %d no_result_for %ds",
                        stage,
                        fold.fold_index,
                        role,
                        _fold_role_period(fold, role),
                        completed,
                        total,
                        elapsed,
                        total_groups,
                        opt_cfg.progress_log_seconds,
                    )
                continue
            except StopIteration:
                break
            results.extend(evaluations)
            completed += len(evaluations)
            _log_evaluation_progress(stage, fold, role, completed, total, progress, opt_cfg, progress_context)
        if completed < total:
            log.warning(
                "Stage %s fold %d %s %s evaluate ended with incomplete results %d/%d groups %d",
                stage,
                fold.fold_index,
                role,
                _fold_role_period(fold, role),
                completed,
                total,
                total_groups,
            )
    _log_evaluation_complete(stage, fold, role, total, started, progress_context)
    return results


@dataclass
class _ProgressLogState:
    started_at: float
    last_logged_at: float
    last_logged_completed: int = 0


def _format_ns_utc(value: int) -> str:
    return ns_to_datetime(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _period_text(start_ns: int, end_ns: int) -> str:
    return f"{_format_ns_utc(start_ns)}..{_format_ns_utc(end_ns)}"


def _fold_role_period(fold: FoldSpec, role: str) -> str:
    slice_start_ns, slice_end_ns, trade_start_ns, trade_end_ns = _window_bounds(fold, role)
    trade_period = _period_text(trade_start_ns, trade_end_ns)
    if slice_start_ns == trade_start_ns and slice_end_ns == trade_end_ns:
        return f"period {trade_period}"
    return f"trade_period {trade_period} data_period {_period_text(slice_start_ns, slice_end_ns)}"


def _log_evaluation_progress(
    stage: str,
    fold: FoldSpec,
    role: str,
    completed: int,
    total: int,
    progress: _ProgressLogState,
    opt_cfg: OptimizerConfig,
    progress_context: EvaluationProgressContext | None = None,
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
    if progress_context:
        fold_completed = min(progress_context.fold_total, progress_context.fold_completed_offset + completed)
        fold_remaining = max(0, progress_context.fold_total - fold_completed)
        eta_fold = fold_remaining / rate if rate > 0 else 0.0
        log.info(
            "Stage %s fold %d %s %s batch %d/%d batch_progress %d/%d fold_progress %d/%d elapsed %.1fs rate %.2f/s eta_batch %.1fs eta_fold %.1fs",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
            progress_context.batch_index,
            progress_context.batch_count,
            completed,
            total,
            fold_completed,
            progress_context.fold_total,
            elapsed,
            rate,
            eta,
            eta_fold,
        )
    else:
        log.info(
            "Stage %s fold %d %s %s progress %d/%d elapsed %.1fs rate %.2f/s eta %.1fs",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
            completed,
            total,
            elapsed,
            rate,
            eta,
        )
    progress.last_logged_at = now
    progress.last_logged_completed = completed


def _log_evaluation_complete(
    stage: str,
    fold: FoldSpec,
    role: str,
    total: int,
    started: float,
    progress_context: EvaluationProgressContext | None = None,
) -> None:
    elapsed = time.monotonic() - started
    rate = total / elapsed if elapsed > 0 else 0.0
    if progress_context:
        fold_completed = min(progress_context.fold_total, progress_context.fold_completed_offset + total)
        log.info(
            "Stage %s fold %d %s %s complete batch %d/%d batch_candidates %d fold_progress %d/%d elapsed %.1fs rate %.2f/s",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
            progress_context.batch_index,
            progress_context.batch_count,
            total,
            fold_completed,
            progress_context.fold_total,
            elapsed,
            rate,
        )
    else:
        log.info(
            "Stage %s fold %d %s %s complete candidates %d elapsed %.1fs rate %.2f/s",
            stage,
            fold.fold_index,
            role,
            _fold_role_period(fold, role),
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


def _chunk_evaluation_tasks(tasks: list[tuple], chunk_size: int) -> list[list[tuple]]:
    chunk_size = max(1, chunk_size)
    return [tasks[index : index + chunk_size] for index in range(0, len(tasks), chunk_size)]


def _evaluation_group_chunk_size(total_groups: int, opt_cfg: OptimizerConfig) -> int:
    if opt_cfg.processes <= 1 or total_groups <= 1:
        return 1
    target_chunks = max(int(opt_cfg.processes) * 4, int(opt_cfg.processes))
    dynamic_size = math.ceil(total_groups / target_chunks)
    return max(1, min(int(opt_cfg.process_chunk_size), int(dynamic_size)))


def _candidate_profile_key(values: dict[str, int | float], base_cfg: RunConfig) -> tuple:
    return _profile_cache_key(apply_parameter_values(base_cfg, values))


def _init_worker(
    ticks: TickData,
    bars: BarData,
    base_cfg: RunConfig,
    fold: FoldSpec,
    role: str,
    profile_cache_size: int,
) -> None:
    global _WORKER_BASE_CFG, _WORKER_WINDOW
    _WORKER_BASE_CFG = base_cfg
    _WORKER_WINDOW = _prepare_window_data(ticks, bars, fold, role, profile_cache_size)


def _evaluate_group_chunk_task(task_chunk: list[tuple]) -> list[Evaluation]:
    if _WORKER_BASE_CFG is None or _WORKER_WINDOW is None:
        raise RuntimeError("Worker data was not initialized")
    evaluations: list[Evaluation] = []
    for task in task_chunk:
        evaluations.extend(_evaluate_group_with_window(task, _WORKER_WINDOW, _WORKER_BASE_CFG))
    return evaluations


def _prepare_window_data(
    ticks: TickData,
    bars: BarData,
    fold: FoldSpec,
    role: str,
    profile_cache_size: int,
) -> WindowData:
    slice_start, slice_end, trade_start, trade_end = _window_bounds(fold, role)
    tick_slice = ticks.slice_time(slice_start, slice_end)
    bar_slice, bar_start_idx, _bar_end_idx = bars.slice_time(slice_start, slice_end)
    local_bar_index = tick_slice.bar_index.astype(np.int64, copy=False) - int(bar_start_idx)
    valid = (local_bar_index >= 0) & (local_bar_index < len(bar_slice))
    tick_bar_index = np.where(valid, local_bar_index, -1).astype(np.int32, copy=False)
    return WindowData(
        ticks=tick_slice,
        bars=bar_slice,
        tick_bar_index=tick_bar_index,
        trade_start_ns=trade_start,
        trade_end_ns=trade_end,
        profile_cache_size=profile_cache_size,
    )


def _window_bounds(fold: FoldSpec, role: str) -> tuple[int, int, int, int]:
    if role == "train":
        return fold.train_start_ns, fold.train_end_ns, fold.train_start_ns, fold.train_end_ns
    return fold.train_start_ns, fold.test_end_ns, fold.test_start_ns, fold.test_end_ns


def _evaluate_group_with_window(task, window: WindowData, base_cfg: RunConfig) -> list[Evaluation]:
    stage, group, fold, role, keep_trades = task
    if not group:
        return []
    profile_cfg = apply_parameter_values(base_cfg, group[0])
    profile = _profile_for_config(window, profile_cfg)
    # Crossing events depend only on the (shared) profile cross levels, the (shared)
    # session/trade-window mask and mid — never on the per-candidate grid params, so
    # detect them once per group instead of once per candidate.
    events = precompute_events(
        window.ticks,
        window.tick_bar_index,
        base_cfg,
        profile,
        window.trade_start_ns,
        window.trade_end_ns,
    )
    return [
        _evaluate_candidate_with_profile(stage, values, fold, role, keep_trades, window, base_cfg, profile, events)
        for values in group
    ]


def _evaluate_candidate_with_profile(
    stage: str,
    values: dict[str, int | float],
    fold: FoldSpec,
    role: str,
    keep_trades: bool,
    window: WindowData,
    base_cfg: RunConfig,
    profile: ProfileArrays,
    events: tuple,
) -> Evaluation:
    cfg = apply_parameter_values(base_cfg, values)
    if keep_trades:
        result = run_simulation(
            window.ticks,
            window.bars,
            window.tick_bar_index,
            cfg,
            trade_start_ns=window.trade_start_ns,
            trade_end_ns=window.trade_end_ns,
            log_result=False,
            profile=profile,
            events=events,
        )
        summary = summarize_trades(result.trades, cfg.initial_equity, result.final_equity)
        session_stats = summarize_trades_by_session(result.trades)
    else:
        result, summary, session_stats = run_simulation_summary(
            window.ticks,
            window.bars,
            window.tick_bar_index,
            cfg,
            trade_start_ns=window.trade_start_ns,
            trade_end_ns=window.trade_end_ns,
            profile=profile,
            events=events,
        )
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
        window_start=ns_to_datetime(window.trade_start_ns),
        window_end=ns_to_datetime(window.trade_end_ns),
        ticks_simulated=result.ticks_simulated,
        bars_total=result.bars_total,
        signals_total=result.signals_total,
        long_signals=result.long_signals,
        short_signals=result.short_signals,
        rejected_missing_band=result.rejected_signals_missing_band,
        rejected_band_too_narrow=result.rejected_signals_band_too_narrow,
        rejected_price_range_position=result.rejected_signals_price_range_position,
        rejected_stop_too_small=result.rejected_signals_stop_too_small,
        rejected_stop_too_large=result.rejected_signals_stop_too_large,
        skipped_no_size=result.skipped_signals_no_size,
        ruined=result.ruined,
        summary=summary,
        score=score,
        session_stats=session_stats,
        trades=result.trades,
    )


def _profile_for_config(window: WindowData, cfg: RunConfig) -> ProfileArrays:
    if window.profile_cache_size <= 0:
        return rolling_profile_arrays(window.bars, cfg)

    key = _profile_cache_key(cfg)
    cached = window.profile_cache.get(key)
    if cached is not None:
        window.profile_cache.move_to_end(key)
        return cached

    profile = rolling_profile_arrays(window.bars, cfg)
    window.profile_cache[key] = profile
    while len(window.profile_cache) > window.profile_cache_size:
        window.profile_cache.popitem(last=False)
    return profile


def _profile_cache_key(cfg: RunConfig) -> tuple:
    return (
        cfg.lookback_bars,
        cfg.min_lookback_bars,
        cfg.profile_max_lookback_seconds,
        round(float(cfg.price_step), 8),
        round(float(cfg.median_quantile), 8),
        round(float(cfg.band_lower_quantile), 8),
        round(float(cfg.band_upper_quantile), 8),
        round(float(cfg.long_cross_quantile), 8),
        round(float(cfg.short_cross_quantile), 8),
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
            "mc_prob_of_ruin_pct": None,
            "passed_pre_mc_filters": passed_pre_mc_filters(oos_metrics, opt_cfg),
            "passed_filters": False,
            "oos_full_coverage": _oos_full_coverage(oos_metrics, expected_folds),
            **_prefix_metrics("train", train_metrics),
            **_prefix_metrics("oos", oos_metrics),
        }
        aggregates.append(row)
    return aggregates


def build_session_aggregates(
    stage: str,
    candidates: list[dict[str, int | float]],
    train_by_hash: dict[str, list[Evaluation]],
    oos_by_hash: dict[str, list[Evaluation]],
    expected_folds: int,
    role_sources: tuple[tuple[str, dict[str, list[Evaluation]]], ...] | None = None,
) -> list[dict]:
    if role_sources is None:
        role_sources = (("train", train_by_hash), ("oos", oos_by_hash))
    rows = []
    for values in candidates:
        digest = parameters.parameter_hash(values)
        for window_role, by_hash in role_sources:
            evaluations = by_hash.get(digest, [])
            session_metrics = aggregate_session_evaluations(evaluations, expected_folds)
            for session_type, _label, _sort_order in SESSION_TYPES:
                rows.append(
                    {
                        "stage": stage,
                        "parameter_hash": digest,
                        "window_role": window_role,
                        "session_type": session_type,
                        "session_label": SESSION_LABELS[session_type],
                        "session_sort_order": SESSION_SORT_ORDERS[session_type],
                        **session_metrics[session_type],
                    }
                )
    return rows


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


def aggregate_session_evaluations(evaluations: list[Evaluation], expected_folds: int) -> dict[str, dict]:
    out = {
        session_type: {
            "folds": len(evaluations),
            "expected_folds": expected_folds,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "breakeven_trades": 0,
            "win_rate_pct": 0.0,
            "gross_profit_eur": 0.0,
            "gross_loss_eur": 0.0,
            "net_profit_eur": 0.0,
            "avg_trade_pnl_eur": 0.0,
        }
        for session_type, _label, _sort_order in SESSION_TYPES
    }
    for evaluation in evaluations:
        for session_type, stats in evaluation.session_stats.items():
            row = out.get(session_type)
            if row is None:
                continue
            row["total_trades"] += int(stats.get("total_trades") or 0)
            row["winning_trades"] += int(stats.get("winning_trades") or 0)
            row["losing_trades"] += int(stats.get("losing_trades") or 0)
            row["breakeven_trades"] += int(stats.get("breakeven_trades") or 0)
            row["gross_profit_eur"] += float(stats.get("gross_profit_eur") or 0.0)
            row["gross_loss_eur"] += float(stats.get("gross_loss_eur") or 0.0)
            row["net_profit_eur"] += float(stats.get("net_profit_eur") or 0.0)

    for row in out.values():
        total = int(row["total_trades"])
        row["win_rate_pct"] = round(float(row["winning_trades"]) / total * 100.0, 4) if total > 0 else 0.0
        row["gross_profit_eur"] = round(float(row["gross_profit_eur"]), 2)
        row["gross_loss_eur"] = round(float(row["gross_loss_eur"]), 2)
        row["net_profit_eur"] = round(float(row["net_profit_eur"]), 2)
        row["avg_trade_pnl_eur"] = round(float(row["net_profit_eur"]) / total, 4) if total > 0 else 0.0
    return out


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
    stage: str | None = None,
    folds: list[FoldSpec] | None = None,
    ticks: TickData | None = None,
    bars: BarData | None = None,
) -> dict[str, dict]:
    if opt_cfg.mc_score_top_n <= 0 or not base_cfg.monte_carlo_enabled:
        return {}
    ranked = [
        row for row in sorted(aggregates, key=lambda item: item["pre_mc_score"], reverse=True)
        if row["oos_folds"] > 0 and row["oos_total_trades"] > 1
    ][: opt_cfg.mc_score_top_n]
    trade_evals_by_hash = oos_by_hash
    missing_trade_evals = [
        evaluation
        for row in ranked
        for evaluation in oos_by_hash.get(row["parameter_hash"], [])
        if int(evaluation.summary.get("total_trades") or 0) > 0 and not evaluation.trades
    ]
    if missing_trade_evals:
        if stage is None or folds is None or ticks is None or bars is None:
            log.warning("Monte-Carlo trade recompute skipped missing window data evaluations %d", len(missing_trade_evals))
        else:
            recomputed = _recompute_oos_evaluations_with_trades(stage, missing_trade_evals, folds, base_cfg, opt_cfg, ticks, bars)
            trade_evals_by_hash = dict(oos_by_hash)
            for evaluation in recomputed:
                existing = [
                    item
                    for item in trade_evals_by_hash.get(evaluation.parameter_hash, [])
                    if item.fold_index != evaluation.fold_index or item.window_role != evaluation.window_role
                ]
                existing.append(evaluation)
                trade_evals_by_hash[evaluation.parameter_hash] = existing

    out: dict[str, dict] = {}
    for rank, row in enumerate(ranked, start=1):
        trades = _oos_trades_for_hash(trade_evals_by_hash, row["parameter_hash"])
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


def persist_top_trades(
    conn,
    run_id: int,
    stage: StageResult,
    opt_cfg: OptimizerConfig,
    base_cfg: RunConfig,
    folds: list[FoldSpec],
    ticks: TickData,
    bars: BarData,
) -> None:
    if opt_cfg.persist_top_trades_n <= 0:
        return
    top = [
        row for row in sorted(stage.aggregates, key=lambda item: item["score"], reverse=True)
        if row["oos_folds"] > 0 and row["parameter_hash"] in stage.parameter_ids
    ][: opt_cfg.persist_top_trades_n]
    hashes = {row["parameter_hash"] for row in top}
    if not hashes:
        return
    top_evals = [evaluation for evaluation in stage.oos_evals if evaluation.parameter_hash in hashes]
    missing_trade_evals = [
        evaluation
        for evaluation in top_evals
        if int(evaluation.summary.get("total_trades") or 0) > 0 and not evaluation.trades
    ]
    if missing_trade_evals:
        recomputed = _recompute_oos_evaluations_with_trades(stage.stage, missing_trade_evals, folds, base_cfg, opt_cfg, ticks, bars)
        replacements = {
            (evaluation.parameter_hash, evaluation.fold_index, evaluation.window_role): evaluation
            for evaluation in recomputed
        }
        top_evals = [
            replacements.get((evaluation.parameter_hash, evaluation.fold_index, evaluation.window_role), evaluation)
            for evaluation in top_evals
        ]

    rows = []
    for evaluation in top_evals:
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


def _recompute_oos_evaluations_with_trades(
    stage: str,
    evaluations: list[Evaluation],
    folds: list[FoldSpec],
    base_cfg: RunConfig,
    opt_cfg: OptimizerConfig,
    ticks: TickData,
    bars: BarData,
) -> list[Evaluation]:
    if not evaluations:
        return []

    fold_by_index = {fold.fold_index: fold for fold in folds}
    by_fold: OrderedDict[int, list[Evaluation]] = OrderedDict()
    for evaluation in evaluations:
        by_fold.setdefault(evaluation.fold_index, []).append(evaluation)

    out: list[Evaluation] = []
    for fold_index, fold_evals in by_fold.items():
        fold = fold_by_index.get(fold_index)
        if fold is None:
            log.warning("Skipping trade recompute for missing fold %d evaluations %d", fold_index, len(fold_evals))
            continue
        candidates = [evaluation.values for evaluation in fold_evals]
        log.info("Recomputing OOS trades stage %s fold %d candidates %d", stage, fold_index, len(candidates))
        out.extend(evaluate_many(stage, candidates, fold, "oos", base_cfg, opt_cfg, ticks, bars, keep_trades=True))
    return out


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
