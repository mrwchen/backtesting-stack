"""Backtest process dispatcher."""

from .app import run_shared_candidate_timeline_prebuilder, run_single_model_worker
from .config import BACKTEST_PARALLEL_CHILD, BACKTEST_SHARED_TIMELINE_PREBUILDER
from .parallel import run_parallel_parent


def main() -> None:
    if BACKTEST_SHARED_TIMELINE_PREBUILDER:
        run_shared_candidate_timeline_prebuilder()
    elif BACKTEST_PARALLEL_CHILD:
        run_single_model_worker()
    else:
        run_parallel_parent()
