"""Backtest process dispatcher."""

from .app import run_single_model_worker
from .config import BACKTEST_PARALLEL_CHILD
from .parallel import run_parallel_parent


def main() -> None:
    if BACKTEST_PARALLEL_CHILD:
        run_single_model_worker()
    else:
        run_parallel_parent()
