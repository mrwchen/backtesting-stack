"""Logging setup shared by parent and model worker processes."""

import logging
import multiprocessing
import os
import re
import time as _time

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.Formatter.converter = _time.gmtime
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    _CONFIGURED = True


def set_log_process_name(name: str) -> None:
    """Set Python logging's processName for clearer interleaved child logs."""
    safe = re.sub(r"[^A-Za-z0-9_.:-]+", "_", name).strip("_")
    multiprocessing.current_process().name = safe[:48] or "backtest"
