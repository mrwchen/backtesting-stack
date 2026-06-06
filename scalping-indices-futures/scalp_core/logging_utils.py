"""Logging configuration."""

import logging
import os
import time


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(
            UtcFormatter(
                "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
                "%Y-%m-%dT%H:%M:%S",
            )
        )
    # arch / statsmodels are chatty at INFO during repeated fits.
    logging.getLogger("arch").setLevel(logging.WARNING)
    logging.getLogger("statsmodels").setLevel(logging.WARNING)
