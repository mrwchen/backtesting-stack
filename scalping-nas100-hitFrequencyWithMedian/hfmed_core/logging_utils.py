"""Logging configuration."""

import logging
import os
import time


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    fmt = "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
    )
    for handler in logging.getLogger().handlers:
        handler.setFormatter(UtcFormatter(fmt, datefmt))

