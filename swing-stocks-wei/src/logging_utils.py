"""Logging setup using the required compact positional UTC format."""
from __future__ import annotations

import logging
import time


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        UtcFormatter(
            "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
