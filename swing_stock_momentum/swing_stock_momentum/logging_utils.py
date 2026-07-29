from __future__ import annotations

import logging
import time


LOG_FORMAT = "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s"
LOG_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        force=True,
    )
    logging.Formatter.converter = time.gmtime
