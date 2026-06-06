"""Logging configuration."""

import logging
import os


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # arch / statsmodels are chatty at INFO during repeated fits.
    logging.getLogger("arch").setLevel(logging.WARNING)
    logging.getLogger("statsmodels").setLevel(logging.WARNING)
