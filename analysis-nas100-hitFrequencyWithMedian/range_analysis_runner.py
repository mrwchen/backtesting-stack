"""Entrypoint for the NAS100 hit-frequency range analysis."""

from __future__ import annotations

import logging

from hfmed_range_analysis.logging_utils import configure_logging
from hfmed_range_analysis.main import main


if __name__ == "__main__":
    configure_logging()
    try:
        main()
    except Exception:
        logging.getLogger(__name__).exception("NAS100 hit-frequency range analysis failed")
        raise

