from __future__ import annotations

import logging

from .config import Config
from .logging_utils import configure_logging
from .runner import run


def main() -> None:
    cfg = Config.from_env()
    configure_logging(cfg.log_level)
    try:
        run(cfg)
    except Exception:
        logging.getLogger(__name__).exception("Swing-stock momentum backtest failed")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
