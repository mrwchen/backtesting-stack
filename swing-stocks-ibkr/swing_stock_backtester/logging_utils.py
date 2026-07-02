import logging
import time


class UtcFormatter(logging.Formatter):
    converter = time.gmtime


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    formatter = UtcFormatter(
        fmt="%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    root.addHandler(handler)
    root.setLevel(level)
