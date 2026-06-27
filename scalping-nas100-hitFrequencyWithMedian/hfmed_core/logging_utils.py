"""Logging configuration."""

import logging
import os
import time
from pathlib import Path


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


def log_resource_snapshot(logger: logging.Logger, label: str) -> None:
    """Log process and cgroup memory without adding external dependencies."""
    proc = _read_proc_status()
    cgroup_current = _read_int_path(
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    )
    cgroup_max = _read_cgroup_max(
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    )
    logger.info(
        "Resource snapshot %s rss_mb %.1f hwm_mb %.1f vms_mb %.1f cgroup_current_mb %.1f cgroup_max_mb %.1f threads %s",
        label,
        _kb_to_mb(proc.get("VmRSS")),
        _kb_to_mb(proc.get("VmHWM")),
        _kb_to_mb(proc.get("VmSize")),
        _bytes_to_mb(cgroup_current),
        _bytes_to_mb(cgroup_max),
        proc.get("Threads", "unknown"),
    )


def _read_proc_status() -> dict[str, int | str]:
    out: dict[str, int | str] = {}
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            parts = raw_value.strip().split()
            if not parts:
                continue
            if parts[0].isdigit():
                out[key] = int(parts[0])
            else:
                out[key] = parts[0]
    except OSError:
        pass
    return out


def _read_int_path(*paths: str) -> int | None:
    for path in paths:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value.isdigit():
            return int(value)
    return None


def _read_cgroup_max(*paths: str) -> int | None:
    for path in paths:
        try:
            value = Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value == "max":
            return None
        if value.isdigit():
            return int(value)
    return None


def _kb_to_mb(value: int | str | None) -> float:
    if not isinstance(value, int):
        return 0.0
    return value / 1024.0


def _bytes_to_mb(value: int | None) -> float:
    if value is None:
        return 0.0
    return value / 1024.0 / 1024.0
