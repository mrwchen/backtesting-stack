"""Parent process orchestration for running multiple model files in parallel."""

import logging
import os
import re
import signal
import subprocess
import sys
import time as _time
from pathlib import Path

from .config import *
from .logging_utils import set_log_process_name
from .model_loader import select_model_files

log = logging.getLogger(__name__)

def _pg_application_name_for_model(model_file: str) -> str:
    model_stem = Path(model_file).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_stem)
    return f"backtest_runner:{safe_stem}"[:63]


def _child_env(model_file: str) -> dict[str, str]:
    env = os.environ.copy()
    env["BACKTEST_PARALLEL_CHILD"] = "1"
    env["MODEL_SELECTION"] = "single"
    env["MODEL_FILE"] = model_file
    env["PGAPPNAME"] = _pg_application_name_for_model(model_file)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _terminate_running_workers(running: dict[subprocess.Popen, str]) -> None:
    for process, model_file in running.items():
        if process.poll() is None:
            log.warning("Terminating model worker pid %d model %s", process.pid, model_file)
            process.terminate()
    for process, model_file in running.items():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.error("Killing model worker pid %d model %s", process.pid, model_file)
            process.kill()
            process.wait(timeout=30)


def run_parallel_parent() -> None:
    set_log_process_name("bt-parent")
    if MODEL_FAILURE_MODE not in {"fail_fast", "continue"}:
        raise ValueError("MODEL_FAILURE_MODE must be one of: fail_fast, continue")

    model_files = select_model_files()
    parallelism = min(MODEL_PARALLELISM, len(model_files))
    script_path = PROJECT_ROOT / "backtest_runner.py"
    pending = list(model_files)
    running: dict[subprocess.Popen, str] = {}
    succeeded: list[str] = []
    failed: list[tuple[str, int]] = []

    def _handle_shutdown(signum: int, _frame: object) -> None:
        log.warning("Parallel parent shutdown requested signal %d with %d workers running", signum, len(running))
        _terminate_running_workers(running)
        raise SystemExit(128 + signum)

    previous_handlers = {
        signal.SIGTERM: signal.getsignal(signal.SIGTERM),
        signal.SIGINT: signal.getsignal(signal.SIGINT),
    }
    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    log.info(
        "Parallel backtest orchestration starting models %d parallelism %d failure mode %s files %s",
        len(model_files),
        parallelism,
        MODEL_FAILURE_MODE,
        ",".join(model_files),
    )

    try:
        while pending or running:
            while pending and len(running) < parallelism:
                model_file = pending.pop(0)
                command = [sys.executable, str(script_path)]
                process = subprocess.Popen(command, env=_child_env(model_file))
                running[process] = model_file
                log.info(
                    "Started model worker pid %d model %s slot %d/%d",
                    process.pid,
                    model_file,
                    len(running),
                    parallelism,
                )

            finished: subprocess.Popen | None = None
            while finished is None:
                for process in list(running):
                    if process.poll() is not None:
                        finished = process
                        break
                if finished is None:
                    _time.sleep(1.0)

            model_file = running.pop(finished)
            exit_code = int(finished.returncode or 0)
            if exit_code == 0:
                succeeded.append(model_file)
                log.info("Model worker finished model %s exit code 0", model_file)
                continue

            failed.append((model_file, exit_code))
            log.error("Model worker failed model %s exit code %d", model_file, exit_code)
            if MODEL_FAILURE_MODE == "fail_fast":
                _terminate_running_workers(running)
                raise SystemExit(exit_code if exit_code > 0 else 1)
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    log.info(
        "Parallel backtest orchestration complete succeeded %d failed %d",
        len(succeeded),
        len(failed),
    )
    if failed:
        log.error(
            "Parallel backtest failures — %s",
            ", ".join(f"{model}:{code}" for model, code in failed),
        )
        raise SystemExit(1)
