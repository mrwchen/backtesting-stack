"""Parent process orchestration for running multiple model files in parallel."""

import logging
import os
import re
import signal
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

from .config import *
from .db import connect_with_retry, validate_result_schema
from .logging_utils import set_log_process_name
from .model_loader import select_model_files
from .persistence import reserve_run_ids

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestJob:
    model_file: str
    account_profile: str

    @property
    def label(self) -> str:
        return f"{self.model_file}:{self.account_profile}"


def _selected_jobs(model_files: list[str]) -> list[BacktestJob]:
    account_profiles = (
        list(ACCOUNT_PROFILE_DEFAULTS)
        if ACCOUNT_PROFILE_REQUEST == "all"
        else [ACCOUNT_PROFILE]
    )
    return [
        BacktestJob(model_file=model_file, account_profile=account_profile)
        for model_file in model_files
        for account_profile in account_profiles
    ]


def _pg_application_name_for_job(job: BacktestJob) -> str:
    model_stem = Path(job.model_file).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_stem)
    safe_profile = re.sub(r"[^A-Za-z0-9_.-]+", "_", job.account_profile)
    prefix = "backtest_runner:"
    suffix = f":{safe_profile}"
    max_stem_len = max(1, 63 - len(prefix) - len(suffix))
    return f"{prefix}{safe_stem[:max_stem_len]}{suffix}"


def _child_env(job: BacktestJob, reserved_run_id: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["BACKTEST_PARALLEL_CHILD"] = "1"
    env["MODEL_SELECTION"] = "single"
    env["MODEL_FILE"] = job.model_file
    env["ACCOUNT_PROFILE"] = job.account_profile
    env["PGAPPNAME"] = _pg_application_name_for_job(job)
    env.pop("BACKTEST_RUN_ID", None)
    if reserved_run_id is not None:
        env["BACKTEST_RUN_ID"] = str(reserved_run_id)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def _terminate_running_workers(running: dict[subprocess.Popen, BacktestJob]) -> None:
    for process, job in running.items():
        if process.poll() is None:
            log.warning("Terminating model worker pid %d job %s", process.pid, job.label)
            process.terminate()
    for process, job in running.items():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            log.error("Killing model worker pid %d job %s", process.pid, job.label)
            process.kill()
            process.wait(timeout=30)


def _reserve_run_ids_for_jobs(jobs: list[BacktestJob]) -> dict[BacktestJob, int]:
    if not jobs:
        return {}
    if GRID_SEARCH_ENABLED:
        log.warning("Run id reservation disabled because grid search can create multiple runs per worker")
        return {}

    conn = connect_with_retry()
    try:
        validate_result_schema(conn)
        run_ids = reserve_run_ids(conn, len(jobs))
    finally:
        conn.close()

    reservations = dict(zip(jobs, run_ids))
    log.info(
        "Reserved run ids for model queue first %s id %d last %s id %d",
        jobs[0].label,
        reservations[jobs[0]],
        jobs[-1].label,
        reservations[jobs[-1]],
    )
    return reservations

def run_parallel_parent() -> None:
    set_log_process_name("bt-parent")
    if MODEL_FAILURE_MODE not in {"fail_fast", "continue"}:
        raise ValueError("MODEL_FAILURE_MODE must be one of: fail_fast, continue")

    model_files = select_model_files()
    jobs = _selected_jobs(model_files)
    script_path = PROJECT_ROOT / "backtest_runner.py"
    reserved_run_ids = _reserve_run_ids_for_jobs(jobs)
    parallelism = min(MODEL_PARALLELISM, len(jobs))
    pending = list(jobs)
    running: dict[subprocess.Popen, BacktestJob] = {}
    succeeded: list[BacktestJob] = []
    failed: list[tuple[BacktestJob, int]] = []

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
        "Parallel backtest orchestration starting models %d jobs %d parallelism %d failure mode %s run id reservation %s files %s profiles %s",
        len(model_files),
        len(jobs),
        parallelism,
        MODEL_FAILURE_MODE,
        "enabled" if reserved_run_ids else "disabled",
        ",".join(model_files),
        ",".join(sorted({job.account_profile for job in jobs})),
    )

    try:
        while pending or running:
            while pending and len(running) < parallelism:
                job = pending.pop(0)
                reserved_run_id = reserved_run_ids.get(job)
                command = [sys.executable, str(script_path)]
                process = subprocess.Popen(command, env=_child_env(job, reserved_run_id))
                running[process] = job
                log.info(
                    "Started model worker pid %d model %s account profile %s reserved run id %s slot %d/%d",
                    process.pid,
                    job.model_file,
                    job.account_profile,
                    str(reserved_run_id) if reserved_run_id is not None else "-",
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

            job = running.pop(finished)
            exit_code = int(finished.returncode or 0)
            if exit_code == 0:
                succeeded.append(job)
                log.info("Model worker finished model %s account profile %s exit code 0", job.model_file, job.account_profile)
                continue

            failed.append((job, exit_code))
            log.error("Model worker failed model %s account profile %s exit code %d", job.model_file, job.account_profile, exit_code)
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
            ", ".join(f"{job.label}:{code}" for job, code in failed),
        )
        raise SystemExit(1)
