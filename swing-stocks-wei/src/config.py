"""All runtime parameters come from environment variables (set in compose.yaml)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date


def _env_date(name: str, default: str) -> date:
    raw = os.getenv(name, "").strip() or default
    return date.fromisoformat(raw)


@dataclass(frozen=True)
class Config:
    symbol: str
    start_date: date
    end_date: date
    ema_fast: int
    ema_slow: int
    stress_enter: float
    stress_exit: float
    cost_bps_per_side: float
    warmup_calendar_days: int
    run_label: str
    log_level: str

    @staticmethod
    def from_env() -> "Config":
        cfg = Config(
            symbol=os.getenv("SYMBOL", "VOO"),
            start_date=_env_date("START_DATE", "2022-01-03"),
            end_date=_env_date("END_DATE", date.today().isoformat()),
            ema_fast=int(os.getenv("EMA_FAST", "9")),
            ema_slow=int(os.getenv("EMA_SLOW", "21")),
            stress_enter=float(os.getenv("STRESS_ENTER", "57")),
            stress_exit=float(os.getenv("STRESS_EXIT", "52")),
            cost_bps_per_side=float(os.getenv("COST_BPS_PER_SIDE", "5")),
            warmup_calendar_days=int(os.getenv("WARMUP_CALENDAR_DAYS", "365")),
            run_label=os.getenv("RUN_LABEL", "wei_regime_ema"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        if cfg.ema_fast >= cfg.ema_slow:
            raise ValueError("EMA_FAST must be smaller than EMA_SLOW")
        if cfg.stress_exit >= cfg.stress_enter:
            raise ValueError("STRESS_EXIT must be below STRESS_ENTER (hysteresis)")
        if cfg.start_date >= cfg.end_date:
            raise ValueError("START_DATE must be before END_DATE")
        return cfg
