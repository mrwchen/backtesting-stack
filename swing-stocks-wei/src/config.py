"""All runtime parameters come from environment variables (set in compose.yaml)."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date


def _env_date(name: str, default: str) -> date:
    raw = os.getenv(name, "").strip() or default
    return date.fromisoformat(raw)


def _env_categories(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "").strip()
    return tuple(c.strip() for c in raw.split(",") if c.strip()) if raw else ()


@dataclass(frozen=True)
class Config:
    start_date: date
    end_date: date
    categories: tuple[str, ...]
    top_n_per_category: int
    min_market_cap_usd: int
    min_coverage_pct: float
    ema_fast: int
    ema_slow: int
    stress_enter: float
    stress_exit: float
    cat_mom_window: int
    cat_mom_deep_threshold: float
    weight_deep_pct: float
    weight_mild_pct: float
    weight_pos_pct: float
    max_positions: int
    max_per_category: int
    cost_bps_per_side: float
    warmup_calendar_days: int
    run_label: str
    table_prefix: str
    metrics_table: str
    symbols_table: str
    scores_table: str
    log_level: str

    @property
    def runs_table(self) -> str:
        return f"{self.table_prefix}runs"

    @property
    def trades_table(self) -> str:
        return f"{self.table_prefix}trades"

    @property
    def equity_table(self) -> str:
        return f"{self.table_prefix}equity_daily"

    @staticmethod
    def from_env() -> "Config":
        cfg = Config(
            start_date=_env_date("START_DATE", "2022-01-03"),
            end_date=_env_date("END_DATE", date.today().isoformat()),
            categories=_env_categories("CATEGORIES"),
            top_n_per_category=int(os.getenv("TOP_N_PER_CATEGORY", "8")),
            min_market_cap_usd=int(os.getenv("MIN_MARKET_CAP_USD", "2000000000")),
            min_coverage_pct=float(os.getenv("MIN_COVERAGE_PCT", "90")),
            ema_fast=int(os.getenv("EMA_FAST", "9")),
            ema_slow=int(os.getenv("EMA_SLOW", "21")),
            stress_enter=float(os.getenv("STRESS_ENTER", "57")),
            stress_exit=float(os.getenv("STRESS_EXIT", "52")),
            cat_mom_window=int(os.getenv("CAT_MOM_WINDOW", "63")),
            cat_mom_deep_threshold=float(os.getenv("CAT_MOM_DEEP_THRESHOLD", "-0.10")),
            weight_deep_pct=float(os.getenv("WEIGHT_DEEP_PCT", "5")),
            weight_mild_pct=float(os.getenv("WEIGHT_MILD_PCT", "3")),
            weight_pos_pct=float(os.getenv("WEIGHT_POS_PCT", "2")),
            max_positions=int(os.getenv("MAX_POSITIONS", "25")),
            max_per_category=int(os.getenv("MAX_PER_CATEGORY", "2")),
            cost_bps_per_side=float(os.getenv("COST_BPS_PER_SIDE", "5")),
            warmup_calendar_days=int(os.getenv("WARMUP_CALENDAR_DAYS", "365")),
            run_label=os.getenv("RUN_LABEL", "wei_stocks_regime_ema"),
            table_prefix=os.getenv("TABLE_PREFIX", "backtest_wei_stocks_"),
            metrics_table=os.getenv("METRICS_TABLE", "stock_core_market_metrics_daily"),
            symbols_table=os.getenv("SYMBOLS_TABLE", "ibkr_symbols"),
            scores_table=os.getenv("SCORES_TABLE", "world_regime_daily_scores_mv"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
        if not re.fullmatch(r"[a-z_][a-z0-9_]*", cfg.table_prefix):
            raise ValueError("TABLE_PREFIX must be a valid lowercase SQL identifier prefix")
        for name, value in (("METRICS_TABLE", cfg.metrics_table),
                            ("SYMBOLS_TABLE", cfg.symbols_table),
                            ("SCORES_TABLE", cfg.scores_table)):
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", value):
                raise ValueError(f"{name} must be a valid lowercase SQL identifier")
        if cfg.ema_fast >= cfg.ema_slow:
            raise ValueError("EMA_FAST must be smaller than EMA_SLOW")
        if cfg.stress_exit >= cfg.stress_enter:
            raise ValueError("STRESS_EXIT must be below STRESS_ENTER (hysteresis)")
        if cfg.start_date >= cfg.end_date:
            raise ValueError("START_DATE must be before END_DATE")
        if cfg.cat_mom_deep_threshold >= 0:
            raise ValueError("CAT_MOM_DEEP_THRESHOLD must be negative (a drawdown)")
        if cfg.max_per_category < 1 or cfg.max_positions < 1:
            raise ValueError("MAX_POSITIONS and MAX_PER_CATEGORY must be >= 1")
        if min(cfg.weight_deep_pct, cfg.weight_mild_pct, cfg.weight_pos_pct) <= 0:
            raise ValueError("all WEIGHT_*_PCT must be positive")
        if not 0 < cfg.min_coverage_pct <= 100:
            raise ValueError("MIN_COVERAGE_PCT must be in (0, 100]")
        return cfg
