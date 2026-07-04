"""Environment-driven configuration for the Wei swing backtester."""
from __future__ import annotations

import os
from dataclasses import dataclass
from hashlib import sha256


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    strategy_version: str
    start_date: str
    end_date: str | None
    warmup_calendar_days: int
    run_label: str
    cache_dir: str
    force_refresh: bool
    log_level: str
    min_price: float
    min_market_cap_usd: float
    revenue_yoy_min: float
    revenue_stale_trading_days: int
    high_lookback_days: int
    high_recent_days: int
    min_pullback_pct: float
    ema_fast_days: int
    ema_slow_days: int
    ema_cross_lookback_days: int
    max_entry_gap_pct: float
    atr_days: int
    initial_stop_mode: str
    atr_stop_multiple: float
    volume_sma_days: int
    volume_filter_enable: bool
    ibkr_category_breadth_filter_enable: bool
    ibkr_category_breadth_on_threshold: float
    ibkr_category_breadth_off_threshold: float
    ibkr_category_breadth_min_symbols: int
    initial_equity: float
    position_size_usd: float
    stop_loss_pct: float
    trailing_activate_pct: float
    trailing_loss_pct: float
    allow_fractional_shares: bool

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            strategy_version=_env("STRATEGY_VERSION", "wei_pullback_reclaim_v2"),
            start_date=_env("START_DATE", "2020-01-02"),
            end_date=_env("END_DATE", "") or None,
            warmup_calendar_days=int(_env("WARMUP_CALENDAR_DAYS", "550")),
            run_label=_env("RUN_LABEL", "wei_52w_pullback_v1"),
            cache_dir=_env("CACHE_DIR", "/cache"),
            force_refresh=_env_bool("FORCE_REFRESH", False),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            min_price=float(_env("MIN_PRICE", "0.01")),
            min_market_cap_usd=float(_env("MIN_MARKET_CAP_USD", "2000000000")),
            revenue_yoy_min=float(_env("REVENUE_YOY_MIN", "0.20")),
            revenue_stale_trading_days=int(_env("REVENUE_STALE_TRADING_DAYS", "280")),
            high_lookback_days=int(_env("HIGH_LOOKBACK_DAYS", "252")),
            high_recent_days=int(_env("HIGH_RECENT_DAYS", "10")),
            min_pullback_pct=float(_env("MIN_PULLBACK_PCT", "0.03")),
            ema_fast_days=int(_env("EMA_FAST_DAYS", "9")),
            ema_slow_days=int(_env("EMA_SLOW_DAYS", "21")),
            ema_cross_lookback_days=int(_env("EMA_CROSS_LOOKBACK_DAYS", "8")),
            max_entry_gap_pct=float(_env("MAX_ENTRY_GAP_PCT", "0.02")),
            atr_days=int(_env("ATR_DAYS", "14")),
            initial_stop_mode=_env("INITIAL_STOP_MODE", "fixed_pct").lower(),
            atr_stop_multiple=float(_env("ATR_STOP_MULTIPLE", "2.0")),
            volume_sma_days=int(_env("VOLUME_SMA_DAYS", "50")),
            volume_filter_enable=_env_bool("VOLUME_FILTER_ENABLE", True),
            ibkr_category_breadth_filter_enable=_env_bool("IBKR_CATEGORY_BREADTH_FILTER_ENABLE", True),
            ibkr_category_breadth_on_threshold=float(_env("IBKR_CATEGORY_BREADTH_ON_THRESHOLD", "0.65")),
            ibkr_category_breadth_off_threshold=float(_env("IBKR_CATEGORY_BREADTH_OFF_THRESHOLD", "0.55")),
            ibkr_category_breadth_min_symbols=int(_env("IBKR_CATEGORY_BREADTH_MIN_SYMBOLS", "5")),
            initial_equity=float(_env("INITIAL_EQUITY", "100000")),
            position_size_usd=float(_env("POSITION_SIZE_USD", "1000")),
            stop_loss_pct=float(_env("STOP_LOSS_PCT", "0.05")),
            trailing_activate_pct=float(_env("TRAILING_ACTIVATE_PCT", "0.10")),
            trailing_loss_pct=float(_env("TRAILING_LOSS_PCT", "0.05")),
            allow_fractional_shares=_env_bool("ALLOW_FRACTIONAL_SHARES", True),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.strategy_version:
            raise ValueError("STRATEGY_VERSION must not be empty")
        if self.warmup_calendar_days < 365:
            raise ValueError("WARMUP_CALENDAR_DAYS should cover at least one 52-week lookback")
        if self.min_price <= 0:
            raise ValueError("MIN_PRICE must be > 0")
        if self.min_market_cap_usd < 0:
            raise ValueError("MIN_MARKET_CAP_USD must be >= 0")
        if self.revenue_yoy_min < -1:
            raise ValueError("REVENUE_YOY_MIN must be >= -1")
        if self.revenue_stale_trading_days < 1:
            raise ValueError("REVENUE_STALE_TRADING_DAYS must be >= 1")
        if self.high_lookback_days < 2:
            raise ValueError("HIGH_LOOKBACK_DAYS must be >= 2")
        if self.high_recent_days < 1:
            raise ValueError("HIGH_RECENT_DAYS must be >= 1")
        if self.min_pullback_pct < 0:
            raise ValueError("MIN_PULLBACK_PCT must be >= 0")
        if self.ema_fast_days < 1:
            raise ValueError("EMA_FAST_DAYS must be >= 1")
        if self.ema_slow_days <= self.ema_fast_days:
            raise ValueError("EMA_SLOW_DAYS must be greater than EMA_FAST_DAYS")
        if self.ema_cross_lookback_days < 0:
            raise ValueError("EMA_CROSS_LOOKBACK_DAYS must be >= 0")
        if self.max_entry_gap_pct < -1:
            raise ValueError("MAX_ENTRY_GAP_PCT must be >= -1")
        if self.atr_days < 1:
            raise ValueError("ATR_DAYS must be >= 1")
        if self.initial_stop_mode not in ("fixed_pct", "atr"):
            raise ValueError("INITIAL_STOP_MODE must be fixed_pct or atr")
        if self.atr_stop_multiple <= 0:
            raise ValueError("ATR_STOP_MULTIPLE must be > 0")
        if self.volume_sma_days < 1:
            raise ValueError("VOLUME_SMA_DAYS must be >= 1")
        if not 0 <= self.ibkr_category_breadth_off_threshold < self.ibkr_category_breadth_on_threshold <= 1:
            raise ValueError(
                "IBKR_CATEGORY_BREADTH_OFF_THRESHOLD must be >= 0 and lower than "
                "IBKR_CATEGORY_BREADTH_ON_THRESHOLD, which must be <= 1"
            )
        if self.ibkr_category_breadth_min_symbols < 1:
            raise ValueError("IBKR_CATEGORY_BREADTH_MIN_SYMBOLS must be >= 1")
        if self.initial_equity <= 0:
            raise ValueError("INITIAL_EQUITY must be > 0")
        if self.position_size_usd <= 0:
            raise ValueError("POSITION_SIZE_USD must be > 0")
        if not 0 < self.stop_loss_pct < 1:
            raise ValueError("STOP_LOSS_PCT must be between 0 and 1")
        if self.trailing_activate_pct <= 0:
            raise ValueError("TRAILING_ACTIVATE_PCT must be > 0")
        if not 0 < self.trailing_loss_pct < 1:
            raise ValueError("TRAILING_LOSS_PCT must be between 0 and 1")

    def fingerprint(self) -> str:
        ignored = {"cache_dir", "force_refresh", "log_level"}
        payload = "|".join(
            f"{name}={getattr(self, name)}"
            for name in sorted(self.__dataclass_fields__)
            if name not in ignored
        )
        return sha256(payload.encode("utf-8")).hexdigest()[:16]
