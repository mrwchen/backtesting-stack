"""Environment-driven configuration for the Minervini swing backtester."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "on")


def _env_int_tuple(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(x) for x in _env(name, default).split(","))


def _env_float_tuple(name: str, default: str) -> tuple[float, ...]:
    return tuple(float(x) for x in _env(name, default).split(","))


def _env_str_tuple(name: str, default: str) -> tuple[str, ...]:
    return tuple(x.strip().upper() for x in _env(name, default).split(",") if x.strip())


@dataclass(frozen=True)
class Config:
    # run control
    stage: str
    start_date: str
    end_date: str | None
    warmup_calendar_days: int
    run_label: str
    cache_dir: str
    force_refresh: bool
    screen_persist: str  # passed | universe
    log_level: str
    simulation_mode: str  # independent | portfolio

    # universe filters
    min_price: float
    min_dollar_volume: float

    # relative strength
    rs_lookbacks: tuple[int, ...]
    rs_weights: tuple[float, ...]
    rs_min: int

    # trend template
    ma200_trend_days: int
    min_above_52w_low: float
    max_below_52w_high: float

    # market regime filter (breadth of stocks above their 200d MA)
    market_filter_enable: bool
    breadth_on_threshold: float
    breadth_off_threshold: float

    # world-regime entry gate
    regime_entry_filter_enable: bool
    regime_allowed_labels: tuple[str, ...]

    # IBKR group leadership filter
    ibkr_group_filter_enable: bool
    ibkr_industry_rs_min: int
    ibkr_category_rs_min: int
    ibkr_stock_industry_rs_min: int
    ibkr_stock_category_rs_min: int
    ibkr_industry_min_symbols: int
    ibkr_category_min_symbols: int
    ibkr_industry_breadth_filter_enable: bool
    ibkr_industry_breadth_on_threshold: float
    ibkr_industry_breadth_off_threshold: float
    ibkr_industry_breadth_min_symbols: int

    # fundamentals
    eps_yoy_min: float
    revenue_yoy_min: float
    fundamentals_min_pass: int
    eps_stale_trading_days: int
    filing_stale_trading_days: int

    # VCP detection
    swing_window: int
    base_min_days: int
    base_max_days: int
    contractions_min: int
    contractions_max: int
    final_depth_max: float
    base_depth_max: float
    pivot_below_base_high_max: float
    dryup_ratio_max: float
    setup_valid_days: int

    # simulation (per-trade, no portfolio constraints)
    initial_equity: float
    risk_pct: float
    stop_max_pct: float
    max_position_pct: float
    max_gap_pct: float
    slippage_pct: float
    commission_pct: float
    partial_at_r: float
    partial_fraction: float
    breakeven_after_partial: bool
    trail_ma_days: int

    # portfolio simulation constraints
    portfolio_max_open_positions: int
    portfolio_max_gross_exposure_pct: float

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            stage=_env("STAGE", "all").lower(),
            start_date=_env("START_DATE", "2020-01-02"),
            end_date=_env("END_DATE", "") or None,
            warmup_calendar_days=int(_env("WARMUP_CALENDAR_DAYS", "550")),
            run_label=_env("RUN_LABEL", "minervini_v1"),
            cache_dir=_env("CACHE_DIR", "/cache"),
            force_refresh=_env_bool("FORCE_REFRESH", False),
            screen_persist=_env("SCREEN_PERSIST", "passed").lower(),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            simulation_mode=_env("SIMULATION_MODE", "independent").lower(),
            min_price=float(_env("MIN_PRICE", "5.0")),
            min_dollar_volume=float(_env("MIN_DOLLAR_VOLUME", "2000000")),
            rs_lookbacks=_env_int_tuple("RS_LOOKBACKS", "63,126,189,252"),
            rs_weights=_env_float_tuple("RS_WEIGHTS", "2,1,1,1"),
            rs_min=int(_env("RS_MIN", "70")),
            ma200_trend_days=int(_env("MA200_TREND_DAYS", "21")),
            min_above_52w_low=float(_env("MIN_ABOVE_52W_LOW", "1.30")),
            max_below_52w_high=float(_env("MAX_BELOW_52W_HIGH", "0.75")),
            market_filter_enable=_env_bool("MARKET_FILTER_ENABLE", True),
            breadth_on_threshold=float(_env("BREADTH_ON_THRESHOLD", "0.50")),
            breadth_off_threshold=float(_env("BREADTH_OFF_THRESHOLD", "0.45")),
            regime_entry_filter_enable=_env_bool("REGIME_ENTRY_FILTER_ENABLE", False),
            regime_allowed_labels=_env_str_tuple("REGIME_ALLOWED_LABELS", "CONSTRUCTIVE,NEUTRAL"),
            ibkr_group_filter_enable=_env_bool("IBKR_GROUP_FILTER_ENABLE", True),
            ibkr_industry_rs_min=int(_env("IBKR_INDUSTRY_RS_MIN", "70")),
            ibkr_category_rs_min=int(_env("IBKR_CATEGORY_RS_MIN", "70")),
            ibkr_stock_industry_rs_min=int(_env("IBKR_STOCK_IN_INDUSTRY_RS_MIN", "70")),
            ibkr_stock_category_rs_min=int(_env("IBKR_STOCK_IN_CATEGORY_RS_MIN", "70")),
            ibkr_industry_min_symbols=int(_env("IBKR_INDUSTRY_MIN_SYMBOLS", "5")),
            ibkr_category_min_symbols=int(_env("IBKR_CATEGORY_MIN_SYMBOLS", "3")),
            ibkr_industry_breadth_filter_enable=_env_bool("IBKR_INDUSTRY_BREADTH_FILTER_ENABLE", True),
            ibkr_industry_breadth_on_threshold=float(_env("IBKR_INDUSTRY_BREADTH_ON_THRESHOLD", "0.55")),
            ibkr_industry_breadth_off_threshold=float(_env("IBKR_INDUSTRY_BREADTH_OFF_THRESHOLD", "0.45")),
            ibkr_industry_breadth_min_symbols=int(_env("IBKR_INDUSTRY_BREADTH_MIN_SYMBOLS", "5")),
            eps_yoy_min=float(_env("EPS_YOY_MIN", "0.20")),
            revenue_yoy_min=float(_env("REVENUE_YOY_MIN", "0.10")),
            fundamentals_min_pass=int(_env("FUNDAMENTALS_MIN_PASS", "2")),
            eps_stale_trading_days=int(_env("EPS_STALE_TRADING_DAYS", "130")),
            filing_stale_trading_days=int(_env("FILING_STALE_TRADING_DAYS", "280")),
            swing_window=int(_env("SWING_WINDOW", "3")),
            base_min_days=int(_env("BASE_MIN_DAYS", "15")),
            base_max_days=int(_env("BASE_MAX_DAYS", "75")),
            contractions_min=int(_env("CONTRACTIONS_MIN", "2")),
            contractions_max=int(_env("CONTRACTIONS_MAX", "4")),
            final_depth_max=float(_env("FINAL_DEPTH_MAX", "0.10")),
            base_depth_max=float(_env("BASE_DEPTH_MAX", "0.35")),
            pivot_below_base_high_max=float(_env("PIVOT_BELOW_BASE_HIGH_MAX", "0.05")),
            dryup_ratio_max=float(_env("DRYUP_RATIO_MAX", "0.70")),
            setup_valid_days=int(_env("SETUP_VALID_DAYS", "15")),
            initial_equity=float(_env("INITIAL_EQUITY", "100000")),
            risk_pct=float(_env("RISK_PCT", "0.01")),
            stop_max_pct=float(_env("STOP_MAX_PCT", "0.08")),
            max_position_pct=float(_env("MAX_POSITION_PCT", "0.25")),
            max_gap_pct=float(_env("MAX_GAP_PCT", "0.05")),
            slippage_pct=float(_env("SLIPPAGE_PCT", "0.001")),
            commission_pct=float(_env("COMMISSION_PCT", "0.0005")),
            partial_at_r=float(_env("PARTIAL_AT_R", "2.0")),
            partial_fraction=float(_env("PARTIAL_FRACTION", "0.5")),
            breakeven_after_partial=_env_bool("BREAKEVEN_AFTER_PARTIAL", True),
            trail_ma_days=int(_env("TRAIL_MA_DAYS", "50")),
            portfolio_max_open_positions=int(_env("PORTFOLIO_MAX_OPEN_POSITIONS", "8")),
            portfolio_max_gross_exposure_pct=float(_env("PORTFOLIO_MAX_GROSS_EXPOSURE_PCT", "1.0")),
        )
        if cfg.stage not in ("screen", "setup", "sim", "all"):
            raise ValueError(f"unsupported STAGE={cfg.stage!r}")
        if cfg.screen_persist not in ("passed", "universe"):
            raise ValueError(f"unsupported SCREEN_PERSIST={cfg.screen_persist!r}")
        if cfg.simulation_mode not in ("independent", "portfolio"):
            raise ValueError(f"unsupported SIMULATION_MODE={cfg.simulation_mode!r}")
        if cfg.portfolio_max_open_positions < 1:
            raise ValueError("PORTFOLIO_MAX_OPEN_POSITIONS must be >= 1")
        if cfg.portfolio_max_gross_exposure_pct <= 0:
            raise ValueError("PORTFOLIO_MAX_GROSS_EXPOSURE_PCT must be > 0")
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)
