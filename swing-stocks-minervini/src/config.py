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


def _env_lower_str_tuple(name: str, default: str) -> tuple[str, ...]:
    return tuple(
        x.strip().lower() for x in _env(name, default).split(",") if x.strip()
    )


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
    log_level: str
    simulation_mode: str  # independent | portfolio | both
    portfolio_ranking_experiment_enable: bool
    portfolio_ranking_mode: str  # neutral | quality_only | validated
    portfolio_setup_types: tuple[str, ...]
    neutral_rank_salt: str

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

    # IBD-inspired index/volume state with stock breadth confirmation
    market_filter_enable: bool
    market_index_symbols: tuple[str, ...]
    market_primary_index: str
    ftd_min_rally_day: int
    ftd_min_gain: float
    distribution_min_loss: float
    distribution_lookback_sessions: int
    distribution_pressure_count: int
    distribution_correction_count: int
    breadth_on_threshold: float
    breadth_off_threshold: float
    market_confirmed_max_exposure_pct: float
    market_confirmed_weak_breadth_max_exposure_pct: float
    market_under_pressure_max_exposure_pct: float
    market_under_pressure_weak_breadth_max_exposure_pct: float

    # world-regime entry gate
    regime_entry_filter_enable: bool
    regime_allowed_labels: tuple[str, ...]

    # IBKR group leadership filter
    ibkr_industry_rs_min: int
    ibkr_category_rs_min: int
    ibkr_stock_industry_rs_min: int
    ibkr_stock_category_rs_min: int
    ibkr_industry_min_symbols: int
    ibkr_category_min_symbols: int
    ibkr_industry_breadth_on_threshold: float
    ibkr_industry_breadth_off_threshold: float
    ibkr_industry_breadth_min_symbols: int

    # fundamentals
    eps_yoy_min: float
    revenue_yoy_min: float
    fundamentals_min_pass: int
    margin_expansion_min: float
    acceleration_min: float
    quarterly_growth_streak_min: int
    quarterly_fundamental_stale_trading_days: int
    institutional_min_managers: int
    institutional_net_activity_min: float
    institutional_activity_lookback_sessions: int

    # VCP detection
    swing_window: int
    base_min_days: int
    base_max_days: int
    contractions_min: int
    contractions_max: int
    final_depth_max: float
    base_depth_max: float
    pivot_below_base_high_max: float
    dryup_score_zero_ratio: float
    setup_valid_days: int
    prior_advance_min: float

    # simulation and pre-session order sizing
    initial_equity: float
    risk_pct: float
    stop_max_pct: float
    max_position_pct: float
    slippage_pct: float
    commission_pct: float
    partial_at_r: float
    partial_fraction: float
    breakeven_after_partial: bool
    trail_ma_days: int
    failed_breakout_exit_enable: bool
    failed_breakout_days: int
    failed_breakout_min_r: float
    pivot_buffer_pct: float
    max_buy_zone_pct: float
    time_stop_sessions: int
    time_stop_min_r: float

    # portfolio simulation constraints
    portfolio_max_open_positions: int
    portfolio_max_daily_orders: int
    portfolio_max_gross_exposure_pct: float
    min_slate_risk_utilization: float
    exposure_levels: tuple[float, ...]
    exposure_winners_to_step_up: int
    exposure_losses_to_reset: int
    exposure_drawdown_reset_pct: float

    @classmethod
    def from_env(cls) -> "Config":
        cfg = cls(
            stage=_env("STAGE", "all").lower(),
            start_date=_env("START_DATE", "2020-01-02"),
            end_date=_env("END_DATE", "") or None,
            warmup_calendar_days=int(_env("WARMUP_CALENDAR_DAYS", "550")),
            run_label=_env(
                "RUN_LABEL", "minervini_sepa_daily_v8_ranking_experiment"
            ),
            cache_dir=_env("CACHE_DIR", "/cache"),
            force_refresh=_env_bool("FORCE_REFRESH", False),
            log_level=_env("LOG_LEVEL", "INFO").upper(),
            simulation_mode=_env("SIMULATION_MODE", "independent").lower(),
            portfolio_ranking_experiment_enable=_env_bool(
                "PORTFOLIO_RANKING_EXPERIMENT_ENABLE", False
            ),
            portfolio_ranking_mode=_env(
                "PORTFOLIO_RANKING_MODE", "validated"
            ).lower(),
            portfolio_setup_types=_env_lower_str_tuple(
                "PORTFOLIO_SETUP_TYPES", "flat_base,vcp"
            ),
            neutral_rank_salt=_env("NEUTRAL_RANK_SALT", "v8-bootstrap-00"),
            min_price=float(_env("MIN_PRICE", "5.0")),
            min_dollar_volume=float(_env("MIN_DOLLAR_VOLUME", "2000000")),
            rs_lookbacks=_env_int_tuple("RS_LOOKBACKS", "63,126,189,252"),
            rs_weights=_env_float_tuple("RS_WEIGHTS", "2,1,1,1"),
            rs_min=int(_env("RS_MIN", "70")),
            ma200_trend_days=int(_env("MA200_TREND_DAYS", "21")),
            min_above_52w_low=float(_env("MIN_ABOVE_52W_LOW", "1.30")),
            max_below_52w_high=float(_env("MAX_BELOW_52W_HIGH", "0.75")),
            market_filter_enable=_env_bool("MARKET_FILTER_ENABLE", True),
            market_index_symbols=_env_str_tuple("MARKET_INDEX_SYMBOLS", "QQQ,VOO"),
            market_primary_index=_env("MARKET_PRIMARY_INDEX", "QQQ").upper(),
            ftd_min_rally_day=int(_env("FTD_MIN_RALLY_DAY", "4")),
            ftd_min_gain=float(_env("FTD_MIN_GAIN", "0.0125")),
            distribution_min_loss=float(_env("DISTRIBUTION_MIN_LOSS", "0.002")),
            distribution_lookback_sessions=int(
                _env("DISTRIBUTION_LOOKBACK_SESSIONS", "25")
            ),
            distribution_pressure_count=int(_env("DISTRIBUTION_PRESSURE_COUNT", "4")),
            distribution_correction_count=int(_env("DISTRIBUTION_CORRECTION_COUNT", "6")),
            breadth_on_threshold=float(_env("BREADTH_ON_THRESHOLD", "0.50")),
            breadth_off_threshold=float(_env("BREADTH_OFF_THRESHOLD", "0.45")),
            market_confirmed_max_exposure_pct=float(
                _env("MARKET_CONFIRMED_MAX_EXPOSURE_PCT", "1.0")
            ),
            market_confirmed_weak_breadth_max_exposure_pct=float(
                _env("MARKET_CONFIRMED_WEAK_BREADTH_MAX_EXPOSURE_PCT", "0.25")
            ),
            market_under_pressure_max_exposure_pct=float(
                _env("MARKET_UNDER_PRESSURE_MAX_EXPOSURE_PCT", "0.50")
            ),
            market_under_pressure_weak_breadth_max_exposure_pct=float(
                _env("MARKET_UNDER_PRESSURE_WEAK_BREADTH_MAX_EXPOSURE_PCT", "0.25")
            ),
            regime_entry_filter_enable=_env_bool("REGIME_ENTRY_FILTER_ENABLE", False),
            regime_allowed_labels=_env_str_tuple(
                "REGIME_ALLOWED_LABELS", "RISK-ON,CONSTRUCTIVE,NEUTRAL"
            ),
            ibkr_industry_rs_min=int(_env("IBKR_INDUSTRY_RS_MIN", "70")),
            ibkr_category_rs_min=int(_env("IBKR_CATEGORY_RS_MIN", "70")),
            ibkr_stock_industry_rs_min=int(_env("IBKR_STOCK_IN_INDUSTRY_RS_MIN", "70")),
            ibkr_stock_category_rs_min=int(_env("IBKR_STOCK_IN_CATEGORY_RS_MIN", "70")),
            ibkr_industry_min_symbols=int(_env("IBKR_INDUSTRY_MIN_SYMBOLS", "5")),
            ibkr_category_min_symbols=int(_env("IBKR_CATEGORY_MIN_SYMBOLS", "3")),
            ibkr_industry_breadth_on_threshold=float(_env("IBKR_INDUSTRY_BREADTH_ON_THRESHOLD", "0.55")),
            ibkr_industry_breadth_off_threshold=float(_env("IBKR_INDUSTRY_BREADTH_OFF_THRESHOLD", "0.45")),
            ibkr_industry_breadth_min_symbols=int(_env("IBKR_INDUSTRY_BREADTH_MIN_SYMBOLS", "5")),
            eps_yoy_min=float(_env("EPS_YOY_MIN", "0.20")),
            revenue_yoy_min=float(_env("REVENUE_YOY_MIN", "0.10")),
            fundamentals_min_pass=int(_env("FUNDAMENTALS_MIN_PASS", "4")),
            margin_expansion_min=float(_env("MARGIN_EXPANSION_MIN", "0.0")),
            acceleration_min=float(_env("ACCELERATION_MIN", "0.0")),
            quarterly_growth_streak_min=int(_env("QUARTERLY_GROWTH_STREAK_MIN", "2")),
            quarterly_fundamental_stale_trading_days=int(
                _env("QUARTERLY_FUNDAMENTAL_STALE_TRADING_DAYS", "130")
            ),
            institutional_min_managers=int(_env("INSTITUTIONAL_MIN_MANAGERS", "10")),
            institutional_net_activity_min=float(_env("INSTITUTIONAL_NET_ACTIVITY_MIN", "0")),
            institutional_activity_lookback_sessions=int(
                _env("INSTITUTIONAL_ACTIVITY_LOOKBACK_SESSIONS", "130")
            ),
            swing_window=int(_env("SWING_WINDOW", "3")),
            base_min_days=int(_env("BASE_MIN_DAYS", "15")),
            base_max_days=int(_env("BASE_MAX_DAYS", "130")),
            contractions_min=int(_env("CONTRACTIONS_MIN", "2")),
            contractions_max=int(_env("CONTRACTIONS_MAX", "4")),
            final_depth_max=float(_env("FINAL_DEPTH_MAX", "0.10")),
            base_depth_max=float(_env("BASE_DEPTH_MAX", "0.35")),
            pivot_below_base_high_max=float(_env("PIVOT_BELOW_BASE_HIGH_MAX", "0.08")),
            dryup_score_zero_ratio=float(_env("DRYUP_SCORE_ZERO_RATIO", "1.25")),
            setup_valid_days=int(_env("SETUP_VALID_DAYS", "20")),
            prior_advance_min=float(_env("PRIOR_ADVANCE_MIN", "0.25")),
            initial_equity=float(_env("INITIAL_EQUITY", "100000")),
            risk_pct=float(_env("RISK_PCT", "0.01")),
            stop_max_pct=float(_env("STOP_MAX_PCT", "0.08")),
            max_position_pct=float(_env("MAX_POSITION_PCT", "0.25")),
            slippage_pct=float(_env("SLIPPAGE_PCT", "0.001")),
            commission_pct=float(_env("COMMISSION_PCT", "0.0005")),
            partial_at_r=float(_env("PARTIAL_AT_R", "2.0")),
            partial_fraction=float(_env("PARTIAL_FRACTION", "0.0")),
            breakeven_after_partial=_env_bool("BREAKEVEN_AFTER_PARTIAL", False),
            trail_ma_days=int(_env("TRAIL_MA_DAYS", "50")),
            failed_breakout_exit_enable=_env_bool("FAILED_BREAKOUT_EXIT_ENABLE", True),
            failed_breakout_days=int(_env("FAILED_BREAKOUT_DAYS", "10")),
            failed_breakout_min_r=float(_env("FAILED_BREAKOUT_MIN_R", "-0.5")),
            pivot_buffer_pct=float(_env("PIVOT_BUFFER_PCT", "0.001")),
            max_buy_zone_pct=float(_env("MAX_BUY_ZONE_PCT", "0.02")),
            time_stop_sessions=int(_env("TIME_STOP_SESSIONS", "10")),
            time_stop_min_r=float(_env("TIME_STOP_MIN_R", "1.0")),
            portfolio_max_open_positions=int(_env("PORTFOLIO_MAX_OPEN_POSITIONS", "8")),
            portfolio_max_daily_orders=int(_env("PORTFOLIO_MAX_DAILY_ORDERS", "3")),
            portfolio_max_gross_exposure_pct=float(_env("PORTFOLIO_MAX_GROSS_EXPOSURE_PCT", "1.0")),
            min_slate_risk_utilization=float(_env("MIN_SLATE_RISK_UTILIZATION", "0.50")),
            exposure_levels=_env_float_tuple("EXPOSURE_LEVELS", "0.25,0.50,0.75,1.00"),
            exposure_winners_to_step_up=int(_env("EXPOSURE_WINNERS_TO_STEP_UP", "2")),
            exposure_losses_to_reset=int(_env("EXPOSURE_LOSSES_TO_RESET", "2")),
            exposure_drawdown_reset_pct=float(_env("EXPOSURE_DRAWDOWN_RESET_PCT", "0.04")),
        )
        if cfg.stage not in ("screen", "setup", "sim", "all", "sensitivity"):
            raise ValueError(f"unsupported STAGE={cfg.stage!r}")
        if cfg.simulation_mode not in ("independent", "portfolio", "both"):
            raise ValueError(f"unsupported SIMULATION_MODE={cfg.simulation_mode!r}")
        if (
            cfg.portfolio_ranking_experiment_enable
            and cfg.simulation_mode == "independent"
        ):
            raise ValueError(
                "PORTFOLIO_RANKING_EXPERIMENT_ENABLE requires "
                "SIMULATION_MODE portfolio or both"
            )
        if cfg.portfolio_ranking_mode not in (
            "neutral",
            "quality_only",
            "validated",
        ):
            raise ValueError(
                "PORTFOLIO_RANKING_MODE must be neutral, quality_only or validated"
            )
        allowed_portfolio_setup_types = {"flat_base", "vcp"}
        if not cfg.portfolio_setup_types:
            raise ValueError("PORTFOLIO_SETUP_TYPES must not be empty")
        unsupported_setup_types = sorted(
            set(cfg.portfolio_setup_types) - allowed_portfolio_setup_types
        )
        if unsupported_setup_types:
            raise ValueError(
                "PORTFOLIO_SETUP_TYPES supports only flat_base and vcp; got "
                + ",".join(unsupported_setup_types)
            )
        if len(set(cfg.portfolio_setup_types)) != len(cfg.portfolio_setup_types):
            raise ValueError("PORTFOLIO_SETUP_TYPES must not contain duplicates")
        if not isinstance(cfg.neutral_rank_salt, str) or not cfg.neutral_rank_salt:
            raise ValueError("NEUTRAL_RANK_SALT must be a non-empty string")
        if not cfg.market_index_symbols:
            raise ValueError("MARKET_INDEX_SYMBOLS must contain at least one symbol")
        if cfg.market_primary_index not in cfg.market_index_symbols:
            raise ValueError("MARKET_PRIMARY_INDEX must be included in MARKET_INDEX_SYMBOLS")
        if cfg.ftd_min_rally_day < 2:
            raise ValueError("FTD_MIN_RALLY_DAY must be >= 2")
        if cfg.ftd_min_gain <= 0:
            raise ValueError("FTD_MIN_GAIN must be > 0")
        if cfg.distribution_min_loss <= 0:
            raise ValueError("DISTRIBUTION_MIN_LOSS must be > 0")
        if cfg.distribution_lookback_sessions < 1:
            raise ValueError("DISTRIBUTION_LOOKBACK_SESSIONS must be >= 1")
        if not (
            1 <= cfg.distribution_pressure_count
            < cfg.distribution_correction_count
        ):
            raise ValueError(
                "DISTRIBUTION counts must satisfy 1 <= pressure < correction"
            )
        if not (0 <= cfg.breadth_off_threshold < cfg.breadth_on_threshold <= 1):
            raise ValueError("breadth thresholds must satisfy 0 <= off < on <= 1")
        market_exposure_caps = (
            cfg.market_confirmed_max_exposure_pct,
            cfg.market_confirmed_weak_breadth_max_exposure_pct,
            cfg.market_under_pressure_max_exposure_pct,
            cfg.market_under_pressure_weak_breadth_max_exposure_pct,
        )
        if any(value < 0 or value > 1 for value in market_exposure_caps):
            raise ValueError("market exposure caps must be between 0 and 1")
        if not (
            cfg.market_confirmed_weak_breadth_max_exposure_pct
            <= cfg.market_confirmed_max_exposure_pct
        ):
            raise ValueError("confirmed weak-breadth cap must not exceed confirmed cap")
        if not (
            cfg.market_under_pressure_weak_breadth_max_exposure_pct
            <= cfg.market_under_pressure_max_exposure_pct
            <= cfg.market_confirmed_max_exposure_pct
        ):
            raise ValueError("under-pressure caps must not exceed confirmed cap")
        if cfg.swing_window < 1:
            raise ValueError("SWING_WINDOW must be >= 1")
        if not 10 <= cfg.base_min_days < cfg.base_max_days:
            raise ValueError("BASE days must satisfy 10 <= MIN < MAX")
        if not 2 <= cfg.contractions_min <= cfg.contractions_max:
            raise ValueError("CONTRACTIONS must satisfy 2 <= MIN <= MAX")
        if not 0 < cfg.final_depth_max <= cfg.base_depth_max < 1:
            raise ValueError("depths must satisfy 0 < FINAL <= BASE < 1")
        if not 0 < cfg.pivot_below_base_high_max < 1:
            raise ValueError("PIVOT_BELOW_BASE_HIGH_MAX must be between 0 and 1")
        if cfg.setup_valid_days < 1:
            raise ValueError("SETUP_VALID_DAYS must be >= 1")
        if cfg.dryup_score_zero_ratio <= 0:
            raise ValueError("DRYUP_SCORE_ZERO_RATIO must be > 0")
        if cfg.prior_advance_min <= 0:
            raise ValueError("PRIOR_ADVANCE_MIN must be > 0")
        if cfg.failed_breakout_days < 1:
            raise ValueError("FAILED_BREAKOUT_DAYS must be >= 1")
        if cfg.portfolio_max_open_positions < 1:
            raise ValueError("PORTFOLIO_MAX_OPEN_POSITIONS must be >= 1")
        if not 1 <= cfg.portfolio_max_daily_orders <= 3:
            raise ValueError("PORTFOLIO_MAX_DAILY_ORDERS must be between 1 and 3")
        if cfg.portfolio_max_gross_exposure_pct <= 0:
            raise ValueError("PORTFOLIO_MAX_GROSS_EXPOSURE_PCT must be > 0")
        if not 0 < cfg.min_slate_risk_utilization <= 1:
            raise ValueError("MIN_SLATE_RISK_UTILIZATION must be in (0, 1]")
        if cfg.exposure_winners_to_step_up < 1:
            raise ValueError("EXPOSURE_WINNERS_TO_STEP_UP must be >= 1")
        if cfg.exposure_losses_to_reset < 1:
            raise ValueError("EXPOSURE_LOSSES_TO_RESET must be >= 1")
        if not cfg.exposure_levels or any(level <= 0 or level > 1 for level in cfg.exposure_levels):
            raise ValueError("EXPOSURE_LEVELS must contain values in (0, 1]")
        if tuple(sorted(cfg.exposure_levels)) != cfg.exposure_levels:
            raise ValueError("EXPOSURE_LEVELS must be sorted ascending")
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)
