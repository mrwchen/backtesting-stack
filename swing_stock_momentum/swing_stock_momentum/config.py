from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import os
import re
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


_QUALIFIED_IDENTIFIER = re.compile(
    r"^(?:[A-Za-z_][A-Za-z0-9_]*\.)?[A-Za-z_][A-Za-z0-9_]*$"
)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_decimal(name: str, default: str) -> Decimal:
    return Decimal(os.getenv(name, default))


def _env_date(name: str, default: str) -> date:
    return date.fromisoformat(os.getenv(name, default))


def _env_int_tuple(name: str, default: str) -> tuple[int, ...]:
    return tuple(int(value.strip()) for value in os.getenv(name, default).split(","))


def _env_decimal_tuple(name: str, default: str) -> tuple[Decimal, ...]:
    return tuple(
        Decimal(value.strip()) for value in os.getenv(name, default).split(",")
    )


def _table_name(name: str, default: str) -> str:
    value = os.getenv(name, default)
    if not _QUALIFIED_IDENTIFIER.fullmatch(value):
        raise ValueError(f"{name} must be an unquoted PostgreSQL table name")
    return value


@dataclass(frozen=True)
class StrategyParameters:
    account_type: str
    currency: str
    trading_timezone: str
    requested_start_date: date
    starting_capital_usd: Decimal
    max_positions: int
    max_new_positions_per_day: int
    risk_per_position_pct: Decimal
    initial_stop_loss_pct: Decimal
    stop_step_interval_sessions: int
    stop_step_pct: Decimal
    initial_take_profit_pct: Decimal
    take_profit_step_interval_sessions: int
    take_profit_step_pct: Decimal
    atr_period_sessions: int
    atr_day1_exit_max_pct: Decimal
    atr_day2_exit_max_pct: Decimal
    prior_high_lookback_sessions: int
    prior_high_max_above_signal_close_pct: Decimal
    min_daily_price_change_pct: Decimal
    max_daily_price_change_pct_exclusive: Decimal
    min_volume_vs_sma21_ratio_exclusive: Decimal
    earnings_blackout_sessions: int
    commission_bps: Decimal
    slippage_bps: Decimal

    def validate(self) -> None:
        if self.account_type != "unlevered":
            raise ValueError("ACCOUNT_TYPE must be unlevered")
        if self.currency != "USD":
            raise ValueError("ACCOUNT_CURRENCY must be USD")
        try:
            ZoneInfo(self.trading_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"TRADING_TIMEZONE is unknown: {self.trading_timezone}"
            ) from exc
        if self.trading_timezone != "America/New_York":
            raise ValueError("TRADING_TIMEZONE must be America/New_York")
        if self.starting_capital_usd <= 0:
            raise ValueError("STARTING_CAPITAL_USD must be positive")
        if self.max_positions < 1:
            raise ValueError("MAX_POSITIONS must be >= 1")
        if self.max_new_positions_per_day < 1:
            raise ValueError("MAX_NEW_POSITIONS_PER_DAY must be >= 1")
        if self.max_new_positions_per_day > self.max_positions:
            raise ValueError(
                "MAX_NEW_POSITIONS_PER_DAY must not exceed MAX_POSITIONS"
            )
        if not Decimal("0") < self.risk_per_position_pct <= Decimal("100"):
            raise ValueError("RISK_PER_POSITION_PCT must be in (0, 100]")
        if self.initial_stop_loss_pct >= 0:
            raise ValueError("INITIAL_STOP_LOSS_PCT must be negative")
        if self.initial_take_profit_pct <= 0:
            raise ValueError("INITIAL_TAKE_PROFIT_PCT must be positive")
        if self.stop_step_interval_sessions < 1:
            raise ValueError("STOP_STEP_INTERVAL_SESSIONS must be >= 1")
        if self.take_profit_step_interval_sessions < 1:
            raise ValueError("TAKE_PROFIT_STEP_INTERVAL_SESSIONS must be >= 1")
        if self.stop_step_pct <= 0 or self.take_profit_step_pct <= 0:
            raise ValueError("stop and take-profit step percentages must be positive")
        if self.atr_period_sessions < 1:
            raise ValueError("ATR_PERIOD_SESSIONS must be >= 1")
        if self.atr_day1_exit_max_pct < 0 or self.atr_day2_exit_max_pct < 0:
            raise ValueError("ATR exit thresholds must be non-negative")
        if self.prior_high_lookback_sessions < 1:
            raise ValueError("PRIOR_HIGH_LOOKBACK_SESSIONS must be >= 1")
        if self.prior_high_max_above_signal_close_pct < 0:
            raise ValueError("PRIOR_HIGH_MAX_ABOVE_SIGNAL_CLOSE_PCT must be >= 0")
        if self.min_daily_price_change_pct >= self.max_daily_price_change_pct_exclusive:
            raise ValueError("daily price-change bounds are invalid")
        if self.min_volume_vs_sma21_ratio_exclusive < 0:
            raise ValueError("MIN_VOLUME_VS_SMA21_RATIO_EXCLUSIVE must be >= 0")
        if self.earnings_blackout_sessions < 1:
            raise ValueError("EARNINGS_BLACKOUT_SESSIONS must be >= 1")
        if self.commission_bps < 0 or self.slippage_bps < 0:
            raise ValueError("cost assumptions must be non-negative")


@dataclass(frozen=True)
class AnalyserParameters:
    min_price_usd: Decimal
    min_dollar_volume_usd: Decimal
    rs_lookbacks: tuple[int, int, int, int]
    rs_weights: tuple[Decimal, Decimal, Decimal, Decimal]
    rs_min: int
    ma200_trend_sessions: int
    min_above_52w_low_ratio: Decimal
    min_near_52w_high_ratio: Decimal

    def validate(self) -> None:
        if self.min_price_usd <= 0 or self.min_dollar_volume_usd <= 0:
            raise ValueError("analyser liquidity thresholds must be positive")
        if len(self.rs_lookbacks) != 4 or len(self.rs_weights) != 4:
            raise ValueError("analyser RS lookbacks and weights must contain four values")
        if tuple(sorted(self.rs_lookbacks)) != self.rs_lookbacks:
            raise ValueError("analyser RS lookbacks must be increasing")
        if any(value <= 0 for value in self.rs_weights):
            raise ValueError("analyser RS weights must be positive")
        if not 1 <= self.rs_min <= 99:
            raise ValueError("ANALYSER_RS_MIN must be between 1 and 99")
        if self.ma200_trend_sessions < 1:
            raise ValueError("ANALYSER_MA200_TREND_SESSIONS must be >= 1")


@dataclass(frozen=True)
class Config:
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str
    pg_app_name: str
    db_connect_timeout_seconds: int
    db_statement_timeout_ms: int
    db_fetch_batch_size: int
    db_write_page_size: int
    progress_log_interval_sessions: int
    log_level: str
    source_market_table: str
    source_analyser_table: str
    source_analyser_state_table: str
    source_earnings_table: str
    runs_table: str
    signals_table: str
    trades_table: str
    equity_table: str
    strategy: StrategyParameters
    analyser: AnalyserParameters

    @classmethod
    def from_env(cls) -> "Config":
        rs_lookbacks = _env_int_tuple("ANALYSER_RS_LOOKBACKS", "63,126,189,252")
        rs_weights = _env_decimal_tuple("ANALYSER_RS_WEIGHTS", "2,1,1,1")
        if len(rs_lookbacks) != 4 or len(rs_weights) != 4:
            raise ValueError(
                "ANALYSER_RS_LOOKBACKS and ANALYSER_RS_WEIGHTS must contain four values"
            )
        cfg = cls(
            pg_host=os.getenv("PGHOST", "timescaledb"),
            pg_port=_env_int("PGPORT", 5432),
            pg_database=os.getenv("PGDATABASE", "postgres"),
            pg_user=os.getenv("PGUSER", "backtesting-account"),
            pg_password=os.getenv("PGPASSWORD", "backtesting-account-pw"),
            pg_app_name=os.getenv("PGAPPNAME", "swing_stock_momentum"),
            db_connect_timeout_seconds=_env_int("DB_CONNECT_TIMEOUT_SECONDS", 15),
            db_statement_timeout_ms=_env_int("DB_STATEMENT_TIMEOUT_MS", 0),
            db_fetch_batch_size=_env_int("DB_FETCH_BATCH_SIZE", 10_000),
            db_write_page_size=_env_int("DB_WRITE_PAGE_SIZE", 1_000),
            progress_log_interval_sessions=_env_int(
                "PROGRESS_LOG_INTERVAL_SESSIONS", 20
            ),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            source_market_table=_table_name(
                "SOURCE_MARKET_TABLE", "stock_core_market_metrics_daily"
            ),
            source_analyser_table=_table_name(
                "SOURCE_ANALYSER_TABLE", "stock_analyser_trend_template_daily"
            ),
            source_analyser_state_table=_table_name(
                "SOURCE_ANALYSER_STATE_TABLE", "stock_analyser_incremental_state"
            ),
            source_earnings_table=_table_name(
                "SOURCE_EARNINGS_TABLE", "stock_core_earnings_calendar_events"
            ),
            runs_table=_table_name("RUNS_TABLE", "backtest_momentum_runs"),
            signals_table=_table_name("SIGNALS_TABLE", "backtest_momentum_signals"),
            trades_table=_table_name("TRADES_TABLE", "backtest_momentum_trades"),
            equity_table=_table_name(
                "EQUITY_TABLE", "backtest_momentum_equity_daily"
            ),
            strategy=StrategyParameters(
                account_type=os.getenv("ACCOUNT_TYPE", "unlevered"),
                currency=os.getenv("ACCOUNT_CURRENCY", "USD"),
                trading_timezone=os.getenv("TRADING_TIMEZONE", "America/New_York"),
                requested_start_date=_env_date("BACKTEST_START_DATE", "2026-01-01"),
                starting_capital_usd=_env_decimal("STARTING_CAPITAL_USD", "30000"),
                max_positions=_env_int("MAX_POSITIONS", 5),
                max_new_positions_per_day=_env_int(
                    "MAX_NEW_POSITIONS_PER_DAY", 2
                ),
                risk_per_position_pct=_env_decimal("RISK_PER_POSITION_PCT", "1"),
                initial_stop_loss_pct=_env_decimal("INITIAL_STOP_LOSS_PCT", "-5"),
                stop_step_interval_sessions=_env_int(
                    "STOP_STEP_INTERVAL_SESSIONS", 5
                ),
                stop_step_pct=_env_decimal("STOP_STEP_PCT", "5"),
                initial_take_profit_pct=_env_decimal(
                    "INITIAL_TAKE_PROFIT_PCT", "10"
                ),
                take_profit_step_interval_sessions=_env_int(
                    "TAKE_PROFIT_STEP_INTERVAL_SESSIONS", 5
                ),
                take_profit_step_pct=_env_decimal("TAKE_PROFIT_STEP_PCT", "7.5"),
                atr_period_sessions=_env_int("ATR_PERIOD_SESSIONS", 14),
                atr_day1_exit_max_pct=_env_decimal("ATR_DAY1_EXIT_MAX_PCT", "1.5"),
                atr_day2_exit_max_pct=_env_decimal("ATR_DAY2_EXIT_MAX_PCT", "2"),
                prior_high_lookback_sessions=_env_int(
                    "PRIOR_HIGH_LOOKBACK_SESSIONS", 10
                ),
                prior_high_max_above_signal_close_pct=_env_decimal(
                    "PRIOR_HIGH_MAX_ABOVE_SIGNAL_CLOSE_PCT", "10"
                ),
                min_daily_price_change_pct=_env_decimal(
                    "MIN_DAILY_PRICE_CHANGE_PCT", "1"
                ),
                max_daily_price_change_pct_exclusive=_env_decimal(
                    "MAX_DAILY_PRICE_CHANGE_PCT_EXCLUSIVE", "5"
                ),
                min_volume_vs_sma21_ratio_exclusive=_env_decimal(
                    "MIN_VOLUME_VS_SMA21_RATIO_EXCLUSIVE", "1.2"
                ),
                earnings_blackout_sessions=_env_int(
                    "EARNINGS_BLACKOUT_SESSIONS", 10
                ),
                commission_bps=_env_decimal("COMMISSION_BPS", "0"),
                slippage_bps=_env_decimal("SLIPPAGE_BPS", "0"),
            ),
            analyser=AnalyserParameters(
                min_price_usd=_env_decimal("ANALYSER_MIN_PRICE_USD", "5"),
                min_dollar_volume_usd=_env_decimal(
                    "ANALYSER_MIN_DOLLAR_VOLUME_USD", "2000000"
                ),
                rs_lookbacks=tuple(rs_lookbacks),  # type: ignore[arg-type]
                rs_weights=tuple(rs_weights),  # type: ignore[arg-type]
                rs_min=_env_int("ANALYSER_RS_MIN", 70),
                ma200_trend_sessions=_env_int(
                    "ANALYSER_MA200_TREND_SESSIONS", 21
                ),
                min_above_52w_low_ratio=_env_decimal(
                    "ANALYSER_MIN_ABOVE_52W_LOW_RATIO", "1.30"
                ),
                min_near_52w_high_ratio=_env_decimal(
                    "ANALYSER_MIN_NEAR_52W_HIGH_RATIO", "0.75"
                ),
            ),
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if not self.pg_app_name.strip():
            raise ValueError("PGAPPNAME must not be empty")
        if self.pg_port < 1 or self.pg_port > 65_535:
            raise ValueError("PGPORT must be between 1 and 65535")
        if self.db_connect_timeout_seconds < 1:
            raise ValueError("DB_CONNECT_TIMEOUT_SECONDS must be >= 1")
        if self.db_statement_timeout_ms < 0:
            raise ValueError("DB_STATEMENT_TIMEOUT_MS must be >= 0")
        if self.db_fetch_batch_size < 1 or self.db_write_page_size < 1:
            raise ValueError("database batch sizes must be >= 1")
        if self.progress_log_interval_sessions < 1:
            raise ValueError("PROGRESS_LOG_INTERVAL_SESSIONS must be >= 1")
        targets = (self.runs_table, self.signals_table, self.trades_table, self.equity_table)
        if len(set(targets)) != len(targets):
            raise ValueError("backtest target tables must be distinct")
        for table in targets:
            if not table.split(".")[-1].startswith("backtest_momentum_"):
                raise ValueError(
                    "backtest target table names must start with backtest_momentum_"
                )
        self.strategy.validate()
        self.analyser.validate()
