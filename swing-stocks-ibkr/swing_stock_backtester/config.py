import os
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def env_int(name: str, default: int, min_value: int | None = None) -> int:
    value = int(os.getenv(name, str(default)).strip())
    if min_value is not None:
        value = max(min_value, value)
    return value


def env_float(name: str, default: float, min_value: float | None = None) -> float:
    value = float(os.getenv(name, str(default)).strip())
    if min_value is not None:
        value = max(min_value, value)
    return value


def parse_date(raw: str | None, default: date | None = None) -> date | None:
    if raw is None or not raw.strip():
        return default
    return datetime.strptime(raw.strip(), "%Y-%m-%d").date()


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return sorted({part.strip().upper().replace("/", ".") for part in raw.split(",") if part.strip()})


@dataclass(frozen=True)
class BacktestConfig:
    pg_host: str
    pg_port: int
    pg_database: str
    pg_user: str
    pg_password: str
    application_name: str
    connect_timeout_seconds: int
    lock_timeout_ms: int
    statement_timeout_ms: int
    idle_in_transaction_timeout_ms: int

    core_table: str
    market_daily_table: str
    fundamental_daily_table: str
    earnings_table: str
    ibkr_symbols_table: str

    runs_table: str
    results_table: str
    trades_table: str
    equity_table: str

    universe_mode: str
    require_ibkr_symbols: bool
    symbols: tuple[str, ...]
    max_symbols: int
    start_date: date
    end_date: date
    lookback_days: int
    process_parallelism: int
    min_history_days: int
    min_price: float
    min_market_cap: float
    min_average_daily_volume: float
    commission_bps: float
    slippage_bps: float
    write_equity_daily: bool
    log_level: str

    @property
    def load_start_date(self) -> date:
        return self.start_date - timedelta(days=self.lookback_days)

    def connection_kwargs(self) -> dict[str, object]:
        options = " ".join(
            [
                f"-c lock_timeout={self.lock_timeout_ms}",
                f"-c statement_timeout={self.statement_timeout_ms}",
                f"-c idle_in_transaction_session_timeout={self.idle_in_transaction_timeout_ms}",
            ]
        )
        return {
            "host": self.pg_host,
            "port": self.pg_port,
            "dbname": self.pg_database,
            "user": self.pg_user,
            "password": self.pg_password,
            "application_name": self.application_name,
            "connect_timeout": self.connect_timeout_seconds,
            "options": options,
        }


def load_config() -> BacktestConfig:
    today_utc = datetime.now(timezone.utc).date()
    end_date = parse_date(os.getenv("END_DATE"), today_utc) or today_utc
    start_date = parse_date(os.getenv("START_DATE"), date(2020, 1, 1)) or date(2020, 1, 1)
    if end_date < start_date:
        raise ValueError(f"END_DATE {end_date} must not be before START_DATE {start_date}.")

    return BacktestConfig(
        pg_host=os.getenv("PGHOST", "timescaledb").strip(),
        pg_port=env_int("PGPORT", 5432, 1),
        pg_database=os.getenv("PGDATABASE", "postgres").strip(),
        pg_user=os.getenv("PGUSER", "market-data-account").strip(),
        pg_password=os.getenv("PGPASSWORD", "market-data-account-pw"),
        application_name=os.getenv("PGAPPNAME", "swing_stocks_ibkr_backtester").strip(),
        connect_timeout_seconds=env_int("DB_CONNECT_TIMEOUT_SECONDS", 15, 1),
        lock_timeout_ms=env_int("DB_LOCK_TIMEOUT_MS", 60000, 1),
        statement_timeout_ms=env_int("DB_STATEMENT_TIMEOUT_MS", 1800000, 1),
        idle_in_transaction_timeout_ms=env_int("DB_IDLE_IN_TRANSACTION_TIMEOUT_MS", 300000, 1),
        core_table=os.getenv("CORE_TABLE", "public.stock_core_security_master_current").strip(),
        market_daily_table=os.getenv("MARKET_DAILY_TABLE", "public.stock_core_market_metrics_daily").strip(),
        fundamental_daily_table=os.getenv(
            "FUNDAMENTAL_DAILY_TABLE",
            "public.stock_core_sec_fundamentals_asof_daily",
        ).strip(),
        earnings_table=os.getenv("EARNINGS_TABLE", "public.stock_core_earnings_calendar_events").strip(),
        ibkr_symbols_table=os.getenv("IBKR_SYMBOLS_TABLE", "public.ibkr_symbols").strip(),
        runs_table=os.getenv("BACKTEST_RUNS_TABLE", "public.backtest_swing_stock_runs").strip(),
        results_table=os.getenv("BACKTEST_RESULTS_TABLE", "public.backtest_swing_stock_strategy_results").strip(),
        trades_table=os.getenv("BACKTEST_TRADES_TABLE", "public.backtest_swing_stock_trades").strip(),
        equity_table=os.getenv("BACKTEST_EQUITY_TABLE", "public.backtest_swing_stock_equity_daily").strip(),
        universe_mode=os.getenv("UNIVERSE_MODE", "stock_core").strip().lower(),
        require_ibkr_symbols=env_bool("REQUIRE_IBKR_SYMBOLS", False),
        symbols=tuple(parse_symbols(os.getenv("SYMBOLS"))),
        max_symbols=env_int("MAX_SYMBOLS", 0, 0),
        start_date=start_date,
        end_date=end_date,
        lookback_days=env_int("LOOKBACK_DAYS", 430, 0),
        process_parallelism=env_int("PROCESS_PARALLELISM", 4, 1),
        min_history_days=env_int("MIN_HISTORY_DAYS", 260, 1),
        min_price=env_float("MIN_PRICE", 5.0, 0.0),
        min_market_cap=env_float("MIN_MARKET_CAP", 500_000_000.0, 0.0),
        min_average_daily_volume=env_float("MIN_AVERAGE_DAILY_VOLUME", 250_000.0, 0.0),
        commission_bps=env_float("COMMISSION_BPS", 1.0, 0.0),
        slippage_bps=env_float("SLIPPAGE_BPS", 2.0, 0.0),
        write_equity_daily=env_bool("WRITE_EQUITY_DAILY", False),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip(),
    )
