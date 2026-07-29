from __future__ import annotations

import logging
from pathlib import Path
import re

import pytest

from swing_stock_momentum.config import Config
from swing_stock_momentum.contracts import (
    EQUITY_RESULT_COLUMNS,
    RUN_RESULT_COLUMNS,
    SIGNAL_RESULT_COLUMNS,
    TRADE_RESULT_COLUMNS,
)
from swing_stock_momentum.logging_utils import configure_logging


def test_required_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "STARTING_CAPITAL_USD",
        "BACKTEST_START_DATE",
        "MAX_POSITIONS",
        "MAX_NEW_POSITIONS_PER_DAY",
        "RISK_PER_POSITION_PCT",
        "COMMISSION_BPS",
        "SLIPPAGE_BPS",
        "PROGRESS_LOG_INTERVAL_SESSIONS",
        "EARNINGS_BLACKOUT_SESSIONS",
        "SOURCE_EARNINGS_TABLE",
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = Config.from_env()

    assert str(cfg.strategy.starting_capital_usd) == "30000"
    assert cfg.strategy.requested_start_date.isoformat() == "2026-01-01"
    assert cfg.strategy.max_positions == 5
    assert cfg.strategy.max_new_positions_per_day == 2
    assert str(cfg.strategy.risk_per_position_pct) == "1"
    assert str(cfg.strategy.commission_bps) == "0"
    assert str(cfg.strategy.slippage_bps) == "0"
    assert cfg.pg_app_name == "swing_stock_momentum"
    assert cfg.progress_log_interval_sessions == 20
    assert cfg.strategy.earnings_blackout_sessions == 10
    assert cfg.source_earnings_table == "stock_core_earnings_calendar_events"


def test_high_lookback_and_atr_period_are_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRIOR_HIGH_LOOKBACK_SESSIONS", "15")
    monkeypatch.setenv("ATR_PERIOD_SESSIONS", "20")

    cfg = Config.from_env()

    assert cfg.strategy.prior_high_lookback_sessions == 15
    assert cfg.strategy.atr_period_sessions == 20


def test_logging_is_compact_positional_utc(capsys: pytest.CaptureFixture[str]) -> None:
    configure_logging("INFO")
    logging.getLogger("contract-test").info("hello")
    line = capsys.readouterr().err.strip()
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z INFO MainProcess MainThread hello",
        line,
    )
    assert "ts_utc=" not in line


def test_init_sql_owns_schema_drop_switch_and_365_day_chunks() -> None:
    sql_text = (Path(__file__).parents[1] / "init" / "schema.sql").read_text(
        encoding="utf-8"
    )
    assert "drop_all_backtest_momentum_tables_on_start" in sql_text
    assert sql_text.count("INTERVAL '365 days'") == 2
    assert "CREATE TABLE IF NOT EXISTS backtest_momentum_runs" in sql_text
    assert "CREATE TABLE IF NOT EXISTS backtest_momentum_signals" in sql_text
    assert "CREATE TABLE IF NOT EXISTS backtest_momentum_trades" in sql_text
    assert "CREATE TABLE IF NOT EXISTS backtest_momentum_equity_daily" in sql_text
    assert "json" not in sql_text.lower()


def _sql_table_columns(sql_text: str, table: str) -> set[str]:
    body = sql_text.split(f"CREATE TABLE IF NOT EXISTS {table} (", 1)[1].split(
        "\n);", 1
    )[0]
    return {
        match.group(1)
        for line in body.splitlines()
        if (match := re.match(r"^    ([a-z][a-z0-9_]*)\s+", line))
    }


def test_python_result_contracts_match_init_sql_columns() -> None:
    sql_text = (Path(__file__).parents[1] / "init" / "schema.sql").read_text(
        encoding="utf-8"
    )
    expected = {
        "backtest_momentum_runs": set(RUN_RESULT_COLUMNS),
        "backtest_momentum_signals": set(SIGNAL_RESULT_COLUMNS),
        "backtest_momentum_trades": set(TRADE_RESULT_COLUMNS),
        "backtest_momentum_equity_daily": set(EQUITY_RESULT_COLUMNS),
    }
    for table, columns in expected.items():
        assert _sql_table_columns(sql_text, table) == columns


def test_runtime_contains_no_schema_ddl_and_sets_application_name() -> None:
    package = Path(__file__).parents[1] / "swing_stock_momentum"
    python_source = "\n".join(
        path.read_text(encoding="utf-8") for path in package.glob("*.py")
    )
    assert "CREATE TABLE" not in python_source.upper()
    assert "ALTER TABLE" not in python_source.upper()
    assert "DROP TABLE" not in python_source.upper()
    assert "application_name=cfg.pg_app_name" in python_source
