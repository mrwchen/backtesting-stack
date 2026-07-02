# Swing Stocks IBKR Backtester

Backtests three long-only daily swing setups for every eligible stock in the configured universe:

- `quality_momentum_swing`
- `trend_pullback`
- `earnings_reaction_drift`

The service reads only the existing stock-core tables and writes results only to new `backtest_...` tables.

## Source Tables

- `stock_core_security_master_current`
- `stock_core_market_metrics_daily`
- `stock_core_sec_fundamentals_asof_daily`
- `stock_core_earnings_calendar_events`

## Result Tables

- `backtest_swing_stock_runs`
- `backtest_swing_stock_strategy_results`
- `backtest_swing_stock_trades`
- `backtest_swing_stock_equity_daily`

`WRITE_EQUITY_DAILY=false` by default to avoid very large daily result writes.

## Execution Model

Each symbol is processed by one worker process. Workers only read market/fundamental/event data. The main process writes all backtest results sequentially, avoiding concurrent writes to the result tables.

Signals are generated after a daily close. Entries are executed at the next available daily open. Stops can trigger intraday using daily low/high information after the entry is active.

## Run

Run on the remote Docker host from this directory:

```bash
docker compose up --build
```

The init container creates or validates the `backtest_...` tables. `DROP_ALL_TABLES_ON_START` defaults to `false` and only affects this service's `backtest_swing_stock_...` tables.

## Diagnostics

Run diagnostics after a backtest:

```bash
docker compose run --rm swing-stocks-ibkr-diagnostics
```

The diagnostics use the latest run by default. Set `DIAGNOSTICS_RUN_ID` to inspect a specific run. Results are written to `backtest_swing_stock_diagnostic_...` tables.

Diagnostics include strategy edge, symbol breadth, yearly stability, exit reasons, top/bottom symbols, feature bucket strength, and feature bucket stability. Feature buckets are based on signal-day data, not post-trade data.
