# swing-stocks-wei

Standalone swing backtester for the requested 52-week-high pullback strategy.

## Data Sources

Only these existing market-data tables are read:

| Table | Used for |
|---|---|
| `stock_core_market_metrics_daily` | adjusted daily OHLCV, Alpaca price feed and USD market cap |
| `stock_core_security_master_current` | USD equity universe |
| `ibkr_symbols` | IBKR-backed USD stock eligibility and industry/category metadata |
| `stock_core_sec_fundamentals_asof_daily` | point-in-time SEC TTM revenue |

No regime data, no external APIs and no scraping are used.

## Strategy

For every IBKR-backed USD equity:

1. A 52-week high is detected with a 252-trading-day rolling high.
2. A setup is active if at least one 52-week high occurred in the last 10 trading days.
3. Entry signal is either an EMA9 cross above EMA21 after a recent 52-week high,
   or a 52-week-high day where EMA9 was already above EMA21.
4. The volume bar must be from
   Alpaca `sip`, not `iex`. With `VOLUME_FILTER_ENABLE=true`, volume must also
   be `volume > SMA50(volume)`.
   Set `VOLUME_FILTER_ENABLE=false` to ignore only the SMA50 volume-size gate;
   the `sip` feed requirement still applies. Diagnostics are stored as
   `volume_sma50_pass`, `volume_feed_pass` and final `volume_pass`.
5. IBKR category breadth must be on: at least 65% of eligible category members
   above MA200 turns the gate on, and it stays on until breadth falls below 55%.
6. Planned entry-day market cap must be at least USD 2,000,000,000 by default
   (`MIN_MARKET_CAP_USD=2000000000`).
7. Latest known TTM revenue must be at least 20% higher than the comparable
   TTM revenue known roughly one year earlier (`REVENUE_YOY_MIN=0.20`).
8. The signal is known after the daily close, so the backtest enters at the next trading day's adjusted open.
9. Each trade gets USD 1000 notional exposure.
10. Initial stop is 5% below entry.
11. Once price has reached +10% from entry, a 5% trailing stop below the highest observed high becomes active for subsequent bars.

Signals are skipped while the same symbol already has an open simulated position.
Different symbols trade independently.

## Result Tables

All result tables are created by `init/schema.sql` and are Timescale hypertables
with `365 days` chunks:

| Table | Content |
|---|---|
| `backtest_wei_signals_daily` | actionable entry signals and indicator values |
| `backtest_wei_runs` | one row per run, parameters and summary metrics |
| `backtest_wei_trades` | simulated trades |
| `backtest_wei_equity_daily` | daily mark-to-market equity curve |

`DROP_ALL_WEI_TABLES_ON_START=false` is the default in `compose.yaml`.

## Run

Run this on the remote Docker host from this directory:

```bash
docker compose up --build
```

The project policy says Docker is not run locally from this workstation.

## Tests

```bash
python -m pytest tests -q
```
