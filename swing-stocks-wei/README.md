# swing-stocks-wei — Regime-Gated Trend Backtester (Stocks)

Standalone backtester for the stock version of the regime-gated trend strategy
(the index version lives in `../swing-indices-wei`). It can run in two modes:

- `independent` tests every valid stock signal as its own hypothetical trade,
  without cash, portfolio slots, or category caps deciding which trades are
  admitted. This is the default in `compose.yaml` and is meant for signal
  quality research.
- `portfolio` simulates a long-only portfolio of US large caps with cash,
  max-position limits, category caps, transaction costs, and a benchmark.

The strategy is driven by three ideas that survived both research windows
(2022–2026 in-sample, 2020–2021 out-of-sample):

1. **Per-stock signal:** stay long unless the stock's EMA9 < EMA21 **and** the
   market-wide stress light is red (composite score hysteresis ≥57 / ≤52,
   previous day's `world_regime_daily_scores_mv.composite_score`).
2. **No new entries while the stress light is red.** This removes the worst
   trade cohort found in research (crash whipsaws, e.g. Feb/Mar 2020: 30% win
   rate) while exits keep working, so the book empties itself in bear phases.
3. **Category-momentum sizing (not filtering):** position size depends on the
   63-day momentum of the stock's IBKR category (equal-weight index of all
   category members) at entry:

   | tier | category momentum | default weight |
   |------|-------------------|----------------|
   | `deep` | ≤ −10 % | 5 % of equity — the robust sweet spot in both windows |
   | `mild` | −10 % … 0 % (or unknown) | 3 % |
   | `pos`  | > 0 % | 2 % — down-weighted, not banned (a hard ban failed out-of-sample 2021) |

Portfolio-mode rules: max 25 positions, max 2 per IBKR category, fills at the
close of the signal day, costs in bps per side, no leverage, no shorts, cash
earns nothing. Benchmark is the equal-weight daily-rebalanced index of the same
universe.

## Known limitations (read before trusting numbers)

- **Survivorship bias:** the universe uses the *latest* market cap and the IBKR
  category mapping is a current snapshot. Recovery-style entries look better
  than they were in real time. A point-in-time universe from
  `stock_core_security_master_history` is the planned next step.
- **Hindsight risk in the regime scores:** `world_regime_daily_scores_mv` is
  recomputed retroactively with today's methodology. The day-lag prevents
  data look-ahead, but not methodology look-ahead.
- Entries happen only on flat→long flips (plus the first evaluation day), so a
  stock whose signal has been long for months does not enter when a slot
  frees up later.

## Run

```bash
cd backtesting-stack/swing-stocks-wei
docker compose up --build        # init schema, then run one backtest
```

All parameters are environment variables in `compose.yaml` (mode, window, EMA
spans, stress thresholds, universe filters, sizing weights, position limits,
costs). `TOP_N_PER_CATEGORY=0` means all stocks per selected category after the
market-cap and coverage filters. Each run appends one row to
`backtest_wei_stocks_runs` plus its trades. Portfolio runs also append the daily
equity curve. Re-running with different parameters and `RUN_LABEL`s is the
intended workflow for comparisons in Grafana.

## Result tables (TimescaleDB, prefix `backtest_wei_stocks_`)

| table | content |
|---|---|
| `backtest_wei_stocks_runs` | one row per run: mode, all parameters, portfolio metrics where applicable, and trade-distribution metrics |
| `backtest_wei_stocks_trades` | one row per trade: symbol, category, entry/exit, gross return, target/effective weight, sizing tier, category momentum at entry, open flag |
| `backtest_wei_stocks_equity_daily` | portfolio-mode hypertable, one row per day per run: equity vs benchmark, position count, gross exposure, composite score, stress state |

## Data sources (read-only)

- `stock_core_market_metrics_daily` — split-adjusted closes + market cap
- `ibkr_symbols` — IBKR category per symbol (latest fetch wins)
- `world_regime_daily_scores_mv` — daily composite stress score

## Tests

```bash
python -m pytest            # pure-Python tests, no DB needed
```
