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

4. **Entry confirmation (`ENTRY_CONFIRM_DAYS`):** entries fire N trading days
   after the flat->long flip, and only if the signal stayed long the whole
   time. This skips the whipsaw cohort right after the stress gate opens
   (10-40 day holds: ~25% win rate in research); 10 days roughly halved the
   max drawdown penalty vs. entering on the flip day.
5. **Catastrophe stop + time stop (`SL_PCT`, `TIME_STOP_DAYS`,
   `TIME_STOP_MIN_RET_PCT`):** exit on the close SL_PCT% below entry (wide by
   design - it fires on ~5-9% of trades and only amputates the left tail;
   tight stops destroyed returns), and exit trades still at/below
   TIME_STOP_MIN_RET_PCT% after TIME_STOP_DAYS trading days (dead-money
   recycling, never touches winners). A catastrophe-stopped symbol is locked
   until its signal resets to flat. Close-based stops do not protect against
   overnight gaps. Deliberately **no take-profit**: every tested TP cut the
   return by a third to half - the edge lives in the right tail.
6. **Re-entry cooldown after a time stop (`REENTRY_COOLDOWN_DAYS`):** a
   time-stopped symbol becomes an entry candidate again after this many
   trading days while its signal is still long, instead of staying locked
   until the next signal reset (0 = old behaviour). This recycles dead money
   ~4 weeks earlier at a cost of 11-16% total return vs. TIME_STOP_DAYS=60
   without re-entry (robust across the 2022-2026 and 2020-2026 windows).
   Holding-time research context: >100% of the strategy P&L sits in trades
   held longer than 70 calendar days (the <=70d cohort is net negative), so
   every stronger holding-time reduction that was tested - take-profits,
   plain and age-conditional trailing stops, hard 25-50-day caps with
   re-entry rolls, EMA-cross exits - cut returns by 30-60%. The average
   holding time (~10-11 weeks) is the edge, not an inefficiency.

Portfolio-mode rules: max 25 positions, max 2 per IBKR category, fills at the
close of the signal day, costs in bps per side, no leverage, no shorts, cash
earns nothing. Entry candidates are ranked by category momentum ascending with
per-stock momentum as tie-breaker (ties are common on mass-entry days when the
gate opens; before this tie-breaker admission degenerated to alphabetical
order, which alone swung the backtest between +83% and +665%). Positions that
grow past `TRIM_ABOVE_PCT` of equity are trimmed back to `TRIM_TARGET_PCT`
(single positions compounding into dominant clumps drove most of the drawdown).
Benchmark is the equal-weight daily-rebalanced index of the same universe.
It is not QQQ or another ETF. Price returns are never calculated across
`stock_core_market_metrics_daily.price_continuity_segment` boundaries. A symbol
with more than one segment in the loaded run window is excluded from that run;
any remaining >10x or <0.1x daily price factor aborts the benchmark instead of
silently contaminating it.

## Known limitations (read before trusting numbers)

- **Survivorship bias, part 1 (selection):** mostly addressed since
  `UNIVERSE_MCAP_ASOF=start` (the default): the universe is selected by the
  market cap as of the *window start* from the daily market caps in
  `stock_core_market_metrics_daily`, so the selection no longer knows the
  winners of the future. `end` restores the old biased behaviour for
  comparisons with legacy runs (in early local research the biased selection
  inflated the baseline from roughly benchmark level to +464%). Still not
  point-in-time: the IBKR category mapping (current snapshot — `ibkr_symbols`
  has no history; `stock_core_security_master_history` tracks master data but
  no market cap) and the coverage filter (looks at the whole run window).
- **Survivorship bias, part 2 (data):** `stock_core_market_metrics_daily`
  contains almost no delisted stocks (2 of ~4500 symbols stop updating), so
  bankruptcies/delistings are missing from the price data itself. Absolute
  returns are overstated even with a point-in-time selection; variant
  comparisons remain meaningful.
- **Hindsight risk in the regime scores:** `world_regime_daily_scores_mv` is
  recomputed retroactively with today's methodology. The day-lag prevents
  data look-ahead, but not methodology look-ahead.
- Entries happen only on flat→long flips (plus the first evaluation day), so a
  stock whose signal has been long for months does not enter when a slot
  frees up later. Exception: a time-stopped symbol re-enters after the
  re-entry cooldown while its signal is still long.

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
| `backtest_wei_stocks_trades` | one row per trade: symbol, category, entry/exit, gross return, target/effective weight, sizing tier, category momentum at entry, open flag, exit reason (`signal`/`sl`/`ts`/`open`, NULL for runs before the column existed) |
| `backtest_wei_stocks_equity_daily` | portfolio-mode hypertable, one row per day per run: equity vs benchmark, position count, gross exposure, composite score, stress state |

## Data sources (read-only)

- `stock_core_market_metrics_daily` — split-adjusted closes + market cap
- `ibkr_symbols` — IBKR category per symbol (latest fetch wins)
- `world_regime_daily_scores_mv` — daily composite stress score

## Tests

```bash
python -m pytest            # pure-Python tests, no DB needed
```
