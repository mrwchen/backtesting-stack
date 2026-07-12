# swing-stocks-minervini

Standalone backtester for Mark Minervini's SEPA swing approach (trend template,
RS rating, VCP breakouts) on the daily stock universe. Fully decoupled from the
other backtesting services.

**Data sources (read-only):**

| Table | Used for |
|---|---|
| `stock_core_market_metrics_current` | canonical current `(symbol, exchange, cik)` identity selection |
| `stock_core_market_metrics_daily` | adjusted daily OHLCV (2020+) |
| `stock_core_security_master_current` | universe filter (`quote_type = EQUITY`) |
| `ibkr_symbols` | IBKR `industry` / `category` taxonomy for group leadership |
| `stock_core_sec_quarterly_fundamental_events` | accession-keyed quarterly EPS, revenue, income and margins with prior-year comparators |
| `stock_core_13f_sponsorship_identity_events` | compact bitemporal institutional sponsorship changes from official SEC 13F filings |
| `alpaca_market_data_1day` | adjusted QQQ/VOO OHLCV for rally attempts, follow-through and distribution days |

All stock inputs are joined to one canonical current `(symbol, exchange, cik)`
identity before caching. This prevents a reused ticker or parallel exchange
identity from mixing another issuer's prices and SEC history into the test.

Quarterly fundamentals use discrete SEC facts where available. Strict,
context-matched YTD/FY subtraction covers Q2-Q4 values that are not reported as
discrete quarters. Currency-inconsistent or unavailable comparisons remain
unknown. Growth, acceleration, margins, streaks and annual stability become
visible only on the filing's `effective_date`.

The market-data service's schema initialization and startup SEC history
backfill and the separate official-SEC 13F service must run before this backtester. An absent quarterly-fundamental table fails at
the SQL boundary; an empty table also fails explicitly instead of silently
disabling the fundamental screen.

**Result tables (all prefixed `backtesting_minervini_`):**

| Table | Content |
|---|---|
| `backtesting_minervini_rs_daily` | daily 1-99 RS rating for every eligible symbol |
| `backtesting_minervini_screen_daily` | trend-template + fundamental + IBKR group-leadership + industry-breadth flags per symbol/day |
| `backtesting_minervini_market_daily` | causal QQQ/VOO market state, distribution count, follow-through events, breadth and entry-exposure cap |
| `backtesting_minervini_setups` | VCP bases with transparent 0-100 component scores + IBKR industry/category |
| `backtesting_minervini_runs` | one row per simulation run (params + metrics) |
| `backtesting_minervini_breakout_events` | every first pivot touch with its same-session stop-buy fill or rejection decision |
| `backtesting_minervini_trades` | trade legs per run + IBKR industry/category + world-regime attribution |
| `backtesting_minervini_equity_daily` | daily equity curve per run |

## Pipeline

Three decoupled stages, selected via `STAGE` (`screen`, `setup`, `sim`, `all`),
plus the bounded `sensitivity` research workflow.
Each stage reads its input from the DB, so stages can be re-run and tuned
independently; source data is cached locally as parquet in `./cache`.

```
screen : RS rating (cross-sectional percentile) + 8-point trend template
         + point-in-time quarterly growth, acceleration, margins, streak and
           annual stability
         + point-in-time 13F manager count and net institutional activity
         + IBKR group leadership:
           strong industry, strong category, strongest stocks inside both
           groups
         + IBKR industry breadth:
           industry gate on at >= 55% of members above their 200d MA, off
           below 45% -> rs_daily, screen_daily
         + QQQ-led market state: correction -> rally attempt -> confirmed
           uptrend -> uptrend under pressure. A day-4-or-later QQQ gain of at
           least 1.25% on higher volume confirms a rally. QQQ/VOO declines of
           at least 0.2% on higher volume add 25-session distribution events;
           four put the uptrend under pressure and six end it. Stock breadth
           above the 200d MA is a secondary exposure confirmation -> market_daily
setup  : causal daily/complete-week VCP detection and transparent scoring for
         contraction quality, final tightness, volume dry-up/slope, tight
         closes, duration, pivot proximity, overhead supply and prior advance.
         A missing global session resets
         that symbol's swing/volume structure, and validity expires in global
         market sessions even while the symbol is halted -> setups
sim    : causal same-session pivot stop-buy entries -> breakout_events, runs,
         trades, equity_daily. A setup detected after close D becomes orderable
         from D+1. Before each session, an order is built only from information
         already known at the previous close. A touch of the pivot trigger by
         the session high fills at max(open, trigger) plus slippage, provided
         the open is inside the 2% buy zone. The breakout day's final close and
         volume never influence that fill. The first pivot touch consumes the
         setup even when a gate, capacity limit or excessive gap blocks entry.
         SIMULATION_MODE=independent trades every triggered setup independently
         with no cash constraint, no position limit and no compounding. That
         research mode measures the signal across the whole universe.
         Before each session, SIMULATION_MODE=portfolio builds a deterministic,
         fully funded stop-buy slate from previous-close equity, cash, gross
         exposure and available position slots. Quantity is fixed using the
         worst permitted buy-zone fill. Same-day exits never fund replacement orders,
         and unused reservations are not reassigned after observing daily highs.
         Portfolio priority uses as-of setup quality: RS rating, stock/group RS,
         contraction count, risk distance, volume dry-up quality and growth
         fields from screen_daily. Independent mode uses the same pre-session
         sizing discipline with INITIAL_EQUITY but no portfolio constraints.
         Exits: structural stop (maximum 8%, also checked on the entry
         bar), failed-breakout exit, ten-session no-progress time stop,
         delayed profit protection, MA-trail, end-of-data when
         the symbol has an executable final-session bar. Otherwise the position
         remains open and is only marked at its last known close.
         Market status and exposure cap computed at close t control entries from
         session t+1 (MARKET_FILTER_ENABLE); open positions keep running into their exits.
         A base is invalidated by a stop-level breach. If trigger and stop both
         occur inside one daily bar, the unknown intraday order is resolved
         adversely as fill first and stop second. An open at or below structural
         invalidation cancels the order before any later pivot touch.
         Optional world-regime entry filtering blocks entries whose latest known
         regime label is not in REGIME_ALLOWED_LABELS. A score for day d is only
         usable from d+1 (its 01:00 America/New_York cutoff); weekend rows remain
         usable for the next session. Trades carry that same causally available
         world-regime composite score at entry as attribution.
         Portfolio exposure starts at 25%, steps up through 50/75/100% only
         after two consecutive closed winners, steps down after a loss, and
         resets after two losses, a 4% drawdown or a non-investable market
         state. The actual entry limit is the lower of that feedback level and
         the market cap: 0% in correction/rally attempt, 25-100% in a confirmed
         uptrend depending on breadth, and 25-50% under pressure. Unrealized
         PnL never raises the exposure level.
sensitivity:
         computes screen and market state once, detects one frozen Moderate
         setup population (VCP score 60, dry-up 0.20-0.85), then simulates it
         twice on 2020-01-02..2023-12-31: market gate off and market gate on.
         Configuration outside that development window is rejected before DB
         access, so no 2024+ data enters this experiment. Same-session execution
         and all other entry, exit and risk rules stay fixed. In independent
         mode the on arm is a binary index-state gate: correction, rally attempt
         and unavailable states block entries; positive exposure caps do not
         scale positions. Each arm produces its own runs, breakout_events,
         trades and equity_daily rows. VCP setups remain in memory because the
         shared setups table is not run-scoped. Negative setup IDs in sensitivity
         events are therefore run-local identifiers, not setup-table keys.
```

The fixed market-filter ablation arms are:

| Variant | Minimum VCP score | Dry-up band | Market gate |
|---|---:|---:|---:|
| `market_off` | 60 | 0.20-0.85 | off |
| `market_on` | 60 | 0.20-0.85 | on |

## Run

First rebuild the upstream SEC 13F data model. This drops the raw and compact
13F tables, backfills the official SEC source events, then builds the compact
identity event stream:

```bash
cd /home/wei/market-data-stack/sec_13f_data_fetcher
DROP_ALL_SEC_13F_TABLES_ON_START=true docker compose up --build
```

Then reset the Minervini result tables and run the development-only ablation:

```bash
cd /home/wei/backtesting-stack/swing-stocks-minervini
DROP_ALL_MINERVINI_TABLES_ON_START=true docker compose up --build
```

Both destructive switches default to `false`; use the explicit overrides only
for this intentionally incompatible source-model rebuild.

All parameters (thresholds, VCP geometry, risk settings) are env vars in
`compose.yaml`. `SCREEN_PERSIST=universe` persists every rankable symbol/day
instead of only passes (larger, useful for Grafana inspection).

## Tests

```bash
python -m pytest tests -q
```

## Known caveats

- **Survivorship bias:** the price history only contains today's listed
  universe backfilled to 2020; delisted losers are missing. Breakout results
  are biased optimistic — read them conservatively.
- Price history starts 2020-01-02; with ~1 year of indicator warmup the
  effective backtest window begins ~2021.
- Quarterly SEC values are GAAP/IFRS facts or strict context-matched
  derivations; non-GAAP values and missing comparators are not estimated.
- 13F is intrinsically delayed. `period_of_report` is the economic date,
  `accepted_at` is the knowledge timestamp when available, and the backtester
  uses only `effective_date` (the next UTC calendar day). Amendments form new
  knowledge states and never rewrite earlier backtest dates.
- This is a deterministic SEPA-inspired research proxy, not a claim to reproduce
  Mark Minervini's complete proprietary discretionary process.
- The 2024+ results have already been inspected during model diagnosis and are
  excluded from the development-only market-filter ablation. A clean forward
  holdout starts after the latest date already examined.
- Historical stock breadth uses the currently canonical stock universe and is
  therefore not a true point-in-time membership series. In independent-mode
  sensitivity runs breadth only changes positive exposure-cap magnitudes, which
  do not affect the binary on/off entry decision; the ablation is consequently
  interpreted as an index-state gate test, not as a point-in-time breadth test.
- The QQQ/VOO state machine is a transparent IBD-inspired approximation, not
  IBD/MarketSurge's proprietary market-status service. QQQ starts rally attempts
  and follow-through days; either configured proxy can add one distribution
  event per session.
- Daily bars do not reveal the intraday order of high and low. When both the
  pivot trigger and stop are inside one bar, the simulator deliberately assumes
  the adverse path: entry first, then stop. A daily model cannot reproduce
  Minervini's intraday volume extrapolation; close and volume from the breakout
  session are therefore excluded from the entry decision entirely.
- The global session grid is the union of available stock bars. A gap in one
  symbol is detected, but a complete provider outage affecting every symbol on
  the same exchange session is not distinguishable without a separate exchange
  calendar source. A missing primary-index bar is marked `DATA_UNAVAILABLE` and
  blocks new entries in the following session.
