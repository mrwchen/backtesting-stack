# swing-stocks-minervini

Daily, causal research implementation of a Minervini/SEPA-style stock process.
The chart model lives in `backtest_models/minervini.py`; data access,
eligibility, persistence and portfolio simulation stay in this service.

The Phase-1 default is `STAGE=all`, `SIMULATION_MODE=both` with the base label
`minervini_sepa_daily_v4_dynamic_rank`
(`MODEL_VERSION=minervini_daily_v4`). One invocation persists two separate
runs in one database transaction: an unfiltered, true first-touch `independent`
research run and the market-gated `portfolio` run with the configured cash,
slot and exposure controls. It uses the known 2020-2023 development window. The 2024-2026 period
has already been inspected and must not be reported as a pristine
out-of-sample test.

## Model

The pipeline has three functional stages:

1. **Screen:** builds the cross-sectional RS rating and Minervini trend
   template. Price eligibility uses unadjusted `raw_close`, so a later stock
   split cannot retroactively turn an historically expensive leader into a
   penny stock. Trends, returns and chart geometry use split-adjusted OHLCV.
   The 12-month RS approximation uses four disjoint quarters with a 40/20/20/20
   recency weighting.
2. **Setup:** eagerly evaluates `vcp`, `flat_base`, `power_play` and
   `tight_shelf` on every eligible detection day. Every class requires a
   material prior advance and a causal, pre-breakout tight area. VCP accepts one
   modestly noisy contraction, keeps confirmed structure across an ambiguous
   outside bar and permits bases up to 130 sessions. Lower volume dry-up is
   always scored better; there is no artificial preferred midpoint, lower
   cutoff or hard setup-score threshold. Missing/zero volume receives no
   dry-up points but does not erase a valid price structure. Range pivots come
   from resistance established before the final tight area, so a breakout day
   cannot redefine its own higher pivot. Stable resistance anchors, nearby
   pivots and overlapping base intervals identify the same price structure;
   nested classifications collapse deterministically to the strongest label
   instead of producing type-dependent duplicates. Genuinely distinct pivots
   on the same detection day remain separate research candidates. Duplicate
   suppression lasts for the configured setup-validity window.
3. **Simulation:** places a stop-buy from information available after setup day
   D for session D+1. For each order session, all active candidates receive a
   fresh snapshot using observations through t-1 only. The dynamic rank combines
   pivot readiness/distance, current RS, refreshed volume dry-up, setup age,
   point-in-time fundamentals and structural setup components. Every component
   is scaled to a common 0-100 range; setup-class-specific weights produce one
   continuous, cross-class comparable score. Missing features receive a neutral
   contribution rather than becoming an implicit pass or failure. Missing or
   expired SEC observations remain unknown, and an incomplete current volume
   window cannot fall back to a stale detect-day dry-up value.

   The `independent` half of the default `both` run records each setup's true
   first touch without cash, slot, exposure, existing-position, market, regime
   or fundamental-gate interaction. The `portfolio` half enforces the configured market state,
   cash, slots, per-trade risk and one aggregate gross-exposure cap. Before the
   session, every portfolio candidate gets its standalone risk-sized target; a
   ranked, capacity-feasible slate is then built incrementally. Its integer
   minimum is funded first; one common scale distributes the remaining budget
   before deterministic integer rounding. A lower-ranked candidate that cannot
   retain the minimum may be skipped while a later, less capital-intensive
   candidate can still qualify. Every nominated order must retain at least
   `MIN_SLATE_RISK_UTILIZATION` (default 50%) of that target.
   Lower-ranked candidates that would create smaller residual orders are
   rejected as portfolio capacity instead of becoming dust trades. Frozen
   reservations are not recycled after observing which daily highs triggered.
   Exposure feedback is weighted by the frozen allocated/standalone share
   ratio, so a half-sized trade contributes 0.5 winner or loser risk units.
   Mixed exit sessions process every risk unit but use the adverse ordering of
   winners first and losses last because daily bars do not reveal cross-symbol
   exit order. Positions can re-enter after a scheduled same-symbol exit when
   conservative cash/capacity remains, without creating overlapping lots. A
   daily trend/fundamental eligibility failure blocks only that portfolio
   session; a pivot touch while blocked is persisted with its rejection reason.
   Stops ratchet causally from the next
   session: +2R protects break-even, +3R protects +1R and so on. The time stop
   requires both insufficient MFE and a close back at or below the pivot.

There is deliberately **no daily-close or breakout-volume confirmation** in
this model. A stop-buy fills on the breakout session when the daily high reaches
the trigger and the open is inside the buy zone. The breakout day's final close
and volume are not known at order time and do not influence that entry. If entry
and stop are both inside one daily bar, the simulator assumes the adverse path:
entry first, stop second.

SEC fundamentals are point-in-time candidate context but are not prerequisites
for chart-setup recognition. Current IBKR taxonomy/group measurements and raw
13F sponsorship counts do not enter candidate ranking. They remain diagnostic
or entry-attribution fields only. The model has no hard IBKR group, IBKR
industry-breadth or institutional-sponsorship gate; their pass fields remain
honest diagnostics instead of being forced to `true` by an enable switch. The
QQQ/VOO market state remains a causal portfolio entry/exposure control; its
close-t state is usable from t+1 and does not filter the default first-touch
research run.

`GOLD_CASES` in the model contains a source-controlled review set of historical
leaders and failed breakouts. It is a regression and chart-review control, not
an optimization target. No gold-case or audit-only records are written to the
database.

## Data sources

| Table | Use |
|---|---|
| `stock_core_market_metrics_current` | canonical current symbol/exchange/CIK identity |
| `stock_core_market_metrics_daily` | adjusted OHLCV plus raw close |
| `stock_core_security_master_current` | current equity universe |
| `stock_core_sec_quarterly_fundamental_events` | filing-effective quarterly fundamentals |
| `stock_core_13f_sponsorship_identity_events` | filing-effective institutional diagnostics; not a rank input |
| `ibkr_symbols` | current industry/category attribution; not a rank input |
| `alpaca_market_data_1day` | QQQ/VOO market-state inputs |
| `world_regime_daily_scores_mv` | optional causally lagged regime gate and attribution |

The versioned parquet-cache contract is validated before reuse. Sponsorship is
checked fresh at the source boundary, so a stale local cache cannot conceal a
changed or empty optional source. Functional stage fingerprints bind the exact
date range, adjusted/raw price matrices and every source input relevant to the
stage, including the daily candidate context used by t-1 ranking. `setup` or
`sim` therefore refuses stale output from a different model, configuration or
source snapshot. A PostgreSQL advisory lock serializes writers because the
stage tables are functional current state, not run-scoped copies. Runtime
validates the incompatible result schema before loading source data and fails
with the required drop instruction if old columns remain.

## Result tables

All service-owned tables use the prefix `backtesting_minervini_`:

| Table | Content |
|---|---|
| `stage_state` | model/config/input/output fingerprint for functional stage hand-off |
| `rs_daily` | daily RS values and cross-sectional ratings |
| `screen_daily` | trend pass plus fundamental, sponsorship and group diagnostics |
| `market_daily` | causal market status, breadth and exposure cap |
| `setups` | typed chart setups, pivot, stop and component scores |
| `runs` | model/input identity, mode, parameters and aggregate metrics; two rows for the default `both` run |
| `breakout_events` | first pivot touch and fill/rejection decision plus scalar t-1 rank snapshot |
| `trades` | completed trade legs and entry-time attribution |
| `equity_daily` | portfolio equity, open positions and exposure state |

Hypertables use 365-day chunks. This service configures no TimescaleDB
compression. Database structure is defined only in `init/schema.sql`; runtime
Python validates and uses the schema but does not create or migrate it.

Each breakout event stores the ordinary scalar research fields
`snapshot_date`, `dynamic_setup_score`, `readiness_score`, `context_score`,
`setup_age_sessions`, `distance_to_pivot_pct` and `candidate_rank`. There is no
JSON snapshot payload and no audit-only table.

## Run

The v4 Phase-1 rebuild is intentionally incompatible with every earlier result
schema. There is no migration and no backward-compatibility path. The first v4
run must drop and recreate all `backtesting_minervini_*` tables through the init
container:

```bash
cd /home/wei/backtesting-stack/swing-stocks-minervini
DROP_ALL_MINERVINI_TABLES_ON_START=true docker compose up --build
```

`DROP_ALL_MINERVINI_TABLES_ON_START` defaults to `false`; do not keep the
destructive override enabled after the one-time reset. This reset deletes all
previous Minervini runs, trades, events, setups and stage state. The Docker
build context is the service directory, which contains both the runtime code
and its service-owned `backtest_models` package.

All runtime parameters are documented in `compose.yaml`. The daily screen table
always stores every rankable symbol/session, including failed trend-template
rows. This complete causal context is required so separate `setup` and `sim`
stages produce the same ranking as `STAGE=all`. Run the unit suite from the
service directory with:

```bash
python -m pytest tests -q
```

## Interpretation limits

- The loader selects the **current** canonical equity universe and current IBKR
  taxonomy. Delisted securities and true historical membership are absent.
  This creates survivorship bias and prevents a fully point-in-time universe.
- Price history starts in 2020; roughly one prior year is needed to warm the
  longest indicators.
- 2024 through the already reviewed 2026 data are development evidence, not a
  clean holdout. Only future, previously unseen data after the latest inspected
  date can provide a new forward test.
- SEC fundamentals and 13F data are used only from their effective knowledge
  dates. Missing/non-comparable filing values remain unknown instead of being
  estimated.
- Daily bars cannot reveal intraday high/low order or reproduce Minervini's
  discretionary intraday volume extrapolation. The deterministic adverse-path
  convention should therefore be read as a research approximation.
- The QQQ/VOO state machine is an explicit IBD-inspired approximation, not the
  proprietary IBD/MarketSurge status service.
- The gold set is intentionally small and human-curated. Thresholds must not be
  tuned to force those named examples to pass; use walk-forward validation and
  a genuinely unseen forward period to control overfitting.
- Gold controls are logged with fixed 90-day-before/5-day-after windows,
  observed setup classes and the source-fingerprint prefix; they never gate a
  run or write audit-only rows to TimescaleDB.
- Pyramiding/add-ons are not simulated. Correct add-ons require lot-level entry,
  risk and exit attribution; silently averaging them into one position would
  make the backtest less trustworthy.
