# swing-stocks-minervini

Daily, causal research implementation of a Minervini/SEPA-style stock process.
The chart model lives in `backtest_models/minervini.py`; data access,
eligibility, persistence and portfolio simulation stay in this service.

The v7 default is `STAGE=all`, `SIMULATION_MODE=both` with the base label
`minervini_sepa_daily_v7_class_local_32salt`
(`MODEL_VERSION=minervini_daily_v7`). With
`PORTFOLIO_RANKING_SENSITIVITY_ENABLE=true`, one invocation persists one
unfiltered, true first-touch `independent` research run and 32 complete,
market-gated `portfolio` paths with the configured cash, slot and exposure
controls. The configured run spans 2020-01-02 through
2026-07-10. Keep 2020-2023 as the calibration/development segment and report
2024-2026 separately: that later period has already been inspected and is a
retrospective validation window, not a pristine out-of-sample test.

The 32 neutral-ranking salts are frozen in source control (`v7-neutral-00`
through `v7-neutral-31`); the canonical single-run default is
`v7-neutral-00`. First-touch quality/fill labels are built exactly once and
reused unchanged by every path. Each portfolio path is independently
persisted in the existing result tables with a deterministic
`_portfolio_salt_NN` run-label suffix and its actual salt in `params`, so its
fingerprint and full equity/event path can be reproduced. Runtime never
selects a best salt. Only after all 32 paths complete, logs report the median,
adverse decile (p10 for return/quality metrics, p90 for drawdown) and worst
value of the core metrics. Arms commit separately to avoid one oversized
database transaction; absence of the final summary therefore identifies an
incomplete matrix. If an arm fails, already committed paths remain as partial
results. A retry creates new run IDs rather than overwriting them; compare only
a complete 32-salt matrix identified by its base label and completion summary.

## Model

The pipeline has three functional stages:

1. **Screen:** builds the cross-sectional RS rating and Minervini trend
   template. Price eligibility uses unadjusted `raw_close`, so a later stock
   split cannot retroactively turn an historically expensive leader into a
   penny stock. Trends, returns and chart geometry use split-adjusted OHLCV.
   The 12-month RS approximation uses four disjoint quarters with a 40/20/20/20
   recency weighting. The setup funnel's `screen_pass` means only that the
   symbol satisfies the price/liquidity rules and has a complete,
   continuity-safe RS observation. It does not require the trend template,
   fundamentals, an IBKR group diagnostic or the template's RS >= 70
   criterion. Those rows remain in the daily screen as causal ranking context.
2. **Setup:** evaluates `vcp`, `flat_base`, `power_play` and `tight_shelf`
   independently on every eligible detection day. Every class requires a
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
   on the same detection day remain separate research candidates. An identical
   structure is emitted once per continuity segment: expiry, later tightness or
   later volume dry-up do not turn the same resistance anchor into a new setup.
   A clearly higher, distinct continuation pivot can form a new base.
   `tight_shelf` remains
   visible in the first-touch research sample but is not a portfolio order
   class. `power_play` is deliberately exceptional rather than a short generic
   base: it requires at least a 100% prior advance within 40 sessions, a 10-30
   session consolidation no deeper than 20%, demand evidence from the mean of
   the three largest thrust-volume observations at least 1.5x their local
   median (with at least 20 valid observations), final tightness no greater
   than 8% and final-close dispersion no greater than 4%. If the whole base is
   deeper than 10%, its final range must also be materially narrower than its
   early range. A Power Play must also remain within 15% of its causally known
   trailing high over up to 252 sessions and close above its 50-session
   average. Young issues need at least 50 continuous sessions; as 150 and 200
   sessions become available, the check progressively adds the 50/150/200
   Stage-2 order and a rising 200-session average. This prevents a mechanical
   doubling from a crash low from being classified as leadership without
   excluding every young post-IPO leader. None of these rules uses the
   breakout day's close or volume.
   Every setup is bound to one positive `price_continuity_segment` and no
   pattern geometry may cross a segment boundary.
3. **Simulation:** places a stop-buy from information available after setup day
   D for session D+1. For each order session, all active candidates receive a
   fresh snapshot using observations through t-1 only. The v7 rank keeps three
   different questions separate: a base quality estimate is a setup-class-
   calibrated expected R-multiple, `fill_probability` estimates a near-term
   trigger, and `slate_priority` is used for pre-session order planning. Pivot
   readiness therefore cannot masquerade as trade quality. Calibration is
   walk-forward: a session may use only labels from trades completed before its
   information date, with causal priors/shrinkage when class history is sparse.
   Every completed label also retains the forecast that was genuinely available
   at its own entry. Per setup class, only those purged walk-forward forecasts
   may validate the quality model. Validation is fail-closed, class-local and
   evaluated only at fixed, non-overlapping two-year boundaries anchored at
   2000Q1. Each review epoch needs all eight quarters with at least 40 fully
   OOF-labelled outcomes per quarter. Every quarter forms four local forecast
   groups with at least five distinct completion dates per group; outcomes from
   the same completion date are first collapsed to one equal-weight cluster.
   The eight quarter-level top-minus-bottom lifts are the independent evidence.
   Aggregate group means must be monotonic and the block-level lift must retain
   a positive lower bound using `t=2.365`. A result is frozen for the following
   two-year epoch, so closing another trade cannot repeatedly retest the same
   history. This proves ranking information, not profitability; all groups may
   still have negative mean R. The causal posterior remains visible as the
   persisted diagnostic `quality_score`, but effective ranking quality and
   `slate_priority` stay neutral until validation. Validated quality,
   fill-weighted slate priority and `quality_rank` order candidates only within
   the same setup class. Across classes, candidates are neutrally interleaved
   with a causal session-and-salt hash; incomparable class posteriors are never
   placed on one global scale. Consequently, an unvalidated zero prior cannot
   systematically outrank a validated negative class. Neutral ties use the
   same stable salted hash rather than raw quality, fill probability or ticker
   order. Within a validated class, `quality_score * fill_probability` retains
   its expected-order-contribution meaning: for a negative expected R, a lower
   fill probability is less adverse and therefore ranks higher. Negative
   quality never acts as an entry gate.
   Current RS, refreshed volume dry-up, setup age, point-in-time fundamentals,
   the trend-template result and structural components are soft quality
   features, not ex-post filters or entry gates. The trend-template contribution
   is 10% of each setup class's raw within-class quality signal. Missing or
   expired observations stay unknown rather than becoming implicit passes or
   failures.

   The `independent` half of the default `both` run records each setup's true
   first touch without cash, slot, exposure, existing-position, market, regime
   or fundamental-gate interaction. Its completed, full-position net R outcome
   supplies one quality label only when the final exit is known; each active
   setup/session also supplies a capacity-independent fill label. The portfolio
   receives these labels but may consume only labels whose availability date is
   no later than that session's t-1 information date, so its own selected trades
   never train the model. A portfolio-only or OOS run builds the same first-touch
   calibration in memory; when OOS state starts before the reporting window,
   that hidden pass includes the pre-roll development sessions. In the 32-salt
   matrix this first-touch pass always uses canonical `v7-neutral-00`, then all
   paths consume the same frozen label tuples. The paths are correlated
   robustness scenarios, not independent samples or a confidence interval. No
   calibration or audit-only table is written. The `portfolio` half enforces
   the configured market state, cash, slots, per-trade risk and one aggregate
   gross-exposure cap. Before the session, every portfolio candidate gets its
   standalone risk-sized target; a
   class-locally ranked, neutrally interleaved, capacity-feasible slate is then
   built incrementally. Its integer
   minimum is funded first; one common scale distributes the remaining budget
   before deterministic integer rounding. A lower-ranked candidate that cannot
   retain the minimum may be skipped while a later, less capital-intensive
   candidate can still qualify. Every nominated order must retain at least
   `MIN_SLATE_RISK_UTILIZATION` (default 50%) of that target. At most
   `PORTFOLIO_MAX_DAILY_ORDERS` (default 3) new orders are nominated per
   session, independent of the larger limit for already open positions.
   Lower-ranked candidates that would create smaller residual orders are
   rejected as portfolio capacity instead of becoming dust trades. Frozen
   reservations are not recycled after observing which daily highs triggered.
   Exposure feedback is weighted by the frozen allocated/standalone share
   ratio, so a half-sized trade contributes 0.5 winner or loser risk units.
   Mixed exit sessions process every risk unit but use the adverse ordering of
   winners first and losses last because daily bars do not reveal cross-symbol
   exit order. Positions can re-enter after a scheduled same-symbol exit when
   conservative cash/capacity remains, without creating overlapping lots.
   Trend-template and fundamental values do not hard-block portfolio orders.
   Stops ratchet causally from the next session. A `power_play` completed high
   of +1R protects break-even on the following session, +2R protects +1R and so
   on; every other setup starts at +2R -> break-even, then +3R -> +1R. The time
   stop requires both insufficient MFE and a close back at or below the pivot.
   A price-continuity break invalidates pending setups and exits an open
   position at the first usable session open in the new segment, including
   normal exit slippage. It never backdates the exit to the old segment and
   never carries chart, trailing-MA or risk state across the discontinuity.

There is deliberately **no daily-close or breakout-volume confirmation** in
this model. A stop-buy fills on the breakout session when the daily high reaches
the trigger and the open is inside the buy zone. The breakout day's final close
and volume are not known at order time and do not influence that entry. If entry
and stop are both inside one daily bar, the simulator assumes the adverse path:
entry first, stop second.

SEC fundamentals are point-in-time candidate features but are neither
prerequisites for chart-setup recognition nor entry hard-gates. Current IBKR
taxonomy/group measurements and raw 13F sponsorship counts do not enter slate
ranking. They remain diagnostic
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
| `stock_core_market_metrics_daily` | adjusted OHLCV, raw close and mandatory price-continuity segment/break markers |
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
stage, including continuity and the daily candidate context used by t-1 ranking. `setup` or
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
| `screen_daily` | every rankable symbol/session, including trend failures, plus fundamental, sponsorship and group diagnostics |
| `market_daily` | causal market status, breadth and exposure cap |
| `setups` | typed chart setups, continuity segment, pivot, stop and component scores |
| `runs` | model/input identity, mode, parameters and aggregate metrics; one first-touch row plus 32 salted portfolio rows for the default sensitivity run |
| `breakout_events` | first pivot touch and fill/rejection decision plus scalar t-1 quality/fill/slate snapshot |
| `trades` | completed trade legs and entry-time attribution |
| `equity_daily` | portfolio equity, open positions and exposure state |

Hypertables use 365-day chunks. This service configures no TimescaleDB
compression. Database structure is defined only in `init/schema.sql`; runtime
Python validates and uses the schema but does not create or migrate it.

Each breakout event stores the ordinary scalar research fields
`snapshot_date`, `quality_score`, `fill_probability`, `slate_priority`,
`setup_age_sessions`, `distance_to_pivot_pct` and class-local `quality_rank`.
There is no
JSON snapshot payload and no audit-only table.

## Run

The v7 model change does not alter the v6 result-table structure. It requires no
migration and no table drop. Existing v5/v6 run rows therefore remain available
as comparison baselines. Rebuild the image and run the complete functional
pipeline so model-version fingerprints replace current stage state and setup
state:

```bash
cd /home/wei/backtesting-stack/swing-stocks-minervini
docker compose up --build
```

Keep `START_DATE=2020-01-02` for this full-history validation. Starting the
process only in 2024 would remove the completed 2022-2023 first-touch labels
that the fixed quality review needs at the 2024 boundary. Aggregate run metrics
cover the complete history; evaluate 2024-2026 separately when judging the
validated ranker.

`DROP_ALL_MINERVINI_TABLES_ON_START` remains `false`. Setting it to `true` would
delete all previous Minervini runs, trades, events, setups and stage state and
is not needed for v7. The Docker build context is the service directory, which
contains both the runtime code and its service-owned `backtest_models` package.

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
