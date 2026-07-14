# swing-stocks-minervini

Daily, causal research implementation of a Minervini/SEPA-style stock process.
The chart model lives in `backtest_models/minervini.py`; data access,
eligibility, persistence and portfolio simulation stay in this service.

The v8 default is `STAGE=all`, `SIMULATION_MODE=both` with the base label
`minervini_sepa_daily_v8_ranking_experiment`
(`MODEL_VERSION=minervini_daily_v8`). With
`PORTFOLIO_RANKING_EXPERIMENT_ENABLE=true`, one invocation performs one hidden,
capacity-independent Market-On calibration, persists one unfiltered true
first-touch research run and then persists a fixed matrix of 288 complete
portfolio paths:

- ranking modes: `neutral`, `quality_only`, `validated`;
- sleeves: `flat_base`, `vcp`, `combined` (`flat_base,vcp`); and
- 32 frozen completion-day-cluster bootstrap salts per mode/sleeve case.

The configured run spans 2020-01-02 through 2026-07-10. Keep 2020-2023 as the
calibration/development segment and report 2024-2026 separately: that later
period has already been inspected and is retrospective validation, not a
pristine out-of-sample test.

The salts are frozen in source control (`v8-bootstrap-00` through
`v8-bootstrap-31`); the canonical single-run default is `v8-bootstrap-00`.
Outside the experiment matrix, a normal single run uses the unbootstrapped
Market-On label set; its `NEUTRAL_RANK_SALT` controls deterministic ranking ties
only. Inside the matrix, each salt additionally applies a deterministic positive
Bayesian-bootstrap weight to every setup-class/completion-day quality-label
cluster. A later cluster cannot change an earlier cluster's weight, and fill
labels are never reweighted. This makes the matrix vary calibration uncertainty
inside setup classes instead of merely rotating class order. Runtime never
selects a best salt or best case. After all 32 paths of a case complete, logs
report its median, adverse decile (p10 for return/quality metrics, p90 for
drawdown) and worst core metric. Arms commit separately; absence of the final
9-case/288-path completion message identifies an incomplete matrix. A retry
creates new run IDs rather than overwriting prior rows. Compare only a complete
matrix identified by its base label and final completion log. The 32 paths
remain correlated robustness scenarios, not independent observations or a
confidence interval.

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
   `tight_shelf` and `power_play` remain visible in the first-touch research
   sample but are not portfolio order classes. `power_play` is deliberately
   exceptional rather than a short generic
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
   D for session D+1. For every order session, all active candidates receive a
   fresh snapshot using observations through t-1 only. Quality is a
   setup-class-calibrated expected **net** R-multiple. Every class targets the
   same outcome unit, so calibrated values can form one global slate.
   `fill_probability` estimates only whether the stop order may touch; it is a
   persisted diagnostic and never changes ranking, tie-breaking or portfolio
   admission. `slate_priority` is the effective quality used by the selected
   ranking mode.

   Calibration is strictly walk-forward: a session may use only labels whose
   trade outcome was complete before its information date. Sparse histories
   remain shrunk toward the explicit zero-R prior. The hidden calibration pass
   is independent of cash, slots, portfolio selection and feedback, but applies
   the exact exogenous market/regime entry gates used by the portfolio. Thus a
   Market-Off first touch remains visible in the separately persisted research
   run but cannot train quality or fill. A quality label becomes available only
   at the final exit of an independently executable Market-On trade; fill labels
   are recorded only for Market-On setup/sessions. Portfolio trades never train
   the model, and no calibration/audit-only rows are written to the database.

   Validation is fail-closed and class-local. Independently fillable Market-On
   entries of the same setup class and information date form one frozen
   competition group. A group becomes usable only after every declared member
   has a final outcome; its completion date is the latest member availability
   date. Single-candidate groups, incomplete groups and groups whose stored
   purged walk-forward predictions cannot distinguish a top candidate provide
   no ranking evidence.

   At each calendar-quarter boundary the validator reviews the latest eight
   fully completed competition-completion quarters and freezes that result for
   the new quarter. Every reviewed quarter needs at least 40 labels from at
   least five complete competition groups. Within each group, the realized net
   R of the highest stored out-of-sample prediction is compared with the rest;
   tied top predictions are averaged. Competition groups are equal-weight when
   forming the quarter result. Across the eight quarter blocks, both
   top-minus-rest lift and the top candidates' own net R must retain positive
   lower confidence bounds using `t=2.365`. Bootstrap weights deliberately do
   not enter this deployment proof: they vary the posterior estimate, not the
   evidence that permits exposure. Therefore a merely less-bad but still
   negative ranking can no longer pass validation.

   Ranking modes have fixed meanings:

   - `neutral` ignores quality and orders only by the stable session/salt hash;
   - `quality_only` orders globally by the causal expected net R without a
     deployment hurdle; and
   - `validated` exposes quality only after the robust review. Cash is an
     explicit zero-return candidate, so an unvalidated or nonpositive setup is
     rejected as `non_positive_quality` before sizing or capacity allocation.

   `quality_rank` is the global expected-net-R rank. Fill probability is absent
   from all three orders, eliminating the old signed-product defect where a
   lower fill probability improved a negative-quality candidate.
   Current RS, refreshed volume dry-up, setup age, point-in-time fundamentals,
   the trend-template result and structural components are soft quality
   features, not ex-post filters or entry gates. The trend-template contribution
   is 10% of each setup class's raw quality signal. Missing or
   expired observations stay unknown rather than becoming implicit passes or
   failures.

   The persisted `independent` half of the default `both` run records every
   setup's true first touch without cash, slot, exposure, same-symbol, market,
   regime or fundamental-gate interaction. It is the complete research
   population, not the training population. The 288 portfolio arms reuse one
   Market-On base-label set. For each salt, quality labels receive the frozen
   completion-day-cluster bootstrap weights and the resulting tuple is reused
   by all nine mode/sleeve cases. No outcome, metric or later cluster influences
   an earlier cluster's weight.

   Flat Base and VCP are evaluated both as isolated sleeves and as one combined
   control. The combined `quality_only` and `validated` slates compare their
   calibrated expected net R directly; there is no forced one-third class
   rotation. Power Play and Tight Shelf remain in setup detection and
   first-touch research but are always `setup_class_research_only` in portfolio
   mode because their current evidence is insufficient for deployment.

   The portfolio enforces the configured market state, cash, slots, per-trade
   risk and one aggregate gross-exposure cap. Before the session, every admitted
   candidate gets its standalone risk-sized target; the globally ordered,
   capacity-feasible slate is then built incrementally. Its integer
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

Every simulation fingerprint also binds the exact order-independent Quality
and Fill calibration-label multisets, including outcome values, weights and
quality-label competition sizes, plus the actual `online_calibration` or
`preloaded` mode. Thus a canonical unbootstrapped single run cannot share an
input identity with a bootstrap arm even when its configured salt is the same.

## Result tables

All service-owned tables use the prefix `backtesting_minervini_`:

| Table | Content |
|---|---|
| `stage_state` | model/config/input/output fingerprint for functional stage hand-off |
| `rs_daily` | daily RS values and cross-sectional ratings |
| `screen_daily` | every rankable symbol/session, including trend failures, plus fundamental, sponsorship and group diagnostics |
| `market_daily` | causal market status, breadth and exposure cap |
| `setups` | typed chart setups, continuity segment, pivot, stop and component scores |
| `runs` | model/input identity, mode, parameters and aggregate metrics; one first-touch row plus 288 portfolio rows for the default v8 experiment |
| `breakout_events` | first pivot touch and fill/rejection decision plus scalar t-1 quality/fill/slate snapshot |
| `trades` | completed trade legs and entry-time attribution |
| `equity_daily` | portfolio equity, open positions and exposure state |

Hypertables use 365-day chunks. This service configures no TimescaleDB
compression. Database structure is defined only in `init/schema.sql`; runtime
Python validates and uses the schema but does not create or migrate it.

Each breakout event stores the ordinary scalar research fields
`snapshot_date`, `quality_score`, `fill_probability`, `slate_priority`,
`setup_age_sessions`, `distance_to_pivot_pct` and global `quality_rank`.
There is no
JSON snapshot payload and no audit-only table.

## Run

The v8 model change does not alter the existing result-table structure. It
requires no migration and no table drop. Existing run rows remain available as
comparison baselines. Rebuild the image and run the complete functional
pipeline so v8 model-version fingerprints replace current stage and setup
state:

```bash
cd /home/wei/backtesting-stack/swing-stocks-minervini
docker compose up --build
```

Keep `START_DATE=2020-01-02` for this full-history validation. Starting the
process only in 2024 would remove the completed 2022-2023 Market-On first-touch
labels that the rolling quality review needs at the 2024 boundary. Aggregate run metrics
cover the complete history; evaluate 2024-2026 separately when judging the
validated ranker.

`DROP_ALL_MINERVINI_TABLES_ON_START` remains `false`. Setting it to `true` would
delete all previous Minervini runs, trades, events, setups and stage state and
is not needed for v8. The Docker build context is the service directory, which
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
