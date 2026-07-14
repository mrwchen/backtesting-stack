# swing-stocks-minervini

Daily, causal research implementation of a Minervini/SEPA-style stock process.
The chart model lives in `backtest_models/minervini.py`; data access,
eligibility, persistence and portfolio simulation stay in this service.

The v9 default is `STAGE=all`, `SIMULATION_MODE=both` with the base label
`minervini_sepa_daily_v9_forward_shadow`
(`MODEL_VERSION=minervini_daily_v9`). It is a frozen forward protocol, not
another retrospective ranking search. `START_DATE=2020-01-02` supplies history,
indicator warm-up, setup lifecycle and causal calibration. Headline performance
starts only at `FORWARD_START_DATE=2026-07-13`, the first US session after the
last inspected data date of 2026-07-10. `END_DATE` is open so each invocation
extends through the newest available daily data.

One invocation performs one hidden, capacity-independent Market-On calibration
and persists exactly 34 forward paths:

- one unfiltered true first-touch research run with all four setup classes;
- one Flat-Base-only `relative_quality` shadow portfolio using the fixed,
  unweighted causal expected-R rank; and
- 32 Flat-Base-only `neutral` control portfolios using frozen session/salt
  hashes (`v9-neutral-control-00` through `v9-neutral-control-31`).

There is no VCP or combined portfolio, no absolute quality/cash gate, no
bootstrap-reweighted quality path and no automatic deployment decision. VCP,
Power Play and Tight Shelf remain visible only in the first-touch research run.
The 32 neutral controls measure sensitivity to arbitrary same-session selection;
they are correlated controls, not independent observations or a confidence
interval. Runtime never selects the best control. A retry creates new run IDs
rather than overwriting prior rows, and a complete batch is identified by its
final 34-run completion log.

Calibration continues causally during the forward period: a new outcome can
affect a later rank only after that outcome is complete and observable. Feature
weights, setup rules, portfolio constraints, costs and exits remain frozen.
Open positions are marked to market at the current data end instead of being
closed by fictional end-of-data trades.

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
   ranking mode. Persisted trade `r_multiple` values and the run-level
   `avg_r_multiple` use commission-aware net P&L divided by initial risk, the
   same outcome unit used by quality calibration.

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

   v9 deliberately separates relative selection from absolute profitability.
   `relative_quality` orders Flat Bases by their causal expected net R but does
   not interpret zero as a cash threshold. `neutral` ignores quality and orders
   the same eligible class only by its frozen session/salt hash. Absolute
   profitability is judged from the resulting forward portfolio, while the 32
   neutral controls show how much of the result could come from arbitrary
   same-session selection.

   `quality_rank` is the actual ordinal pre-session slate position (`1..N`),
   including neutral controls and score ties. Fill probability remains a
   diagnostic and is absent from both orders, eliminating the old signed-product
   defect where a lower fill probability improved a negative-quality candidate.
   Current RS, refreshed volume dry-up, setup age, point-in-time fundamentals,
   the trend-template result and structural components are soft quality
   features, not ex-post filters or entry gates. The trend-template contribution
   is 10% of each setup class's raw quality signal. Missing or
   expired observations stay unknown rather than becoming implicit passes or
   failures.

   The persisted `independent` forward run records every setup's true first
   touch without cash, slot, exposure, same-symbol, market, regime or
   fundamental-gate interaction. It is the complete forward research population,
   not the training population. The Quality shadow and all neutral controls
   reuse one causally filtered Market-On calibration set, but Flat portfolio
   fingerprints contain only Flat-Base setups and labels. VCP, Power Play and
   Tight Shelf remain in setup detection and first-touch research and can never
   enter a v9 portfolio arm.

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

   After each portfolio path, logs report closed-position concentration: Top
   1/3/5 share of positive gross R, net R and profit factor after removing those
   winners, and the number of distinct winning symbols. This is an ex-post
   robustness diagnostic only; it never changes historical orders or creates an
   alternative equity curve.

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
and Fill calibration-label multisets, including outcome values and availability
dates, plus the actual `online_calibration` or `preloaded` mode, Forward cutoff,
ranking mode, neutral salt and end-of-data policy. The Quality path is fixed and
unweighted; only neutral controls vary their deterministic pre-session order.

## Result tables

All service-owned tables use the prefix `backtesting_minervini_`:

| Table | Content |
|---|---|
| `stage_state` | model/config/input/output fingerprint for functional stage hand-off |
| `rs_daily` | daily RS values and cross-sectional ratings |
| `screen_daily` | every rankable symbol/session, including trend failures, plus fundamental, sponsorship and group diagnostics |
| `market_daily` | causal market status, breadth and exposure cap |
| `setups` | typed chart setups, continuity segment, pivot, stop and component scores |
| `runs` | model/input identity, Forward contract, parameters and aggregate metrics; one first-touch row, one Quality shadow row and 32 neutral control rows per complete v9 batch |
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

The v9 Forward protocol does not alter the result-table structure. It requires
no migration and no table drop; existing v8 rows remain available as historical
comparison baselines. Rebuild the image and run the complete functional
pipeline so v9 model-version fingerprints replace current stage and setup
state:

```bash
cd /home/wei/backtesting-stack/swing-stocks-minervini
docker compose up --build
```

Keep `START_DATE=2020-01-02`: it is the causal history/calibration start, not the
performance start. `FORWARD_START_DATE=2026-07-13` is frozen in configuration
and code. All persisted trades, events, equity and headline metrics begin there.
With open `END_DATE`, rerunning later extends the same deterministic Shadow
protocol through newly available data. A completed new trade may train later
sessions only after its exit result is observable.

`DROP_ALL_MINERVINI_TABLES_ON_START` remains `false`. Setting it to `true` would
delete all previous Minervini runs, trades, events, setups and stage state and
is not needed for v9. The Docker build context is the service directory, which
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
- The v9 Forward clock starts on 2026-07-13. Early runs can contain few or no
  completed trades and must not be treated as evidence. Changing ranking
  features, weights, gates, costs or exits creates a new model version and
  restarts that clock instead of rewriting v9 history.
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
