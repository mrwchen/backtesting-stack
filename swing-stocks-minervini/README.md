# swing-stocks-minervini

Standalone backtester for Mark Minervini's SEPA swing approach (trend template,
RS rating, VCP breakouts) on the daily stock universe. Fully decoupled from the
other backtesting services.

**Data sources (read-only):**

| Table | Used for |
|---|---|
| `stock_core_market_metrics_daily` | adjusted daily OHLCV (2020+) |
| `stock_core_security_master_current` | universe filter (`quote_type = EQUITY`) |
| `ibkr_symbols` | IBKR `industry` / `category` taxonomy for group leadership |
| `stock_core_sec_fundamentals_asof_daily` | SEC TTM revenue/margins/net income (point-in-time: `period_end_date` is the filing availability date) |

Quarterly EPS is derived from consecutive quarterly TTM net-income diffs
divided by diluted shares (the earnings calendar carries no reported EPS in
practice). Annual-only filers never qualify for the EPS flag.

**Result tables (all prefixed `backtesting_minervini_`):**

| Table | Content |
|---|---|
| `backtesting_minervini_rs_daily` | daily 1-99 RS rating for every eligible symbol |
| `backtesting_minervini_screen_daily` | trend-template + fundamental + IBKR group-leadership + industry-breadth flags per symbol/day |
| `backtesting_minervini_market_daily` | daily market breadth (% above 200d MA) + hysteresis gate |
| `backtesting_minervini_setups` | detected VCP bases (pivot, stop, contraction chain) + IBKR industry/category |
| `backtesting_minervini_runs` | one row per simulation run (params + metrics) |
| `backtesting_minervini_trades` | trade legs per run + IBKR industry/category + world-regime attribution |
| `backtesting_minervini_equity_daily` | daily equity curve per run |

## Pipeline

Three decoupled stages, selected via `STAGE` (`screen`, `setup`, `sim`, `all`).
Each stage reads its input from the DB, so stages can be re-run and tuned
independently; source data is cached locally as parquet in `./cache`.

```
screen : RS rating (cross-sectional percentile) + 8-point trend template
         + point-in-time fundamentals
         + IBKR group leadership:
           strong industry, strong category, strongest stocks inside both
           groups
         + IBKR industry breadth:
           industry gate on at >= 55% of members above their 200d MA, off
           below 45% -> rs_daily, screen_daily
         + market breadth gate (share of stocks above their 200d MA, on >= 50%
         / off < 45% hysteresis) -> market_daily
setup  : VCP detection on screen-pass days -> setups
sim    : stop-buy breakout entries over the pivot -> runs, trades, equity_daily
         SIMULATION_MODE=independent trades every triggered setup independently
         with no cash constraint, no position limit and no compounding. That
         research mode measures the signal across the whole universe.
         SIMULATION_MODE=portfolio applies cash, max gross exposure and max
         open-position constraints. Portfolio sizing uses current marked-to-
         market equity, while independent sizing uses INITIAL_EQUITY. When
         several entries compete for limited portfolio capacity, the portfolio
         mode prioritizes as-of setup quality: RS rating, stock/group RS,
         contraction count, volume dry-up and growth fields from screen_daily.
         Exits: stop (also checked on the entry bar itself), partial at
         PARTIAL_AT_R, MA-trail, end-of-data.
         New entries are blocked while the market breadth gate is off
         (MARKET_FILTER_ENABLE); open positions keep running into their exits.
         Optional world-regime entry filtering blocks entries whose latest known
         regime label is not in REGIME_ALLOWED_LABELS. Trades carry the latest
         known world-regime composite score at entry as attribution.
```

## Run

```bash
docker compose up --build        # runs STAGE=all over START_DATE..END_DATE
```

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
- Fundamentals are TTM (SEC); quarterly EPS is reconstructed from TTM diffs
  and is therefore slightly noisier than reported quarterly EPS.
