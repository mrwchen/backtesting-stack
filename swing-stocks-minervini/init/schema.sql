-- Idempotent schema for the Minervini swing backtester.
-- All tables are prefixed backtesting_minervini_.
-- Safe to re-run: statements use IF NOT EXISTS or equivalent guards.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE ON SCHEMA public TO "market-data-account";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT, INSERT, UPDATE ON TABLES TO "market-data-account";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT DELETE ON TABLES TO "market-data-account";
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT USAGE, SELECT ON SEQUENCES TO "market-data-account";

\if :drop_all_minervini_tables_on_start
DROP TABLE IF EXISTS backtesting_minervini_equity_daily CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_trades CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_runs CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_setups CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_screen_daily CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_market_daily CASCADE;
DROP TABLE IF EXISTS backtesting_minervini_rs_daily CASCADE;
\endif

-- ---------------------------------------------------------------------------
-- Stage 1a: cross-sectional relative-strength ranking (full eligible universe)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtesting_minervini_rs_daily (
    period_end_date     DATE NOT NULL,
    symbol              TEXT NOT NULL,
    rs_raw              NUMERIC(18,6),
    rs_rating           SMALLINT,
    universe_size       INTEGER,
    PRIMARY KEY (symbol, period_end_date)
);

SELECT create_hypertable(
    'backtesting_minervini_rs_daily',
    'period_end_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_bm_rs_period
    ON backtesting_minervini_rs_daily (period_end_date DESC, rs_rating DESC);

-- ---------------------------------------------------------------------------
-- Stage 1b: daily screen result (trend template + fundamentals per symbol/day)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtesting_minervini_screen_daily (
    period_end_date             DATE NOT NULL,
    symbol                      TEXT NOT NULL,
    ibkr_industry               TEXT,
    ibkr_category               TEXT,
    close                       NUMERIC(15,4),
    rs_rating                   SMALLINT,
    ibkr_industry_rs_rating     SMALLINT,
    ibkr_category_rs_rating     SMALLINT,
    stock_industry_rs_rating    SMALLINT,
    stock_category_rs_rating    SMALLINT,
    ibkr_industry_pass          BOOLEAN,
    ibkr_category_pass          BOOLEAN,
    stock_industry_pass         BOOLEAN,
    stock_category_pass         BOOLEAN,
    group_filter_pass           BOOLEAN NOT NULL,
    crit_price_above_ma150_200  BOOLEAN,
    crit_ma150_above_ma200      BOOLEAN,
    crit_ma200_rising           BOOLEAN,
    crit_ma50_above_ma150_200   BOOLEAN,
    crit_price_above_ma50       BOOLEAN,
    crit_above_52w_low          BOOLEAN,
    crit_near_52w_high          BOOLEAN,
    crit_rs_rating              BOOLEAN,
    trend_template_pass         BOOLEAN NOT NULL,
    eps_pass                    BOOLEAN,
    revenue_pass                BOOLEAN,
    margin_pass                 BOOLEAN,
    fundamentals_pass           BOOLEAN NOT NULL,
    screen_pass                 BOOLEAN NOT NULL,
    eps_yoy                     NUMERIC(18,6),
    revenue_yoy                 NUMERIC(18,6),
    PRIMARY KEY (symbol, period_end_date)
);

SELECT create_hypertable(
    'backtesting_minervini_screen_daily',
    'period_end_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_bm_screen_period_pass
    ON backtesting_minervini_screen_daily (period_end_date DESC, screen_pass);

CREATE INDEX IF NOT EXISTS idx_bm_screen_group_filter
    ON backtesting_minervini_screen_daily (
        period_end_date DESC, ibkr_industry, ibkr_category, group_filter_pass
    );

-- ---------------------------------------------------------------------------
-- Stage 1c: daily market regime (breadth of stocks above their 200d MA)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtesting_minervini_market_daily (
    period_end_date     DATE PRIMARY KEY,
    market_breadth_pct  NUMERIC(8,4),
    market_on           BOOLEAN NOT NULL
);

-- ---------------------------------------------------------------------------
-- Stage 2: detected VCP setups (pre-breakout bases with pivot and stop)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtesting_minervini_setups (
    setup_id            BIGSERIAL PRIMARY KEY,
    symbol              TEXT NOT NULL,
    ibkr_industry       TEXT,
    ibkr_category       TEXT,
    detect_date         DATE NOT NULL,
    pivot               NUMERIC(15,4) NOT NULL,
    last_low            NUMERIC(15,4) NOT NULL,
    stop_level          NUMERIC(15,4) NOT NULL,
    base_start_date     DATE NOT NULL,
    base_days           INTEGER NOT NULL,
    n_contractions      INTEGER NOT NULL,
    contraction_depths  JSONB NOT NULL,
    dryup_ratio         NUMERIC(10,4),
    close               NUMERIC(15,4),
    valid_until         DATE NOT NULL,
    created_ts          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_bm_setups_detect
    ON backtesting_minervini_setups (detect_date, symbol);

-- ---------------------------------------------------------------------------
-- Stage 3: simulation results
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtesting_minervini_runs (
    run_id              BIGSERIAL PRIMARY KEY,
    run_ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
    run_label           TEXT NOT NULL,
    start_date          DATE NOT NULL,
    end_date            DATE NOT NULL,
    params              JSONB NOT NULL,
    initial_equity      NUMERIC(18,2) NOT NULL,
    final_equity        NUMERIC(18,2),
    total_return        NUMERIC(18,6),
    cagr                NUMERIC(18,6),
    max_drawdown        NUMERIC(18,6),
    win_rate            NUMERIC(18,6),
    profit_factor       NUMERIC(18,6),
    avg_r_multiple      NUMERIC(18,6),
    num_positions       INTEGER,
    num_trade_legs      INTEGER,
    avg_exposure        NUMERIC(18,6)
);

CREATE TABLE IF NOT EXISTS backtesting_minervini_trades (
    trade_id            BIGSERIAL PRIMARY KEY,
    run_id              BIGINT NOT NULL REFERENCES backtesting_minervini_runs (run_id) ON DELETE CASCADE,
    position_id         INTEGER NOT NULL,
    setup_id            BIGINT,
    symbol              TEXT NOT NULL,
    ibkr_industry       TEXT,
    ibkr_category       TEXT,
    leg                 TEXT NOT NULL,
    exit_reason         TEXT NOT NULL,
    entry_date          DATE NOT NULL,
    entry_price         NUMERIC(15,4) NOT NULL,
    stop_price          NUMERIC(15,4) NOT NULL,
    pivot               NUMERIC(15,4),
    shares              INTEGER NOT NULL,
    exit_date           DATE NOT NULL,
    exit_price          NUMERIC(15,4) NOT NULL,
    pnl                 NUMERIC(18,2) NOT NULL,
    r_multiple          NUMERIC(18,6),
    holding_days        INTEGER NOT NULL,
    regime_composite    NUMERIC(18,6),
    regime_label        TEXT
);

CREATE INDEX IF NOT EXISTS idx_bm_trades_run
    ON backtesting_minervini_trades (run_id, symbol);

-- Aggregate research curve (fixed sizing base + cumulative PnL over all
-- independent trades) — NOT a cash-constrained account equity.
CREATE TABLE IF NOT EXISTS backtesting_minervini_equity_daily (
    run_id              BIGINT NOT NULL REFERENCES backtesting_minervini_runs (run_id) ON DELETE CASCADE,
    period_end_date     DATE NOT NULL,
    equity              NUMERIC(18,2) NOT NULL,
    open_positions      INTEGER NOT NULL,
    exposure_pct        NUMERIC(18,6) NOT NULL,
    PRIMARY KEY (run_id, period_end_date)
);

-- Ensure the runtime account can use everything created above (existing objects).
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO "market-data-account";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "market-data-account";
