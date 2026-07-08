-- Idempotent schema for the regime-gated trend portfolio backtester (stocks).
-- The table prefix comes from the psql variable :table_prefix (compose env
-- TABLE_PREFIX, default backtest_wei_stocks_).
-- Safe to re-run: statements use IF NOT EXISTS or equivalent guards.

\set runs_table   :table_prefix runs
\set trades_table :table_prefix trades
\set equity_table :table_prefix equity_daily
\set equity_index :table_prefix equity_run_day_idx

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

\if :drop_all_wei_stocks_tables_on_start
DROP TABLE IF EXISTS :equity_table CASCADE;
DROP TABLE IF EXISTS :trades_table CASCADE;
DROP TABLE IF EXISTS :runs_table CASCADE;
\endif

-- ---------------------------------------------------------------------------
-- One row per backtest run: parameters and summary metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS :runs_table (
    run_id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    simulation_mode         TEXT NOT NULL CHECK (simulation_mode IN ('portfolio', 'independent')),
    run_label               TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    start_date              DATE NOT NULL,
    end_date                DATE NOT NULL,
    universe_size           INTEGER NOT NULL,
    top_n_per_category      INTEGER NOT NULL,
    min_market_cap_usd      BIGINT NOT NULL,
    ema_fast                INTEGER NOT NULL,
    ema_slow                INTEGER NOT NULL,
    stress_enter            NUMERIC(5,1) NOT NULL,
    stress_exit             NUMERIC(5,1) NOT NULL,
    cat_mom_window          INTEGER NOT NULL,
    cat_mom_deep_threshold  NUMERIC(6,4) NOT NULL,
    weight_deep_pct         NUMERIC(5,2) NOT NULL,
    weight_mild_pct         NUMERIC(5,2) NOT NULL,
    weight_pos_pct          NUMERIC(5,2) NOT NULL,
    max_positions           INTEGER NOT NULL,
    max_per_category        INTEGER NOT NULL,
    entry_confirm_days      INTEGER NOT NULL DEFAULT 0,
    trim_above_pct          NUMERIC(5,2) NOT NULL DEFAULT 0,
    trim_target_pct         NUMERIC(5,2) NOT NULL DEFAULT 0,
    sl_pct                  NUMERIC(5,2) NOT NULL DEFAULT 0,
    time_stop_days          INTEGER NOT NULL DEFAULT 0,
    time_stop_min_ret_pct   NUMERIC(6,2) NOT NULL DEFAULT 0,
    cost_bps_per_side       NUMERIC(6,2) NOT NULL,
    total_return_pct        NUMERIC(10,2),
    bh_return_pct           NUMERIC(10,2),
    max_drawdown_pct        NUMERIC(10,2),
    bh_max_drawdown_pct     NUMERIC(10,2),
    cagr_pct                NUMERIC(10,2),
    n_trades                INTEGER,
    n_winning_trades        INTEGER,
    n_open_trades           INTEGER,
    win_rate_pct            NUMERIC(5,1),
    avg_trade_return_pct    NUMERIC(10,2),
    median_trade_return_pct NUMERIC(10,2),
    avg_holding_days        NUMERIC(10,1),
    avg_gross_exposure_pct  NUMERIC(5,1)
);

-- Columns added after the first release (no-ops on fresh tables).
ALTER TABLE :runs_table
    ADD COLUMN IF NOT EXISTS entry_confirm_days    INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS trim_above_pct        NUMERIC(5,2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS trim_target_pct       NUMERIC(5,2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS sl_pct                NUMERIC(5,2) NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS time_stop_days        INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS time_stop_min_ret_pct NUMERIC(6,2) NOT NULL DEFAULT 0;

-- ---------------------------------------------------------------------------
-- One row per completed (or still open) long trade of a run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS :trades_table (
    run_id                  BIGINT NOT NULL REFERENCES :runs_table (run_id) ON DELETE CASCADE,
    trade_no                INTEGER NOT NULL,
    symbol                  TEXT NOT NULL,
    ibkr_category           TEXT NOT NULL,
    entry_date              DATE NOT NULL,
    exit_date               DATE,
    entry_price             NUMERIC(18,6) NOT NULL,
    exit_price              NUMERIC(18,6),
    gross_return_pct        NUMERIC(10,2),
    holding_days            INTEGER,
    target_weight_pct       NUMERIC(5,2) NOT NULL,
    effective_weight_pct    NUMERIC(5,2) NOT NULL,
    sizing_tier             TEXT NOT NULL,       -- deep | mild | pos
    cat_mom_at_entry_pct    NUMERIC(8,2),        -- category momentum at entry, NULL if unknown
    is_open                 BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, trade_no)
);

-- ---------------------------------------------------------------------------
-- Daily portfolio state of a run: equity curves and exposure (for Grafana).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS :equity_table (
    day                     DATE NOT NULL,
    run_id                  BIGINT NOT NULL,
    equity                  NUMERIC(18,8) NOT NULL,
    bh_equity               NUMERIC(18,8) NOT NULL,
    n_positions             INTEGER NOT NULL,
    gross_exposure_pct      NUMERIC(5,1) NOT NULL,
    composite_score         NUMERIC(5,1),
    stress_on               BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, day)
);

SELECT create_hypertable(
    :'equity_table',
    'day',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS :equity_index
    ON :equity_table (run_id, day DESC);

GRANT SELECT, INSERT, UPDATE, DELETE
    ON :runs_table, :trades_table, :equity_table
    TO "market-data-account";
