-- Idempotent schema for the regime-gated EMA index backtester.
-- All tables are prefixed backtest_wei_.
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

\if :drop_all_wei_tables_on_start
DROP TABLE IF EXISTS backtest_wei_equity_daily CASCADE;
DROP TABLE IF EXISTS backtest_wei_trades CASCADE;
DROP TABLE IF EXISTS backtest_wei_runs CASCADE;
\endif

-- ---------------------------------------------------------------------------
-- One row per backtest run: parameters and summary metrics.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_wei_runs (
    run_id                  BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_label               TEXT NOT NULL,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    symbol                  TEXT NOT NULL,
    start_date              DATE NOT NULL,
    end_date                DATE NOT NULL,
    ema_fast                INTEGER NOT NULL,
    ema_slow                INTEGER NOT NULL,
    stress_enter            NUMERIC(5,1) NOT NULL,
    stress_exit             NUMERIC(5,1) NOT NULL,
    cost_bps_per_side       NUMERIC(6,2) NOT NULL,
    total_return_pct        NUMERIC(10,2),
    bh_return_pct           NUMERIC(10,2),
    max_drawdown_pct        NUMERIC(10,2),
    bh_max_drawdown_pct     NUMERIC(10,2),
    cagr_pct                NUMERIC(10,2),
    n_trades                INTEGER,
    n_winning_trades        INTEGER,
    days_invested_pct       NUMERIC(5,1)
);

-- ---------------------------------------------------------------------------
-- One row per completed (or still open) long trade of a run.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_wei_trades (
    run_id                  BIGINT NOT NULL REFERENCES backtest_wei_runs (run_id) ON DELETE CASCADE,
    trade_no                INTEGER NOT NULL,
    entry_date              DATE NOT NULL,
    exit_date               DATE,
    entry_price             NUMERIC(18,6) NOT NULL,
    exit_price              NUMERIC(18,6),
    gross_return_pct        NUMERIC(10,2),
    holding_days            INTEGER,
    is_open                 BOOLEAN NOT NULL DEFAULT FALSE,
    PRIMARY KEY (run_id, trade_no)
);

-- ---------------------------------------------------------------------------
-- Daily state of a run: signals, position and equity curves (for Grafana).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_wei_equity_daily (
    day                     DATE NOT NULL,
    run_id                  BIGINT NOT NULL,
    close                   NUMERIC(18,6) NOT NULL,
    ema_fast_value          NUMERIC(18,6) NOT NULL,
    ema_slow_value          NUMERIC(18,6) NOT NULL,
    composite_score         NUMERIC(5,1),
    stress_on               BOOLEAN NOT NULL,
    position                SMALLINT NOT NULL,
    equity                  NUMERIC(18,8) NOT NULL,
    bh_equity               NUMERIC(18,8) NOT NULL,
    PRIMARY KEY (run_id, day)
);

SELECT create_hypertable(
    'backtest_wei_equity_daily',
    'day',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_bw_equity_run_day
    ON backtest_wei_equity_daily (run_id, day DESC);

GRANT SELECT, INSERT, UPDATE, DELETE
    ON backtest_wei_runs, backtest_wei_trades, backtest_wei_equity_daily
    TO "market-data-account";
