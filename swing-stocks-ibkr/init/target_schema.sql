\if :drop_all_tables_on_start
DROP TABLE IF EXISTS backtest_swing_stock_equity_daily CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_trades CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_strategy_results CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_runs CASCADE;
\endif

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE ON SCHEMA public TO "market-data-account";

CREATE TABLE IF NOT EXISTS backtest_swing_stock_runs (
    run_id                         BIGSERIAL PRIMARY KEY,
    started_at_utc                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at_utc                TIMESTAMPTZ,
    status                         TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'ok', 'error')),
    strategy_count                 INTEGER NOT NULL,
    symbol_count                   INTEGER NOT NULL,
    processed_symbol_count         INTEGER NOT NULL DEFAULT 0,
    failed_symbol_count            INTEGER NOT NULL DEFAULT 0,
    start_date                     DATE NOT NULL,
    end_date                       DATE NOT NULL,
    universe_mode                  TEXT NOT NULL,
    max_symbols                    INTEGER NOT NULL,
    process_parallelism            INTEGER NOT NULL,
    min_price                      NUMERIC(18,6) NOT NULL,
    min_market_cap                 NUMERIC(20,2) NOT NULL,
    min_average_daily_volume       NUMERIC(20,2) NOT NULL,
    commission_bps                 NUMERIC(12,6) NOT NULL,
    slippage_bps                   NUMERIC(12,6) NOT NULL,
    write_equity_daily             BOOLEAN NOT NULL,
    source_core_table              TEXT NOT NULL,
    source_market_table            TEXT NOT NULL,
    source_fundamental_table       TEXT NOT NULL,
    source_earnings_table          TEXT NOT NULL,
    error_text                     TEXT
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_strategy_results (
    run_id                         BIGINT NOT NULL REFERENCES backtest_swing_stock_runs(run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    strategy_version               TEXT NOT NULL,
    symbol                         TEXT NOT NULL,
    exchange                       TEXT NOT NULL,
    cik                            BIGINT NOT NULL,
    status                         TEXT NOT NULL
        CHECK (status IN ('ok', 'no_data', 'insufficient_history', 'error')),
    first_trade_date               DATE,
    last_trade_date                DATE,
    trade_count                    INTEGER NOT NULL,
    win_count                      INTEGER NOT NULL,
    loss_count                     INTEGER NOT NULL,
    flat_count                     INTEGER NOT NULL,
    avg_return_pct                 NUMERIC(18,6),
    median_return_pct              NUMERIC(18,6),
    best_return_pct                NUMERIC(18,6),
    worst_return_pct               NUMERIC(18,6),
    total_compounded_return_pct    NUMERIC(18,6),
    max_drawdown_pct               NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    expectancy_pct                 NUMERIC(18,6),
    avg_holding_days               NUMERIC(18,6),
    exposure_days                  INTEGER NOT NULL,
    signal_count                   INTEGER NOT NULL,
    skipped_signal_count           INTEGER NOT NULL,
    error_text                     TEXT,
    PRIMARY KEY (run_id, strategy_name, symbol, exchange, cik)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_trades (
    run_id                         BIGINT NOT NULL REFERENCES backtest_swing_stock_runs(run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    strategy_version               TEXT NOT NULL,
    symbol                         TEXT NOT NULL,
    exchange                       TEXT NOT NULL,
    cik                            BIGINT NOT NULL,
    trade_number                   INTEGER NOT NULL,
    signal_date                    DATE NOT NULL,
    entry_date                     DATE NOT NULL,
    exit_date                      DATE NOT NULL,
    entry_price                    NUMERIC(18,6) NOT NULL,
    exit_price                     NUMERIC(18,6) NOT NULL,
    stop_price                     NUMERIC(18,6),
    gross_return_pct               NUMERIC(18,6) NOT NULL,
    net_return_pct                 NUMERIC(18,6) NOT NULL,
    holding_days                   INTEGER NOT NULL,
    exit_reason                    TEXT NOT NULL,
    signal_score                   NUMERIC(18,6),
    quality_score                  NUMERIC(18,6),
    momentum_score                 NUMERIC(18,6),
    entry_condition                TEXT NOT NULL,
    fundamental_asof_date          DATE,
    earnings_event_date            DATE,
    earnings_known_asof_ts         TIMESTAMPTZ,
    PRIMARY KEY (run_id, strategy_name, symbol, exchange, cik, trade_number)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_equity_daily (
    run_id                         BIGINT NOT NULL REFERENCES backtest_swing_stock_runs(run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    strategy_version               TEXT NOT NULL,
    symbol                         TEXT NOT NULL,
    exchange                       TEXT NOT NULL,
    cik                            BIGINT NOT NULL,
    day                            DATE NOT NULL,
    equity                         NUMERIC(18,8) NOT NULL,
    drawdown_pct                   NUMERIC(18,6) NOT NULL,
    in_position                    BOOLEAN NOT NULL,
    PRIMARY KEY (run_id, strategy_name, symbol, exchange, cik, day)
);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_results_run_strategy
    ON backtest_swing_stock_strategy_results (run_id, strategy_name, status);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_results_symbol
    ON backtest_swing_stock_strategy_results (symbol, strategy_name, trade_count DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_trades_run_strategy
    ON backtest_swing_stock_trades (run_id, strategy_name, entry_date, exit_date);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_trades_symbol
    ON backtest_swing_stock_trades (symbol, strategy_name, entry_date);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_equity_run_day
    ON backtest_swing_stock_equity_daily (run_id, day, strategy_name);

ALTER TABLE backtest_swing_stock_runs OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_strategy_results OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_trades OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_equity_daily OWNER TO "market-data-account";

GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_runs TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_strategy_results TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_trades TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_equity_daily TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_swing_stock_runs_run_id_seq TO "market-data-account";
