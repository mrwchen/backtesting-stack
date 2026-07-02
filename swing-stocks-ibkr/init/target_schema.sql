\if :drop_all_tables_on_start
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_feature_bucket_stability CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_feature_bucket_strength CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_top_bottom_symbols CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_holding_period_buckets CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_exit_reason_yearly CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_exit_reasons CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_yearly_stability CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_symbol_breadth CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_strategy_edge CASCADE;
DROP TABLE IF EXISTS backtest_swing_stock_diagnostic_runs CASCADE;
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

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_runs (
    diagnostic_run_id              BIGSERIAL PRIMARY KEY,
    source_run_id                  BIGINT NOT NULL REFERENCES backtest_swing_stock_runs(run_id) ON DELETE CASCADE,
    started_at_utc                 TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at_utc                TIMESTAMPTZ,
    status                         TEXT NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'ok', 'error')),
    top_n                          INTEGER NOT NULL,
    min_bucket_trades              INTEGER NOT NULL,
    min_year_trades                INTEGER NOT NULL,
    feature_lookback_days          INTEGER NOT NULL,
    error_text                     TEXT
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_strategy_edge (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    trades                         INTEGER NOT NULL,
    symbols                        INTEGER NOT NULL,
    first_exit                     DATE,
    last_exit                      DATE,
    avg_gross_return_pct           NUMERIC(18,6),
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    p10_net_return_pct             NUMERIC(18,6),
    p90_net_return_pct             NUMERIC(18,6),
    worst_net_return_pct           NUMERIC(18,6),
    best_net_return_pct            NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    avg_net_return_trim_1pct_each_tail NUMERIC(18,6),
    profit_factor_trim_1pct_each_tail NUMERIC(18,6),
    avg_holding_days               NUMERIC(18,6),
    median_holding_days            NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, strategy_name)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_symbol_breadth (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    symbol_strategy_rows           INTEGER NOT NULL,
    ok_rows                        INTEGER NOT NULL,
    insufficient_history_rows      INTEGER NOT NULL,
    error_rows                     INTEGER NOT NULL,
    symbols_with_trades            INTEGER NOT NULL,
    symbols_positive_compounded    INTEGER NOT NULL,
    positive_symbol_share_pct      NUMERIC(18,6),
    p10_symbol_compounded_pct      NUMERIC(18,6),
    p50_symbol_compounded_pct      NUMERIC(18,6),
    p90_symbol_compounded_pct      NUMERIC(18,6),
    best_symbol_compounded_pct     NUMERIC(18,6),
    worst_symbol_compounded_pct    NUMERIC(18,6),
    signals                        INTEGER,
    skipped_signals                INTEGER,
    PRIMARY KEY (diagnostic_run_id, strategy_name)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_yearly_stability (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    exit_year                      INTEGER NOT NULL,
    trades                         INTEGER NOT NULL,
    symbols                        INTEGER NOT NULL,
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, strategy_name, exit_year)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_exit_reasons (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    exit_reason                    TEXT NOT NULL,
    trades                         INTEGER NOT NULL,
    share_pct                      NUMERIC(18,6),
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, strategy_name, exit_reason)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_exit_reason_yearly (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    exit_year                      INTEGER NOT NULL,
    exit_reason                    TEXT NOT NULL,
    trades                         INTEGER NOT NULL,
    symbols                        INTEGER NOT NULL,
    share_pct                      NUMERIC(18,6),
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    avg_holding_days               NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, strategy_name, exit_year, exit_reason)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_holding_period_buckets (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    holding_days_bucket            TEXT NOT NULL,
    trades                         INTEGER NOT NULL,
    symbols                        INTEGER NOT NULL,
    share_pct                      NUMERIC(18,6),
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    avg_holding_days               NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, strategy_name, holding_days_bucket)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_top_bottom_symbols (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    side                           TEXT NOT NULL CHECK (side IN ('best', 'worst')),
    strategy_name                  TEXT NOT NULL,
    rank                           INTEGER NOT NULL,
    symbol                         TEXT NOT NULL,
    exchange                       TEXT NOT NULL,
    trade_count                    INTEGER NOT NULL,
    win_count                      INTEGER NOT NULL,
    loss_count                     INTEGER NOT NULL,
    total_compounded_return_pct    NUMERIC(18,6),
    max_drawdown_pct               NUMERIC(18,6),
    avg_return_pct                 NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, side, strategy_name, rank)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_feature_bucket_strength (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    side                           TEXT NOT NULL CHECK (side IN ('strongest', 'weakest')),
    strategy_name                  TEXT NOT NULL,
    feature_name                   TEXT NOT NULL,
    rank                           INTEGER NOT NULL,
    feature_bucket                 TEXT NOT NULL,
    trades                         INTEGER NOT NULL,
    symbols                        INTEGER NOT NULL,
    years                          INTEGER NOT NULL,
    avg_net_return_pct             NUMERIC(18,6),
    median_net_return_pct          NUMERIC(18,6),
    win_rate_pct                   NUMERIC(18,6),
    profit_factor                  NUMERIC(18,6),
    avg_holding_days               NUMERIC(18,6),
    PRIMARY KEY (diagnostic_run_id, side, strategy_name, feature_name, rank)
);

CREATE TABLE IF NOT EXISTS backtest_swing_stock_diagnostic_feature_bucket_stability (
    diagnostic_run_id              BIGINT NOT NULL REFERENCES backtest_swing_stock_diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
    strategy_name                  TEXT NOT NULL,
    feature_name                   TEXT NOT NULL,
    rank                           INTEGER NOT NULL,
    feature_bucket                 TEXT NOT NULL,
    qualified_years                INTEGER NOT NULL,
    trades                         INTEGER NOT NULL,
    avg_yearly_return_pct          NUMERIC(18,6),
    worst_yearly_return_pct        NUMERIC(18,6),
    avg_yearly_profit_factor       NUMERIC(18,6),
    positive_avg_years             INTEGER NOT NULL,
    profit_factor_above_one_years  INTEGER NOT NULL,
    PRIMARY KEY (diagnostic_run_id, strategy_name, feature_name, rank)
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

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_runs_source
    ON backtest_swing_stock_diagnostic_runs (source_run_id, diagnostic_run_id DESC);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_edge_strategy
    ON backtest_swing_stock_diagnostic_strategy_edge (diagnostic_run_id, strategy_name);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_exit_year_lookup
    ON backtest_swing_stock_diagnostic_exit_reason_yearly
    (diagnostic_run_id, strategy_name, exit_year, exit_reason);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_holding_lookup
    ON backtest_swing_stock_diagnostic_holding_period_buckets
    (diagnostic_run_id, strategy_name, holding_days_bucket);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_strength_lookup
    ON backtest_swing_stock_diagnostic_feature_bucket_strength
    (diagnostic_run_id, strategy_name, feature_name, side, rank);

CREATE INDEX IF NOT EXISTS idx_backtest_swing_stock_diag_stability_lookup
    ON backtest_swing_stock_diagnostic_feature_bucket_stability
    (diagnostic_run_id, strategy_name, feature_name, rank);

ALTER TABLE backtest_swing_stock_runs OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_strategy_results OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_trades OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_equity_daily OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_runs OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_strategy_edge OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_symbol_breadth OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_yearly_stability OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_exit_reasons OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_exit_reason_yearly OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_holding_period_buckets OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_top_bottom_symbols OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_feature_bucket_strength OWNER TO "market-data-account";
ALTER TABLE backtest_swing_stock_diagnostic_feature_bucket_stability OWNER TO "market-data-account";

GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_runs TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_strategy_results TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_trades TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_equity_daily TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_runs TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_strategy_edge TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_symbol_breadth TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_yearly_stability TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_exit_reasons TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_exit_reason_yearly TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_holding_period_buckets TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_top_bottom_symbols TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_feature_bucket_strength TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_swing_stock_diagnostic_feature_bucket_stability TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_swing_stock_runs_run_id_seq TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_swing_stock_diagnostic_runs_diagnostic_run_id_seq TO "market-data-account";
