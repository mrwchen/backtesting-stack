-- Fresh schema for swing trade backtesting.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END;
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE, CREATE ON SCHEMA public TO "market-data-account";

\if :drop_backtest_tables_on_start
DROP TABLE IF EXISTS backtest_monte_carlo CASCADE;
DROP TABLE IF EXISTS backtest_account_curve CASCADE;
DROP TABLE IF EXISTS backtest_decision_events CASCADE;
DROP TABLE IF EXISTS backtest_trades CASCADE;
DROP TABLE IF EXISTS backtest_runs CASCADE;
\endif

-- ── Run metadata ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id               SERIAL        PRIMARY KEY,
    created_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    notes                TEXT,
    run_label            TEXT          NOT NULL,
    model_file           TEXT          NOT NULL,
    account_profile      TEXT          NOT NULL,

    -- Time range
    start_date           DATE          NOT NULL,
    end_date             DATE          NOT NULL,

    -- Margin account parameters
    initial_equity       NUMERIC(15,2) NOT NULL,
    risk_per_trade_pct   NUMERIC(5,2)  NOT NULL,   -- % of equity risked per trade
    max_open_positions   INTEGER       NOT NULL,
    margin_requirement_pct NUMERIC(5,2) NOT NULL,  -- margin needed as % of position size
    maintenance_margin_pct NUMERIC(5,2) NOT NULL,
    min_free_margin_pct  NUMERIC(5,2)  NOT NULL,   -- halt new trades if free margin < X% of equity
    allow_fractional_shares BOOLEAN,
    spread_bps           NUMERIC(6,2),
    slippage_bps         NUMERIC(6,2),
    commission_per_order_usd NUMERIC(8,4),
    commission_per_share_usd NUMERIC(10,6) NOT NULL DEFAULT 0,
    commission_min_per_order_usd NUMERIC(8,4) NOT NULL DEFAULT 0,
    commission_max_pct    NUMERIC(6,3) NOT NULL DEFAULT 0,
    commission_bps       NUMERIC(6,2),
    margin_financing_rate_pct NUMERIC(5,2),
    entry_window_enabled BOOLEAN,
    entry_window_tz      TEXT,
    entry_window_start   TEXT,
    entry_window_end     TEXT,

    -- Signal parameters (snapshot of what was used)
    long_max_score       NUMERIC(5,2)  NOT NULL,
    short_min_score      NUMERIC(5,2)  NOT NULL,
    long_min_fundamental NUMERIC(5,2)  NOT NULL,
    short_max_fundamental NUMERIC(5,2) NOT NULL,
    min_market_cap_m     NUMERIC(10,2),
    long_min_pullback    NUMERIC(5,2)  NOT NULL,
    long_max_pullback    NUMERIC(5,2),
    long_ideal_pullback  NUMERIC(5,2),
    long_max_rsi         NUMERIC(5,2)  NOT NULL,
    short_min_bounce     NUMERIC(5,2)  NOT NULL,
    short_max_bounce     NUMERIC(5,2),
    short_ideal_bounce   NUMERIC(5,2),
    short_min_rsi        NUMERIC(5,2),
    short_max_rsi        NUMERIC(5,2)  NOT NULL,
    long_sl_buffer       NUMERIC(8,4),
    short_sl_buffer      NUMERIC(8,4),
    long_tp1_pct         NUMERIC(6,4)  NOT NULL,
    long_tp2_pct         NUMERIC(6,4)  NOT NULL,
    short_tp1_pct        NUMERIC(6,4)  NOT NULL,
    short_tp2_pct        NUMERIC(6,4)  NOT NULL,
    long_max_hold_days   NUMERIC(6,2),
    short_max_hold_days  NUMERIC(6,2),
    tp1_close_ratio      NUMERIC(4,3),

    -- Results summary (filled after run completes)
    final_equity         NUMERIC(15,2),
    total_trades         INTEGER,
    winning_trades       INTEGER,
    losing_trades        INTEGER,
    breakeven_trades     INTEGER,
    expired_trades       INTEGER,
    win_rate_pct         NUMERIC(5,2),
    total_return_pct     NUMERIC(12,2),
    max_drawdown_pct     NUMERIC(8,2),
    avg_return_pct       NUMERIC(12,4),
    avg_win_pct          NUMERIC(12,4),
    avg_loss_pct         NUMERIC(12,4),
    profit_factor        NUMERIC(12,4)
);

-- ── Per-day decision trace ───────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_decision_events (
    id                   BIGSERIAL     PRIMARY KEY,
    run_id               INTEGER       NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    created_at           TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    signal_date          DATE          NOT NULL,
    as_of_ts             TIMESTAMPTZ,
    symbol               TEXT,
    direction            TEXT,

    -- Decision taxonomy
    decision_stage       TEXT          NOT NULL, -- regime_filter | candidate_filter | bar_load | signal_eval | portfolio_filter | order_open
    decision             TEXT          NOT NULL, -- skipped_day | no_candidates | rejected | signal | blocked | opened
    reason_code          TEXT          NOT NULL,
    reason_text          TEXT,
    signal_passed        BOOLEAN       NOT NULL DEFAULT FALSE,
    opened               BOOLEAN       NOT NULL DEFAULT FALSE,
    candidate_rank       INTEGER,
    signal_rank          INTEGER,

    -- Market and fundamental context
    world_regime_label   TEXT,
    world_regime_score   NUMERIC(5,2),
    valuation_label      TEXT,
    sector               TEXT,
    industry             TEXT,
    fundamental_score    NUMERIC(8,4),
    mispricing_score     NUMERIC(8,4),
    market_cap_m         NUMERIC(18,2),

    -- Bar and signal context
    bar_count            INTEGER,
    min_bars             INTEGER,
    entry_ts             TIMESTAMPTZ,
    entry_price          NUMERIC(15,4),
    stop_loss            NUMERIC(15,4),
    take_profit_1        NUMERIC(15,4),
    take_profit_2        NUMERIC(15,4),
    pullback_pct         NUMERIC(6,2),
    rsi_1h               NUMERIC(5,2),
    volume_ratio         NUMERIC(6,3),
    entry_score          NUMERIC(8,4),
    combined_score       NUMERIC(8,4),

    -- Portfolio context at decision time
    open_positions       INTEGER,
    max_open_positions   INTEGER,
    account_equity       NUMERIC(15,2),
    used_margin          NUMERIC(15,2),
    free_margin          NUMERIC(15,2),
    required_margin      NUMERIC(15,2),
    free_margin_after    NUMERIC(15,2),
    min_free_margin_pct  NUMERIC(5,2),
    position_size_usd    NUMERIC(15,2),
    shares               NUMERIC(15,6)
);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_run_day
    ON backtest_decision_events (run_id, signal_date, decision_stage, decision);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_symbol_day
    ON backtest_decision_events (run_id, symbol, signal_date);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_reason
    ON backtest_decision_events (run_id, reason_code, signal_date);

-- ── Individual trades ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_trades (
    id                   SERIAL        PRIMARY KEY,
    run_id               INTEGER       NOT NULL REFERENCES backtest_runs(run_id),
    signal_date          DATE          NOT NULL,
    symbol               TEXT          NOT NULL,
    direction            TEXT          NOT NULL,   -- LONG | SHORT

    -- World regime at signal time
    world_regime_label   TEXT,
    world_regime_score   NUMERIC(5,2),

    -- Fundamental label at signal time
    valuation_label      TEXT,

    -- Signal scores
    fundamental_score    NUMERIC(5,2),
    entry_score          NUMERIC(5,2),
    combined_score       NUMERIC(5,2),

    -- Entry
    entry_price          NUMERIC(15,4) NOT NULL,
    stop_loss            NUMERIC(15,4) NOT NULL,
    take_profit_1        NUMERIC(15,4) NOT NULL,
    take_profit_2        NUMERIC(15,4) NOT NULL,
    pullback_pct         NUMERIC(6,2),
    rsi_1h               NUMERIC(5,2),
    volume_ratio         NUMERIC(6,3),
    entry_reason         TEXT,

    -- Position sizing
    position_size_usd    NUMERIC(15,2),
    shares               NUMERIC(15,6),
    margin_used          NUMERIC(15,2),
    equity_before        NUMERIC(15,2),

    -- Outcome
    outcome_status       TEXT,          -- HIT_TP2 | HIT_TP1_THEN_BE | HIT_SL | MAX_HOLD | MAX_HOLD_TP1 | FORCE_CLOSED
    outcome_price        NUMERIC(15,4),
    outcome_date         DATE,
    outcome_bars         INTEGER,       -- 1h bars from entry to close
    tp1_hit              BOOLEAN        NOT NULL DEFAULT FALSE,
    return_pct           NUMERIC(8,4),
    pnl_usd              NUMERIC(15,2),
    equity_after         NUMERIC(15,2),
    entry_ts             TIMESTAMPTZ,
    tp1_exit_ts          TIMESTAMPTZ,
    exit_ts              TIMESTAMPTZ,

    UNIQUE (run_id, signal_date, symbol)
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id
    ON backtest_trades (run_id, signal_date);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol
    ON backtest_trades (symbol, signal_date);

-- ── Account curve snapshots ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_account_curve (
    run_id               INTEGER       NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    ts                   TIMESTAMPTZ   NOT NULL,
    trade_date           DATE          NOT NULL,
    seq_in_run           INTEGER       NOT NULL,
    balance_usd          NUMERIC(15,2) NOT NULL,
    open_pnl_usd         NUMERIC(15,2) NOT NULL,
    equity_usd           NUMERIC(15,2) NOT NULL,
    used_margin_usd      NUMERIC(15,2) NOT NULL,
    free_margin_usd      NUMERIC(15,2) NOT NULL,
    open_positions       INTEGER       NOT NULL,
    realized_pnl_usd     NUMERIC(15,2) NOT NULL,
    closed_trades        INTEGER       NOT NULL,
    PRIMARY KEY (run_id, ts, seq_in_run)
);

SELECT create_hypertable(
    'backtest_account_curve',
    'ts',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_backtest_account_curve_run_ts
    ON backtest_account_curve (run_id, ts DESC);

-- ── Dashboard lookup indexes ─────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_backtest_runs_model_created
    ON backtest_runs (model_file, created_at DESC, run_id DESC);

-- ── Monte Carlo results ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_monte_carlo (
    run_id                 INTEGER       PRIMARY KEY REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    n_simulations          INTEGER       NOT NULL,
    -- Final equity percentiles
    final_equity_p05       NUMERIC(14,2),
    final_equity_p25       NUMERIC(14,2),
    final_equity_p50       NUMERIC(14,2),
    final_equity_p75       NUMERIC(14,2),
    final_equity_p95       NUMERIC(14,2),
    -- Max drawdown percentiles (negative %)
    max_drawdown_p05       NUMERIC(10,4),  -- worst (most negative)
    max_drawdown_p25       NUMERIC(10,4),
    max_drawdown_p50       NUMERIC(10,4),
    max_drawdown_p75       NUMERIC(10,4),
    max_drawdown_p95       NUMERIC(10,4),  -- mildest
    -- Total return percentiles (%)
    total_return_p05       NUMERIC(12,4),
    total_return_p25       NUMERIC(12,4),
    total_return_p50       NUMERIC(12,4),
    total_return_p75       NUMERIC(12,4),
    total_return_p95       NUMERIC(12,4),
    -- Risk summary
    prob_of_ruin_pct       NUMERIC(6,2),   -- % of sims where final_equity < 50% of initial
    prob_profitable_pct    NUMERIC(6,2),   -- % of sims with positive total return
    -- Extremes
    worst_final_equity     NUMERIC(14,2),
    worst_max_drawdown_pct NUMERIC(10,4),
    best_final_equity      NUMERIC(14,2),
    created_at             TIMESTAMPTZ    NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_runs  TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_decision_events TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_trades TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_account_curve TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_monte_carlo TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_runs_run_id_seq   TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_decision_events_id_seq TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_trades_id_seq     TO "market-data-account";

