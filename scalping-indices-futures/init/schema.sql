-- Schema for the statistical scalping backtester (scalping-indices-futures).
-- Tables are prefixed `backtest2_scalp_` so they never collide with other stacks.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END;
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE, CREATE ON SCHEMA public TO "market-data-account";

\if :drop_scalp_tables_on_start
DROP TABLE IF EXISTS backtest2_scalp_monte_carlo CASCADE;
DROP TABLE IF EXISTS backtest2_scalp_trades CASCADE;
DROP TABLE IF EXISTS backtest2_scalp_runs CASCADE;
\endif

-- ── Run metadata + config snapshot + result summary ─────────────────────────────

CREATE TABLE IF NOT EXISTS backtest2_scalp_runs (
    run_id                 SERIAL        PRIMARY KEY,
    created_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    run_label              TEXT          NOT NULL,
    notes                  TEXT,

    -- data source
    symbol                 TEXT          NOT NULL,
    bar_size               TEXT          NOT NULL,
    data_start_ts          TIMESTAMPTZ,
    data_end_ts            TIMESTAMPTZ,
    bars_total             INTEGER,
    bars_simulated         INTEGER,

    -- layer switches
    price_model            TEXT          NOT NULL,   -- kalman | state_space
    vol_model              TEXT          NOT NULL,   -- garch | egarch
    decision_model         TEXT          NOT NULL,   -- bayes | logistic

    -- feature windows
    rsi_period             INTEGER,
    roll_vol_bars          INTEGER,
    momentum_bars          INTEGER,
    atr_bars               INTEGER,

    -- regime (HMM) detail
    regime_states          INTEGER       NOT NULL,
    regime_block_high_vol_state BOOLEAN,
    hmm_n_iter             INTEGER,
    hmm_covariance_type    TEXT,

    -- volatility model detail
    garch_p                INTEGER,
    garch_q                INTEGER,
    garch_dist             TEXT,

    -- decision model detail
    logistic_c             NUMERIC(12,4),
    min_train_rows         INTEGER,

    -- walk-forward / decision params
    warmup_bars            INTEGER,
    train_window_bars      INTEGER,
    refit_every_bars       INTEGER,
    prob_threshold         NUMERIC(6,4),

    -- trade-level params
    stop_mode              TEXT,                      -- vol | atr
    tp_mode                TEXT,                      -- fixed | trailing
    stop_vol_mult          NUMERIC(8,4),
    tp_vol_mult            NUMERIC(8,4),
    stop_atr_mult          NUMERIC(8,4),
    tp_atr_mult            NUMERIC(8,4),
    trailing_activation_mult NUMERIC(8,4),
    trailing_distance_mult NUMERIC(8,4),
    min_stop_pct           NUMERIC(8,4),
    max_stop_pct           NUMERIC(8,4),
    max_hold_bars          INTEGER,
    allow_short            BOOLEAN,
    reentry_cooldown_bars  INTEGER,
    intrabar_fill_priority TEXT,                      -- stop | tp
    session_flat_time      TEXT,
    session_tz             TEXT,

    -- account: PS_ACC
    account_profile        TEXT          NOT NULL,
    initial_equity         NUMERIC(15,2) NOT NULL,
    account_currency       TEXT,
    margin_requirement_pct NUMERIC(6,3),
    risk_per_trade_pct     NUMERIC(6,3),
    max_margin_pct         NUMERIC(6,3),
    contract_multiplier    NUMERIC(12,4),
    lot_size               NUMERIC(12,4),
    eurusd_rate            NUMERIC(10,6),

    -- costs
    spread_points          NUMERIC(10,4),
    slippage_points        NUMERIC(10,4),
    spread_bps             NUMERIC(8,4),
    slippage_bps           NUMERIC(8,4),
    commission_per_unit    NUMERIC(12,6),
    mc_extra_slippage_points NUMERIC(10,4),
    mc_extra_slippage_bps  NUMERIC(8,4),
    mc_random_seed         BIGINT,

    -- result summary
    run_duration_seconds   NUMERIC(12,3),
    final_equity           NUMERIC(15,2),
    total_return_pct       NUMERIC(14,4),
    total_trades           INTEGER,
    winning_trades         INTEGER,
    losing_trades          INTEGER,
    breakeven_trades       INTEGER,
    win_rate_pct           NUMERIC(6,2),
    profit_factor          NUMERIC(14,4),
    max_drawdown_pct       NUMERIC(10,4),
    avg_win_pct            NUMERIC(12,4),
    avg_loss_pct           NUMERIC(12,4),
    ruined                 BOOLEAN
);

ALTER TABLE backtest2_scalp_runs ADD COLUMN IF NOT EXISTS spread_points NUMERIC(10,4);
ALTER TABLE backtest2_scalp_runs ADD COLUMN IF NOT EXISTS slippage_points NUMERIC(10,4);
ALTER TABLE backtest2_scalp_runs ADD COLUMN IF NOT EXISTS lot_size NUMERIC(12,4);
ALTER TABLE backtest2_scalp_runs ADD COLUMN IF NOT EXISTS mc_extra_slippage_points NUMERIC(10,4);
ALTER TABLE backtest2_scalp_runs ADD COLUMN IF NOT EXISTS mc_extra_slippage_bps NUMERIC(8,4);

-- ── Individual trades ───────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest2_scalp_trades (
    id                     BIGSERIAL     PRIMARY KEY,
    run_id                 INTEGER       NOT NULL REFERENCES backtest2_scalp_runs(run_id) ON DELETE CASCADE,

    intent_ts              TIMESTAMPTZ   NOT NULL,
    entry_ts               TIMESTAMPTZ   NOT NULL,
    entry_price            NUMERIC(15,4) NOT NULL,
    direction              TEXT          NOT NULL,   -- LONG | SHORT
    units                  NUMERIC(18,8) NOT NULL,
    notional_eur           NUMERIC(18,2),
    margin_used_eur        NUMERIC(18,2),

    regime_state           INTEGER,
    prob_up                NUMERIC(8,6),
    sigma_pts              NUMERIC(15,6),
    stop_price             NUMERIC(15,4),
    take_profit_price      NUMERIC(15,4),

    outcome_status         TEXT          NOT NULL,   -- HIT_TP | HIT_SL | MAX_HOLD | SESSION_FLAT
    exit_ts                TIMESTAMPTZ   NOT NULL,
    exit_price             NUMERIC(15,4) NOT NULL,
    bars_held              INTEGER       NOT NULL,

    return_pct             NUMERIC(14,4) NOT NULL,
    pnl_eur                NUMERIC(15,2) NOT NULL,
    costs_eur              NUMERIC(15,2),
    equity_before          NUMERIC(15,2),
    equity_after           NUMERIC(15,2),

    created_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS backtest2_scalp_trades_run_idx ON backtest2_scalp_trades(run_id);

-- ── Monte-Carlo risk (base / slippage stress / sequence) ────────────────────────

CREATE TABLE IF NOT EXISTS backtest2_scalp_monte_carlo (
    run_id                 INTEGER       PRIMARY KEY REFERENCES backtest2_scalp_runs(run_id) ON DELETE CASCADE,
    n_simulations          INTEGER       NOT NULL,

    -- base (permuted trade order)
    base_final_equity_p05  NUMERIC(15,2), base_final_equity_p25 NUMERIC(15,2),
    base_final_equity_p50  NUMERIC(15,2), base_final_equity_p75 NUMERIC(15,2),
    base_final_equity_p95  NUMERIC(15,2),
    base_max_drawdown_p05  NUMERIC(10,4), base_max_drawdown_p25 NUMERIC(10,4),
    base_max_drawdown_p50  NUMERIC(10,4), base_max_drawdown_p75 NUMERIC(10,4),
    base_max_drawdown_p95  NUMERIC(10,4),
    base_total_return_p05  NUMERIC(14,4), base_total_return_p25 NUMERIC(14,4),
    base_total_return_p50  NUMERIC(14,4), base_total_return_p75 NUMERIC(14,4),
    base_total_return_p95  NUMERIC(14,4),
    base_prob_of_ruin_pct  NUMERIC(6,2),  base_prob_profitable_pct NUMERIC(6,2),
    base_worst_final_equity NUMERIC(15,2), base_best_final_equity NUMERIC(15,2),
    base_worst_max_drawdown_pct NUMERIC(10,4),

    -- slippage stress (extra slippage deducted, permuted)
    slip_final_equity_p05  NUMERIC(15,2), slip_final_equity_p25 NUMERIC(15,2),
    slip_final_equity_p50  NUMERIC(15,2), slip_final_equity_p75 NUMERIC(15,2),
    slip_final_equity_p95  NUMERIC(15,2),
    slip_max_drawdown_p05  NUMERIC(10,4), slip_max_drawdown_p25 NUMERIC(10,4),
    slip_max_drawdown_p50  NUMERIC(10,4), slip_max_drawdown_p75 NUMERIC(10,4),
    slip_max_drawdown_p95  NUMERIC(10,4),
    slip_total_return_p05  NUMERIC(14,4), slip_total_return_p25 NUMERIC(14,4),
    slip_total_return_p50  NUMERIC(14,4), slip_total_return_p75 NUMERIC(14,4),
    slip_total_return_p95  NUMERIC(14,4),
    slip_prob_of_ruin_pct  NUMERIC(6,2),  slip_prob_profitable_pct NUMERIC(6,2),
    slip_worst_final_equity NUMERIC(15,2), slip_best_final_equity NUMERIC(15,2),
    slip_worst_max_drawdown_pct NUMERIC(10,4),

    -- sequence risk (block bootstrap, preserves streaks)
    seq_final_equity_p05   NUMERIC(15,2), seq_final_equity_p25 NUMERIC(15,2),
    seq_final_equity_p50   NUMERIC(15,2), seq_final_equity_p75 NUMERIC(15,2),
    seq_final_equity_p95   NUMERIC(15,2),
    seq_max_drawdown_p05   NUMERIC(10,4), seq_max_drawdown_p25 NUMERIC(10,4),
    seq_max_drawdown_p50   NUMERIC(10,4), seq_max_drawdown_p75 NUMERIC(10,4),
    seq_max_drawdown_p95   NUMERIC(10,4),
    seq_total_return_p05   NUMERIC(14,4), seq_total_return_p25 NUMERIC(14,4),
    seq_total_return_p50   NUMERIC(14,4), seq_total_return_p75 NUMERIC(14,4),
    seq_total_return_p95   NUMERIC(14,4),
    seq_prob_of_ruin_pct   NUMERIC(6,2),  seq_prob_profitable_pct NUMERIC(6,2),
    seq_worst_final_equity NUMERIC(15,2), seq_best_final_equity NUMERIC(15,2),
    seq_worst_max_drawdown_pct NUMERIC(10,4),

    created_at             TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON
    backtest2_scalp_runs, backtest2_scalp_trades, backtest2_scalp_monte_carlo
    TO "market-data-account";
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "market-data-account";
