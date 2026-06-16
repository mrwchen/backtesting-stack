-- Schema for the NAS100 hit-frequency median tick backtester.
-- Tables are prefixed `backtest2_nas100_hfmed_` to avoid collisions.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END;
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE, CREATE ON SCHEMA public TO "market-data-account";

\if :drop_hfmed_tables_on_start
DROP TABLE IF EXISTS backtest2_nas100_hfmed_monte_carlo CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_trades CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_runs CASCADE;
\endif

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_runs (
    run_id                    SERIAL        PRIMARY KEY,
    created_at                TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    run_label                 TEXT          NOT NULL,
    notes                     TEXT,

    source_table              TEXT          NOT NULL,
    symbol                    TEXT          NOT NULL,
    start_ts_utc              TIMESTAMPTZ,
    end_ts_utc                TIMESTAMPTZ,
    data_start_ts             TIMESTAMPTZ,
    data_end_ts               TIMESTAMPTZ,
    ticks_loaded              BIGINT,
    ticks_simulated           BIGINT,
    bars_built                INTEGER,

    bar_seconds               INTEGER       NOT NULL,
    lookback_bars             INTEGER       NOT NULL,
    min_lookback_bars         INTEGER       NOT NULL,
    price_step                NUMERIC(12,6) NOT NULL,
    median_quantile           NUMERIC(8,6)  NOT NULL,
    stop_points               NUMERIC(12,4) NOT NULL,
    take_profit_points        NUMERIC(12,4) NOT NULL,

    account_profile           TEXT          NOT NULL,
    initial_equity            NUMERIC(15,2) NOT NULL,
    account_currency          TEXT          NOT NULL,
    margin_requirement_pct    NUMERIC(8,4),
    risk_per_trade_pct        NUMERIC(8,4),
    max_margin_pct            NUMERIC(8,4),
    contract_multiplier       NUMERIC(12,4),
    lot_size                  NUMERIC(12,4),
    eurusd_rate               NUMERIC(12,6),

    spread_points             NUMERIC(12,4),
    slippage_points           NUMERIC(12,4),
    commission_per_unit       NUMERIC(12,6),

    monte_carlo_enabled       BOOLEAN,
    monte_carlo_simulations   INTEGER,
    mc_extra_slippage_points  NUMERIC(12,4),
    mc_block_size             INTEGER,
    mc_ruin_drawdown_pct      NUMERIC(8,4),
    mc_random_seed            BIGINT,

    signals_total             BIGINT,
    long_signals              BIGINT,
    short_signals             BIGINT,
    skipped_signals_no_size   BIGINT,

    run_duration_seconds      NUMERIC(12,3),
    final_equity              NUMERIC(15,2),
    total_return_pct          NUMERIC(14,4),
    total_trades              INTEGER,
    winning_trades            INTEGER,
    losing_trades             INTEGER,
    breakeven_trades          INTEGER,
    win_rate_pct              NUMERIC(8,4),
    profit_factor             NUMERIC(14,4),
    max_drawdown_pct          NUMERIC(10,4),
    avg_win_pct               NUMERIC(12,4),
    avg_loss_pct              NUMERIC(12,4),
    ruined                    BOOLEAN
);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_trades (
    id                        BIGSERIAL     PRIMARY KEY,
    run_id                    INTEGER       NOT NULL REFERENCES backtest2_nas100_hfmed_runs(run_id) ON DELETE CASCADE,
    signal_ts                 TIMESTAMPTZ   NOT NULL,
    entry_ts                  TIMESTAMPTZ   NOT NULL,
    exit_ts                   TIMESTAMPTZ   NOT NULL,
    direction                 TEXT          NOT NULL,

    median_level              NUMERIC(15,4) NOT NULL,
    signal_mid                NUMERIC(15,4) NOT NULL,
    previous_mid              NUMERIC(15,4) NOT NULL,

    entry_bid                 NUMERIC(15,4) NOT NULL,
    entry_ask                 NUMERIC(15,4) NOT NULL,
    entry_price               NUMERIC(15,4) NOT NULL,
    exit_bid                  NUMERIC(15,4) NOT NULL,
    exit_ask                  NUMERIC(15,4) NOT NULL,
    exit_price                NUMERIC(15,4) NOT NULL,
    stop_price                NUMERIC(15,4) NOT NULL,
    take_profit_price         NUMERIC(15,4) NOT NULL,

    units                     NUMERIC(18,8) NOT NULL,
    notional_eur              NUMERIC(18,2),
    margin_used_eur           NUMERIC(18,2),
    gross_pnl_eur             NUMERIC(15,2),
    extra_costs_eur           NUMERIC(15,2),
    pnl_eur                   NUMERIC(15,2) NOT NULL,
    equity_before             NUMERIC(15,2),
    equity_after              NUMERIC(15,2),
    return_pct                NUMERIC(14,4),
    price_pnl_points          NUMERIC(12,4),

    outcome_status            TEXT          NOT NULL,
    ticks_held                INTEGER       NOT NULL,
    seconds_held              NUMERIC(12,3) NOT NULL,
    created_at                TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_run_idx ON backtest2_nas100_hfmed_trades(run_id);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_run_entry_idx ON backtest2_nas100_hfmed_trades(run_id, entry_ts);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_run_direction_idx ON backtest2_nas100_hfmed_trades(run_id, direction);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_run_outcome_idx ON backtest2_nas100_hfmed_trades(run_id, outcome_status);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_monte_carlo (
    run_id                    INTEGER       PRIMARY KEY REFERENCES backtest2_nas100_hfmed_runs(run_id) ON DELETE CASCADE,
    n_simulations             INTEGER       NOT NULL,

    base_final_equity_p05     NUMERIC(15,2),
    base_final_equity_p25     NUMERIC(15,2),
    base_final_equity_p50     NUMERIC(15,2),
    base_final_equity_p75     NUMERIC(15,2),
    base_final_equity_p95     NUMERIC(15,2),
    base_max_drawdown_p05     NUMERIC(10,4),
    base_max_drawdown_p25     NUMERIC(10,4),
    base_max_drawdown_p50     NUMERIC(10,4),
    base_max_drawdown_p75     NUMERIC(10,4),
    base_max_drawdown_p95     NUMERIC(10,4),
    base_total_return_p05     NUMERIC(14,4),
    base_total_return_p25     NUMERIC(14,4),
    base_total_return_p50     NUMERIC(14,4),
    base_total_return_p75     NUMERIC(14,4),
    base_total_return_p95     NUMERIC(14,4),
    base_prob_of_ruin_pct     NUMERIC(8,4),
    base_prob_profitable_pct  NUMERIC(8,4),
    base_worst_final_equity   NUMERIC(15,2),
    base_best_final_equity    NUMERIC(15,2),
    base_worst_max_drawdown_pct NUMERIC(10,4),

    slip_final_equity_p05     NUMERIC(15,2),
    slip_final_equity_p25     NUMERIC(15,2),
    slip_final_equity_p50     NUMERIC(15,2),
    slip_final_equity_p75     NUMERIC(15,2),
    slip_final_equity_p95     NUMERIC(15,2),
    slip_max_drawdown_p05     NUMERIC(10,4),
    slip_max_drawdown_p25     NUMERIC(10,4),
    slip_max_drawdown_p50     NUMERIC(10,4),
    slip_max_drawdown_p75     NUMERIC(10,4),
    slip_max_drawdown_p95     NUMERIC(10,4),
    slip_total_return_p05     NUMERIC(14,4),
    slip_total_return_p25     NUMERIC(14,4),
    slip_total_return_p50     NUMERIC(14,4),
    slip_total_return_p75     NUMERIC(14,4),
    slip_total_return_p95     NUMERIC(14,4),
    slip_prob_of_ruin_pct     NUMERIC(8,4),
    slip_prob_profitable_pct  NUMERIC(8,4),
    slip_worst_final_equity   NUMERIC(15,2),
    slip_best_final_equity    NUMERIC(15,2),
    slip_worst_max_drawdown_pct NUMERIC(10,4),

    seq_final_equity_p05      NUMERIC(15,2),
    seq_final_equity_p25      NUMERIC(15,2),
    seq_final_equity_p50      NUMERIC(15,2),
    seq_final_equity_p75      NUMERIC(15,2),
    seq_final_equity_p95      NUMERIC(15,2),
    seq_max_drawdown_p05      NUMERIC(10,4),
    seq_max_drawdown_p25      NUMERIC(10,4),
    seq_max_drawdown_p50      NUMERIC(10,4),
    seq_max_drawdown_p75      NUMERIC(10,4),
    seq_max_drawdown_p95      NUMERIC(10,4),
    seq_total_return_p05      NUMERIC(14,4),
    seq_total_return_p25      NUMERIC(14,4),
    seq_total_return_p50      NUMERIC(14,4),
    seq_total_return_p75      NUMERIC(14,4),
    seq_total_return_p95      NUMERIC(14,4),
    seq_prob_of_ruin_pct      NUMERIC(8,4),
    seq_prob_profitable_pct   NUMERIC(8,4),
    seq_worst_final_equity    NUMERIC(15,2),
    seq_best_final_equity     NUMERIC(15,2),
    seq_worst_max_drawdown_pct NUMERIC(10,4),

    created_at                TIMESTAMPTZ   NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE, DELETE ON
    backtest2_nas100_hfmed_runs,
    backtest2_nas100_hfmed_trades,
    backtest2_nas100_hfmed_monte_carlo
    TO "market-data-account";

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "market-data-account";

