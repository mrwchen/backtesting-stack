-- Schema for the NAS100 hit-frequency median tick backtester optimizer.
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
GRANT SELECT ON public.pepperstone_ticks_data TO "market-data-account";

\if :drop_hfmed_tables_on_start
DROP TABLE IF EXISTS backtest2_nas100_hfmed_trades CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_monte_carlo CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_parameter_session_stats CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_fold_results CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_parameter_sets CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_runs CASCADE;
\endif

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_runs (
    run_id                              BIGSERIAL     PRIMARY KEY,
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    run_label                           TEXT          NOT NULL,
    mode                                TEXT          NOT NULL,
    status                              TEXT          NOT NULL,
    notes                               TEXT,

    source_table                        TEXT          NOT NULL,
    symbol                              TEXT          NOT NULL,
    start_ts_utc                        TIMESTAMPTZ,
    end_ts_utc                          TIMESTAMPTZ,
    data_start_ts                       TIMESTAMPTZ,
    data_end_ts                         TIMESTAMPTZ,
    ticks_loaded                        BIGINT,
    bars_built                          INTEGER,
    folds_built                         INTEGER,

    bar_seconds                         INTEGER       NOT NULL,
    baseline_lookback_bars              INTEGER       NOT NULL,
    min_lookback_bars                   INTEGER       NOT NULL,
    price_step                          NUMERIC(12,6) NOT NULL,
    median_quantile                     NUMERIC(8,6)  NOT NULL,
    band_lower_quantile                 NUMERIC(8,6)  NOT NULL,
    band_upper_quantile                 NUMERIC(8,6)  NOT NULL,
    baseline_long_cross_quantile        NUMERIC(8,6)  NOT NULL,
    baseline_short_cross_quantile       NUMERIC(8,6)  NOT NULL,
    stop_mode                           TEXT          NOT NULL,
    baseline_stop_points                NUMERIC(12,4) NOT NULL,
    baseline_take_profit_points         NUMERIC(12,4) NOT NULL,
    baseline_min_profile_range_points   NUMERIC(12,4) NOT NULL,
    baseline_stop_profile_lower_quantile NUMERIC(8,6) NOT NULL,
    baseline_stop_profile_upper_quantile NUMERIC(8,6) NOT NULL,
    baseline_stop_profile_buffer_points NUMERIC(12,4) NOT NULL,
    baseline_min_stop_distance_points   NUMERIC(12,4) NOT NULL,
    baseline_max_stop_distance_points   NUMERIC(12,4) NOT NULL,

    account_profile                     TEXT          NOT NULL,
    initial_equity                      NUMERIC(15,2) NOT NULL,
    account_currency                    TEXT          NOT NULL,
    margin_requirement_pct              NUMERIC(8,4),
    risk_per_trade_pct                  NUMERIC(8,4),
    max_margin_pct                      NUMERIC(8,4),
    contract_multiplier                 NUMERIC(12,4),
    lot_size                            NUMERIC(12,4),
    eurusd_rate                         NUMERIC(12,6),

    spread_points                       NUMERIC(12,4),
    slippage_points                     NUMERIC(12,4),
    commission_per_unit                 NUMERIC(12,6),

    monte_carlo_enabled                 BOOLEAN,
    monte_carlo_simulations             INTEGER,
    mc_extra_slippage_points            NUMERIC(12,4),
    mc_block_size                       INTEGER,
    mc_ruin_drawdown_pct                NUMERIC(8,4),
    mc_random_seed                      BIGINT,

    wf_train_days                       INTEGER,
    wf_test_days                        INTEGER,
    wf_step_days                        INTEGER,
    wf_train_top_n_per_fold             INTEGER,
    optimizer_processes                 INTEGER,
    stage1_max_parameter_sets           INTEGER,
    stage2_enabled                      BOOLEAN,
    stage2_seed_top_n                   INTEGER,
    stage2_max_parameter_sets           INTEGER,
    mc_score_top_n                      INTEGER,
    persist_top_trades_n                INTEGER,
    min_oos_trades                      INTEGER,
    min_oos_profit_factor               NUMERIC(12,4),
    max_oos_drawdown_pct                NUMERIC(12,4),
    max_mc_ruin_pct                     NUMERIC(12,4),

    stage1_parameter_sets               INTEGER,
    stage2_parameter_sets               INTEGER,
    best_parameter_set_id               BIGINT,
    best_score                          NUMERIC(18,6),
    run_duration_seconds                NUMERIC(14,3)
);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_parameter_sets (
    parameter_set_id                    BIGSERIAL     PRIMARY KEY,
    run_id                              BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_runs(run_id) ON DELETE CASCADE,
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    stage                               TEXT          NOT NULL,
    stage_rank                          INTEGER,
    parameter_hash                      TEXT          NOT NULL,
    parameter_label                     TEXT          NOT NULL,
    parameter_signature                 TEXT          NOT NULL,

    lookback_bars                       INTEGER       NOT NULL,
    long_cross_quantile                 NUMERIC(8,6)  NOT NULL,
    short_cross_quantile                NUMERIC(8,6)  NOT NULL,
    take_profit_points                  NUMERIC(12,4) NOT NULL,
    min_profile_range_points            NUMERIC(12,4) NOT NULL,
    stop_profile_lower_quantile         NUMERIC(8,6)  NOT NULL,
    stop_profile_upper_quantile         NUMERIC(8,6)  NOT NULL,
    stop_profile_buffer_points          NUMERIC(12,4) NOT NULL,
    min_stop_distance_points            NUMERIC(12,4) NOT NULL,
    max_stop_distance_points            NUMERIC(12,4) NOT NULL,

    pre_mc_score                        NUMERIC(18,6),
    score                               NUMERIC(18,6),
    mc_scored                           BOOLEAN       NOT NULL DEFAULT FALSE,
    mc_prob_of_ruin_pct                 NUMERIC(12,4),
    passed_pre_mc_filters               BOOLEAN       NOT NULL DEFAULT FALSE,
    passed_filters                      BOOLEAN       NOT NULL DEFAULT FALSE,
    top_trades_persisted                BOOLEAN       NOT NULL DEFAULT FALSE,

    train_folds                         INTEGER,
    train_expected_folds                INTEGER,
    train_total_trades                  INTEGER,
    train_total_return_pct              NUMERIC(14,4),
    train_mean_return_pct               NUMERIC(14,4),
    train_median_return_pct             NUMERIC(14,4),
    train_std_return_pct                NUMERIC(14,4),
    train_max_drawdown_pct              NUMERIC(10,4),
    train_profit_factor                 NUMERIC(14,4),
    train_win_rate_pct                  NUMERIC(8,4),
    train_profitable_folds_pct          NUMERIC(8,4),
    train_gross_profit_eur              NUMERIC(18,2),
    train_gross_loss_eur                NUMERIC(18,2),
    train_net_profit_eur                NUMERIC(18,2),
    train_avg_trade_pnl_eur             NUMERIC(14,4),
    train_signals_total                 BIGINT,
    train_ruined_folds                  INTEGER,

    oos_folds                           INTEGER,
    oos_expected_folds                  INTEGER,
    oos_total_trades                    INTEGER,
    oos_total_return_pct                NUMERIC(14,4),
    oos_mean_return_pct                 NUMERIC(14,4),
    oos_median_return_pct               NUMERIC(14,4),
    oos_std_return_pct                  NUMERIC(14,4),
    oos_max_drawdown_pct                NUMERIC(10,4),
    oos_profit_factor                   NUMERIC(14,4),
    oos_win_rate_pct                    NUMERIC(8,4),
    oos_profitable_folds_pct            NUMERIC(8,4),
    oos_gross_profit_eur                NUMERIC(18,2),
    oos_gross_loss_eur                  NUMERIC(18,2),
    oos_net_profit_eur                  NUMERIC(18,2),
    oos_avg_trade_pnl_eur               NUMERIC(14,4),
    oos_signals_total                   BIGINT,
    oos_ruined_folds                    INTEGER,

    UNIQUE (run_id, stage, parameter_hash)
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_parameter_sets_run_stage_rank_idx
    ON backtest2_nas100_hfmed_parameter_sets(run_id, stage, stage_rank);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_parameter_sets_run_score_idx
    ON backtest2_nas100_hfmed_parameter_sets(run_id, score DESC);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_parameter_sets_filters_idx
    ON backtest2_nas100_hfmed_parameter_sets(run_id, passed_filters, score DESC);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_parameter_session_stats (
    session_stat_id                    BIGSERIAL     PRIMARY KEY,
    run_id                             BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_runs(run_id) ON DELETE CASCADE,
    parameter_set_id                   BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_parameter_sets(parameter_set_id) ON DELETE CASCADE,
    created_at                         TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    stage                              TEXT          NOT NULL,
    window_role                        TEXT          NOT NULL,
    session_type                       TEXT          NOT NULL,
    session_label                      TEXT          NOT NULL,
    session_sort_order                 INTEGER       NOT NULL,

    folds                              INTEGER       NOT NULL,
    expected_folds                     INTEGER       NOT NULL,
    total_trades                       INTEGER       NOT NULL,
    winning_trades                     INTEGER       NOT NULL,
    losing_trades                      INTEGER       NOT NULL,
    breakeven_trades                   INTEGER       NOT NULL,
    win_rate_pct                       NUMERIC(8,4)  NOT NULL,
    gross_profit_eur                   NUMERIC(18,2) NOT NULL,
    gross_loss_eur                     NUMERIC(18,2) NOT NULL,
    net_profit_eur                     NUMERIC(18,2) NOT NULL,
    avg_trade_pnl_eur                  NUMERIC(14,4) NOT NULL,

    UNIQUE (parameter_set_id, window_role, session_type)
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_parameter_session_stats_run_idx
    ON backtest2_nas100_hfmed_parameter_session_stats(run_id, stage, window_role, session_sort_order);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_parameter_session_stats_param_idx
    ON backtest2_nas100_hfmed_parameter_session_stats(parameter_set_id, window_role, session_sort_order);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_fold_results (
    fold_result_id                      BIGSERIAL     PRIMARY KEY,
    run_id                              BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_runs(run_id) ON DELETE CASCADE,
    parameter_set_id                    BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_parameter_sets(parameter_set_id) ON DELETE CASCADE,
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    stage                               TEXT          NOT NULL,
    fold_index                          INTEGER       NOT NULL,
    window_role                         TEXT          NOT NULL,
    window_start                        TIMESTAMPTZ   NOT NULL,
    window_end                          TIMESTAMPTZ   NOT NULL,

    ticks_simulated                     BIGINT,
    bars_total                          INTEGER,
    signals_total                       BIGINT,
    long_signals                        BIGINT,
    short_signals                       BIGINT,
    rejected_missing_band               BIGINT,
    rejected_band_too_narrow            BIGINT,
    rejected_stop_too_small             BIGINT,
    rejected_stop_too_large             BIGINT,
    skipped_no_size                     BIGINT,
    ruined                              BOOLEAN,
    score                               NUMERIC(18,6),

    total_trades                        INTEGER,
    winning_trades                      INTEGER,
    losing_trades                       INTEGER,
    breakeven_trades                    INTEGER,
    win_rate_pct                        NUMERIC(8,4),
    profit_factor                       NUMERIC(14,4),
    total_return_pct                    NUMERIC(14,4),
    max_drawdown_pct                    NUMERIC(10,4),
    avg_win_pct                         NUMERIC(12,4),
    avg_loss_pct                        NUMERIC(12,4),
    gross_profit_eur                    NUMERIC(18,2),
    gross_loss_eur                      NUMERIC(18,2),
    net_profit_eur                      NUMERIC(18,2),
    avg_trade_pnl_eur                   NUMERIC(14,4),
    final_equity                        NUMERIC(18,2),
    avg_realized_risk_pct               NUMERIC(8,4),
    median_realized_risk_pct            NUMERIC(8,4),
    max_realized_risk_pct               NUMERIC(8,4),
    margin_capped_share_pct             NUMERIC(8,4)
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_fold_results_param_idx
    ON backtest2_nas100_hfmed_fold_results(parameter_set_id, fold_index, window_role);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_fold_results_run_idx
    ON backtest2_nas100_hfmed_fold_results(run_id, stage, fold_index, window_role);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_monte_carlo (
    parameter_set_id                    BIGINT        PRIMARY KEY REFERENCES backtest2_nas100_hfmed_parameter_sets(parameter_set_id) ON DELETE CASCADE,
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    mc_score_rank                       INTEGER,
    n_simulations                       INTEGER       NOT NULL,

    base_final_equity_p05               NUMERIC(15,2),
    base_final_equity_p25               NUMERIC(15,2),
    base_final_equity_p50               NUMERIC(15,2),
    base_final_equity_p75               NUMERIC(15,2),
    base_final_equity_p95               NUMERIC(15,2),
    base_max_drawdown_p05               NUMERIC(10,4),
    base_max_drawdown_p25               NUMERIC(10,4),
    base_max_drawdown_p50               NUMERIC(10,4),
    base_max_drawdown_p75               NUMERIC(10,4),
    base_max_drawdown_p95               NUMERIC(10,4),
    base_total_return_p05               NUMERIC(14,4),
    base_total_return_p25               NUMERIC(14,4),
    base_total_return_p50               NUMERIC(14,4),
    base_total_return_p75               NUMERIC(14,4),
    base_total_return_p95               NUMERIC(14,4),
    base_prob_of_ruin_pct               NUMERIC(8,4),
    base_prob_profitable_pct            NUMERIC(8,4),
    base_worst_final_equity             NUMERIC(15,2),
    base_best_final_equity              NUMERIC(15,2),
    base_worst_max_drawdown_pct         NUMERIC(10,4),

    slip_final_equity_p05               NUMERIC(15,2),
    slip_final_equity_p25               NUMERIC(15,2),
    slip_final_equity_p50               NUMERIC(15,2),
    slip_final_equity_p75               NUMERIC(15,2),
    slip_final_equity_p95               NUMERIC(15,2),
    slip_max_drawdown_p05               NUMERIC(10,4),
    slip_max_drawdown_p25               NUMERIC(10,4),
    slip_max_drawdown_p50               NUMERIC(10,4),
    slip_max_drawdown_p75               NUMERIC(10,4),
    slip_max_drawdown_p95               NUMERIC(10,4),
    slip_total_return_p05               NUMERIC(14,4),
    slip_total_return_p25               NUMERIC(14,4),
    slip_total_return_p50               NUMERIC(14,4),
    slip_total_return_p75               NUMERIC(14,4),
    slip_total_return_p95               NUMERIC(14,4),
    slip_prob_of_ruin_pct               NUMERIC(8,4),
    slip_prob_profitable_pct            NUMERIC(8,4),
    slip_worst_final_equity             NUMERIC(15,2),
    slip_best_final_equity              NUMERIC(15,2),
    slip_worst_max_drawdown_pct         NUMERIC(10,4),

    seq_final_equity_p05                NUMERIC(15,2),
    seq_final_equity_p25                NUMERIC(15,2),
    seq_final_equity_p50                NUMERIC(15,2),
    seq_final_equity_p75                NUMERIC(15,2),
    seq_final_equity_p95                NUMERIC(15,2),
    seq_max_drawdown_p05                NUMERIC(10,4),
    seq_max_drawdown_p25                NUMERIC(10,4),
    seq_max_drawdown_p50                NUMERIC(10,4),
    seq_max_drawdown_p75                NUMERIC(10,4),
    seq_max_drawdown_p95                NUMERIC(10,4),
    seq_total_return_p05                NUMERIC(14,4),
    seq_total_return_p25                NUMERIC(14,4),
    seq_total_return_p50                NUMERIC(14,4),
    seq_total_return_p75                NUMERIC(14,4),
    seq_total_return_p95                NUMERIC(14,4),
    seq_prob_of_ruin_pct                NUMERIC(8,4),
    seq_prob_profitable_pct             NUMERIC(8,4),
    seq_worst_final_equity              NUMERIC(15,2),
    seq_best_final_equity               NUMERIC(15,2),
    seq_worst_max_drawdown_pct          NUMERIC(10,4)
);

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_trades (
    trade_id                            BIGSERIAL     PRIMARY KEY,
    parameter_set_id                    BIGINT        NOT NULL REFERENCES backtest2_nas100_hfmed_parameter_sets(parameter_set_id) ON DELETE CASCADE,
    created_at                          TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    stage                               TEXT          NOT NULL,
    fold_index                          INTEGER       NOT NULL,
    window_role                         TEXT          NOT NULL,
    entry_session                       TEXT          NOT NULL,

    signal_ts                           TIMESTAMPTZ   NOT NULL,
    entry_ts                            TIMESTAMPTZ   NOT NULL,
    exit_ts                             TIMESTAMPTZ   NOT NULL,
    direction                           TEXT          NOT NULL,

    cross_quantile                      NUMERIC(8,6)  NOT NULL,
    cross_level                         NUMERIC(15,4) NOT NULL,
    median_level                        NUMERIC(15,4) NOT NULL,
    signal_mid                          NUMERIC(15,4) NOT NULL,
    previous_mid                        NUMERIC(15,4) NOT NULL,

    entry_bid                           NUMERIC(15,4) NOT NULL,
    entry_ask                           NUMERIC(15,4) NOT NULL,
    entry_price                         NUMERIC(15,4) NOT NULL,
    exit_bid                            NUMERIC(15,4) NOT NULL,
    exit_ask                            NUMERIC(15,4) NOT NULL,
    exit_price                          NUMERIC(15,4) NOT NULL,
    stop_price                          NUMERIC(15,4) NOT NULL,
    take_profit_price                   NUMERIC(15,4) NOT NULL,

    units                               NUMERIC(18,8) NOT NULL,
    notional_eur                        NUMERIC(18,2),
    margin_used_eur                     NUMERIC(18,2),
    gross_pnl_eur                       NUMERIC(15,2),
    extra_costs_eur                     NUMERIC(15,2),
    pnl_eur                             NUMERIC(15,2) NOT NULL,
    equity_before                       NUMERIC(15,2),
    equity_after                        NUMERIC(15,2),
    return_pct                          NUMERIC(14,4),
    price_pnl_points                    NUMERIC(12,4),

    outcome_status                      TEXT          NOT NULL,
    ticks_held                          INTEGER       NOT NULL,
    seconds_held                        NUMERIC(12,3) NOT NULL,
    realized_risk_eur                   NUMERIC(15,2),
    realized_risk_pct                   NUMERIC(8,4),
    margin_capped                       BOOLEAN
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_param_entry_idx
    ON backtest2_nas100_hfmed_trades(parameter_set_id, entry_ts);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_param_direction_idx
    ON backtest2_nas100_hfmed_trades(parameter_set_id, direction);
CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_trades_param_outcome_idx
    ON backtest2_nas100_hfmed_trades(parameter_set_id, outcome_status);

-- Backfill realized-risk columns on already-existing deployments (no-op on fresh installs).
ALTER TABLE backtest2_nas100_hfmed_fold_results
    ADD COLUMN IF NOT EXISTS avg_realized_risk_pct    NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS median_realized_risk_pct NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS max_realized_risk_pct    NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS margin_capped_share_pct  NUMERIC(8,4);

ALTER TABLE backtest2_nas100_hfmed_trades
    ADD COLUMN IF NOT EXISTS realized_risk_eur        NUMERIC(15,2),
    ADD COLUMN IF NOT EXISTS realized_risk_pct        NUMERIC(8,4),
    ADD COLUMN IF NOT EXISTS margin_capped            BOOLEAN;

GRANT SELECT, INSERT, UPDATE, DELETE ON
    backtest2_nas100_hfmed_runs,
    backtest2_nas100_hfmed_parameter_sets,
    backtest2_nas100_hfmed_parameter_session_stats,
    backtest2_nas100_hfmed_fold_results,
    backtest2_nas100_hfmed_monte_carlo,
    backtest2_nas100_hfmed_trades
    TO "market-data-account";

GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO "market-data-account";
