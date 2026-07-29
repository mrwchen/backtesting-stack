-- Service-owned schema for the point-in-time swing-stock momentum backtest.
-- Re-runnable without migrations. Runtime Python only validates this schema.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'backtesting-account') THEN
        CREATE USER "backtesting-account" WITH PASSWORD 'backtesting-account-pw';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO "backtesting-account";
GRANT USAGE ON SCHEMA public TO "backtesting-account";

\if :drop_all_backtest_momentum_tables_on_start
DROP TABLE IF EXISTS backtest_momentum_equity_daily CASCADE;
DROP TABLE IF EXISTS backtest_momentum_trades CASCADE;
DROP TABLE IF EXISTS backtest_momentum_signals CASCADE;
DROP TABLE IF EXISTS backtest_momentum_runs CASCADE;
\endif

CREATE TABLE IF NOT EXISTS backtest_momentum_runs (
    run_id                                      UUID PRIMARY KEY,
    started_at_utc                             TIMESTAMPTZ NOT NULL,
    completed_at_utc                           TIMESTAMPTZ NOT NULL,
    account_type                               TEXT NOT NULL,
    currency                                   TEXT NOT NULL,
    trading_timezone                           TEXT NOT NULL,
    requested_start_date                       DATE NOT NULL,
    actual_start_date                          DATE NOT NULL,
    end_date                                   DATE NOT NULL,
    starting_capital_usd                       NUMERIC(24,8) NOT NULL,
    ending_cash_usd                            NUMERIC(24,8) NOT NULL,
    ending_market_value_usd                    NUMERIC(24,8) NOT NULL,
    ending_equity_usd                          NUMERIC(24,8) NOT NULL,
    realized_pnl_usd                           NUMERIC(24,8) NOT NULL,
    unrealized_pnl_usd                         NUMERIC(24,8) NOT NULL,
    total_return_pct                           NUMERIC(20,8) NOT NULL,
    max_drawdown_pct                           NUMERIC(20,8) NOT NULL,
    total_commission_usd                       NUMERIC(24,8) NOT NULL,
    signal_count                               INTEGER NOT NULL,
    selected_signal_count                      INTEGER NOT NULL,
    closed_trade_count                         INTEGER NOT NULL,
    open_trade_count                           INTEGER NOT NULL,
    winning_trade_count                        INTEGER NOT NULL,
    losing_trade_count                         INTEGER NOT NULL,
    source_watermark_utc                       TIMESTAMPTZ NOT NULL,
    analyser_watermark_utc                     TIMESTAMPTZ NOT NULL,
    source_market_table                        TEXT NOT NULL,
    source_analyser_table                      TEXT NOT NULL,
    price_basis                                TEXT NOT NULL,
    entry_execution_model                      TEXT NOT NULL,
    atr_exit_execution_model                   TEXT NOT NULL,
    intraday_conflict_policy                   TEXT NOT NULL,
    gap_execution_policy                       TEXT NOT NULL,
    end_of_data_policy                         TEXT NOT NULL,
    prior_high_price_field                     TEXT NOT NULL,
    atr_method                                 TEXT NOT NULL,
    fractional_shares_allowed                  BOOLEAN NOT NULL,
    risk_equity_basis                          TEXT NOT NULL,
    ranking_policy                             TEXT NOT NULL,
    symbol_reentry_policy                      TEXT NOT NULL,
    max_positions                              SMALLINT NOT NULL,
    max_new_positions_per_day                  SMALLINT NOT NULL,
    risk_per_position_pct                      NUMERIC(12,8) NOT NULL,
    initial_stop_loss_pct                      NUMERIC(12,8) NOT NULL,
    stop_step_interval_sessions                SMALLINT NOT NULL,
    stop_step_pct                              NUMERIC(12,8) NOT NULL,
    initial_take_profit_pct                    NUMERIC(12,8) NOT NULL,
    take_profit_step_interval_sessions         SMALLINT NOT NULL,
    take_profit_step_pct                       NUMERIC(12,8) NOT NULL,
    atr_period_sessions                        SMALLINT NOT NULL,
    atr_day1_exit_max_pct                      NUMERIC(12,8) NOT NULL,
    atr_day2_exit_max_pct                      NUMERIC(12,8) NOT NULL,
    prior_high_lookback_sessions               SMALLINT NOT NULL,
    prior_high_max_above_signal_close_pct      NUMERIC(12,8) NOT NULL,
    min_daily_price_change_pct                 NUMERIC(12,8) NOT NULL,
    max_daily_price_change_pct_exclusive       NUMERIC(12,8) NOT NULL,
    min_volume_vs_sma21_ratio_exclusive        NUMERIC(12,8) NOT NULL,
    commission_bps                             NUMERIC(12,8) NOT NULL,
    slippage_bps                               NUMERIC(12,8) NOT NULL,
    analyser_min_price_usd                     NUMERIC(20,8) NOT NULL,
    analyser_min_dollar_volume_usd             NUMERIC(24,8) NOT NULL,
    analyser_rs_lookback_1_sessions            SMALLINT NOT NULL,
    analyser_rs_lookback_2_sessions            SMALLINT NOT NULL,
    analyser_rs_lookback_3_sessions            SMALLINT NOT NULL,
    analyser_rs_lookback_4_sessions            SMALLINT NOT NULL,
    analyser_rs_weight_1                       NUMERIC(12,8) NOT NULL,
    analyser_rs_weight_2                       NUMERIC(12,8) NOT NULL,
    analyser_rs_weight_3                       NUMERIC(12,8) NOT NULL,
    analyser_rs_weight_4                       NUMERIC(12,8) NOT NULL,
    analyser_rs_min                            SMALLINT NOT NULL,
    analyser_ma200_trend_sessions              SMALLINT NOT NULL,
    analyser_min_above_52w_low_ratio           NUMERIC(12,8) NOT NULL,
    analyser_min_near_52w_high_ratio           NUMERIC(12,8) NOT NULL,
    CHECK (completed_at_utc >= started_at_utc),
    CHECK (account_type = 'unlevered'),
    CHECK (currency = 'USD'),
    CHECK (trading_timezone = 'America/New_York'),
    CHECK (price_basis = 'adjusted_ohlc'),
    CHECK (entry_execution_model = 'signal_day_close'),
    CHECK (atr_exit_execution_model = 'signal_day_close'),
    CHECK (intraday_conflict_policy = 'low_before_high'),
    CHECK (gap_execution_policy = 'open_when_beyond_active_level'),
    CHECK (end_of_data_policy = 'mark_to_market'),
    CHECK (prior_high_price_field = 'adjusted_high'),
    CHECK (atr_method = 'simple_tr14_mean_over_adjusted_close_pct'),
    CHECK (NOT fractional_shares_allowed),
    CHECK (risk_equity_basis = 'current_account_equity_before_entry'),
    CHECK (ranking_policy = 'volume_ratio_desc_return_desc_symbol_asc'),
    CHECK (symbol_reentry_policy = 'allowed_after_exit'),
    CHECK (actual_start_date >= requested_start_date),
    CHECK (end_date >= actual_start_date),
    CHECK (starting_capital_usd > 0),
    CHECK (ending_cash_usd >= 0),
    CHECK (ending_market_value_usd >= 0),
    CHECK (max_positions > 0),
    CHECK (max_new_positions_per_day > 0),
    CHECK (max_new_positions_per_day <= max_positions),
    CHECK (risk_per_position_pct > 0),
    CHECK (initial_stop_loss_pct < 0),
    CHECK (initial_take_profit_pct > 0),
    CHECK (commission_bps >= 0),
    CHECK (slippage_bps >= 0),
    CHECK (source_watermark_utc = analyser_watermark_utc),
    CHECK (signal_count >= selected_signal_count),
    CHECK (selected_signal_count = closed_trade_count + open_trade_count)
);

CREATE TABLE IF NOT EXISTS backtest_momentum_signals (
    run_id                                      UUID NOT NULL REFERENCES backtest_momentum_runs(run_id) ON DELETE CASCADE,
    signal_date                                DATE NOT NULL,
    symbol                                     TEXT NOT NULL,
    exchange                                   TEXT NOT NULL,
    cik                                        BIGINT NOT NULL,
    selection_rank                             INTEGER,
    decision                                   TEXT NOT NULL,
    selected                                   BOOLEAN NOT NULL,
    prior_high_observation_count               SMALLINT NOT NULL,
    prior_max_adjusted_high                     NUMERIC(20,8),
    prior_high_limit_adjusted_price             NUMERIC(20,8),
    account_equity_before_entry_usd            NUMERIC(24,8),
    available_cash_before_entry_usd            NUMERIC(24,8),
    risk_budget_usd                            NUMERIC(24,8),
    risk_per_share_usd                         NUMERIC(20,8),
    risk_sized_shares                          INTEGER,
    cash_limited_shares                        INTEGER,
    selected_shares                            INTEGER,
    entry_reference_price                      NUMERIC(20,8),
    entry_fill_price                           NUMERIC(20,8),
    entry_commission_usd                       NUMERIC(24,8),

    price_continuity_segment                   INTEGER NOT NULL,
    currency                                   TEXT NOT NULL,
    raw_close                                  NUMERIC(20,8),
    adjusted_close                             NUMERIC(20,8),
    adjusted_high                              NUMERIC(20,8),
    adjusted_low                               NUMERIC(20,8),
    adjusted_volume                            BIGINT,
    daily_price_change_pct                     NUMERIC(20,8),
    adjusted_volume_sma21_prior                NUMERIC(30,8),
    adjusted_volume_vs_sma21_prior_ratio       NUMERIC(30,8),
    adjusted_volume_sma50_prior                NUMERIC(30,8),
    adjusted_volume_vs_sma50_prior_ratio       NUMERIC(30,8),
    daily_traded_notional_usd                  NUMERIC(24,2),
    daily_traded_notional_sma21_prior_usd      NUMERIC(30,8),
    daily_traded_notional_vs_sma21_prior_ratio NUMERIC(30,8),
    daily_traded_notional_sma50_prior_usd      NUMERIC(30,8),
    daily_traded_notional_vs_sma50_prior_ratio NUMERIC(30,8),
    dollar_volume_63d                          NUMERIC(30,8),
    ma5                                        NUMERIC(20,8),
    ma9                                        NUMERIC(20,8),
    ma21                                       NUMERIC(20,8),
    ma50                                       NUMERIC(20,8),
    ma150                                      NUMERIC(20,8),
    ma200                                      NUMERIC(20,8),
    ma200_21_sessions_ago                      NUMERIC(20,8),
    low_52w                                    NUMERIC(20,8),
    high_52w                                   NUMERIC(20,8),
    rs_raw                                     NUMERIC(24,10),
    rs_rating                                  SMALLINT,
    rs_universe_size                           INTEGER NOT NULL,
    forward_5d_max_gain_pct                    NUMERIC(20,8),
    forward_5d_max_loss_pct                    NUMERIC(20,8),
    forward_10d_max_gain_pct                   NUMERIC(20,8),
    forward_10d_max_loss_pct                   NUMERIC(20,8),
    forward_15d_max_gain_pct                   NUMERIC(20,8),
    forward_15d_max_loss_pct                   NUMERIC(20,8),
    forward_20d_max_gain_pct                   NUMERIC(20,8),
    forward_20d_max_loss_pct                   NUMERIC(20,8),
    forward_30d_max_gain_pct                   NUMERIC(20,8),
    forward_30d_max_loss_pct                   NUMERIC(20,8),
    forward_45d_max_gain_pct                   NUMERIC(20,8),
    forward_45d_max_loss_pct                   NUMERIC(20,8),
    forward_60d_max_gain_pct                   NUMERIC(20,8),
    forward_60d_max_loss_pct                   NUMERIC(20,8),
    forward_90d_max_gain_pct                   NUMERIC(20,8),
    forward_90d_max_loss_pct                   NUMERIC(20,8),
    crit_price_above_ma150_200                 BOOLEAN NOT NULL,
    crit_ma150_above_ma200                     BOOLEAN NOT NULL,
    crit_ma200_rising                          BOOLEAN NOT NULL,
    crit_ma50_above_ma150_200                  BOOLEAN NOT NULL,
    crit_price_above_ma50                      BOOLEAN NOT NULL,
    crit_above_52w_low                         BOOLEAN NOT NULL,
    crit_near_52w_high                         BOOLEAN NOT NULL,
    crit_rs_rating                             BOOLEAN NOT NULL,
    trend_template_pass                        BOOLEAN NOT NULL,

    PRIMARY KEY (signal_date, run_id, symbol, exchange, cik),
    CHECK (price_continuity_segment > 0),
    CHECK (currency = 'USD'),
    CHECK (prior_high_observation_count >= 0),
    CHECK (selection_rank IS NULL OR selection_rank > 0),
    CHECK (selected = (decision = 'selected')),
    CHECK (selected_shares IS NULL OR selected_shares > 0),
    CHECK (
        decision IN (
            'selected',
            'invalid_execution_price',
            'prior_high_history_incomplete',
            'prior_high_limit_exceeded',
            'symbol_already_open',
            'position_limit_reached',
            'daily_entry_limit_reached',
            'risk_budget_below_one_share',
            'insufficient_cash'
        )
    ),
    CHECK (
        trend_template_pass = (
            crit_price_above_ma150_200
            AND crit_ma150_above_ma200
            AND crit_ma200_rising
            AND crit_ma50_above_ma150_200
            AND crit_price_above_ma50
            AND crit_above_52w_low
            AND crit_near_52w_high
            AND crit_rs_rating
        )
    )
);

SELECT create_hypertable(
    'backtest_momentum_signals',
    'signal_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_backtest_momentum_signals_run_date
    ON backtest_momentum_signals (run_id, signal_date DESC, selection_rank);
CREATE INDEX IF NOT EXISTS idx_backtest_momentum_signals_selected
    ON backtest_momentum_signals (run_id, selected, signal_date DESC);

CREATE TABLE IF NOT EXISTS backtest_momentum_trades (
    run_id                              UUID NOT NULL REFERENCES backtest_momentum_runs(run_id) ON DELETE CASCADE,
    trade_id                            INTEGER NOT NULL,
    symbol                              TEXT NOT NULL,
    exchange                            TEXT NOT NULL,
    cik                                 BIGINT NOT NULL,
    entry_date                          DATE NOT NULL,
    exit_date                           DATE,
    status                              TEXT NOT NULL,
    exit_reason                         TEXT,
    shares                              INTEGER NOT NULL,
    holding_sessions                    INTEGER NOT NULL,
    entry_reference_price               NUMERIC(20,8) NOT NULL,
    entry_fill_price                    NUMERIC(20,8) NOT NULL,
    entry_notional_usd                  NUMERIC(24,8) NOT NULL,
    entry_commission_usd                NUMERIC(24,8) NOT NULL,
    risk_budget_usd                     NUMERIC(24,8) NOT NULL,
    planned_initial_stop_loss_usd       NUMERIC(24,8) NOT NULL,
    initial_stop_price                  NUMERIC(20,8) NOT NULL,
    initial_take_profit_price           NUMERIC(20,8) NOT NULL,
    active_stop_price_at_exit           NUMERIC(20,8),
    active_take_profit_price_at_exit    NUMERIC(20,8),
    exit_reference_price                NUMERIC(20,8),
    exit_fill_price                     NUMERIC(20,8),
    exit_notional_usd                   NUMERIC(24,8),
    exit_commission_usd                 NUMERIC(24,8),
    last_valuation_date                 DATE NOT NULL,
    last_mark_price                     NUMERIC(20,8) NOT NULL,
    market_value_usd                    NUMERIC(24,8) NOT NULL,
    gross_pnl_usd                       NUMERIC(24,8) NOT NULL,
    net_pnl_usd                         NUMERIC(24,8) NOT NULL,
    return_pct                          NUMERIC(20,8) NOT NULL,
    PRIMARY KEY (run_id, trade_id),
    CHECK (status IN ('open', 'closed')),
    CHECK (shares > 0),
    CHECK (holding_sessions >= 0),
    CHECK (entry_reference_price > 0),
    CHECK (entry_fill_price > 0),
    CHECK (entry_notional_usd > 0),
    CHECK (entry_commission_usd >= 0),
    CHECK (planned_initial_stop_loss_usd <= risk_budget_usd),
    CHECK (initial_stop_price < entry_fill_price),
    CHECK (initial_take_profit_price > entry_fill_price),
    CHECK (market_value_usd >= 0),
    CHECK (
        (status = 'open' AND exit_date IS NULL AND exit_reason IS NULL
            AND exit_fill_price IS NULL AND exit_commission_usd IS NULL)
        OR
        (status = 'closed' AND exit_date IS NOT NULL AND exit_reason IS NOT NULL
            AND exit_fill_price IS NOT NULL AND exit_commission_usd IS NOT NULL
            AND market_value_usd = 0)
    ),
    CHECK (
        exit_reason IS NULL OR exit_reason IN (
            'stop_loss_gap',
            'take_profit_gap',
            'stop_loss_intraday',
            'take_profit_intraday',
            'atr_day1',
            'atr_day2'
        )
    )
);

CREATE INDEX IF NOT EXISTS idx_backtest_momentum_trades_run_entry
    ON backtest_momentum_trades (run_id, entry_date, trade_id);
CREATE INDEX IF NOT EXISTS idx_backtest_momentum_trades_run_status
    ON backtest_momentum_trades (run_id, status, symbol);

CREATE TABLE IF NOT EXISTS backtest_momentum_equity_daily (
    run_id                              UUID NOT NULL REFERENCES backtest_momentum_runs(run_id) ON DELETE CASCADE,
    valuation_date                     DATE NOT NULL,
    cash_usd                            NUMERIC(24,8) NOT NULL,
    positions_market_value_usd         NUMERIC(24,8) NOT NULL,
    total_equity_usd                    NUMERIC(24,8) NOT NULL,
    realized_pnl_usd                    NUMERIC(24,8) NOT NULL,
    unrealized_pnl_usd                  NUMERIC(24,8) NOT NULL,
    cumulative_commission_usd           NUMERIC(24,8) NOT NULL,
    open_position_count                 SMALLINT NOT NULL,
    closed_trade_count                  INTEGER NOT NULL,
    daily_return_pct                    NUMERIC(20,8) NOT NULL,
    total_return_pct                    NUMERIC(20,8) NOT NULL,
    drawdown_pct                        NUMERIC(20,8) NOT NULL,
    PRIMARY KEY (valuation_date, run_id),
    CHECK (cash_usd >= 0),
    CHECK (positions_market_value_usd >= 0),
    CHECK (abs(total_equity_usd - cash_usd - positions_market_value_usd) <= 0.00000002),
    CHECK (cumulative_commission_usd >= 0),
    CHECK (open_position_count >= 0),
    CHECK (closed_trade_count >= 0),
    CHECK (drawdown_pct <= 0)
);

SELECT create_hypertable(
    'backtest_momentum_equity_daily',
    'valuation_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_backtest_momentum_equity_run_date
    ON backtest_momentum_equity_daily (run_id, valuation_date DESC);

GRANT SELECT, INSERT ON backtest_momentum_runs TO "backtesting-account";
GRANT SELECT, INSERT ON backtest_momentum_signals TO "backtesting-account";
GRANT SELECT, INSERT ON backtest_momentum_trades TO "backtesting-account";
GRANT SELECT, INSERT ON backtest_momentum_equity_daily TO "backtesting-account";
GRANT SELECT ON stock_core_market_metrics_daily TO "backtesting-account";
GRANT SELECT ON stock_analyser_trend_template_daily TO "backtesting-account";
GRANT SELECT ON stock_analyser_incremental_state TO "backtesting-account";

COMMENT ON TABLE backtest_momentum_runs IS
    'Completed atomic backtest runs and all strategy/analyser parameters used for reproducibility.';
COMMENT ON TABLE backtest_momentum_signals IS
    'Entry candidates, deterministic selection decisions, and the complete stock-analyser source row.';
COMMENT ON TABLE backtest_momentum_trades IS
    'Selected positions with execution, exit, risk, cost, and final mark results.';
COMMENT ON TABLE backtest_momentum_equity_daily IS
    'Daily unlevered account equity in USD; 365-day chunks and no compression.';
COMMENT ON COLUMN backtest_momentum_signals.forward_5d_max_gain_pct IS
    'Copied analyser outcome label for result analysis only; never read by the entry or exit policy.';
