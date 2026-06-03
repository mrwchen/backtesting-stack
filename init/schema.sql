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
DROP TABLE IF EXISTS backtest_daily_policy_snapshots CASCADE;
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
    ps_margin_requirement_pct NUMERIC(5,2),
    ps_margin_stop_out_level_pct NUMERIC(5,2),
    ps_min_entry_margin_level_pct NUMERIC(5,2),
    ibkr_long_initial_margin_pct NUMERIC(5,2),
    ibkr_long_maintenance_margin_pct NUMERIC(5,2),
    ibkr_short_initial_margin_pct NUMERIC(5,2),
    ibkr_short_maintenance_margin_pct NUMERIC(5,2),
    allow_fractional_shares BOOLEAN,
    spread_bps           NUMERIC(6,2),
    slippage_bps         NUMERIC(6,2),
    commission_per_order_usd NUMERIC(8,4),
    commission_per_share_usd NUMERIC(10,6) NOT NULL DEFAULT 0,
    commission_min_per_order_usd NUMERIC(8,4) NOT NULL DEFAULT 0,
    commission_max_pct    NUMERIC(6,3) NOT NULL DEFAULT 0,
    commission_bps       NUMERIC(6,2),
    margin_financing_rate_pct NUMERIC(5,2),
    ps_share_cfd_arr_pct NUMERIC(6,3),
    ps_share_cfd_admin_fee_pct NUMERIC(6,3),
    ps_share_cfd_short_borrow_rate_pct NUMERIC(6,3),
    ps_share_cfd_overnight_day_count NUMERIC(8,2),
    entry_window_enabled BOOLEAN,
    entry_window_tz      TEXT,
    entry_window_start   TEXT,
    entry_window_end     TEXT,

    -- Common policy snapshot
    long_min_fundamental NUMERIC(5,2)  NOT NULL,
    short_max_fundamental NUMERIC(5,2) NOT NULL,
    min_market_cap_m     NUMERIC(10,2),

    -- Model parameter snapshot
    long_min_pullback    NUMERIC(5,2),
    long_max_pullback    NUMERIC(5,2),
    long_ideal_pullback  NUMERIC(5,2),
    long_max_rsi         NUMERIC(5,2),
    short_min_bounce     NUMERIC(5,2),
    short_max_bounce     NUMERIC(5,2),
    short_ideal_bounce   NUMERIC(5,2),
    short_min_rsi        NUMERIC(5,2),
    short_max_rsi        NUMERIC(5,2),

    -- Central execution/risk policy snapshot
    take_profit_mode TEXT NOT NULL,
    execution_long_take_profit_pct NUMERIC(6,4) NOT NULL,
    execution_short_take_profit_pct NUMERIC(6,4) NOT NULL,
    execution_long_trailing_activation_pct NUMERIC(6,4) NOT NULL,
    execution_short_trailing_activation_pct NUMERIC(6,4) NOT NULL,
    execution_long_trailing_distance_pct NUMERIC(6,4) NOT NULL,
    execution_short_trailing_distance_pct NUMERIC(6,4) NOT NULL,
    execution_long_max_hold_days NUMERIC(6,2) NOT NULL,
    execution_short_max_hold_days NUMERIC(6,2) NOT NULL,
    common_stop_loss_enabled BOOLEAN NOT NULL,
    common_stop_lookback_bars INTEGER NOT NULL,
    common_stop_buffer NUMERIC(8,4) NOT NULL,
    common_stop_atr_lookback_bars INTEGER NOT NULL,
    common_stop_atr_mult NUMERIC(8,4) NOT NULL,
    common_min_stop_pct NUMERIC(6,3) NOT NULL,
    common_max_stop_pct NUMERIC(6,3) NOT NULL,

    -- Results summary (filled after run completes)
    run_duration_seconds NUMERIC(12,3),
    final_equity         NUMERIC(15,2),
    total_trades         INTEGER,
    winning_trades       INTEGER,
    losing_trades        INTEGER,
    breakeven_trades     INTEGER,
    expired_trades       INTEGER,
    win_rate_pct         NUMERIC(5,2),
    total_return_pct     NUMERIC(12,2),
    margin_hours_usd     NUMERIC(20,4),
    return_per_margin_hour_pct NUMERIC(18,8),
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
    intent_date          DATE          NOT NULL,
    as_of_ts             TIMESTAMPTZ,
    symbol               TEXT,
    exchange             TEXT,
    cik                  BIGINT,
    direction            TEXT,

    -- Decision taxonomy
    decision_stage       TEXT          NOT NULL, -- regime_filter | candidate_filter | bar_load | intent_eval | portfolio_filter | order_open
    decision             TEXT          NOT NULL, -- skipped_day | no_candidates | rejected | intent | blocked | opened
    reason_code          TEXT          NOT NULL,
    reason_text          TEXT,
    intent_passed        BOOLEAN       NOT NULL DEFAULT FALSE,
    opened               BOOLEAN       NOT NULL DEFAULT FALSE,
    candidate_rank       INTEGER,
    intent_rank          INTEGER,

    -- Market and fundamental context
    world_regime_label   TEXT,
    world_regime_score   NUMERIC(5,2),
    valuation_label      TEXT,
    sector               TEXT,
    industry             TEXT,
    fundamental_score    NUMERIC(8,4),
    mispricing_score     NUMERIC(8,4),
    market_cap_m         NUMERIC(18,2),

    -- Bar, intent, and execution context
    bar_count            INTEGER,
    min_bars             INTEGER,
    intent_score         NUMERIC(8,4),
    intent_reason        TEXT,
    entry_ts             TIMESTAMPTZ,
    entry_price          NUMERIC(15,4),
    stop_loss            NUMERIC(15,4),
    take_profit          NUMERIC(15,4),
    trailing_activation_price NUMERIC(15,4),
    trailing_distance_pct NUMERIC(6,4),

    -- Portfolio context at decision time
    open_positions       INTEGER,
    max_open_positions   INTEGER,
    account_equity       NUMERIC(15,2),
    initial_margin       NUMERIC(15,2),
    maintenance_margin   NUMERIC(15,2),
    available_funds      NUMERIC(15,2),
    excess_liquidity     NUMERIC(15,2),
    required_initial_margin NUMERIC(15,2),
    required_maintenance_margin NUMERIC(15,2),
    available_funds_after NUMERIC(15,2),
    excess_liquidity_after NUMERIC(15,2),
    position_size_usd    NUMERIC(15,2),
    shares               NUMERIC(15,6)
);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_run_day
    ON backtest_decision_events (run_id, intent_date, decision_stage, decision);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_symbol_day
    ON backtest_decision_events (run_id, symbol, intent_date);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_identity_day
    ON backtest_decision_events (run_id, symbol, exchange, cik, intent_date);

CREATE INDEX IF NOT EXISTS idx_backtest_decision_events_reason
    ON backtest_decision_events (run_id, reason_code, intent_date);

-- ── Per-day Daily Policy snapshot ────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_daily_policy_snapshots (
    run_id                     INTEGER       NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    day                        DATE          NOT NULL,
    created_at                 TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    as_of_ts                   TIMESTAMPTZ,
    policy_available           BOOLEAN       NOT NULL DEFAULT FALSE,
    model_file                 TEXT          NOT NULL,
    account_profile            TEXT          NOT NULL,

    world_regime_label         TEXT,
    world_regime_score         NUMERIC(5,2),
    daily_policy_phase         TEXT,
    world_regime_ma_score      NUMERIC(5,2),
    max_long_positions         INTEGER,
    max_short_positions        INTEGER,
    max_total_positions        INTEGER,
    long_risk_multiplier       NUMERIC(8,4),
    short_risk_multiplier      NUMERIC(8,4),
    risk_per_trade_pct         NUMERIC(8,4),

    halted                     BOOLEAN       NOT NULL DEFAULT FALSE,
    halt_reason_code           TEXT,
    halt_reason_text           TEXT,

    prune_enabled              BOOLEAN       NOT NULL DEFAULT FALSE,
    prune_checked              BOOLEAN       NOT NULL DEFAULT FALSE,
    prune_triggered            BOOLEAN       NOT NULL DEFAULT FALSE,
    prune_closed_positions     INTEGER       NOT NULL DEFAULT 0,
    prune_pnl_usd              NUMERIC(15,2) NOT NULL DEFAULT 0,

    opens_today                INTEGER       NOT NULL DEFAULT 0,
    refill_opens_today         INTEGER       NOT NULL DEFAULT 0,
    sl_closes_today            INTEGER       NOT NULL DEFAULT 0,
    closed_today               INTEGER       NOT NULL DEFAULT 0,
    policy_block_events        INTEGER       NOT NULL DEFAULT 0,
    daily_policy_block_events  INTEGER       NOT NULL DEFAULT 0,
    portfolio_block_events     INTEGER       NOT NULL DEFAULT 0,

    signal_decisions           INTEGER       NOT NULL DEFAULT 0,
    candidate_count_long       INTEGER       NOT NULL DEFAULT 0,
    candidate_count_short      INTEGER       NOT NULL DEFAULT 0,
    intent_count_long          INTEGER       NOT NULL DEFAULT 0,
    intent_count_short         INTEGER       NOT NULL DEFAULT 0,

    open_positions_start       INTEGER       NOT NULL DEFAULT 0,
    long_positions_start       INTEGER       NOT NULL DEFAULT 0,
    short_positions_start      INTEGER       NOT NULL DEFAULT 0,
    open_positions_before_prune INTEGER,
    long_positions_before_prune INTEGER,
    short_positions_before_prune INTEGER,
    open_positions_after_prune INTEGER,
    long_positions_after_prune INTEGER,
    short_positions_after_prune INTEGER,
    open_positions_end         INTEGER       NOT NULL DEFAULT 0,
    long_positions_end         INTEGER       NOT NULL DEFAULT 0,
    short_positions_end        INTEGER       NOT NULL DEFAULT 0,

    day_start_equity           NUMERIC(15,2),
    day_end_equity             NUMERIC(15,2),
    day_return_pct             NUMERIC(12,4),
    day_pnl_usd                NUMERIC(15,2) NOT NULL DEFAULT 0,

    PRIMARY KEY (run_id, day)
);

CREATE INDEX IF NOT EXISTS idx_backtest_daily_policy_snapshots_run_day
    ON backtest_daily_policy_snapshots (run_id, day);

CREATE INDEX IF NOT EXISTS idx_backtest_daily_policy_snapshots_phase
    ON backtest_daily_policy_snapshots (run_id, daily_policy_phase, day);

CREATE INDEX IF NOT EXISTS idx_backtest_daily_policy_snapshots_halt
    ON backtest_daily_policy_snapshots (run_id, halted, halt_reason_code, day);

-- ── Individual trades ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_trades (
    id                   SERIAL        PRIMARY KEY,
    run_id               INTEGER       NOT NULL REFERENCES backtest_runs(run_id),
    intent_date          DATE          NOT NULL,
    symbol               TEXT          NOT NULL,
    exchange             TEXT          NOT NULL,
    cik                  BIGINT        NOT NULL,
    direction            TEXT          NOT NULL,   -- LONG | SHORT

    -- World regime at intent time
    world_regime_label   TEXT,
    world_regime_score   NUMERIC(5,2),

    -- Fundamental label at intent time
    valuation_label      TEXT,
    sector               TEXT,
    industry             TEXT,

    -- Intent scores
    fundamental_score    NUMERIC(5,2),
    intent_score         NUMERIC(8,4),
    intent_reason        TEXT,

    -- Execution levels
    entry_price          NUMERIC(15,4) NOT NULL,
    stop_loss            NUMERIC(15,4) NOT NULL,
    take_profit_mode     TEXT          NOT NULL,
    take_profit          NUMERIC(15,4),
    trailing_activation_price NUMERIC(15,4),
    trailing_distance_pct NUMERIC(6,4),

    -- Position sizing
    position_size_usd    NUMERIC(15,2),
    shares               NUMERIC(15,6),
    margin_used          NUMERIC(15,2),
    maintenance_margin_used NUMERIC(15,2),
    equity_before        NUMERIC(15,2),

    -- Outcome
    outcome_status       TEXT,          -- HIT_TP | HIT_TRAILING_STOP | HIT_SL | MAX_HOLD | FORCE_CLOSED | MARGIN_STOP_OUT | IBKR_MARGIN_LIQUIDATION
    outcome_price        NUMERIC(15,4),
    outcome_date         DATE,
    outcome_bars         INTEGER,       -- 1h bars from entry to close
    trailing_activated   BOOLEAN        NOT NULL DEFAULT FALSE,
    trailing_stop        NUMERIC(15,4),
    return_pct           NUMERIC(8,4),
    margin_hours_usd     NUMERIC(20,4),
    return_per_margin_hour_pct NUMERIC(18,8),
    pnl_usd              NUMERIC(15,2),
    equity_after         NUMERIC(15,2),
    entry_ts             TIMESTAMPTZ   NOT NULL,
    trailing_activated_ts TIMESTAMPTZ,
    exit_ts              TIMESTAMPTZ,

    UNIQUE (run_id, intent_date, symbol, exchange, cik, direction, entry_ts)
);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_run_id
    ON backtest_trades (run_id, intent_date);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_symbol
    ON backtest_trades (symbol, intent_date);

CREATE INDEX IF NOT EXISTS idx_backtest_trades_identity
    ON backtest_trades (symbol, exchange, cik, intent_date);

-- ── Account curve snapshots ─────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS backtest_account_curve (
    run_id               INTEGER       NOT NULL REFERENCES backtest_runs(run_id) ON DELETE CASCADE,
    ts                   TIMESTAMPTZ   NOT NULL,
    trade_date           DATE          NOT NULL,
    seq_in_run           INTEGER       NOT NULL,
    balance_usd          NUMERIC(15,2) NOT NULL,
    open_pnl_usd         NUMERIC(15,2) NOT NULL,
    equity_usd           NUMERIC(15,2) NOT NULL,
    initial_margin_usd   NUMERIC(15,2) NOT NULL,
    maintenance_margin_usd NUMERIC(15,2) NOT NULL,
    available_funds_usd  NUMERIC(15,2) NOT NULL,
    excess_liquidity_usd NUMERIC(15,2) NOT NULL,
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

-- ── Source lookup indexes for point-in-time candidate selection ──────────────

DO $$
BEGIN
    IF to_regclass('public.alpaca_market_data_1h') IS NOT NULL THEN
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_backtest_amd_1h_identity_ts_cover
                ON public.alpaca_market_data_1h (symbol, exchange, cik, ts)
                INCLUDE (open, high, low, close, volume)
        ';
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_backtest_amd_1h_ts
                ON public.alpaca_market_data_1h (ts DESC)
        ';
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.world_regime_daily_scores_mv') IS NOT NULL THEN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'world_regime_daily_scores_mv'
              AND column_name IN (
                  'dominant_shock_type',
                  'max_shock_type_score',
                  'defensive_risk_off_score',
                  'energy_commodity_shock_score',
                  'rates_inflation_usd_shock_score',
                  'credit_banking_stress_score',
                  'policy_geopolitical_score',
                  'tech_stress_shock_score',
                  'precious_metals_score',
                  'industrial_metals_score',
                  'metals_mining_shock_score',
                  'metals_mining_subtype'
              )
            GROUP BY table_schema, table_name
            HAVING COUNT(DISTINCT column_name) = 12
        ) THEN
            EXECUTE '
                CREATE INDEX IF NOT EXISTS idx_backtest_world_regime_day_score
                    ON public.world_regime_daily_scores_mv (day DESC)
                    INCLUDE (
                        regime_label,
                        composite_score,
                        dominant_shock_type,
                        max_shock_type_score,
                        defensive_risk_off_score,
                        energy_commodity_shock_score,
                        rates_inflation_usd_shock_score,
                        credit_banking_stress_score,
                        policy_geopolitical_score,
                        tech_stress_shock_score,
                        precious_metals_score,
                        industrial_metals_score,
                        metals_mining_shock_score,
                        metals_mining_subtype
                    )
                    WHERE composite_score IS NOT NULL
            ';
        ELSE
            EXECUTE '
                CREATE INDEX IF NOT EXISTS idx_backtest_world_regime_day_score
                    ON public.world_regime_daily_scores_mv (day DESC)
                    INCLUDE (regime_label, composite_score)
                    WHERE composite_score IS NOT NULL
            ';
        END IF;
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.stock_scorer_fundamental_scores') IS NOT NULL THEN
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_backtest_safs_identity_pit_cover
                ON public.stock_scorer_fundamental_scores (
                    symbol,
                    exchange,
                    cik,
                    (COALESCE(data_available_at, fundamental_data_available_at)) DESC,
                    "time" DESC
                )
                INCLUDE (
                    composite_score,
                    composite_score_abs,
                    sector,
                    industry,
                    valuation_label,
                    mispricing_score,
                    negative_earnings_flag,
                    high_leverage_flag,
                    long_eligible,
                    short_eligible,
                    market_cap_m,
                    current_price_currency,
                    market_cap_currency,
                    currency,
                    financial_currency
                )
                WHERE symbol IS NOT NULL
                  AND exchange IS NOT NULL
                  AND cik IS NOT NULL
                  AND composite_score IS NOT NULL
        ';
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_backtest_safs_identity_available_full_cover
                ON public.stock_scorer_fundamental_scores (
                    symbol,
                    exchange,
                    cik,
                    (COALESCE(data_available_at, fundamental_data_available_at)) DESC NULLS LAST,
                    "time" DESC
                )
                INCLUDE (
                    composite_score,
                    composite_score_abs,
                    sector,
                    industry,
                    valuation_label,
                    mispricing_score,
                    negative_earnings_flag,
                    high_leverage_flag,
                    long_eligible,
                    short_eligible,
                    market_cap_m,
                    relative_absolute_divergence,
                    long_block_reason,
                    short_block_reason,
                    current_price_currency,
                    market_cap_currency,
                    currency,
                    financial_currency
                )
                WHERE symbol IS NOT NULL
                  AND exchange IS NOT NULL
                  AND cik IS NOT NULL
        ';
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.pepperstone_data') IS NOT NULL
       AND EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'pepperstone_data'
              AND column_name = 'symbol'
       )
       AND EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'pepperstone_data'
              AND column_name = 'symbol_ps'
       )
       AND EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'pepperstone_data'
              AND column_name = 'is_trading_enabled'
       ) THEN
        IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'pepperstone_data'
              AND column_name = 'symbol_ps24'
        ) THEN
            EXECUTE '
                CREATE INDEX IF NOT EXISTS idx_backtest_pepperstone_symbol_tradable
                    ON public.pepperstone_data (symbol)
                    WHERE symbol IS NOT NULL
                      AND is_trading_enabled IS NOT FALSE
                      AND (
                          NULLIF(BTRIM(symbol_ps), '''') IS NOT NULL
                          OR NULLIF(BTRIM(symbol_ps24), '''') IS NOT NULL
                      )
            ';
        ELSE
            EXECUTE '
                CREATE INDEX IF NOT EXISTS idx_backtest_pepperstone_symbol_tradable
                    ON public.pepperstone_data (symbol)
                    WHERE symbol IS NOT NULL
                      AND is_trading_enabled IS NOT FALSE
                      AND NULLIF(BTRIM(symbol_ps), '''') IS NOT NULL
            ';
        END IF;
    END IF;
END;
$$;

DO $$
BEGIN
    IF to_regclass('public.ibkr_symbol_margin_requirements') IS NOT NULL THEN
        EXECUTE '
            CREATE INDEX IF NOT EXISTS idx_backtest_ibkr_margin_action_symbol_usable
                ON public.ibkr_symbol_margin_requirements (
                    (UPPER(TRIM(action))),
                    (UPPER(TRIM(source_symbol)))
                )
                WHERE quantity > 0
                  AND initial_margin_pct > 0
                  AND maintenance_margin_pct > 0
                  AND source_symbol IS NOT NULL
        ';
    END IF;
END;
$$;

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
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_daily_policy_snapshots TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_trades TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_account_curve TO "market-data-account";
GRANT SELECT, INSERT, UPDATE, DELETE ON backtest_monte_carlo TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_runs_run_id_seq   TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_decision_events_id_seq TO "market-data-account";
GRANT USAGE, SELECT ON SEQUENCE backtest_trades_id_seq     TO "market-data-account";
