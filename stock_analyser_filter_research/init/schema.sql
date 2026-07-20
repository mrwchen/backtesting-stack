-- Service-owned schema for the simple stock-analyser filter research.
-- Runtime Python validates and writes these objects but never creates or alters them.

DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'market-data-account') THEN
        CREATE USER "market-data-account" WITH PASSWORD 'market-data-account-pw';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE postgres TO "market-data-account";
GRANT USAGE ON SCHEMA public TO "market-data-account";

\if :drop_all_stock_analyser_filter_research_tables_on_start
DROP TABLE IF EXISTS stock_analyser_filter_research_rule_results CASCADE;
DROP TABLE IF EXISTS stock_analyser_filter_research_signal_results CASCADE;
\endif

CREATE TABLE IF NOT EXISTS stock_analyser_filter_research_signal_results (
    signal_date                                  DATE NOT NULL,
    previous_session_date                        DATE NOT NULL,
    symbol                                       TEXT NOT NULL,
    exchange                                     TEXT NOT NULL,
    cik                                          BIGINT NOT NULL,
    price_continuity_segment                     INTEGER NOT NULL,
    currency                                     TEXT NOT NULL,

    raw_close                                    NUMERIC(20,8),
    adjusted_close                               NUMERIC(20,8),
    adjusted_high                                NUMERIC(20,8),
    adjusted_low                                 NUMERIC(20,8),
    adjusted_volume                              BIGINT,
    daily_price_change_pct                       NUMERIC(20,8),
    adjusted_volume_sma21_prior                  NUMERIC(30,8),
    adjusted_volume_vs_sma21_prior_ratio         NUMERIC(30,8),
    adjusted_volume_sma50_prior                  NUMERIC(30,8),
    adjusted_volume_vs_sma50_prior_ratio         NUMERIC(30,8),
    daily_traded_notional_usd                    NUMERIC(24,2),
    daily_traded_notional_sma21_prior_usd        NUMERIC(30,8),
    daily_traded_notional_vs_sma21_prior_ratio   NUMERIC(30,8),
    daily_traded_notional_sma50_prior_usd        NUMERIC(30,8),
    daily_traded_notional_vs_sma50_prior_ratio   NUMERIC(30,8),
    dollar_volume_63d                            NUMERIC(30,8),
    ma50                                         NUMERIC(20,8),
    ma150                                        NUMERIC(20,8),
    ma200                                        NUMERIC(20,8),
    ma200_21_sessions_ago                        NUMERIC(20,8),
    low_52w                                      NUMERIC(20,8),
    high_52w                                     NUMERIC(20,8),
    rs_raw                                       NUMERIC(24,10),
    rs_rating                                    SMALLINT,

    trigger_criteria                             TEXT NOT NULL,
    trigger_count                                SMALLINT NOT NULL,
    previous_criteria_pass_count                 SMALLINT NOT NULL,
    prior_7_of_8_count_10d                       SMALLINT,
    sessions_since_previous_pass                 INTEGER,

    distance_to_ma50_pct                         NUMERIC(20,8),
    distance_to_ma150_pct                        NUMERIC(20,8),
    distance_to_ma200_pct                        NUMERIC(20,8),
    ma50_vs_ma150_pct                            NUMERIC(20,8),
    ma50_vs_ma200_pct                            NUMERIC(20,8),
    ma150_vs_ma200_pct                           NUMERIC(20,8),
    ma200_slope_21d_pct                          NUMERIC(20,8),
    price_vs_52w_low_pct                         NUMERIC(20,8),
    price_vs_52w_high_pct                        NUMERIC(20,8),

    prior_return_5d_pct                          NUMERIC(20,8),
    prior_return_10d_pct                         NUMERIC(20,8),
    prior_return_21d_pct                         NUMERIC(20,8),
    prior_momentum_acceleration_5d_pct_points    NUMERIC(20,8),
    prior_atr_14d_pct                            NUMERIC(20,8),
    prior_max_drawdown_21d_pct                   NUMERIC(20,8),
    prior_close_vs_20d_high_pct                  NUMERIC(20,8),
    prior_close_vs_63d_high_pct                  NUMERIC(20,8),
    signal_close_vs_prior_20d_high_pct           NUMERIC(20,8),
    prior_range_compression_10_vs_10_ratio       NUMERIC(20,8),
    signal_close_location_value                  NUMERIC(20,8),
    prior_rs_rating_change_5d                    NUMERIC(20,8),

    prior_volume_sma5_vs21_ratio                 NUMERIC(20,8),
    prior_volume_sma10_vs21_ratio                NUMERIC(20,8),
    prior_notional_sma5_vs21_ratio               NUMERIC(20,8),
    prior_notional_sma10_vs21_ratio              NUMERIC(20,8),
    prior_up_volume_share21                      NUMERIC(20,8),
    prior_up_notional_share21                    NUMERIC(20,8),
    prior_price_volume_corr21                    NUMERIC(20,8),
    prior_price_notional_corr21                  NUMERIC(20,8),

    forward_5d_max_gain_pct                      NUMERIC(20,8),
    forward_5d_max_loss_pct                      NUMERIC(20,8),
    forward_10d_max_gain_pct                     NUMERIC(20,8),
    forward_10d_max_loss_pct                     NUMERIC(20,8),
    forward_20d_max_gain_pct                     NUMERIC(20,8),
    forward_20d_max_loss_pct                     NUMERIC(20,8),
    weak_5d                                      BOOLEAN,
    strong_5d                                    BOOLEAN,
    deep_loss_5d                                 BOOLEAN,
    bad_5d                                       BOOLEAN,
    late_strong_10d                              BOOLEAN,
    late_strong_20d                              BOOLEAN,

    analysis_split                               TEXT NOT NULL,
    include_stage_a                              BOOLEAN NOT NULL,
    include_stage_ab                             BOOLEAN NOT NULL,
    include_stage_abc                            BOOLEAN NOT NULL,
    matched_rule_ids                             TEXT,
    filter_decision                              TEXT NOT NULL,
    exclusion_reason                             TEXT,

    PRIMARY KEY (signal_date, symbol, exchange, cik),
    CHECK (previous_session_date < signal_date),
    CHECK (price_continuity_segment > 0),
    CHECK (currency = 'USD'),
    CHECK (rs_rating IS NULL OR rs_rating BETWEEN 1 AND 99),
    CHECK (trigger_count BETWEEN 1 AND 8),
    CHECK (previous_criteria_pass_count BETWEEN 0 AND 7),
    CHECK (prior_7_of_8_count_10d IS NULL OR prior_7_of_8_count_10d BETWEEN 0 AND 10),
    CHECK (forward_5d_max_gain_pct IS NULL OR forward_5d_max_gain_pct >= 0),
    CHECK (forward_5d_max_loss_pct IS NULL OR forward_5d_max_loss_pct <= 0),
    CHECK (forward_10d_max_gain_pct IS NULL OR forward_10d_max_gain_pct >= 0),
    CHECK (forward_10d_max_loss_pct IS NULL OR forward_10d_max_loss_pct <= 0),
    CHECK (forward_20d_max_gain_pct IS NULL OR forward_20d_max_gain_pct >= 0),
    CHECK (forward_20d_max_loss_pct IS NULL OR forward_20d_max_loss_pct <= 0),
    CHECK (
        (weak_5d IS NULL AND strong_5d IS NULL AND deep_loss_5d IS NULL AND bad_5d IS NULL)
        OR
        (weak_5d IS NOT NULL AND strong_5d IS NOT NULL AND deep_loss_5d IS NOT NULL
         AND bad_5d = (weak_5d OR deep_loss_5d))
    ),
    CHECK (analysis_split IN ('discovery', 'validation', 'test', 'purged')),
    CHECK (NOT include_stage_ab OR include_stage_a),
    CHECK (NOT include_stage_abc OR include_stage_ab),
    CHECK (filter_decision IN ('include', 'exclude')),
    CHECK ((filter_decision = 'include') = include_stage_abc),
    CHECK (
        (filter_decision = 'include' AND exclusion_reason IS NULL)
        OR
        (filter_decision = 'exclude' AND NULLIF(TRIM(exclusion_reason), '') IS NOT NULL)
    )
);

SELECT create_hypertable(
    'stock_analyser_filter_research_signal_results',
    'signal_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_safr_signal_identity
    ON stock_analyser_filter_research_signal_results
    (symbol, exchange, cik, signal_date DESC);

CREATE INDEX IF NOT EXISTS idx_safr_signal_decision
    ON stock_analyser_filter_research_signal_results
    (analysis_split, filter_decision, signal_date DESC);

CREATE TABLE IF NOT EXISTS stock_analyser_filter_research_rule_results (
    result_id                      BIGSERIAL PRIMARY KEY,
    rule_id                        TEXT NOT NULL,
    result_kind                    TEXT NOT NULL,
    stage                          TEXT NOT NULL,
    feature_group                  TEXT NOT NULL,
    feature_name                   TEXT,
    operator                       TEXT,
    threshold_value                NUMERIC(30,10),
    bin_number                     SMALLINT,
    bin_lower_bound                NUMERIC(30,10),
    bin_upper_bound                NUMERIC(30,10),
    rule_text                      TEXT NOT NULL,
    selection_order                SMALLINT,
    evaluation_scope               TEXT NOT NULL,
    scope_year                     SMALLINT,
    period_start                   DATE,
    period_end                     DATE,
    is_selected                    BOOLEAN NOT NULL,
    is_final_filter                BOOLEAN NOT NULL,
    passes_holdout                 BOOLEAN,
    component_count                SMALLINT NOT NULL,
    signal_count                   INTEGER NOT NULL,
    sample_count                   INTEGER NOT NULL,
    unlabeled_count                INTEGER NOT NULL,
    matched_signal_count           INTEGER NOT NULL,
    matched_unlabeled_count        INTEGER NOT NULL,
    weak_count                     INTEGER NOT NULL,
    deep_loss_count                INTEGER NOT NULL,
    bad_count                      INTEGER NOT NULL,
    strong_count                   INTEGER NOT NULL,
    late_strong_10d_count          INTEGER NOT NULL,
    late_strong_20d_count          INTEGER NOT NULL,
    excluded_count                 INTEGER NOT NULL,
    excluded_weak_count            INTEGER NOT NULL,
    excluded_deep_loss_count       INTEGER NOT NULL,
    excluded_bad_count             INTEGER NOT NULL,
    excluded_strong_count          INTEGER NOT NULL,
    excluded_late_strong_10d_count INTEGER NOT NULL,
    excluded_late_strong_20d_count INTEGER NOT NULL,
    label_coverage_rate            NUMERIC(18,8),
    matched_label_coverage_rate    NUMERIC(18,8),
    exclusion_rate                 NUMERIC(18,8),
    weak_capture_rate              NUMERIC(18,8),
    deep_loss_capture_rate         NUMERIC(18,8),
    bad_capture_rate               NUMERIC(18,8),
    strong_rejection_rate          NUMERIC(18,8),
    strong_retention_rate          NUMERIC(18,8),
    excluded_bad_rate              NUMERIC(18,8),
    bad_lift                       NUMERIC(18,8),
    retained_bad_rate              NUMERIC(18,8),
    retained_strong_rate           NUMERIC(18,8),
    late_strong_10d_rejection_rate NUMERIC(18,8),
    late_strong_20d_rejection_rate NUMERIC(18,8),

    CHECK (result_kind IN ('baseline', 'quantile_bin', 'candidate_rule', 'selected_filter')),
    CHECK (stage IN ('baseline', 'A', 'B', 'C', 'A_B', 'A_B_C')),
    CHECK (feature_group IN ('none', 'A', 'B', 'C', 'multiple')),
    CHECK (operator IS NULL OR operator IN ('le', 'ge')),
    CHECK (evaluation_scope IN ('discovery', 'validation', 'test', 'all_signals', 'calendar_year')),
    CHECK (
        (evaluation_scope = 'calendar_year' AND scope_year IS NOT NULL)
        OR
        (evaluation_scope <> 'calendar_year' AND scope_year IS NULL)
    ),
    CHECK (period_end IS NULL OR period_start IS NOT NULL),
    CHECK (period_end IS NULL OR period_end >= period_start),
    CHECK (selection_order IS NULL OR selection_order BETWEEN 1 AND 3),
    CHECK (NOT is_final_filter OR is_selected),
    CHECK (component_count >= 0),
    CHECK (signal_count >= 0),
    CHECK (sample_count >= 0),
    CHECK (sample_count <= signal_count),
    CHECK (unlabeled_count = signal_count - sample_count),
    CHECK (matched_signal_count BETWEEN 0 AND signal_count),
    CHECK (matched_unlabeled_count BETWEEN 0 AND unlabeled_count),
    CHECK (weak_count BETWEEN 0 AND sample_count),
    CHECK (deep_loss_count BETWEEN 0 AND sample_count),
    CHECK (bad_count BETWEEN 0 AND sample_count),
    CHECK (strong_count BETWEEN 0 AND sample_count),
    CHECK (late_strong_10d_count BETWEEN 0 AND sample_count),
    CHECK (late_strong_20d_count BETWEEN 0 AND sample_count),
    CHECK (excluded_count BETWEEN 0 AND sample_count),
    CHECK (excluded_weak_count BETWEEN 0 AND weak_count),
    CHECK (excluded_deep_loss_count BETWEEN 0 AND deep_loss_count),
    CHECK (excluded_bad_count BETWEEN 0 AND bad_count),
    CHECK (excluded_strong_count BETWEEN 0 AND strong_count),
    CHECK (excluded_late_strong_10d_count BETWEEN 0 AND late_strong_10d_count),
    CHECK (excluded_late_strong_20d_count BETWEEN 0 AND late_strong_20d_count),
    CHECK (matched_signal_count = excluded_count + matched_unlabeled_count),
    CHECK (label_coverage_rate IS NULL OR label_coverage_rate BETWEEN 0 AND 1),
    CHECK (matched_label_coverage_rate IS NULL OR matched_label_coverage_rate BETWEEN 0 AND 1),
    CHECK (exclusion_rate IS NULL OR exclusion_rate BETWEEN 0 AND 1),
    CHECK (weak_capture_rate IS NULL OR weak_capture_rate BETWEEN 0 AND 1),
    CHECK (deep_loss_capture_rate IS NULL OR deep_loss_capture_rate BETWEEN 0 AND 1),
    CHECK (bad_capture_rate IS NULL OR bad_capture_rate BETWEEN 0 AND 1),
    CHECK (strong_rejection_rate IS NULL OR strong_rejection_rate BETWEEN 0 AND 1),
    CHECK (strong_retention_rate IS NULL OR strong_retention_rate BETWEEN 0 AND 1),
    CHECK (excluded_bad_rate IS NULL OR excluded_bad_rate BETWEEN 0 AND 1),
    CHECK (retained_bad_rate IS NULL OR retained_bad_rate BETWEEN 0 AND 1),
    CHECK (retained_strong_rate IS NULL OR retained_strong_rate BETWEEN 0 AND 1),
    CHECK (late_strong_10d_rejection_rate IS NULL OR late_strong_10d_rejection_rate BETWEEN 0 AND 1),
    CHECK (late_strong_20d_rejection_rate IS NULL OR late_strong_20d_rejection_rate BETWEEN 0 AND 1)
);

CREATE INDEX IF NOT EXISTS idx_safr_rule_recommended
    ON stock_analyser_filter_research_rule_results
    (is_final_filter, is_selected, evaluation_scope, result_kind);

CREATE INDEX IF NOT EXISTS idx_safr_rule_feature
    ON stock_analyser_filter_research_rule_results
    (feature_group, feature_name, evaluation_scope);

GRANT SELECT, INSERT
    ON stock_analyser_filter_research_signal_results TO "market-data-account";
GRANT SELECT, INSERT
    ON stock_analyser_filter_research_rule_results TO "market-data-account";
GRANT USAGE, SELECT
    ON SEQUENCE stock_analyser_filter_research_rule_results_result_id_seq
    TO "market-data-account";

COMMENT ON TABLE stock_analyser_filter_research_signal_results IS
    'One continuity-safe false-to-true trend-template event per stock, with causal A-C features, forward research outcomes and the selected simple filter decision.';
COMMENT ON TABLE stock_analyser_filter_research_rule_results IS
    'Discovery-only quantile boundaries, exclusion-rule evaluations and the validation-selected simple rule evaluated on untouched test data.';
