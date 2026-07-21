-- Service-owned schema for stock-analyser filter and early-cut research.
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

\if :{?drop_all_stock_analyser_filter_research_tables_on_start}
\else
\set drop_all_stock_analyser_filter_research_tables_on_start false
\endif

\if :drop_all_stock_analyser_filter_research_tables_on_start
DROP TABLE IF EXISTS stock_analyser_filter_research_rule_results CASCADE;
DROP TABLE IF EXISTS stock_analyser_filter_research_early_cut_results CASCADE;
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

    prior_base_width_20_pct                      NUMERIC(20,8),
    prior_trend_slope_20_pct_per_session         NUMERIC(20,8),
    prior_trend_r2_20                            NUMERIC(20,8),
    prior_trend_efficiency_20                    NUMERIC(20,8),
    prior_positive_return_share_20               NUMERIC(20,8),
    prior_peak_age_40_sessions                   SMALLINT,
    prior_pullback_from_40d_high_pct             NUMERIC(20,8),
    prior_trough_age_40_sessions                 SMALLINT,
    prior_drawdown_to_trough_40_pct              NUMERIC(20,8),
    prior_recovery_from_trough_40_pct            NUMERIC(20,8),
    prior_v_recovery_fraction_40                 NUMERIC(20,8),
    prior_distribution_day_count_20              SMALLINT,
    prior_churning_day_count_20                  SMALLINT,
    prior_failed_breakout_count_20               SMALLINT,

    fundamental_snapshot_age_days                NUMERIC(20,8),
    fundamental_report_age_days                  NUMERIC(20,8),
    fundamental_gross_margin_ttm_ratio           NUMERIC(20,8),
    fundamental_operating_margin_ttm_ratio       NUMERIC(20,8),
    fundamental_net_margin_ttm_ratio             NUMERIC(20,8),
    fundamental_fcf_margin_ttm_ratio             NUMERIC(20,8),
    fundamental_fcf_sbc_adjusted_margin_ttm_ratio NUMERIC(20,8),
    fundamental_debt_to_capital_ratio            NUMERIC(20,8),
    fundamental_cash_to_assets_ratio             NUMERIC(20,8),
    fundamental_current_ratio                    NUMERIC(20,8),
    fundamental_accruals_ratio                   NUMERIC(20,8),
    fundamental_sbc_to_revenue_ttm_ratio         NUMERIC(20,8),
    fundamental_quarter_filing_age_days          NUMERIC(20,8),
    fundamental_quarter_age_days                 NUMERIC(20,8),
    fundamental_quarterly_revenue_yoy_growth_ratio NUMERIC(20,8),
    fundamental_quarterly_eps_yoy_change_ratio   NUMERIC(20,8),
    fundamental_quarterly_operating_margin_ratio NUMERIC(20,8),
    fundamental_quarterly_operating_margin_yoy_change NUMERIC(20,8),
    fundamental_quarterly_net_margin_ratio       NUMERIC(20,8),
    fundamental_quarterly_net_margin_yoy_change  NUMERIC(20,8),

    market_cap_usd                              BIGINT,
    log_market_cap_usd                          NUMERIC(20,8),
    market_cap_shares_staleness_days            INTEGER,

    forward_5d_max_gain_pct                      NUMERIC(20,8),
    forward_5d_max_loss_pct                      NUMERIC(20,8),
    forward_10d_max_gain_pct                     NUMERIC(20,8),
    forward_10d_max_loss_pct                     NUMERIC(20,8),
    forward_20d_max_gain_pct                     NUMERIC(20,8),
    forward_20d_max_loss_pct                     NUMERIC(20,8),
    forward_5d_label_end_date                    DATE,
    terminal_close_return_5d_pct                 NUMERIC(20,8),
    first_gain_2pct_day                          SMALLINT,
    first_gain_5pct_day                          SMALLINT,
    first_loss_5pct_day                          SMALLINT,
    gain_loss_order_5d                           TEXT,
    weak_5d                                      BOOLEAN,
    strong_5d                                    BOOLEAN,
    deep_loss_5d                                 BOOLEAN,
    bad_5d                                       BOOLEAN,
    loss_first_5d                                BOOLEAN,
    strong_first_5d                              BOOLEAN,
    late_strong_10d                              BOOLEAN,
    late_strong_20d                              BOOLEAN,

    analysis_split                               TEXT NOT NULL,
    include_weak_filter                          BOOLEAN NOT NULL,
    include_loss_first_filter                    BOOLEAN NOT NULL,
    include_final                                BOOLEAN NOT NULL,
    weak_matched_rule_ids                        TEXT,
    loss_first_matched_rule_ids                  TEXT,
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
    CHECK (sessions_since_previous_pass IS NULL OR sessions_since_previous_pass > 0),
    CHECK (prior_base_width_20_pct IS NULL OR prior_base_width_20_pct >= 0),
    CHECK (prior_trend_r2_20 IS NULL OR prior_trend_r2_20 BETWEEN 0 AND 1),
    CHECK (prior_trend_efficiency_20 IS NULL OR prior_trend_efficiency_20 BETWEEN 0 AND 1),
    CHECK (prior_positive_return_share_20 IS NULL OR prior_positive_return_share_20 BETWEEN 0 AND 1),
    CHECK (prior_peak_age_40_sessions IS NULL OR prior_peak_age_40_sessions BETWEEN 0 AND 39),
    CHECK (prior_trough_age_40_sessions IS NULL OR prior_trough_age_40_sessions BETWEEN 0 AND 39),
    CHECK (prior_drawdown_to_trough_40_pct IS NULL OR prior_drawdown_to_trough_40_pct <= 0),
    CHECK (prior_recovery_from_trough_40_pct IS NULL OR prior_recovery_from_trough_40_pct >= 0),
    CHECK (prior_v_recovery_fraction_40 IS NULL OR prior_v_recovery_fraction_40 >= 0),
    CHECK (prior_distribution_day_count_20 IS NULL OR prior_distribution_day_count_20 BETWEEN 0 AND 20),
    CHECK (prior_churning_day_count_20 IS NULL OR prior_churning_day_count_20 BETWEEN 0 AND 20),
    CHECK (prior_failed_breakout_count_20 IS NULL OR prior_failed_breakout_count_20 BETWEEN 0 AND 20),
    CHECK (fundamental_snapshot_age_days IS NULL OR fundamental_snapshot_age_days >= 0),
    CHECK (fundamental_report_age_days IS NULL OR fundamental_report_age_days >= 0),
    CHECK (fundamental_quarter_filing_age_days IS NULL OR fundamental_quarter_filing_age_days >= 0),
    CHECK (fundamental_quarter_age_days IS NULL OR fundamental_quarter_age_days >= 0),
    CHECK (market_cap_usd IS NULL OR market_cap_usd > 0),
    CHECK (log_market_cap_usd IS NULL OR log_market_cap_usd >= 0),
    CHECK ((market_cap_usd IS NULL) = (log_market_cap_usd IS NULL)),
    CHECK (
        market_cap_shares_staleness_days IS NULL
        OR market_cap_shares_staleness_days >= 0
    ),
    CHECK (
        fundamental_quarterly_eps_yoy_change_ratio IS NULL
        OR fundamental_quarterly_eps_yoy_change_ratio BETWEEN -1 AND 1
    ),
    CHECK (forward_5d_max_gain_pct IS NULL OR forward_5d_max_gain_pct >= 0),
    CHECK (forward_5d_max_loss_pct IS NULL OR forward_5d_max_loss_pct <= 0),
    CHECK (forward_10d_max_gain_pct IS NULL OR forward_10d_max_gain_pct >= 0),
    CHECK (forward_10d_max_loss_pct IS NULL OR forward_10d_max_loss_pct <= 0),
    CHECK (forward_20d_max_gain_pct IS NULL OR forward_20d_max_gain_pct >= 0),
    CHECK (forward_20d_max_loss_pct IS NULL OR forward_20d_max_loss_pct <= 0),
    CHECK (forward_5d_label_end_date IS NULL OR forward_5d_label_end_date > signal_date),
    CHECK (first_gain_2pct_day IS NULL OR first_gain_2pct_day BETWEEN 1 AND 5),
    CHECK (first_gain_5pct_day IS NULL OR first_gain_5pct_day BETWEEN 1 AND 5),
    CHECK (first_loss_5pct_day IS NULL OR first_loss_5pct_day BETWEEN 1 AND 5),
    CHECK (
        gain_loss_order_5d IS NULL OR gain_loss_order_5d IN (
            'gain_first', 'loss_first', 'same_day_ambiguous',
            'gain_only', 'loss_only', 'neither'
        )
    ),
    CHECK (
        (weak_5d IS NULL AND strong_5d IS NULL AND deep_loss_5d IS NULL
         AND bad_5d IS NULL)
        OR
        (weak_5d IS NOT NULL AND strong_5d IS NOT NULL AND deep_loss_5d IS NOT NULL
         AND bad_5d IS NOT NULL AND bad_5d = (weak_5d OR deep_loss_5d))
    ),
    CHECK ((loss_first_5d IS NULL) = (strong_first_5d IS NULL)),
    CHECK (
        gain_loss_order_5d IS NOT NULL
        OR (loss_first_5d IS NULL AND strong_first_5d IS NULL)
    ),
    CHECK (
        gain_loss_order_5d <> 'same_day_ambiguous'
        OR (loss_first_5d IS NULL AND strong_first_5d IS NULL)
    ),
    CHECK (analysis_split IN ('discovery', 'validation', 'diagnostic', 'holdout', 'purged')),
    CHECK (include_final = (include_weak_filter AND include_loss_first_filter)),
    CHECK (filter_decision IN ('include', 'exclude')),
    CHECK ((filter_decision = 'include') = include_final),
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

CREATE TABLE IF NOT EXISTS stock_analyser_filter_research_early_cut_results (
    signal_date                                      DATE NOT NULL,
    landmark_day                                     SMALLINT NOT NULL,
    landmark_date                                    DATE,
    effective_session_date                           DATE,
    horizon_end_date                                 DATE,
    symbol                                           TEXT NOT NULL,
    exchange                                         TEXT NOT NULL,
    cik                                              BIGINT NOT NULL,
    price_continuity_segment                         INTEGER NOT NULL,
    currency                                         TEXT NOT NULL,

    landmark_observed                                BOOLEAN NOT NULL,
    same_continuity_segment                          BOOLEAN NOT NULL,
    eligible_at_landmark                             BOOLEAN NOT NULL,
    active_at_landmark                               BOOLEAN NOT NULL,
    prior_policy_cut_day                             SMALLINT,
    full_outcome_available                           BOOLEAN NOT NULL,
    signal_adjusted_close                            NUMERIC(20,8),
    landmark_adjusted_close                          NUMERIC(20,8),
    landmark_adjusted_high                           NUMERIC(20,8),
    landmark_adjusted_low                            NUMERIC(20,8),
    landmark_adjusted_volume                         BIGINT,
    landmark_daily_traded_notional_usd               NUMERIC(24,2),
    landmark_daily_price_change_pct                  NUMERIC(20,8),
    landmark_volume_vs_sma21_prior_ratio             NUMERIC(30,8),
    landmark_volume_vs_sma50_prior_ratio             NUMERIC(30,8),
    landmark_notional_vs_sma21_prior_ratio           NUMERIC(30,8),
    landmark_notional_vs_sma50_prior_ratio           NUMERIC(30,8),
    landmark_rs_rating                               SMALLINT,
    landmark_criteria_pass_count                     SMALLINT,
    landmark_trend_template_pass                     BOOLEAN,

    close_return_from_signal_pct                     NUMERIC(20,8),
    max_gain_to_landmark_pct                         NUMERIC(20,8),
    max_loss_to_landmark_pct                         NUMERIC(20,8),
    drawdown_from_post_signal_high_pct               NUMERIC(20,8),
    rebound_from_post_signal_low_pct                 NUMERIC(20,8),
    landmark_true_range_pct                          NUMERIC(20,8),
    landmark_close_location_value                    NUMERIC(20,8),
    volume_vs_signal_ratio                           NUMERIC(20,8),
    notional_vs_signal_ratio                         NUMERIC(20,8),
    rs_rating_change_from_signal                     NUMERIC(20,8),
    landmark_distance_to_ma50_pct                    NUMERIC(20,8),
    landmark_distance_to_ma150_pct                   NUMERIC(20,8),
    landmark_distance_to_ma200_pct                   NUMERIC(20,8),
    landmark_price_vs_52w_high_pct                   NUMERIC(20,8),
    landmark_atr_14d_pct                             NUMERIC(20,8),
    mean_volume_since_signal_vs_prior21_ratio        NUMERIC(30,8),
    mean_notional_since_signal_vs_prior21_ratio      NUMERIC(30,8),

    hit_gain_2pct_so_far                             BOOLEAN,
    hit_gain_5pct_so_far                             BOOLEAN,
    hit_loss_5pct_so_far                             BOOLEAN,
    first_gain_2pct_day_so_far                       SMALLINT,
    first_gain_5pct_day_so_far                       SMALLINT,
    first_loss_5pct_day_so_far                       SMALLINT,
    remaining_max_gain_from_signal_pct               NUMERIC(20,8),
    remaining_max_loss_from_signal_pct               NUMERIC(20,8),
    remaining_max_gain_from_landmark_pct             NUMERIC(20,8),
    remaining_max_loss_from_landmark_pct             NUMERIC(20,8),
    terminal_close_return_from_signal_pct             NUMERIC(20,8),
    terminal_close_return_from_landmark_pct           NUMERIC(20,8),
    future_first_gain_2pct_day                        SMALLINT,
    future_first_gain_5pct_day                        SMALLINT,
    future_first_loss_5pct_day                        SMALLINT,
    continuation_outcome                              TEXT,
    stagnant_to_day5                                  BOOLEAN,
    loss_first_to_day5                                BOOLEAN,
    strong_first_to_day5                              BOOLEAN,
    bad_to_day5                                       BOOLEAN,

    analysis_split                                    TEXT NOT NULL,
    include_stagnation_filter                         BOOLEAN NOT NULL,
    include_loss_filter                               BOOLEAN NOT NULL,
    include_final                                     BOOLEAN NOT NULL,
    stagnation_matched_rule_ids                       TEXT,
    loss_matched_rule_ids                             TEXT,
    matched_rule_ids                                  TEXT,
    cut_decision                                      TEXT NOT NULL,
    cut_reason                                        TEXT,

    PRIMARY KEY (signal_date, symbol, exchange, cik, landmark_day),
    CHECK (landmark_day BETWEEN 1 AND 3),
    CHECK (price_continuity_segment > 0),
    CHECK (currency = 'USD'),
    CHECK (landmark_date IS NULL OR landmark_date > signal_date),
    CHECK (effective_session_date IS NULL OR effective_session_date >= signal_date),
    CHECK (horizon_end_date IS NULL OR horizon_end_date > signal_date),
    CHECK (landmark_rs_rating IS NULL OR landmark_rs_rating BETWEEN 1 AND 99),
    CHECK (landmark_criteria_pass_count IS NULL OR landmark_criteria_pass_count BETWEEN 0 AND 8),
    CHECK (max_gain_to_landmark_pct IS NULL OR max_gain_to_landmark_pct >= 0),
    CHECK (max_loss_to_landmark_pct IS NULL OR max_loss_to_landmark_pct <= 0),
    CHECK (remaining_max_gain_from_signal_pct IS NULL OR remaining_max_gain_from_signal_pct >= 0),
    CHECK (remaining_max_loss_from_signal_pct IS NULL OR remaining_max_loss_from_signal_pct <= 0),
    CHECK (remaining_max_gain_from_landmark_pct IS NULL OR remaining_max_gain_from_landmark_pct >= 0),
    CHECK (remaining_max_loss_from_landmark_pct IS NULL OR remaining_max_loss_from_landmark_pct <= 0),
    CHECK (first_gain_2pct_day_so_far IS NULL OR first_gain_2pct_day_so_far BETWEEN 1 AND landmark_day),
    CHECK (first_gain_5pct_day_so_far IS NULL OR first_gain_5pct_day_so_far BETWEEN 1 AND landmark_day),
    CHECK (first_loss_5pct_day_so_far IS NULL OR first_loss_5pct_day_so_far BETWEEN 1 AND landmark_day),
    CHECK (future_first_gain_2pct_day IS NULL OR future_first_gain_2pct_day BETWEEN landmark_day + 1 AND 5),
    CHECK (future_first_gain_5pct_day IS NULL OR future_first_gain_5pct_day BETWEEN landmark_day + 1 AND 5),
    CHECK (future_first_loss_5pct_day IS NULL OR future_first_loss_5pct_day BETWEEN landmark_day + 1 AND 5),
    CHECK (
        continuation_outcome IS NULL OR continuation_outcome IN (
            'loss_first', 'strong_first', 'same_session_ambiguous',
            'stagnant', 'neutral'
        )
    ),
    CHECK (
        CASE
            WHEN continuation_outcome IS NULL THEN
                stagnant_to_day5 IS NULL
                AND loss_first_to_day5 IS NULL
                AND strong_first_to_day5 IS NULL
                AND bad_to_day5 IS NULL
            WHEN continuation_outcome = 'same_session_ambiguous' THEN
                stagnant_to_day5 IS NULL
                AND loss_first_to_day5 IS NULL
                AND strong_first_to_day5 IS NULL
                AND bad_to_day5 IS NULL
            WHEN continuation_outcome = 'loss_first' THEN
                stagnant_to_day5 IS FALSE
                AND loss_first_to_day5 IS TRUE
                AND strong_first_to_day5 IS FALSE
                AND bad_to_day5 IS TRUE
            WHEN continuation_outcome = 'strong_first' THEN
                stagnant_to_day5 IS FALSE
                AND loss_first_to_day5 IS FALSE
                AND strong_first_to_day5 IS TRUE
                AND bad_to_day5 IS FALSE
            WHEN continuation_outcome = 'stagnant' THEN
                stagnant_to_day5 IS TRUE
                AND loss_first_to_day5 IS FALSE
                AND strong_first_to_day5 IS FALSE
                AND bad_to_day5 IS TRUE
            WHEN continuation_outcome = 'neutral' THEN
                stagnant_to_day5 IS FALSE
                AND loss_first_to_day5 IS FALSE
                AND strong_first_to_day5 IS FALSE
                AND bad_to_day5 IS FALSE
            ELSE FALSE
        END
    ),
    CHECK (analysis_split IN ('discovery', 'validation', 'diagnostic', 'holdout', 'purged')),
    CHECK (NOT same_continuity_segment OR landmark_observed),
    CHECK (NOT eligible_at_landmark OR (landmark_observed AND same_continuity_segment)),
    CHECK (
        prior_policy_cut_day IS NULL
        OR prior_policy_cut_day BETWEEN 1 AND landmark_day - 1
    ),
    CHECK (
        active_at_landmark = (
            eligible_at_landmark AND prior_policy_cut_day IS NULL
        )
    ),
    CHECK (include_final = (include_stagnation_filter AND include_loss_filter)),
    CHECK (
        cut_decision IN (
            'cut', 'hold', 'not_active', 'not_eligible', 'not_evaluable'
        )
    ),
    CHECK (
        (active_at_landmark AND include_final AND cut_decision = 'hold')
        OR
        (active_at_landmark AND NOT include_final AND cut_decision = 'cut')
        OR
        (NOT active_at_landmark
         AND NOT include_stagnation_filter
         AND NOT include_loss_filter
         AND NOT include_final
         AND (
             (prior_policy_cut_day IS NOT NULL AND cut_decision = 'not_active')
             OR
             (prior_policy_cut_day IS NULL
              AND NOT eligible_at_landmark
              AND cut_decision IN ('not_eligible', 'not_evaluable'))
         ))
    ),
    CHECK (
        (cut_decision = 'cut' AND NULLIF(TRIM(cut_reason), '') IS NOT NULL)
        OR
        (cut_decision <> 'cut' AND cut_reason IS NULL)
    )
);

SELECT create_hypertable(
    'stock_analyser_filter_research_early_cut_results',
    'signal_date',
    chunk_time_interval => INTERVAL '365 days',
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_safr_early_cut_identity
    ON stock_analyser_filter_research_early_cut_results
    (symbol, exchange, cik, signal_date DESC, landmark_day);

CREATE INDEX IF NOT EXISTS idx_safr_early_cut_decision
    ON stock_analyser_filter_research_early_cut_results
    (analysis_split, cut_decision, landmark_day, signal_date DESC);

CREATE TABLE IF NOT EXISTS stock_analyser_filter_research_rule_results (
    result_id                          BIGSERIAL PRIMARY KEY,
    rule_id                            TEXT NOT NULL,
    result_kind                        TEXT NOT NULL,
    decision_family                    TEXT NOT NULL,
    objective                          TEXT NOT NULL,
    protected_outcome                  TEXT NOT NULL,
    landmark_day                       SMALLINT,
    feature_group                      TEXT NOT NULL,
    feature_name                       TEXT,
    operator                           TEXT,
    quantile_value                     NUMERIC(30,10),
    threshold_value                    NUMERIC(30,10),
    rule_text                          TEXT NOT NULL,
    selection_order                    SMALLINT,
    evaluation_scope                   TEXT NOT NULL,
    scope_year                         SMALLINT,
    period_start                       DATE,
    period_end                         DATE,
    threshold_fit_end_date             DATE,
    is_selected                        BOOLEAN NOT NULL,
    is_final_filter                    BOOLEAN NOT NULL,
    passes_holdout                     BOOLEAN,
    passes_development_gates           BOOLEAN,
    passes_stability_gates             BOOLEAN,
    component_count                    SMALLINT NOT NULL,
    population_count                   INTEGER NOT NULL,
    sample_count                       INTEGER NOT NULL,
    unlabeled_count                    INTEGER NOT NULL,
    matched_count                      INTEGER NOT NULL,
    matched_labeled_count              INTEGER NOT NULL,
    matched_unlabeled_count            INTEGER NOT NULL,
    objective_count                    INTEGER NOT NULL,
    protected_count                    INTEGER NOT NULL,
    matched_objective_count            INTEGER NOT NULL,
    matched_protected_count            INTEGER NOT NULL,
    label_coverage_rate                NUMERIC(18,8),
    matched_label_coverage_rate        NUMERIC(18,8),
    match_rate                         NUMERIC(18,8),
    population_objective_rate          NUMERIC(18,8),
    matched_objective_rate             NUMERIC(18,8),
    objective_capture_rate             NUMERIC(18,8),
    objective_lift                     NUMERIC(18,8),
    protected_rejection_rate           NUMERIC(18,8),
    protected_retention_rate           NUMERIC(18,8),
    retained_objective_rate            NUMERIC(18,8),
    retained_protected_rate            NUMERIC(18,8),
    eligible_fold_count                INTEGER NOT NULL,
    positive_lift_fold_count           INTEGER NOT NULL,
    positive_lift_fold_fraction        NUMERIC(18,8),
    median_fold_objective_lift         NUMERIC(18,8),
    min_fold_objective_lift            NUMERIC(18,8),
    min_fold_protected_retention_rate  NUMERIC(18,8),
    max_fold_match_rate                NUMERIC(18,8),
    selection_score                    NUMERIC(18,8),

    CHECK (result_kind IN ('baseline', 'candidate_rule', 'selected_filter')),
    CHECK (decision_family IN ('entry_filter', 'early_cut')),
    CHECK (
        objective IN (
            'weak_5d', 'loss_first_5d', 'stagnant_to_day5',
            'loss_first_to_day5', 'bad_to_day5'
        )
    ),
    CHECK (protected_outcome IN ('strong_first_5d', 'strong_first_to_day5')),
    CHECK (
        (decision_family = 'entry_filter'
         AND landmark_day IS NULL
         AND objective IN ('weak_5d', 'loss_first_5d')
         AND protected_outcome = 'strong_first_5d')
        OR
        (decision_family = 'early_cut'
         AND landmark_day BETWEEN 1 AND 3
         AND (
             objective IN ('stagnant_to_day5', 'loss_first_to_day5')
             OR (objective = 'bad_to_day5' AND landmark_day = 1)
         )
         AND protected_outcome = 'strong_first_to_day5')
    ),
    CHECK (feature_group IN ('none', 'A', 'B', 'C', 'D', 'E', 'F', 'M', 'multiple')),
    CHECK (operator IS NULL OR operator IN ('le', 'ge')),
    CHECK (quantile_value IS NULL OR quantile_value BETWEEN 0 AND 1),
    CHECK (
        evaluation_scope IN (
            'development', 'discovery', 'validation', 'diagnostic', 'holdout',
            'all_signals', 'calendar_year', 'walk_forward_year',
            'walk_forward_pooled'
        )
    ),
    CHECK (
        (evaluation_scope IN ('calendar_year', 'walk_forward_year') AND scope_year IS NOT NULL)
        OR
        (evaluation_scope NOT IN ('calendar_year', 'walk_forward_year') AND scope_year IS NULL)
    ),
    CHECK (period_end IS NULL OR period_start IS NOT NULL),
    CHECK (period_end IS NULL OR period_end >= period_start),
    CHECK (threshold_fit_end_date IS NULL OR period_end IS NULL OR threshold_fit_end_date <= period_end),
    CHECK (selection_order IS NULL OR selection_order BETWEEN 1 AND 2),
    CHECK (NOT is_final_filter OR is_selected),
    CHECK (component_count >= 0),
    CHECK (population_count >= 0),
    CHECK (sample_count BETWEEN 0 AND population_count),
    CHECK (unlabeled_count = population_count - sample_count),
    CHECK (matched_count BETWEEN 0 AND population_count),
    CHECK (matched_labeled_count BETWEEN 0 AND sample_count),
    CHECK (matched_unlabeled_count BETWEEN 0 AND unlabeled_count),
    CHECK (matched_count = matched_labeled_count + matched_unlabeled_count),
    CHECK (objective_count BETWEEN 0 AND sample_count),
    CHECK (protected_count BETWEEN 0 AND sample_count),
    CHECK (matched_objective_count BETWEEN 0 AND objective_count),
    CHECK (matched_protected_count BETWEEN 0 AND protected_count),
    CHECK (matched_objective_count <= matched_labeled_count),
    CHECK (matched_protected_count <= matched_labeled_count),
    CHECK (objective_count + protected_count <= sample_count),
    CHECK (matched_objective_count + matched_protected_count <= matched_labeled_count),
    CHECK (eligible_fold_count >= 0),
    CHECK (positive_lift_fold_count BETWEEN 0 AND eligible_fold_count),
    CHECK (label_coverage_rate IS NULL OR label_coverage_rate BETWEEN 0 AND 1),
    CHECK (matched_label_coverage_rate IS NULL OR matched_label_coverage_rate BETWEEN 0 AND 1),
    CHECK (match_rate IS NULL OR match_rate BETWEEN 0 AND 1),
    CHECK (population_objective_rate IS NULL OR population_objective_rate BETWEEN 0 AND 1),
    CHECK (matched_objective_rate IS NULL OR matched_objective_rate BETWEEN 0 AND 1),
    CHECK (objective_capture_rate IS NULL OR objective_capture_rate BETWEEN 0 AND 1),
    CHECK (objective_lift IS NULL OR objective_lift >= 0),
    CHECK (protected_rejection_rate IS NULL OR protected_rejection_rate BETWEEN 0 AND 1),
    CHECK (protected_retention_rate IS NULL OR protected_retention_rate BETWEEN 0 AND 1),
    CHECK (retained_objective_rate IS NULL OR retained_objective_rate BETWEEN 0 AND 1),
    CHECK (retained_protected_rate IS NULL OR retained_protected_rate BETWEEN 0 AND 1),
    CHECK (positive_lift_fold_fraction IS NULL OR positive_lift_fold_fraction BETWEEN 0 AND 1),
    CHECK (median_fold_objective_lift IS NULL OR median_fold_objective_lift >= 0),
    CHECK (min_fold_objective_lift IS NULL OR min_fold_objective_lift >= 0),
    CHECK (min_fold_protected_retention_rate IS NULL OR min_fold_protected_retention_rate BETWEEN 0 AND 1),
    CHECK (max_fold_match_rate IS NULL OR max_fold_match_rate BETWEEN 0 AND 1),
    CHECK (
        evaluation_scope NOT IN ('walk_forward_year', 'diagnostic', 'holdout')
        OR threshold_fit_end_date IS NULL
        OR (period_start IS NOT NULL AND threshold_fit_end_date < period_start)
    )
);

CREATE INDEX IF NOT EXISTS idx_safr_rule_recommended
    ON stock_analyser_filter_research_rule_results
    (decision_family, is_final_filter, is_selected, evaluation_scope, result_kind);

CREATE INDEX IF NOT EXISTS idx_safr_rule_feature
    ON stock_analyser_filter_research_rule_results
    (decision_family, landmark_day, feature_group, feature_name, evaluation_scope);

GRANT SELECT, INSERT
    ON stock_analyser_filter_research_signal_results TO "market-data-account";
GRANT SELECT, INSERT
    ON stock_analyser_filter_research_early_cut_results TO "market-data-account";
GRANT SELECT, INSERT
    ON stock_analyser_filter_research_rule_results TO "market-data-account";
GRANT USAGE, SELECT
    ON SEQUENCE stock_analyser_filter_research_rule_results_result_id_seq
    TO "market-data-account";

COMMENT ON TABLE stock_analyser_filter_research_signal_results IS
    'One continuity-safe false-to-true trend-template event per stock, with causal entry features, forward outcomes and entry-filter decisions.';
COMMENT ON TABLE stock_analyser_filter_research_early_cut_results IS
    'D+1, D+2 and D+3 landmark observations for each signal, with point-in-time features, landmark-relative continuation outcomes and sequential early-cut decisions.';
COMMENT ON TABLE stock_analyser_filter_research_rule_results IS
    'Entry-filter and early-cut rule candidates, stability gates and out-of-sample evaluations.';
