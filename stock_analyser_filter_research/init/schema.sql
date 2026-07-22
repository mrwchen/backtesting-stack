-- Service-owned schema for stock-analyser filter and position-management research.
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
    adjusted_volume_vs_sma7_prior_ratio          NUMERIC(30,8),
    adjusted_volume_vs_sma14_prior_ratio         NUMERIC(30,8),
    adjusted_volume_sma50_prior                  NUMERIC(30,8),
    adjusted_volume_vs_sma50_prior_ratio         NUMERIC(30,8),
    adjusted_volume_vs_sma100_prior_ratio        NUMERIC(30,8),
    daily_traded_notional_usd                    NUMERIC(24,2),
    daily_traded_notional_sma21_prior_usd        NUMERIC(30,8),
    daily_traded_notional_vs_sma21_prior_ratio   NUMERIC(30,8),
    daily_traded_notional_vs_sma7_prior_ratio    NUMERIC(30,8),
    daily_traded_notional_vs_sma14_prior_ratio   NUMERIC(30,8),
    daily_traded_notional_sma50_prior_usd        NUMERIC(30,8),
    daily_traded_notional_vs_sma50_prior_ratio   NUMERIC(30,8),
    daily_traded_notional_vs_sma100_prior_ratio  NUMERIC(30,8),
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

    prior_adjusted_close                         NUMERIC(20,8),
    prior_return_42d_pct                         NUMERIC(20,8),
    prior_return_63d_pct                         NUMERIC(20,8),
    prior_return_126d_pct                        NUMERIC(20,8),
    prior_return_252d_pct                        NUMERIC(20,8),
    prior_return_acceleration_21d_pct_points     NUMERIC(20,8),
    prior_daily_return_std_21d_pct               NUMERIC(20,8),
    prior_daily_return_std_63d_pct               NUMERIC(20,8),
    prior_downside_return_std_21d_pct            NUMERIC(20,8),
    prior_max_drawdown_63d_pct                   NUMERIC(20,8),
    prior_max_drawdown_126d_pct                  NUMERIC(20,8),
    prior_atr_5d_pct                             NUMERIC(20,8),
    prior_atr_21d_pct                            NUMERIC(20,8),
    prior_atr_5_vs21_ratio                       NUMERIC(20,8),
    prior_atr_10_vs21_ratio                      NUMERIC(20,8),
    prior_close_vs_126d_high_pct                 NUMERIC(20,8),
    prior_close_vs_252d_high_pct                 NUMERIC(20,8),
    prior_volume_sma5_vs50_ratio                 NUMERIC(20,8),
    prior_volume_sma10_vs50_ratio                NUMERIC(20,8),
    prior_volume_sma21_vs50_ratio                NUMERIC(20,8),
    prior_notional_sma5_vs50_ratio               NUMERIC(20,8),
    prior_notional_sma10_vs50_ratio              NUMERIC(20,8),
    prior_notional_sma21_vs50_ratio              NUMERIC(20,8),
    prior_up_down_volume_ratio21                 NUMERIC(20,8),
    prior_up_down_notional_ratio21               NUMERIC(20,8),
    prior_volume_dryup_share10                   NUMERIC(20,8),
    prior_volume_dryup_share20                   NUMERIC(20,8),
    prior_obv_slope_20                           NUMERIC(20,8),
    prior_accumulation_day_count_20              NUMERIC(20,8),
    prior_high_volume_down_day_count_20          NUMERIC(20,8),
    prior_base_width_10_pct                      NUMERIC(20,8),
    prior_base_width_40_pct                      NUMERIC(20,8),
    prior_base_width_63_pct                      NUMERIC(20,8),
    prior_tight_close_range_5_pct                NUMERIC(20,8),
    prior_tight_close_range_10_pct               NUMERIC(20,8),
    prior_tight_close_range_15_pct               NUMERIC(20,8),
    prior_range_compression_5_vs20_ratio         NUMERIC(20,8),
    ma50_slope_21d_pct                           NUMERIC(20,8),
    ma150_slope_21d_pct                          NUMERIC(20,8),
    ma200_slope_63d_pct                          NUMERIC(20,8),
    prior_overhead_supply_share63                NUMERIC(20,8),
    prior_high_test_count_20                     NUMERIC(20,8),
    prior_high_slope_20_pct_per_session          NUMERIC(20,8),
    prior_low_slope_20_pct_per_session           NUMERIC(20,8),
    prior_contraction_count_40                   NUMERIC(20,8),
    prior_return_efficiency_63                   NUMERIC(20,8),
    prior_rs_rating_change_21d                   NUMERIC(20,8),
    prior_history_sessions                       INTEGER,
    signal_undercut_reclaim_10                   NUMERIC(20,8),
    signal_volume_vs_prior_10d_max_down_volume_ratio NUMERIC(20,8),

    pattern_flat_base_setup_score                       NUMERIC(20,8),
    pattern_flat_base_trigger_score                     NUMERIC(20,8),
    pattern_flat_base_score_10d                         NUMERIC(20,8),
    pattern_flat_base_score_15d                         NUMERIC(20,8),
    pattern_flat_base_score_20d                         NUMERIC(20,8),
    pattern_flat_base_score_30d                         NUMERIC(20,8),
    pattern_flat_base_score_40d                         NUMERIC(20,8),
    pattern_flat_base_score_63d                         NUMERIC(20,8),
    pattern_ordered_uptrend_setup_score                 NUMERIC(20,8),
    pattern_ordered_uptrend_trigger_score               NUMERIC(20,8),
    pattern_ordered_uptrend_score_10d                   NUMERIC(20,8),
    pattern_ordered_uptrend_score_15d                   NUMERIC(20,8),
    pattern_ordered_uptrend_score_20d                   NUMERIC(20,8),
    pattern_ordered_uptrend_score_30d                   NUMERIC(20,8),
    pattern_ordered_uptrend_score_40d                   NUMERIC(20,8),
    pattern_ordered_uptrend_score_63d                   NUMERIC(20,8),
    pattern_pullback_setup_score                        NUMERIC(20,8),
    pattern_pullback_trigger_score                      NUMERIC(20,8),
    pattern_pullback_score_10d                          NUMERIC(20,8),
    pattern_pullback_score_15d                          NUMERIC(20,8),
    pattern_pullback_score_20d                          NUMERIC(20,8),
    pattern_pullback_score_30d                          NUMERIC(20,8),
    pattern_pullback_score_40d                          NUMERIC(20,8),
    pattern_pullback_score_63d                          NUMERIC(20,8),
    pattern_v_recovery_setup_score                      NUMERIC(20,8),
    pattern_v_recovery_trigger_score                    NUMERIC(20,8),
    pattern_v_recovery_score_20d                        NUMERIC(20,8),
    pattern_v_recovery_score_30d                        NUMERIC(20,8),
    pattern_v_recovery_score_40d                        NUMERIC(20,8),
    pattern_v_recovery_score_63d                        NUMERIC(20,8),
    pattern_v_recovery_score_126d                       NUMERIC(20,8),
    pattern_volume_dryup_breakout_setup_score           NUMERIC(20,8),
    pattern_volume_dryup_breakout_trigger_score         NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_10d             NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_15d             NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_20d             NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_30d             NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_40d             NUMERIC(20,8),
    pattern_volume_dryup_breakout_score_63d             NUMERIC(20,8),
    pattern_distribution_top_setup_score                NUMERIC(20,8),
    pattern_distribution_top_trigger_score              NUMERIC(20,8),
    pattern_distribution_top_score_10d                  NUMERIC(20,8),
    pattern_distribution_top_score_15d                  NUMERIC(20,8),
    pattern_distribution_top_score_20d                  NUMERIC(20,8),
    pattern_distribution_top_score_30d                  NUMERIC(20,8),
    pattern_distribution_top_score_40d                  NUMERIC(20,8),
    pattern_distribution_top_score_63d                  NUMERIC(20,8),
    pattern_vcp_setup_score                             NUMERIC(20,8),
    pattern_vcp_trigger_score                           NUMERIC(20,8),
    pattern_vcp_score_20d                               NUMERIC(20,8),
    pattern_vcp_score_30d                               NUMERIC(20,8),
    pattern_vcp_score_40d                               NUMERIC(20,8),
    pattern_vcp_score_63d                               NUMERIC(20,8),
    pattern_vcp_score_126d                              NUMERIC(20,8),
    pattern_cup_with_handle_setup_score                 NUMERIC(20,8),
    pattern_cup_with_handle_trigger_score               NUMERIC(20,8),
    pattern_cup_with_handle_score_63d                   NUMERIC(20,8),
    pattern_cup_with_handle_score_126d                  NUMERIC(20,8),
    pattern_high_tight_flag_setup_score                 NUMERIC(20,8),
    pattern_high_tight_flag_trigger_score               NUMERIC(20,8),
    pattern_high_tight_flag_score_20d                   NUMERIC(20,8),
    pattern_high_tight_flag_score_30d                   NUMERIC(20,8),
    pattern_high_tight_flag_score_40d                   NUMERIC(20,8),
    pattern_high_tight_flag_score_63d                   NUMERIC(20,8),
    pattern_pullback_notional_setup_score               NUMERIC(20,8),
    pattern_pullback_notional_trigger_score             NUMERIC(20,8),
    pattern_pullback_notional_score_10d                 NUMERIC(20,8),
    pattern_pullback_notional_score_15d                 NUMERIC(20,8),
    pattern_pullback_notional_score_20d                 NUMERIC(20,8),
    pattern_pullback_notional_score_30d                 NUMERIC(20,8),
    pattern_pullback_notional_score_40d                 NUMERIC(20,8),
    pattern_pullback_notional_score_63d                 NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_setup_score  NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_trigger_score NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_10d    NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_15d    NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_20d    NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_30d    NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_40d    NUMERIC(20,8),
    pattern_volume_dryup_breakout_notional_score_63d    NUMERIC(20,8),
    pattern_distribution_top_notional_setup_score       NUMERIC(20,8),
    pattern_distribution_top_notional_trigger_score     NUMERIC(20,8),
    pattern_distribution_top_notional_score_10d         NUMERIC(20,8),
    pattern_distribution_top_notional_score_15d         NUMERIC(20,8),
    pattern_distribution_top_notional_score_20d         NUMERIC(20,8),
    pattern_distribution_top_notional_score_30d         NUMERIC(20,8),
    pattern_distribution_top_notional_score_40d         NUMERIC(20,8),
    pattern_distribution_top_notional_score_63d         NUMERIC(20,8),
    pattern_vcp_notional_setup_score                    NUMERIC(20,8),
    pattern_vcp_notional_trigger_score                  NUMERIC(20,8),
    pattern_vcp_notional_score_20d                      NUMERIC(20,8),
    pattern_vcp_notional_score_30d                      NUMERIC(20,8),
    pattern_vcp_notional_score_40d                      NUMERIC(20,8),
    pattern_vcp_notional_score_63d                      NUMERIC(20,8),
    pattern_vcp_notional_score_126d                     NUMERIC(20,8),
    pattern_cup_with_handle_notional_setup_score        NUMERIC(20,8),
    pattern_cup_with_handle_notional_trigger_score      NUMERIC(20,8),
    pattern_cup_with_handle_notional_score_63d          NUMERIC(20,8),
    pattern_cup_with_handle_notional_score_126d         NUMERIC(20,8),
    pattern_high_tight_flag_notional_setup_score        NUMERIC(20,8),
    pattern_high_tight_flag_notional_trigger_score      NUMERIC(20,8),
    pattern_high_tight_flag_notional_score_20d          NUMERIC(20,8),
    pattern_high_tight_flag_notional_score_30d          NUMERIC(20,8),
    pattern_high_tight_flag_notional_score_40d          NUMERIC(20,8),
    pattern_high_tight_flag_notional_score_63d          NUMERIC(20,8),

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
    fundamental_operating_cashflow_margin_ttm_ratio NUMERIC(20,8),
    fundamental_rd_to_revenue_ttm_ratio          NUMERIC(20,8),
    fundamental_sga_to_revenue_ttm_ratio         NUMERIC(20,8),
    fundamental_capex_to_revenue_ttm_ratio       NUMERIC(20,8),
    fundamental_da_to_revenue_ttm_ratio          NUMERIC(20,8),
    fundamental_roe_ttm_ratio                    NUMERIC(20,8),
    fundamental_roa_ttm_ratio                    NUMERIC(20,8),
    fundamental_cash_conversion_ttm_ratio        NUMERIC(20,8),
    fundamental_fcf_conversion_ttm_ratio         NUMERIC(20,8),
    fundamental_fcf_sbc_adjusted_conversion_ttm_ratio NUMERIC(20,8),
    fundamental_interest_coverage_ttm_ratio      NUMERIC(20,8),
    fundamental_debt_to_assets_ratio             NUMERIC(20,8),
    fundamental_net_debt_to_assets_ratio         NUMERIC(20,8),
    fundamental_debt_to_operating_income_ttm_ratio NUMERIC(20,8),
    fundamental_net_debt_to_fcf_ttm_ratio        NUMERIC(20,8),
    fundamental_quick_ratio                      NUMERIC(20,8),
    fundamental_working_capital_to_assets_ratio  NUMERIC(20,8),
    fundamental_goodwill_intangibles_to_assets_ratio NUMERIC(20,8),
    fundamental_inventory_to_revenue_ttm_ratio   NUMERIC(20,8),
    fundamental_receivables_to_revenue_ttm_ratio NUMERIC(20,8),
    fundamental_payables_to_revenue_ttm_ratio    NUMERIC(20,8),
    fundamental_asset_turnover_ttm_ratio         NUMERIC(20,8),
    fundamental_diluted_share_pressure_ratio     NUMERIC(20,8),
    fundamental_buyback_to_revenue_ttm_ratio     NUMERIC(20,8),
    fundamental_sec_shares_change_1y_ratio       NUMERIC(20,8),
    fundamental_quarter_filing_age_days          NUMERIC(20,8),
    fundamental_quarter_age_days                 NUMERIC(20,8),
    fundamental_quarterly_revenue_yoy_growth_ratio NUMERIC(20,8),
    fundamental_quarterly_eps_yoy_change_ratio   NUMERIC(20,8),
    fundamental_quarterly_eps_yoy_growth_ratio   NUMERIC(20,8),
    fundamental_quarterly_loss_to_profit         NUMERIC(20,8),
    fundamental_quarterly_revenue_growth_acceleration NUMERIC(20,8),
    fundamental_quarterly_eps_growth_acceleration NUMERIC(20,8),
    fundamental_quarterly_revenue_sequential_growth_ratio NUMERIC(20,8),
    fundamental_quarterly_eps_sequential_change_ratio NUMERIC(20,8),
    fundamental_revenue_growth_streak_4q         NUMERIC(20,8),
    fundamental_eps_growth_streak_4q             NUMERIC(20,8),
    fundamental_quarterly_operating_margin_ratio NUMERIC(20,8),
    fundamental_quarterly_operating_margin_yoy_change NUMERIC(20,8),
    fundamental_quarterly_net_margin_ratio       NUMERIC(20,8),
    fundamental_quarterly_net_margin_yoy_change  NUMERIC(20,8),
    fundamental_quarterly_operating_margin_acceleration NUMERIC(20,8),
    fundamental_quarterly_net_margin_acceleration NUMERIC(20,8),

    earnings_event_age_days                      NUMERIC(20,8),
    earnings_event_on_signal_day                 NUMERIC(20,8),
    earnings_event_within_5d                     NUMERIC(20,8),
    earnings_event_within_21d                    NUMERIC(20,8),

    market_cap_usd                              BIGINT,
    log_market_cap_usd                          NUMERIC(20,8),
    market_cap_shares_staleness_days            INTEGER,

    signal_adjusted_open                        NUMERIC(20,8),
    signal_gap_pct                              NUMERIC(20,8),
    signal_intraday_return_pct                  NUMERIC(20,8),
    shares_outstanding                          BIGINT,
    log_shares_outstanding                      NUMERIC(20,8),
    signal_turnover_ratio                       NUMERIC(20,8),

    cross_sectional_rs_21d_pct_rank             NUMERIC(20,8),
    cross_sectional_rs_63d_pct_rank             NUMERIC(20,8),
    cross_sectional_rs_126d_pct_rank            NUMERIC(20,8),
    cross_sectional_rs_252d_pct_rank            NUMERIC(20,8),
    market_breadth_above_ma50_ratio             NUMERIC(20,8),
    market_breadth_above_ma150_ratio            NUMERIC(20,8),
    market_breadth_above_ma200_ratio            NUMERIC(20,8),
    market_breadth_trend_template_ratio         NUMERIC(20,8),
    market_breadth_rs70_ratio                   NUMERIC(20,8),
    market_breadth_rs90_ratio                   NUMERIC(20,8),
    market_advancer_ratio                       NUMERIC(20,8),
    market_median_daily_return_pct              NUMERIC(20,8),
    market_breadth_above_ma50_change_5d         NUMERIC(20,8),
    market_breadth_above_ma200_change_21d       NUMERIC(20,8),
    market_spy_prior_return_5d_pct              NUMERIC(20,8),
    market_spy_prior_return_21d_pct             NUMERIC(20,8),
    market_spy_prior_return_63d_pct             NUMERIC(20,8),
    market_qqq_prior_return_5d_pct              NUMERIC(20,8),
    market_qqq_prior_return_21d_pct             NUMERIC(20,8),
    market_qqq_prior_return_63d_pct             NUMERIC(20,8),
    market_iwm_prior_return_5d_pct              NUMERIC(20,8),
    market_iwm_prior_return_21d_pct             NUMERIC(20,8),
    market_iwm_prior_return_63d_pct             NUMERIC(20,8),
    market_dia_prior_return_21d_pct             NUMERIC(20,8),
    relative_return_vs_spy_21d_pct_points       NUMERIC(20,8),
    relative_return_vs_spy_63d_pct_points       NUMERIC(20,8),
    relative_return_vs_qqq_21d_pct_points       NUMERIC(20,8),
    relative_return_vs_qqq_63d_pct_points       NUMERIC(20,8),
    relative_return_vs_iwm_21d_pct_points       NUMERIC(20,8),
    relative_return_vs_iwm_63d_pct_points       NUMERIC(20,8),
    market_vix_level                            NUMERIC(20,8),
    market_vxn_level                            NUMERIC(20,8),
    market_vvix_level                           NUMERIC(20,8),
    market_skew_level                           NUMERIC(20,8),
    market_vix9d_to_vix_ratio                   NUMERIC(20,8),
    market_vix_to_vix3m_ratio                   NUMERIC(20,8),

    current_taxonomy_backcast_industry          TEXT,
    current_taxonomy_backcast_category          TEXT,
    current_taxonomy_backcast_subcategory       TEXT,
    current_taxonomy_backcast_industry_group_member_count INTEGER,
    current_taxonomy_backcast_industry_group_median_return_21d_pct NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_median_return_63d_pct NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_median_return_126d_pct NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_median_return_252d_pct NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_return_21d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_return_63d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_return_126d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_return_252d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_rs_raw_median NUMERIC(20,8),
    current_taxonomy_backcast_industry_group_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_stock_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma50_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma200_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma50_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma50_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma200_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_industry_above_ma200_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_industry_new_52w_high_count INTEGER,
    current_taxonomy_backcast_industry_new_52w_high_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_new_52w_high_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_industry_new_52w_high_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_industry_rs70_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_rs90_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_trend_template_ratio NUMERIC(20,8),
    current_taxonomy_backcast_industry_leadership_breadth_score NUMERIC(20,8),
    current_taxonomy_backcast_industry_leadership_breadth_score_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_industry_leadership_breadth_score_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_industry_new_8of8_signal_count INTEGER,
    current_taxonomy_backcast_industry_new_8of8_signal_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_member_count INTEGER,
    current_taxonomy_backcast_category_path_group_median_return_21d_pct NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_median_return_63d_pct NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_median_return_126d_pct NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_median_return_252d_pct NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_return_21d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_return_63d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_return_126d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_return_252d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_rs_raw_median NUMERIC(20,8),
    current_taxonomy_backcast_category_path_group_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_stock_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma50_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma200_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma50_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma50_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma200_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_above_ma200_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_new_52w_high_count INTEGER,
    current_taxonomy_backcast_category_path_new_52w_high_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_new_52w_high_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_new_52w_high_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_rs70_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_rs90_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_trend_template_ratio NUMERIC(20,8),
    current_taxonomy_backcast_category_path_leadership_breadth_score NUMERIC(20,8),
    current_taxonomy_backcast_category_path_leadership_breadth_score_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_leadership_breadth_score_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_category_path_new_8of8_signal_count INTEGER,
    current_taxonomy_backcast_category_path_new_8of8_signal_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_member_count INTEGER,
    current_taxonomy_backcast_subcategory_path_group_median_return_21d_pct NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_median_return_63d_pct NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_median_return_126d_pct NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_median_return_252d_pct NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_return_21d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_return_63d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_return_126d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_return_252d_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_rs_raw_median NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_group_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_stock_rs_raw_pct_rank NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma50_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma200_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma50_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma50_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma200_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_above_ma200_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_new_52w_high_count INTEGER,
    current_taxonomy_backcast_subcategory_path_new_52w_high_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_new_52w_high_ratio_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_new_52w_high_ratio_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_rs70_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_rs90_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_trend_template_ratio NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_leadership_breadth_score NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_leadership_breadth_score_change_5d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_leadership_breadth_score_change_21d NUMERIC(20,8),
    current_taxonomy_backcast_subcategory_path_new_8of8_signal_count INTEGER,
    current_taxonomy_backcast_subcategory_path_new_8of8_signal_ratio NUMERIC(20,8),

    forward_5d_max_gain_pct                      NUMERIC(20,8),
    forward_5d_max_loss_pct                      NUMERIC(20,8),
    forward_10d_max_gain_pct                     NUMERIC(20,8),
    forward_10d_max_loss_pct                     NUMERIC(20,8),
    forward_20d_max_gain_pct                     NUMERIC(20,8),
    forward_20d_max_loss_pct                     NUMERIC(20,8),
    forward_30d_max_gain_pct                     NUMERIC(20,8),
    forward_30d_max_loss_pct                     NUMERIC(20,8),
    forward_40d_max_gain_pct                     NUMERIC(20,8),
    forward_40d_max_loss_pct                     NUMERIC(20,8),
    forward_60d_max_gain_pct                     NUMERIC(20,8),
    forward_60d_max_loss_pct                     NUMERIC(20,8),
    forward_90d_max_gain_pct                     NUMERIC(20,8),
    forward_90d_max_loss_pct                     NUMERIC(20,8),
    forward_5d_label_end_date                    DATE,
    forward_10d_label_end_date                   DATE,
    forward_20d_label_end_date                   DATE,
    forward_30d_label_end_date                   DATE,
    forward_40d_label_end_date                   DATE,
    forward_60d_label_end_date                   DATE,
    forward_90d_label_end_date                   DATE,
    terminal_close_return_5d_pct                 NUMERIC(20,8),
    terminal_close_return_10d_pct                NUMERIC(20,8),
    terminal_close_return_20d_pct                NUMERIC(20,8),
    terminal_close_return_30d_pct                NUMERIC(20,8),
    terminal_close_return_40d_pct                NUMERIC(20,8),
    terminal_close_return_60d_pct                NUMERIC(20,8),
    terminal_close_return_90d_pct                NUMERIC(20,8),
    first_gain_2pct_day                          SMALLINT,
    first_gain_1pct_day                          SMALLINT,
    first_gain_3pct_day                          SMALLINT,
    first_gain_5pct_day                          SMALLINT,
    first_loss_5pct_day                          SMALLINT,
    first_loss_10pct_day                         SMALLINT,
    gain_loss_order_5d                           TEXT,
    weak_5d                                      BOOLEAN,
    strong_5d                                    BOOLEAN,
    deep_loss_5d                                 BOOLEAN,
    bad_5d                                       BOOLEAN,
    loss_first_5d                                BOOLEAN,
    strong_first_5d                              BOOLEAN,
    terminal_stagnant_5d                         BOOLEAN,
    terminal_winner_5d                           BOOLEAN,
    stagnant_5d                                  BOOLEAN,
    hard_stop_10pct_5d                           BOOLEAN,
    terminal_nonpositive_20d                     BOOLEAN,
    terminal_winner_20d                          BOOLEAN,
    terminal_nonpositive_30d                     BOOLEAN,
    terminal_winner_30d                          BOOLEAN,
    runner_60d                                   BOOLEAN,
    runner_90d                                   BOOLEAN,
    mfe_to_abs_mae_5d_ratio                      NUMERIC(20,8),
    terminal_return_to_mfe_5d_ratio              NUMERIC(20,8),
    late_strong_10d                              BOOLEAN,
    late_strong_20d                              BOOLEAN,

    analysis_split                               TEXT NOT NULL,
    include_weak_filter                          BOOLEAN NOT NULL,
    include_loss_first_filter                    BOOLEAN NOT NULL,
    include_final                                BOOLEAN NOT NULL,
    strong_confirmation                          BOOLEAN NOT NULL,
    weak_matched_rule_ids                        TEXT,
    loss_first_matched_rule_ids                  TEXT,
    matched_rule_ids                             TEXT,
    confirmation_matched_rule_ids                TEXT,
    confirmation_reason                          TEXT,
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
    CHECK (
        LEAST(
            pattern_flat_base_setup_score,
            pattern_flat_base_trigger_score,
            pattern_flat_base_score_10d,
            pattern_flat_base_score_15d,
            pattern_flat_base_score_20d,
            pattern_flat_base_score_30d,
            pattern_flat_base_score_40d,
            pattern_flat_base_score_63d,
            pattern_ordered_uptrend_setup_score,
            pattern_ordered_uptrend_trigger_score,
            pattern_ordered_uptrend_score_10d,
            pattern_ordered_uptrend_score_15d,
            pattern_ordered_uptrend_score_20d,
            pattern_ordered_uptrend_score_30d,
            pattern_ordered_uptrend_score_40d,
            pattern_ordered_uptrend_score_63d,
            pattern_pullback_setup_score,
            pattern_pullback_trigger_score,
            pattern_pullback_score_10d,
            pattern_pullback_score_15d,
            pattern_pullback_score_20d,
            pattern_pullback_score_30d,
            pattern_pullback_score_40d,
            pattern_pullback_score_63d,
            pattern_v_recovery_setup_score,
            pattern_v_recovery_trigger_score,
            pattern_v_recovery_score_20d,
            pattern_v_recovery_score_30d,
            pattern_v_recovery_score_40d,
            pattern_v_recovery_score_63d,
            pattern_v_recovery_score_126d,
            pattern_volume_dryup_breakout_setup_score,
            pattern_volume_dryup_breakout_trigger_score,
            pattern_volume_dryup_breakout_score_10d,
            pattern_volume_dryup_breakout_score_15d,
            pattern_volume_dryup_breakout_score_20d,
            pattern_volume_dryup_breakout_score_30d,
            pattern_volume_dryup_breakout_score_40d,
            pattern_volume_dryup_breakout_score_63d,
            pattern_distribution_top_setup_score,
            pattern_distribution_top_trigger_score,
            pattern_distribution_top_score_10d,
            pattern_distribution_top_score_15d,
            pattern_distribution_top_score_20d,
            pattern_distribution_top_score_30d,
            pattern_distribution_top_score_40d,
            pattern_distribution_top_score_63d,
            pattern_vcp_setup_score,
            pattern_vcp_trigger_score,
            pattern_vcp_score_20d,
            pattern_vcp_score_30d,
            pattern_vcp_score_40d,
            pattern_vcp_score_63d,
            pattern_vcp_score_126d,
            pattern_cup_with_handle_setup_score,
            pattern_cup_with_handle_trigger_score,
            pattern_cup_with_handle_score_63d,
            pattern_cup_with_handle_score_126d,
            pattern_high_tight_flag_setup_score,
            pattern_high_tight_flag_trigger_score,
            pattern_high_tight_flag_score_20d,
            pattern_high_tight_flag_score_30d,
            pattern_high_tight_flag_score_40d,
            pattern_high_tight_flag_score_63d,
            pattern_pullback_notional_setup_score,
            pattern_pullback_notional_trigger_score,
            pattern_pullback_notional_score_10d,
            pattern_pullback_notional_score_15d,
            pattern_pullback_notional_score_20d,
            pattern_pullback_notional_score_30d,
            pattern_pullback_notional_score_40d,
            pattern_pullback_notional_score_63d,
            pattern_volume_dryup_breakout_notional_setup_score,
            pattern_volume_dryup_breakout_notional_trigger_score,
            pattern_volume_dryup_breakout_notional_score_10d,
            pattern_volume_dryup_breakout_notional_score_15d,
            pattern_volume_dryup_breakout_notional_score_20d,
            pattern_volume_dryup_breakout_notional_score_30d,
            pattern_volume_dryup_breakout_notional_score_40d,
            pattern_volume_dryup_breakout_notional_score_63d,
            pattern_distribution_top_notional_setup_score,
            pattern_distribution_top_notional_trigger_score,
            pattern_distribution_top_notional_score_10d,
            pattern_distribution_top_notional_score_15d,
            pattern_distribution_top_notional_score_20d,
            pattern_distribution_top_notional_score_30d,
            pattern_distribution_top_notional_score_40d,
            pattern_distribution_top_notional_score_63d,
            pattern_vcp_notional_setup_score,
            pattern_vcp_notional_trigger_score,
            pattern_vcp_notional_score_20d,
            pattern_vcp_notional_score_30d,
            pattern_vcp_notional_score_40d,
            pattern_vcp_notional_score_63d,
            pattern_vcp_notional_score_126d,
            pattern_cup_with_handle_notional_setup_score,
            pattern_cup_with_handle_notional_trigger_score,
            pattern_cup_with_handle_notional_score_63d,
            pattern_cup_with_handle_notional_score_126d,
            pattern_high_tight_flag_notional_setup_score,
            pattern_high_tight_flag_notional_trigger_score,
            pattern_high_tight_flag_notional_score_20d,
            pattern_high_tight_flag_notional_score_30d,
            pattern_high_tight_flag_notional_score_40d,
            pattern_high_tight_flag_notional_score_63d
        ) >= 0
    ),
    CHECK (
        GREATEST(
            pattern_flat_base_setup_score,
            pattern_flat_base_trigger_score,
            pattern_flat_base_score_10d,
            pattern_flat_base_score_15d,
            pattern_flat_base_score_20d,
            pattern_flat_base_score_30d,
            pattern_flat_base_score_40d,
            pattern_flat_base_score_63d,
            pattern_ordered_uptrend_setup_score,
            pattern_ordered_uptrend_trigger_score,
            pattern_ordered_uptrend_score_10d,
            pattern_ordered_uptrend_score_15d,
            pattern_ordered_uptrend_score_20d,
            pattern_ordered_uptrend_score_30d,
            pattern_ordered_uptrend_score_40d,
            pattern_ordered_uptrend_score_63d,
            pattern_pullback_setup_score,
            pattern_pullback_trigger_score,
            pattern_pullback_score_10d,
            pattern_pullback_score_15d,
            pattern_pullback_score_20d,
            pattern_pullback_score_30d,
            pattern_pullback_score_40d,
            pattern_pullback_score_63d,
            pattern_v_recovery_setup_score,
            pattern_v_recovery_trigger_score,
            pattern_v_recovery_score_20d,
            pattern_v_recovery_score_30d,
            pattern_v_recovery_score_40d,
            pattern_v_recovery_score_63d,
            pattern_v_recovery_score_126d,
            pattern_volume_dryup_breakout_setup_score,
            pattern_volume_dryup_breakout_trigger_score,
            pattern_volume_dryup_breakout_score_10d,
            pattern_volume_dryup_breakout_score_15d,
            pattern_volume_dryup_breakout_score_20d,
            pattern_volume_dryup_breakout_score_30d,
            pattern_volume_dryup_breakout_score_40d,
            pattern_volume_dryup_breakout_score_63d,
            pattern_distribution_top_setup_score,
            pattern_distribution_top_trigger_score,
            pattern_distribution_top_score_10d,
            pattern_distribution_top_score_15d,
            pattern_distribution_top_score_20d,
            pattern_distribution_top_score_30d,
            pattern_distribution_top_score_40d,
            pattern_distribution_top_score_63d,
            pattern_vcp_setup_score,
            pattern_vcp_trigger_score,
            pattern_vcp_score_20d,
            pattern_vcp_score_30d,
            pattern_vcp_score_40d,
            pattern_vcp_score_63d,
            pattern_vcp_score_126d,
            pattern_cup_with_handle_setup_score,
            pattern_cup_with_handle_trigger_score,
            pattern_cup_with_handle_score_63d,
            pattern_cup_with_handle_score_126d,
            pattern_high_tight_flag_setup_score,
            pattern_high_tight_flag_trigger_score,
            pattern_high_tight_flag_score_20d,
            pattern_high_tight_flag_score_30d,
            pattern_high_tight_flag_score_40d,
            pattern_high_tight_flag_score_63d,
            pattern_pullback_notional_setup_score,
            pattern_pullback_notional_trigger_score,
            pattern_pullback_notional_score_10d,
            pattern_pullback_notional_score_15d,
            pattern_pullback_notional_score_20d,
            pattern_pullback_notional_score_30d,
            pattern_pullback_notional_score_40d,
            pattern_pullback_notional_score_63d,
            pattern_volume_dryup_breakout_notional_setup_score,
            pattern_volume_dryup_breakout_notional_trigger_score,
            pattern_volume_dryup_breakout_notional_score_10d,
            pattern_volume_dryup_breakout_notional_score_15d,
            pattern_volume_dryup_breakout_notional_score_20d,
            pattern_volume_dryup_breakout_notional_score_30d,
            pattern_volume_dryup_breakout_notional_score_40d,
            pattern_volume_dryup_breakout_notional_score_63d,
            pattern_distribution_top_notional_setup_score,
            pattern_distribution_top_notional_trigger_score,
            pattern_distribution_top_notional_score_10d,
            pattern_distribution_top_notional_score_15d,
            pattern_distribution_top_notional_score_20d,
            pattern_distribution_top_notional_score_30d,
            pattern_distribution_top_notional_score_40d,
            pattern_distribution_top_notional_score_63d,
            pattern_vcp_notional_setup_score,
            pattern_vcp_notional_trigger_score,
            pattern_vcp_notional_score_20d,
            pattern_vcp_notional_score_30d,
            pattern_vcp_notional_score_40d,
            pattern_vcp_notional_score_63d,
            pattern_vcp_notional_score_126d,
            pattern_cup_with_handle_notional_setup_score,
            pattern_cup_with_handle_notional_trigger_score,
            pattern_cup_with_handle_notional_score_63d,
            pattern_cup_with_handle_notional_score_126d,
            pattern_high_tight_flag_notional_setup_score,
            pattern_high_tight_flag_notional_trigger_score,
            pattern_high_tight_flag_notional_score_20d,
            pattern_high_tight_flag_notional_score_30d,
            pattern_high_tight_flag_notional_score_40d,
            pattern_high_tight_flag_notional_score_63d
        ) <= 100
    ),
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
        current_taxonomy_backcast_category IS NULL
        OR current_taxonomy_backcast_industry IS NOT NULL
    ),
    CHECK (
        current_taxonomy_backcast_subcategory IS NULL
        OR current_taxonomy_backcast_category IS NOT NULL
    ),
    CHECK (
        LEAST(
            current_taxonomy_backcast_industry_group_member_count,
            current_taxonomy_backcast_industry_new_52w_high_count,
            current_taxonomy_backcast_industry_new_8of8_signal_count,
            current_taxonomy_backcast_category_path_group_member_count,
            current_taxonomy_backcast_category_path_new_52w_high_count,
            current_taxonomy_backcast_category_path_new_8of8_signal_count,
            current_taxonomy_backcast_subcategory_path_group_member_count,
            current_taxonomy_backcast_subcategory_path_new_52w_high_count,
            current_taxonomy_backcast_subcategory_path_new_8of8_signal_count
        ) >= 0
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
    CHECK (forward_30d_max_gain_pct IS NULL OR forward_30d_max_gain_pct >= 0),
    CHECK (forward_30d_max_loss_pct IS NULL OR forward_30d_max_loss_pct <= 0),
    CHECK (forward_40d_max_gain_pct IS NULL OR forward_40d_max_gain_pct >= 0),
    CHECK (forward_40d_max_loss_pct IS NULL OR forward_40d_max_loss_pct <= 0),
    CHECK (forward_60d_max_gain_pct IS NULL OR forward_60d_max_gain_pct >= 0),
    CHECK (forward_60d_max_loss_pct IS NULL OR forward_60d_max_loss_pct <= 0),
    CHECK (forward_90d_max_gain_pct IS NULL OR forward_90d_max_gain_pct >= 0),
    CHECK (forward_90d_max_loss_pct IS NULL OR forward_90d_max_loss_pct <= 0),
    CHECK (forward_5d_label_end_date IS NULL OR forward_5d_label_end_date > signal_date),
    CHECK (forward_10d_label_end_date IS NULL OR forward_10d_label_end_date > signal_date),
    CHECK (forward_20d_label_end_date IS NULL OR forward_20d_label_end_date > signal_date),
    CHECK (forward_30d_label_end_date IS NULL OR forward_30d_label_end_date > signal_date),
    CHECK (forward_40d_label_end_date IS NULL OR forward_40d_label_end_date > signal_date),
    CHECK (forward_60d_label_end_date IS NULL OR forward_60d_label_end_date > signal_date),
    CHECK (forward_90d_label_end_date IS NULL OR forward_90d_label_end_date > signal_date),
    CHECK (first_gain_2pct_day IS NULL OR first_gain_2pct_day BETWEEN 1 AND 5),
    CHECK (first_gain_1pct_day IS NULL OR first_gain_1pct_day BETWEEN 1 AND 5),
    CHECK (first_gain_3pct_day IS NULL OR first_gain_3pct_day BETWEEN 1 AND 5),
    CHECK (first_gain_5pct_day IS NULL OR first_gain_5pct_day BETWEEN 1 AND 5),
    CHECK (first_loss_5pct_day IS NULL OR first_loss_5pct_day BETWEEN 1 AND 5),
    CHECK (first_loss_10pct_day IS NULL OR first_loss_10pct_day BETWEEN 1 AND 5),
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
    CHECK (
        (terminal_close_return_5d_pct IS NULL)
        = (terminal_stagnant_5d IS NULL)
    ),
    CHECK ((terminal_stagnant_5d IS NULL) = (terminal_winner_5d IS NULL)),
    CHECK ((stagnant_5d IS NULL) = (hard_stop_10pct_5d IS NULL)),
    CHECK ((terminal_nonpositive_20d IS NULL) = (terminal_winner_20d IS NULL)),
    CHECK ((terminal_nonpositive_30d IS NULL) = (terminal_winner_30d IS NULL)),
    CHECK (
        terminal_stagnant_5d IS NULL
        OR NOT (terminal_stagnant_5d AND terminal_winner_5d)
    ),
    CHECK (analysis_split IN ('discovery', 'validation', 'diagnostic', 'holdout', 'purged')),
    CHECK (
        NOT include_final
        OR (include_weak_filter AND include_loss_first_filter)
    ),
    CHECK (filter_decision IN ('include', 'exclude')),
    CHECK ((filter_decision = 'include') = include_final),
    CHECK (
        (filter_decision = 'include' AND exclusion_reason IS NULL)
        OR
        (filter_decision = 'exclude' AND NULLIF(TRIM(exclusion_reason), '') IS NOT NULL)
    ),
    CHECK (
        (strong_confirmation
         AND NULLIF(TRIM(confirmation_reason), '') IS NOT NULL)
        OR
        (NOT strong_confirmation AND confirmation_reason IS NULL)
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
    day20_end_date                                   DATE,
    day40_end_date                                   DATE,
    day60_end_date                                   DATE,
    day90_end_date                                   DATE,
    symbol                                           TEXT NOT NULL,
    exchange                                         TEXT NOT NULL,
    cik                                              BIGINT NOT NULL,
    price_continuity_segment                         INTEGER NOT NULL,
    currency                                         TEXT NOT NULL,
    decision_stage                                   TEXT NOT NULL,

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
    landmark_volume_vs_sma7_prior_ratio              NUMERIC(30,8),
    landmark_volume_vs_sma14_prior_ratio             NUMERIC(30,8),
    landmark_volume_vs_sma100_prior_ratio            NUMERIC(30,8),
    landmark_notional_vs_sma7_prior_ratio            NUMERIC(30,8),
    landmark_notional_vs_sma14_prior_ratio           NUMERIC(30,8),
    landmark_notional_vs_sma100_prior_ratio          NUMERIC(30,8),
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
    landmark_return_5d_pct                           NUMERIC(20,8),
    landmark_return_10d_pct                          NUMERIC(20,8),
    landmark_return_20d_pct                          NUMERIC(20,8),
    landmark_max_drawdown_20d_pct                    NUMERIC(20,8),
    landmark_trend_slope_20_pct_per_session          NUMERIC(20,8),
    landmark_trend_r2_20                             NUMERIC(20,8),
    landmark_trend_efficiency_20                     NUMERIC(20,8),
    landmark_range_compression_10_vs_10_ratio        NUMERIC(20,8),
    landmark_distribution_day_count_20               NUMERIC(20,8),
    landmark_churning_day_count_20                   NUMERIC(20,8),
    mean_volume_since_signal_vs_prior21_ratio        NUMERIC(30,8),
    mean_notional_since_signal_vs_prior21_ratio      NUMERIC(30,8),

    hit_gain_2pct_so_far                             BOOLEAN,
    hit_gain_5pct_so_far                             BOOLEAN,
    hit_loss_5pct_so_far                             BOOLEAN,
    hit_loss_10pct_so_far                            BOOLEAN,
    first_gain_2pct_day_so_far                       SMALLINT,
    first_gain_5pct_day_so_far                       SMALLINT,
    first_loss_5pct_day_so_far                       SMALLINT,
    first_loss_10pct_day_so_far                      SMALLINT,
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
    stagnant_at_day5                                  BOOLEAN,
    effective_adjusted_open                           NUMERIC(20,8),
    effective_open_return_from_signal_pct             NUMERIC(20,8),
    max_gain_from_effective_open_to_day20_pct         NUMERIC(20,8),
    max_loss_from_effective_open_to_day20_pct         NUMERIC(20,8),
    terminal_return_from_effective_open_to_day20_pct  NUMERIC(20,8),
    take_profit_better_to_day20                       BOOLEAN,
    continue_winner_to_day20                          BOOLEAN,
    max_gain_from_effective_open_to_day40_pct         NUMERIC(20,8),
    max_loss_from_effective_open_to_day40_pct         NUMERIC(20,8),
    terminal_return_from_effective_open_to_day40_pct  NUMERIC(20,8),
    take_profit_better_to_day40                       BOOLEAN,
    continue_winner_to_day40                          BOOLEAN,
    max_gain_from_effective_open_to_day60_pct         NUMERIC(20,8),
    max_loss_from_effective_open_to_day60_pct         NUMERIC(20,8),
    terminal_return_from_effective_open_to_day60_pct  NUMERIC(20,8),
    take_profit_better_to_day60                       BOOLEAN,
    continue_winner_to_day60                          BOOLEAN,
    max_gain_from_effective_open_to_day90_pct         NUMERIC(20,8),
    max_loss_from_effective_open_to_day90_pct         NUMERIC(20,8),
    terminal_return_from_effective_open_to_day90_pct  NUMERIC(20,8),
    take_profit_better_to_day90                       BOOLEAN,
    continue_winner_to_day90                          BOOLEAN,

    market_breadth_above_ma50_ratio                   NUMERIC(20,8),
    market_breadth_above_ma200_ratio                  NUMERIC(20,8),
    market_breadth_trend_template_ratio               NUMERIC(20,8),
    market_advancer_ratio                             NUMERIC(20,8),
    market_median_daily_return_pct                    NUMERIC(20,8),
    market_spy_prior_return_5d_pct                    NUMERIC(20,8),
    market_qqq_prior_return_5d_pct                    NUMERIC(20,8),
    market_iwm_prior_return_5d_pct                    NUMERIC(20,8),
    market_vix_level                                  NUMERIC(20,8),
    market_vxn_level                                  NUMERIC(20,8),
    market_vix9d_to_vix_ratio                         NUMERIC(20,8),
    market_vix_to_vix3m_ratio                         NUMERIC(20,8),
    market_spy_return_since_signal_pct                NUMERIC(20,8),
    market_qqq_return_since_signal_pct                NUMERIC(20,8),
    market_iwm_return_since_signal_pct                NUMERIC(20,8),
    relative_return_vs_spy_since_signal_pct_points    NUMERIC(20,8),
    relative_return_vs_qqq_since_signal_pct_points    NUMERIC(20,8),
    relative_return_vs_iwm_since_signal_pct_points    NUMERIC(20,8),

    analysis_split                                    TEXT NOT NULL,
    include_stagnation_filter                         BOOLEAN NOT NULL,
    include_loss_filter                               BOOLEAN NOT NULL,
    include_final                                     BOOLEAN NOT NULL,
    stagnation_matched_rule_ids                       TEXT,
    loss_matched_rule_ids                             TEXT,
    matched_rule_ids                                  TEXT,
    cut_decision                                      TEXT NOT NULL,
    cut_reason                                        TEXT,
    management_include_final                          BOOLEAN NOT NULL,
    management_matched_rule_ids                       TEXT,
    management_decision                               TEXT NOT NULL,
    management_reason                                 TEXT,

    PRIMARY KEY (signal_date, symbol, exchange, cik, landmark_day),
    CHECK (landmark_day IN (1, 2, 3, 5, 20, 30)),
    CHECK (price_continuity_segment > 0),
    CHECK (currency = 'USD'),
    CHECK (
        (landmark_day IN (1, 2, 3) AND decision_stage = 'early_cut')
        OR (landmark_day = 5 AND decision_stage = 'stagnation_review')
        OR (landmark_day IN (20, 30) AND decision_stage = 'profit_review')
    ),
    CHECK (landmark_date IS NULL OR landmark_date > signal_date),
    CHECK (effective_session_date IS NULL OR effective_session_date >= signal_date),
    CHECK (horizon_end_date IS NULL OR horizon_end_date > signal_date),
    CHECK (day20_end_date IS NULL OR day20_end_date > signal_date),
    CHECK (day40_end_date IS NULL OR day40_end_date > signal_date),
    CHECK (day60_end_date IS NULL OR day60_end_date > signal_date),
    CHECK (day90_end_date IS NULL OR day90_end_date > signal_date),
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
    CHECK (first_loss_10pct_day_so_far IS NULL OR first_loss_10pct_day_so_far BETWEEN 1 AND landmark_day),
    CHECK (landmark_trend_r2_20 IS NULL OR landmark_trend_r2_20 BETWEEN 0 AND 1),
    CHECK (landmark_trend_efficiency_20 IS NULL OR landmark_trend_efficiency_20 BETWEEN 0 AND 1),
    CHECK (max_gain_from_effective_open_to_day20_pct IS NULL OR max_gain_from_effective_open_to_day20_pct >= 0),
    CHECK (max_loss_from_effective_open_to_day20_pct IS NULL OR max_loss_from_effective_open_to_day20_pct <= 0),
    CHECK (max_gain_from_effective_open_to_day40_pct IS NULL OR max_gain_from_effective_open_to_day40_pct >= 0),
    CHECK (max_loss_from_effective_open_to_day40_pct IS NULL OR max_loss_from_effective_open_to_day40_pct <= 0),
    CHECK (max_gain_from_effective_open_to_day60_pct IS NULL OR max_gain_from_effective_open_to_day60_pct >= 0),
    CHECK (max_loss_from_effective_open_to_day60_pct IS NULL OR max_loss_from_effective_open_to_day60_pct <= 0),
    CHECK (max_gain_from_effective_open_to_day90_pct IS NULL OR max_gain_from_effective_open_to_day90_pct >= 0),
    CHECK (max_loss_from_effective_open_to_day90_pct IS NULL OR max_loss_from_effective_open_to_day90_pct <= 0),
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
        decision_stage <> 'early_cut'
        OR active_at_landmark = (
            eligible_at_landmark AND prior_policy_cut_day IS NULL
        )
    ),
    CHECK (
        decision_stage <> 'early_cut'
        OR include_final = (include_stagnation_filter AND include_loss_filter)
    ),
    CHECK (
        cut_decision IN (
            'cut', 'hold', 'not_active', 'not_eligible', 'not_evaluable'
        )
    ),
    CHECK (
        decision_stage <> 'early_cut'
        OR (
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
        )
    ),
    CHECK (
        (cut_decision = 'cut' AND NULLIF(TRIM(cut_reason), '') IS NOT NULL)
        OR
        (cut_decision <> 'cut' AND cut_reason IS NULL)
    ),
    CHECK (
        management_decision IN (
            'hold', 'hard_stop', 'cut_stagnation', 'take_profit',
            'not_eligible', 'not_evaluable'
        )
    ),
    CHECK (
        decision_stage = 'early_cut'
        OR (
            NOT active_at_landmark
            AND NOT include_stagnation_filter
            AND NOT include_loss_filter
            AND NOT include_final
            AND cut_decision = 'not_evaluable'
        )
    ),
    CHECK (
        decision_stage <> 'early_cut'
        OR (
            NOT management_include_final
            AND management_decision = 'not_evaluable'
            AND management_reason IS NULL
        )
    ),
    CHECK (
        decision_stage = 'early_cut'
        OR (
            (management_decision = 'hold' AND management_include_final)
            OR
            (management_decision IN ('cut_stagnation', 'take_profit')
             AND NOT management_include_final
             AND NULLIF(TRIM(management_reason), '') IS NOT NULL)
            OR
            (management_decision IN ('hard_stop', 'not_eligible', 'not_evaluable')
             AND NOT management_include_final
             AND management_reason IS NULL)
        )
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
    pattern_name                       TEXT,
    pattern_match_mode                 TEXT,
    pattern_total_clause_count         SMALLINT,
    pattern_required_clause_count      SMALLINT,
    pattern_score_window_sessions      SMALLINT,
    pattern_score_threshold_pct        NUMERIC(20,8),
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
    passes_multiple_testing            BOOLEAN,
    multiple_testing_candidate_count   INTEGER,
    permutation_trial_count            INTEGER,
    max_stat_permutation_p_value       NUMERIC(18,8),
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
    CHECK (decision_family IN ('entry_filter', 'entry_confirmation', 'early_cut', 'position_management')),
    CHECK (
        objective IN (
            'weak_5d', 'loss_first_5d', 'terminal_stagnant_5d',
            'stagnant_5d', 'hard_stop_10pct_5d',
            'terminal_nonpositive_20d', 'terminal_nonpositive_30d',
            'strong_first_5d', 'terminal_winner_5d',
            'terminal_winner_20d', 'terminal_winner_30d',
            'runner_60d', 'runner_90d',
            'stagnant_to_day5', 'loss_first_to_day5', 'bad_to_day5',
            'take_profit_better_to_day20',
            'take_profit_better_to_day40',
            'take_profit_better_to_day60',
            'take_profit_better_to_day90'
        )
    ),
    CHECK (
        protected_outcome IN (
            'strong_first_5d', 'terminal_winner_5d', 'bad_5d',
            'terminal_winner_20d', 'terminal_winner_30d', 'runner_60d',
            'terminal_stagnant_5d', 'terminal_nonpositive_20d',
            'terminal_nonpositive_30d', 'strong_first_to_day5',
            'continue_winner_to_day20', 'continue_winner_to_day40',
            'continue_winner_to_day60', 'continue_winner_to_day90'
        )
    ),
    CHECK (
        (decision_family = 'entry_filter'
         AND landmark_day IS NULL
         AND (
             (objective IN ('weak_5d', 'loss_first_5d')
              AND protected_outcome = 'strong_first_5d')
             OR
             (objective = 'terminal_stagnant_5d'
              AND protected_outcome = 'terminal_winner_5d')
             OR (objective = 'stagnant_5d'
                 AND protected_outcome = 'terminal_winner_20d')
             OR (objective = 'hard_stop_10pct_5d'
                 AND protected_outcome = 'runner_60d')
             OR (objective = 'terminal_nonpositive_20d'
                 AND protected_outcome = 'terminal_winner_20d')
             OR (objective = 'terminal_nonpositive_30d'
                 AND protected_outcome = 'terminal_winner_30d')
         ))
        OR
        (decision_family = 'entry_confirmation'
         AND landmark_day IS NULL
         AND (
             (objective = 'strong_first_5d' AND protected_outcome = 'bad_5d')
             OR
             (objective = 'terminal_winner_5d'
              AND protected_outcome = 'terminal_stagnant_5d')
             OR (objective = 'terminal_winner_20d'
                 AND protected_outcome = 'terminal_nonpositive_20d')
             OR (objective = 'terminal_winner_30d'
                 AND protected_outcome = 'terminal_nonpositive_30d')
             OR (objective IN ('runner_60d', 'runner_90d')
                 AND protected_outcome = 'terminal_nonpositive_30d')
         ))
        OR
        (decision_family = 'early_cut'
         AND landmark_day BETWEEN 1 AND 3
         AND (
             objective IN ('stagnant_to_day5', 'loss_first_to_day5')
             OR (objective = 'bad_to_day5' AND landmark_day = 1)
         )
         AND protected_outcome = 'strong_first_to_day5')
        OR
        (decision_family = 'position_management'
         AND landmark_day IN (5, 20, 30)
         AND objective = 'take_profit_better_to_day'
             || SUBSTRING(objective FROM '[0-9]+$')
         AND protected_outcome = 'continue_winner_to_day'
             || SUBSTRING(objective FROM '[0-9]+$')
         AND (
             (landmark_day = 5 AND objective = 'take_profit_better_to_day20')
             OR
             (landmark_day IN (20, 30)
              AND objective IN (
                  'take_profit_better_to_day40',
                  'take_profit_better_to_day60',
                  'take_profit_better_to_day90'
              ))
         ))
    ),
    CHECK (
        feature_group IN (
            'none', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'M', 'N', 'P',
            'R', 'S', 'T', 'multiple'
        )
    ),
    CHECK (operator IS NULL OR operator IN ('le', 'ge')),
    CHECK (quantile_value IS NULL OR quantile_value BETWEEN 0 AND 1),
    CHECK (
        pattern_match_mode IS NULL
        OR pattern_match_mode IN ('all', 'k_of_n', 'score_threshold')
    ),
    CHECK (
        (pattern_match_mode IS NULL
         AND pattern_name IS NULL
         AND pattern_total_clause_count IS NULL
         AND pattern_required_clause_count IS NULL
         AND pattern_score_window_sessions IS NULL
         AND pattern_score_threshold_pct IS NULL)
        OR
        (pattern_match_mode = 'all'
         AND pattern_name IS NOT NULL
         AND pattern_total_clause_count >= 1
         AND pattern_required_clause_count = pattern_total_clause_count
         AND pattern_score_window_sessions IS NULL
         AND pattern_score_threshold_pct IS NULL)
        OR
        (pattern_match_mode = 'k_of_n'
         AND pattern_name IS NOT NULL
         AND pattern_total_clause_count >= 2
         AND pattern_required_clause_count BETWEEN 2 AND pattern_total_clause_count
         AND pattern_score_window_sessions IS NULL
         AND pattern_score_threshold_pct IS NULL)
        OR
        (pattern_match_mode = 'score_threshold'
         AND pattern_name IS NOT NULL
         AND pattern_total_clause_count IS NULL
         AND pattern_required_clause_count IS NULL
         AND (pattern_score_window_sessions IS NULL
              OR pattern_score_window_sessions IN (10, 15, 20, 30, 40, 63, 126))
         AND (pattern_score_threshold_pct IS NULL
              OR pattern_score_threshold_pct BETWEEN 0 AND 100))
    ),
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
    CHECK (selection_order IS NULL OR selection_order BETWEEN 1 AND 3),
    CHECK (NOT is_final_filter OR is_selected),
    CHECK (component_count >= 0),
    CHECK (
        multiple_testing_candidate_count IS NULL
        OR multiple_testing_candidate_count > 0
    ),
    CHECK (permutation_trial_count IS NULL OR permutation_trial_count >= 19),
    CHECK (
        max_stat_permutation_p_value IS NULL
        OR max_stat_permutation_p_value BETWEEN 0 AND 1
    ),
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
    -- Objective and protected outcomes are independent labels and may overlap.
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

CREATE INDEX IF NOT EXISTS idx_safr_rule_pattern
    ON stock_analyser_filter_research_rule_results
    (pattern_name, pattern_match_mode, pattern_score_window_sessions,
     decision_family, evaluation_scope);

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
    'One continuity-safe false-to-true trend-template event per stock, with causal entry features, diagnostic-only current-taxonomy backcasts, forward outcomes and entry-filter decisions.';
COMMENT ON COLUMN stock_analyser_filter_research_signal_results.current_taxonomy_backcast_industry IS
    'Current IBKR taxonomy snapshot backcast to historical signals; diagnostic-only and never eligible for final filter selection.';
COMMENT ON COLUMN stock_analyser_filter_research_signal_results.current_taxonomy_backcast_category IS
    'Current IBKR taxonomy snapshot backcast to historical signals; interpreted together with Industry and diagnostic-only.';
COMMENT ON COLUMN stock_analyser_filter_research_signal_results.current_taxonomy_backcast_subcategory IS
    'Current IBKR taxonomy snapshot backcast to historical signals; interpreted as the full Industry/Category/Subcategory path and diagnostic-only.';
COMMENT ON TABLE stock_analyser_filter_research_early_cut_results IS
    'D+1, D+2, D+3, D+5, D+20 and D+30 point-in-time landmark observations for each signal, including sequential early-cut and independent next-open position-management outcomes.';
COMMENT ON TABLE stock_analyser_filter_research_rule_results IS
    'Entry-filter, entry-confirmation, early-cut and position-management rule candidates, stability gates and out-of-sample evaluations.';
