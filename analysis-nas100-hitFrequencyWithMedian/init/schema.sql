-- Schema for the NAS100 hit-frequency median range analysis.
-- Tables are prefixed `backtest2_nas100_hfmed_` to avoid collisions.

CREATE EXTENSION IF NOT EXISTS timescaledb;

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

\if :drop_hfmed_range_analysis_tables_on_start
DROP TABLE IF EXISTS backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana CASCADE;
DROP TABLE IF EXISTS backtest2_nas100_hfmed_range_analysis CASCADE;
\endif

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_range_analysis (
    cross_ts                        TIMESTAMPTZ    NOT NULL,
    analysis_id                     UUID           NOT NULL,
    created_at                      TIMESTAMPTZ    NOT NULL,
    symbol                          TEXT           NOT NULL,
    source_table                    TEXT           NOT NULL,
    source_start_ts                 TIMESTAMPTZ,
    source_end_ts                   TIMESTAMPTZ,
    data_start_ts                   TIMESTAMPTZ    NOT NULL,
    data_end_ts                     TIMESTAMPTZ    NOT NULL,
    ticks_loaded                    BIGINT         NOT NULL,
    bars_loaded                     BIGINT         NOT NULL,

    bar_seconds                     INTEGER        NOT NULL,
    price_step                      NUMERIC(12,6)  NOT NULL,
    lookback_bars                   INTEGER        NOT NULL,
    min_lookback_bars               INTEGER        NOT NULL,
    profile_max_lookback_seconds    INTEGER,

    event_tick_index                BIGINT         NOT NULL,
    bar_start_ts                    TIMESTAMPTZ    NOT NULL,
    direction_code                  SMALLINT       NOT NULL,
    direction                       TEXT           NOT NULL,
    previous_mid                    NUMERIC(18,6)  NOT NULL,
    signal_mid                      NUMERIC(18,6)  NOT NULL,
    q50_level                       NUMERIC(18,6)  NOT NULL,
    profile_low                     NUMERIC(18,6)  NOT NULL,
    profile_high                    NUMERIC(18,6)  NOT NULL,
    profile_range_points            NUMERIC(18,6)  NOT NULL,

    CHECK (direction_code IN (-1, 1)),
    CHECK (direction IN ('UP', 'DOWN')),
    PRIMARY KEY (cross_ts, analysis_id, lookback_bars, event_tick_index)
);

SELECT create_hypertable(
    'backtest2_nas100_hfmed_range_analysis',
    'cross_ts',
    chunk_time_interval => :'hfmed_range_analysis_chunk_interval'::interval,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_analysis_symbol_cross_lookup_idx
    ON backtest2_nas100_hfmed_range_analysis(symbol, lookback_bars, cross_ts DESC);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_analysis_run_cross_idx
    ON backtest2_nas100_hfmed_range_analysis(analysis_id, lookback_bars, cross_ts);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE backtest2_nas100_hfmed_range_analysis TO "market-data-account";

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana (
    week_start_ts                  TIMESTAMPTZ    NOT NULL,
    analysis_id                    UUID           NOT NULL,
    created_at                     TIMESTAMPTZ    NOT NULL,
    symbol                         TEXT           NOT NULL,
    source_table                   TEXT           NOT NULL,
    source_start_ts                TIMESTAMPTZ,
    source_end_ts                  TIMESTAMPTZ,
    data_start_ts                  TIMESTAMPTZ    NOT NULL,
    data_end_ts                    TIMESTAMPTZ    NOT NULL,
    bar_seconds                    INTEGER        NOT NULL,
    price_step                     NUMERIC(12,6)  NOT NULL,
    lookback_bars                  INTEGER        NOT NULL,
    min_lookback_bars              INTEGER        NOT NULL,
    profile_max_lookback_seconds   INTEGER,
    session_timezone               TEXT           NOT NULL,
    session_sort_order             INTEGER        NOT NULL,
    session_label                  TEXT           NOT NULL,
    session_start_local_time       TIME           NOT NULL,
    session_end_local_time         TIME           NOT NULL,
    crossings_total                BIGINT         NOT NULL,
    week_first_cross_ts            TIMESTAMPTZ    NOT NULL,
    week_last_cross_ts             TIMESTAMPTZ    NOT NULL,
    min_range_points               NUMERIC(18,6),
    avg_range_points               NUMERIC(18,6),
    median_range_points            NUMERIC(18,6),
    p75_range_points               NUMERIC(18,6),
    p95_range_points               NUMERIC(18,6),
    max_range_points               NUMERIC(18,6),

    PRIMARY KEY (week_start_ts, analysis_id, session_sort_order, lookback_bars)
);

SELECT create_hypertable(
    'backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana',
    'week_start_ts',
    chunk_time_interval => :'hfmed_range_analysis_chunk_interval'::interval,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_weekly_session_stats_lookup_idx
    ON backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana(symbol, session_sort_order, lookback_bars, week_start_ts DESC);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_weekly_session_stats_run_idx
    ON backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana(analysis_id, session_sort_order, lookback_bars, week_start_ts);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE backtest2_nas100_hfmed_range_weekly_session_stats_for_grafana TO "market-data-account";
