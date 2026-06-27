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
DROP TABLE IF EXISTS backtest2_nas100_hfmed_range_analysis CASCADE;
\endif

CREATE TABLE IF NOT EXISTS backtest2_nas100_hfmed_range_analysis (
    bar_start_ts                    TIMESTAMPTZ    NOT NULL,
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

    bar_open                        NUMERIC(18,6)  NOT NULL,
    bar_high                        NUMERIC(18,6)  NOT NULL,
    bar_low                         NUMERIC(18,6)  NOT NULL,
    bar_close                       NUMERIC(18,6)  NOT NULL,
    bar_tick_count                  INTEGER        NOT NULL,

    profile_low                     NUMERIC(18,6),
    profile_high                    NUMERIC(18,6),
    profile_range_points            NUMERIC(18,6),

    PRIMARY KEY (bar_start_ts, lookback_bars, analysis_id)
);

SELECT create_hypertable(
    'backtest2_nas100_hfmed_range_analysis',
    'bar_start_ts',
    chunk_time_interval => :'hfmed_range_analysis_chunk_interval'::interval,
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_analysis_symbol_lookup_idx
    ON backtest2_nas100_hfmed_range_analysis(symbol, lookback_bars, bar_start_ts DESC);

CREATE INDEX IF NOT EXISTS backtest2_nas100_hfmed_range_analysis_run_idx
    ON backtest2_nas100_hfmed_range_analysis(analysis_id, lookback_bars, bar_start_ts);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE backtest2_nas100_hfmed_range_analysis TO "market-data-account";

