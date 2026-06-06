-- QQQ engine/regime diagnostics.
--
-- Read-only report. Change params.run_id/start_day/end_day as needed.
-- The goal is to separate:
--   1. engine accounting sanity,
--   2. QQQ market replay versus the run curve,
--   3. world-regime detection timing during QQQ drawdowns,
--   4. trade concentration created by runner policy.

WITH params AS (
  SELECT
    1::integer AS run_id,
    DATE '2024-07-01' AS start_day,
    DATE '2026-05-01' AS end_day,
    5.0::double precision AS drawdown_warn_pct,
    10.0::double precision AS drawdown_stress_pct
)
SELECT
  'engine_pnl_reconciliation' AS check_name,
  r.run_id,
  r.model_file,
  r.account_profile,
  r.initial_equity,
  r.final_equity,
  round((r.final_equity - r.initial_equity)::numeric, 2) AS final_minus_initial,
  round(sum(t.pnl_usd)::numeric, 2) AS sum_trade_pnl,
  round(((r.final_equity - r.initial_equity) - sum(t.pnl_usd))::numeric, 2) AS diff_usd,
  count(*) AS trade_rows,
  count(*) FILTER (WHERE t.exit_ts < t.entry_ts) AS exits_before_entries,
  count(*) FILTER (WHERE t.pnl_usd IS NULL OR t.return_pct IS NULL) AS missing_results
FROM params p
JOIN public.backtest_runs r ON r.run_id = p.run_id
JOIN public.backtest_trades t ON t.run_id = r.run_id
GROUP BY r.run_id, r.model_file, r.account_profile, r.initial_equity, r.final_equity;

WITH params AS (
  SELECT 1::integer AS run_id, DATE '2024-07-01' AS start_day, DATE '2026-05-01' AS end_day
),
curve_daily AS (
  SELECT DISTINCT ON (trade_date)
    trade_date,
    equity_usd::double precision AS equity_usd,
    open_positions,
    initial_margin_usd::double precision AS initial_margin_usd,
    open_pnl_usd::double precision AS open_pnl_usd,
    closed_trades
  FROM public.backtest_account_curve
  WHERE run_id = (SELECT run_id FROM params)
  ORDER BY trade_date, ts DESC, seq_in_run DESC
),
qqq_daily AS (
  SELECT DISTINCT ON ((ts AT TIME ZONE 'America/New_York')::date)
    (ts AT TIME ZONE 'America/New_York')::date AS trade_date,
    close::double precision AS qqq_close
  FROM public.alpaca_market_data_1h
  WHERE symbol = 'QQQ'
  ORDER BY (ts AT TIME ZONE 'America/New_York')::date, ts DESC
),
joined AS (
  SELECT
    c.*,
    q.qqq_close,
    first_value(c.equity_usd) OVER (ORDER BY c.trade_date) AS base_equity,
    first_value(q.qqq_close) OVER (ORDER BY c.trade_date) AS base_qqq
  FROM params p
  JOIN curve_daily c ON c.trade_date BETWEEN p.start_day AND p.end_day
  JOIN qqq_daily q USING (trade_date)
)
SELECT
  date_trunc('month', trade_date)::date AS month,
  max(trade_date) AS month_end_day,
  round(((last(equity_usd, trade_date) / first(equity_usd, trade_date) - 1.0) * 100.0)::numeric, 2) AS run_month_return_pct,
  round(((last(qqq_close, trade_date) / first(qqq_close, trade_date) - 1.0) * 100.0)::numeric, 2) AS qqq_month_return_pct,
  round(((last(equity_usd / base_equity, trade_date) - last(qqq_close / base_qqq, trade_date)) * 100.0)::numeric, 2) AS cumulative_active_gap_pct,
  round(avg(open_positions)::numeric, 2) AS avg_open_positions,
  round((avg(initial_margin_usd / NULLIF(equity_usd, 0.0)) * 5.0 * 100.0)::numeric, 1) AS avg_gross_notional_pct_equity_est
FROM joined
GROUP BY date_trunc('month', trade_date)::date
ORDER BY month;

WITH params AS (
  SELECT
    DATE '2024-07-01' AS start_day,
    DATE '2026-05-01' AS end_day,
    5.0::double precision AS drawdown_warn_pct,
    10.0::double precision AS drawdown_stress_pct
),
qqq_daily AS (
  SELECT DISTINCT ON ((ts AT TIME ZONE 'America/New_York')::date)
    (ts AT TIME ZONE 'America/New_York')::date AS day,
    close::double precision AS qqq_close
  FROM public.alpaca_market_data_1h
  WHERE symbol = 'QQQ'
  ORDER BY (ts AT TIME ZONE 'America/New_York')::date, ts DESC
),
qqq_dd AS (
  SELECT
    q.day,
    q.qqq_close,
    max(q.qqq_close) OVER (ORDER BY q.day ROWS BETWEEN 60 PRECEDING AND CURRENT ROW) AS qqq_peak_60d
  FROM qqq_daily q
),
joined AS (
  SELECT
    q.day,
    q.qqq_close,
    (q.qqq_close / NULLIF(q.qqq_peak_60d, 0.0) - 1.0) * 100.0 AS qqq_60d_drawdown_pct,
    r.regime_label AS same_day_regime_label,
    r.composite_score AS same_day_composite,
    r.max_shock_type_score AS same_day_max_shock,
    r.dominant_shock_type AS same_day_shock_type,
    prev.regime_label AS engine_asof_regime_label,
    prev.composite_score AS engine_asof_composite,
    prev.max_shock_type_score AS engine_asof_max_shock,
    prev.dominant_shock_type AS engine_asof_shock_type
  FROM params p
  JOIN qqq_dd q ON q.day BETWEEN p.start_day AND p.end_day
  LEFT JOIN public.world_regime_daily_scores_mv r ON r.day = q.day
  LEFT JOIN public.world_regime_daily_scores_mv prev ON prev.day = q.day - 1
)
SELECT
  day,
  round(qqq_close::numeric, 2) AS qqq_close,
  round(qqq_60d_drawdown_pct::numeric, 2) AS qqq_60d_drawdown_pct,
  same_day_regime_label,
  round(same_day_composite::numeric, 2) AS same_day_composite,
  round(same_day_max_shock::numeric, 2) AS same_day_max_shock,
  same_day_shock_type,
  engine_asof_regime_label,
  round(engine_asof_composite::numeric, 2) AS engine_asof_composite,
  round(engine_asof_max_shock::numeric, 2) AS engine_asof_max_shock,
  engine_asof_shock_type,
  CASE
    WHEN qqq_60d_drawdown_pct <= -10.0 AND COALESCE(engine_asof_max_shock, 0.0) < 70.0 THEN 'LATE_FOR_10PCT_DRAWDOWN'
    WHEN qqq_60d_drawdown_pct <= -5.0 AND COALESCE(engine_asof_max_shock, 0.0) < 60.0 THEN 'LATE_FOR_5PCT_DRAWDOWN'
    WHEN qqq_60d_drawdown_pct <= -5.0 THEN 'DETECTED'
    ELSE 'NO_STRESS'
  END AS detection_check
FROM joined
WHERE qqq_60d_drawdown_pct <= -5.0
ORDER BY day;

WITH params AS (
  SELECT 1::integer AS run_id, DATE '2024-07-01' AS start_day, DATE '2026-05-01' AS end_day
)
SELECT
  date_trunc('month', t.entry_ts)::date AS entry_month,
  t.direction,
  t.world_regime_label,
  count(*) AS opened_trades,
  round(sum(t.pnl_usd)::numeric, 2) AS pnl_usd,
  round(avg(t.position_size_usd / NULLIF(t.equity_before, 0.0) * 100.0)::numeric, 1) AS avg_notional_pct_equity,
  round(max(t.position_size_usd / NULLIF(t.equity_before, 0.0) * 100.0)::numeric, 1) AS max_notional_pct_equity,
  count(*) FILTER (WHERE t.outcome_status = 'HIT_SL') AS hit_sl,
  count(*) FILTER (WHERE t.outcome_status = 'MAX_HOLD') AS max_hold
FROM params p
JOIN public.backtest_trades t ON t.run_id = p.run_id
WHERE t.entry_ts::date BETWEEN p.start_day AND p.end_day
GROUP BY date_trunc('month', t.entry_ts)::date, t.direction, t.world_regime_label
ORDER BY entry_month, direction, opened_trades DESC;

SELECT
  day,
  bucket,
  round(bucket_score::numeric, 2) AS bucket_score,
  round(bucket_coverage_pct::numeric, 1) AS bucket_coverage_pct,
  available_components,
  configured_components
FROM public.world_regime_bucket_scores_mv
WHERE day IN (
    DATE '2024-07-10',
    DATE '2024-07-17',
    DATE '2024-07-24',
    DATE '2024-08-02',
    DATE '2024-08-05',
    DATE '2025-03-03',
    DATE '2025-04-04',
    DATE '2025-11-20'
  )
  AND bucket IN (
    'Volatility',
    'Risk Assets',
    'Credit / FCI',
    'Liquidity / Systemic',
    'Rates Level',
    'Curve / Recession',
    'USD / FX',
    'Macro / Growth',
    'Positioning / Crowding',
    'Policy / Fiscal News',
    'Geopolitics / Conflict'
  )
ORDER BY day, bucket;
