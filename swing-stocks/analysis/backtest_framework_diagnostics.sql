-- Backtest framework diagnostics.
-- Edit the run ids in each selected_runs CTE before execution.
-- These queries are read-only and do not create tables or views.

-- 1) Run summary.
WITH selected_runs AS (
    SELECT unnest(ARRAY[1, 2]::int[]) AS run_id
)
SELECT
    r.run_id,
    r.created_at,
    r.model_file,
    r.account_profile,
    r.start_date,
    r.end_date,
    r.initial_equity,
    r.final_equity,
    r.total_return_pct,
    r.max_drawdown_pct,
    r.total_trades,
    r.win_rate_pct,
    r.profit_factor,
    r.risk_per_trade_pct,
    r.max_open_positions
FROM public.backtest_runs r
JOIN selected_runs s USING (run_id)
ORDER BY r.run_id;

-- 2) Exit-type attribution by direction.
WITH selected_runs AS (
    SELECT unnest(ARRAY[1, 2]::int[]) AS run_id
)
SELECT
    t.run_id,
    t.direction,
    t.outcome_status,
    COUNT(*) AS trades,
    ROUND(SUM(t.pnl_usd)::numeric, 2) AS pnl_usd,
    ROUND(AVG(t.return_pct)::numeric, 4) AS avg_return_pct,
    ROUND(100.0 * AVG((t.pnl_usd > 0)::int)::numeric, 2) AS win_rate_pct,
    ROUND((
        SUM(CASE WHEN t.pnl_usd > 0 THEN t.pnl_usd ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_usd < 0 THEN t.pnl_usd ELSE 0 END)), 0)
    )::numeric, 3) AS profit_factor
FROM public.backtest_trades t
JOIN selected_runs s USING (run_id)
GROUP BY t.run_id, t.direction, t.outcome_status
ORDER BY t.run_id, t.direction, pnl_usd;

-- 3) Entry regime x direction.
WITH selected_runs AS (
    SELECT unnest(ARRAY[1, 2]::int[]) AS run_id
)
SELECT
    t.run_id,
    t.direction,
    COALESCE(t.world_regime_label, '') AS entry_regime,
    COUNT(*) AS trades,
    ROUND(SUM(t.pnl_usd)::numeric, 2) AS pnl_usd,
    ROUND(AVG(t.return_pct)::numeric, 4) AS avg_return_pct,
    ROUND(100.0 * AVG((t.pnl_usd > 0)::int)::numeric, 2) AS win_rate_pct,
    ROUND((
        SUM(CASE WHEN t.pnl_usd > 0 THEN t.pnl_usd ELSE 0 END)
        / NULLIF(ABS(SUM(CASE WHEN t.pnl_usd < 0 THEN t.pnl_usd ELSE 0 END)), 0)
    )::numeric, 3) AS profit_factor
FROM public.backtest_trades t
JOIN selected_runs s USING (run_id)
GROUP BY t.run_id, t.direction, COALESCE(t.world_regime_label, '')
ORDER BY t.run_id, t.direction, pnl_usd;

-- 4) Worst monthly periods.
WITH selected_runs AS (
    SELECT unnest(ARRAY[1, 2]::int[]) AS run_id
)
SELECT
    t.run_id,
    date_trunc('month', t.outcome_date)::date AS month,
    COUNT(*) AS trades,
    ROUND(SUM(t.pnl_usd)::numeric, 2) AS pnl_usd,
    ROUND(AVG(t.return_pct)::numeric, 4) AS avg_return_pct
FROM public.backtest_trades t
JOIN selected_runs s USING (run_id)
GROUP BY t.run_id, date_trunc('month', t.outcome_date)::date
ORDER BY pnl_usd
LIMIT 30;
