-- Optional destructive reset for the backtesting stack.
-- Keep this limited to result tables; source market-data tables share this DB.

DROP TABLE IF EXISTS backtest_monte_carlo CASCADE;
DROP TABLE IF EXISTS backtest_trades CASCADE;
DROP TABLE IF EXISTS backtest_runs CASCADE;
