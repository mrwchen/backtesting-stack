"""
Swing trade backtester.

Generates swing signals day-by-day on historical data, simulates a margin
account, and writes results to backtest_runs / backtest_trades /
backtest_decision_events / backtest_account_curve / backtest_monte_carlo.

Point-in-time data used:
  - world_regime_daily_scores_mv / _historical_mv : as_of each simulated day
  - stock_scorer_fundamental_scores : available at each simulated entry cutoff (true PIT by default)
  - alpaca_market_data_1h         : up_to each simulated entry cutoff (true PIT)

Run once, write results, exit.
"""

from backtest_core.logging_utils import configure_logging

configure_logging()

from backtest_core.main import main


if __name__ == "__main__":
    main()
