"""Entry point for the statistical scalping backtester.

Five-layer pipeline (see scalp_core/):
  1. Regime      — Hidden Markov Model
  2. Price       — Kalman Filter | State-Space Model      (PRICE_MODEL)
  3. Volatility  — GARCH | EGARCH                         (VOL_MODEL)
  4. Decision    — Bayesian classifier | Logistic model   (DECISION_MODEL)
  5. Risk        — Monte-Carlo (drawdown / slippage / sequence risk)
"""

from scalp_core.logging_utils import configure_logging

configure_logging()

from scalp_core.main import main

if __name__ == "__main__":
    main()
