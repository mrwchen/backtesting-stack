# Backtest Model Configs

Each file is loaded automatically by model stem:

```text
backtest_models/pullback_bounce_fundamental_v1.py
backtest_model_configs/pullback_bounce_fundamental_v1.env
```

These files contain model-specific entry, timing, probability, and grid
parameters. `docker-compose.yml` should contain runtime, datasource, common
policy, account, portfolio-risk settings, and the default common stop-loss
policy.

Normal strategy models provide entries; the common backtest policy sets the
initial stop loss and take profits after the next-bar-open entry fill. A model
may own its stop-loss logic only when its config exposes an explicit stop-loss
parameter set, for example `statistical_regime_probability_v1.py` with
`LONG_STOP_VOL_MULT`, `SHORT_STOP_VOL_MULT`, `MIN_STOP_PCT`, and `MAX_STOP_PCT`.
