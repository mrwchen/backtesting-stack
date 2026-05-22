# Backtest Model Configs

Each file is loaded automatically by model stem:

```text
backtest_models/pullback_bounce_fundamental_v1.py
backtest_model_configs/pullback_bounce_fundamental_v1.env
```

These files contain model-specific entry, exit, timing, probability, and grid
parameters. `docker-compose.yml` should only contain runtime, datasource,
common policy, account, and portfolio-risk settings.
