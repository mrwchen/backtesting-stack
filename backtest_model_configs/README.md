# Backtest Model Configs

Each file is loaded automatically by model stem:

```text
backtest_models/pullback_bounce_fundamental_v1.py
backtest_model_configs/pullback_bounce_fundamental_v1.env
```

Each file contains only model-specific selection parameters. Models return a
mandatory `TradeIntent`: symbol, direction, score, and reason.

`docker-compose.yml` owns runtime, datasource, common policy, account,
portfolio-risk settings, central take-profit/TCR and max-hold-duration settings,
and the central stop-loss policy. Model config files must not set execution
levels such as stop loss, take profits, sizing, or hold duration.
