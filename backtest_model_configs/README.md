# Backtest Model Configs

Each file is loaded automatically by model stem:

```text
backtest_models/pullback_bounce_fundamental_v1.py
backtest_model_configs/pullback_bounce_fundamental_v1.env
```

Each file is split into two sections:

```text
# Overrides for docker-compose defaults
# Model-specific parameters
```

The override section may contain values that intentionally replace defaults from
`docker-compose.yml`. The model-specific section contains entry, timing,
probability, and grid parameters that are not docker-compose defaults.

`docker-compose.yml` should contain runtime, datasource, common policy, account,
portfolio-risk settings, default take-profit/TCR and max-hold-duration settings,
and the default common stop-loss policy.

Normal strategy models provide entries; the common backtest policy sets the
initial stop loss and take profits after the next-bar-open entry fill. Normal
models use the `LONG_TP1_PCT`, `LONG_TP2_PCT`, `SHORT_TP1_PCT`,
`SHORT_TP2_PCT`, `TP1_CLOSE_RATIO`, `LONG_MAX_HOLD_DAYS`, and
`SHORT_MAX_HOLD_DAYS` defaults from `docker-compose.yml`. Only
`statistical_regime_probability_v1.py` currently carries model-specific
take-profit/TCR overrides; max-hold duration remains common for all models. A
model may own its stop-loss logic only when its config exposes an explicit
stop-loss parameter set, for example `statistical_regime_probability_v1.py`
with `LONG_STOP_VOL_MULT`, `SHORT_STOP_VOL_MULT`, `MIN_STOP_PCT`, and
`MAX_STOP_PCT`.
