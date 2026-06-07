# Swing Alpha Forward Research

Dieses Research-Skript prueft, ob der aktuelle Fundamental-/Swing-Scorer wirklich spaetere Swing-Rendite erklaert. Es schreibt keine Daten in die Datenbank und baut kein neues Modell. Es liest point-in-time Kandidaten ueber die gleiche Candidate-Logik wie der Backtest und misst danach Forward-Return, MAE und MFE.

## Serverlauf

Auf dem Server im Stack-Verzeichnis ausfuehren:

```bash
cd backtesting-stack/swing-stocks
docker compose run --rm backtest-runner python analysis/swing_alpha_forward_research.py \
  --start-date 2024-01-01 \
  --end-date 2026-05-01 \
  --frequency daily \
  --directions LONG,SHORT \
  --horizons 5,10,20,60 \
  --write-candidates
```

Schneller Smoke-Test:

```bash
docker compose run --rm backtest-runner python analysis/swing_alpha_forward_research.py \
  --start-date 2024-01-01 \
  --end-date 2026-05-01 \
  --frequency weekly \
  --directions LONG,SHORT \
  --no-write-candidates
```

## Outputs

Standardausgabe:

```text
analysis/output/swing_alpha_forward_research/
```

Wichtige Dateien:

- `swing_alpha_forward_summary.csv`: Decile-Auswertung je Score, Richtung und Horizont.
- `swing_alpha_forward_spreads.csv`: Top-Decile minus Bottom-Decile. Das ist die wichtigste Datei.
- `swing_alpha_forward_slices.csv`: Slice-Auswertung nach Richtung, Horizont, Regime, Price-Momentum-Decile und Entry-Pullback-Bucket.
- `swing_alpha_forward_slice_leaders.csv`: Die gleichen Slices, aber nach Edge sortiert und mit `--min-slice-count` gefiltert.
- `swing_alpha_forward_candidates.csv`: Einzelne Kandidaten mit Forward-Return, MAE, MFE und Deciles.
- `swing_alpha_forward_run.json`: Parameter des Research-Laufs.

## Interpretation

Ein brauchbarer Long-Scorer sollte in `swing_alpha_forward_spreads.csv` fuer `LONG` bei `scorer_alpha`, `price_alpha` oder `swing_alpha` positive `top_minus_bottom_return_pct` und idealerweise positive `top_minus_bottom_excess_pct` zeigen. Wenn die Top-Deciles keine bessere Rendite, aber schlechtere MAE zeigen, ist der Score fuer Swing-Entries nicht geeignet.

Ein brauchbarer Short-Scorer sollte bei `SHORT` ebenfalls positive Top-minus-Bottom-Spreads zeigen, weil die directional Scores fuer Shorts invertiert werden. Wenn die Short-Top-Deciles positive absolute Long-Returns oder negative Short-Returns liefern, sind die Scorer-Flags eher Underperformance-Hinweise als echte Naked-Short-Signale.

Das Skript nutzt als Entry die erste 1h-Bar nach `as_of_ts` und ist damit point-in-time konservativ. Default ist `23:59 UTC` pro Sample-Tag.

## Slice-Research

Die Slice-Auswertung beantwortet die naechste Frage: Funktioniert Momentum nur dann, wenn die Aktie nach einem kontrollierten Pullback gekauft wird?

Wichtige Spalten:

- `regime_source`: `market` nutzt das Benchmark-Regime aus `alpaca_market_data_1h_daily_regime_scores`; `world` nutzt optional `world_regime_daily_scores_mv`, falls diese Relation vorhanden ist.
- `regime_label`: z.B. `TREND_UP`, `RANGE`, `UNCLEAR` oder `UNKNOWN`.
- `price_momentum_decile`: Tages-Decile des directional Price-Momentum-Scores.
- `entry_pullback_bucket`: fuer Longs `long_drawdown_00_02`, `long_drawdown_02_05`, `long_drawdown_05_10`, `long_drawdown_10_15`, `long_drawdown_15_plus`; fuer Shorts analog `short_bounce_*`.
- `avg_excess_benchmark_pct`: entscheidend fuer QQQ-Vergleich.

Fuer ein neues Long-Modell sind die besten Kandidaten Slices mit:

- `direction=LONG`
- ausreichendem `count`, mindestens `--min-slice-count`
- positivem `avg_excess_benchmark_pct`
- positiver `avg_mfe_minus_abs_mae_pct`
- akzeptabler `avg_mae_pct`

Wenn die besten Slices eher `price_momentum_decile` 8-10 und `long_drawdown_02_05` oder `long_drawdown_05_10` sind, spricht das fuer ein Momentum-Pullback-Modell. Wenn `long_drawdown_00_02` schlecht bleibt, bestaetigt das, dass Breakout-nahe Entries vermieden werden sollten.
