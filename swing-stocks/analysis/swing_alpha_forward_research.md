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
- `swing_alpha_forward_candidates.csv`: Einzelne Kandidaten mit Forward-Return, MAE, MFE und Deciles.
- `swing_alpha_forward_run.json`: Parameter des Research-Laufs.

## Interpretation

Ein brauchbarer Long-Scorer sollte in `swing_alpha_forward_spreads.csv` fuer `LONG` bei `scorer_alpha`, `price_alpha` oder `swing_alpha` positive `top_minus_bottom_return_pct` und idealerweise positive `top_minus_bottom_excess_pct` zeigen. Wenn die Top-Deciles keine bessere Rendite, aber schlechtere MAE zeigen, ist der Score fuer Swing-Entries nicht geeignet.

Ein brauchbarer Short-Scorer sollte bei `SHORT` ebenfalls positive Top-minus-Bottom-Spreads zeigen, weil die directional Scores fuer Shorts invertiert werden. Wenn die Short-Top-Deciles positive absolute Long-Returns oder negative Short-Returns liefern, sind die Scorer-Flags eher Underperformance-Hinweise als echte Naked-Short-Signale.

Das Skript nutzt als Entry die erste 1h-Bar nach `as_of_ts` und ist damit point-in-time konservativ. Default ist `23:59 UTC` pro Sample-Tag.
