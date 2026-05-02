# NAS100 Backtester

## Start
Bearbeite alle Variablen direkt in `docker-compose.yml` und starte dann:

```bash
docker compose up --build
```

## Wohin geschrieben wird
Der Lauf schreibt an zwei Stellen:

1. **Dateien lokal im Projektordner `./output`**
   - `summary.json`
   - `trades.csv`

2. **Trades in `backtesting.public.bt_trade_history`**
   - `account_number` kommt aus `ACCOUNT_NUMBER`
   - `account_type` kommt aus `ACCOUNT_TYPE`
   - `closing_deal_id` und `position_id` werden automatisch fortlaufend ab dem aktuellen Tabellenmaximum vergeben

## Wichtige Compose-Variablen
- `START_TIME_UTC` und `END_TIME_UTC`
  - leer = ganze verfügbare Historie
  - sonst UTC-ISO-Zeit, z. B. `2024-01-01T00:00:00Z`
- `ACCOUNT_NUMBER`
  - Standard: `00001`
- `ACCOUNT_TYPE`
  - Standard: `backtester`

## Strategie-Regeln
- Entry-Signale kommen aus `backtesting.public.bt_signal`
- Gültige Actions: `buy`, `sell`
- Entry-Fill = Open der ersten `market_data_1min`-Kerze mit `bar_time >= bt_signal.event_time`
- Regime-Filter kommt aus `backtesting.public.bt_regime`
- Long nur wenn `is_strong_long OR is_weak_long`
- Short nur wenn `is_strong_short OR is_weak_short`
- Maximal eine offene Position gleichzeitig
- Positionsgröße = 45% der maximal möglichen margin-basierten Größe
- Lots werden auf 0.1 abgerundet
- Risiko = 2% der eingesetzten Margin
- TP = CRV 2:1
- Exit bei SL, TP, Regimewechsel oder Datenende
- Spread ist konfigurierbar; Standard = 1.5 Punkte
- Wenn SL und TP in derselben 1m-Kerze getroffen werden, gewinnt **SL**
- Keine Kommission, keine Slippage, keine Finanzierungskosten

## Mapping nach `bt_trade_history`
- `trade_type` = `buy` oder `sell`
- `quantity_lots` = Backtest-Volume
- `volume_in_units` = derselbe Wert wie `quantity_lots` für dieses NAS100-Setup
- `gross_profit` = `net_profit`, weil Kommission und Swap = 0
- `pips` = realisierte Punkte
- `balance` = Equity nach Trade-Schließung

## Wichtige Annahmen
- `ticker = symbol`
- Preisquelle ist `market-data.public.market_data_1min`
- `1.0 lot = 1 USD pro Punkt` für NAS100
- Regimewechsel wird auf der ersten 1m-Kerze mit `bar_time >= bt_regime.event_time` wirksam
