# swing-stocks-wei — Regime-gated EMA Index Backtester

Backtestet eine Long/Flat-Strategie auf einem S&P-500-Proxy (default: VOO):
ein EMA-Trendfilter, dessen Ausstiegssignal durch den Makro-Stress-Score aus
`world_regime_daily_scores_mv` gegated wird.

## Strategie

Jeden Handelstag zum Schlusskurs eine Entscheidung:

```
flat  wenn  EMA9 < EMA21  UND  Stress-Ampel ROT
long  sonst
```

**Trendfilter:** EMA9 vs. EMA21 auf Tages-Schlusskursen (`alpaca_market_data_1day`,
split-adjusted).

**Stress-Ampel (Hysterese):** getrieben vom `composite_score` (0–100) der
`world_regime_daily_scores_mv`, immer mit dem letzten Score **vor** dem
Handelstag (der Score fuer Kalendertag d hat seinen Cutoff erst um 05:00 UTC
an d+1 — das Lag verhindert Look-ahead):

- Ampel GRUEN und Score ≥ `STRESS_ENTER` (57) → ROT
- Ampel ROT und Score ≤ `STRESS_EXIT` (52) → GRUEN
- dazwischen: Zustand bleibt

Die Kombination filtert die Fehlsignale beider Einzelsignale: EMA-Whipsaws in
ruhigen Maerkten werden ignoriert (kein Stress → long bleiben), und nach
Crashs holt die schneller entspannende Score-Ampel frueher zurueck in den
Markt als der EMA-Cross. Keine Shorts — die Short-Seite hat in allen Tests
auf dem S&P 500 Geld verloren.

## Historische Ergebnisse (VOO, Schlusskurs-Ausfuehrung, 5 bps/Seite)

| Zeitraum | Strategie | Buy & Hold | MaxDD Strategie | MaxDD B&H | Trades |
|---|---|---|---|---|---|
| 2020-01 – 2021-12 (out-of-sample) | +59,9 % | +46,3 % | −9,4 % | −34,3 % | 5 |
| 2022-01 – 2026-07 (Auswahlzeitraum) | +57,6 % | +57,2 % | −13,8 % | −25,4 % | ~23 |

Charakter der Strategie: Buy-&-Hold-Rendite bei einem Drittel bis der Haelfte
des Drawdowns — kein Rendite-Booster. Die Schwellen 57/52 wurden auf
2022–2026 selektiert; 2020–2021 war echter Out-of-sample-Test der Parameter.

**Caveats:** Die Regime-MV wird mit heutiger Methodik rueckwirkend berechnet
(moeglicher Hindsight-Bias in den Score-Gewichten). Flat-Phasen sind mit 0 %
verzinst (real: Geldmarktzins). Ausfuehrung zum selben Schlusskurs wie das
Signal ist eine Idealisierung.

## Ablauf

1. `init-wei-schema` (postgres:17-alpine) legt die Ergebnistabellen an
   (`DROP_ALL_WEI_TABLES_ON_START=true` fuer destruktive Neuanlage).
2. `wei-backtester` laedt Preise + Scores (inkl. `WARMUP_CALENDAR_DAYS`
   EMA-Warmup vor `START_DATE`), berechnet Signale, simuliert und persistiert.

```bash
docker compose up --build
```

Tests (lokal, ohne DB):

```bash
python -m pytest tests/
```

## Ergebnistabellen

Der Tabellen-Praefix ist ueber `TABLE_PREFIX` konfigurierbar (default
`backtest_wei_`; muss in beiden compose-Services identisch gesetzt sein):

| Tabelle | Inhalt |
|---|---|
| `<TABLE_PREFIX>runs` | ein Datensatz pro Run: Parameter + Kennzahlen (Total, CAGR, MaxDD, Trades, investierte Tage; jeweils inkl. Buy-&-Hold-Vergleich) |
| `<TABLE_PREFIX>trades` | alle Long-Trades eines Runs (Entry/Exit-Datum und -Kurs, Brutto-Rendite, Haltedauer, offen-Flag) |
| `<TABLE_PREFIX>equity_daily` | Tageszustand (Hypertable): Close, EMAs, gelaggter Score, Ampel, Position, Equity-Kurve + B&H-Kurve — direkt in Grafana plottbar |

Equity-Kurve ist **netto** (Kosten pro Positionswechsel), Trade-Renditen sind
**brutto** (Kurs zu Kurs).

## Parameter (compose.yaml)

| Env | Default | Bedeutung |
|---|---|---|
| `PRICES_TABLE` | `alpaca_market_data_1day` | Kurstabelle (braucht Spalten `symbol`, `ts`, `close`; split-adjusted) |
| `SCORES_TABLE` | `world_regime_daily_scores_mv` | Score-Quelle (braucht Spalten `day`, `composite_score`) |
| `SYMBOL` | `VOO` | gehandeltes Symbol aus `PRICES_TABLE` |
| `START_DATE` / `END_DATE` | `2022-01-03` / heute | Auswertungsfenster |
| `EMA_FAST` / `EMA_SLOW` | `9` / `21` | Trendfilter-Spannen |
| `STRESS_ENTER` / `STRESS_EXIT` | `57` / `52` | Hysterese-Schwellen der Stress-Ampel |
| `COST_BPS_PER_SIDE` | `5` | Transaktionskosten je Kauf/Verkauf |
| `WARMUP_CALENDAR_DAYS` | `365` | Preishistorie vor START_DATE fuer EMA-Warmup |
| `RUN_LABEL` | `wei_regime_ema_9_21_57_52` | Freitext-Label des Runs |
| `TABLE_PREFIX` | `backtest_wei_` | Praefix der Ergebnistabellen (Init-Container **und** Runner; nur Kleinbuchstaben, Ziffern, Unterstriche) |
