# Stock Analyser Filter Research

Dieses eigenstaendige Research-Programm untersucht alle kausalen
`false -> true`-Ereignisse des 8-von-8-Trend-Templates aus
`stock_analyser_trend_template_daily`. Es simuliert kein Portfolio und
beruecksichtigt weder Orders, Kosten noch Slippage. Das Ergebnis ist eine kleine,
interpretierbare Menge von Ausschlussregeln fuer Signale, die in den folgenden
fuenf Handelssessions schwach bleiben oder stark verlieren.

## Research-Vertrag

Ein Signal entsteht nur, wenn dieselbe `(symbol, exchange, cik)` in der aktuellen
globalen Handelssession alle acht Kriterien besteht und in der unmittelbar
vorherigen globalen Session nicht bestanden hat. Fehlende Sessions,
Kontinuitaetssegment-Wechsel, erste Beobachtungen und `true -> true` erzeugen kein
neues Signal.

Die Version 1 untersucht drei Stufen:

- **A – Zustand am Signaltag:** unter anderem Volume/21d Avg, Volume/50d Avg,
  Notional/21d Avg, Notional/50d Avg, Liquiditaet, RS-Rating sowie Abstaende zu
  gleitenden Durchschnitten und 52-Wochen-Grenzen.
- **B – vorheriger Chartverlauf:** Renditen, Momentum-Beschleunigung, ATR,
  Drawdown, Position im vorherigen Kursbereich, Range-Kompression und
  RS-Veraenderung. Historische Fenster enden ausnahmslos am Vortag.
- **C – vorheriger Aktivitaetsverlauf:** kurz-/mittelfristige Volume- und
  Notional-Verhaeltnisse, Up-Session-Anteile sowie Preis-Aktivitaets-Korrelationen.

Fundamentaldaten sind bewusst noch nicht Teil dieser ersten, einfachen Version.
Sie koennen spaeter als Stufe D ergaenzt werden, sobald A-C belastbare
Out-of-sample-Ergebnisse liefern.

Die Zielvariablen beziehen sich auf den Adjusted Close des Signaltags:

- `weak_5d`: maximaler 5-Session-Gewinn `< 2 %`
- `strong_5d`: maximaler 5-Session-Gewinn `>= 5 %`
- `deep_loss_5d`: maximaler 5-Session-Verlust `<= -5 %`
- `bad_5d`: `weak_5d OR deep_loss_5d`

Unvollstaendige Forward-Horizonte bleiben `NULL` und werden nicht zur
Regelauswahl verwendet.

## Schutz vor Look-ahead und Overfitting

Schwellenwerte werden nur aus dem Discovery-Zeitraum bis 2022-12-31 gewonnen.
Eine Regel muss danach sowohl in Discovery als auch in Validation (2023-2024)
Mindestbedingungen fuer Bad-Trade-Lift, Ausschlussrate und Erhalt starker Signale
erfuellen. Pro Stufe wird hoechstens eine einfache Bedingung gewaehlt; die finale
Entscheidung ist `A OR B OR C`. Daten ab 2025 bleiben bis zur fertigen Auswahl
unangetasteter Testbestand.

Die jeweils letzten fuenf globalen Handelssessions vor dem Ende von Discovery
und Validation tragen den Split `purged`: Ihre Signale werden gespeichert, aber
ihre erst im folgenden Zeitraum bekannten 5-Session-Ergebnisse duerfen nicht in
die Schwellen- oder Regelauswahl einfliessen.

Standardbedingungen:

- mindestens `max(50, 1 %)` gelabelte Signale ausgeschlossen
- hoechstens 35 % gelabelte Signale ausgeschlossen
- mindestens 90 % der starken Signale behalten
- mindestens 90 % vollstaendige 5-Tage-Labels unter den gematchten Signalen
- Bad-Trade-Lift mindestens 1,05
- jede weitere Stufe verbessert die Bad-Trade-Capture-Rate in Discovery und
  Validation um mindestens einen Prozentpunkt

Diese Grenzen sind Research-Hyperparameter und stehen explizit in `compose.yaml`.

## Datenbanktabellen

Das Init-SQL erzeugt ausschliesslich:

- `stock_analyser_filter_research_signal_results`: unkomprimierte
  Timescale-Hypertable mit 365-Tage-Chunks, allen Signalfeatures, Labels und der
  finalen Include-/Exclude-Entscheidung.
- `stock_analyser_filter_research_rule_results`: Kandidaten-, Quantil- und
  ausgewaehlte Regelmetriken fuer Discovery, Validation, Test, Gesamtbestand und
  Kalenderjahre. Signal-, Unlabeled- und Coverage-Zaehler machen unvollstaendige
  Forward-Horizonte sichtbar.

Es gibt keine JSON-Spalten und keine Audit-/Run-Tabelle. Runtime-Code erzeugt oder
veraendert keine DB-Struktur; er validiert den Schema-Vertrag und schreibt beide
leeren Zieltabellen atomar. Sind Zielzeilen vorhanden, bricht der Lauf ab.

## Ausfuehrung

Auf der Trading-VM kann der erste Lauf beide Tabellen ohne Drop erzeugen und
danach direkt befuellen:

```bash
docker compose up --build --force-recreate stock-analyser-filter-research
```

Ein spaeterer vollstaendiger Neuaufbau benoetigt explizit den bereits
freigegebenen Drop-Schalter:

```bash
DROP_ALL_STOCK_ANALYSER_FILTER_RESEARCH_TABLES_ON_START=true \
  docker compose up --build --force-recreate stock-analyser-filter-research
```

Der Prozess nutzt nach Zeilenanzahl balancierte, disjunkte Aktienpartitionen.
Jeder Worker-Prozess besitzt eine eigene DB-Verbindung; nur der Hauptprozess
schreibt. Alle Worker importieren denselben exportierten PostgreSQL-Snapshot und
sehen dadurch exakt denselben Source-Stand. Die Anzahl der Prozesse und die Identity-Batchgroesse sind ueber
`MAX_WORKERS` und `WORKER_IDENTITY_BATCH_SIZE` steuerbar.

## Tests

```bash
python -m pytest -q
```
