# Swing Stock Momentum Backtest

Deterministischer Portfolio-Backtest mit point-in-time abgegrenzten Quelldaten
und der unten erläuterten idealisierten Same-Close-Ausführung für die
Ergebnisse von `stock_analyser`. Der Lauf beginnt am ersten vorhandenen
New-York-Handelstag ab dem 01.01.2026 mit 30.000 USD und endet am letzten
vollständig zwischen Core-Daten und Analyzer abgeglichenen Handelstag.

## Entry

Eine Zeile ist zunächst Kandidat, wenn alle acht Trend-Template-Kriterien wahr
sind, `daily_price_change_pct >= 1` und `< 5` ist sowie
`adjusted_volume_vs_sma21_prior_ratio > 1.2` ist. Zusätzlich müssen die zehn
unmittelbar vorherigen globalen Handelssessions für dieselbe Identität und
denselben `price_continuity_segment` vollständig sein. Kein `adjusted_high`
dieser zehn Sessions darf mehr als 10 % über dem adjustierten Close des
Signaltages liegen.

Aus allen gültigen Kandidaten werden bei freien Portfolio-Slots zuerst das
höhere relative Volumen, dann die höhere Tagesrendite und schließlich Symbol,
Börse und CIK aufsteigend berücksichtigt. Maximal zwei Symbole können
gleichzeitig offen sein. Eine Position wird mit ganzen Aktien zum adjustierten
Close eröffnet. Das Stückzahlrisiko einschließlich konfigurierbarer Kosten und
Slippage überschreitet beim anfänglichen Stop nie 1 % des jeweils aktuellen
Account-Equity. Da das Konto `unlevered` ist, begrenzt das vorhandene Cash die
Stückzahl zusätzlich.

Die Symbol-Eindeutigkeit gilt für das aktuell offene Portfolio. Nach einem
Verkauf darf dasselbe Symbol bei einem späteren gültigen Signal erneut gekauft
werden.

Der ausdrücklich gewünschte Kauf anhand des vollständigen Signaltages zum
Close ist eine idealisierte Same-Close-Ausführung: Tagesrendite und
Tagesvolumen stehen endgültig erst mit diesem Close fest. Ohne intraday
verfügbaren, vor Auktionsschluss berechneten Signalstand kann das einen
Execution-Look-ahead erzeugen und das Ergebnis optimistisch machen. Für einen
direkt handelbaren Folgetest sollte der Entry auf den nächsten Open gelegt
werden; dieses Framework bildet zunächst bewusst die angeforderte
Same-Close-Regel ab und kennzeichnet sie im Run-Datensatz.

## Exit-Reihenfolge

Der erste vorhandene Bar nach dem Kauf ist D+1. An jedem gehaltenen
Handelstag gilt strikt:

1. Liegt der Open bereits jenseits von Stop oder Take Profit, erfolgt der
   Verkauf zum Open.
2. Berührt dieselbe Tageskerze Low und High beide Levels, gewinnt konservativ
   das Low und damit der Stop. Sonst wird am zuerst geprüften Stop- oder
   Take-Profit-Level verkauft.
3. Ist die Position danach noch offen, wird sie an D+1 bei ATR/14 <= 1,5 %
   beziehungsweise an D+2 bei ATR/14 <= 2 % zum Close verkauft.

ATR/14 ist der einfache Mittelwert der True Range der aktuellen und 13
vorherigen vollständigen, zusammenhängenden Handelssessions, geteilt durch den
aktuellen adjustierten Close. Die anfänglichen Levels liegen bei -5 % und
+10 % vom Entry-Fill. Nach jedem vollständigen Fünf-Tage-Block steigen sie ab
der nächsten Session erneut um 5 beziehungsweise 7,5 Prozentpunkte: D+1 bis
D+5 `-5/+10`, D+6 bis D+10 `0/+17,5`, D+11 bis D+15 `+5/+25` und so weiter.

Offene Positionen werden am Datenende nicht künstlich geschlossen. Sie werden
zum letzten verfügbaren adjustierten Close bewertet und getrennt als offen
ausgewiesen. Provision und Slippage sind in Basispunkten konfigurierbar und
standardmäßig beide null.

## Point-in-time- und Datenvertrag

Die Ausführung verwendet `adjusted_open/high/low/close` aus
`stock_core_market_metrics_daily`. Dadurch bleiben Splits und die vom Analyzer
verwendete Preisbasis konsistent. Der Backtest bricht ab, wenn der
`stock_analyser`-Checkpoint nicht exakt dem Core-Quell-Watermark entspricht
oder im Backtestzeitraum Identitätszeilen fehlen. So kann kein teilweise
aktualisierter Tag unbemerkt verwendet werden.

Die Forward-Outcome-Spalten des Analyzers werden ausschließlich zur vom Nutzer
gewünschten Nachvollziehbarkeit in die Signaltabelle kopiert. Keine Entry-,
Exit-, Ranking- oder Sizing-Entscheidung liest diese Spalten. Spätere
Fundamentaldaten werden ebenfalls nicht nachträglich verbunden.

## Ergebnistabellen

- `backtest_momentum_runs`: Ergebniskennzahlen, Source-Watermarks und alle
  Strategie- sowie Analyzer-Konfigurationsparameter.
- `backtest_momentum_signals`: jeder Entry-Kandidat mit Auswahlentscheidung,
  Sizing-Zwischenwerten und der vollständigen Analyzer-Quellzeile.
- `backtest_momentum_trades`: offene und geschlossene Positionen mit Fill,
  Exitgrund, Risiko, Kosten und P&L.
- `backtest_momentum_equity_daily`: tägliches Cash, Marktwert, Equity, P&L und
  Drawdown.

Signal- und Equity-Tabelle sind unkomprimierte Timescale-Hypertables mit
365-Tage-Chunks. Die anderen beiden Tabellen sind normale PostgreSQL-Tabellen.
Die Runtime erzeugt oder verändert keine Datenbankobjekte; das geschieht nur
in `init/schema.sql`. Der Init-Schalter
`DROP_ALL_BACKTEST_MOMENTUM_TABLES_ON_START` ist standardmäßig `false`.

Jeder vollständige Lauf wird unter einer neuen UUID in einer Transaktion
geschrieben. Ein PostgreSQL Advisory Lock verhindert parallele Portfolio-Läufe.
Alle DB-Verbindungen übergeben `PGAPPNAME`. Logs verwenden UTC und das Format
`timestamp level process thread message`.

## Ausführung

```powershell
docker compose up --build --abort-on-container-exit
```

Der Backtest ist ein einmaliger Batch-Lauf und kein Scheduler. Vor einem Lauf
muss `stock_analyser` seinen aktuellen Zyklus vollständig abgeschlossen haben.

Tests und Compose-Validierung:

```powershell
python -m pytest -q
docker compose config
```
