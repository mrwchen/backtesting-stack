# Swing Stock Momentum Backtest

Deterministischer Portfolio-Backtest mit grundsätzlich point-in-time
abgegrenzten Quelldaten, der unten ausdrücklich beschriebenen retrospektiven
Earnings-Ausnahme und einer idealisierten Same-Close-Ausführung für die
Ergebnisse von `stock_analyser`. Der Lauf beginnt am ersten vorhandenen
New-York-Handelstag ab `BACKTEST_START_DATE` mit dem über
`STARTING_CAPITAL_USD` konfigurierten Kapital und endet am letzten vollständig
zwischen Core-Daten und Analyzer abgeglichenen Handelstag.

## Entry

Eine Zeile ist zunächst Kandidat, wenn alle acht Trend-Template-Kriterien wahr
sind, `daily_price_change_pct` innerhalb der konfigurierten Grenzen liegt und
`adjusted_volume_vs_sma21_prior_ratio` die konfigurierte Untergrenze
überschreitet. Zusätzlich müssen genau so viele unmittelbar vorherige globale
Handelssessions vollständig sein, wie `PRIOR_HIGH_LOOKBACK_SESSIONS` vorgibt.
Kein `adjusted_high` dieser Sessions darf den mit
`PRIOR_HIGH_MAX_ABOVE_SIGNAL_CLOSE_PCT` festgelegten Abstand über dem
adjustierten Signal-Close überschreiten.

Ein Einstieg wird außerdem verworfen, wenn für dieselbe Identität ein
bestätigtes SEC-8-K-Earnings-Ereignis in D+1 bis einschließlich der mit
`EARNINGS_BLACKOUT_SESSIONS` konfigurierten Handelssession liegt. Die
Standardgrenze beträgt zehn Sessions. Maßgeblich ist der globale
New-York-Handelskalender der vollständig vorhandenen Marktdaten. Sind am
Datenende weniger als zehn folgende Sessions vorhanden, werden konservativ
keine neuen Positionen mehr eröffnet.

Aus allen gültigen Kandidaten werden bei freien Portfolio-Slots zuerst das
höhere relative Volumen, dann die höhere Tagesrendite und schließlich Symbol,
Börse und CIK aufsteigend berücksichtigt. Pro Handelstag werden höchstens zwei
neue Positionen eröffnet; gleichzeitig können maximal fünf Symbole offen sein.
Beide Grenzen werden getrennt mit `MAX_NEW_POSITIONS_PER_DAY` und
`MAX_POSITIONS` konfiguriert. Eine Position wird mit ganzen Aktien zum
adjustierten Close eröffnet. Das Stückzahlrisiko einschließlich
konfigurierbarer Kosten und Slippage überschreitet beim anfänglichen Stop nie
1 % des jeweils aktuellen Account-Equity. Da das Konto `unlevered` ist,
begrenzt das vorhandene Cash die Stückzahl zusätzlich.

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

Der ATR ist der einfache Mittelwert der True Range über die mit
`ATR_PERIOD_SESSIONS` konfigurierte Zahl vollständiger, zusammenhängender
Handelssessions, geteilt durch den aktuellen adjustierten Close. Die
anfänglichen Levels liegen bei -5 % und
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

Der Earnings-Filter ist eine vom Nutzer bewusst gewählte Ausnahme von diesem
Point-in-time-Prinzip. Er verwendet bestätigte tatsächliche Termine aus
`stock_core_earnings_calendar_events` mit
`source = 'sec_8k_item_2_02'`, auch wenn diese Information am früheren
Signaltag noch nicht veröffentlicht war. Dadurch enthält das Ergebnis beim
Earnings-Filter ausdrücklich Look-ahead Bias. Policy, Quelle, Fensterlänge,
gefundener Termin und Sessionabstand werden in den Ergebnistabellen
mitgeschrieben.

Für große Zeiträume werden die Quelldaten innerhalb desselben wiederholbar
lesbaren DB-Snapshots über zwei getrennte Ströme verarbeitet. Der breite
Analyzer-Datensatz wird bereits in PostgreSQL auf tatsächliche Entry-Kandidaten
gefiltert. Der vollständige Analyzer-Inhalt wird daher nur für diese Kandidaten
geladen. Alle übrigen Zeilen laufen als kompakter Kursstrom mit den für
Bewertung, ATR und vorheriges Hoch erforderlichen zwölf Feldern. Datum, Symbol,
Börse und CIK verbinden beide Ströme deterministisch. Dadurch bleiben
Entscheidungen und Nachvollziehbarkeit unverändert, ohne die breiten
Analyzer-Werte für jede Nicht-Signal-Zeile nach Python zu übertragen.

## Ergebnistabellen

- `backtest_momentum_runs`: Ergebniskennzahlen, Source-Watermarks und alle
  Strategie- sowie Analyzer-Konfigurationsparameter.
- `backtest_momentum_signals`: jeder Entry-Kandidat mit Auswahlentscheidung,
  Earnings-Horizont und -Treffer, Sizing-Zwischenwerten und der vollständigen
  Analyzer-Quellzeile.
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
