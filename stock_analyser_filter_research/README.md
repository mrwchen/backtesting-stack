# Stock Analyser Filter Research V3

Dieses eigenstaendige Research-Programm untersucht alle kausalen
`false -> true`-Ereignisse des 8-von-8-Trend-Templates aus
`stock_analyser_trend_template_daily`. Es simuliert weder Portfolio noch
Orders, Kosten oder Slippage. Das Ergebnis ist eine breite, reproduzierbare
Untersuchung von Entry-Ausschluessen, Entry-Bestaetigungen und einfachen
D+1-bis-D+3-Cut-Regeln.

## Signal und Entscheidungszeitpunkt

Ein Signal entsteht nur, wenn dieselbe `(symbol, exchange, cik)` in der
aktuellen globalen Handelssession alle acht Kriterien besteht und in der
unmittelbar vorherigen globalen Session nicht bestanden hat. Beide Sessions
muessen beobachtet sein und zum selben Kontinuitaetssegment gehoeren.
`true -> true`, erste Beobachtungen, Luecken und Segmentwechsel erzeugen kein
neues Signal.

Ein Teil der Merkmale verwendet die vollstaendige Signaltagskerze. Die
Entry-Entscheidung ist daher eine Entscheidung nach dem Close, fruehestens fuer
die naechste Session. SEC-Daten muessen am Signaltag bis 16:00 Uhr
`America/New_York` bekannt sein. Marktindex- und Volatilitaetsdaten werden bis
17:00 Uhr New York beruecksichtigt, weil die auf der Trading VM gespeicherten
finalen Tageswerte erst 30 bis 45 Minuten nach dem Close verfuegbar sind.

## Untersuchte Outcomes

Die Entry-Ausschluesse werden getrennt fuer drei negative Outcomes gesucht:

- `weak_5d`: D+1 bis D+5 erreicht nie +2 % gegenueber dem Signaltags-Close.
- `loss_first_5d`: -5 % wird vor +5 % erreicht.
- `terminal_stagnant_5d`: Der D+5-Close liegt unter +1 %.

Dabei werden die jeweils passenden positiven Klassen geschuetzt:

- `strong_first_5d`: +5 % wird vor -5 % erreicht.
- `terminal_winner_5d`: Der D+5-Close liegt bei mindestens +3 %.

Wenn +5 % und -5 % innerhalb derselben Tageskerze erstmals erreicht werden,
ist die Reihenfolge mit Tagesdaten unbekannt. Der Fall wird als
`same_day_ambiguous` gespeichert und nicht einer Reihenfolgeklasse zugerechnet.

Zusaetzlich sucht V3 unabhaengige Entry-Bestaetigungen fuer
`strong_first_5d` und `terminal_winner_5d`. Sie ueberschreiben keinen
Ausschluss. `include_final` bleibt die gemeinsame Ausschlussentscheidung;
`strong_confirmation` kann zum Priorisieren oder fuer eine spaetere zweite
Research-Stufe verwendet werden.

Gespeichert werden ferner der erste +1-%-, +2-%-, +3-%-, +5-%- und -5-%-Tag,
der D+5-Schlussreturn, MFE/MAE und deren Verhaeltnisse. So ist sichtbar, ob ein
Signal nur intraday kurz steigt oder den Gewinn bis D+5 behaelt.

## Entry-Merkmale A bis V3

Alle Merkmalsgruppen konkurrieren gleichzeitig; es gibt keine irreversible
Abfolge A -> B -> C -> D.

- **A – Signaltag:** Volume/21D, Volume/50D, Notional/21D, Notional/50D,
  Liquiditaet, RS, Triggerkontext, Trendgeometrie, 52-Wochen-Position und
  Kerzenqualitaet.
- **B – Preisverlauf bis D-1:** Rendite und Beschleunigung, ATR, Drawdown,
  Abstand zu vorherigen Hochs, Range-Kompression und RS-Aenderung.
- **C – Aktivitaetsverlauf bis D-1:** kurz-/mittelfristige Volume- und
  Notional-Verhaeltnisse, Up-/Down-Aktivitaet und Preis-Aktivitaets-
  Korrelationen.
- **D – Chartstruktur bis D-1:** Basisbreite, Trendsteigung und R²,
  Trend-Effizienz, Pullback-, Hoch-/Tiefalter, V-Erholung sowie Distribution-,
  Churning- und Failed-Breakout-Tage.
- **T – erweiterte Technik:** 42/63/126/252-Tage-Momentum, Volatilitaets-
  Kontraktion, mehrere Basisbreiten, Tight Closes, MA-Steigungen,
  Overhead-Supply, High Tests, steigende Tiefs, Kontraktionsfolgen und
  Undercut/Reclaim.
- **S – Supply/Demand:** Gap und Intraday-Staerke des Signaltags, SEC-basierte
  Aktienzahl und Turnover aus as-traded Raw Volume.
- **F – Fundamentals:** Profitabilitaet, Margen, Cash Conversion, ROE/ROA,
  Verschuldung, Liquiditaet, Accruals, SBC, Investitionsintensitaet,
  Asset-Qualitaet, Verwässerungs-/Buyback-Proxies sowie Quartalswachstum,
  Beschleunigung, Streaks und Margin-Entwicklung.
- **N – Earnings-Naehe:** Alter eines bestaetigten SEC-8-K-Item-2.02-Ereignisses
  und Ereignisfenster von 0, 5 und 21 Tagen.
- **M – Unternehmensgroesse:** point-in-time Market Cap in USD, logarithmische
  Market Cap und Alter des verwendeten Aktienstands.
- **R – Markt und Leadership:** Marktbreite, Breitenveraenderung, SPY-/QQQ-/
  IWM-/DIA-Momentum, relative Rendite, VIX/VXN/VVIX/SKEW und VIX-Termstruktur.
  Die gespeicherten `cross_sectional_rs_*`-Ränge vergleichen die an demselben
  Tag neu ausgeloesten 8-von-8-Signale; der vorhandene `rs_rating` bleibt der
  breitere Universe-Rang des Stock Analysers.

Die atomare Quantilsuche prueft fuer jedes numerische Merkmal die unteren und
oberen Tails. Zusaetzlich werden 25 fachlich vorab definierte Faktorenpaare in
allen vier Tail-Kombinationen mit `AND` untersucht, unter anderem Market Cap ×
ATR, Market Cap × Volume/Notional, Kompression × Dry-up, Wachstum × RS,
Earnings-Naehe × Gap/Volume und Marktbreite × Leadership.

## Vordefinierte Setups

Neben den atomaren Merkmalen untersucht das Programm nachvollziehbare,
vorab definierte Mehrfaktor-Setups:

- flache Basis, geordneter Aufwaertstrend, Pullback vom Hoch, V-Erholung,
  Volume-Dry-up-Breakout und Distribution Top,
- VCP, High Tight Flag, Bull Flag, Darvas Box, Ascending Triangle,
  Cup-with-Handle, Three-Weeks-Tight und Weinstein Stage 2,
- Pocket Pivot und RS Leader,
- Zanger-Volume-Breakout,
- Growth Leader und Quality Growth,
- Post-Earnings Power und Market-Confirmed Leader,
- feste Market-Cap-/Volumen-Setups fuer Large-Cap-Low-ATR,
  Small-Cap-Bearish-High-Volume und High-Volume-Strong-Close.

Market Cap wird ausserdem in feste, disjunkte Baender unterteilt: Micro unter
300 Mio. USD, Small 300 Mio. bis unter 2 Mrd., Mid 2 bis unter 10 Mrd., Large
10 bis unter 200 Mrd. und Mega ab 200 Mrd. USD. Jedes Band wird allein sowie
mit niedrigem/hohem Volume/21D und Notional/21D untersucht.

Diese Namen bezeichnen messbare, datenbasierte Annaeherungen an Ideen von
Minervini, Ryan/O'Neil, Weinstein, Darvas und Zanger. Sie sind keine Behauptung,
dass ein diskretionaerer Trader einen Chart genauso klassifiziert haette.
Nicht point-in-time abgesicherte Sektorzuordnung, Fondsbesitz, Analysten-
Schaetzungen und Revisionen werden bewusst nicht geraten oder rueckwirkend
verwendet.

Maximal zwei vollstaendige Kandidaten koennen je Outcome mit `OR` verbunden
werden. Jede vordefinierte Mehrfaktor-Regel bleibt intern ein `AND`. Auch
verworfene Zweierkombinationen werden als Kandidatenzeilen gespeichert und in
der Korrektur fuer multiples Testen mitgezaehlt.

## Point-in-time-Fundamentals und Earnings

Fundamentals stammen ausschliesslich aus
`stock_core_sec_fundamentals_asof_daily` und
`stock_core_sec_quarterly_fundamental_events`. Pro Signal wird nur ein
Datensatz verwendet, dessen Periode nicht in der Zukunft liegt und dessen
`sec_data_available_at` beziehungsweise `accepted_at` bis zum Signaltags-Close
bekannt war. Spaetere Amendments bleiben fuer die historische Entscheidung
unsichtbar.

Earnings-Merkmale stammen nur aus bestaetigten
`sec_8k_item_2_02`-Ereignissen in `stock_core_earnings_calendar_events`.
Aktuelle historische Snapshots externer Kalender werden nicht als damaliges
Wissen behandelt. Es wird keine neue Schaetzungsquelle abgefragt.

Es werden keine absoluten Fundamentalbetraege zwischen Aktien verglichen.
Ratios werden nur bei bekannter Berichtswährung gebildet. Das Alter der
verwendeten SEC-Daten bleibt als eigenes Merkmal sichtbar.

`stock_core_market_metrics_daily` wird nur fuer exakte Signal-Keys geladen.
Es gibt kein Forward-Fill. Market Cap wird nur in USD verglichen. Open- und
SEC-Aktienzahl-Merkmale sind davon unabhaengig; nicht SEC-basierte Aktienzahlen
werden nicht fuer Float-/Turnover-Research verwendet.

## Walk-forward, Refit und Multiple Testing

Quantilschwellen werden expanding walk-forward geprueft. Standardmaessig
verwendet jeder Jahresfold 2020 bis 2024 nur fruehere Featurezeilen fuer seinen
Schwellenwert. Signale, deren D+5-Label am Fold-Ende noch nicht vollstaendig
bekannt ist, werden aus diesem Fold gepurgt.

Eine Regel muss Mindestanzahl, Matchrate, Label-Coverage, Objective-Lift und
Retention der geschuetzten Klasse sowohl gepoolt als auch ueber ausreichend
viele Jahresfolds bestehen. Eine zweite OR-Komponente muss den Netto-Score
`Objective-Capture - Protected-Rejection` zusaetzlich verbessern.

Weil V3 sehr viele Merkmale und Kombinationen untersucht, reicht ein einfacher
Lift nicht. Fuer jede Entscheidungsfamilie wird deshalb auf dem konfigurierten
Validation-Zeitraum ein jahresstratifizierter Max-Statistic-Permutationstest
ueber die komplette evaluierbare Kandidatenfamilie gerechnet. Standard sind
999 deterministische Permutationen und eine family-wise Schwelle von 0,05.
`passes_multiple_testing`, Kandidatenzahl, Trial-Zahl und korrigierter
`max_stat_permutation_p_value` stehen in der Regeltabelle.

Nach der Auswahl werden Schwellen bis zum Validation-Ende neu gefittet und
erneut durch Safety-Gates geprueft. Der Zeitraum 2025 bis 20.07.2026 ist nur
`diagnostic`. Die erste globale Handelssession danach beginnt den unangetasteten
`holdout`; Diagnostic und Holdout beeinflussen Auswahl und Schwellen nicht.
Ein positives Research-Ergebnis ist damit robuster, aber noch kein Beweis fuer
zukuenftige Profitabilitaet.

## Early Cut D+1 bis D+3

Zu jedem Signal werden genau drei Landmark-Zeilen gespeichert:

- Entscheidung nach Close D+1, wirksam ab D+2,
- Entscheidung nach Close D+2, wirksam ab D+3,
- Entscheidung nach Close D+3, wirksam ab D+4.

Jede Zeile verwendet nur Informationen bis zu ihrem Close. Neben Preis-,
Volumen-, Notional-, RS- und Trenddaten stehen in V3 auch das bis dahin bekannte
Marktregime und die relative Entwicklung gegen SPY, QQQ und IWM zur Auswahl.
Die verbleibende Entwicklung wird erst ab der jeweils naechsten Session bis D+5
bewertet.

Die Policy wird gemeinsam als D+1 -> D+2 -> D+3-Sequenz ausgewaehlt. Ein Cut
deaktiviert spaetere Landmark-Entscheidungen. Luecken, Segmentwechsel,
Delistings und unvollstaendige Horizonte werden zensiert und nicht als Verlust
oder Erfolg unterstellt. Die Schutzklasse ist weiterhin
`strong_first_to_day5`.

## Datenbanktabellen

Das Init-SQL besitzt ausschliesslich drei serviceeigene Ergebnistabellen:

- `stock_analyser_filter_research_signal_results`: Signale, alle Entry-
  Merkmale und Outcomes sowie Ausschluss und Bestaetigung.
- `stock_analyser_filter_research_early_cut_results`: drei kausale
  Landmark-Zeilen je Signal mit sequenziellem Hold-/Cut-Status.
- `stock_analyser_filter_research_rule_results`: alle atomaren, Pattern-,
  Interaktions- und OR-Kandidaten, Walk-forward-/Stabilitaetsmetriken,
  Multiple-Testing-Ergebnisse sowie Diagnostic und Holdout.

Signal- und Early-Cut-Tabelle sind unkomprimierte Timescale-Hypertables mit
365-Tage-Chunks. Es gibt keine JSON-, Audit- oder Run-Tabelle. Der Runtime-Code
erzeugt oder aendert keine DB-Struktur; er validiert den vollstaendigen Vertrag
und schreibt alle drei zuvor leeren Tabellen atomar.

## Ausfuehrung

V3 ist ein inkompatibler Neuaufbau. Die vorhandenen Research-Tabellen werden
einmalig explizit gedroppt und aus `init/schema.sql` neu erzeugt:

```bash
DROP_ALL_STOCK_ANALYSER_FILTER_RESEARCH_TABLES_ON_START=true \
  docker compose up --build --force-recreate stock-analyser-filter-research
```

Der Drop-Schalter ist standardmaessig `false`. Ohne expliziten Drop bricht das
Programm bei einer alten Struktur oder bereits vorhandenen Zielzeilen ab.

Die Berechnung nutzt disjunkte, nach Quellzeilenzahl balancierte
Aktienpartitionen. Jeder Worker-Prozess besitzt eine eigene DB-Verbindung mit
AppName und importiert denselben exportierten PostgreSQL-Snapshot. Nur der
Hauptprozess schreibt die drei Ergebnistabellen. Wegen der breiten Suche und
999 Permutationen ist V3 bewusst deutlich rechenintensiver als V2.

## Tests

```bash
python -m pytest -q
docker compose config
```
