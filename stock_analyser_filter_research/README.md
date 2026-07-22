# Stock Analyser Filter Research V5

Dieses eigenstaendige Research-Programm untersucht alle kausalen
`false -> true`-Ereignisse des 8-von-8-Trend-Templates aus
`stock_analyser_trend_template_daily`. Es simuliert weder Portfolio noch
Orders, Kosten oder Slippage. Das Ergebnis ist eine breite, reproduzierbare
Untersuchung von Entry-Ausschluessen, Entry-Bestaetigungen, fruehen
D+1-bis-D+3-Cut-Regeln und Positionsentscheidungen an D+5, D+20 und D+30.
Die Outcome-Pfade reichen bis D+90.

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

Fuer das konkrete Halteziel kommen weitere Outcomes hinzu:

- `stagnant_5d`: Bis D+5 wurde nie +2 % erreicht und der D+5-Close liegt bei
  hoechstens 0 %.
- `hard_stop_10pct_5d`: Mindestens ein Tagestief bis D+5 beruehrt -10 %.
- `terminal_nonpositive_20d` und `terminal_nonpositive_30d`: Der Schlusskurs
  liegt am jeweiligen Horizont nicht ueber dem Signaltags-Close.
- `terminal_winner_20d` und `terminal_winner_30d`: Der Schlussreturn erreicht
  mindestens +5 %.
- `runner_60d` und `runner_90d`: Der Schlussreturn erreicht mindestens +5 %
  und der Pfad hat zuvor nicht den -10-%-Hard-Stop beruehrt.

Zusaetzlich sucht V5 unabhaengige Entry-Bestaetigungen fuer die positiven
D+5-, D+20-, D+30-, D+60- und D+90-Outcomes. Sie ueberschreiben keinen
Ausschluss. `include_final` bleibt die gemeinsame Ausschlussentscheidung;
`strong_confirmation` kann zum Priorisieren oder fuer eine spaetere zweite
Research-Stufe verwendet werden.

Gespeichert werden ferner der erste +1-%-, +2-%-, +3-%-, +5-%-, -5-%- und
-10-%-Tag, Schlussreturns sowie MFE/MAE fuer D+5, D+10, D+20, D+30, D+40,
D+60 und D+90. So ist sichtbar, ob ein Signal nur intraday kurz steigt, spaet
anspringt oder seine Staerke laenger behaelt.

Der Hard-Stop ist mit Tagesdaten ein Schwellenwert-Outcome: Ein Low bei oder
unter -10 % gilt als Stop-Treffer. Ein exakter Fill bei -10 % wird damit nicht
behauptet; insbesondere kann ein Gap unter das Stop-Niveau mit Tagesdaten nicht
realistisch ausgefuehrt werden.

## Entry-Merkmale A bis V5

Alle Merkmalsgruppen konkurrieren gleichzeitig; es gibt keine irreversible
Abfolge A -> B -> C -> D.

- **A – Signaltag:** Volume und Notional jeweils relativ zum vorherigen
  7-/14-/21-/50-/100-Tage-Durchschnitt, Liquiditaet, RS, Triggerkontext,
  Trendgeometrie, 52-Wochen-Position und Kerzenqualitaet.
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
- **P – weiche Chartmuster:** kausale Setup-, Trigger- und Gesamtscores von
  0 bis 100 fuer Flat Base, Ordered Uptrend, Pullback, V-Recovery,
  Volume-Dry-up-Breakout, Distribution Top, VCP, Cup-with-Handle und
  High-Tight-Flag. Je nach Muster werden 10, 15, 20, 30, 40, 63 und 126
  Handelstage parallel untersucht. Fuer die sechs aktivitaetsabhaengigen
  Muster Pullback, Volume-Dry-up-Breakout, Distribution Top, VCP,
  Cup-with-Handle und High-Tight-Flag existiert zusaetzlich eine getrennte
  USD-Notional-Variante; die klassischen Scores behalten Stueckvolumen.
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
- **G – aktuelle IBKR-Gruppen (nur diagnostisch):** relative Gruppenstaerke,
  Anteil ueber MA50/MA200, neue 52-Wochen-Hochs, Leadership-Breite und neue
  8-von-8-Signale. Die Analyse erfolgt getrennt auf `Industry`, dem
  vollstaendigen Pfad `Industry -> Category` und dem vollstaendigen Pfad
  `Industry -> Category -> Subcategory`.

Die atomare Quantilsuche prueft fuer jedes numerische Merkmal die unteren und
oberen Tails. Zusaetzlich werden 31 fachlich vorab definierte Faktorenpaare in
allen vier Tail-Kombinationen mit `AND` untersucht, unter anderem Market Cap ×
ATR, Market Cap × Volume/Notional, Kompression × Dry-up, Wachstum × RS,
Earnings-Naehe × Gap/Volume, Marktbreite × Leadership sowie fuer jedes der
sechs aktivitaetsabhaengigen Muster der klassische Volume-Score × sein
Notional-Gegenstueck.

Fuer alle direkten Volume-/Notional-Ratios der Fenster 7, 14, 21, 50 und 100
werden neben Quantilen feste Grenzen geprueft: 0,5x, 0,75x, 1x, 1,25x, 1,5x,
2x, 3x, 5x und 10x, jeweils als Unter- und Obergrenze. Damit wird zum Beispiel
`Vol/21D <= 1` explizit getestet und nicht nur indirekt ueber ein Quantil.

## IBKR-Industry-Hierarchie: diagnostischer Backcast

Die Klassifikation stammt aus dem aktuellen Snapshot
`stock_core_security_master_current`. Fuer die historische Research-Periode
liegt noch keine damalige, point-in-time gueltige Industry-Historie vor.
Deshalb wird die heutige Zuordnung rueckwirkend nur als Hypothesengenerator
verwendet. Sie darf unabhaengig von Lift, Stabilitaet oder korrigiertem
p-Wert weder `include_final` noch `strong_confirmation` beeinflussen und wird
in der Regeltabelle immer als nicht ausgewaehlter `candidate_rule` gespeichert.

Pro Handelstag und Hierarchieebene werden folgende Gruppenmerkmale berechnet:

- Median-Rendite ueber 21, 63, 126 und 252 Sessions sowie der taegliche Rang
  dieses Gruppenmedians gegen die anderen Gruppen derselben Ebene,
- Median von `rs_raw`, Gruppenrang dieses Medians und der RS-Rang der einzelnen
  Aktie innerhalb ihrer Gruppe,
- Anteil der Gruppenmitglieder ueber MA50 und MA200 samt Veraenderung ueber
  exakt 5 und 21 globale Handelssessions,
- Anzahl und Anteil neuer 52-Wochen-Hochs samt 5-/21-Session-Aenderung,
- Anteil der Aktien mit RS >= 70, RS >= 90 und bestandenem 8-von-8-
  Trend-Template,
- Leadership-Breite als Mittelwert dieser drei Anteile samt 5-/21-Session-
  Aenderung sowie Anzahl und Anteil neuer `false -> true`-8-von-8-Signale.

Kleine Gruppen werden merkmalsspezifisch auf `NULL` gesetzt: mindestens 20
auswertbare Mitglieder fuer Industry, 10 fuer den Category-Pfad und 5 fuer
den Subcategory-Pfad. Dadurch kann beispielsweise eine Gruppe mit 20 Aktien,
aber nur 14 vorhandenen MA200-Werten fuer das MA200-Merkmal trotzdem nicht als
ausreichend gross erscheinen.

Neben allen atomaren Quantil-Tails werden Kombinationen wie Gruppen-RS x
MA-Breite, Rendite x Breite, RS x neue Hochs, Aktie-in-Gruppe-RS x
Gruppenstaerke, Breitenrichtung x Leadership, Gruppenstaerke x Volume/21D und
gruppenuebergreifende Industry-/Category-/Subcategory-Uebereinstimmung
untersucht. Diese Kandidaten durchlaufen dieselben Walk-forward-, Stabilitaets-
und Max-Statistic-Pruefungen wie die kausalen Merkmale; die erzwungene
Nichtauswahl verhindert lediglich ihre operative Verwendung.

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

Maximal drei vollstaendige Kandidaten koennen je Outcome mit `OR` verbunden
werden. Die bisherigen vollstaendigen `AND`-Muster bleiben als Kontrollgruppe
erhalten. Fuer Drei-Faktor-Muster wird zusaetzlich 2-von-3 und fuer
Vier-Faktor-Muster 2-von-4 sowie 3-von-4 getestet. Fehlende Komponenten gelten
nicht als erfuellt; eine k-von-n-Regel benoetigt weiterhin alle Eingangsdaten.
Auch verworfene, tatsaechlich getestete Kombinationen werden als
Kandidatenzeilen gespeichert und in der Korrektur fuer multiples Testen
mitgezaehlt.

Die weichen Scores trennen das bis D-1 bekannte Setup vom Signaltag-Trigger.
Die Gesamtscores kombinieren beide Teile gewichtet, ohne eine spaetere Kerze
zu verwenden. Neben kausal gefitteten Quantilen prueft die Forschung feste
Scoregrenzen von 40, 50, 60, 70, 80 und 90 jeweils in beide Richtungen. Die
Regeltabelle kennzeichnet `all`, `k_of_n` und `score_threshold` getrennt und
speichert Mustername, Klauselanzahl, Mindestanzahl, Fenster und den finalen
Score-Schwellenwert in eigenen Spalten.

Stueckvolumen und Notional werden absichtlich nicht vorab vermischt. Das erste
misst die Zahl gehandelter Aktien; das zweite misst den umgesetzten USD-Wert
und enthaelt damit auch die Preisentwicklung. Getrennte Scores erlauben der
Regelsuche zu bestimmen, ob klassische Aktivitaet, Kapitalumsatz oder erst die
Kombination einen stabilen Edge besitzt. Ein Aktivitaetsverhaeltnis mit
fehlendem oder nichtpositivem Nenner wird als `NULL` behandelt und erzeugt
weder `Infinity` noch eine RuntimeWarning.

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

Market Cap aus `stock_core_market_metrics_daily` wird nur fuer exakte
Signal-Keys geladen. Fuer die spaeteren Ausstiegsvergleiche wird
`adjusted_open` mit dem exakten Identitaets- und Datums-Key der jeweiligen
Session verbunden. Es gibt kein Forward-Fill. Market Cap wird nur in USD
verglichen. Open- und SEC-Aktienzahl-Merkmale sind davon unabhaengig; nicht
SEC-basierte Aktienzahlen werden nicht fuer Float-/Turnover-Research verwendet.

## Walk-forward, Refit und Multiple Testing

Quantilschwellen werden expanding walk-forward geprueft. Standardmaessig
verwendet jeder Jahresfold 2020 bis 2024 nur fruehere Featurezeilen fuer seinen
Schwellenwert. Je nach Outcome wird bis zum tatsaechlichen Label-Ende D+5,
D+20, D+30, D+40, D+60 oder D+90 gepurgt. Ein Signal darf erst in einen Fold
eingehen, wenn das fuer genau dieses Objective benoetigte Ergebnis damals
vollstaendig bekannt war.

Eine Regel muss Mindestanzahl, Matchrate, Label-Coverage, Objective-Lift und
Retention der geschuetzten Klasse sowohl gepoolt als auch ueber ausreichend
viele Jahresfolds bestehen. Jede weitere OR-Komponente muss den Netto-Score
`Objective-Capture - Protected-Rejection` zusaetzlich verbessern.

Weil V5 sehr viele Merkmale und Kombinationen untersucht, reicht ein einfacher
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

## Early Cut und Positionsmanagement

Zu jedem Signal werden genau sechs Landmark-Zeilen gespeichert:

- Entscheidung nach Close D+1, wirksam ab D+2,
- Entscheidung nach Close D+2, wirksam ab D+3,
- Entscheidung nach Close D+3, wirksam ab D+4,
- Stagnationsentscheidung nach Close D+5, wirksam zum Open D+6,
- Gewinnmitnahmeentscheidung nach Close D+20, wirksam zum Open D+21,
- Gewinnmitnahmeentscheidung nach Close D+30, wirksam zum Open D+31.

Jede Zeile verwendet nur Informationen bis zu ihrem Close. Neben Preis-,
Volumen-, Notional-, RS- und Trenddaten stehen in V5 auch das bis dahin bekannte
Marktregime und die relative Entwicklung gegen SPY, QQQ und IWM zur Auswahl.
Bei D+1 bis D+3 wird die verbleibende Entwicklung erst ab der jeweils naechsten
Session bis D+5 bewertet.

Die Policy wird gemeinsam als D+1 -> D+2 -> D+3-Sequenz ausgewaehlt. Ein Cut
deaktiviert spaetere Landmark-Entscheidungen. Luecken, Segmentwechsel,
Delistings und unvollstaendige Horizonte werden zensiert und nicht als Verlust
oder Erfolg unterstellt. Die Schutzklasse ist weiterhin
`strong_first_to_day5`.

Die D+5-Regelsuche betrachtet nur Positionen, die nach der vereinbarten
Definition stagnieren und zuvor nicht den -10-%-Hard-Stop beruehrt haben. Sie
prueft, welche am D+5-Close bekannten Merkmale einen Ausstieg am D+6-Open
gegenueber einem Halten bis D+20 unterscheiden.

An D+20 und D+30 werden nur noch positive, nicht ausgestoppte Pfade untersucht.
Der exakte `adjusted_open` der naechsten Session ist der Referenzpreis. Von dort
werden Halten bis D+40, D+60 und D+90 sowie der jeweilige weitere MFE/MAE-Pfad
verglichen. `take_profit_better` bedeutet, dass der spaetere Schlussreturn vom
naechsten Open nicht positiv war; `continue_winner` verlangt mindestens +5 %
und keinen anschliessenden -10-%-Pfad.

Diese D+5-/D+20-/D+30-Vergleiche sind bewusst unabhaengige, kontrafaktische
Research-Fragen. Sie sind noch keine sequenzielle Portfolio- oder
Order-Simulation. Dadurch bleibt sichtbar, welche Faktoren an jedem einzelnen
Entscheidungstag tragen, ohne Kosten, Slippage oder Positionsgroessen
einzumischen.

## Datenbanktabellen

Das Init-SQL besitzt ausschliesslich drei serviceeigene Ergebnistabellen:

- `stock_analyser_filter_research_signal_results`: Signale, alle Entry-
  Merkmale einschliesslich der 105 expliziten Pattern-Score-Spalten (64 mit
  klassischem Volume und 41 getrennte Notional-Varianten), drei
  aktuellen Taxonomiebezeichnungen und 90 diagnostischen Gruppenmerkmalen,
  Outcomes sowie Ausschluss und Bestaetigung.
- `stock_analyser_filter_research_early_cut_results`: sechs kausale
  Landmark-Zeilen je Signal mit D+1-bis-D+3-Cut-Status und unabhaengigen
  D+5-/D+20-/D+30-Management-Ergebnissen. Der bestehende Tabellenpraefix und
  Tabellenname bleiben trotz des breiteren Inhalts erhalten.
- `stock_analyser_filter_research_rule_results`: alle atomaren, Pattern-,
  Interaktions- und OR-Kandidaten, Walk-forward-/Stabilitaetsmetriken,
  Multiple-Testing-Ergebnisse, explizite Pattern-Metadaten sowie Diagnostic
  und Holdout.

Signal- und Landmark-Tabelle sind unkomprimierte Timescale-Hypertables mit
365-Tage-Chunks. Es gibt keine JSON-, Audit- oder Run-Tabelle. Der Runtime-Code
erzeugt oder aendert keine DB-Struktur; er validiert den vollstaendigen Vertrag
und schreibt alle drei zuvor leeren Tabellen atomar.

## Ausfuehrung

V5 ist ein inkompatibler Neuaufbau. Die vorhandenen Research-Tabellen werden
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
Hauptprozess schreibt die drei Ergebnistabellen. Wegen der zusaetzlichen
Horizonte, festen Ratio-Grenzen, Management-Objectives, Dreier-OR-Suche und 999
Permutationen, k-von-n-Muster, 105 Multi-Window-Scores, zwei zusaetzlichen
gruppenweiten SQL-Auswertungen und der diagnostischen Taxonomie-Kandidaten ist
V5 bewusst deutlich rechenintensiver als V4. Die taeglichen Gruppenaggregate
werden mit PostgreSQL `GROUPING SETS` in einem gemeinsamen Quellscan berechnet.

## Tests

```bash
python -m pytest -q
docker compose config
```
