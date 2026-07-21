# Stock Analyser Filter Research V2.2 / C3

Dieses eigenstaendige Research-Programm untersucht kausale `false -> true`-
Ereignisse des 8-von-8-Trend-Templates aus
`stock_analyser_trend_template_daily`. Es simuliert weder ein Portfolio noch
Orders, Kosten oder Slippage. Das Ergebnis sind einfache, nachvollziehbare
Include-/Exclude-Regeln am Signaltag sowie kausale Hold-/Cut-Regeln nach D+1,
D+2 und D+3.

## Signalvertrag

Ein Signal entsteht nur, wenn dieselbe `(symbol, exchange, cik)` in der
aktuellen globalen Handelssession alle acht Kriterien besteht und in der
unmittelbar vorherigen globalen Session nicht bestanden hat. Beide Sessions
muessen beobachtet sein und zum selben Kontinuitaetssegment gehoeren.
`true -> true`, erste Beobachtungen, Luecken und Segmentwechsel erzeugen kein
neues Signal.

Die A-Features verwenden teilweise die komplette Signaltagskerze. Eine
Signaltagsentscheidung ist daher erst nach Handelsschluss bekannt. Das Programm
berechnet bewusst keinen fiktiven Same-Close- oder Next-Open-Return.

## Getrennte Entry-Ziele

V2.2 vermischt schwache und stark verlierende Signale nicht zu einem
Optimierungsziel:

- `weak_5d`: In D+1 bis D+5 wird ausgehend vom Signaltags-Close nie +2 %
  erreicht.
- `loss_first_5d`: Die -5-%-Barriere wird vor der +5-%-Barriere erreicht.
- `strong_first_5d`: Die +5-%-Barriere wird vor der -5-%-Barriere erreicht und
  bildet die geschuetzte Klasse.

Wenn beide Barrieren in derselben Tageskerze erreicht werden, ist die
Intraday-Reihenfolge mit Tagesdaten unbekannt. Solche Faelle werden als
`same_day_ambiguous` gespeichert und nicht gewaltsam einem der beiden
Reihenfolgeziele zugeordnet. `deep_loss_5d`, `strong_5d` und `bad_5d` bleiben
als Diagnosen erhalten, steuern die Auswahl aber nicht gemeinsam.

## Feature-Gruppen A-D

- **A – Zustand am Signaltag:** Volume/21d, Volume/50d, Notional/21d,
  Notional/50d, Liquiditaet, RS, Triggerkontext, Trendgeometrie und Position zu
  den 52-Wochen-Grenzen.
- **B – Chartverlauf bis D-1:** Renditen, Beschleunigung, ATR, Drawdown,
  vorherige Hochs, Range-Kompression und RS-Aenderung.
- **C – Aktivitaetsverlauf bis D-1:** kurz-/mittelfristige Volume- und
  Notional-Verhaeltnisse, Up-Session-Anteile sowie Preis-Aktivitaets-
  Korrelationen.
- **D – geordnete Chart- und Volumenmuster:** Basisbreite, logarithmische
  Trendsteigung und R², Trend-Effizienz, Anteil positiver Sessions, Alter und
  Abstand des letzten Hochs, Tiefe/Alter/Erholung des letzten V-Tiefs sowie
  Distribution-, Churning- und Failed-Breakout-Zaehler.

A, B, C und D sind bei der Auswahl gleichberechtigt. Es gibt keine irreversible
Reihenfolge A -> B -> C -> D. Neben einfachen Quantilbedingungen konkurrieren
sechs vorab festgelegte Musterkandidaten: `flat_base`, `ordered_uptrend`,
`pullback_from_high`, `v_recovery`, `volume_dry_up_breakout` und
`distribution_top`. Die Teilbedingungen innerhalb eines solchen Musters
werden zwingend mit `AND` verbunden. Maximal zwei vollstaendige Kandidaten
werden pro Ziel mit `OR` verbunden; es gibt keine freie kombinatorische Suche
nach beliebigen AND-Mustern. Weak- und Loss-First-
Spezialregeln muessen nur ihr eigenes Ziel stabil anreichern. Eine valide
Weak-Regel wird daher nicht mehr verworfen, nur weil sie kein Loss-First-Signal
liefert. Ihre gemeinsame Entry-Union wird ausschliesslich auf Matchrate,
Labelabdeckung und Strong-Retention geprueft. Das Ranking maximiert
`Objective-Capture - Protected-Rejection`; eine zweite Bedingung muss diesen
Netto-Score um mindestens den konfigurierten Wert verbessern.

Alle Vorlaufmerkmale verwenden nur zusammenhaengende Sessions bis D-1. Das
Volume-Dry-up-Breakout-Muster kombiniert diesen Vorlauf mit Volumen und
Breakout-Bestaetigung der Signaltagskerze; es ist deshalb wie alle A-Features
erst nach dem Signaltags-Close entscheidbar. Fundamentaldaten sind weiterhin
nicht Teil von C3.

## Walk-forward und neuer Holdout

Quantil-Policies werden expanding walk-forward geprueft. Standardmaessig wird
fuer jedes Evaluationsjahr 2020 bis 2024 der Schwellenwert ausschliesslich aus
frueheren Featurezeilen bestimmt. Die letzten Signale, deren D+5-Label erst im
Folgejahr bekannt waere, bleiben sichtbar, werden fuer den Fold aber gepurgt.

Eine Regel muss neben den gepoolten Safety-Gates in mindestens vier
aussagefaehigen Jahresfolds stabil sein. Standardbedingungen sind unter
anderem:

- mindestens `max(50, 1 %)` gelabelte Matches,
- hoechstens 35 % Matches,
- mindestens 92 % gepoolte und finale Retention der geschuetzten
  Strong-First-Klasse,
- mindestens 90 % Retention in jedem aussagefaehigen Walk-forward-Fold,
- mindestens 90 % Labelabdeckung unter den Matches,
- Objective-Lift mindestens 1,05,
- positiver Lift in mindestens 75 % der aussagefaehigen Jahresfolds,
- mindestens ein Prozentpunkt zusaetzlicher Netto-Score fuer eine zweite
  Bedingung.

Nach der Walk-forward-Auswahl werden die finalen Schwellen bis zum
Validation-Ende neu gefittet. Diese festen Schwellen muessen Development und
Validation nochmals bestehen. Verletzt ein Refit Match-, Coverage-, Lift- oder
Strong-Safety-Gates, faellt die Regel deterministisch auf einen kuerzeren
Prefix bis hin zu `no filter` zurueck. Diagnostic und Holdout werden fuer diese
Entscheidung nicht gelesen.

Der bereits betrachtete Zeitraum 2025 bis einschliesslich 20.07.2026 ist nur
`diagnostic` und darf die Auswahl oder den finalen Schwellenwert nicht aendern.
Die erste globale Handelssession nach dem 20.07.2026 beginnt den neuen,
unangetasteten `holdout`. Solange dort weniger als die konfigurierte
Mindestanzahl gelabelter Beobachtungen vorliegt, bleibt `passes_holdout` NULL
statt faelschlich `false` zu werden.

## Early Cut D+1 bis D+3

Zu jedem Signal werden genau drei Landmark-Zeilen gespeichert:

- Entscheidung nach Close D+1, wirksam ab D+2,
- Entscheidung nach Close D+2, wirksam ab D+3,
- Entscheidung nach Close D+3, wirksam ab D+4.

Jede Landmark-Zeile verwendet nur Informationen bis zu ihrem Close. Dazu
gehoeren bisherige MFE/MAE, Rendite seit dem Signal, Drawdown/Rebound,
Tagesrange, Close-Position, Volume-/Notional-Verhaeltnisse, RS- und
Trendabstaende. Future-Spalten der Source werden nie als Feature verwendet.

Die verbleibende Entwicklung beginnt strikt mit der naechsten Session
`effective_session_date` und wird bis zum unveraenderten Ende D+5 bewertet.
Alle zukuenftigen +2-%-, +5-%- und -5-%-Barrieren beziehen sich auf den Close
des jeweiligen Landmarks, nicht mehr auf den urspruenglichen Signalkurs. Die
pfadbewusste Klasse ist eine von:

- `loss_first`,
- `strong_first`,
- `same_session_ambiguous`,
- `stagnant`,
- `neutral`.

`bad_to_day5` fasst die beiden handlungsrelevanten Klassen `stagnant` und
`loss_first` fuer die gemeinsame sequenzielle Policy zusammen.

Bereits bis zum Landmark erreichte +/-5-%-Barrieren verlassen das primaere
Early-Cut-Risk-Set. Luecken, Segmentwechsel, Delistings und unvollstaendige
Horizonte werden zensiert und niemals als Verlust oder Erfolg unterstellt.
Alle drei Landmark-Zeilen bleiben trotzdem gespeichert, damit Coverage
sichtbar ist. Die Policy wird jedoch gemeinsam als D+1 -> D+2 -> D+3-Sequenz
ausgewaehlt. Ein Cut an D+1 deaktiviert D+2 und D+3; ein Cut an D+2 deaktiviert
D+3. `active_at_landmark` kennzeichnet das unmittelbar vor der jeweiligen
Entscheidung noch aktive Risk-Set. `prior_policy_cut_day` verweist bei einer
spaeteren `not_active`-Zeile auf den frueheren Cut-Tag. Die intrinsische
`eligible_at_landmark`-Information bleibt davon unveraendert.

Die kumulative Policy wird an der vollstaendigen D+1-Kohorte bewertet.
Population und Objective-/Strong-Nenner bleiben fest am D+1-Outcome verankert.
Ein spaeterer Cut erhaelt Objective-Credit aber nur, wenn dasselbe Outcome am
tatsaechlichen ersten Cut-Landmark noch gilt. Jeder Cut eines am D+1
geschuetzten Strong-Signals bleibt eine Protected-Rejection. Die Policy muss die
gepoolte/finale 92-%- sowie die foldweise 90-%-Strong-Retention als Gesamtregel
einhalten. Fuer die sequenzielle Auswahl wird jede vollstaendige
D+1 -> D+3-Folge separat an ihrem D+1-Landmark einem Walk-forward-Fold
zugeordnet. Reicht ihr unveraenderter D+5-Horizont ueber das Fold-Ende hinaus,
wird die gesamte Folge aus dieser Auswahlkohorte gepurgt. Diagnostic und
Holdout werden ebenfalls als D+1-geankerte Gesamtsequenzen ausgewertet; noch
nicht abgeschlossene Horizonte bleiben dort ungelabelt. Der gespeicherte finale
Aktivstatus wird nie fuer die Walk-forward-Auswahl wiederverwendet.

Davon getrennt beschreibt `analysis_split` den Diagnose-Split jeder einzelnen
gespeicherten Landmark-Zeile anhand ihres eigenen `landmark_date` und ihres
D+5-Horizonts. Deshalb koennen die drei gespeicherten Zeilen eines Signals an
einer Split-Grenze unterschiedliche Werte tragen; diese Zeilenwerte steuern
nicht die D+1-geankerte sequenzielle Regelauswahl.

## Datenbanktabellen

Das Init-SQL besitzt ausschliesslich diese serviceeigenen Ergebnistabellen:

- `stock_analyser_filter_research_signal_results`: Entry-Signale, A-D-Features,
  pfadbewusste Labels und finale Entry-Entscheidungen.
- `stock_analyser_filter_research_early_cut_results`: genau drei kausale
  Landmark-Zeilen je Signal mit landmark-relativen Outcomes, intrinsischer
  Eligibility und sequenziellem Aktiv-/Hold-/Cut-Status.
- `stock_analyser_filter_research_rule_results`: generische Kandidaten-,
  Walk-forward-, Diagnose-, Holdout- und ausgewählte Regelmetriken fuer beide
  Entscheidungsfamilien und Ziele.

Signal- und Early-Cut-Tabelle sind unkomprimierte Timescale-Hypertables mit
365-Tage-Chunks. Es gibt keine JSON-, Audit- oder Run-Tabelle. Runtime-Code
erzeugt oder aendert keine DB-Struktur; er validiert den vollstaendigen Vertrag
und schreibt alle drei zuvor leeren Tabellen atomar.

## Ausfuehrung

Der inkompatible V2.2-Neuaufbau der freigegebenen, reproduzierbaren
Research-Tabellen erfolgt einmalig mit:

```bash
DROP_ALL_STOCK_ANALYSER_FILTER_RESEARCH_TABLES_ON_START=true \
  docker compose up --build --force-recreate stock-analyser-filter-research
```

Der Drop-Schalter ist und bleibt standardmaessig `false`. Ohne expliziten Drop
bricht das Programm bei bereits vorhandenen Zielzeilen ab.

Die Berechnung nutzt disjunkte, nach Source-Zeilenzahl balancierte
Aktienpartitionen. Jeder Worker-Prozess besitzt eine eigene DB-Verbindung und
importiert denselben exportierten PostgreSQL-Snapshot. Nur der Hauptprozess
schreibt die drei Ergebnistabellen.

## Tests

```bash
python -m pytest -q
```
