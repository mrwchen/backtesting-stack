# Stock Analyser Filter Research V2

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

V2 vermischt schwache und stark verlierende Signale nicht mehr zu einem
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

## Feature-Gruppen A-C

- **A – Zustand am Signaltag:** Volume/21d, Volume/50d, Notional/21d,
  Notional/50d, Liquiditaet, RS, Triggerkontext, Trendgeometrie und Position zu
  den 52-Wochen-Grenzen.
- **B – Chartverlauf bis D-1:** Renditen, Beschleunigung, ATR, Drawdown,
  vorherige Hochs, Range-Kompression und RS-Aenderung.
- **C – Aktivitaetsverlauf bis D-1:** kurz-/mittelfristige Volume- und
  Notional-Verhaeltnisse, Up-Session-Anteile sowie Preis-Aktivitaets-
  Korrelationen.

A, B und C sind bei der Auswahl gleichberechtigt. Es gibt keine irreversible
Reihenfolge A -> B -> C mehr. Pro Ziel werden hoechstens zwei einfache
Quantilbedingungen gewaehlt und mit `OR` verknuepft. Die beiden Zielregeln
werden anschliessend nur in einer Kombination verwendet, welche die gemeinsame
Strong-Retention- und Ausschlussgrenze weiterhin einhaelt.

Fundamentaldaten sind weiterhin nicht Teil von V2.

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
- mindestens 90 % Retention der geschuetzten Strong-First-Klasse,
- mindestens 90 % Labelabdeckung unter den Matches,
- Objective-Lift mindestens 1,05,
- positiver Lift in mindestens 75 % der aussagefaehigen Jahresfolds,
- mindestens ein Prozentpunkt zusaetzliche Capture fuer eine zweite Bedingung.

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

Die verbleibende Entwicklung wird bis zum unveraenderten Ende D+5 bewertet.
Die pfadbewusste Klasse ist eine von:

- `loss_first`,
- `strong_first`,
- `same_session_ambiguous`,
- `stagnant`,
- `neutral`.

Bereits bis zum Landmark erreichte +/-5-%-Barrieren verlassen das primaere
Early-Cut-Risk-Set. Luecken, Segmentwechsel, Delistings und unvollstaendige
Horizonte werden zensiert und niemals als Verlust oder Erfolg unterstellt.
Alle drei Landmark-Zeilen bleiben trotzdem gespeichert, damit Coverage
sichtbar ist. Die drei Landmarks sind unabhaengige Research-Entscheidungen;
eine hypothetische fruehere Cut-Regel entfernt keine spaetere Zeile.
Der Split einer Landmark-Zeile richtet sich nach ihrem eigenen
`landmark_date`; ein Entry unmittelbar vor einer Jahres- oder Holdout-Grenze
zieht seine spaeteren Early-Cut-Entscheidungen daher nicht in den alten Split.

## Datenbanktabellen

Das Init-SQL besitzt ausschliesslich diese serviceeigenen Ergebnistabellen:

- `stock_analyser_filter_research_signal_results`: Entry-Signale, A-C-Features,
  pfadbewusste Labels und finale Entry-Entscheidungen.
- `stock_analyser_filter_research_early_cut_results`: genau drei kausale
  Landmark-Zeilen je Signal mit Hold-/Cut-Entscheidungen.
- `stock_analyser_filter_research_rule_results`: generische Kandidaten-,
  Walk-forward-, Diagnose-, Holdout- und ausgewählte Regelmetriken fuer beide
  Entscheidungsfamilien und Ziele.

Signal- und Early-Cut-Tabelle sind unkomprimierte Timescale-Hypertables mit
365-Tage-Chunks. Es gibt keine JSON-, Audit- oder Run-Tabelle. Runtime-Code
erzeugt oder aendert keine DB-Struktur; er validiert den vollstaendigen Vertrag
und schreibt alle drei zuvor leeren Tabellen atomar.

## Ausfuehrung

Der inkompatible V2-Neuaufbau der freigegebenen, reproduzierbaren
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
