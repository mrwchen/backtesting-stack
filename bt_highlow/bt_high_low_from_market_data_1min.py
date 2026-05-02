from __future__ import annotations

import argparse
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


TICKER_MAP = {
    "SPOTCRUDE": "SpotCrude",
    "COPPER": "Copper",
    "QQQ": "QQQ.US",
}


# =========================================================================
# Parameter
# =========================================================================

@dataclass(frozen=True)
class Params:
    lb: int = 5              # Left Bars fuer Pivot-Bestaetigung
    rb: int = 5              # Right Bars fuer Pivot-Bestaetigung
    bot: str = "py-backtester"


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Berechnet den HH/HL/LH/LL-Pivot-Indikator (Wei-HH-LL-HL-LH-Marker-v2) auf market_data_1min und schreibt nach bt_high_low."
    )
    parser.add_argument("--start-date", required=True,
                        help="Startdatum oder -zeit, z.B. 2025-01-01 oder 2025-01-01T00:00:00Z")
    parser.add_argument("--lb", type=int, default=5, help="Left Bars fuer Pivot-Bestaetigung")
    parser.add_argument("--rb", type=int, default=5, help="Right Bars fuer Pivot-Bestaetigung")
    parser.add_argument("--bot", default="py-backtester", help="Wert fuer die bot-Spalte in bt_high_low")
    parser.add_argument("--timeframe", default="1min")
    parser.add_argument("--warmup-bars", type=int, default=2000,
                        help="Bars vor start-date als Warmup; wichtig fuer die ZigZag-Kette und findprevious()")
    parser.add_argument("--symbol", action="append", dest="symbols", default=None,
                        help="Mehrfach nutzbar: nur diese Symbole verarbeiten")
    parser.add_argument("--host", default=os.getenv("PGHOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PGPORT", "5432")))
    parser.add_argument("--user", default=os.getenv("PGUSER", "backtesting-account"))
    parser.add_argument("--password", default=os.getenv("PGPASSWORD", "backtesting-account-pw"))
    parser.add_argument("--source-db", default=os.getenv("PGSOURCE_DB", "market-data"))
    parser.add_argument("--target-db", default=os.getenv("PGTARGET_DB", "backtesting"))
    parser.add_argument("--source-table", default="public.market_data_1min")
    parser.add_argument("--target-table", default="public.bt_high_low")
    return parser.parse_args()


def parse_start_ts(value: str) -> datetime:
    text = value.strip()
    if len(text) == 10:
        return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)
    text = text.replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def mapped_ticker(symbol: str) -> str:
    return TICKER_MAP.get(symbol, symbol)


# =========================================================================
# Pivot-Berechnung
# =========================================================================

def confirmed_pivot(values: np.ndarray, left: int, right: int, is_high: bool) -> np.ndarray:
    """Emuliert Pine ta.pivothigh / ta.pivotlow.
    Der Wert erscheint am Index t, wenn an Position t-right ein bestaetigter Pivot liegt.
    Zentrum muss strikt hoeher (bzw. niedriger) sein als alle anderen Werte im Fenster.
    """
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if n < left + right + 1:
        return out
    for t in range(left + right, n):
        pivot_idx = t - right
        window = values[pivot_idx - left: pivot_idx + right + 1]
        if np.isnan(window).any():
            continue
        center = values[pivot_idx]
        others = np.concatenate([window[:left], window[left + 1:]])
        if is_high:
            if center == np.max(window) and center > np.max(others):
                out[t] = center
        else:
            if center == np.min(window) and center < np.min(others):
                out[t] = center
    return out


def value_when(condition: np.ndarray, source: np.ndarray, occurrence: int) -> np.ndarray:
    """Emuliert Pine ta.valuewhen(condition, source, occurrence).
    occurrence=0 liefert den juengsten passenden Wert, 1 den vorherigen usw.
    Wichtig: in Pine wird der aktuelle Bar-Wert eingerechnet, wenn condition dort true ist.
    """
    n = len(condition)
    out = np.full(n, np.nan, dtype=float)
    buf: list[float] = []
    for i in range(n):
        if condition[i]:
            src = source[i]
            if not np.isnan(src):
                buf.append(float(src))
                if len(buf) > occurrence + 1:
                    buf.pop(0)
        if len(buf) >= occurrence + 1:
            out[i] = buf[-1 - occurrence]
    return out


# =========================================================================
# Kernlogik: ZigZag-Kette + HH/HL/LH/LL-Klassifikation
# =========================================================================

def compute_pivots_for_symbol(df_symbol: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Baut pro Symbol die ZigZag-Kette und klassifiziert jeden bestaetigten
    Pivot als HH/HL/LH/LL (oder keiner). Gibt einen DataFrame nur mit
    bestaetigten, klassifizierten Pivots zurueck.
    """
    n = len(df_symbol)
    if n == 0:
        return df_symbol.iloc[0:0].copy()

    high = df_symbol["high"].to_numpy(dtype=float)
    low = df_symbol["low"].to_numpy(dtype=float)
    bar_time = pd.to_datetime(df_symbol["bar_time"], utc=True).to_numpy()

    # Pine ta.pivothigh(lb, rb) mit source=high bzw. low
    ph = confirmed_pivot(high, params.lb, params.rb, True)
    pl = confirmed_pivot(low, params.lb, params.rb, False)

    # Initiales hl und zz analog Pine Zeile 25-26:
    # hl = not na(ph) ? 1 : (not na(pl) ? -1 : na)
    # zz = not na(ph) ? ph  : (not na(pl) ? pl : na)
    hl = np.where(~np.isnan(ph), 1.0, np.where(~np.isnan(pl), -1.0, np.nan))
    zz = np.where(~np.isnan(ph), ph, np.where(~np.isnan(pl), pl, np.nan))

    # Erste Filterrunde (Pine Zeile 28-29):
    # zz := not na(pl) and hl == -1 and valuewhen(not na(hl), hl, 1) == -1 and ph > valuewhen(not na(zz), zz, 1) ? na : zz
    # zz := not na(ph) and hl ==  1 and valuewhen(not na(hl), hl, 1) ==  1 and pl < valuewhen(not na(zz), zz, 1) ? na : zz
    #
    # Wichtig: die valuewhen-Aufrufe beziehen sich hier auf den Stand VOR dem
    # Filter, also auf das ursprungliche hl und zz. In Pine ist das so, weil
    # die rechte Seite des :=-Assignments die "alten" Werte liest. valuewhen(..., 1)
    # liefert den vorletzten gesetzten Wert - auf dem aktuellen Bar ist das
    # also der letzte Pivot VOR dem aktuellen.
    hl_cond = ~np.isnan(hl)
    hl_prev = value_when(hl_cond, hl, 1)
    zz_cond = ~np.isnan(zz)
    zz_prev = value_when(zz_cond, zz, 1)

    # Vorsicht bei Vergleichen mit NaN: Pine behandelt na-Vergleich als false.
    def _lt(a, b):
        return np.where(np.isnan(a) | np.isnan(b), False, a < b)

    def _gt(a, b):
        return np.where(np.isnan(a) | np.isnan(b), False, a > b)

    mask1 = (~np.isnan(pl)) & (hl == -1.0) & (hl_prev == -1.0) & _gt(ph, zz_prev)
    mask2 = (~np.isnan(ph)) & (hl == 1.0) & (hl_prev == 1.0) & _lt(pl, zz_prev)
    zz = np.where(mask1 | mask2, np.nan, zz)

    # Zweite Filterrunde (Pine Zeile 31-32):
    # hl := hl == 1  and valuewhen(not na(hl), hl, 1) == -1 and zz < valuewhen(not na(zz), zz, 1) ? na : hl
    # hl := hl == -1 and valuewhen(not na(hl), hl, 1) == 1  and zz > valuewhen(not na(zz), zz, 1) ? na : hl
    # Hier werden valuewhen-Aufrufe erneut gemacht, basierend auf dem
    # jetzt bereits gefilterten zz. Wir bauen die valuewhen-Werte erneut.
    zz_cond2 = ~np.isnan(zz)
    zz_prev2 = value_when(zz_cond2, zz, 1)

    mask3 = (hl == 1.0) & (hl_prev == -1.0) & _lt(zz, zz_prev2)
    mask4 = (hl == -1.0) & (hl_prev == 1.0) & _gt(zz, zz_prev2)
    hl = np.where(mask3 | mask4, np.nan, hl)

    # Pine Zeile 33: zz := na(hl) ? na : zz
    zz = np.where(np.isnan(hl), np.nan, zz)

    # Jetzt haben wir die finalen ZigZag-Punkte: Indizes mit not na(zz) sind
    # "gueltige" Pivots in der ZigZag-Kette; hl[i] ist die Richtung (1=high, -1=low).
    # Fuer jeden gueltigen Pivot bestimmen wir a, b, c, d, e (der aktuelle +
    # die 4 vorherigen ZigZag-Punkte, mit alternierender Richtung - genauso
    # wie findprevious() in Pine).

    # Sammle die Indizes der gueltigen ZigZag-Punkte in Reihenfolge.
    valid_indices = np.flatnonzero(~np.isnan(zz))
    if len(valid_indices) == 0:
        return df_symbol.iloc[0:0].copy().assign(
            signal=pd.Series(dtype=str),
            pivot_price=pd.Series(dtype=float),
            pivot_direction=pd.Series(dtype=int),
            confirmation_idx=pd.Series(dtype=int),
        )

    zz_values_at_valid = zz[valid_indices]
    hl_values_at_valid = hl[valid_indices]

    # Fuer findprevious(): wir brauchen fuer jeden Pivot i in der Liste die
    # Werte loc1..loc4 = letzte 4 ZigZag-Punkte mit alternierender Richtung
    # ausgehend von Pivot i (Pine schaut rueckwaerts, erwartet 1.ter mit
    # entgegengesetzter Richtung, 2.ter mit gleicher, 3.ter mit entgegengesetzter,
    # 4.ter mit gleicher).
    records = []
    num_valid = len(valid_indices)

    for k in range(num_valid):
        cur_idx = int(valid_indices[k])
        cur_dir = float(hl_values_at_valid[k])
        if np.isnan(cur_dir):
            continue
        a = float(zz_values_at_valid[k])

        # Pine: ehl = hl == 1 ? -1 : 1  -> wir suchen ersten vorherigen mit
        # entgegengesetzter Richtung, dann gleicher, dann entgegen, dann gleich.
        # Default in Pine: loc1..loc4 = 0.0 wenn nichts gefunden. Wir emulieren
        # das mit 0.0 Fallback, damit die Vergleiche a>b, b>d usw. dasselbe
        # Ergebnis liefern.
        loc = [0.0, 0.0, 0.0, 0.0]
        expected_dirs = [
            -cur_dir,   # loc1 entgegen zur aktuellen
             cur_dir,   # loc2 gleich zur aktuellen
            -cur_dir,   # loc3 entgegen
             cur_dir,   # loc4 gleich
        ]
        j = k - 1
        slot = 0
        while j >= 0 and slot < 4:
            if hl_values_at_valid[j] == expected_dirs[slot]:
                loc[slot] = float(zz_values_at_valid[j])
                slot += 1
            j -= 1

        loc1, loc2, loc3, loc4 = loc
        # Pine Zuordnung:
        # a := zz ; b := loc1 ; c := loc2 ; d := loc3 ; e := loc4
        b, c, d, e = loc1, loc2, loc3, loc4

        # Bedingungen wie in Pine (Zeile 87-90). not na(hl) ist hier immer true,
        # weil wir ueber valid_indices iterieren.
        is_hh = (a > b) and (a > c) and (c > b) and (c > d)
        is_ll = (a < b) and (a < c) and (c < b) and (c < d)
        is_hl = ((a >= c) and (b > c and b > d and d > c and d > e)) or ((a < b) and (a > c) and (b < d))
        is_lh = ((a <= c) and (b < c and b < d and d < c and d < e)) or ((a > b) and (a < c) and (b > d))

        # Reihenfolge wie in Pine: HH -> HL -> LH -> LL (Zeile 109-124).
        # Achtung: pivot_direction in Pine folgt NICHT streng der Pivot-Art,
        # sondern ist laut Pine:
        #   HH -> 1, HL -> -1, LH -> 1, LL -> -1
        # (siehe Pine Zeilen 109-124; _hl hat pivotDir = -1 und _lh hat 1).
        signal = None
        pivot_dir = 0
        if is_hh:
            signal = "HH"
            pivot_dir = 1
        elif is_hl:
            signal = "HL"
            pivot_dir = -1
        elif is_lh:
            signal = "LH"
            pivot_dir = 1
        elif is_ll:
            signal = "LL"
            pivot_dir = -1

        if signal is None:
            continue

        # Pivot-Kerzen-Index: der Pivot wurde rb Bars frueher gebildet.
        pivot_cand_idx = cur_idx - params.rb
        if pivot_cand_idx < 0:
            continue

        records.append({
            "confirmation_idx": cur_idx,
            "pivot_idx": pivot_cand_idx,
            "signal": signal,
            "pivot_price": a,
            "pivot_direction": pivot_dir,
        })

    if not records:
        return df_symbol.iloc[0:0].copy().assign(
            signal=pd.Series(dtype=str),
            pivot_price=pd.Series(dtype=float),
            pivot_direction=pd.Series(dtype=int),
            confirmation_idx=pd.Series(dtype=int),
            pivot_idx=pd.Series(dtype=int),
        )

    pivots_df = pd.DataFrame(records)
    pivots_df["confirmation_bar_time"] = bar_time[pivots_df["confirmation_idx"].to_numpy()]
    pivots_df["pivot_bar_time"] = bar_time[pivots_df["pivot_idx"].to_numpy()]
    # symbol in jede Zeile
    if "symbol" in df_symbol.columns and n > 0:
        pivots_df["symbol"] = df_symbol["symbol"].iloc[0]
    return pivots_df


# =========================================================================
# DB
# =========================================================================

def _make_engine(host: str, port: int, user: str, password: str, dbname: str) -> Engine:
    from urllib.parse import quote_plus
    url = (
        f"postgresql+psycopg://{quote_plus(user)}:{quote_plus(password)}"
        f"@{host}:{port}/{quote_plus(dbname)}"
    )
    return create_engine(url, future=True)


def fetch_market_data(engine: Engine, table_name: str, start_ts: datetime,
                      warmup_bars: int, symbols: list[str] | None) -> pd.DataFrame:
    from sqlalchemy import text
    fetch_from = start_ts - timedelta(minutes=warmup_bars)
    symbol_filter = "AND symbol = ANY(:symbols)" if symbols else ""
    sql = text(
        f"""
        SELECT symbol, bar_time, high, low
        FROM {table_name}
        WHERE bar_time >= :fetch_from
        {symbol_filter}
        ORDER BY symbol, bar_time
        """
    )
    query_params: dict[str, object] = {"fetch_from": fetch_from}
    if symbols:
        query_params["symbols"] = symbols
    df = pd.read_sql_query(sql, engine, params=query_params, parse_dates=["bar_time"])
    if df.empty:
        return df
    df["bar_time"] = pd.to_datetime(df["bar_time"], utc=True)
    for col in ["high", "low"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["symbol", "bar_time", "high", "low"]).copy()
    return df


def build_output_rows(pivots_all: pd.DataFrame, start_ts: datetime,
                       timeframe: str, bot: str) -> pd.DataFrame:
    if pivots_all.empty:
        return pivots_all
    # Nur Zeilen, deren Bestaetigung im Zielzeitraum liegt.
    confirm_utc = pd.to_datetime(pivots_all["confirmation_bar_time"], utc=True)
    mask = confirm_utc >= pd.Timestamp(start_ts)
    hits = pivots_all.loc[mask].copy()
    if hits.empty:
        return hits

    hits["id"] = [str(uuid.uuid4()) for _ in range(len(hits))]
    hits["bot"] = bot
    hits["ticker"] = hits["symbol"].map(mapped_ticker)
    hits["exchange"] = ""
    # Fuer "symbol" in bt_high_low nutzen wir den raw-Symbolnamen aus market_data_1min.
    hits["symbol_out"] = hits["symbol"]
    hits["timeframe"] = timeframe

    # ms-Epoch fuer bigint-Spalten. Pine time ist UTC ms seit Epoch.
    pivot_utc = pd.to_datetime(hits["pivot_bar_time"], utc=True)
    confirm_utc = pd.to_datetime(hits["confirmation_bar_time"], utc=True)
    hits["pivot_bar_time_ms"] = (pivot_utc.astype("int64") // 1_000_000).astype("int64")
    hits["confirmation_bar_time_ms"] = (confirm_utc.astype("int64") // 1_000_000).astype("int64")
    hits["pivot_bar_time_ts"] = pivot_utc
    hits["confirmation_bar_time_ts"] = confirm_utc

    now_utc = datetime.now(timezone.utc)
    hits["created_at"] = now_utc

    cols = [
        "id", "bot", "ticker", "exchange", "symbol_out", "timeframe", "signal",
        "pivot_price", "pivot_direction",
        "pivot_bar_time_ms", "confirmation_bar_time_ms",
        "pivot_bar_time_ts", "confirmation_bar_time_ts",
        "created_at",
    ]
    return hits[cols].sort_values(["ticker", "confirmation_bar_time_ts"]).reset_index(drop=True)


def delete_existing_rows(conn, table_name: str, start_ts: datetime,
                          timeframe: str, tickers: list[str]) -> None:
    if not tickers:
        return
    sql = f"""
        DELETE FROM {table_name}
        WHERE confirmation_bar_time >= %s
          AND timeframe = %s
          AND ticker = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(sql, (start_ts, timeframe, list(tickers)))


def insert_rows(conn, table_name: str, df_out: pd.DataFrame) -> int:
    if df_out.empty:
        return 0
    sql = f"""
        INSERT INTO {table_name} (
            id, bot, ticker, exchange, symbol, timeframe, signal,
            pivot_price, pivot_direction,
            pivot_bar_time_ms, confirmation_bar_time_ms,
            pivot_bar_time, confirmation_bar_time,
            created_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s,
            %s
        )
    """
    records = []
    for row in df_out.itertuples(index=False):
        records.append((
            row.id,
            row.bot,
            row.ticker,
            row.exchange,
            row.symbol_out,
            row.timeframe,
            row.signal,
            float(row.pivot_price),
            int(row.pivot_direction),
            int(row.pivot_bar_time_ms),
            int(row.confirmation_bar_time_ms),
            row.pivot_bar_time_ts.to_pydatetime() if hasattr(row.pivot_bar_time_ts, "to_pydatetime") else row.pivot_bar_time_ts,
            row.confirmation_bar_time_ts.to_pydatetime() if hasattr(row.confirmation_bar_time_ts, "to_pydatetime") else row.confirmation_bar_time_ts,
            row.created_at.to_pydatetime() if hasattr(row.created_at, "to_pydatetime") else row.created_at,
        ))
    with conn.cursor() as cur:
        cur.executemany(sql, records)
    return len(records)


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    args = parse_args()
    start_ts = parse_start_ts(args.start_date)
    params = Params(lb=args.lb, rb=args.rb, bot=args.bot)

    source_engine = _make_engine(args.host, args.port, args.user, args.password, args.source_db)
    try:
        market_df = fetch_market_data(
            engine=source_engine,
            table_name=args.source_table,
            start_ts=start_ts,
            warmup_bars=args.warmup_bars,
            symbols=args.symbols,
        )
    finally:
        source_engine.dispose()

    if market_df.empty:
        print("Keine Marktdaten gefunden.")
        return

    per_symbol_results = []
    for symbol, df_symbol in market_df.groupby("symbol", sort=True):
        df_symbol = df_symbol.sort_values("bar_time").reset_index(drop=True)
        pivots_df = compute_pivots_for_symbol(df_symbol, params)
        if pivots_df.empty:
            continue
        per_symbol_results.append(pivots_df)

    # Ticker, die verarbeitet wurden (auch wenn kein Pivot gefunden wurde,
    # wollen wir alte Zeilen ab start_ts trotzdem loeschen).
    processed_symbols = sorted(market_df["symbol"].dropna().unique().tolist())
    processed_tickers = sorted({mapped_ticker(s) for s in processed_symbols})

    if per_symbol_results:
        pivots_all = pd.concat(per_symbol_results, ignore_index=True)
    else:
        pivots_all = pd.DataFrame()

    output_df = build_output_rows(pivots_all, start_ts, args.timeframe, args.bot)

    # DELETE ab start_ts fuer die verarbeiteten Ticker, dann INSERT.
    target_engine = _make_engine(args.host, args.port, args.user, args.password, args.target_db)
    try:
        raw_conn = target_engine.raw_connection()
        try:
            delete_existing_rows(
                conn=raw_conn,
                table_name=args.target_table,
                start_ts=start_ts,
                timeframe=args.timeframe,
                tickers=processed_tickers,
            )
            inserted = insert_rows(raw_conn, args.target_table, output_df)
            raw_conn.commit()
        except Exception:
            raw_conn.rollback()
            raise
        finally:
            raw_conn.close()
    finally:
        target_engine.dispose()

    if output_df.empty:
        print(f"Keine Pivot-Signale ab {start_ts.isoformat()} gefunden. "
              f"Alte Zeilen fuer {processed_tickers} entfernt.")
        return

    counts = output_df["signal"].value_counts().to_dict()
    print(
        f"Fertig. {inserted} Signale nach {args.target_table} geschrieben "
        f"({counts}) ab {start_ts.isoformat()} fuer {output_df['ticker'].nunique()} Ticker "
        f"({', '.join(sorted(output_df['ticker'].unique()))})."
    )


if __name__ == "__main__":
    main()
