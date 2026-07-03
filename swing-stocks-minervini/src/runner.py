"""Entry point: orchestrates screen -> setup -> sim stages (STAGE env var)."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import data_loader, db, fundamentals, persistence, rs_rating, trend_template, vcp
from .config import Config
from .simulator import simulate

log = logging.getLogger("runner")


def _long_frame(mask: pd.DataFrame, dates: pd.DatetimeIndex, symbols: pd.Index, columns: dict) -> pd.DataFrame:
    """Turn a boolean selection matrix plus per-cell/per-row matrices into a long frame."""
    row_idx, col_idx = np.where(mask.to_numpy())
    out = {
        "period_end_date": pd.Series(dates[row_idx]).dt.date,
        "symbol": symbols.to_numpy()[col_idx],
    }
    for name, source in columns.items():
        if isinstance(source, pd.DataFrame):
            out[name] = source.to_numpy()[row_idx, col_idx]
        else:  # per-day series
            out[name] = source.to_numpy()[row_idx]
    return pd.DataFrame(out)


def run_screen(conn, cfg: Config, matrices: dict, window: np.ndarray) -> pd.DataFrame:
    dates, symbols = matrices["dates"], matrices["symbols"]
    close_m, volume_m = matrices["close"], matrices["volume"]

    log.info("computing relative strength for %d symbols", len(symbols))
    rs = rs_rating.compute_rs(close_m, volume_m, cfg)
    log.info("computing trend template")
    template = trend_template.compute_template(close_m, rs["rs_rating"], cfg)

    log.info("computing point-in-time fundamentals")
    filings = data_loader.load_fundamentals(conn, cfg)
    eps_pass, eps_yoy = fundamentals.eps_flags(filings, dates, symbols, cfg)
    revenue_pass, revenue_yoy, margin_pass = fundamentals.revenue_margin_flags(
        filings, dates, symbols, cfg
    )
    fundamentals_pass = fundamentals.combine(eps_pass, revenue_pass, margin_pass, cfg)
    screen_pass = template["template_pass"] & fundamentals_pass

    window_m = pd.DataFrame(
        np.broadcast_to(window[:, None], (len(dates), len(symbols))),
        index=dates, columns=symbols,
    )

    rs_df = _long_frame(
        rs["eligible"] & window_m, dates, symbols,
        {"rs_raw": rs["rs_raw"].round(6), "rs_rating": rs["rs_rating"], "universe_size": rs["universe_size"]},
    )
    rs_df["rs_rating"] = rs_df["rs_rating"].astype(int)
    start, end = dates[window][0].date(), dates[window][-1].date()
    persistence.write_rs(conn, rs_df, start, end)

    persist_mask = screen_pass if cfg.screen_persist == "passed" else rs["eligible"]
    screen_df = _long_frame(
        persist_mask & window_m, dates, symbols,
        {
            "close": close_m.round(4),
            "rs_rating": rs["rs_rating"],
            "crit_price_above_ma150_200": template["crit_price_above_ma150_200"],
            "crit_ma150_above_ma200": template["crit_ma150_above_ma200"],
            "crit_ma200_rising": template["crit_ma200_rising"],
            "crit_ma50_above_ma150_200": template["crit_ma50_above_ma150_200"],
            "crit_price_above_ma50": template["crit_price_above_ma50"],
            "crit_above_52w_low": template["crit_above_52w_low"],
            "crit_near_52w_high": template["crit_near_52w_high"],
            "crit_rs_rating": template["crit_rs_rating"],
            "trend_template_pass": template["template_pass"],
            "eps_pass": eps_pass,
            "revenue_pass": revenue_pass,
            "margin_pass": margin_pass,
            "fundamentals_pass": fundamentals_pass,
            "screen_pass": screen_pass,
            "eps_yoy": eps_yoy.round(6),
            "revenue_yoy": revenue_yoy.round(6),
        },
    )
    screen_df["rs_rating"] = screen_df["rs_rating"].astype("Int16")
    persistence.write_screen(conn, screen_df, start, end)
    log.info(
        "screen done: %d rs rows, %d screen rows (%d screen passes)",
        len(rs_df), len(screen_df), int(screen_df["screen_pass"].sum()),
    )
    return _long_frame(screen_pass & window_m, dates, symbols, {})


def run_setup(conn, cfg: Config, prices: pd.DataFrame, pass_days: pd.DataFrame, start, end) -> None:
    if pass_days.empty:
        log.warning("no screen passes -> no setup detection")
        persistence.write_setups(conn, [], start, end)
        return

    pass_days = pass_days.copy()
    pass_days["period_end_date"] = pd.to_datetime(pass_days["period_end_date"])
    pass_by_symbol = pass_days.groupby("symbol")["period_end_date"].apply(list)

    all_setups: list[vcp.Setup] = []
    grouped = prices[prices["symbol"].isin(pass_by_symbol.index)].groupby("symbol")
    for symbol, sub in grouped:
        sub = sub.sort_values("date")
        dates_s = pd.DatetimeIndex(sub["date"])
        wanted = pd.DatetimeIndex(pass_by_symbol[symbol])
        local_idx = dates_s.searchsorted(wanted)
        local_idx = local_idx[
            (local_idx < len(dates_s)) & (dates_s[np.minimum(local_idx, len(dates_s) - 1)].isin(wanted))
        ]
        if len(local_idx) == 0:
            continue
        all_setups.extend(
            vcp.find_setups(
                symbol,
                dates_s,
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float),
                sub["close"].to_numpy(dtype=float),
                sub["volume"].to_numpy(dtype=float),
                np.asarray(local_idx),
                cfg,
            )
        )
    log.info("detected %d VCP setups across %d symbols", len(all_setups), len(pass_by_symbol))
    persistence.write_setups(conn, all_setups, start, end)


def run_sim(conn, cfg: Config, matrices: dict, start, end) -> None:
    setups = persistence.read_setups(conn, start, end)
    if setups.empty:
        log.warning("no setups in %s..%s -> nothing to simulate", start, end)
        return
    dates = matrices["dates"]
    sim_start_idx = int(dates.searchsorted(pd.Timestamp(start)))
    result = simulate(
        dates, matrices["symbols"],
        matrices["open"], matrices["high"], matrices["low"], matrices["close"],
        setups, cfg, sim_start_idx=sim_start_idx,
    )
    run_id = persistence.create_run(conn, cfg, result.metrics, start, end)
    persistence.write_trades(conn, run_id, result.trades)
    persistence.write_equity(conn, run_id, result.equity)
    log.info("run %d persisted: %s", run_id, result.metrics)


def main() -> None:
    cfg = Config.from_env()
    logging.basicConfig(
        level=cfg.log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    log.info("stage=%s start=%s end=%s label=%s", cfg.stage, cfg.start_date, cfg.end_date, cfg.run_label)

    conn = db.get_conn()
    prices = data_loader.load_prices(conn, cfg)
    equity_symbols = data_loader.load_equity_symbols(conn, cfg)
    prices = prices[prices["symbol"].isin(equity_symbols)]
    log.info("loaded %d daily bars for %d equity symbols", len(prices), prices["symbol"].nunique())

    matrices = {"close": data_loader.pivot_field(prices, "close")}
    matrices["dates"] = matrices["close"].index
    matrices["symbols"] = matrices["close"].columns
    for field in ("open", "high", "low", "volume"):
        matrices[field] = data_loader.pivot_field(prices, field)

    dates = matrices["dates"]
    window = (dates >= pd.Timestamp(cfg.start_date)).to_numpy()
    if not window.any():
        raise SystemExit(f"no price data on/after START_DATE={cfg.start_date}")
    start, end = dates[window][0].date(), dates[window][-1].date()

    stages = ("screen", "setup", "sim") if cfg.stage == "all" else (cfg.stage,)
    pass_days = None
    if "screen" in stages:
        pass_days = run_screen(conn, cfg, matrices, window)
    if "setup" in stages:
        if pass_days is None:
            pass_days = persistence.read_screen_pass_days(conn, start, end)
        run_setup(conn, cfg, prices, pass_days, start, end)
    if "sim" in stages:
        run_sim(conn, cfg, matrices, start, end)
    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
