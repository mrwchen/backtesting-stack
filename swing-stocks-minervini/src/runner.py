"""Entry point: orchestrates screen -> setup -> sim stages (STAGE env var)."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from . import data_loader, db, fundamentals, group_filter, market_filter, persistence, rs_rating, trend_template, vcp
from .config import Config
from .simulator import simulate

log = logging.getLogger("runner")


class _UtcFormatter(logging.Formatter):
    converter = time.gmtime


def _configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        _UtcFormatter(
            "%(asctime)sZ %(levelname)s %(processName)s %(threadName)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)


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


def run_screen(
    conn, cfg: Config, matrices: dict, market: pd.DataFrame,
    universe: pd.DataFrame, window: np.ndarray,
) -> pd.DataFrame:
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

    log.info("computing IBKR group leadership filter")
    leadership = group_filter.compute_leadership(rs["rs_raw"], universe, cfg)
    screen_pass = (
        template["template_pass"]
        & fundamentals_pass
        & leadership["group_filter_pass"]
    )

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
            "ibkr_industry_rs_rating": leadership["ibkr_industry_rs_rating"],
            "ibkr_category_rs_rating": leadership["ibkr_category_rs_rating"],
            "stock_industry_rs_rating": leadership["stock_industry_rs_rating"],
            "stock_category_rs_rating": leadership["stock_category_rs_rating"],
            "ibkr_industry_pass": leadership["ibkr_industry_pass"],
            "ibkr_category_pass": leadership["ibkr_category_pass"],
            "stock_industry_pass": leadership["stock_industry_pass"],
            "stock_category_pass": leadership["stock_category_pass"],
            "group_filter_pass": leadership["group_filter_pass"],
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
    screen_df = screen_df.merge(
        universe[["symbol", "ibkr_industry", "ibkr_category"]], on="symbol", how="left"
    )
    for column in (
        "rs_rating",
        "ibkr_industry_rs_rating",
        "ibkr_category_rs_rating",
        "stock_industry_rs_rating",
        "stock_category_rs_rating",
    ):
        screen_df[column] = screen_df[column].astype("Int16")
    persistence.write_screen(conn, screen_df, start, end)

    market_df = market.loc[window].reset_index(names="period_end_date")
    market_df["period_end_date"] = market_df["period_end_date"].dt.date
    market_df["market_breadth_pct"] = (market_df["market_breadth"] * 100).round(4)
    persistence.write_market(conn, market_df, start, end)
    log.info(
        "screen done: %d rs rows, %d screen rows (%d screen passes, %d group passes)",
        len(rs_df), len(screen_df), int(screen_df["screen_pass"].sum()),
        int(screen_df["group_filter_pass"].sum()),
    )
    return _long_frame(screen_pass & window_m, dates, symbols, {})


def run_setup(
    conn, cfg: Config, prices: pd.DataFrame, pass_days: pd.DataFrame,
    universe: pd.DataFrame, start, end,
) -> None:
    if pass_days.empty:
        log.warning("no screen passes -> no setup detection")
        persistence.write_setups(conn, pd.DataFrame(), start, end)
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
    setups_df = pd.DataFrame([vars(s) for s in all_setups])
    if not setups_df.empty:
        setups_df = setups_df.merge(
            universe[["symbol", "ibkr_industry", "ibkr_category"]], on="symbol", how="left"
        )
    persistence.write_setups(conn, setups_df, start, end)


def run_sim(
    conn, cfg: Config, matrices: dict, universe: pd.DataFrame,
    market: pd.DataFrame, start, end,
) -> None:
    setups = persistence.read_setups(conn, start, end)
    if setups.empty:
        log.warning("no setups in %s..%s -> nothing to simulate", start, end)
        return
    dates = matrices["dates"]
    sim_start_idx = int(dates.searchsorted(pd.Timestamp(start)))
    market_on = market["market_on"].to_numpy() if cfg.market_filter_enable else None
    result = simulate(
        dates, matrices["symbols"],
        matrices["open"], matrices["high"], matrices["low"], matrices["close"],
        setups, cfg, sim_start_idx=sim_start_idx, market_on=market_on,
    )
    run_id = persistence.create_run(conn, cfg, result.metrics, start, end)
    trades = result.trades
    if not trades.empty:
        trades = trades.merge(
            universe[["symbol", "ibkr_industry", "ibkr_category"]], on="symbol", how="left"
        )
        regime = data_loader.load_regime_scores(conn, cfg)
        if not regime.empty:
            regime = regime.copy()
            regime["day"] = pd.to_datetime(regime["day"]).dt.date
            trades = trades.merge(
                regime.rename(columns={"day": "entry_date"}), on="entry_date", how="left"
            )
        else:
            trades["regime_composite"] = None
            trades["regime_label"] = None
    persistence.write_trades(conn, run_id, trades)
    persistence.write_equity(conn, run_id, result.equity)
    log.info("run %d persisted: %s", run_id, result.metrics)


def main() -> None:
    cfg = Config.from_env()
    _configure_logging(cfg.log_level)
    log.info("Stage %s start %s end %s label %s", cfg.stage, cfg.start_date, cfg.end_date, cfg.run_label)

    conn = db.get_conn()
    prices = data_loader.load_prices(conn, cfg)
    universe = data_loader.load_universe(conn, cfg)
    prices = prices[prices["symbol"].isin(set(universe["symbol"]))]
    taxonomy_ok = universe["ibkr_industry"].notna() & universe["ibkr_category"].notna()
    log.info(
        "loaded %d daily bars for %d equity symbols, IBKR taxonomy for %d of %d symbols",
        len(prices), prices["symbol"].nunique(), int(taxonomy_ok.sum()), len(universe),
    )

    matrices = {"close": data_loader.pivot_field(prices, "close")}
    matrices["dates"] = matrices["close"].index
    matrices["symbols"] = matrices["close"].columns
    for field in ("open", "high", "low", "volume"):
        matrices[field] = data_loader.pivot_field(prices, field)

    dates = matrices["dates"]
    window = np.asarray(dates >= pd.Timestamp(cfg.start_date))
    if not window.any():
        raise SystemExit(f"no price data on/after START_DATE={cfg.start_date}")
    start, end = dates[window][0].date(), dates[window][-1].date()

    market = market_filter.compute_breadth(matrices["close"], cfg)
    log.info(
        "market breadth: %.1f%% latest, gate on for %d of %d days in window",
        100 * market["market_breadth"].iloc[-1],
        int(market.loc[window, "market_on"].sum()), int(window.sum()),
    )

    stages = ("screen", "setup", "sim") if cfg.stage == "all" else (cfg.stage,)
    pass_days = None
    if "screen" in stages:
        pass_days = run_screen(conn, cfg, matrices, market, universe, window)
    if "setup" in stages:
        if pass_days is None:
            pass_days = persistence.read_screen_pass_days(conn, start, end)
        run_setup(conn, cfg, prices, pass_days, universe, start, end)
    if "sim" in stages:
        run_sim(conn, cfg, matrices, universe, market, start, end)
    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
