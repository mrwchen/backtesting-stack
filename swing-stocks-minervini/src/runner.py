"""Entry point: orchestrates screen -> setup -> sim stages (STAGE env var)."""
from __future__ import annotations

import logging
import time
from dataclasses import fields, replace
from datetime import date

import numpy as np
import pandas as pd

from . import (
    data_loader,
    db,
    fundamentals,
    group_filter,
    market_filter,
    persistence,
    rs_rating,
    sensitivity,
    trend_template,
    vcp,
)
from .config import Config
from .simulator import simulate

log = logging.getLogger("runner")

SETUP_CONTEXT_COLUMNS = [
    "symbol",
    "period_end_date",
    "rs_rating",
    "ibkr_industry_rs_rating",
    "ibkr_category_rs_rating",
    "stock_industry_rs_rating",
    "stock_category_rs_rating",
    "eps_yoy",
    "revenue_yoy",
]


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


def _market_data_config(cfg: Config) -> Config:
    """Require complete index inputs when any sensitivity arm uses the gate."""
    if cfg.stage == "sensitivity" and any(
        variant.market_filter_enable for variant in sensitivity.VARIANTS
    ):
        return replace(cfg, market_filter_enable=True)
    return cfg


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


def _effective_regime(regime: pd.DataFrame) -> pd.DataFrame:
    """Map each score day to the first calendar day on which it is knowable.

    The source materialized view closes ``day=d`` at 01:00 America/New_York on
    ``d+1``.  US cash-session decisions can therefore use the row from the next
    calendar date onward.  Calendar dates deliberately preserve weekend rows:
    Sunday's score is available before Monday's session.
    """
    effective = regime.copy()
    effective["day"] = pd.to_datetime(effective["day"]).astype("datetime64[ns]")
    effective["effective_date"] = (
        effective["day"].dt.normalize() + pd.Timedelta(days=1)
    ).astype("datetime64[ns]")
    return (
        effective.sort_values(["effective_date", "day"], kind="stable")
        .drop_duplicates("effective_date", keep="last")
        .reset_index(drop=True)
    )


def _regime_entry_allowed(dates: pd.DatetimeIndex, regime: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Per-session gate using only regime rows finalized before that session."""
    if regime.empty:
        log.warning("regime entry filter enabled but no regime data is available - blocking all entries")
        return np.zeros(len(dates), dtype=bool)

    labels = _effective_regime(
        regime[["day", "regime_label"]].dropna(subset=["day"])
    )
    labels["regime_label"] = labels["regime_label"].astype(str).str.upper()
    sessions = pd.DataFrame(
        {
            "session_date": pd.DatetimeIndex(dates)
            .normalize()
            .astype("datetime64[ns]")
        }
    )
    aligned = pd.merge_asof(
        sessions.sort_values("session_date"),
        labels[["effective_date", "regime_label"]].sort_values("effective_date"),
        left_on="session_date",
        right_on="effective_date",
        direction="backward",
    )["regime_label"]
    allowed = aligned.isin(cfg.regime_allowed_labels).fillna(False).to_numpy(dtype=bool)
    log.info(
        "regime entry gate allows %d of %d days; allowed labels: %s",
        int(allowed.sum()), len(allowed), ",".join(cfg.regime_allowed_labels),
    )
    return allowed


def _attach_regime_attribution(trades: pd.DataFrame, regime: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    if regime.empty:
        trades = trades.copy()
        trades["regime_composite"] = None
        trades["regime_label"] = None
        return trades

    trades = trades.copy()
    trades["_order"] = np.arange(len(trades))
    trades["_entry_ts"] = pd.to_datetime(trades["entry_date"]).astype(
        "datetime64[ns]"
    )
    regime = _effective_regime(
        regime[["day", "regime_composite", "regime_label"]].dropna(subset=["day"])
    )
    trades = pd.merge_asof(
        trades.sort_values("_entry_ts"),
        regime.sort_values("effective_date"),
        left_on="_entry_ts",
        right_on="effective_date",
        direction="backward",
    )
    return trades.sort_values("_order").drop(
        columns=["_order", "_entry_ts", "day", "effective_date"]
    )


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
    quarterly = data_loader.load_quarterly_fundamentals(conn, cfg)
    fundamental = fundamentals.quarterly_flags(quarterly, dates, symbols, cfg)
    sponsorship_events = data_loader.load_sponsorship_events(conn, cfg)
    sponsorship = fundamentals.sponsorship_flags(sponsorship_events, dates, symbols, cfg)

    log.info("computing IBKR group leadership filter")
    leadership = group_filter.compute_leadership(rs["rs_raw"], universe, cfg)
    log.info("computing IBKR industry breadth gate")
    industry_breadth = group_filter.compute_industry_breadth(close_m, universe, cfg)
    screen_pass = (
        template["template_pass"]
        & fundamental["fundamentals_pass"]
        & sponsorship["institutional_sponsorship_pass"]
        & leadership["group_filter_pass"]
        & industry_breadth["ibkr_industry_breadth_pass"]
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
            "ibkr_industry_breadth_pct": (
                industry_breadth["ibkr_industry_breadth"] * 100
            ).round(4),
            "ibkr_industry_breadth_on": industry_breadth["ibkr_industry_breadth_on"],
            "ibkr_industry_breadth_pass": industry_breadth["ibkr_industry_breadth_pass"],
            "crit_price_above_ma150_200": template["crit_price_above_ma150_200"],
            "crit_ma150_above_ma200": template["crit_ma150_above_ma200"],
            "crit_ma200_rising": template["crit_ma200_rising"],
            "crit_ma50_above_ma150_200": template["crit_ma50_above_ma150_200"],
            "crit_price_above_ma50": template["crit_price_above_ma50"],
            "crit_above_52w_low": template["crit_above_52w_low"],
            "crit_near_52w_high": template["crit_near_52w_high"],
            "crit_rs_rating": template["crit_rs_rating"],
            "trend_template_pass": template["template_pass"],
            "eps_pass": fundamental["eps_pass"],
            "revenue_pass": fundamental["revenue_pass"],
            "margin_pass": fundamental["margin_pass"],
            "acceleration_pass": fundamental["acceleration_pass"],
            "streak_pass": fundamental["streak_pass"],
            "stability_pass": fundamental["stability_pass"],
            "fundamental_score": fundamental["fundamental_score"],
            "fundamentals_pass": fundamental["fundamentals_pass"],
            "institutional_manager_count": sponsorship["institutional_manager_count"],
            "institutional_net_activity": sponsorship["institutional_net_activity"],
            "institutional_sponsorship_pass": sponsorship["institutional_sponsorship_pass"],
            "screen_pass": screen_pass,
            "eps_yoy": fundamental["eps_yoy"].round(6),
            "revenue_yoy": fundamental["revenue_yoy"].round(6),
            "eps_acceleration": fundamental["eps_acceleration"].round(6),
            "revenue_acceleration": fundamental["revenue_acceleration"].round(6),
            "margin_delta": fundamental["margin_delta"].round(6),
            "growth_streak": fundamental["growth_streak"],
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
        "fundamental_score",
        "growth_streak",
        "institutional_manager_count",
    ):
        screen_df[column] = screen_df[column].round().astype(
            "Int32" if column == "institutional_manager_count" else "Int16"
        )
    persistence.write_screen(conn, screen_df, start, end)

    market_df = market.loc[window].reset_index(names="period_end_date")
    market_df["period_end_date"] = market_df["period_end_date"].dt.date
    market_df["market_breadth_pct"] = (market_df["market_breadth"] * 100).round(4)
    persistence.write_market(conn, market_df, start, end)
    log.info(
        "screen done: %d rs rows, %d screen rows (%d screen passes, %d group passes, %d industry breadth passes)",
        len(rs_df), len(screen_df), int(screen_df["screen_pass"].sum()),
        int(screen_df["group_filter_pass"].sum()),
        int(screen_df["ibkr_industry_breadth_pass"].sum()),
    )
    return screen_df.loc[screen_df["screen_pass"]].reset_index(drop=True)


def detect_setups(
    cfg: Config,
    prices: pd.DataFrame,
    pass_days: pd.DataFrame,
    universe: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if pass_days.empty:
        return pd.DataFrame(
            columns=[field.name for field in fields(vcp.Setup)]
            + ["ibkr_industry", "ibkr_category"]
        )

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
                trading_dates=trading_dates,
            )
        )
    log.info("detected %d VCP setups across %d symbols", len(all_setups), len(pass_by_symbol))
    setups_df = pd.DataFrame([vars(s) for s in all_setups])
    if setups_df.empty:
        return pd.DataFrame(
            columns=[field.name for field in fields(vcp.Setup)]
            + ["ibkr_industry", "ibkr_category"]
        )
    setups_df = setups_df.merge(
        universe[["symbol", "ibkr_industry", "ibkr_category"]],
        on="symbol",
        how="left",
    )
    return setups_df


def run_setup(
    conn, cfg: Config, prices: pd.DataFrame, pass_days: pd.DataFrame,
    universe: pd.DataFrame, trading_dates: pd.DatetimeIndex, start, end,
) -> pd.DataFrame:
    setups_df = detect_setups(cfg, prices, pass_days, universe, trading_dates)
    persistence.write_setups(conn, setups_df, start, end)
    return setups_df


def _prepare_simulation_setups(
    setups: pd.DataFrame,
    screen_passes: pd.DataFrame,
) -> pd.DataFrame:
    prepared = setups.copy()
    if "setup_id" not in prepared:
        # Sensitivity setups stay in memory because the shared setup table is
        # not run-scoped. Negative identifiers cannot be mistaken for its
        # positive BIGSERIAL values when events are inspected later.
        prepared["setup_id"] = -np.arange(1, len(prepared) + 1, dtype=np.int64)
    available_context = [
        column for column in SETUP_CONTEXT_COLUMNS if column in screen_passes.columns
    ]
    if len(available_context) == len(SETUP_CONTEXT_COLUMNS):
        context = screen_passes[SETUP_CONTEXT_COLUMNS].rename(
            columns={"period_end_date": "detect_date"}
        )
        context = context.copy()
        context["detect_date"] = pd.to_datetime(context["detect_date"]).dt.date
        prepared = prepared.merge(
            context,
            on=["symbol", "detect_date"],
            how="left",
            validate="many_to_one",
        )
    return prepared


def _slice_matrices(matrices: dict, end: date) -> dict:
    mask = matrices["dates"] <= pd.Timestamp(end)
    sliced = {
        "dates": matrices["dates"][mask],
        "symbols": matrices["symbols"],
    }
    for field in ("open", "high", "low", "close", "volume"):
        sliced[field] = matrices[field].loc[mask]
    return sliced


def _run_simulation(
    conn,
    cfg: Config,
    matrices: dict,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    setups: pd.DataFrame,
    regime: pd.DataFrame,
    start: date,
    end: date,
    *,
    state_start: date | None = None,
) -> tuple[int, dict]:
    dates = matrices["dates"]
    sim_start_idx = int(dates.searchsorted(pd.Timestamp(start)))
    state_start_idx = int(
        dates.searchsorted(pd.Timestamp(state_start or start))
    )
    market_for_dates = market.reindex(dates)
    market_exposure_cap = (
        market_for_dates["entry_exposure_cap"].to_numpy()
        if cfg.market_filter_enable
        else None
    )
    regime_entry_allowed = (
        _regime_entry_allowed(dates, regime, cfg)
        if cfg.regime_entry_filter_enable
        else None
    )
    result = simulate(
        dates, matrices["symbols"],
        matrices["open"], matrices["high"], matrices["low"], matrices["close"],
        setups, cfg, sim_start_idx=sim_start_idx, state_start_idx=state_start_idx,
        market_exposure_cap=market_exposure_cap,
        regime_entry_allowed=regime_entry_allowed,
    )
    run_id = persistence.create_run(conn, cfg, result.metrics, start, end)
    trades = result.trades
    breakout_events = result.breakout_events
    if not breakout_events.empty:
        breakout_date = pd.to_datetime(breakout_events["breakout_date"]).dt.date
        breakout_events = breakout_events.loc[
            breakout_date >= start
        ].reset_index(drop=True)
    if not trades.empty:
        trades = trades.merge(
            universe[["symbol", "ibkr_industry", "ibkr_category"]], on="symbol", how="left"
        )
        trades = _attach_regime_attribution(trades, regime)
    persistence.write_trades(conn, run_id, trades)
    persistence.write_breakout_events(conn, run_id, breakout_events)
    persistence.write_equity(conn, run_id, result.equity)
    decisions = (
        " ".join(
            f"{decision} {count}"
            for decision, count in breakout_events["decision"]
            .value_counts()
            .sort_index()
            .items()
        )
        if not breakout_events.empty
        else "none 0"
    )
    log.info(
        "run %d %s %s setups %d breakout-events %d filled %d positions %s return %s decisions %s",
        run_id,
        cfg.run_label,
        cfg.simulation_mode,
        len(setups),
        len(breakout_events),
        int(breakout_events["entry_filled"].sum())
        if not breakout_events.empty else 0,
        result.metrics.get("num_positions", 0),
        result.metrics.get("total_return", 0.0),
        decisions,
    )
    return run_id, result.metrics


def run_sim(
    conn, cfg: Config, matrices: dict, universe: pd.DataFrame,
    market: pd.DataFrame, start, end,
) -> tuple[int, dict]:
    setups = persistence.read_setups(conn, start, end)
    if setups.empty:
        log.warning("no setups in %s..%s -> persisting zero-signal run", start, end)
    regime = data_loader.load_regime_scores(conn, cfg)
    return _run_simulation(
        conn, cfg, matrices, universe, market, setups, regime, start, end
    )


def run_sensitivity(
    conn,
    cfg: Config,
    prices: pd.DataFrame,
    screen_passes: pd.DataFrame,
    matrices: dict,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    start: date,
    end: date,
) -> None:
    periods = sensitivity.phases(start, end)
    regime = data_loader.load_regime_scores(conn, cfg)
    detection_cache: dict[tuple[float, float, float], pd.DataFrame] = {}

    log.info(
        "sensitivity start development-only market-filter ablation variants %d periods %d end-limit %s screen-pass-days %d symbols %d",
        len(sensitivity.VARIANTS),
        len(periods),
        sensitivity.DEVELOPMENT_END_DATE,
        len(screen_passes),
        screen_passes["symbol"].nunique(),
    )
    for variant in sensitivity.VARIANTS:
        if variant.detection_key not in detection_cache:
            detection_cfg = variant.apply(cfg, "full", start, end)
            detection_cache[variant.detection_key] = detect_setups(
                detection_cfg,
                prices,
                screen_passes,
                universe,
                matrices["dates"],
            )
        all_setups = detection_cache[variant.detection_key]

        for phase, phase_start, phase_end in periods:
            phase_cfg = variant.apply(cfg, phase, phase_start, phase_end)
            detect_date = pd.to_datetime(all_setups["detect_date"]).dt.date
            valid_until = pd.to_datetime(all_setups["valid_until"]).dt.date
            phase_setups = all_setups.loc[
                (detect_date <= phase_end) & (valid_until >= phase_start)
            ].reset_index(drop=True)
            phase_setups = _prepare_simulation_setups(
                phase_setups, screen_passes
            )
            phase_matrices = _slice_matrices(matrices, phase_end)

            log.info(
                "sensitivity variant %s period %s %s %s score %.0f dryup %.2f %.2f market-filter %s setups %d",
                variant.name,
                phase,
                phase_start,
                phase_end,
                variant.vcp_score_min,
                variant.dryup_ratio_min,
                variant.dryup_ratio_max,
                "on" if variant.market_filter_enable else "off",
                len(phase_setups),
            )
            _run_simulation(
                conn,
                phase_cfg,
                phase_matrices,
                universe,
                market,
                phase_setups,
                regime,
                phase_start,
                phase_end,
                state_start=start,
            )

    log.info(
        "sensitivity done detection-configurations %d runs %d",
        len(detection_cache),
        len(sensitivity.VARIANTS) * len(periods),
    )


def main() -> None:
    cfg = Config.from_env()
    _configure_logging(cfg.log_level)
    log.info("Stage %s start %s end %s label %s", cfg.stage, cfg.start_date, cfg.end_date, cfg.run_label)

    if cfg.stage == "sensitivity":
        if cfg.end_date is None:
            raise ValueError(
                "development-only market-filter ablation requires END_DATE=2023-12-31"
            )
        sensitivity.validate_configured_window(
            date.fromisoformat(cfg.start_date), date.fromisoformat(cfg.end_date)
        )

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

    index_bars = data_loader.load_market_indexes(conn, _market_data_config(cfg))
    market = market_filter.compute_market_model(matrices["close"], index_bars, cfg)
    log.info(
        "Market model latest %s breadth %.1f%% entry cap %.0f%% active %d %d days",
        market["market_status"].iloc[-1],
        100 * market["market_breadth"].iloc[-1],
        100 * market["entry_exposure_cap"].iloc[-1],
        int(market.loc[window, "market_on"].sum()), int(window.sum()),
    )

    if cfg.stage == "all":
        stages = ("screen", "setup", "sim")
    elif cfg.stage == "sensitivity":
        stages = ("screen", "sensitivity")
    else:
        stages = (cfg.stage,)
    pass_days = None
    if "screen" in stages:
        pass_days = run_screen(conn, cfg, matrices, market, universe, window)
    if "setup" in stages:
        if pass_days is None:
            pass_days = persistence.read_screen_pass_days(conn, start, end)
        run_setup(
            conn, cfg, prices, pass_days, universe, matrices["dates"], start, end
        )
    if "sim" in stages:
        run_sim(conn, cfg, matrices, universe, market, start, end)
    if "sensitivity" in stages:
        assert pass_days is not None
        run_sensitivity(
            conn,
            cfg,
            prices,
            pass_days,
            matrices,
            universe,
            market,
            start,
            end,
        )
    conn.close()
    log.info("done")


if __name__ == "__main__":
    main()
