"""Entry point: orchestrates screen -> setup -> sim stages (STAGE env var)."""
from __future__ import annotations

import logging
import time
from dataclasses import fields, replace
from datetime import date

import numpy as np
import pandas as pd
from backtest_models import minervini as model

from . import (
    data_loader,
    db,
    fundamentals,
    group_filter,
    market_filter,
    persistence,
    ranking_sensitivity,
    reproducibility,
    rs_rating,
    sensitivity,
    trend_template,
)
from .config import Config
from .candidate_ranking import FillCalibrationLabel, QualityCalibrationLabel
from .simulator import simulate

log = logging.getLogger("runner")

CANDIDATE_CONTEXT_COLUMNS = (
    "trend_template_pass",
    "rs_rating",
    "fundamental_score",
    "fundamental_coverage",
    "eps_yoy",
    "revenue_yoy",
)

SCREEN_FINGERPRINT_COLUMNS = tuple(persistence.SCREEN_COLUMNS)
SETUP_FINGERPRINT_COLUMNS = (
    "symbol", "setup_type", "price_continuity_segment", "detect_date",
    "pivot", "last_low", "stop_level",
    "base_start_date", "base_days", "n_contractions", "base_count",
    "contraction_depths", "dryup_ratio", "setup_score", "prior_advance_pct",
    "final_tightness_pct", "structure_quality_score",
    "volume_dryup_score", "tightness_score", "pivot_proximity_score",
    "prior_advance_score", "close", "valid_until",
)
SIMULATION_FINGERPRINT_COLUMNS = SETUP_FINGERPRINT_COLUMNS


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


def _fingerprint_available(
    frame: pd.DataFrame,
    preferred_columns: tuple[str, ...],
) -> str:
    columns = tuple(column for column in preferred_columns if column in frame.columns)
    if not columns:
        raise ValueError("cannot fingerprint a frame without stable model columns")
    return reproducibility.frame_fingerprint(frame, columns)


def _screen_config_fingerprint(cfg: Config) -> str:
    return reproducibility.config_fingerprint(
        cfg,
        reproducibility.SCREEN_CONFIG_FIELDS,
        model_version=model.MODEL_VERSION,
    )


def _setup_input_fingerprint(
    screen_fingerprint: str,
    source_fingerprint: str,
) -> str:
    return reproducibility.combine_fingerprints(
        screen=screen_fingerprint,
        source=source_fingerprint,
    )


def _setup_config_fingerprint(cfg: Config, setup_input_fingerprint: str) -> str:
    return reproducibility.config_fingerprint(
        cfg,
        reproducibility.SETUP_CONFIG_FIELDS,
        model_version=model.MODEL_VERSION,
        upstream_fingerprint=setup_input_fingerprint,
    )


def _source_input_fingerprint(matrices: dict, universe: pd.DataFrame) -> str:
    price_fingerprint = reproducibility.matrix_fingerprint(
        matrices["dates"],
        matrices["symbols"],
        {
            field: matrices[field]
            for field in (
                "open", "high", "low", "close", "raw_close", "volume",
                "price_continuity_segment", "price_continuity_break",
            )
        },
    )
    universe_columns = tuple(
        column
        for column in (
            "symbol", "exchange", "cik", "ibkr_industry", "ibkr_category"
        )
        if column in universe.columns
    )
    universe_fingerprint = reproducibility.frame_fingerprint(
        universe, universe_columns
    )
    return reproducibility.combine_fingerprints(
        prices=price_fingerprint,
        universe=universe_fingerprint,
    )


def _require_matching_fingerprint(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    expected: str,
    stage: str,
) -> None:
    actual = _fingerprint_available(frame, columns)
    if actual != expected:
        raise RuntimeError(
            f"persisted {stage} output fingerprint does not match its stage state"
        )


def _simulation_input_fingerprint(
    cfg: Config,
    setups: pd.DataFrame,
    matrices: dict,
    universe: pd.DataFrame,
    market_exposure_cap: np.ndarray | None,
    regime_entry_allowed: np.ndarray | None,
    regime: pd.DataFrame,
    candidate_context: dict[str, pd.DataFrame],
    *,
    quality_labels_fingerprint: str,
    fill_labels_fingerprint: str,
    online_calibration: bool,
) -> str:
    dates = matrices["dates"]
    price_matrices = {
        field: matrices[field]
        for field in (
            "open", "high", "low", "close", "volume",
            "price_continuity_segment", "price_continuity_break",
        )
    }
    price_fingerprint = reproducibility.matrix_fingerprint(
        dates,
        matrices["symbols"],
        price_matrices,
    )
    context_fingerprint = reproducibility.matrix_fingerprint(
        dates,
        matrices["symbols"],
        {
            f"candidate_{field}": candidate_context[field]
            for field in CANDIDATE_CONTEXT_COLUMNS
        },
    )
    attribution_columns = ("symbol", "ibkr_industry", "ibkr_category")
    attribution_fingerprint = reproducibility.frame_fingerprint(
        universe, attribution_columns
    )
    gate_frame = pd.DataFrame({"date": dates})
    if market_exposure_cap is not None:
        gate_frame["market_exposure_cap"] = market_exposure_cap
    if regime_entry_allowed is not None:
        gate_frame["regime_entry_allowed"] = regime_entry_allowed
    gate_fingerprint = reproducibility.frame_fingerprint(
        gate_frame, tuple(gate_frame.columns)
    )
    regime_columns = tuple(
        column
        for column in ("day", "regime_composite", "regime_label")
        if column in regime.columns
    )
    regime_fingerprint = (
        reproducibility.frame_fingerprint(regime, regime_columns)
        if regime_columns
        else reproducibility.combine_fingerprints(regime="empty")
    )
    source_fingerprint = reproducibility.combine_fingerprints(
        setups=_fingerprint_available(setups, SIMULATION_FINGERPRINT_COLUMNS),
        prices=price_fingerprint,
        candidate_context=context_fingerprint,
        entry_attribution=attribution_fingerprint,
        gates=gate_fingerprint,
        regime=regime_fingerprint,
        quality_calibration_labels=quality_labels_fingerprint,
        fill_calibration_labels=fill_labels_fingerprint,
        calibration_mode=(
            "online_calibration" if online_calibration else "preloaded"
        ),
    )
    return reproducibility.config_fingerprint(
        cfg,
        reproducibility.SIM_CONFIG_FIELDS,
        model_version=model.MODEL_VERSION,
        upstream_fingerprint=source_fingerprint,
    )


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
    rs = rs_rating.compute_rs(
        close_m,
        volume_m,
        cfg,
        raw_close=matrices["raw_close"],
        continuity_segment=matrices["price_continuity_segment"],
        continuity_break=matrices["price_continuity_break"],
    )
    log.info("computing trend template")
    template = trend_template.compute_template(
        close_m,
        rs["rs_rating"],
        cfg,
        high=matrices["high"],
        low=matrices["low"],
        continuity_segment=matrices["price_continuity_segment"],
        continuity_break=matrices["price_continuity_break"],
    )

    log.info("computing point-in-time fundamentals")
    quarterly = data_loader.load_quarterly_fundamentals(conn, cfg)
    fundamental = fundamentals.quarterly_flags(quarterly, dates, symbols, cfg)
    sponsorship_events = data_loader.load_sponsorship_events(conn, cfg)
    sponsorship = fundamentals.sponsorship_flags(sponsorship_events, dates, symbols, cfg)

    log.info("computing IBKR group leadership filter")
    leadership = group_filter.compute_leadership(rs["rs_raw"], universe, cfg)
    log.info("computing IBKR industry breadth gate")
    industry_breadth = group_filter.compute_industry_breadth(close_m, universe, cfg)
    # Candidate generation is deliberately independent of optional quality
    # data. Fundamentals, sponsorship, group leadership and industry breadth
    # are retained as point-in-time ranking/context fields, but a missing or
    # lagging optional field must not hide a valid Stage-2 chart structure.
    screen_pass = rs["eligible"]

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

    # The simulation rank is rebuilt for every session, including sessions on
    # which the trend template is false. Persist the complete rankable daily
    # population so a later STAGE=sim has the same causal context as STAGE=all.
    persist_mask = rs["eligible"]
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
            # Match NUMERIC(7,6) exactly so STAGE=all and a later STAGE=sim
            # build byte-for-byte equivalent ranking inputs.
            "fundamental_coverage": fundamental["fundamental_coverage"].round(6),
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
    window_cells = int(window.sum()) * len(symbols)
    log.info(
        "screen funnel cells %d rankable %d trend %d fundamentals %d sponsorship %d group %d industry-breadth %d candidates %d persisted %d",
        window_cells,
        int((rs["eligible"] & window_m).to_numpy().sum()),
        int((template["template_pass"] & window_m).to_numpy().sum()),
        int((fundamental["fundamentals_pass"] & window_m).to_numpy().sum()),
        int((sponsorship["institutional_sponsorship_pass"] & window_m).to_numpy().sum()),
        int((leadership["group_filter_pass"] & window_m).to_numpy().sum()),
        int((industry_breadth["ibkr_industry_breadth_pass"] & window_m).to_numpy().sum()),
        int((screen_pass & window_m).to_numpy().sum()),
        len(screen_df),
    )
    return screen_df.reset_index(drop=True)


def detect_setups(
    cfg: Config,
    prices: pd.DataFrame,
    pass_days: pd.DataFrame,
    universe: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    if pass_days.empty:
        return pd.DataFrame(
            columns=[field.name for field in fields(model.Setup)]
            + ["ibkr_industry", "ibkr_category"]
        )

    pass_days = pass_days.copy()
    pass_days["period_end_date"] = pd.to_datetime(pass_days["period_end_date"])
    pass_by_symbol = pass_days.groupby("symbol")["period_end_date"].apply(list)

    all_setups: list[model.Setup] = []
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
            model.find_setups(
                symbol,
                dates_s,
                sub["high"].to_numpy(dtype=float),
                sub["low"].to_numpy(dtype=float),
                sub["close"].to_numpy(dtype=float),
                sub["volume"].to_numpy(dtype=float),
                sub["price_continuity_segment"].to_numpy(),
                sub["price_continuity_break"].to_numpy(),
                np.asarray(local_idx),
                cfg,
                trading_dates=trading_dates,
            )
        )
    setups_df = pd.DataFrame([vars(s) for s in all_setups])
    if setups_df.empty:
        log.info("detected 0 setups across %d candidate symbols", len(pass_by_symbol))
        return pd.DataFrame(
            columns=[field.name for field in fields(model.Setup)]
            + ["ibkr_industry", "ibkr_category"]
        )
    setup_counts = " ".join(
        f"{setup_type} {count}"
        for setup_type, count in setups_df["setup_type"].value_counts().sort_index().items()
    )
    log.info(
        "detected %d setups across %d candidate symbols types %s",
        len(setups_df),
        len(pass_by_symbol),
        setup_counts,
    )
    setups_df = setups_df.merge(
        universe[["symbol", "ibkr_industry", "ibkr_category"]],
        on="symbol",
        how="left",
    )
    return setups_df


def _log_gold_control_coverage(
    setups: pd.DataFrame,
    start: date,
    end: date,
    source_fingerprint: str,
) -> None:
    """Report frozen chart-review controls without turning them into targets."""
    applicable = [
        case for case in model.GOLD_CASES if start <= case.reference_date <= end
    ]
    if not applicable:
        log.info("gold controls applicable 0")
        return
    detected = setups.copy()
    if detected.empty:
        detected_dates = pd.Series(dtype="datetime64[ns]")
    else:
        detected_dates = pd.to_datetime(detected["detect_date"])
    covered: list[model.GoldCase] = []
    for case in applicable:
        reference = pd.Timestamp(case.reference_date)
        in_window = (
            (detected["symbol"] == case.symbol)
            & detected_dates.between(
                reference - pd.Timedelta(days=case.lookback_days),
                reference + pd.Timedelta(days=case.forward_days),
            )
        ) if not detected.empty else pd.Series(dtype=bool)
        case_setups = detected.loc[in_window] if len(in_window) else detected.iloc[0:0]
        if not case_setups.empty:
            covered.append(case)
        setup_types = (
            ",".join(sorted(case_setups["setup_type"].unique()))
            if not case_setups.empty
            else "none"
        )
        log.info(
            "gold control %s %s reference %s window %d %d observed %d types %s source %s",
            case.symbol,
            case.role,
            case.reference_date,
            case.lookback_days,
            case.forward_days,
            len(case_setups),
            setup_types,
            source_fingerprint[:12],
        )
    for role in ("positive", "negative"):
        role_cases = [case for case in applicable if case.role == role]
        role_covered = [case for case in covered if case.role == role]
        missing = ",".join(case.symbol for case in role_cases if case not in role_covered)
        log.info(
            "gold controls %s applicable %d observed %d missing %s",
            role,
            len(role_cases),
            len(role_covered),
            missing or "none",
        )


def run_setup(
    conn, cfg: Config, prices: pd.DataFrame, pass_days: pd.DataFrame,
    universe: pd.DataFrame, trading_dates: pd.DatetimeIndex, start, end,
    *, source_fingerprint: str,
) -> pd.DataFrame:
    setups_df = detect_setups(cfg, prices, pass_days, universe, trading_dates)
    _log_gold_control_coverage(setups_df, start, end, source_fingerprint)
    persistence.write_setups(conn, setups_df, start, end)
    return setups_df


def _prepare_simulation_setups(setups: pd.DataFrame) -> pd.DataFrame:
    prepared = setups.copy()
    if "setup_id" not in prepared:
        # Sensitivity setups stay in memory because the shared setup table is
        # not run-scoped. Negative identifiers cannot be mistaken for its
        # positive BIGSERIAL values when events are inspected later.
        prepared["setup_id"] = -np.arange(1, len(prepared) + 1, dtype=np.int64)
    return prepared


def _screen_pass_rows(screen_daily: pd.DataFrame) -> pd.DataFrame:
    if "screen_pass" not in screen_daily.columns:
        raise ValueError("daily screen rows are missing screen_pass")
    return screen_daily.loc[
        screen_daily["screen_pass"].fillna(False).astype(bool)
    ].reset_index(drop=True)


def _candidate_context_matrices(
    screen_daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
) -> dict[str, pd.DataFrame]:
    """Return exact daily as-of inputs for pre-session candidate snapshots.

    Rows are observations for their own completed session. ``CandidateRanker``
    applies the causal one-session lag when it builds the next session's order
    slate. Missing symbol/session observations deliberately remain NaN; in
    particular, an absent trend-template pass must never be forward-filled.
    """
    required = {"period_end_date", "symbol", *CANDIDATE_CONTEXT_COLUMNS}
    missing = sorted(required - set(screen_daily.columns))
    if missing:
        raise ValueError(
            "daily screen rows are missing candidate fields: "
            + ", ".join(missing)
        )

    context = screen_daily.loc[:, sorted(required)].copy()
    context["period_end_date"] = pd.to_datetime(
        context["period_end_date"]
    ).dt.normalize()
    duplicate = context.duplicated(["period_end_date", "symbol"], keep=False)
    if duplicate.any():
        raise ValueError(
            "daily screen rows contain duplicate symbol/session candidate context"
        )

    aligned: dict[str, pd.DataFrame] = {}
    exact_dates = pd.DatetimeIndex(dates)
    exact_symbols = pd.Index(symbols)
    for field in CANDIDATE_CONTEXT_COLUMNS:
        values = context.pivot(
            index="period_end_date", columns="symbol", values=field
        )
        values = values.reindex(index=exact_dates, columns=exact_symbols)
        aligned[field] = values.apply(pd.to_numeric, errors="coerce").astype(float)
    return aligned


def _slice_matrices(matrices: dict, end: date) -> dict:
    mask = matrices["dates"] <= pd.Timestamp(end)
    sliced = {
        "dates": matrices["dates"][mask],
        "symbols": matrices["symbols"],
    }
    for field in (
        "open", "high", "low", "close", "volume",
        "price_continuity_segment", "price_continuity_break",
    ):
        sliced[field] = matrices[field].loc[mask]
    return sliced


def _first_touch_calibration_labels(
    cfg: Config,
    matrices: dict,
    setups: pd.DataFrame,
    candidate_context: dict[str, pd.DataFrame],
    *,
    state_start_idx: int,
    market_exposure_cap: np.ndarray | None,
    regime_entry_allowed: np.ndarray | None,
) -> tuple[
    tuple[QualityCalibrationLabel, ...],
    tuple[FillCalibrationLabel, ...],
]:
    """Build capacity-independent labels entirely in memory.

    Future labels may be present in the returned tuple because the ranker
    applies ``available_date <= information_date`` on every estimate. This
    gives OOS and portfolio runs a causal walk-forward history without storing
    a research/audit table or learning from portfolio selection.
    """
    calibration_cfg = replace(
        cfg,
        simulation_mode="independent",
        portfolio_ranking_mode="quality_only",
    )
    result = simulate(
        matrices["dates"],
        matrices["symbols"],
        matrices["open"],
        matrices["high"],
        matrices["low"],
        matrices["close"],
        setups,
        calibration_cfg,
        sim_start_idx=state_start_idx,
        state_start_idx=state_start_idx,
        volume_m=matrices["volume"],
        candidate_context=candidate_context,
        continuity_segment_m=matrices["price_continuity_segment"],
        market_exposure_cap=market_exposure_cap,
        regime_entry_allowed=regime_entry_allowed,
        online_calibration=True,
    )
    log.info(
        "market-on first-touch calibration quality-labels %d fill-labels %d",
        len(result.quality_labels),
        len(result.fill_labels),
    )
    return result.quality_labels, result.fill_labels


def _entry_gate_arrays(
    cfg: Config,
    dates: pd.DatetimeIndex,
    market: pd.DataFrame,
    regime: pd.DataFrame,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return the exact exogenous entry gates used by calibration and portfolio."""
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
    return market_exposure_cap, regime_entry_allowed


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
    candidate_context: dict[str, pd.DataFrame],
    state_start: date | None = None,
    commit: bool = True,
    quality_labels: tuple[QualityCalibrationLabel, ...] = (),
    fill_labels: tuple[FillCalibrationLabel, ...] = (),
    calibration_labels_supplied: bool = False,
    calibration_label_fingerprints: tuple[str, str] | None = None,
) -> tuple[int, dict]:
    dates = matrices["dates"]
    sim_start_idx = int(dates.searchsorted(pd.Timestamp(start)))
    state_start_idx = int(
        dates.searchsorted(pd.Timestamp(state_start or start))
    )
    market_exposure_cap, regime_entry_allowed = _entry_gate_arrays(
        cfg, dates, market, regime
    )
    preloaded_calibration = (
        calibration_labels_supplied or bool(quality_labels) or bool(fill_labels)
    )
    if (
        (state_start_idx < sim_start_idx or cfg.simulation_mode == "portfolio")
        and not preloaded_calibration
    ):
        quality_labels, fill_labels = _first_touch_calibration_labels(
            cfg,
            matrices,
            setups,
            candidate_context,
            state_start_idx=state_start_idx,
            market_exposure_cap=market_exposure_cap,
            regime_entry_allowed=regime_entry_allowed,
        )
        # Even an empty hidden calibration result is an intentionally frozen,
        # preloaded label set.  It must not silently fall back to online labels.
        preloaded_calibration = True
    online_calibration = not preloaded_calibration
    if calibration_label_fingerprints is None:
        calibration_label_fingerprints = (
            reproducibility.quality_calibration_labels_fingerprint(
                quality_labels
            ),
            reproducibility.fill_calibration_labels_fingerprint(fill_labels),
        )
    elif (
        len(calibration_label_fingerprints) != 2
        or not all(
            isinstance(fingerprint, str) and fingerprint
            for fingerprint in calibration_label_fingerprints
        )
    ):
        raise ValueError(
            "calibration_label_fingerprints must contain quality and fill hashes"
        )
    input_fingerprint = _simulation_input_fingerprint(
        cfg,
        setups,
        matrices,
        universe,
        market_exposure_cap,
        regime_entry_allowed,
        regime,
        candidate_context,
        quality_labels_fingerprint=calibration_label_fingerprints[0],
        fill_labels_fingerprint=calibration_label_fingerprints[1],
        online_calibration=online_calibration,
    )
    result = simulate(
        dates, matrices["symbols"],
        matrices["open"], matrices["high"], matrices["low"], matrices["close"],
        setups, cfg, sim_start_idx=sim_start_idx, state_start_idx=state_start_idx,
        market_exposure_cap=market_exposure_cap,
        regime_entry_allowed=regime_entry_allowed,
        volume_m=matrices["volume"],
        candidate_context=candidate_context,
        continuity_segment_m=matrices["price_continuity_segment"],
        quality_labels=quality_labels,
        fill_labels=fill_labels,
        online_calibration=online_calibration,
    )
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
    try:
        run_id = persistence.create_run(
            conn,
            cfg,
            result.metrics,
            start,
            end,
            model_version=model.MODEL_VERSION,
            input_fingerprint=input_fingerprint,
        )
        persistence.write_trades(conn, run_id, trades)
        persistence.write_breakout_events(conn, run_id, breakout_events)
        persistence.write_equity(conn, run_id, result.equity)
        if commit:
            conn.commit()
    except Exception:
        if commit:
            conn.rollback()
        raise
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


def _log_portfolio_ranking_experiment_summary(
    case_name: str,
    metrics_by_salt: list[dict],
) -> None:
    for metric, values in ranking_sensitivity.summarize(metrics_by_salt).items():
        log.info(
            "portfolio ranking experiment summary %s %s samples %d missing %d median %.6f adverse-quantile %.6f worst %.6f",
            case_name,
            metric,
            values["count"],
            values["missing_count"],
            values["median"],
            values["adverse_quantile"],
            values["worst"],
        )


def _run_portfolio_ranking_experiment(
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
    candidate_context: dict[str, pd.DataFrame],
) -> tuple[tuple[int, dict] | None, tuple[tuple[int, dict], ...]]:
    """Run the frozen ranking-mode, setup-sleeve and bootstrap-salt matrix.

    Market-on calibration is generated exactly once.  Each salt applies a
    deterministic completion-day cluster bootstrap to those immutable base
    labels, and that same weighted tuple is reused by every case for the salt.
    Each arm commits independently so the full equity/event paths do not
    accumulate in one oversized database transaction.  No arm is selected as
    a winner; only predeclared case distributions are logged.
    """
    if cfg.simulation_mode not in ("portfolio", "both"):
        raise ValueError(
            "portfolio ranking experiment requires SIMULATION_MODE portfolio or both"
        )

    total_paths = (
        len(ranking_sensitivity.EXPERIMENT_CASES)
        * len(ranking_sensitivity.NEUTRAL_RANK_SALTS)
    )
    log.info(
        "portfolio ranking experiment start cases %d salts %d paths %d label %s",
        len(ranking_sensitivity.EXPERIMENT_CASES),
        len(ranking_sensitivity.NEUTRAL_RANK_SALTS),
        total_paths,
        cfg.run_label,
    )
    market_exposure_cap, regime_entry_allowed = _entry_gate_arrays(
        cfg, matrices["dates"], market, regime
    )
    state_start_idx = int(matrices["dates"].searchsorted(pd.Timestamp(start)))
    calibration_cfg = replace(
        cfg,
        simulation_mode="independent",
        portfolio_ranking_mode="quality_only",
        portfolio_setup_types=("flat_base", "vcp"),
        neutral_rank_salt=ranking_sensitivity.NEUTRAL_RANK_SALTS[0],
    )
    quality_labels, fill_labels = _first_touch_calibration_labels(
        calibration_cfg,
        matrices,
        setups,
        candidate_context,
        state_start_idx=state_start_idx,
        market_exposure_cap=market_exposure_cap,
        regime_entry_allowed=regime_entry_allowed,
    )
    base_calibration_label_fingerprints = (
        reproducibility.quality_calibration_labels_fingerprint(quality_labels),
        reproducibility.fill_calibration_labels_fingerprint(fill_labels),
    )

    first_touch: tuple[int, dict] | None = None
    if cfg.simulation_mode == "both":
        first_touch_cfg = replace(
            cfg,
            simulation_mode="independent",
            market_filter_enable=False,
            regime_entry_filter_enable=False,
            portfolio_ranking_mode="quality_only",
            portfolio_setup_types=("flat_base", "vcp"),
            neutral_rank_salt=ranking_sensitivity.NEUTRAL_RANK_SALTS[0],
            run_label=f"{cfg.run_label}_first_touch",
        )
        first_touch = _run_simulation(
            conn,
            first_touch_cfg,
            matrices,
            universe,
            market,
            setups,
            regime,
            start,
            end,
            candidate_context=candidate_context,
            quality_labels=quality_labels,
            fill_labels=fill_labels,
            calibration_labels_supplied=True,
            calibration_label_fingerprints=(
                base_calibration_label_fingerprints
            ),
        )

    portfolio_results: list[tuple[int, dict]] = []
    metrics_by_case: dict[str, list[dict]] = {
        case.name: [] for case in ranking_sensitivity.EXPERIMENT_CASES
    }
    for salt_index, salt in enumerate(ranking_sensitivity.NEUTRAL_RANK_SALTS):
        weighted_quality_labels = ranking_sensitivity.bootstrap_quality_labels(
            quality_labels, salt
        )
        weighted_calibration_label_fingerprints = (
            reproducibility.quality_calibration_labels_fingerprint(
                weighted_quality_labels
            ),
            base_calibration_label_fingerprints[1],
        )
        for case in ranking_sensitivity.EXPERIMENT_CASES:
            portfolio_cfg = replace(
                cfg,
                simulation_mode="portfolio",
                portfolio_ranking_mode=case.ranking_mode,
                portfolio_setup_types=case.setup_types,
                neutral_rank_salt=salt,
                run_label=(
                    f"{cfg.run_label}_{case.name}_salt_{salt_index:02d}"
                ),
            )
            run_result = _run_simulation(
                conn,
                portfolio_cfg,
                matrices,
                universe,
                market,
                setups,
                regime,
                start,
                end,
                candidate_context=candidate_context,
                quality_labels=weighted_quality_labels,
                fill_labels=fill_labels,
                calibration_labels_supplied=True,
                calibration_label_fingerprints=(
                    weighted_calibration_label_fingerprints
                ),
            )
            portfolio_results.append(run_result)
            metrics_by_case[case.name].append(run_result[1])

    for case in ranking_sensitivity.EXPERIMENT_CASES:
        _log_portfolio_ranking_experiment_summary(
            case.name, metrics_by_case[case.name]
        )

    log.info(
        "portfolio ranking experiment done cases %d salts %d portfolio-paths %d persisted-runs %d",
        len(ranking_sensitivity.EXPERIMENT_CASES),
        len(ranking_sensitivity.NEUTRAL_RANK_SALTS),
        len(portfolio_results),
        len(portfolio_results) + int(first_touch is not None),
    )
    return first_touch, tuple(portfolio_results)


def run_sim(
    conn, cfg: Config, matrices: dict, universe: pd.DataFrame,
    market: pd.DataFrame, screen_daily: pd.DataFrame, start, end, *,
    setups: pd.DataFrame | None = None,
) -> (
    tuple[int, dict]
    | tuple[tuple[int, dict], tuple[int, dict]]
    | tuple[tuple[int, dict] | None, tuple[tuple[int, dict], ...]]
):
    if setups is None:
        setups = persistence.read_setups(conn, start, end)
    if setups.empty:
        log.warning("no setups in %s..%s -> persisting zero-signal run", start, end)
    regime = data_loader.load_regime_scores(conn, cfg)
    candidate_context = _candidate_context_matrices(
        screen_daily, matrices["dates"], matrices["symbols"]
    )

    if cfg.portfolio_ranking_experiment_enable:
        return _run_portfolio_ranking_experiment(
            conn,
            cfg,
            matrices,
            universe,
            market,
            setups,
            regime,
            start,
            end,
            candidate_context=candidate_context,
        )

    if cfg.simulation_mode == "both":
        market_exposure_cap, regime_entry_allowed = _entry_gate_arrays(
            cfg, matrices["dates"], market, regime
        )
        state_start_idx = int(
            matrices["dates"].searchsorted(pd.Timestamp(start))
        )
        quality_labels, fill_labels = _first_touch_calibration_labels(
            cfg,
            matrices,
            setups,
            candidate_context,
            state_start_idx=state_start_idx,
            market_exposure_cap=market_exposure_cap,
            regime_entry_allowed=regime_entry_allowed,
        )
        calibration_label_fingerprints = (
            reproducibility.quality_calibration_labels_fingerprint(
                quality_labels
            ),
            reproducibility.fill_calibration_labels_fingerprint(fill_labels),
        )
        first_touch_cfg = replace(
            cfg,
            simulation_mode="independent",
            market_filter_enable=False,
            regime_entry_filter_enable=False,
            portfolio_ranking_mode="quality_only",
            portfolio_setup_types=("flat_base", "vcp"),
            run_label=f"{cfg.run_label}_first_touch",
        )
        portfolio_cfg = replace(
            cfg,
            simulation_mode="portfolio",
            run_label=f"{cfg.run_label}_portfolio",
        )
        try:
            first_touch = _run_simulation(
                conn,
                first_touch_cfg,
                matrices,
                universe,
                market,
                setups,
                regime,
                start,
                end,
                candidate_context=candidate_context,
                commit=False,
                quality_labels=quality_labels,
                fill_labels=fill_labels,
                calibration_labels_supplied=True,
                calibration_label_fingerprints=calibration_label_fingerprints,
            )
            portfolio = _run_simulation(
                conn,
                portfolio_cfg,
                matrices,
                universe,
                market,
                setups,
                regime,
                start,
                end,
                candidate_context=candidate_context,
                commit=False,
                quality_labels=quality_labels,
                fill_labels=fill_labels,
                calibration_labels_supplied=True,
                calibration_label_fingerprints=calibration_label_fingerprints,
            )
            conn.commit()
            return first_touch, portfolio
        except Exception:
            conn.rollback()
            raise

    if cfg.simulation_mode == "independent":
        market_exposure_cap, regime_entry_allowed = _entry_gate_arrays(
            cfg, matrices["dates"], market, regime
        )
        state_start_idx = int(
            matrices["dates"].searchsorted(pd.Timestamp(start))
        )
        quality_labels, fill_labels = _first_touch_calibration_labels(
            cfg,
            matrices,
            setups,
            candidate_context,
            state_start_idx=state_start_idx,
            market_exposure_cap=market_exposure_cap,
            regime_entry_allowed=regime_entry_allowed,
        )
        calibration_label_fingerprints = (
            reproducibility.quality_calibration_labels_fingerprint(
                quality_labels
            ),
            reproducibility.fill_calibration_labels_fingerprint(fill_labels),
        )
        first_touch_cfg = replace(
            cfg,
            market_filter_enable=False,
            regime_entry_filter_enable=False,
            portfolio_ranking_mode="quality_only",
            portfolio_setup_types=("flat_base", "vcp"),
        )
        return _run_simulation(
            conn,
            first_touch_cfg,
            matrices,
            universe,
            market,
            setups,
            regime,
            start,
            end,
            candidate_context=candidate_context,
            quality_labels=quality_labels,
            fill_labels=fill_labels,
            calibration_labels_supplied=True,
            calibration_label_fingerprints=calibration_label_fingerprints,
        )

    return _run_simulation(
        conn,
        cfg,
        matrices,
        universe,
        market,
        setups,
        regime,
        start,
        end,
        candidate_context=candidate_context,
    )


def run_sensitivity(
    conn,
    cfg: Config,
    prices: pd.DataFrame,
    screen_daily: pd.DataFrame,
    matrices: dict,
    universe: pd.DataFrame,
    market: pd.DataFrame,
    start: date,
    end: date,
) -> None:
    if cfg.simulation_mode == "both":
        raise ValueError(
            "sensitivity requires SIMULATION_MODE independent or portfolio"
        )
    periods = sensitivity.phases(start, end)
    regime = data_loader.load_regime_scores(conn, cfg)
    detection_cache: dict[str, pd.DataFrame] = {}
    setup_pass_days = _screen_pass_rows(screen_daily)

    log.info(
        "sensitivity start development-only market-filter ablation variants %d periods %d end-limit %s screen-pass-days %d symbols %d",
        len(sensitivity.VARIANTS),
        len(periods),
        sensitivity.DEVELOPMENT_END_DATE,
        len(setup_pass_days),
        setup_pass_days["symbol"].nunique(),
    )
    for variant in sensitivity.VARIANTS:
        if variant.detection_key not in detection_cache:
            detection_cfg = variant.apply(cfg, "full", start, end)
            detection_cache[variant.detection_key] = detect_setups(
                detection_cfg,
                prices,
                setup_pass_days,
                universe,
                matrices["dates"],
            )
        all_setups = detection_cache[variant.detection_key]

        for phase, phase_start, phase_end in periods:
            phase_cfg = variant.apply(cfg, phase, phase_start, phase_end)
            detect_date = pd.to_datetime(all_setups["detect_date"]).dt.date
            phase_setups = all_setups.loc[
                detect_date <= phase_end
            ].reset_index(drop=True)
            phase_setups = _prepare_simulation_setups(phase_setups)
            phase_matrices = _slice_matrices(matrices, phase_end)
            candidate_context = _candidate_context_matrices(
                screen_daily,
                phase_matrices["dates"],
                phase_matrices["symbols"],
            )

            log.info(
                "sensitivity variant %s period %s %s %s market-filter %s setups %d",
                variant.name,
                phase,
                phase_start,
                phase_end,
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
                candidate_context=candidate_context,
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
        if cfg.simulation_mode == "both":
            raise ValueError(
                "sensitivity requires SIMULATION_MODE independent or portfolio"
            )
        if cfg.end_date is None:
            raise ValueError(
                "development-only market-filter ablation requires END_DATE=2023-12-31"
            )
        sensitivity.validate_configured_window(
            date.fromisoformat(cfg.start_date), date.fromisoformat(cfg.end_date)
        )

    if cfg.stage == "all":
        stages = ("screen", "setup", "sim")
    elif cfg.stage == "sensitivity":
        stages = ("screen", "sensitivity")
    else:
        stages = (cfg.stage,)

    conn = db.get_conn()
    db.acquire_pipeline_lock(conn)
    db.validate_result_schema(conn)
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
    for field in (
        "open", "high", "low", "volume", "raw_close",
        "price_continuity_segment", "price_continuity_break",
    ):
        matrices[field] = data_loader.pivot_field(prices, field)
    source_input_fingerprint = _source_input_fingerprint(matrices, universe)

    dates = matrices["dates"]
    window = np.asarray(dates >= pd.Timestamp(cfg.start_date))
    if not window.any():
        raise SystemExit(f"no price data on/after START_DATE={cfg.start_date}")
    start, end = dates[window][0].date(), dates[window][-1].date()

    if {"screen", "sim", "sensitivity"}.intersection(stages):
        index_bars = data_loader.load_market_indexes(
            conn, _market_data_config(cfg)
        )
        market = market_filter.compute_market_model(
            matrices["close"], index_bars, cfg
        )
        log.info(
            "Market model latest %s breadth %.1f%% entry cap %.0f%% active %d %d days",
            market["market_status"].iloc[-1],
            100 * market["market_breadth"].iloc[-1],
            100 * market["entry_exposure_cap"].iloc[-1],
            int(market.loc[window, "market_on"].sum()), int(window.sum()),
        )
    else:
        market = pd.DataFrame(index=matrices["dates"])
    pass_days = None
    setups = None
    screen_output_fingerprint = None
    setup_output_fingerprint = None
    setup_input_fingerprint = None
    screen_config_fingerprint = _screen_config_fingerprint(cfg)
    if "screen" in stages:
        pass_days = run_screen(conn, cfg, matrices, market, universe, window)
        screen_output_fingerprint = _fingerprint_available(
            pass_days, SCREEN_FINGERPRINT_COLUMNS
        )
        persistence.write_stage_state(
            conn,
            stage="screen",
            model_version=model.MODEL_VERSION,
            config_fingerprint=screen_config_fingerprint,
            input_fingerprint=source_input_fingerprint,
            output_fingerprint=screen_output_fingerprint,
            start=start,
            end=end,
        )
    if "setup" in stages:
        if pass_days is None:
            screen_output_fingerprint = persistence.require_stage_state(
                conn,
                stage="screen",
                model_version=model.MODEL_VERSION,
                config_fingerprint=screen_config_fingerprint,
                input_fingerprint=source_input_fingerprint,
                start=start,
                end=end,
            )
            pass_days = persistence.read_screen_stage_output(conn, start, end)
            _require_matching_fingerprint(
                pass_days,
                SCREEN_FINGERPRINT_COLUMNS,
                screen_output_fingerprint,
                "screen",
            )
        assert screen_output_fingerprint is not None
        setups = run_setup(
            conn,
            cfg,
            prices,
            _screen_pass_rows(pass_days),
            universe,
            matrices["dates"],
            start,
            end,
            source_fingerprint=source_input_fingerprint,
        )
        setup_input_fingerprint = _setup_input_fingerprint(
            screen_output_fingerprint, source_input_fingerprint
        )
        setup_config_fingerprint = _setup_config_fingerprint(
            cfg, setup_input_fingerprint
        )
        setup_output_fingerprint = _fingerprint_available(
            setups, SETUP_FINGERPRINT_COLUMNS
        )
        persistence.write_stage_state(
            conn,
            stage="setup",
            model_version=model.MODEL_VERSION,
            config_fingerprint=setup_config_fingerprint,
            input_fingerprint=setup_input_fingerprint,
            output_fingerprint=setup_output_fingerprint,
            start=start,
            end=end,
        )
    if "sim" in stages:
        if screen_output_fingerprint is None:
            screen_output_fingerprint = persistence.require_stage_state(
                conn,
                stage="screen",
                model_version=model.MODEL_VERSION,
                config_fingerprint=screen_config_fingerprint,
                input_fingerprint=source_input_fingerprint,
                start=start,
                end=end,
            )
            pass_days = persistence.read_screen_stage_output(conn, start, end)
            _require_matching_fingerprint(
                pass_days,
                SCREEN_FINGERPRINT_COLUMNS,
                screen_output_fingerprint,
                "screen",
            )
        assert pass_days is not None
        setup_input_fingerprint = _setup_input_fingerprint(
            screen_output_fingerprint, source_input_fingerprint
        )
        setup_config_fingerprint = _setup_config_fingerprint(
            cfg, setup_input_fingerprint
        )
        if setup_output_fingerprint is None:
            setup_output_fingerprint = persistence.require_stage_state(
                conn,
                stage="setup",
                model_version=model.MODEL_VERSION,
                config_fingerprint=setup_config_fingerprint,
                input_fingerprint=setup_input_fingerprint,
                start=start,
                end=end,
            )
        setups = persistence.read_setups(conn, start, end)
        _require_matching_fingerprint(
            setups,
            SETUP_FINGERPRINT_COLUMNS,
            setup_output_fingerprint,
            "setup",
        )
        run_sim(
            conn,
            cfg,
            matrices,
            universe,
            market,
            pass_days,
            start,
            end,
            setups=setups,
        )
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
