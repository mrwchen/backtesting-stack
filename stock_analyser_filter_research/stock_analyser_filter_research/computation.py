from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    CRITERION_COLUMNS,
    IDENTITY_COLUMNS,
    SIGNAL_COLUMNS,
    SOURCE_BOOLEAN_COLUMNS,
    SOURCE_COLUMNS,
)


def empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _rolling_in_segment(
    values: pd.Series,
    segment: pd.Series,
    window: int,
    operation: str,
    *,
    offset: int = 0,
) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    for segment_id in segment.dropna().unique():
        in_segment = segment.eq(segment_id)
        isolated = values.where(in_segment).shift(offset)
        rolled = getattr(
            isolated.rolling(window, min_periods=window), operation
        )()
        result.loc[in_segment] = rolled.loc[in_segment]
    return result


def _lag_in_segment(
    values: pd.Series, segment: pd.Series, periods: int
) -> pd.Series:
    shifted = values.shift(periods)
    return shifted.where(segment.eq(segment.shift(periods)))


def _complete_prior_window(
    values: pd.Series, segment: pd.Series, window: int
) -> pd.Series:
    return _rolling_in_segment(
        values, segment, window, "count", offset=1
    ).eq(window)


def _pct_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return ((numerator / denominator) - 1.0).mul(100.0).where(
        numerator.notna() & denominator.gt(0)
    )


def _prior_return(
    close: pd.Series, segment: pd.Series, sessions: int
) -> pd.Series:
    end = _lag_in_segment(close, segment, 1)
    start = _lag_in_segment(close, segment, sessions + 1)
    complete = _complete_prior_window(close, segment, sessions + 1)
    return _pct_ratio(end, start).where(complete)


def _prior_correlation(
    left: pd.Series,
    right: pd.Series,
    segment: pd.Series,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=left.index, dtype=float)
    for segment_id in segment.dropna().unique():
        in_segment = segment.eq(segment_id)
        prior_left = left.where(in_segment).shift(1)
        prior_right = right.where(in_segment).shift(1)
        correlated = prior_left.rolling(
            window, min_periods=window
        ).corr(prior_right)
        result.loc[in_segment] = correlated.loc[in_segment]
    return result


def _nullable_flag(
    index: pd.Index,
    available: pd.Series,
    predicate: Callable[[pd.Index], pd.Series | np.ndarray],
) -> pd.Series:
    result = pd.Series(pd.NA, index=index, dtype="boolean")
    selected = available[available].index
    if len(selected):
        result.loc[selected] = np.asarray(predicate(selected), dtype=bool)
    return result


def _analysis_split(
    signal_dates: pd.Series,
    label_available_dates: pd.Series,
    cfg: Config,
) -> pd.Series:
    dates = pd.to_datetime(signal_dates).dt.normalize()
    availability = pd.to_datetime(label_available_dates).dt.normalize()
    discovery_end = pd.Timestamp(cfg.discovery_end_date)
    validation_end = pd.Timestamp(cfg.validation_end_date)
    discovery_eligible = (
        (dates <= discovery_end)
        & availability.notna()
        & (availability <= discovery_end)
    )
    validation_eligible = (
        (dates > discovery_end)
        & (dates <= validation_end)
        & availability.notna()
        & (availability <= validation_end)
    )
    values = np.where(
        discovery_eligible,
        "discovery",
        np.where(
            validation_eligible,
            "validation",
            np.where(dates > validation_end, "test", "purged"),
        ),
    )
    return pd.Series(values, index=signal_dates.index, dtype="string")


def _event_max_drawdown(
    close: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
    window: int,
) -> pd.Series:
    result = pd.Series(np.nan, index=close.index, dtype=float)
    for position in signal_positions:
        if position < window:
            continue
        values = close.iloc[position - window : position].to_numpy(dtype=float)
        segments = segment.iloc[position - window : position]
        current_segment = segment.iloc[position]
        if (
            len(values) != window
            or not np.isfinite(values).all()
            or current_segment != current_segment
            or not segments.eq(current_segment).all()
            or np.any(values <= 0)
        ):
            continue
        running_peak = np.maximum.accumulate(values)
        result.iloc[position] = float(
            np.min(values / running_peak - 1.0) * 100.0
        )
    return result


def _event_text_and_history(
    criteria: dict[str, pd.Series],
    trend_pass: pd.Series,
    segment: pd.Series,
    signal_positions: np.ndarray,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    trigger_text = pd.Series(pd.NA, index=trend_pass.index, dtype="string")
    trigger_count = pd.Series(np.nan, index=trend_pass.index, dtype=float)
    previous_count = pd.Series(np.nan, index=trend_pass.index, dtype=float)
    sessions_since = pd.Series(np.nan, index=trend_pass.index, dtype=float)

    signal_set = set(int(value) for value in signal_positions)
    last_pass_by_segment: dict[int, int] = {}
    last_observed_segment: int | None = None
    for position in range(len(trend_pass)):
        segment_value = segment.iloc[position]
        if pd.isna(segment_value):
            # A missing global session terminates every history chain for this
            # identity, even if an upstream segment identifier were reused.
            last_pass_by_segment.clear()
            last_observed_segment = None
            continue
        segment_id = int(segment_value)
        if (
            last_observed_segment is not None
            and segment_id != last_observed_segment
        ):
            last_pass_by_segment.clear()
        last_observed_segment = segment_id
        pass_value = trend_pass.iloc[position]
        is_pass = bool(pass_value) if pd.notna(pass_value) else False
        if position in signal_set:
            triggered: list[str] = []
            for name, values in criteria.items():
                current_value = values.iloc[position]
                previous_value = values.iloc[position - 1]
                if (
                    pd.notna(current_value)
                    and pd.notna(previous_value)
                    and bool(current_value)
                    and not bool(previous_value)
                ):
                    triggered.append(name)
            trigger_text.iloc[position] = ",".join(triggered)
            trigger_count.iloc[position] = len(triggered)
            previous_count.iloc[position] = sum(
                bool(values.iloc[position - 1]) for values in criteria.values()
            )
            previous_position = last_pass_by_segment.get(segment_id)
            if previous_position is not None:
                sessions_since.iloc[position] = position - previous_position
        if is_pass:
            last_pass_by_segment[segment_id] = position
    return trigger_text, trigger_count, previous_count, sessions_since


def calculate_identity_signals(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    if rows.empty:
        return empty_signal_frame()
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")
    if rows.duplicated(["period_end_date"]).any():
        raise ValueError("one identity contains duplicate dates")
    identities = rows.loc[:, list(IDENTITY_COLUMNS)].drop_duplicates()
    if len(identities) != 1:
        raise ValueError("calculate_identity_signals requires exactly one identity")

    rows = rows.sort_values("period_end_date", kind="stable")
    indexed = rows.set_index("period_end_date").reindex(trading_dates)
    observed = indexed["symbol"].notna()
    segment = _numeric(indexed["price_continuity_segment"])
    same_previous_segment = (
        observed
        & observed.shift(1, fill_value=False)
        & segment.notna()
        & segment.eq(segment.shift(1))
    )

    criteria = {
        column: indexed[column].astype("boolean")
        for column in CRITERION_COLUMNS
    }
    trend_pass = indexed["trend_template_pass"].astype("boolean")
    previous_pass = trend_pass.shift(1)
    signal_mask = (
        trend_pass.eq(True).fillna(False)
        & previous_pass.eq(False).fillna(False)
        & same_previous_segment
    )
    signal_positions = np.flatnonzero(signal_mask.to_numpy(dtype=bool))
    if not len(signal_positions):
        return empty_signal_frame()

    close = _numeric(indexed["adjusted_close"])
    high = _numeric(indexed["adjusted_high"])
    low = _numeric(indexed["adjusted_low"])
    volume = _numeric(indexed["adjusted_volume"])
    notional = _numeric(indexed["daily_traded_notional_usd"])
    rs_rating = _numeric(indexed["rs_rating"])

    previous_close = _lag_in_segment(close, segment, 1)
    daily_return = (close / previous_close - 1.0).where(
        close.gt(0) & previous_close.gt(0)
    )
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    normalized_true_range = (true_range / previous_close).where(
        previous_close.gt(0)
    )

    prior_close = _lag_in_segment(close, segment, 1)
    prior_high20 = _rolling_in_segment(
        high, segment, 20, "max", offset=1
    )
    prior_high63 = _rolling_in_segment(
        high, segment, 63, "max", offset=1
    )
    recent_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=1
    )
    older_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=11
    )

    prior_volume5 = _rolling_in_segment(
        volume, segment, 5, "mean", offset=1
    )
    prior_volume10 = _rolling_in_segment(
        volume, segment, 10, "mean", offset=1
    )
    prior_volume21 = _rolling_in_segment(
        volume, segment, 21, "mean", offset=1
    )
    prior_notional5 = _rolling_in_segment(
        notional, segment, 5, "mean", offset=1
    )
    prior_notional10 = _rolling_in_segment(
        notional, segment, 10, "mean", offset=1
    )
    prior_notional21 = _rolling_in_segment(
        notional, segment, 21, "mean", offset=1
    )

    up_volume = volume.where(daily_return.gt(0), 0.0).where(
        volume.notna() & daily_return.notna()
    )
    nonflat_volume = volume.where(daily_return.ne(0), 0.0).where(
        volume.notna() & daily_return.notna()
    )
    up_notional = notional.where(daily_return.gt(0), 0.0).where(
        notional.notna() & daily_return.notna()
    )
    nonflat_notional = notional.where(daily_return.ne(0), 0.0).where(
        notional.notna() & daily_return.notna()
    )
    prior_up_volume = _rolling_in_segment(
        up_volume, segment, 21, "sum", offset=1
    )
    prior_nonflat_volume = _rolling_in_segment(
        nonflat_volume, segment, 21, "sum", offset=1
    )
    prior_up_notional = _rolling_in_segment(
        up_notional, segment, 21, "sum", offset=1
    )
    prior_nonflat_notional = _rolling_in_segment(
        nonflat_notional, segment, 21, "sum", offset=1
    )

    criteria_count = sum(
        values.astype("Float64") for values in criteria.values()
    ).astype(float)
    exactly_seven = criteria_count.eq(7).astype(float).where(observed)
    prior_seven_count = _rolling_in_segment(
        exactly_seven, segment, 10, "sum", offset=1
    )
    (
        trigger_text,
        trigger_count,
        previous_criteria_count,
        sessions_since_previous_pass,
    ) = _event_text_and_history(
        criteria, trend_pass, segment, signal_positions
    )

    ma50 = _numeric(indexed["ma50"])
    ma150 = _numeric(indexed["ma150"])
    ma200 = _numeric(indexed["ma200"])
    ma200_prior = _numeric(indexed["ma200_21_sessions_ago"])
    low52 = _numeric(indexed["low_52w"])
    high52 = _numeric(indexed["high_52w"])

    prior_return5 = _prior_return(close, segment, 5)
    prior_return10 = _prior_return(close, segment, 10)
    prior_return21 = _prior_return(close, segment, 21)
    prior_return_previous5 = _pct_ratio(
        _lag_in_segment(close, segment, 6),
        _lag_in_segment(close, segment, 11),
    ).where(_complete_prior_window(close, segment, 11))
    prior_momentum_acceleration = prior_return5 - prior_return_previous5

    prior_atr14 = _rolling_in_segment(
        true_range, segment, 14, "mean", offset=1
    )
    prior_atr14_pct = (prior_atr14 / prior_close).mul(100.0).where(
        prior_close.gt(0)
    )
    max_drawdown21 = _event_max_drawdown(
        close, segment, signal_positions, 21
    )
    close_location = ((close - low) / (high - low)).where(high.gt(low))
    prior_rs_change5 = (
        _lag_in_segment(rs_rating, segment, 1)
        - _lag_in_segment(rs_rating, segment, 6)
    ).where(_complete_prior_window(rs_rating, segment, 6))

    log_volume = np.log1p(volume.where(volume.ge(0)))
    log_notional = np.log1p(notional.where(notional.ge(0)))

    features: dict[str, pd.Series] = {
        "trigger_criteria": trigger_text,
        "trigger_count": trigger_count,
        "previous_criteria_pass_count": previous_criteria_count,
        "prior_7_of_8_count_10d": prior_seven_count,
        "sessions_since_previous_pass": sessions_since_previous_pass,
        "distance_to_ma50_pct": _pct_ratio(close, ma50),
        "distance_to_ma150_pct": _pct_ratio(close, ma150),
        "distance_to_ma200_pct": _pct_ratio(close, ma200),
        "ma50_vs_ma150_pct": _pct_ratio(ma50, ma150),
        "ma50_vs_ma200_pct": _pct_ratio(ma50, ma200),
        "ma150_vs_ma200_pct": _pct_ratio(ma150, ma200),
        "ma200_slope_21d_pct": _pct_ratio(ma200, ma200_prior),
        "price_vs_52w_low_pct": _pct_ratio(close, low52),
        "price_vs_52w_high_pct": _pct_ratio(close, high52),
        "prior_return_5d_pct": prior_return5,
        "prior_return_10d_pct": prior_return10,
        "prior_return_21d_pct": prior_return21,
        "prior_momentum_acceleration_5d_pct_points": (
            prior_momentum_acceleration
        ),
        "prior_atr_14d_pct": prior_atr14_pct,
        "prior_max_drawdown_21d_pct": max_drawdown21,
        "prior_close_vs_20d_high_pct": _pct_ratio(
            prior_close, prior_high20
        ),
        "prior_close_vs_63d_high_pct": _pct_ratio(
            prior_close, prior_high63
        ),
        "signal_close_vs_prior_20d_high_pct": _pct_ratio(
            close, prior_high20
        ),
        "prior_range_compression_10_vs_10_ratio": (
            recent_normalized_range10 / older_normalized_range10
        ).where(older_normalized_range10.gt(0)),
        "signal_close_location_value": close_location,
        "prior_rs_rating_change_5d": prior_rs_change5,
        "prior_volume_sma5_vs21_ratio": (
            prior_volume5 / prior_volume21
        ).where(prior_volume21.gt(0)),
        "prior_volume_sma10_vs21_ratio": (
            prior_volume10 / prior_volume21
        ).where(prior_volume21.gt(0)),
        "prior_notional_sma5_vs21_ratio": (
            prior_notional5 / prior_notional21
        ).where(prior_notional21.gt(0)),
        "prior_notional_sma10_vs21_ratio": (
            prior_notional10 / prior_notional21
        ).where(prior_notional21.gt(0)),
        "prior_up_volume_share21": (
            prior_up_volume / prior_nonflat_volume
        ).where(prior_nonflat_volume.gt(0)),
        "prior_up_notional_share21": (
            prior_up_notional / prior_nonflat_notional
        ).where(prior_nonflat_notional.gt(0)),
        "prior_price_volume_corr21": _prior_correlation(
            daily_return, log_volume, segment, 21
        ),
        "prior_price_notional_corr21": _prior_correlation(
            daily_return, log_notional, segment, 21
        ),
    }

    signal_index = trading_dates[signal_positions]
    output = indexed.loc[signal_index, [
        *IDENTITY_COLUMNS,
        "price_continuity_segment",
        "currency",
        "raw_close",
        "adjusted_close",
        "adjusted_high",
        "adjusted_low",
        "adjusted_volume",
        "daily_price_change_pct",
        "adjusted_volume_sma21_prior",
        "adjusted_volume_vs_sma21_prior_ratio",
        "adjusted_volume_sma50_prior",
        "adjusted_volume_vs_sma50_prior_ratio",
        "daily_traded_notional_usd",
        "daily_traded_notional_sma21_prior_usd",
        "daily_traded_notional_vs_sma21_prior_ratio",
        "daily_traded_notional_sma50_prior_usd",
        "daily_traded_notional_vs_sma50_prior_ratio",
        "dollar_volume_63d",
        "ma50",
        "ma150",
        "ma200",
        "ma200_21_sessions_ago",
        "low_52w",
        "high_52w",
        "rs_raw",
        "rs_rating",
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_10d_max_gain_pct",
        "forward_10d_max_loss_pct",
        "forward_20d_max_gain_pct",
        "forward_20d_max_loss_pct",
    ]].copy()
    output.insert(0, "signal_date", signal_index)
    output.insert(
        1,
        "previous_session_date",
        trading_dates[signal_positions - 1],
    )
    label_available_dates = pd.Series(
        [
            (
                trading_dates[position + 5]
                if position + 5 < len(trading_dates)
                else pd.NaT
            )
            for position in signal_positions
        ]
    )
    for name, values in features.items():
        output[name] = values.loc[signal_index].to_numpy()

    output = output.reset_index(drop=True)
    output["signal_date"] = pd.to_datetime(output["signal_date"])
    output["previous_session_date"] = pd.to_datetime(
        output["previous_session_date"]
    )
    in_date_range = output["signal_date"].dt.date >= cfg.signal_start_date
    if cfg.signal_end_date is not None:
        in_date_range &= output["signal_date"].dt.date <= cfg.signal_end_date
    output = output.loc[in_date_range].reset_index(drop=True)
    if output.empty:
        return empty_signal_frame()

    gain5 = _numeric(output["forward_5d_max_gain_pct"])
    loss5 = _numeric(output["forward_5d_max_loss_pct"])
    gain10 = _numeric(output["forward_10d_max_gain_pct"])
    gain20 = _numeric(output["forward_20d_max_gain_pct"])
    complete5 = gain5.notna() & loss5.notna()
    output["weak_5d"] = _nullable_flag(
        output.index,
        complete5,
        lambda selected: gain5.loc[selected] < cfg.weak_5d_max_gain_pct,
    )
    output["strong_5d"] = _nullable_flag(
        output.index,
        complete5,
        lambda selected: gain5.loc[selected] >= cfg.strong_5d_min_gain_pct,
    )
    output["deep_loss_5d"] = _nullable_flag(
        output.index,
        complete5,
        lambda selected: loss5.loc[selected] <= cfg.deep_loss_5d_max_loss_pct,
    )
    output["bad_5d"] = (
        output["weak_5d"] | output["deep_loss_5d"]
    ).astype("boolean")
    complete10 = complete5 & gain10.notna()
    complete20 = complete5 & gain20.notna()
    output["late_strong_10d"] = _nullable_flag(
        output.index,
        complete10,
        lambda selected: (
            gain5.loc[selected] < cfg.strong_5d_min_gain_pct
        )
        & (gain10.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    output["late_strong_20d"] = _nullable_flag(
        output.index,
        complete20,
        lambda selected: (
            gain5.loc[selected] < cfg.strong_5d_min_gain_pct
        )
        & (gain20.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    # The five-session outcome must be fully observable inside Discovery or
    # Validation. Boundary signals remain stored as "purged" but cannot leak
    # labels across the next split.
    label_available_dates = label_available_dates.loc[in_date_range].reset_index(
        drop=True
    )
    output["analysis_split"] = _analysis_split(
        output["signal_date"], label_available_dates, cfg
    )
    output["include_stage_a"] = True
    output["include_stage_ab"] = True
    output["include_stage_abc"] = True
    output["matched_rule_ids"] = pd.NA
    output["filter_decision"] = "include"
    output["exclusion_reason"] = pd.NA

    for column in SIGNAL_COLUMNS:
        if column not in output:
            raise AssertionError(f"calculation did not produce {column}")
    return output.loc[:, SIGNAL_COLUMNS]


def calculate_signal_batch(
    source: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> pd.DataFrame:
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError("source data is missing columns: " + ", ".join(missing))
    if source.empty:
        return empty_signal_frame()

    data = source.copy()
    data["period_end_date"] = pd.to_datetime(
        data["period_end_date"], errors="raise"
    ).dt.normalize()
    if data.duplicated([*IDENTITY_COLUMNS, "period_end_date"]).any():
        raise ValueError("source contains duplicate identity/date rows")
    if data.loc[:, list(IDENTITY_COLUMNS)].isna().any().any():
        raise ValueError("source identity columns must not be null")
    currency = data["currency"].astype("string").str.upper()
    if currency.isna().any() or currency.ne("USD").any():
        raise ValueError("source currency must be USD")
    data["currency"] = currency
    for column in SOURCE_BOOLEAN_COLUMNS:
        data[column] = data[column].astype(bool)

    results = [
        calculate_identity_signals(group, trading_dates, cfg)
        for _, group in data.groupby(
            list(IDENTITY_COLUMNS), sort=True, observed=True
        )
    ]
    nonempty = [frame for frame in results if not frame.empty]
    if not nonempty:
        return empty_signal_frame()
    return pd.concat(nonempty, ignore_index=True).loc[:, SIGNAL_COLUMNS]
