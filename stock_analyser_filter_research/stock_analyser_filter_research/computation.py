from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .contracts import (
    CRITERION_COLUMNS,
    EARLY_CUT_COLUMNS,
    EARLY_CUT_LANDMARK_DAYS,
    IDENTITY_COLUMNS,
    SIGNAL_COLUMNS,
    SOURCE_BOOLEAN_COLUMNS,
    SOURCE_COLUMNS,
)


@dataclass(frozen=True)
class CalculationBatchResult:
    """One bounded worker result, before any research rules are fitted."""

    signals: pd.DataFrame
    early_cut: pd.DataFrame

    @property
    def early_cut_results(self) -> pd.DataFrame:
        """Descriptive alias for callers that prefer the table-oriented name."""

        return self.early_cut


def empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=SIGNAL_COLUMNS)


def empty_early_cut_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EARLY_CUT_COLUMNS)


def empty_calculation_batch() -> CalculationBatchResult:
    return CalculationBatchResult(
        signals=empty_signal_frame(),
        early_cut=empty_early_cut_frame(),
    )


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
        rolled = getattr(isolated.rolling(window, min_periods=window), operation)()
        result.loc[in_segment] = rolled.loc[in_segment]
    return result


def _lag_in_segment(values: pd.Series, segment: pd.Series, periods: int) -> pd.Series:
    shifted = values.shift(periods)
    return shifted.where(segment.eq(segment.shift(periods)))


def _complete_prior_window(
    values: pd.Series, segment: pd.Series, window: int
) -> pd.Series:
    return _rolling_in_segment(values, segment, window, "count", offset=1).eq(window)


def _pct_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (
        ((numerator / denominator) - 1.0)
        .mul(100.0)
        .where(numerator.notna() & denominator.gt(0))
    )


def _prior_return(close: pd.Series, segment: pd.Series, sessions: int) -> pd.Series:
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
        correlated = prior_left.rolling(window, min_periods=window).corr(prior_right)
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
    holdout_cutoff = pd.Timestamp(cfg.holdout_cutoff_date)
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
    diagnostic_eligible = (
        (dates > validation_end)
        & (dates <= holdout_cutoff)
        & availability.notna()
        & (availability <= holdout_cutoff)
    )
    values = np.where(
        discovery_eligible,
        "discovery",
        np.where(
            validation_eligible,
            "validation",
            np.where(
                diagnostic_eligible,
                "diagnostic",
                np.where(dates > holdout_cutoff, "holdout", "purged"),
            ),
        ),
    )
    return pd.Series(values, index=signal_dates.index, dtype="string")


def _nullable_bool(value: bool | None) -> Any:
    return pd.NA if value is None else bool(value)


def _finite_positive(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(number) and number > 0)


def _date_at(trading_dates: pd.DatetimeIndex, position: int) -> pd.Timestamp:
    if 0 <= position < len(trading_dates):
        return pd.Timestamp(trading_dates[position])
    return pd.NaT


def _observed_same_segment(
    indexed: pd.DataFrame,
    segment: pd.Series,
    start: int,
    end: int,
    segment_id: float,
) -> bool:
    """Return whether every inclusive global-session position is one chain."""

    if start < 0 or end >= len(indexed) or end < start or pd.isna(segment_id):
        return False
    rows = indexed.iloc[start : end + 1]
    if not rows["symbol"].notna().all():
        return False
    return bool(segment.iloc[start : end + 1].eq(segment_id).all())


def _complete_price_path(
    indexed: pd.DataFrame,
    segment: pd.Series,
    start: int,
    end: int,
    segment_id: float,
) -> bool:
    if not _observed_same_segment(indexed, segment, start, end, segment_id):
        return False
    values = indexed.iloc[start : end + 1].loc[
        :, ["adjusted_close", "adjusted_high", "adjusted_low"]
    ]
    numeric = values.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return bool(numeric.size and np.isfinite(numeric).all() and (numeric > 0).all())


def _first_hit_day(mask: np.ndarray, first_day: int) -> int | None:
    positions = np.flatnonzero(mask)
    return None if not len(positions) else int(first_day + positions[0])


def _gain_loss_order(
    first_strong_day: int | None,
    first_loss_day: int | None,
) -> str:
    if first_strong_day is None and first_loss_day is None:
        return "neither"
    if first_loss_day is None:
        return "gain_only"
    if first_strong_day is None:
        return "loss_only"
    if first_strong_day == first_loss_day:
        return "same_day_ambiguous"
    return "gain_first" if first_strong_day < first_loss_day else "loss_first"


def _continuation_order(
    first_strong_day: int | None,
    first_loss_day: int | None,
) -> str:
    if first_strong_day is None and first_loss_day is None:
        return "neither"
    if first_loss_day is None:
        return "strong_first"
    if first_strong_day is None:
        return "loss_first"
    if first_strong_day == first_loss_day:
        return "same_session_ambiguous"
    return "strong_first" if first_strong_day < first_loss_day else "loss_first"


def _entry_path_outcome(
    indexed: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segment: pd.Series,
    signal_position: int,
    cfg: Config,
) -> dict[str, Any]:
    """Calculate D1..D5 outcomes from actual adjusted high/low paths."""

    label_end_date = _date_at(trading_dates, signal_position + 5)
    result: dict[str, Any] = {
        "forward_5d_max_gain_pct": np.nan,
        "forward_5d_max_loss_pct": np.nan,
        "forward_5d_label_end_date": label_end_date,
        "terminal_close_return_5d_pct": np.nan,
        "first_gain_2pct_day": pd.NA,
        "first_gain_5pct_day": pd.NA,
        "first_loss_5pct_day": pd.NA,
        "gain_loss_order_5d": pd.NA,
        "weak_5d": pd.NA,
        "strong_5d": pd.NA,
        "deep_loss_5d": pd.NA,
        "bad_5d": pd.NA,
        "loss_first_5d": pd.NA,
        "strong_first_5d": pd.NA,
        "_full_path_available": False,
    }
    if pd.isna(label_end_date):
        return result

    segment_id = segment.iloc[signal_position]
    if not _complete_price_path(
        indexed, segment, signal_position, signal_position + 5, segment_id
    ):
        return result

    signal_close = float(indexed["adjusted_close"].iloc[signal_position])
    highs = _numeric(
        indexed["adjusted_high"].iloc[signal_position + 1 : signal_position + 6]
    ).to_numpy(dtype=float)
    lows = _numeric(
        indexed["adjusted_low"].iloc[signal_position + 1 : signal_position + 6]
    ).to_numpy(dtype=float)
    terminal_close = float(indexed["adjusted_close"].iloc[signal_position + 5])
    high_returns = (highs / signal_close - 1.0) * 100.0
    low_returns = (lows / signal_close - 1.0) * 100.0
    max_gain = max(0.0, float(high_returns.max()))
    max_loss = min(0.0, float(low_returns.min()))
    first_gain_2 = _first_hit_day(high_returns >= cfg.weak_5d_max_gain_pct, 1)
    first_gain_5 = _first_hit_day(high_returns >= cfg.strong_5d_min_gain_pct, 1)
    first_loss_5 = _first_hit_day(low_returns <= cfg.deep_loss_5d_max_loss_pct, 1)
    order = _gain_loss_order(first_gain_5, first_loss_5)
    weak = max_gain < cfg.weak_5d_max_gain_pct
    strong = max_gain >= cfg.strong_5d_min_gain_pct
    deep_loss = max_loss <= cfg.deep_loss_5d_max_loss_pct

    result.update(
        {
            "forward_5d_max_gain_pct": max_gain,
            "forward_5d_max_loss_pct": max_loss,
            "terminal_close_return_5d_pct": (
                (terminal_close / signal_close - 1.0) * 100.0
            ),
            "first_gain_2pct_day": (pd.NA if first_gain_2 is None else first_gain_2),
            "first_gain_5pct_day": (pd.NA if first_gain_5 is None else first_gain_5),
            "first_loss_5pct_day": (pd.NA if first_loss_5 is None else first_loss_5),
            "gain_loss_order_5d": order,
            "weak_5d": weak,
            "strong_5d": strong,
            "deep_loss_5d": deep_loss,
            "bad_5d": weak or deep_loss,
            "loss_first_5d": _nullable_bool(
                None
                if order == "same_day_ambiguous"
                else order in {"loss_first", "loss_only"}
            ),
            "strong_first_5d": _nullable_bool(
                None
                if order == "same_day_ambiguous"
                else order in {"gain_first", "gain_only"}
            ),
            "_full_path_available": True,
        }
    )
    return result


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
        result.iloc[position] = float(np.min(values / running_peak - 1.0) * 100.0)
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
        if last_observed_segment is not None and segment_id != last_observed_segment:
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
        column: indexed[column].astype("boolean") for column in CRITERION_COLUMNS
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
    normalized_true_range = (true_range / previous_close).where(previous_close.gt(0))

    prior_close = _lag_in_segment(close, segment, 1)
    prior_high20 = _rolling_in_segment(high, segment, 20, "max", offset=1)
    prior_high63 = _rolling_in_segment(high, segment, 63, "max", offset=1)
    recent_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=1
    )
    older_normalized_range10 = _rolling_in_segment(
        normalized_true_range, segment, 10, "mean", offset=11
    )

    prior_volume5 = _rolling_in_segment(volume, segment, 5, "mean", offset=1)
    prior_volume10 = _rolling_in_segment(volume, segment, 10, "mean", offset=1)
    prior_volume21 = _rolling_in_segment(volume, segment, 21, "mean", offset=1)
    prior_notional5 = _rolling_in_segment(notional, segment, 5, "mean", offset=1)
    prior_notional10 = _rolling_in_segment(notional, segment, 10, "mean", offset=1)
    prior_notional21 = _rolling_in_segment(notional, segment, 21, "mean", offset=1)

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
    prior_up_volume = _rolling_in_segment(up_volume, segment, 21, "sum", offset=1)
    prior_nonflat_volume = _rolling_in_segment(
        nonflat_volume, segment, 21, "sum", offset=1
    )
    prior_up_notional = _rolling_in_segment(up_notional, segment, 21, "sum", offset=1)
    prior_nonflat_notional = _rolling_in_segment(
        nonflat_notional, segment, 21, "sum", offset=1
    )

    criteria_count = sum(
        values.astype("Float64") for values in criteria.values()
    ).astype(float)
    exactly_seven = criteria_count.eq(7).astype(float).where(observed)
    prior_seven_count = _rolling_in_segment(exactly_seven, segment, 10, "sum", offset=1)
    (
        trigger_text,
        trigger_count,
        previous_criteria_count,
        sessions_since_previous_pass,
    ) = _event_text_and_history(criteria, trend_pass, segment, signal_positions)

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

    prior_atr14 = _rolling_in_segment(true_range, segment, 14, "mean", offset=1)
    prior_atr14_pct = (prior_atr14 / prior_close).mul(100.0).where(prior_close.gt(0))
    max_drawdown21 = _event_max_drawdown(close, segment, signal_positions, 21)
    close_location = ((close - low) / (high - low)).where(high.gt(low))
    prior_rs_change5 = (
        _lag_in_segment(rs_rating, segment, 1) - _lag_in_segment(rs_rating, segment, 6)
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
        "prior_momentum_acceleration_5d_pct_points": (prior_momentum_acceleration),
        "prior_atr_14d_pct": prior_atr14_pct,
        "prior_max_drawdown_21d_pct": max_drawdown21,
        "prior_close_vs_20d_high_pct": _pct_ratio(prior_close, prior_high20),
        "prior_close_vs_63d_high_pct": _pct_ratio(prior_close, prior_high63),
        "signal_close_vs_prior_20d_high_pct": _pct_ratio(close, prior_high20),
        "prior_range_compression_10_vs_10_ratio": (
            recent_normalized_range10 / older_normalized_range10
        ).where(older_normalized_range10.gt(0)),
        "signal_close_location_value": close_location,
        "prior_rs_rating_change_5d": prior_rs_change5,
        "prior_volume_sma5_vs21_ratio": (prior_volume5 / prior_volume21).where(
            prior_volume21.gt(0)
        ),
        "prior_volume_sma10_vs21_ratio": (prior_volume10 / prior_volume21).where(
            prior_volume21.gt(0)
        ),
        "prior_notional_sma5_vs21_ratio": (prior_notional5 / prior_notional21).where(
            prior_notional21.gt(0)
        ),
        "prior_notional_sma10_vs21_ratio": (prior_notional10 / prior_notional21).where(
            prior_notional21.gt(0)
        ),
        "prior_up_volume_share21": (prior_up_volume / prior_nonflat_volume).where(
            prior_nonflat_volume.gt(0)
        ),
        "prior_up_notional_share21": (prior_up_notional / prior_nonflat_notional).where(
            prior_nonflat_notional.gt(0)
        ),
        "prior_price_volume_corr21": _prior_correlation(
            daily_return, log_volume, segment, 21
        ),
        "prior_price_notional_corr21": _prior_correlation(
            daily_return, log_notional, segment, 21
        ),
    }

    signal_index = trading_dates[signal_positions]
    output = indexed.loc[
        signal_index,
        [
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
        ],
    ].copy()
    output.insert(0, "signal_date", signal_index)
    output.insert(
        1,
        "previous_session_date",
        trading_dates[signal_positions - 1],
    )
    for name, values in features.items():
        output[name] = values.loc[signal_index].to_numpy()

    output = output.reset_index(drop=True)
    output["signal_date"] = pd.to_datetime(output["signal_date"])
    output["previous_session_date"] = pd.to_datetime(output["previous_session_date"])
    in_date_range = output["signal_date"].dt.date >= cfg.signal_start_date
    if cfg.signal_end_date is not None:
        in_date_range &= output["signal_date"].dt.date <= cfg.signal_end_date
    kept_signal_positions = signal_positions[in_date_range.to_numpy(dtype=bool)]
    output = output.loc[in_date_range].reset_index(drop=True)
    if output.empty:
        return empty_signal_frame()

    path_outcomes = [
        _entry_path_outcome(indexed, trading_dates, segment, int(position), cfg)
        for position in kept_signal_positions
    ]
    for column in (
        "forward_5d_max_gain_pct",
        "forward_5d_max_loss_pct",
        "forward_5d_label_end_date",
        "terminal_close_return_5d_pct",
        "first_gain_2pct_day",
        "first_gain_5pct_day",
        "first_loss_5pct_day",
        "gain_loss_order_5d",
        "weak_5d",
        "strong_5d",
        "deep_loss_5d",
        "bad_5d",
        "loss_first_5d",
        "strong_first_5d",
    ):
        output[column] = [item[column] for item in path_outcomes]

    gain5 = _numeric(output["forward_5d_max_gain_pct"])
    gain10 = _numeric(output["forward_10d_max_gain_pct"])
    gain20 = _numeric(output["forward_20d_max_gain_pct"])
    complete5 = pd.Series(
        [bool(item["_full_path_available"]) for item in path_outcomes],
        index=output.index,
    )
    complete10 = complete5 & gain10.notna()
    complete20 = complete5 & gain20.notna()
    output["late_strong_10d"] = _nullable_flag(
        output.index,
        complete10,
        lambda selected: (gain5.loc[selected] < cfg.strong_5d_min_gain_pct)
        & (gain10.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    output["late_strong_20d"] = _nullable_flag(
        output.index,
        complete20,
        lambda selected: (gain5.loc[selected] < cfg.strong_5d_min_gain_pct)
        & (gain20.loc[selected] >= cfg.strong_5d_min_gain_pct),
    )
    output["analysis_split"] = _analysis_split(
        output["signal_date"], output["forward_5d_label_end_date"], cfg
    )
    output["include_weak_filter"] = True
    output["include_loss_first_filter"] = True
    output["include_final"] = True
    output["weak_matched_rule_ids"] = pd.NA
    output["loss_first_matched_rule_ids"] = pd.NA
    output["matched_rule_ids"] = pd.NA
    output["filter_decision"] = "include"
    output["exclusion_reason"] = pd.NA

    for column in SIGNAL_COLUMNS:
        if column not in output:
            raise AssertionError(f"calculation did not produce {column}")
    return output.loc[:, SIGNAL_COLUMNS]


def _empty_early_cut_row(signal: pd.Series, landmark_day: int) -> dict[str, Any]:
    row = {column: pd.NA for column in EARLY_CUT_COLUMNS}
    row.update(
        {
            "signal_date": pd.Timestamp(signal["signal_date"]),
            "landmark_day": int(landmark_day),
            "symbol": signal["symbol"],
            "exchange": signal["exchange"],
            "cik": signal["cik"],
            "price_continuity_segment": signal["price_continuity_segment"],
            "currency": signal["currency"],
            "landmark_observed": False,
            "same_continuity_segment": False,
            "eligible_at_landmark": False,
            "full_outcome_available": False,
            "analysis_split": signal["analysis_split"],
            "include_stagnation_filter": False,
            "include_loss_filter": False,
            "include_final": False,
            "stagnation_matched_rule_ids": pd.NA,
            "loss_matched_rule_ids": pd.NA,
            "matched_rule_ids": pd.NA,
            "cut_decision": "not_evaluable",
            "cut_reason": pd.NA,
        }
    )
    return row


def _safe_scalar_ratio(
    numerator: Any,
    denominator: Any,
    *,
    percentage: bool = False,
) -> float:
    try:
        left = float(numerator)
        right = float(denominator)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(left) or not np.isfinite(right) or right <= 0:
        return np.nan
    value = left / right - (1.0 if percentage else 0.0)
    return float(value * 100.0 if percentage else value)


def _early_cut_landmark_row(
    indexed: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    segment: pd.Series,
    true_range: pd.Series,
    landmark_atr14: pd.Series,
    signal: pd.Series,
    signal_position: int,
    landmark_day: int,
    cfg: Config,
) -> dict[str, Any]:
    """Build one causal D1-D3 decision row and its censored outcome."""

    row = _empty_early_cut_row(signal, landmark_day)
    landmark_position = signal_position + landmark_day
    effective_position = landmark_position + 1
    horizon_position = signal_position + 5
    landmark_date = _date_at(trading_dates, landmark_position)
    effective_date = _date_at(trading_dates, effective_position)
    horizon_date = _date_at(trading_dates, horizon_position)
    row.update(
        {
            "landmark_date": landmark_date,
            "effective_session_date": effective_date,
            "horizon_end_date": horizon_date,
            # Early-cut folds are anchored at the close where their features
            # become known, not at the original entry signal.  The unchanged
            # D+5 horizon still determines whether a pre-holdout label belongs
            # wholly inside its split.
            "analysis_split": _analysis_split(
                pd.Series([landmark_date]),
                pd.Series([horizon_date]),
                cfg,
            ).iloc[0],
        }
    )

    if signal_position < 0 or signal_position >= len(indexed):
        return row
    segment_id = segment.iloc[signal_position]
    signal_close = indexed["adjusted_close"].iloc[signal_position]
    if _finite_positive(signal_close):
        row["signal_adjusted_close"] = float(signal_close)

    landmark_observed = False
    if 0 <= landmark_position < len(indexed):
        landmark_symbol = indexed["symbol"].iloc[landmark_position]
        signal_symbol = signal["symbol"]
        landmark_observed = bool(
            pd.notna(landmark_symbol)
            and pd.notna(signal_symbol)
            and landmark_symbol == signal_symbol
        )
    row["landmark_observed"] = landmark_observed
    # Eligibility is decided at the landmark close.  The following session is
    # only when a possible cut becomes effective; requiring that observation
    # here would leak future survivorship into the decision population.
    same_through_landmark = _observed_same_segment(
        indexed,
        segment,
        signal_position,
        landmark_position,
        segment_id,
    )
    row["same_continuity_segment"] = same_through_landmark
    complete_through_landmark = _complete_price_path(
        indexed,
        segment,
        signal_position,
        landmark_position,
        segment_id,
    )
    full_outcome_available = _complete_price_path(
        indexed,
        segment,
        signal_position,
        horizon_position,
        segment_id,
    )
    row["full_outcome_available"] = full_outcome_available

    if complete_through_landmark:
        signal_close = float(signal_close)
        landmark = indexed.iloc[landmark_position]
        landmark_close = float(landmark["adjusted_close"])
        landmark_high = float(landmark["adjusted_high"])
        landmark_low = float(landmark["adjusted_low"])
        path = indexed.iloc[signal_position + 1 : landmark_position + 1]
        path_highs = _numeric(path["adjusted_high"]).to_numpy(dtype=float)
        path_lows = _numeric(path["adjusted_low"]).to_numpy(dtype=float)
        high_returns = (path_highs / signal_close - 1.0) * 100.0
        low_returns = (path_lows / signal_close - 1.0) * 100.0
        max_gain = max(0.0, float(high_returns.max()))
        max_loss = min(0.0, float(low_returns.min()))
        first_gain_2 = _first_hit_day(high_returns >= cfg.weak_5d_max_gain_pct, 1)
        first_gain_5 = _first_hit_day(high_returns >= cfg.strong_5d_min_gain_pct, 1)
        first_loss_5 = _first_hit_day(low_returns <= cfg.deep_loss_5d_max_loss_pct, 1)
        hit_gain_2 = first_gain_2 is not None
        hit_gain_5 = first_gain_5 is not None
        hit_loss_5 = first_loss_5 is not None

        prior_close = float(indexed["adjusted_close"].iloc[landmark_position - 1])
        current_true_range = true_range.iloc[landmark_position]
        current_atr14 = landmark_atr14.iloc[landmark_position]
        landmark_volume = landmark["adjusted_volume"]
        landmark_notional = landmark["daily_traded_notional_usd"]
        signal_volume = indexed["adjusted_volume"].iloc[signal_position]
        signal_notional = indexed["daily_traded_notional_usd"].iloc[signal_position]
        signal_rs = indexed["rs_rating"].iloc[signal_position]
        landmark_rs = landmark["rs_rating"]
        criteria_values = pd.Series(
            [landmark[column] for column in CRITERION_COLUMNS],
            dtype="boolean",
        )
        criteria_count: Any = (
            int(criteria_values.sum()) if criteria_values.notna().all() else pd.NA
        )
        trend_value = landmark["trend_template_pass"]

        row.update(
            {
                "landmark_adjusted_close": landmark_close,
                "landmark_adjusted_high": landmark_high,
                "landmark_adjusted_low": landmark_low,
                "landmark_adjusted_volume": landmark_volume,
                "landmark_daily_traded_notional_usd": landmark_notional,
                "landmark_daily_price_change_pct": landmark["daily_price_change_pct"],
                "landmark_volume_vs_sma21_prior_ratio": landmark[
                    "adjusted_volume_vs_sma21_prior_ratio"
                ],
                "landmark_volume_vs_sma50_prior_ratio": landmark[
                    "adjusted_volume_vs_sma50_prior_ratio"
                ],
                "landmark_notional_vs_sma21_prior_ratio": landmark[
                    "daily_traded_notional_vs_sma21_prior_ratio"
                ],
                "landmark_notional_vs_sma50_prior_ratio": landmark[
                    "daily_traded_notional_vs_sma50_prior_ratio"
                ],
                "landmark_rs_rating": landmark_rs,
                "landmark_criteria_pass_count": criteria_count,
                "landmark_trend_template_pass": (
                    pd.NA if pd.isna(trend_value) else bool(trend_value)
                ),
                "close_return_from_signal_pct": (
                    (landmark_close / signal_close - 1.0) * 100.0
                ),
                "max_gain_to_landmark_pct": max_gain,
                "max_loss_to_landmark_pct": max_loss,
                "drawdown_from_post_signal_high_pct": (
                    (landmark_close / float(path_highs.max()) - 1.0) * 100.0
                ),
                "rebound_from_post_signal_low_pct": (
                    (landmark_close / float(path_lows.min()) - 1.0) * 100.0
                ),
                "landmark_true_range_pct": (
                    float(current_true_range) / prior_close * 100.0
                    if pd.notna(current_true_range) and prior_close > 0
                    else np.nan
                ),
                "landmark_close_location_value": (
                    (landmark_close - landmark_low) / (landmark_high - landmark_low)
                    if landmark_high > landmark_low
                    else np.nan
                ),
                "volume_vs_signal_ratio": _safe_scalar_ratio(
                    landmark_volume, signal_volume
                ),
                "notional_vs_signal_ratio": _safe_scalar_ratio(
                    landmark_notional, signal_notional
                ),
                "rs_rating_change_from_signal": (
                    float(landmark_rs) - float(signal_rs)
                    if pd.notna(landmark_rs)
                    and pd.notna(signal_rs)
                    and np.isfinite(float(landmark_rs))
                    and np.isfinite(float(signal_rs))
                    else np.nan
                ),
                "landmark_distance_to_ma50_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma50"], percentage=True
                ),
                "landmark_distance_to_ma150_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma150"], percentage=True
                ),
                "landmark_distance_to_ma200_pct": _safe_scalar_ratio(
                    landmark_close, landmark["ma200"], percentage=True
                ),
                "landmark_price_vs_52w_high_pct": _safe_scalar_ratio(
                    landmark_close, landmark["high_52w"], percentage=True
                ),
                "landmark_atr_14d_pct": (
                    float(current_atr14) / landmark_close * 100.0
                    if pd.notna(current_atr14) and np.isfinite(float(current_atr14))
                    else np.nan
                ),
                "mean_volume_since_signal_vs_prior21_ratio": (
                    _safe_scalar_ratio(
                        _numeric(path["adjusted_volume"]).mean(),
                        indexed["adjusted_volume_sma21_prior"].iloc[signal_position],
                    )
                ),
                "mean_notional_since_signal_vs_prior21_ratio": (
                    _safe_scalar_ratio(
                        _numeric(path["daily_traded_notional_usd"]).mean(),
                        indexed["daily_traded_notional_sma21_prior_usd"].iloc[
                            signal_position
                        ],
                    )
                ),
                "hit_gain_2pct_so_far": hit_gain_2,
                "hit_gain_5pct_so_far": hit_gain_5,
                "hit_loss_5pct_so_far": hit_loss_5,
                "first_gain_2pct_day_so_far": (
                    pd.NA if first_gain_2 is None else first_gain_2
                ),
                "first_gain_5pct_day_so_far": (
                    pd.NA if first_gain_5 is None else first_gain_5
                ),
                "first_loss_5pct_day_so_far": (
                    pd.NA if first_loss_5 is None else first_loss_5
                ),
            }
        )
        row["eligible_at_landmark"] = bool(
            same_through_landmark and not hit_gain_5 and not hit_loss_5
        )
        row["cut_decision"] = (
            "hold"
            if row["eligible_at_landmark"]
            else "not_eligible" if same_through_landmark else "not_evaluable"
        )
        if row["eligible_at_landmark"]:
            row["include_stagnation_filter"] = True
            row["include_loss_filter"] = True
            row["include_final"] = True

    if full_outcome_available:
        signal_close = float(indexed["adjusted_close"].iloc[signal_position])
        landmark_close = float(indexed["adjusted_close"].iloc[landmark_position])
        future = indexed.iloc[effective_position : horizon_position + 1]
        future_highs = _numeric(future["adjusted_high"]).to_numpy(dtype=float)
        future_lows = _numeric(future["adjusted_low"]).to_numpy(dtype=float)
        signal_high_returns = (future_highs / signal_close - 1.0) * 100.0
        signal_low_returns = (future_lows / signal_close - 1.0) * 100.0
        landmark_high_returns = (future_highs / landmark_close - 1.0) * 100.0
        landmark_low_returns = (future_lows / landmark_close - 1.0) * 100.0
        first_future_gain_2 = _first_hit_day(
            signal_high_returns >= cfg.weak_5d_max_gain_pct,
            landmark_day + 1,
        )
        first_future_gain_5 = _first_hit_day(
            signal_high_returns >= cfg.strong_5d_min_gain_pct,
            landmark_day + 1,
        )
        first_future_loss_5 = _first_hit_day(
            signal_low_returns <= cfg.deep_loss_5d_max_loss_pct,
            landmark_day + 1,
        )
        terminal_close = float(indexed["adjusted_close"].iloc[horizon_position])
        row.update(
            {
                "remaining_max_gain_from_signal_pct": max(
                    0.0, float(signal_high_returns.max())
                ),
                "remaining_max_loss_from_signal_pct": min(
                    0.0, float(signal_low_returns.min())
                ),
                "remaining_max_gain_from_landmark_pct": max(
                    0.0, float(landmark_high_returns.max())
                ),
                "remaining_max_loss_from_landmark_pct": min(
                    0.0, float(landmark_low_returns.min())
                ),
                "terminal_close_return_from_signal_pct": (
                    (terminal_close / signal_close - 1.0) * 100.0
                ),
                "terminal_close_return_from_landmark_pct": (
                    (terminal_close / landmark_close - 1.0) * 100.0
                ),
                "future_first_gain_2pct_day": (
                    pd.NA if first_future_gain_2 is None else first_future_gain_2
                ),
                "future_first_gain_5pct_day": (
                    pd.NA if first_future_gain_5 is None else first_future_gain_5
                ),
                "future_first_loss_5pct_day": (
                    pd.NA if first_future_loss_5 is None else first_future_loss_5
                ),
            }
        )

        if bool(row["eligible_at_landmark"]):
            continuation = _continuation_order(first_future_gain_5, first_future_loss_5)
            if continuation == "neither":
                full_path = indexed.iloc[signal_position + 1 : horizon_position + 1]
                full_high_returns = (
                    _numeric(full_path["adjusted_high"]).to_numpy(dtype=float)
                    / signal_close
                    - 1.0
                ) * 100.0
                continuation = (
                    "stagnant"
                    if float(full_high_returns.max()) < cfg.weak_5d_max_gain_pct
                    else "neutral"
                )
            row["continuation_outcome"] = continuation
            if continuation == "same_session_ambiguous":
                row["stagnant_to_day5"] = pd.NA
                row["loss_first_to_day5"] = pd.NA
                row["strong_first_to_day5"] = pd.NA
            else:
                row["stagnant_to_day5"] = continuation == "stagnant"
                row["loss_first_to_day5"] = continuation == "loss_first"
                row["strong_first_to_day5"] = continuation == "strong_first"

    return row


def calculate_early_cut_landmarks(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
    signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return exactly three censored D1-D3 observations per entry signal."""

    if signals is None:
        signals = calculate_identity_signals(rows, trading_dates, cfg)
    if signals.empty:
        return empty_early_cut_frame()
    if rows.empty:
        raise ValueError("non-empty signals require identity source rows")
    if not trading_dates.is_monotonic_increasing or not trading_dates.is_unique:
        raise ValueError("trading_dates must be sorted and unique")

    ordered = rows.sort_values("period_end_date", kind="stable").copy()
    ordered["period_end_date"] = pd.to_datetime(
        ordered["period_end_date"], errors="raise"
    ).dt.normalize()
    indexed = ordered.set_index("period_end_date").reindex(trading_dates)
    segment = _numeric(indexed["price_continuity_segment"])
    close = _numeric(indexed["adjusted_close"])
    high = _numeric(indexed["adjusted_high"])
    low = _numeric(indexed["adjusted_low"])
    previous_close = _lag_in_segment(close, segment, 1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=False)
    landmark_atr14 = _rolling_in_segment(true_range, segment, 14, "mean")
    positions_by_date = {
        pd.Timestamp(value): position for position, value in enumerate(trading_dates)
    }

    output_rows: list[dict[str, Any]] = []
    for _, signal in signals.iterrows():
        signal_date = pd.Timestamp(signal["signal_date"])
        if signal_date not in positions_by_date:
            raise ValueError("a signal date is absent from trading_dates")
        signal_position = positions_by_date[signal_date]
        for landmark_day in EARLY_CUT_LANDMARK_DAYS:
            output_rows.append(
                _early_cut_landmark_row(
                    indexed,
                    trading_dates,
                    segment,
                    true_range,
                    landmark_atr14,
                    signal,
                    signal_position,
                    landmark_day,
                    cfg,
                )
            )

    output = pd.DataFrame(output_rows, columns=EARLY_CUT_COLUMNS)
    expected = len(signals) * len(EARLY_CUT_LANDMARK_DAYS)
    if len(output) != expected:
        raise AssertionError(
            f"expected {expected} early-cut rows, calculated {len(output)}"
        )
    return output.loc[:, EARLY_CUT_COLUMNS]


def calculate_identity_results(
    rows: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> CalculationBatchResult:
    """Calculate entry signals and their paired landmark observations."""

    signals = calculate_identity_signals(rows, trading_dates, cfg)
    early_cut = calculate_early_cut_landmarks(rows, trading_dates, cfg, signals=signals)
    return CalculationBatchResult(signals=signals, early_cut=early_cut)


def calculate_signal_batch(
    source: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
    cfg: Config,
) -> CalculationBatchResult:
    missing = sorted(set(SOURCE_COLUMNS) - set(source.columns))
    if missing:
        raise ValueError("source data is missing columns: " + ", ".join(missing))
    if source.empty:
        return empty_calculation_batch()

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
        calculate_identity_results(group, trading_dates, cfg)
        for _, group in data.groupby(list(IDENTITY_COLUMNS), sort=True, observed=True)
    ]
    signal_frames = [result.signals for result in results if not result.signals.empty]
    early_cut_frames = [
        result.early_cut for result in results if not result.early_cut.empty
    ]
    if not signal_frames:
        return empty_calculation_batch()
    signals = pd.concat(signal_frames, ignore_index=True).loc[:, SIGNAL_COLUMNS]
    early_cut = pd.concat(early_cut_frames, ignore_index=True).loc[:, EARLY_CUT_COLUMNS]
    expected = len(signals) * len(EARLY_CUT_LANDMARK_DAYS)
    if len(early_cut) != expected:
        raise AssertionError(
            f"expected {expected} early-cut rows, calculated {len(early_cut)}"
        )
    return CalculationBatchResult(signals=signals, early_cut=early_cut)
