"""Deterministic fingerprints for independently runnable pipeline stages."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal

import numpy as np
import pandas as pd

from .config import Config


SCREEN_CONFIG_FIELDS = (
    "min_price", "min_dollar_volume", "rs_lookbacks", "rs_weights", "rs_min",
    "ma200_trend_days", "min_above_52w_low", "max_below_52w_high",
    "eps_yoy_min", "revenue_yoy_min", "fundamentals_min_pass",
    "margin_expansion_min", "acceleration_min", "quarterly_growth_streak_min",
    "quarterly_fundamental_stale_trading_days", "institutional_min_managers",
    "institutional_net_activity_min", "institutional_activity_lookback_sessions",
    "ibkr_industry_min_symbols", "ibkr_category_min_symbols",
    "ibkr_industry_rs_min", "ibkr_category_rs_min",
    "ibkr_stock_industry_rs_min", "ibkr_stock_category_rs_min",
    "ibkr_industry_breadth_min_symbols", "ibkr_industry_breadth_on_threshold",
    "ibkr_industry_breadth_off_threshold",
)

SETUP_CONFIG_FIELDS = (
    "swing_window", "base_min_days", "base_max_days", "contractions_min",
    "contractions_max", "final_depth_max", "base_depth_max",
    "pivot_below_base_high_max", "dryup_score_zero_ratio", "setup_valid_days",
    "prior_advance_min", "stop_max_pct",
)

SIM_CONFIG_FIELDS = (
    "simulation_mode", "initial_equity", "risk_pct", "stop_max_pct",
    "max_position_pct", "slippage_pct", "commission_pct", "partial_at_r",
    "partial_fraction", "breakeven_after_partial", "trail_ma_days",
    "failed_breakout_exit_enable", "failed_breakout_days",
    "failed_breakout_min_r",
    "eps_yoy_min", "revenue_yoy_min", "dryup_score_zero_ratio",
    "pivot_buffer_pct", "max_buy_zone_pct", "time_stop_sessions",
    "time_stop_min_r", "portfolio_max_open_positions",
    "portfolio_max_daily_orders",
    "portfolio_max_gross_exposure_pct", "min_slate_risk_utilization",
    "neutral_rank_salt",
    "exposure_levels",
    "exposure_winners_to_step_up", "exposure_losses_to_reset",
    "exposure_drawdown_reset_pct", "market_filter_enable",
    "regime_entry_filter_enable", "regime_allowed_labels",
)


def config_fingerprint(
    cfg: Config,
    fields: tuple[str, ...],
    *,
    model_version: str,
    upstream_fingerprint: str = "",
) -> str:
    values = asdict(cfg)
    payload = {
        "model_version": model_version,
        "upstream_fingerprint": upstream_fingerprint,
        "config": {name: values[name] for name in fields},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def frame_fingerprint(frame: pd.DataFrame, columns: tuple[str, ...]) -> str:
    """Hash stage rows independently of pandas and PostgreSQL scalar dtypes."""
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError("fingerprint frame missing columns: " + ",".join(missing))

    def canonical(value) -> str:
        if isinstance(value, (tuple, list, np.ndarray)):
            return "[" + ",".join(canonical(item) for item in value) + "]"
        if value is None or (not isinstance(value, str) and bool(pd.isna(value))):
            return ""
        if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
            return pd.Timestamp(value).isoformat()
        if isinstance(value, (bool, np.bool_)):
            return "true" if bool(value) else "false"
        if isinstance(value, (Decimal, float, np.floating)):
            number = Decimal(str(value)).normalize()
            return format(number, "f")
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    modulus = 1 << 256
    row_sum = 0
    row_square_sum = 0
    count = 0
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        encoded_row = json.dumps(
            [canonical(value) for value in row],
            separators=(",", ":"),
        ).encode("utf-8")
        value = int.from_bytes(hashlib.sha256(encoded_row).digest(), "big")
        row_sum = (row_sum + value) % modulus
        row_square_sum = (row_square_sum + value * value) % modulus
        count += 1
    payload = {
        "columns": list(columns),
        "count": count,
        "row_sum": f"{row_sum:064x}",
        "row_square_sum": f"{row_square_sum:064x}",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def matrix_fingerprint(
    dates: pd.DatetimeIndex,
    symbols: pd.Index,
    matrices: dict[str, pd.DataFrame],
) -> str:
    """Hash aligned numeric matrices without materializing a long frame."""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "dates": [pd.Timestamp(value).isoformat() for value in dates],
                "symbols": [str(value) for value in symbols],
                "fields": sorted(matrices),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for field in sorted(matrices):
        values = np.asarray(matrices[field], dtype="<f8")
        if values.shape != (len(dates), len(symbols)):
            raise ValueError(f"matrix {field} is not aligned to dates and symbols")
        digest.update(field.encode("utf-8"))
        for row_start in range(0, len(dates), 256):
            block = values[row_start : row_start + 256]
            finite = np.isfinite(block)
            canonical = np.where(finite, block, 0.0).astype("<f8", copy=False)
            digest.update(finite.astype(np.uint8, copy=False).tobytes(order="C"))
            digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def combine_fingerprints(**components: str) -> str:
    encoded = json.dumps(components, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
