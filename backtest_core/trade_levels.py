"""Common stop-loss and take-profit policy for accepted model entries."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from backtest_shared import Bar, Signal

from .config import (
    COMMON_MAX_STOP_PCT,
    COMMON_MIN_STOP_PCT,
    COMMON_STOP_ATR_LOOKBACK_BARS,
    COMMON_STOP_ATR_MULT,
    COMMON_STOP_BUFFER,
    COMMON_STOP_LOOKBACK_BARS,
    COMMON_STOP_LOSS_ENABLED,
)

MODEL_STOP_LOSS_PARAM_NAMES = (
    "long_stop_vol_mult",
    "short_stop_vol_mult",
    "min_stop_pct",
    "max_stop_pct",
)


@dataclass(frozen=True)
class TradeLevelResult:
    accepted: bool
    source: str
    reason_code: str
    reason_text: str


def model_uses_own_stop_loss(cfg: Any) -> bool:
    """Return true when a strategy config carries an explicit SL concept."""
    explicit_flag = getattr(cfg, "model_stop_loss_enabled", None)
    if explicit_flag is not None:
        return bool(explicit_flag)
    return all(hasattr(cfg, name) for name in MODEL_STOP_LOSS_PARAM_NAMES)


def common_stop_required_lookback() -> int:
    return max(COMMON_STOP_LOOKBACK_BARS, COMMON_STOP_ATR_LOOKBACK_BARS + 1)


def apply_trade_levels(
    signal: Signal,
    bars: list[Bar],
    cfg: Any,
    entry_ts: datetime,
    entry_open: float,
) -> TradeLevelResult:
    signal.entry_ts = entry_ts
    signal.entry_price = float(entry_open)
    _apply_common_take_profits(signal, cfg)

    if model_uses_own_stop_loss(cfg):
        return _validate_model_stop_loss(signal)

    if not COMMON_STOP_LOSS_ENABLED:
        return _validate_model_stop_loss(signal)

    stop_loss = _common_stop_loss(signal.direction, signal.entry_price, bars)
    if stop_loss is None:
        return TradeLevelResult(
            False,
            "common",
            "common_stop_loss_unavailable",
            "Common stop-loss policy could not calculate a valid stop from recent bars.",
        )

    signal.stop_loss = stop_loss
    return TradeLevelResult(True, "common", "common_stop_loss_applied", "Common stop-loss policy applied.")


def _apply_common_take_profits(signal: Signal, cfg: Any) -> None:
    if signal.direction == "LONG":
        signal.take_profit_1 = signal.entry_price * (1.0 + cfg.long_tp1_pct)
        signal.take_profit_2 = signal.entry_price * (1.0 + cfg.long_tp2_pct)
    else:
        signal.take_profit_1 = signal.entry_price * (1.0 - cfg.short_tp1_pct)
        signal.take_profit_2 = signal.entry_price * (1.0 - cfg.short_tp2_pct)


def _validate_model_stop_loss(signal: Signal) -> TradeLevelResult:
    try:
        stop_loss = float(signal.stop_loss)
    except (TypeError, ValueError):
        return TradeLevelResult(
            False,
            "model",
            "model_stop_loss_missing",
            "Model-owned stop-loss policy did not provide a numeric stop.",
        )

    if signal.direction == "LONG" and stop_loss >= signal.entry_price:
        return TradeLevelResult(
            False,
            "model",
            "model_stop_loss_invalid",
            "Model-owned long stop-loss was not below entry.",
        )
    if signal.direction == "SHORT" and stop_loss <= signal.entry_price:
        return TradeLevelResult(
            False,
            "model",
            "model_stop_loss_invalid",
            "Model-owned short stop-loss was not above entry.",
        )
    return TradeLevelResult(True, "model", "model_stop_loss_applied", "Model-owned stop-loss policy applied.")


def _common_stop_loss(direction: str, entry_price: float, bars: list[Bar]) -> float | None:
    if entry_price <= 0.0 or not bars:
        return None

    structure = _structure_stop(direction, entry_price, bars)
    atr_stop = _atr_stop(direction, entry_price, bars)
    min_stop = _min_stop(direction, entry_price)

    candidates = [value for value in (structure, atr_stop, min_stop) if value is not None]
    if not candidates:
        return None

    if direction == "LONG":
        stop = min(candidates)
        if COMMON_MAX_STOP_PCT > 0.0:
            stop = max(stop, entry_price * (1.0 - COMMON_MAX_STOP_PCT / 100.0))
        return stop if 0.0 < stop < entry_price else None

    stop = max(candidates)
    if COMMON_MAX_STOP_PCT > 0.0:
        stop = min(stop, entry_price * (1.0 + COMMON_MAX_STOP_PCT / 100.0))
    return stop if stop > entry_price else None


def _structure_stop(direction: str, entry_price: float, bars: list[Bar]) -> float | None:
    recent = bars[-COMMON_STOP_LOOKBACK_BARS:]
    if not recent:
        return None
    if direction == "LONG":
        raw = min(float(bar.low) for bar in recent)
        stop = raw * (1.0 - COMMON_STOP_BUFFER)
        return stop if 0.0 < stop < entry_price else None
    raw = max(float(bar.high) for bar in recent)
    stop = raw * (1.0 + COMMON_STOP_BUFFER)
    return stop if stop > entry_price else None


def _atr_stop(direction: str, entry_price: float, bars: list[Bar]) -> float | None:
    atr = _average_true_range(bars, COMMON_STOP_ATR_LOOKBACK_BARS)
    if atr is None or atr <= 0.0 or COMMON_STOP_ATR_MULT <= 0.0:
        return None
    if direction == "LONG":
        stop = entry_price - atr * COMMON_STOP_ATR_MULT
        return stop if 0.0 < stop < entry_price else None
    stop = entry_price + atr * COMMON_STOP_ATR_MULT
    return stop if stop > entry_price else None


def _min_stop(direction: str, entry_price: float) -> float | None:
    if COMMON_MIN_STOP_PCT <= 0.0:
        return None
    if direction == "LONG":
        return entry_price * (1.0 - COMMON_MIN_STOP_PCT / 100.0)
    return entry_price * (1.0 + COMMON_MIN_STOP_PCT / 100.0)


def _average_true_range(bars: list[Bar], lookback: int) -> float | None:
    if len(bars) < 2:
        return None
    recent = bars[-max(1, lookback):]
    values: list[float] = []
    previous_close = float(bars[-len(recent) - 1].close) if len(bars) > len(recent) else float(recent[0].close)
    for bar in recent:
        high = float(bar.high)
        low = float(bar.low)
        true_range = max(high - low, abs(high - previous_close), abs(low - previous_close))
        values.append(true_range)
        previous_close = float(bar.close)
    return sum(values) / len(values) if values else None
