"""Central execution levels for accepted model intents."""

from dataclasses import dataclass
from datetime import datetime

from backtest_shared import Bar, FundamentalRow, TradeIntent, TradePlan

from .config import (
    COMMON_MAX_STOP_PCT,
    COMMON_MIN_STOP_PCT,
    COMMON_STOP_ATR_LOOKBACK_BARS,
    COMMON_STOP_ATR_MULT,
    COMMON_STOP_BUFFER,
    COMMON_STOP_LOOKBACK_BARS,
    COMMON_STOP_LOSS_ENABLED,
    EXECUTION_LONG_MAX_HOLD_DAYS,
    EXECUTION_LONG_TAKE_PROFIT_PCT,
    EXECUTION_LONG_TRAILING_ACTIVATION_PCT,
    EXECUTION_LONG_TRAILING_DISTANCE_PCT,
    EXECUTION_SHORT_MAX_HOLD_DAYS,
    EXECUTION_SHORT_TAKE_PROFIT_PCT,
    EXECUTION_SHORT_TRAILING_ACTIVATION_PCT,
    EXECUTION_SHORT_TRAILING_DISTANCE_PCT,
    TAKE_PROFIT_MODE,
)


@dataclass(frozen=True)
class TradePlanResult:
    accepted: bool
    source: str
    reason_code: str
    reason_text: str
    plan: TradePlan | None = None


def common_stop_required_lookback() -> int:
    return max(COMMON_STOP_LOOKBACK_BARS, COMMON_STOP_ATR_LOOKBACK_BARS + 1)


def build_trade_plan(
    intent: TradeIntent,
    fundamental: FundamentalRow,
    bars: list[Bar],
    entry_ts: datetime,
    entry_open: float,
) -> TradePlanResult:
    entry_price = float(entry_open)
    if not COMMON_STOP_LOSS_ENABLED:
        return TradePlanResult(
            False,
            "execution",
            "common_stop_loss_disabled",
            "Central stop-loss policy is disabled; execution risk engine cannot size this trade.",
        )

    stop_loss = _common_stop_loss(intent.direction, entry_price, bars)
    if stop_loss is None:
        return TradePlanResult(
            False,
            "execution",
            "common_stop_loss_unavailable",
            "Central stop-loss policy could not calculate a valid stop from recent bars.",
        )

    take_profit = None
    trailing_activation_price = None
    trailing_distance_pct = None
    if TAKE_PROFIT_MODE == "fixed":
        if intent.direction == "LONG":
            take_profit = entry_price * (1.0 + EXECUTION_LONG_TAKE_PROFIT_PCT)
        else:
            take_profit = entry_price * (1.0 - EXECUTION_SHORT_TAKE_PROFIT_PCT)
    elif TAKE_PROFIT_MODE == "trailing":
        if intent.direction == "LONG":
            trailing_activation_price = entry_price * (1.0 + EXECUTION_LONG_TRAILING_ACTIVATION_PCT)
            trailing_distance_pct = EXECUTION_LONG_TRAILING_DISTANCE_PCT
        else:
            trailing_activation_price = entry_price * (1.0 - EXECUTION_SHORT_TRAILING_ACTIVATION_PCT)
            trailing_distance_pct = EXECUTION_SHORT_TRAILING_DISTANCE_PCT
    else:
        raise ValueError(f"Unknown take-profit mode: {TAKE_PROFIT_MODE!r}")

    plan = TradePlan(
        symbol=fundamental.symbol,
        direction=intent.direction,
        fundamental_score=fundamental.composite_score,
        intent_score=intent.score,
        intent_reason=intent.reason,
        entry_price=entry_price,
        stop_loss=stop_loss,
        take_profit_mode=TAKE_PROFIT_MODE,
        take_profit=take_profit,
        trailing_activation_price=trailing_activation_price,
        trailing_distance_pct=trailing_distance_pct,
        valuation_label=fundamental.valuation_label,
        sector=fundamental.sector,
        industry=fundamental.industry,
        entry_ts=entry_ts,
        exchange=fundamental.exchange,
        cik=fundamental.cik,
    )
    return TradePlanResult(
        True,
        "execution",
        "execution_levels_applied",
        "Central execution risk engine applied entry, stop-loss, and take-profit policy.",
        plan,
    )


def execution_max_hold_days(direction: str) -> float:
    if direction == "LONG":
        return EXECUTION_LONG_MAX_HOLD_DAYS
    if direction == "SHORT":
        return EXECUTION_SHORT_MAX_HOLD_DAYS
    raise ValueError(f"Unknown direction: {direction!r}")


def validate_intent_for_candidate(intent: TradeIntent, fundamental: FundamentalRow, direction: str) -> TradePlanResult:
    if intent.direction != direction:
        return TradePlanResult(
            False,
            "intent",
            "intent_direction_mismatch",
            f"Model returned {intent.direction} while runner was evaluating {direction}.",
        )
    if intent.symbol != fundamental.symbol.strip().upper():
        return TradePlanResult(
            False,
            "intent",
            "intent_symbol_mismatch",
            f"Model returned symbol {intent.symbol} for candidate {fundamental.symbol}.",
        )
    return TradePlanResult(True, "intent", "intent_valid", "Model intent matched the evaluated candidate and direction.")


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
