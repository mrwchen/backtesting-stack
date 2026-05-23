"""Pullback/Bounce fundamental swing model.

Model idea:
  - LONG: strong fundamentals plus pullback from recent high.
  - SHORT: weak fundamentals plus bounce from recent low.
  - Direction is selected by the generic runner from the world-regime score.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, mean


@dataclass
class SignalConfig:
    """Parameters for pullback_bounce_fundamental_v1."""

    # Signal limits
    min_bars: int = 150

    # Entry filters — LONG
    long_min_pullback: float = 5.0
    long_max_pullback: float = 25.0
    long_ideal_pullback: float = 12.5
    long_max_rsi: float = 50.0

    # Entry filters — SHORT
    short_min_bounce: float = 3.0
    short_max_bounce: float = 20.0
    short_ideal_bounce: float = 8.5
    short_min_rsi: float = 35.0
    short_max_rsi: float = 65.0

    # Stop / TP
    long_sl_buffer: float = 0.003
    short_sl_buffer: float = 0.003
    long_tp1_pct: float = 0.06
    long_tp2_pct: float = 0.12
    short_tp1_pct: float = 0.06
    short_tp2_pct: float = 0.12
    long_max_hold_days: float = 12.0
    short_max_hold_days: float = 5.0
    tp1_close_ratio: float = 0.6

    # Mispricing score blending
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.3

    # Lookback windows
    price_lookback_bars: int = 320
    sl_lookback_bars: int = 5
    vol_short_bars: int = 5
    vol_long_bars: int = 25


def signal_config_from_env() -> SignalConfig:
    """Build a model config from environment variables."""
    defaults = SignalConfig()
    return SignalConfig(
        min_bars=env_int("MIN_BARS", defaults.min_bars),
        long_min_pullback=env_float("LONG_MIN_PULLBACK", defaults.long_min_pullback),
        long_max_pullback=env_float("LONG_MAX_PULLBACK", defaults.long_max_pullback),
        long_ideal_pullback=env_float("LONG_IDEAL_PULLBACK", defaults.long_ideal_pullback),
        long_max_rsi=env_float("LONG_MAX_RSI", defaults.long_max_rsi),
        short_min_bounce=env_float("SHORT_MIN_BOUNCE", defaults.short_min_bounce),
        short_max_bounce=env_float("SHORT_MAX_BOUNCE", defaults.short_max_bounce),
        short_ideal_bounce=env_float("SHORT_IDEAL_BOUNCE", defaults.short_ideal_bounce),
        short_min_rsi=env_float("SHORT_MIN_RSI", defaults.short_min_rsi),
        short_max_rsi=env_float("SHORT_MAX_RSI", defaults.short_max_rsi),
        long_sl_buffer=env_float("LONG_SL_BUFFER", defaults.long_sl_buffer),
        short_sl_buffer=env_float("SHORT_SL_BUFFER", defaults.short_sl_buffer),
        long_tp1_pct=env_float("LONG_TP1_PCT", defaults.long_tp1_pct),
        long_tp2_pct=env_float("LONG_TP2_PCT", defaults.long_tp2_pct),
        short_tp1_pct=env_float("SHORT_TP1_PCT", defaults.short_tp1_pct),
        short_tp2_pct=env_float("SHORT_TP2_PCT", defaults.short_tp2_pct),
        long_max_hold_days=env_float("LONG_MAX_HOLD_DAYS", defaults.long_max_hold_days),
        short_max_hold_days=env_float("SHORT_MAX_HOLD_DAYS", defaults.short_max_hold_days),
        tp1_close_ratio=env_float("TP1_CLOSE_RATIO", defaults.tp1_close_ratio),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", defaults.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", defaults.mispricing_weight),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", defaults.price_lookback_bars),
        sl_lookback_bars=env_int("SL_LOOKBACK_BARS", defaults.sl_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", defaults.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", defaults.vol_long_bars),
    )


def required_bar_lookback(cfg: SignalConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.sl_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(
    base_cfg: SignalConfig,
    parse_grid_vals,
    parse_hold_grid_vals,
):
    """Yield model-specific grid-search configs for this strategy."""
    long_tp1_vals = parse_grid_vals("GRID_LONG_TP1_PCT", base_cfg.long_tp1_pct)
    long_tp2_vals = parse_grid_vals("GRID_LONG_TP2_PCT", base_cfg.long_tp2_pct)
    short_tp1_vals = parse_grid_vals("GRID_SHORT_TP1_PCT", base_cfg.short_tp1_pct)
    short_tp2_vals = parse_grid_vals("GRID_SHORT_TP2_PCT", base_cfg.short_tp2_pct)
    long_max_hold_days_vals = parse_hold_grid_vals("GRID_LONG_MAX_HOLD_DAYS", base_cfg.long_max_hold_days)
    short_max_hold_days_vals = parse_hold_grid_vals("GRID_SHORT_MAX_HOLD_DAYS", base_cfg.short_max_hold_days)
    tp1_ratio_vals = parse_grid_vals("GRID_TP1_CLOSE_RATIO", base_cfg.tp1_close_ratio)

    for ltp1, ltp2, stp1, stp2, lmhd, smhd, tcr in product(
        long_tp1_vals,
        long_tp2_vals,
        short_tp1_vals,
        short_tp2_vals,
        long_max_hold_days_vals,
        short_max_hold_days_vals,
        tp1_ratio_vals,
    ):
        if ltp2 <= ltp1 or stp2 <= stp1:
            continue

        cfg = dataclasses.replace(
            base_cfg,
            long_tp1_pct=ltp1,
            long_tp2_pct=ltp2,
            short_tp1_pct=stp1,
            short_tp2_pct=stp2,
            long_max_hold_days=lmhd,
            short_max_hold_days=smhd,
            tp1_close_ratio=tcr,
        )
        notes = (
            f"grid model=pullback_bounce_fundamental_v1 "
            f"ltp1={ltp1:.3f} ltp2={ltp2:.3f} "
            f"stp1={stp1:.3f} stp2={stp2:.3f} "
            f"lmhd={lmhd:.1f} smhd={smhd:.1f} tcr={tcr:.2f}"
        )
        yield {
            "config": cfg,
            "notes": notes,
            "summary": {
                "long_tp1_pct": ltp1,
                "long_tp2_pct": ltp2,
                "short_tp1_pct": stp1,
                "short_tp2_pct": stp2,
                "long_max_hold_days": lmhd,
                "short_max_hold_days": smhd,
                "tp1_close_ratio": tcr,
            },
        }


def compute_long_signal(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: SignalConfig,
) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: SignalConfig,
) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]

    entry_price = closes[-1]
    lookback_highs = highs[-cfg.price_lookback_bars:]
    high_20d = max(lookback_highs) if lookback_highs else entry_price

    if high_20d <= 0 or entry_price <= 0:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="invalid_price",
            reason_text="Entry price or lookback high is not positive.",
            entry_price=entry_price,
        )

    pullback_pct = (high_20d - entry_price) / high_20d * 100.0
    if pullback_pct < cfg.long_min_pullback:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="pullback_below_min",
            reason_text=f"Pullback {pullback_pct:.2f}% is below minimum {cfg.long_min_pullback:.2f}%.",
            entry_price=entry_price,
            pullback_pct=round(pullback_pct, 2),
        )
    if pullback_pct > cfg.long_max_pullback:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="pullback_above_max",
            reason_text=f"Pullback {pullback_pct:.2f}% is above maximum {cfg.long_max_pullback:.2f}%.",
            entry_price=entry_price,
            pullback_pct=round(pullback_pct, 2),
        )

    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="rsi_above_max",
            reason_text=f"RSI {rsi:.2f} is above maximum {cfg.long_max_rsi:.2f}.",
            entry_price=entry_price,
            pullback_pct=round(pullback_pct, 2),
            rsi_1h=round(rsi, 2),
        )

    vol_short = mean(volumes[-cfg.vol_short_bars:])
    vol_long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else vol_short
    vol_ratio = (vol_short / vol_long) if vol_long > 0 else 1.0

    pullback_score = clamp(1.0 - abs((pullback_pct - cfg.long_ideal_pullback) / cfg.long_ideal_pullback), 0.0, 1.0)
    rsi_score = clamp((40.0 - rsi) / 20.0, 0.0, 1.0)
    vol_score = clamp((1.0 - vol_ratio) / 0.5, 0.0, 1.0)
    entry_score = pullback_score * 0.5 + rsi_score * 0.35 + vol_score * 0.15

    if cfg.use_mispricing_score and fundamental.mispricing_score is not None:
        fund_raw = (
            (fundamental.composite_score / 100.0) * (1.0 - cfg.mispricing_weight)
            + (fundamental.mispricing_score / 100.0) * cfg.mispricing_weight
        )
    else:
        fund_raw = fundamental.composite_score / 100.0

    combined = (fund_raw * 0.375 + entry_score * 0.625) * 10.0

    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    tp1 = entry_price * (1.0 + cfg.long_tp1_pct)
    tp2 = entry_price * (1.0 + cfg.long_tp2_pct)

    reason = f"Pullback {pullback_pct:.1f}% | RSI {rsi:.0f} | Vol {vol_ratio:.2f}x"

    signal = Signal(
        symbol=fundamental.symbol,
        direction="LONG",
        fundamental_score=fundamental.composite_score,
        entry_score=round(entry_score, 4),
        combined_score=round(combined, 4),
        entry_price=entry_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        pullback_pct=round(pullback_pct, 2),
        rsi_1h=round(rsi, 2),
        volume_ratio=round(vol_ratio, 3),
        entry_reason=reason,
        valuation_label=fundamental.valuation_label,
        sector=fundamental.sector,
        industry=fundamental.industry,
    )
    return SignalEvaluation(
        signal=signal,
        decision="signal",
        reason_code="signal_passed",
        reason_text=reason,
        entry_price=entry_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        pullback_pct=round(pullback_pct, 2),
        rsi_1h=round(rsi, 2),
        volume_ratio=round(vol_ratio, 3),
        entry_score=round(entry_score, 4),
        combined_score=round(combined, 4),
    )


def compute_short_signal(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: SignalConfig,
) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(
    bars: list[Bar],
    fundamental: FundamentalRow,
    now: datetime,
    cfg: SignalConfig,
) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]

    entry_price = closes[-1]
    lookback_lows = lows[-cfg.price_lookback_bars:]
    low_20d = min(lookback_lows) if lookback_lows else entry_price

    if low_20d <= 0 or entry_price <= 0:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="invalid_price",
            reason_text="Entry price or lookback low is not positive.",
            entry_price=entry_price,
        )

    bounce_pct = (entry_price - low_20d) / low_20d * 100.0
    if bounce_pct < cfg.short_min_bounce:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="bounce_below_min",
            reason_text=f"Bounce {bounce_pct:.2f}% is below minimum {cfg.short_min_bounce:.2f}%.",
            entry_price=entry_price,
            pullback_pct=round(bounce_pct, 2),
        )
    if bounce_pct > cfg.short_max_bounce:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="bounce_above_max",
            reason_text=f"Bounce {bounce_pct:.2f}% is above maximum {cfg.short_max_bounce:.2f}%.",
            entry_price=entry_price,
            pullback_pct=round(bounce_pct, 2),
        )

    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="rsi_below_min",
            reason_text=f"RSI {rsi:.2f} is below minimum {cfg.short_min_rsi:.2f}.",
            entry_price=entry_price,
            pullback_pct=round(bounce_pct, 2),
            rsi_1h=round(rsi, 2),
        )
    if rsi > cfg.short_max_rsi:
        return SignalEvaluation(
            signal=None,
            decision="rejected",
            reason_code="rsi_above_max",
            reason_text=f"RSI {rsi:.2f} is above maximum {cfg.short_max_rsi:.2f}.",
            entry_price=entry_price,
            pullback_pct=round(bounce_pct, 2),
            rsi_1h=round(rsi, 2),
        )

    vol_short = mean(volumes[-cfg.vol_short_bars:])
    vol_long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else vol_short
    vol_ratio = (vol_short / vol_long) if vol_long > 0 else 1.0

    bounce_score = clamp(1.0 - abs((bounce_pct - cfg.short_ideal_bounce) / cfg.short_ideal_bounce), 0.0, 1.0)
    rsi_ideal = (cfg.short_min_rsi + cfg.short_max_rsi) / 2.0
    rsi_score = clamp(1.0 - abs((rsi - rsi_ideal) / 15.0), 0.0, 1.0)
    vol_score = clamp((1.0 - vol_ratio) / 0.5, 0.0, 1.0)
    entry_score = bounce_score * 0.5 + rsi_score * 0.35 + vol_score * 0.15

    if cfg.use_mispricing_score and fundamental.mispricing_score is not None:
        inv_fund = (
            ((100.0 - fundamental.composite_score) / 100.0) * (1.0 - cfg.mispricing_weight)
            + ((100.0 - fundamental.mispricing_score) / 100.0) * cfg.mispricing_weight
        )
    else:
        inv_fund = (100.0 - fundamental.composite_score) / 100.0

    combined = (inv_fund * 0.375 + entry_score * 0.625) * 10.0

    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    tp1 = entry_price * (1.0 - cfg.short_tp1_pct)
    tp2 = entry_price * (1.0 - cfg.short_tp2_pct)

    reason = f"Bounce {bounce_pct:.1f}% | RSI {rsi:.0f} | Vol {vol_ratio:.2f}x"

    signal = Signal(
        symbol=fundamental.symbol,
        direction="SHORT",
        fundamental_score=fundamental.composite_score,
        entry_score=round(entry_score, 4),
        combined_score=round(combined, 4),
        entry_price=entry_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        pullback_pct=round(bounce_pct, 2),
        rsi_1h=round(rsi, 2),
        volume_ratio=round(vol_ratio, 3),
        entry_reason=reason,
        valuation_label=fundamental.valuation_label,
        sector=fundamental.sector,
        industry=fundamental.industry,
    )
    return SignalEvaluation(
        signal=signal,
        decision="signal",
        reason_code="signal_passed",
        reason_text=reason,
        entry_price=entry_price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        pullback_pct=round(bounce_pct, 2),
        rsi_1h=round(rsi, 2),
        volume_ratio=round(vol_ratio, 3),
        entry_score=round(entry_score, 4),
        combined_score=round(combined, 4),
    )
