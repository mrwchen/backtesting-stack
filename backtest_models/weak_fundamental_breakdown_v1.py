"""Weak fundamental breakdown swing model.

Model idea:
  - LONG side is conservative and only buys quality reclaim breakouts.
  - SHORT side is the primary edge: weak fundamentals plus price breakdown.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, mean


@dataclass
class SignalConfig:
    min_bars: int = 180
    long_min_pullback: float = 0.0
    long_max_pullback: float = 10.0
    long_ideal_pullback: float = 3.0
    long_max_rsi: float = 68.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 5.0
    short_ideal_bounce: float = 1.0
    short_min_rsi: float = 20.0
    short_max_rsi: float = 55.0
    long_sl_buffer: float = 0.006
    short_sl_buffer: float = 0.007
    long_tp1_pct: float = 0.045
    long_tp2_pct: float = 0.09
    short_tp1_pct: float = 0.06
    short_tp2_pct: float = 0.13
    long_max_hold_days: float = 12.0
    short_max_hold_days: float = 5.0
    tp1_close_ratio: float = 0.6
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.30
    price_lookback_bars: int = 260
    sl_lookback_bars: int = 14
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    breakdown_tolerance_pct: float = 1.2
    min_downtrend_pct: float = 6.0
    min_volume_ratio: float = 0.9


def signal_config_from_env() -> SignalConfig:
    d = SignalConfig()
    return SignalConfig(
        min_bars=env_int("MIN_BARS", d.min_bars),
        long_min_pullback=env_float("LONG_MIN_PULLBACK", d.long_min_pullback),
        long_max_pullback=env_float("LONG_MAX_PULLBACK", d.long_max_pullback),
        long_ideal_pullback=env_float("LONG_IDEAL_PULLBACK", d.long_ideal_pullback),
        long_max_rsi=env_float("LONG_MAX_RSI", d.long_max_rsi),
        short_min_bounce=env_float("SHORT_MIN_BOUNCE", d.short_min_bounce),
        short_max_bounce=env_float("SHORT_MAX_BOUNCE", d.short_max_bounce),
        short_ideal_bounce=env_float("SHORT_IDEAL_BOUNCE", d.short_ideal_bounce),
        short_min_rsi=env_float("SHORT_MIN_RSI", d.short_min_rsi),
        short_max_rsi=env_float("SHORT_MAX_RSI", d.short_max_rsi),
        long_sl_buffer=env_float("LONG_SL_BUFFER", d.long_sl_buffer),
        short_sl_buffer=env_float("SHORT_SL_BUFFER", d.short_sl_buffer),
        long_tp1_pct=env_float("LONG_TP1_PCT", d.long_tp1_pct),
        long_tp2_pct=env_float("LONG_TP2_PCT", d.long_tp2_pct),
        short_tp1_pct=env_float("SHORT_TP1_PCT", d.short_tp1_pct),
        short_tp2_pct=env_float("SHORT_TP2_PCT", d.short_tp2_pct),
        long_max_hold_days=env_float("LONG_MAX_HOLD_DAYS", d.long_max_hold_days),
        short_max_hold_days=env_float("SHORT_MAX_HOLD_DAYS", d.short_max_hold_days),
        tp1_close_ratio=env_float("TP1_CLOSE_RATIO", d.tp1_close_ratio),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        sl_lookback_bars=env_int("SL_LOOKBACK_BARS", d.sl_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        breakdown_tolerance_pct=env_float("BREAKDOWN_TOLERANCE_PCT", d.breakdown_tolerance_pct),
        min_downtrend_pct=env_float("MIN_DOWNTREND_PCT", d.min_downtrend_pct),
        min_volume_ratio=env_float("MIN_VOLUME_RATIO", d.min_volume_ratio),
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=weak_fundamental_breakdown_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: SignalConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund_raw(f: FundamentalRow, cfg: SignalConfig, short: bool) -> float:
    score = f.composite_score
    if cfg.use_mispricing_score and f.mispricing_score is not None:
        score = score * (1.0 - cfg.mispricing_weight) + f.mispricing_score * cfg.mispricing_weight
    return (100.0 - score if short else score) / 100.0


def compute_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    gap = (high - entry) / high * 100.0 if high > 0 else 999.0
    if gap > cfg.long_max_pullback:
        return SignalEvaluation(None, "rejected", "quality_reclaim_not_close_to_high", f"Close is {gap:.2f}% below high.", entry_price=entry, pullback_pct=round(gap, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above maximum.", entry_price=entry, rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    entry_score = clamp(1.0 - gap / max(cfg.long_max_pullback, 0.01), 0.0, 1.0) * 0.55 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.20 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.25
    combined = (_fund_raw(fundamental, cfg, short=False) * 0.45 + entry_score * 0.55) * 10.0
    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    signal = Signal(fundamental.symbol, "LONG", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 + cfg.long_tp1_pct), entry * (1.0 + cfg.long_tp2_pct), round(gap, 2), round(rsi, 2), round(vol_ratio, 3), f"Quality reclaim gap {gap:.1f}% | RSI {rsi:.0f}", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "quality_reclaim_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)


def compute_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    low_gap = (entry - lookback_low) / lookback_low * 100.0 if lookback_low > 0 else 999.0
    if low_gap > cfg.breakdown_tolerance_pct:
        return SignalEvaluation(None, "rejected", "not_breaking_down", f"Close is {low_gap:.2f}% above lookback low.", entry_price=entry, pullback_pct=round(low_gap, 2))
    trend_base = closes[-cfg.price_lookback_bars] if len(closes) > cfg.price_lookback_bars and closes[-cfg.price_lookback_bars] > 0 else closes[0]
    downtrend = (1.0 - entry / trend_base) * 100.0
    if downtrend < cfg.min_downtrend_pct:
        return SignalEvaluation(None, "rejected", "downtrend_too_small", f"Downtrend {downtrend:.2f}% below minimum {cfg.min_downtrend_pct:.2f}%.", entry_price=entry)
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_outside_breakdown_range", f"RSI {rsi:.2f} outside range.", entry_price=entry, rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    if vol_ratio < cfg.min_volume_ratio:
        return SignalEvaluation(None, "rejected", "breakdown_volume_too_low", f"Volume ratio {vol_ratio:.2f} below minimum.", entry_price=entry, volume_ratio=round(vol_ratio, 3))
    breakdown_score = clamp(1.0 - low_gap / max(cfg.breakdown_tolerance_pct, 0.01), 0.0, 1.0)
    trend_score = clamp(downtrend / 25.0, 0.0, 1.0)
    vol_score = clamp(vol_ratio / 1.8, 0.0, 1.0)
    entry_score = breakdown_score * 0.45 + trend_score * 0.35 + vol_score * 0.20
    combined = (_fund_raw(fundamental, cfg, short=True) * 0.50 + entry_score * 0.50) * 10.0
    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    signal = Signal(fundamental.symbol, "SHORT", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 - cfg.short_tp1_pct), entry * (1.0 - cfg.short_tp2_pct), round(low_gap, 2), round(rsi, 2), round(vol_ratio, 3), f"Breakdown gap {low_gap:.1f}% | Downtrend {downtrend:.1f}% | Vol {vol_ratio:.2f}x", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "weak_fundamental_breakdown_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)
