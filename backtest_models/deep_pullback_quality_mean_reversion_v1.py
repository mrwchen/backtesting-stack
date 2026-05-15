"""Deep pullback quality mean-reversion swing model.

Model idea:
  - LONG: buy high-quality stocks after a deep but not catastrophic pullback.
  - SHORT: short weak stocks after an extended relief bounce.
  - Uses only information available in the provided PIT fundamental row and bars.
"""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, env_list, mean


@dataclass
class SignalConfig:
    long_max_score: float = 58.0
    short_min_score: float = 65.0
    long_min_fundamental: float = 72.0
    short_max_fundamental: float = 32.0
    min_bars: int = 220

    long_min_pullback: float = 15.0
    long_max_pullback: float = 35.0
    long_ideal_pullback: float = 22.0
    long_max_rsi: float = 45.0
    short_min_bounce: float = 12.0
    short_max_bounce: float = 32.0
    short_ideal_bounce: float = 20.0
    short_min_rsi: float = 45.0
    short_max_rsi: float = 75.0

    long_sl_buffer: float = 0.008
    short_sl_buffer: float = 0.008
    long_tp1_pct: float = 0.07
    long_tp2_pct: float = 0.14
    short_tp1_pct: float = 0.06
    short_tp2_pct: float = 0.12

    long_label_blocklist: list = field(default_factory=lambda: ["value_trap", "overvalued_weak"])
    short_label_blocklist: list = field(default_factory=lambda: ["deep_value", "quality_value", "compounder"])
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.35

    price_lookback_bars: int = 420
    sl_lookback_bars: int = 16
    vol_short_bars: int = 5
    vol_long_bars: int = 30
    stabilization_bars: int = 3
    min_reclaim_pct: float = 0.5


def signal_config_from_env() -> SignalConfig:
    d = SignalConfig()
    return SignalConfig(
        long_max_score=env_float("LONG_MAX_SCORE", d.long_max_score),
        short_min_score=env_float("SHORT_MIN_SCORE", d.short_min_score),
        long_min_fundamental=env_float("LONG_MIN_FUNDAMENTAL", d.long_min_fundamental),
        short_max_fundamental=env_float("SHORT_MAX_FUNDAMENTAL", d.short_max_fundamental),
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
        long_label_blocklist=env_list("LONG_LABEL_BLOCKLIST", d.long_label_blocklist),
        short_label_blocklist=env_list("SHORT_LABEL_BLOCKLIST", d.short_label_blocklist),
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        sl_lookback_bars=env_int("SL_LOOKBACK_BARS", d.sl_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        stabilization_bars=env_int("STABILIZATION_BARS", d.stabilization_bars),
        min_reclaim_pct=env_float("MIN_RECLAIM_PCT", d.min_reclaim_pct),
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals, long_max_hold_days, short_max_hold_days, tp1_close_ratio):
    yield {"config": dataclasses.replace(base_cfg), "long_max_hold_days": long_max_hold_days, "short_max_hold_days": short_max_hold_days, "tp1_close_ratio": tp1_close_ratio, "notes": "grid model=deep_pullback_quality_mean_reversion_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: SignalConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund_score(f: FundamentalRow, cfg: SignalConfig, short: bool = False) -> float:
    raw = f.composite_score
    if cfg.use_mispricing_score and f.mispricing_score is not None:
        raw = raw * (1.0 - cfg.mispricing_weight) + f.mispricing_score * cfg.mispricing_weight
    return (100.0 - raw if short else raw) / 100.0


def compute_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    pullback = (high - entry) / high * 100.0 if high > 0 else 999.0
    if pullback < cfg.long_min_pullback:
        return SignalEvaluation(None, "rejected", "pullback_not_deep_enough", f"Pullback {pullback:.2f}% is below minimum {cfg.long_min_pullback:.2f}%.", entry_price=entry, pullback_pct=round(pullback, 2))
    if pullback > cfg.long_max_pullback:
        return SignalEvaluation(None, "rejected", "pullback_too_deep", f"Pullback {pullback:.2f}% is above maximum {cfg.long_max_pullback:.2f}%.", entry_price=entry, pullback_pct=round(pullback, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_not_oversold", f"RSI {rsi:.2f} is above maximum {cfg.long_max_rsi:.2f}.", entry_price=entry, pullback_pct=round(pullback, 2), rsi_1h=round(rsi, 2))
    recent_low = min(lows[-max(1, cfg.stabilization_bars):])
    reclaim = (entry / recent_low - 1.0) * 100.0 if recent_low > 0 else 0.0
    if reclaim < cfg.min_reclaim_pct:
        return SignalEvaluation(None, "rejected", "no_stabilization_reclaim", f"Close reclaimed only {reclaim:.2f}% from recent low.", entry_price=entry, pullback_pct=round(pullback, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    pullback_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / cfg.long_ideal_pullback, 0.0, 1.0)
    rsi_score = clamp((cfg.long_max_rsi - rsi) / 25.0, 0.0, 1.0)
    reclaim_score = clamp(reclaim / 4.0, 0.0, 1.0)
    entry_score = pullback_score * 0.45 + rsi_score * 0.30 + reclaim_score * 0.20 + clamp(1.2 - vol_ratio, 0.0, 1.0) * 0.05
    combined = (_fund_score(fundamental, cfg) * 0.50 + entry_score * 0.50) * 10.0
    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    signal = Signal(fundamental.symbol, "LONG", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 + cfg.long_tp1_pct), entry * (1.0 + cfg.long_tp2_pct), round(pullback, 2), round(rsi, 2), round(vol_ratio, 3), f"Deep pullback {pullback:.1f}% | RSI {rsi:.0f} | Reclaim {reclaim:.1f}%", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "deep_quality_reversion_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)


def compute_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (entry - low) / low * 100.0 if low > 0 else 999.0
    if bounce < cfg.short_min_bounce:
        return SignalEvaluation(None, "rejected", "bounce_not_extended_enough", f"Bounce {bounce:.2f}% is below minimum {cfg.short_min_bounce:.2f}%.", entry_price=entry, pullback_pct=round(bounce, 2))
    if bounce > cfg.short_max_bounce:
        return SignalEvaluation(None, "rejected", "bounce_too_extended", f"Bounce {bounce:.2f}% is above maximum {cfg.short_max_bounce:.2f}%.", entry_price=entry, pullback_pct=round(bounce, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_not_in_relief_zone", f"RSI {rsi:.2f} outside short relief range.", entry_price=entry, pullback_pct=round(bounce, 2), rsi_1h=round(rsi, 2))
    recent_high = max(highs[-max(1, cfg.stabilization_bars):])
    rejection = (recent_high / entry - 1.0) * 100.0 if entry > 0 else 0.0
    if rejection < cfg.min_reclaim_pct:
        return SignalEvaluation(None, "rejected", "no_bounce_rejection", f"Close rejected only {rejection:.2f}% from recent high.", entry_price=entry, pullback_pct=round(bounce, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    bounce_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / cfg.short_ideal_bounce, 0.0, 1.0)
    rsi_score = clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0)
    rejection_score = clamp(rejection / 4.0, 0.0, 1.0)
    entry_score = bounce_score * 0.45 + rsi_score * 0.25 + rejection_score * 0.25 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.05
    combined = (_fund_score(fundamental, cfg, short=True) * 0.50 + entry_score * 0.50) * 10.0
    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    signal = Signal(fundamental.symbol, "SHORT", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 - cfg.short_tp1_pct), entry * (1.0 - cfg.short_tp2_pct), round(bounce, 2), round(rsi, 2), round(vol_ratio, 3), f"Relief bounce {bounce:.1f}% | RSI {rsi:.0f} | Rejection {rejection:.1f}%", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "weak_relief_reversion_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)
