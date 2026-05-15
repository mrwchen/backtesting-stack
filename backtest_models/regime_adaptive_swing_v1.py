"""Regime-adaptive swing model.

Model idea:
  - The runner selects LONG/SHORT from world regime.
  - This model changes the entry style by side:
    LONG = quality pullback with moderate momentum confirmation.
    SHORT = weak-fundamental breakdown/failed bounce.
"""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, env_list, mean


@dataclass
class SignalConfig:
    long_max_score: float = 55.0
    short_min_score: float = 62.0
    long_min_fundamental: float = 65.0
    short_max_fundamental: float = 38.0
    min_bars: int = 180
    long_min_pullback: float = 4.0
    long_max_pullback: float = 22.0
    long_ideal_pullback: float = 10.0
    long_max_rsi: float = 58.0
    short_min_bounce: float = 1.0
    short_max_bounce: float = 14.0
    short_ideal_bounce: float = 5.0
    short_min_rsi: float = 30.0
    short_max_rsi: float = 62.0
    long_sl_buffer: float = 0.007
    short_sl_buffer: float = 0.008
    long_tp1_pct: float = 0.055
    long_tp2_pct: float = 0.12
    short_tp1_pct: float = 0.06
    short_tp2_pct: float = 0.11
    long_label_blocklist: list = field(default_factory=lambda: ["value_trap", "overvalued_weak"])
    short_label_blocklist: list = field(default_factory=lambda: ["deep_value", "quality_value", "compounder"])
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.30
    price_lookback_bars: int = 300
    sl_lookback_bars: int = 14
    vol_short_bars: int = 5
    vol_long_bars: int = 25
    momentum_lookback_bars: int = 40
    min_long_momentum_pct: float = -4.0
    max_short_momentum_pct: float = 4.0


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
        momentum_lookback_bars=env_int("MOMENTUM_LOOKBACK_BARS", d.momentum_lookback_bars),
        min_long_momentum_pct=env_float("MIN_LONG_MOMENTUM_PCT", d.min_long_momentum_pct),
        max_short_momentum_pct=env_float("MAX_SHORT_MOMENTUM_PCT", d.max_short_momentum_pct),
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals, long_max_hold_days, short_max_hold_days, tp1_close_ratio):
    yield {"config": dataclasses.replace(base_cfg), "long_max_hold_days": long_max_hold_days, "short_max_hold_days": short_max_hold_days, "tp1_close_ratio": tp1_close_ratio, "notes": "grid model=regime_adaptive_swing_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: SignalConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund(f: FundamentalRow, cfg: SignalConfig, short: bool) -> float:
    score = f.composite_score
    if cfg.use_mispricing_score and f.mispricing_score is not None:
        score = score * (1.0 - cfg.mispricing_weight) + f.mispricing_score * cfg.mispricing_weight
    return (100.0 - score if short else score) / 100.0


def _momentum(closes: list[float], lookback: int) -> float:
    base = closes[-lookback] if len(closes) > lookback and closes[-lookback] > 0 else closes[0]
    return (closes[-1] / base - 1.0) * 100.0


def compute_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    lookback_high = max(highs[-cfg.price_lookback_bars:])
    if lookback_high <= 0 or entry <= 0:
        return SignalEvaluation(None, "rejected", "invalid_price", "Entry price or lookback high is not positive.", entry_price=entry)
    pullback = (lookback_high - entry) / lookback_high * 100.0
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return SignalEvaluation(None, "rejected", "adaptive_long_pullback_outside_range", f"Pullback {pullback:.2f}% outside adaptive long range.", entry_price=entry, pullback_pct=round(pullback, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(None, "rejected", "adaptive_long_rsi_too_high", f"RSI {rsi:.2f} above max.", entry_price=entry, pullback_pct=round(pullback, 2), rsi_1h=round(rsi, 2))
    mom = _momentum(closes, cfg.momentum_lookback_bars)
    if mom < cfg.min_long_momentum_pct:
        return SignalEvaluation(None, "rejected", "adaptive_long_momentum_too_weak", f"Momentum {mom:.2f}% below minimum.", entry_price=entry)
    vol_ratio = _vol_ratio(volumes, cfg)
    entry_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / cfg.long_ideal_pullback, 0.0, 1.0) * 0.40 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.25 + clamp((mom - cfg.min_long_momentum_pct) / 14.0, 0.0, 1.0) * 0.25 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.10
    combined = (_fund(fundamental, cfg, False) * 0.45 + entry_score * 0.55) * 10.0
    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    signal = Signal(fundamental.symbol, "LONG", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 + cfg.long_tp1_pct), entry * (1.0 + cfg.long_tp2_pct), round(pullback, 2), round(rsi, 2), round(vol_ratio, 3), f"Adaptive long pullback {pullback:.1f}% | Momentum {mom:.1f}%", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "regime_adaptive_long_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)


def compute_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    lookback_low = min(lows[-cfg.price_lookback_bars:])
    if lookback_low <= 0 or entry <= 0:
        return SignalEvaluation(None, "rejected", "invalid_price", "Entry price or lookback low is not positive.", entry_price=entry)
    bounce = (entry - lookback_low) / lookback_low * 100.0
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return SignalEvaluation(None, "rejected", "adaptive_short_bounce_outside_range", f"Bounce {bounce:.2f}% outside adaptive short range.", entry_price=entry, pullback_pct=round(bounce, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return SignalEvaluation(None, "rejected", "adaptive_short_rsi_outside_range", f"RSI {rsi:.2f} outside range.", entry_price=entry, pullback_pct=round(bounce, 2), rsi_1h=round(rsi, 2))
    mom = _momentum(closes, cfg.momentum_lookback_bars)
    if mom > cfg.max_short_momentum_pct:
        return SignalEvaluation(None, "rejected", "adaptive_short_momentum_too_strong", f"Momentum {mom:.2f}% above maximum.", entry_price=entry)
    vol_ratio = _vol_ratio(volumes, cfg)
    entry_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / cfg.short_ideal_bounce, 0.0, 1.0) * 0.35 + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.20 + clamp((cfg.max_short_momentum_pct - mom) / 14.0, 0.0, 1.0) * 0.35 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.10
    combined = (_fund(fundamental, cfg, True) * 0.50 + entry_score * 0.50) * 10.0
    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    signal = Signal(fundamental.symbol, "SHORT", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 - cfg.short_tp1_pct), entry * (1.0 - cfg.short_tp2_pct), round(bounce, 2), round(rsi, 2), round(vol_ratio, 3), f"Adaptive short bounce {bounce:.1f}% | Momentum {mom:.1f}%", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "regime_adaptive_short_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)
