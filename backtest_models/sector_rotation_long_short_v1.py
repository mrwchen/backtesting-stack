"""Sector rotation long/short swing model.

Model idea:
  - LONG: strongest fundamental names from allowed/leading sectors.
  - SHORT: weakest fundamental names from allowed/lagging sectors.
  - Sector preference is explicit and PIT-safe because it uses the candidate row only.
"""

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, env_list, mean


@dataclass
class SignalConfig:
    min_bars: int = 160
    long_min_pullback: float = 2.0
    long_max_pullback: float = 18.0
    long_ideal_pullback: float = 7.0
    long_max_rsi: float = 62.0
    short_min_bounce: float = 2.0
    short_max_bounce: float = 18.0
    short_ideal_bounce: float = 7.0
    short_min_rsi: float = 35.0
    short_max_rsi: float = 68.0
    long_sl_buffer: float = 0.007
    short_sl_buffer: float = 0.007
    long_tp1_pct: float = 0.055
    long_tp2_pct: float = 0.11
    short_tp1_pct: float = 0.055
    short_tp2_pct: float = 0.10
    long_max_hold_days: float = 12.0
    short_max_hold_days: float = 5.0
    tp1_close_ratio: float = 0.6
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25
    price_lookback_bars: int = 260
    sl_lookback_bars: int = 12
    vol_short_bars: int = 5
    vol_long_bars: int = 25
    long_sector_allowlist: list = field(default_factory=list)
    short_sector_allowlist: list = field(default_factory=list)
    sector_bonus_score: float = 0.8


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
        long_sector_allowlist=env_list("LONG_SECTOR_ALLOWLIST", d.long_sector_allowlist),
        short_sector_allowlist=env_list("SHORT_SECTOR_ALLOWLIST", d.short_sector_allowlist),
        sector_bonus_score=env_float("SECTOR_BONUS_SCORE", d.sector_bonus_score),
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


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=sector_rotation_long_short_v1", "summary": {}}


def _sector_allowed(sector: str, allowlist: list) -> bool:
    return not allowlist or sector.lower() in {s.lower() for s in allowlist}


def _vol_ratio(volumes: list[float], cfg: SignalConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund(f: FundamentalRow, cfg: SignalConfig, short: bool) -> float:
    score = f.composite_score
    if cfg.use_mispricing_score and f.mispricing_score is not None:
        score = score * (1.0 - cfg.mispricing_weight) + f.mispricing_score * cfg.mispricing_weight
    return (100.0 - score if short else score) / 100.0


def compute_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    if not _sector_allowed(fundamental.sector, cfg.long_sector_allowlist):
        return SignalEvaluation(None, "rejected", "sector_not_in_long_rotation", f"Sector {fundamental.sector or 'unknown'} is not in long allowlist.")
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    pullback = (high - entry) / high * 100.0 if high > 0 else 999.0
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return SignalEvaluation(None, "rejected", "pullback_outside_sector_rotation_range", f"Pullback {pullback:.2f}% outside long range.", entry_price=entry, pullback_pct=round(pullback, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above max.", entry_price=entry, pullback_pct=round(pullback, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    sector_score = 1.0 if cfg.long_sector_allowlist else 0.5
    entry_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / cfg.long_ideal_pullback, 0.0, 1.0) * 0.35 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.25 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.15 + sector_score * 0.25
    combined = (_fund(fundamental, cfg, False) * 0.55 + entry_score * 0.45) * 10.0 + cfg.sector_bonus_score * sector_score
    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    signal = Signal(fundamental.symbol, "LONG", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 + cfg.long_tp1_pct), entry * (1.0 + cfg.long_tp2_pct), round(pullback, 2), round(rsi, 2), round(vol_ratio, 3), f"Sector {fundamental.sector or 'unknown'} | Pullback {pullback:.1f}% | RSI {rsi:.0f}", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "sector_rotation_long_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)


def compute_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    if not _sector_allowed(fundamental.sector, cfg.short_sector_allowlist):
        return SignalEvaluation(None, "rejected", "sector_not_in_short_rotation", f"Sector {fundamental.sector or 'unknown'} is not in short allowlist.")
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (entry - low) / low * 100.0 if low > 0 else 999.0
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return SignalEvaluation(None, "rejected", "bounce_outside_sector_rotation_range", f"Bounce {bounce:.2f}% outside short range.", entry_price=entry, pullback_pct=round(bounce, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return SignalEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside range.", entry_price=entry, pullback_pct=round(bounce, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    sector_score = 1.0 if cfg.short_sector_allowlist else 0.5
    entry_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / cfg.short_ideal_bounce, 0.0, 1.0) * 0.35 + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.25 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.15 + sector_score * 0.25
    combined = (_fund(fundamental, cfg, True) * 0.55 + entry_score * 0.45) * 10.0 + cfg.sector_bonus_score * sector_score
    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    signal = Signal(fundamental.symbol, "SHORT", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 - cfg.short_tp1_pct), entry * (1.0 - cfg.short_tp2_pct), round(bounce, 2), round(rsi, 2), round(vol_ratio, 3), f"Sector {fundamental.sector or 'unknown'} | Bounce {bounce:.1f}% | RSI {rsi:.0f}", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "sector_rotation_short_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)
