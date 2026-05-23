"""PIT fundamental momentum swing model.

Model idea:
  - Uses point-in-time filtered fundamental rows from the runner.
  - LONG: high composite score plus high mispricing/valuation momentum proxy.
  - SHORT: low composite score plus weak mispricing proxy.

Note:
  The current model API does not pass filing timestamps into FundamentalRow.
  Therefore this file is PIT-safe via the runner filters, but not truly
  filing-fresh until those timestamps are added to the model input.
"""

import dataclasses
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from backtest_shared import Bar, FundamentalRow, Signal, SignalEvaluation
from backtest_shared import clamp, compute_rsi, env_bool, env_float, env_int, mean


@dataclass
class SignalConfig:
    min_bars: int = 160
    long_min_pullback: float = 0.0
    long_max_pullback: float = 16.0
    long_ideal_pullback: float = 6.0
    long_max_rsi: float = 64.0
    short_min_bounce: float = 0.0
    short_max_bounce: float = 16.0
    short_ideal_bounce: float = 6.0
    short_min_rsi: float = 30.0
    short_max_rsi: float = 66.0
    long_sl_buffer: float = 0.007
    short_sl_buffer: float = 0.007
    long_tp1_pct: float = 0.06
    long_tp2_pct: float = 0.13
    short_tp1_pct: float = 0.06
    short_tp2_pct: float = 0.12
    long_max_hold_days: float = 12.0
    short_max_hold_days: float = 5.0
    tp1_close_ratio: float = 0.6
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.45
    price_lookback_bars: int = 260
    sl_lookback_bars: int = 14
    vol_short_bars: int = 5
    vol_long_bars: int = 25
    min_long_mispricing_score: float = 60.0
    max_short_mispricing_score: float = 40.0
    require_mispricing_score: bool = True


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
        min_long_mispricing_score=env_float("MIN_LONG_MISPRICING_SCORE", d.min_long_mispricing_score),
        max_short_mispricing_score=env_float("MAX_SHORT_MISPRICING_SCORE", d.max_short_mispricing_score),
        require_mispricing_score=env_bool("REQUIRE_MISPRICING_SCORE", d.require_mispricing_score),
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
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=filing_aware_fundamental_momentum_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: SignalConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _blended(f: FundamentalRow, cfg: SignalConfig, short: bool) -> float:
    score = f.composite_score
    if cfg.use_mispricing_score and f.mispricing_score is not None:
        score = score * (1.0 - cfg.mispricing_weight) + f.mispricing_score * cfg.mispricing_weight
    return (100.0 - score if short else score) / 100.0


def _check_mispricing_long(fundamental: FundamentalRow, cfg: SignalConfig) -> Optional[SignalEvaluation]:
    if fundamental.mispricing_score is None:
        if cfg.require_mispricing_score:
            return SignalEvaluation(None, "rejected", "missing_fundamental_momentum_proxy", "Mispricing score is required but missing.")
        return None
    if fundamental.mispricing_score < cfg.min_long_mispricing_score:
        return SignalEvaluation(None, "rejected", "fundamental_momentum_proxy_too_low", f"Mispricing score {fundamental.mispricing_score:.2f} below minimum {cfg.min_long_mispricing_score:.2f}.")
    return None


def _check_mispricing_short(fundamental: FundamentalRow, cfg: SignalConfig) -> Optional[SignalEvaluation]:
    if fundamental.mispricing_score is None:
        if cfg.require_mispricing_score:
            return SignalEvaluation(None, "rejected", "missing_fundamental_momentum_proxy", "Mispricing score is required but missing.")
        return None
    if fundamental.mispricing_score > cfg.max_short_mispricing_score:
        return SignalEvaluation(None, "rejected", "fundamental_weakness_proxy_too_high", f"Mispricing score {fundamental.mispricing_score:.2f} above maximum {cfg.max_short_mispricing_score:.2f}.")
    return None


def compute_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_long_signal(bars, fundamental, now, cfg).signal


def evaluate_long_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    early = _check_mispricing_long(fundamental, cfg)
    if early is not None:
        return early
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    pullback = (high - entry) / high * 100.0 if high > 0 else 999.0
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return SignalEvaluation(None, "rejected", "filing_proxy_pullback_outside_range", f"Pullback {pullback:.2f}% outside range.", entry_price=entry, pullback_pct=round(pullback, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return SignalEvaluation(None, "rejected", "filing_proxy_rsi_too_high", f"RSI {rsi:.2f} above max.", entry_price=entry, pullback_pct=round(pullback, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    mispricing_component = clamp(((fundamental.mispricing_score or cfg.min_long_mispricing_score) - cfg.min_long_mispricing_score) / 30.0, 0.0, 1.0)
    entry_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / max(cfg.long_ideal_pullback, 1.0), 0.0, 1.0) * 0.30 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.20 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.15 + mispricing_component * 0.35
    combined = (_blended(fundamental, cfg, False) * 0.60 + entry_score * 0.40) * 10.0
    sl = min(lows[-cfg.sl_lookback_bars:]) * (1.0 - cfg.long_sl_buffer)
    signal = Signal(fundamental.symbol, "LONG", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 + cfg.long_tp1_pct), entry * (1.0 + cfg.long_tp2_pct), round(pullback, 2), round(rsi, 2), round(vol_ratio, 3), f"PIT fundamental momentum proxy | Pullback {pullback:.1f}% | Mispricing {fundamental.mispricing_score}", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "pit_fundamental_momentum_long_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)


def compute_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> Optional[Signal]:
    return evaluate_short_signal(bars, fundamental, now, cfg).signal


def evaluate_short_signal(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: SignalConfig) -> SignalEvaluation:
    early = _check_mispricing_short(fundamental, cfg)
    if early is not None:
        return early
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (entry - low) / low * 100.0 if low > 0 else 999.0
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return SignalEvaluation(None, "rejected", "filing_proxy_bounce_outside_range", f"Bounce {bounce:.2f}% outside range.", entry_price=entry, pullback_pct=round(bounce, 2))
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return SignalEvaluation(None, "rejected", "filing_proxy_rsi_outside_short_range", f"RSI {rsi:.2f} outside range.", entry_price=entry, pullback_pct=round(bounce, 2), rsi_1h=round(rsi, 2))
    vol_ratio = _vol_ratio(volumes, cfg)
    mispricing_component = clamp((cfg.max_short_mispricing_score - (fundamental.mispricing_score or cfg.max_short_mispricing_score)) / 30.0, 0.0, 1.0)
    entry_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / max(cfg.short_ideal_bounce, 1.0), 0.0, 1.0) * 0.30 + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.20 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.15 + mispricing_component * 0.35
    combined = (_blended(fundamental, cfg, True) * 0.60 + entry_score * 0.40) * 10.0
    sl = max(highs[-cfg.sl_lookback_bars:]) * (1.0 + cfg.short_sl_buffer)
    signal = Signal(fundamental.symbol, "SHORT", fundamental.composite_score, round(entry_score, 4), round(combined, 4), entry, sl, entry * (1.0 - cfg.short_tp1_pct), entry * (1.0 - cfg.short_tp2_pct), round(bounce, 2), round(rsi, 2), round(vol_ratio, 3), f"PIT fundamental weakness proxy | Bounce {bounce:.1f}% | Mispricing {fundamental.mispricing_score}", fundamental.valuation_label, fundamental.sector, fundamental.industry)
    return SignalEvaluation(signal, "signal", "pit_fundamental_momentum_short_passed", signal.entry_reason, entry, signal.stop_loss, signal.take_profit_1, signal.take_profit_2, signal.pullback_pct, signal.rsi_1h, signal.volume_ratio, signal.entry_score, signal.combined_score)
