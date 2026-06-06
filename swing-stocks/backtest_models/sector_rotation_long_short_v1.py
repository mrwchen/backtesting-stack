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

from backtest_shared import Bar, FundamentalRow, TradeIntent, IntentEvaluation
from backtest_shared import (
    clamp,
    compute_rsi,
    directional_fundamental_score,
    env_bool,
    env_float,
    env_int,
    env_list,
    env_optional_float,
    env_str,
    mean,
)


@dataclass
class IntentConfig:
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
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.25
    fundamental_score_mode: str = "peer"
    fundamental_peer_weight: float = 1.0
    fundamental_abs_weight: float = 0.0
    long_min_absolute_score: Optional[float] = 50.0
    short_max_absolute_score: Optional[float] = 50.0
    price_lookback_bars: int = 260
    vol_short_bars: int = 5
    vol_long_bars: int = 25
    long_sector_allowlist: list = field(default_factory=list)
    short_sector_allowlist: list = field(default_factory=list)
    sector_bonus_score: float = 0.8


def intent_config_from_env() -> IntentConfig:
    d = IntentConfig()
    return IntentConfig(
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
        use_mispricing_score=env_bool("USE_MISPRICING_SCORE", d.use_mispricing_score),
        mispricing_weight=env_float("MISPRICING_WEIGHT", d.mispricing_weight),
        fundamental_score_mode=env_str("FUNDAMENTAL_SCORE_MODE", d.fundamental_score_mode),
        fundamental_peer_weight=env_float("FUNDAMENTAL_PEER_WEIGHT", d.fundamental_peer_weight),
        fundamental_abs_weight=env_float("FUNDAMENTAL_ABS_WEIGHT", d.fundamental_abs_weight),
        long_min_absolute_score=env_optional_float("LONG_MIN_ABSOLUTE_SCORE", d.long_min_absolute_score),
        short_max_absolute_score=env_optional_float("SHORT_MAX_ABSOLUTE_SCORE", d.short_max_absolute_score),
        price_lookback_bars=env_int("PRICE_LOOKBACK_BARS", d.price_lookback_bars),
        vol_short_bars=env_int("VOL_SHORT_BARS", d.vol_short_bars),
        vol_long_bars=env_int("VOL_LONG_BARS", d.vol_long_bars),
        long_sector_allowlist=env_list("LONG_SECTOR_ALLOWLIST", d.long_sector_allowlist),
        short_sector_allowlist=env_list("SHORT_SECTOR_ALLOWLIST", d.short_sector_allowlist),
        sector_bonus_score=env_float("SECTOR_BONUS_SCORE", d.sector_bonus_score),
    )


def required_bar_lookback(cfg: IntentConfig) -> int:
    return max(
        cfg.min_bars,
        cfg.price_lookback_bars,
        cfg.vol_long_bars,
        cfg.vol_short_bars,
        50,
    )


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=sector_rotation_long_short_v1", "summary": {}}


def _sector_allowed(sector: str, allowlist: list) -> bool:
    return not allowlist or sector.lower() in {s.lower() for s in allowlist}


def _vol_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _fund(f: FundamentalRow, cfg: IntentConfig, short: bool) -> float:
    return directional_fundamental_score(
        f,
        short=short,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    if not _sector_allowed(fundamental.sector, cfg.long_sector_allowlist):
        return IntentEvaluation(None, "rejected", "sector_not_in_long_rotation", f"Sector {fundamental.sector or 'unknown'} is not in long allowlist.")
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    high = max(highs[-cfg.price_lookback_bars:])
    pullback = (high - entry) / high * 100.0 if high > 0 else 999.0
    if pullback < cfg.long_min_pullback or pullback > cfg.long_max_pullback:
        return IntentEvaluation(None, "rejected", "pullback_outside_sector_rotation_range", f"Pullback {pullback:.2f}% outside long range.")
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_above_max", f"RSI {rsi:.2f} above max.")
    vol_ratio = _vol_ratio(volumes, cfg)
    sector_score = 1.0 if cfg.long_sector_allowlist else 0.5
    entry_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / cfg.long_ideal_pullback, 0.0, 1.0) * 0.35 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.25 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.15 + sector_score * 0.25
    combined = (_fund(fundamental, cfg, False) * 0.55 + entry_score * 0.45) * 10.0 + cfg.sector_bonus_score * sector_score
    reason = f"Sector {fundamental.sector or 'unknown'} | Pullback {pullback:.1f}% | RSI {rsi:.0f}"
    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "sector_rotation_long_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
    if not _sector_allowed(fundamental.sector, cfg.short_sector_allowlist):
        return IntentEvaluation(None, "rejected", "sector_not_in_short_rotation", f"Sector {fundamental.sector or 'unknown'} is not in short allowlist.")
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [float(b.volume) for b in bars]
    entry = closes[-1]
    low = min(lows[-cfg.price_lookback_bars:])
    bounce = (entry - low) / low * 100.0 if low > 0 else 999.0
    if bounce < cfg.short_min_bounce or bounce > cfg.short_max_bounce:
        return IntentEvaluation(None, "rejected", "bounce_outside_sector_rotation_range", f"Bounce {bounce:.2f}% outside short range.")
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "rsi_outside_short_range", f"RSI {rsi:.2f} outside range.")
    vol_ratio = _vol_ratio(volumes, cfg)
    sector_score = 1.0 if cfg.short_sector_allowlist else 0.5
    entry_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / cfg.short_ideal_bounce, 0.0, 1.0) * 0.35 + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.25 + clamp(vol_ratio / 1.4, 0.0, 1.0) * 0.15 + sector_score * 0.25
    combined = (_fund(fundamental, cfg, True) * 0.55 + entry_score * 0.45) * 10.0 + cfg.sector_bonus_score * sector_score
    reason = f"Sector {fundamental.sector or 'unknown'} | Bounce {bounce:.1f}% | RSI {rsi:.0f}"
    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "sector_rotation_short_passed", reason)
