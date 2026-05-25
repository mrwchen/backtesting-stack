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

from backtest_shared import Bar, FundamentalRow, TradeIntent, IntentEvaluation
from backtest_shared import (
    clamp,
    compute_rsi,
    directional_fundamental_score,
    env_bool,
    env_float,
    env_int,
    env_optional_float,
    env_str,
    mean,
)


@dataclass
class IntentConfig:
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
    use_mispricing_score: bool = True
    mispricing_weight: float = 0.45
    fundamental_score_mode: str = "blend"
    fundamental_peer_weight: float = 0.50
    fundamental_abs_weight: float = 0.50
    long_min_absolute_score: Optional[float] = None
    short_max_absolute_score: Optional[float] = None
    price_lookback_bars: int = 260
    vol_short_bars: int = 5
    vol_long_bars: int = 25
    min_long_mispricing_score: float = 60.0
    max_short_mispricing_score: float = 40.0
    require_mispricing_score: bool = True


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
        min_long_mispricing_score=env_float("MIN_LONG_MISPRICING_SCORE", d.min_long_mispricing_score),
        max_short_mispricing_score=env_float("MAX_SHORT_MISPRICING_SCORE", d.max_short_mispricing_score),
        require_mispricing_score=env_bool("REQUIRE_MISPRICING_SCORE", d.require_mispricing_score),
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
    yield {"config": dataclasses.replace(base_cfg), "notes": "grid model=filing_aware_fundamental_momentum_v1", "summary": {}}


def _vol_ratio(volumes: list[float], cfg: IntentConfig) -> float:
    short = mean(volumes[-cfg.vol_short_bars:])
    long = mean(volumes[-cfg.vol_long_bars:-cfg.vol_short_bars]) if len(volumes) > cfg.vol_long_bars else short
    return short / long if long > 0 else 1.0


def _blended(f: FundamentalRow, cfg: IntentConfig, short: bool) -> float:
    return directional_fundamental_score(
        f,
        short=short,
        score_mode=cfg.fundamental_score_mode,
        peer_weight=cfg.fundamental_peer_weight,
        abs_weight=cfg.fundamental_abs_weight,
        use_mispricing_score=cfg.use_mispricing_score,
        mispricing_weight=cfg.mispricing_weight,
    )


def _check_mispricing_long(fundamental: FundamentalRow, cfg: IntentConfig) -> Optional[IntentEvaluation]:
    if fundamental.mispricing_score is None:
        if cfg.require_mispricing_score:
            return IntentEvaluation(None, "rejected", "missing_fundamental_momentum_proxy", "Mispricing score is required but missing.")
        return None
    if fundamental.mispricing_score < cfg.min_long_mispricing_score:
        return IntentEvaluation(None, "rejected", "fundamental_momentum_proxy_too_low", f"Mispricing score {fundamental.mispricing_score:.2f} below minimum {cfg.min_long_mispricing_score:.2f}.")
    return None


def _check_mispricing_short(fundamental: FundamentalRow, cfg: IntentConfig) -> Optional[IntentEvaluation]:
    if fundamental.mispricing_score is None:
        if cfg.require_mispricing_score:
            return IntentEvaluation(None, "rejected", "missing_fundamental_momentum_proxy", "Mispricing score is required but missing.")
        return None
    if fundamental.mispricing_score > cfg.max_short_mispricing_score:
        return IntentEvaluation(None, "rejected", "fundamental_weakness_proxy_too_high", f"Mispricing score {fundamental.mispricing_score:.2f} above maximum {cfg.max_short_mispricing_score:.2f}.")
    return None


def compute_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_long_intent(bars, fundamental, now, cfg).intent


def evaluate_long_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
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
        return IntentEvaluation(None, "rejected", "filing_proxy_pullback_outside_range", f"Pullback {pullback:.2f}% outside range.")
    rsi = compute_rsi(closes[-50:])
    if rsi > cfg.long_max_rsi:
        return IntentEvaluation(None, "rejected", "filing_proxy_rsi_too_high", f"RSI {rsi:.2f} above max.")
    vol_ratio = _vol_ratio(volumes, cfg)
    mispricing_component = clamp(((fundamental.mispricing_score or cfg.min_long_mispricing_score) - cfg.min_long_mispricing_score) / 30.0, 0.0, 1.0)
    entry_score = clamp(1.0 - abs(pullback - cfg.long_ideal_pullback) / max(cfg.long_ideal_pullback, 1.0), 0.0, 1.0) * 0.30 + clamp((cfg.long_max_rsi - rsi) / 30.0, 0.0, 1.0) * 0.20 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.15 + mispricing_component * 0.35
    combined = (_blended(fundamental, cfg, False) * 0.60 + entry_score * 0.40) * 10.0
    reason = f"PIT fundamental momentum proxy | Pullback {pullback:.1f}% | Mispricing {fundamental.mispricing_score}"
    intent = TradeIntent(fundamental.symbol, "LONG", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "pit_fundamental_momentum_long_passed", reason)


def compute_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> Optional[TradeIntent]:
    return evaluate_short_intent(bars, fundamental, now, cfg).intent


def evaluate_short_intent(bars: list[Bar], fundamental: FundamentalRow, now: datetime, cfg: IntentConfig) -> IntentEvaluation:
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
        return IntentEvaluation(None, "rejected", "filing_proxy_bounce_outside_range", f"Bounce {bounce:.2f}% outside range.")
    rsi = compute_rsi(closes[-50:])
    if rsi < cfg.short_min_rsi or rsi > cfg.short_max_rsi:
        return IntentEvaluation(None, "rejected", "filing_proxy_rsi_outside_short_range", f"RSI {rsi:.2f} outside range.")
    vol_ratio = _vol_ratio(volumes, cfg)
    mispricing_component = clamp((cfg.max_short_mispricing_score - (fundamental.mispricing_score or cfg.max_short_mispricing_score)) / 30.0, 0.0, 1.0)
    entry_score = clamp(1.0 - abs(bounce - cfg.short_ideal_bounce) / max(cfg.short_ideal_bounce, 1.0), 0.0, 1.0) * 0.30 + clamp((rsi - cfg.short_min_rsi) / max(cfg.short_max_rsi - cfg.short_min_rsi, 1.0), 0.0, 1.0) * 0.20 + clamp(vol_ratio / 1.5, 0.0, 1.0) * 0.15 + mispricing_component * 0.35
    combined = (_blended(fundamental, cfg, True) * 0.60 + entry_score * 0.40) * 10.0
    reason = f"PIT fundamental weakness proxy | Bounce {bounce:.1f}% | Mispricing {fundamental.mispricing_score}"
    intent = TradeIntent(fundamental.symbol, "SHORT", round(combined, 4), reason)
    return IntentEvaluation(intent, "intent", "pit_fundamental_momentum_short_passed", reason)
