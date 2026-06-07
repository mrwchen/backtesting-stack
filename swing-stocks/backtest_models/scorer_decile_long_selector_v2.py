"""V2 long-only scorer selector.

This version keeps the v1 selector mechanics, but uses additional config
guards for the two issues observed in the first live backtest:
  - overextended 130-hour momentum,
  - score saturation at price_momentum_score=100.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path


MODEL_NAME = "scorer_decile_long_selector_v2"

_BASE_PATH = Path(__file__).with_name("scorer_decile_long_selector_v1.py")
_SPEC = importlib.util.spec_from_file_location("_scorer_decile_long_selector_v1_base", _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load base scorer selector model from {_BASE_PATH}")
_BASE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _BASE
_SPEC.loader.exec_module(_BASE)

BENCHMARK_SYMBOL = _BASE.BENCHMARK_SYMBOL
BENCHMARK_SYMBOLS = _BASE.BENCHMARK_SYMBOLS
BENCHMARK_BAR_LOOKBACK = _BASE.BENCHMARK_BAR_LOOKBACK
IntentConfig = _BASE.IntentConfig


def intent_config_from_env() -> IntentConfig:
    return _BASE.intent_config_from_env()


def required_bar_lookback(cfg: IntentConfig) -> int:
    return _BASE.required_bar_lookback(cfg)


def iter_grid_search_configs(base_cfg, parse_grid_vals, parse_hold_grid_vals):
    yield {"config": dataclasses.replace(base_cfg), "notes": f"grid model={MODEL_NAME}", "summary": {}}


def set_market_context(cfg: IntentConfig, as_of_ts, bars_by_symbol) -> None:
    _BASE.set_market_context(cfg, as_of_ts, bars_by_symbol)


def compute_long_intent(bars, fundamental, now, cfg):
    return _BASE.compute_long_intent(bars, fundamental, now, cfg)


def evaluate_long_intent(bars, fundamental, now, cfg):
    return _BASE.evaluate_long_intent(bars, fundamental, now, cfg)


def compute_short_intent(bars, fundamental, now, cfg):
    return _BASE.compute_short_intent(bars, fundamental, now, cfg)


def evaluate_short_intent(bars, fundamental, now, cfg):
    return _BASE.evaluate_short_intent(bars, fundamental, now, cfg)


def evaluate_position_exit(pos, ts, open_, high, low, close, total_bars, cfg, *, exit_active: bool):
    return _BASE.evaluate_position_exit(
        pos,
        ts,
        open_,
        high,
        low,
        close,
        total_bars,
        cfg,
        exit_active=exit_active,
    )
