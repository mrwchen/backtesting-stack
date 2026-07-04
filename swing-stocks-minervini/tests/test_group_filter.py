import numpy as np
import pandas as pd

from src.group_filter import compute_leadership
from tests.util import make_cfg


def _close_frame(periods: int = 300) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=periods)
    return pd.DataFrame(
        {
            "AAA": 50 * (1.006 ** np.arange(periods)),
            "AAB": 50 * (1.004 ** np.arange(periods)),
            "AAC": 50 * (1.002 ** np.arange(periods)),
            "BBA": 50 * (0.999 ** np.arange(periods)),
            "BBB": 50 * (0.997 ** np.arange(periods)),
            "BBC": 50 * (0.995 ** np.arange(periods)),
        },
        index=dates,
    )


def _rs_raw(close: pd.DataFrame, cfg) -> pd.DataFrame:
    out = None
    for weight, lookback in zip(cfg.rs_weights, cfg.rs_lookbacks):
        term = weight * (close / close.shift(lookback))
        out = term if out is None else out + term
    return out


def test_ibkr_group_filter_requires_strong_group_and_stock_leadership():
    close = _close_frame()
    cfg = make_cfg(
        ibkr_industry_min_symbols=3,
        ibkr_category_min_symbols=3,
        ibkr_industry_rs_min=70,
        ibkr_category_rs_min=70,
        ibkr_stock_industry_rs_min=70,
        ibkr_stock_category_rs_min=70,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "AAB", "AAC", "BBA", "BBB", "BBC"],
            "ibkr_industry": ["TECH", "TECH", "TECH", "FIN", "FIN", "FIN"],
            "ibkr_category": ["SOFTWARE", "SOFTWARE", "SOFTWARE", "BANKS", "BANKS", "BANKS"],
        }
    )

    leadership = compute_leadership(_rs_raw(close, cfg), universe, cfg)
    last_pass = leadership["group_filter_pass"].iloc[-1]

    assert bool(leadership["ibkr_industry_pass"].iloc[-1]["AAA"]) is True
    assert bool(leadership["ibkr_industry_pass"].iloc[-1]["BBA"]) is False
    assert bool(last_pass["AAA"]) is True
    assert bool(last_pass["AAB"]) is False
    assert bool(last_pass["BBA"]) is False


def test_ibkr_group_filter_rejects_missing_taxonomy():
    close = _close_frame()
    cfg = make_cfg(
        ibkr_industry_min_symbols=3,
        ibkr_category_min_symbols=3,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "AAB", "AAC", "BBA", "BBB", "BBC"],
            "ibkr_industry": ["TECH", "TECH", "TECH", "FIN", "FIN", None],
            "ibkr_category": ["SOFTWARE", "SOFTWARE", "SOFTWARE", "BANKS", "BANKS", None],
        }
    )

    leadership = compute_leadership(_rs_raw(close, cfg), universe, cfg)

    assert bool(leadership["group_filter_pass"].iloc[-1]["BBC"]) is False
