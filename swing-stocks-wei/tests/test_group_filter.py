from __future__ import annotations

import numpy as np
import pandas as pd

from src.group_filter import compute_industry_breadth
from tests.util import make_config


def test_industry_breadth_gate_uses_ma200_hysteresis():
    dates = pd.bdate_range("2023-01-02", periods=220)
    close = pd.DataFrame(
        {
            "AAA": [100.0] * 210 + [102.0] * 10,
            "AAB": [100.0] * 210 + [103.0] * 10,
            "AAC": [100.0] * 210 + [90.0] * 10,
            "BBA": [100.0] * 210 + [90.0] * 10,
            "BBB": [100.0] * 210 + [90.0] * 10,
            "BBC": [100.0] * 210 + [90.0] * 10,
        },
        index=dates,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "AAB", "AAC", "BBA", "BBB", "BBC"],
            "ibkr_industry": ["TECH", "TECH", "TECH", "FIN", "FIN", "FIN"],
            "ibkr_category": ["SOFTWARE", "SOFTWARE", "SOFTWARE", "BANKS", "BANKS", "BANKS"],
        }
    )
    cfg = make_config(
        ibkr_industry_breadth_filter_enable=True,
        ibkr_industry_breadth_min_symbols=3,
    )

    breadth = compute_industry_breadth(close, universe, cfg)

    assert abs(breadth["ibkr_industry_breadth"].iloc[-1]["AAA"] - (2 / 3)) < 1e-9
    assert abs(breadth["ibkr_industry_breadth"].iloc[-1]["BBA"] - 0.0) < 1e-9
    assert bool(breadth["ibkr_industry_breadth_pass"].iloc[-1]["AAA"]) is True
    assert bool(breadth["ibkr_industry_breadth_pass"].iloc[-1]["BBA"]) is False


def test_industry_breadth_rejects_small_or_missing_industries():
    dates = pd.bdate_range("2023-01-02", periods=220)
    close = pd.DataFrame(
        {
            "AAA": [100.0] * 210 + [102.0] * 10,
            "BBA": [100.0] * 210 + [102.0] * 10,
            "BBB": [100.0] * 210 + [102.0] * 10,
            "BBC": [100.0] * 210 + [102.0] * 10,
        },
        index=dates,
    )
    universe = pd.DataFrame(
        {
            "symbol": ["AAA", "BBA", "BBB", "BBC"],
            "ibkr_industry": ["TECH", "FIN", "FIN", None],
            "ibkr_category": ["SOFTWARE", "BANKS", "BANKS", None],
        }
    )
    cfg = make_config(
        ibkr_industry_breadth_filter_enable=True,
        ibkr_industry_breadth_min_symbols=3,
    )

    breadth = compute_industry_breadth(close, universe, cfg)

    assert np.isnan(breadth["ibkr_industry_breadth"].iloc[-1]["AAA"])
    assert bool(breadth["ibkr_industry_breadth_pass"].iloc[-1]["AAA"]) is False
    assert bool(breadth["ibkr_industry_breadth_pass"].iloc[-1]["BBC"]) is False
