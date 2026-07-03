import numpy as np
import pandas as pd

from src.trend_template import compute_template
from tests.util import make_cfg


def test_uptrend_passes_downtrend_fails():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(
        {"UP": np.linspace(50, 150, n), "DOWN": np.linspace(150, 50, n)}, index=dates
    )
    rs = pd.DataFrame({"UP": np.full(n, 90.0), "DOWN": np.full(n, 90.0)}, index=dates)

    cfg = make_cfg()
    template = compute_template(close, rs, cfg)
    last = template["template_pass"].iloc[-1]

    assert bool(last["UP"]) is True
    assert bool(last["DOWN"]) is False


def test_rs_criterion_gates_template():
    n = 300
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n)}, index=dates)
    rs_low = pd.DataFrame({"UP": np.full(n, 50.0)}, index=dates)

    cfg = make_cfg(rs_min=70)
    template = compute_template(close, rs_low, cfg)

    assert bool(template["template_pass"].iloc[-1]["UP"]) is False
    assert bool(template["crit_price_above_ma50"].iloc[-1]["UP"]) is True


def test_no_pass_without_ma_history():
    n = 100  # fewer bars than the 200d MA needs
    dates = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame({"UP": np.linspace(50, 150, n)}, index=dates)
    rs = pd.DataFrame({"UP": np.full(n, 90.0)}, index=dates)

    template = compute_template(close, rs, make_cfg())
    assert not template["template_pass"].to_numpy().any()
