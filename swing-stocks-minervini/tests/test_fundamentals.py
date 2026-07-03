import numpy as np
import pandas as pd

from src.fundamentals import combine, eps_flags, revenue_margin_flags
from tests.util import make_cfg


def _grid():
    dates = pd.bdate_range("2022-01-03", "2024-12-31")
    symbols = pd.Index(["AAA"])
    return dates, symbols


def _quarterly_filings(n: int, quarterly_growth: float = 0.25) -> pd.DataFrame:
    """TTM net income built so consecutive diffs give quarterly NI 100*(1+g)^i."""
    quarterly_ni = 100.0 * (1 + quarterly_growth) ** np.arange(n)
    ttm = 400.0 + np.cumsum(quarterly_ni)
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "available_date": pd.date_range("2022-02-01", periods=n, freq="91D"),
            "net_income_ttm": ttm,
            "shares_diluted": 100.0,
            "revenue_ttm": 1000.0 * 1.05 ** np.arange(n),
            "net_margin_ttm": 0.10 + 0.005 * np.arange(n),
        }
    )


def test_eps_yoy_pass_needs_five_quarterly_diffs():
    dates, symbols = _grid()
    filings = _quarterly_filings(10)
    cfg = make_cfg(eps_yoy_min=0.20)

    eps_pass, eps_yoy = eps_flags(filings, dates, symbols, cfg)

    avail = filings["available_date"]
    # quarterly EPS exists from filing 2; YoY needs 4 more -> first pass at filing 6
    assert not eps_pass.loc[: avail[4], "AAA"].any()
    after = avail[5] + pd.Timedelta(days=3)
    assert bool(eps_pass.loc[after:, "AAA"].iloc[0]) is True
    # 25% quarterly growth -> YoY = 1.25^4 - 1 ~ 144%
    assert eps_yoy.loc[after:, "AAA"].iloc[0] > 1.4


def test_annual_filers_never_qualify():
    dates, symbols = _grid()
    filings = _quarterly_filings(10)
    filings["available_date"] = pd.date_range("2022-02-01", periods=10, freq="365D")

    eps_pass, _ = eps_flags(filings, dates, symbols, make_cfg())
    assert not eps_pass.to_numpy().any()  # gap > 130 days -> no quarterly diff


def test_eps_flag_goes_stale():
    dates, symbols = _grid()
    filings = _quarterly_filings(6)  # reporting stops after filing 6
    cfg = make_cfg(eps_stale_trading_days=130)

    eps_pass, _ = eps_flags(filings, dates, symbols, cfg)

    last_avail = filings["available_date"].iloc[-1]
    assert bool(eps_pass.loc[last_avail + pd.Timedelta(days=3):, "AAA"].iloc[0]) is True
    assert not eps_pass.loc[last_avail + pd.Timedelta(days=280):, "AAA"].any()


def test_revenue_and_margin_flags():
    dates, symbols = _grid()
    filings = _quarterly_filings(9)
    cfg = make_cfg(revenue_yoy_min=0.10)

    revenue_pass, revenue_yoy, margin_pass = revenue_margin_flags(filings, dates, symbols, cfg)

    late = filings["available_date"].iloc[-1] + pd.Timedelta(days=5)
    late_row = revenue_pass.loc[late:].iloc[0]
    # 4 filings/year at 5% each ~ 21% YoY -> pass; margin strictly rising -> pass
    assert bool(late_row["AAA"]) is True
    assert revenue_yoy.loc[late:, "AAA"].iloc[0] > 0.15
    assert bool(margin_pass.loc[late:, "AAA"].iloc[0]) is True


def test_combine_requires_min_pass():
    dates, symbols = _grid()
    yes = pd.DataFrame(True, index=dates, columns=symbols)
    no = pd.DataFrame(False, index=dates, columns=symbols)

    assert combine(yes, yes, no, make_cfg(fundamentals_min_pass=2)).all().all()
    assert not combine(yes, no, no, make_cfg(fundamentals_min_pass=2)).any().any()
